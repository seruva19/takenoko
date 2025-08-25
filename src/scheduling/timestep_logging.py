"""Utilities for logging timestep distributions to TensorBoard.

This module centralizes all visualization and logging of timestep distributions
to keep training core clean and maintainable.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol
import os
from collections import deque
from typing import Deque, List

from common.logger import get_logger

logger = get_logger(__name__)

# One-time guard to avoid spamming warnings
_WARNED_VIZ_MISMATCH: bool = False


class _TBWriter(Protocol):
    def add_histogram(
        self, tag: str, values: Any, global_step: int, bins: Optional[int] = None
    ) -> None: ...
    def add_figure(self, tag: str, figure: Any, global_step: int) -> None: ...
    def add_scalar(self, tag: str, scalar_value: Any, global_step: int) -> None: ...


def _get_tensorboard_writer(accelerator) -> Optional[_TBWriter]:
    try:
        for tracker in accelerator.trackers:
            if getattr(tracker, "name", None) == "tensorboard" and hasattr(
                tracker, "writer"
            ):
                return tracker.writer
    except Exception:
        return None
    return None


def log_initial_timestep_distribution(accelerator, args, timestep_distribution) -> None:
    """Log the initial/expected timestep distribution.

    Preference order:
    1) Precomputed buckets if enabled and initialized
    2) Simulate via configured sampler and apply constraints
    """
    writer = _get_tensorboard_writer(accelerator)
    if writer is None:
        return

    import torch

    # Warn once when advanced sampling/regime features may cause mismatch
    global _WARNED_VIZ_MISMATCH
    if not _WARNED_VIZ_MISMATCH:
        try:
            features: list[str] = []
            if str(getattr(args, "timestep_sampling", "")).lower() == "fopp":
                features.append("FoPP AR-Diffusion")
            if bool(getattr(args, "enable_fvdm", False)):
                features.append("FVDM")
            if bool(getattr(args, "enable_dual_model_training", False)):
                features.append("Dual-model regime switching")
            if features:
                logger.warning(
                    "Timestep visualization may not fully reflect training due to: "
                    + ", ".join(features)
                )
                _WARNED_VIZ_MISMATCH = True
        except Exception:
            pass

    log_mode = str(getattr(args, "log_timestep_distribution", "off")).lower()
    if log_mode == "off":
        return
    num_bins = max(10, int(getattr(args, "log_timestep_distribution_bins", 100)))

    # Optional: only log once per logging directory using a marker file
    try:
        init_once = bool(getattr(args, "log_timestep_distribution_init_once", True))
        if init_once:
            logdir = getattr(args, "logging_dir", None)
            if logdir:
                os.makedirs(logdir, exist_ok=True)
                marker_path = os.path.join(
                    logdir, "timestep_initial_expected_logged.marker"
                )
                if os.path.exists(marker_path):
                    return
    except Exception:
        # Fallback to normal behavior if anything goes wrong
        pass

    # Prefer precomputed buckets if available
    dist = None
    try:
        from scheduling.timestep_distribution import should_use_precomputed_timesteps

        if (
            should_use_precomputed_timesteps(args)
            and getattr(timestep_distribution, "is_initialized", False)
            and getattr(timestep_distribution, "distribution", None) is not None
        ):
            dist = timestep_distribution.distribution
    except Exception:
        dist = None

    if dist is None:
        # Simulate by sampling many timesteps via the configured sampler
        try:
            method = str(getattr(args, "timestep_sampling", "")).lower()
            samples = int(getattr(args, "log_timestep_distribution_samples", 20000))
            if method == "sigma":
                # Mirror sigma-path sampling used in training
                from utils.train_utils import compute_density_for_timestep_sampling
                import torch as _torch

                _wm = str(getattr(args, "weighting_scheme", "none"))
                _lm = float(getattr(args, "logit_mean", 0.0) or 0.0)
                _ls = float(getattr(args, "logit_std", 1.0) or 1.0)
                _ms = float(getattr(args, "mode_scale", 1.29) or 1.29)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=_wm,
                    batch_size=samples,
                    logit_mean=_lm,
                    logit_std=_ls,
                    mode_scale=_ms,
                )
                t_min = int(getattr(args, "min_timestep", 0) or 0)
                t_max = int(getattr(args, "max_timestep", 1000) or 1000)
                indices = (u * (t_max - t_min) + t_min).long()
                # Convert to normalized [0,1] like the non-sigma path
                t = (indices.to(dtype=_torch.float32) + 1.0) / 1000.0
                dist = t
            else:
                from scheduling.timestep_utils import (
                    _generate_timesteps_from_distribution,
                    _apply_timestep_constraints,
                )

                device = accelerator.device
                t = _generate_timesteps_from_distribution(args, samples, device)
                t = _apply_timestep_constraints(t, args, samples, device)
                dist = t
        except Exception:
            return

    # Histogram
    try:
        writer.add_histogram(
            "timestep/initial_expected",
            dist.detach().cpu(),
            global_step=0,
            bins=num_bins,
        )
    except TypeError:
        writer.add_histogram(
            "timestep/initial_expected", dist.detach().cpu(), global_step=0
        )

    # Chart image (optional)
    if log_mode in ["chart", "both"]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            values = dist.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=120)
            ax.hist(values, bins=num_bins, range=(0.0, 1.0), density=True, alpha=0.8)
            ax.set_title("Initial timestep distribution (expected)")
            ax.set_xlabel("t in [0,1]")
            ax.set_ylabel("density")
            fig.tight_layout()
            writer.add_figure("timestep/initial_expected_chart", fig, global_step=0)
            plt.close(fig)
        except Exception:
            pass

    # Optional probability mass function (PMF) bar chart over [1..1000]
    try:
        if bool(getattr(args, "log_timestep_distribution_pmf", False)):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import torch as _torch
            import numpy as _np

            idx_exp = (_torch.round(dist * 1000.0 + 1.0)).clamp_(1, 1000).long()
            counts = _torch.zeros(1000, dtype=_torch.float32)
            counts.index_add_(
                0, idx_exp - 1, _torch.ones_like(idx_exp, dtype=_torch.float32)
            )
            pmf = counts / counts.sum().clamp_min(1.0)

            x = _np.arange(1, 1001, dtype=_np.int32)
            y = pmf.numpy()

            tmin = int(getattr(args, "min_timestep", 0))
            tmax = int(getattr(args, "max_timestep", 1000))
            lo = max(1, tmin)
            hi = max(lo + 1, min(1000, tmax))

            fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=120)
            ax.bar(x[lo - 1 : hi], y[lo - 1 : hi], width=1.0)
            ax.set_title("Initial expected PMF over indices")
            ax.set_xlabel("timestep index [1..1000]")
            ax.set_ylabel("probability")
            fig.tight_layout()
            writer.add_figure("timestep/initial_expected_pmf", fig, global_step=0)
            plt.close(fig)
    except Exception:
        pass

    # Write marker so restarts do not duplicate this log
    try:
        if bool(getattr(args, "log_timestep_distribution_init_once", True)):
            logdir = getattr(args, "logging_dir", None)
            if logdir:
                marker_path = os.path.join(
                    logdir, "timestep_initial_expected_logged.marker"
                )
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write("logged\n")
    except Exception:
        pass


def log_live_timestep_distribution(
    accelerator, args, timesteps, global_step: int
) -> None:
    """Log the live (used) timestep indices during training.

    Logs histogram of indices in 1..1000, and an optional chart of normalized
    values in [0,1].
    """
    writer = _get_tensorboard_writer(accelerator)
    if writer is None:
        return

    import torch

    log_mode = str(getattr(args, "log_timestep_distribution", "off")).lower()
    if log_mode == "off":
        return
    interval = int(getattr(args, "log_timestep_distribution_interval", 1000))
    if interval <= 0 or (global_step % interval != 0):
        return

    num_bins = max(10, int(getattr(args, "log_timestep_distribution_bins", 100)))

    # Histogram over integer indices 1..1000
    ts = timesteps.detach().float().cpu()
    try:
        writer.add_histogram(
            "timestep/used_indices", ts, global_step=global_step, bins=num_bins
        )
    except TypeError:
        writer.add_histogram("timestep/used_indices", ts, global_step=global_step)

    # Rolling-window histogram over recent indices
    try:
        global _USED_INDICES_BUFFER
        window_size = int(getattr(args, "log_timestep_distribution_window", 10000))
        if window_size > 0:
            # Initialize or resize the global buffer lazily
            try:
                _ = _USED_INDICES_BUFFER  # type: ignore
            except NameError:
                _USED_INDICES_BUFFER = None  # type: ignore

            if (
                _USED_INDICES_BUFFER is None  # type: ignore
                or getattr(_USED_INDICES_BUFFER, "maxlen", None) != window_size  # type: ignore
            ):
                _USED_INDICES_BUFFER = deque(maxlen=window_size)  # type: ignore

            # Extend buffer with current step values
            values: List[float] = ts.tolist()  # type: ignore
            _USED_INDICES_BUFFER.extend(values)  # type: ignore

            # Log histogram of the aggregated window
            import torch as _torch  # local import to avoid polluting module scope

            if len(_USED_INDICES_BUFFER) > 0:  # type: ignore
                agg = _torch.tensor(list(_USED_INDICES_BUFFER), dtype=_torch.float32)  # type: ignore
                try:
                    writer.add_histogram(
                        "timestep/used_indices_window",
                        agg,
                        global_step=global_step,
                        bins=num_bins,
                    )
                except TypeError:
                    writer.add_histogram(
                        "timestep/used_indices_window", agg, global_step=global_step
                    )

                # Scalars: mean/std over window (indices and normalized t)
                try:
                    mean_idx = float(agg.mean().item())
                    std_idx = float(agg.std(unbiased=False).item())
                    writer.add_scalar(
                        "timestep/window/mean_index", mean_idx, global_step
                    )
                    writer.add_scalar("timestep/window/std_index", std_idx, global_step)
                    mean_t = float((agg / 1000.0).mean().item())
                    writer.add_scalar("timestep/window/mean_t", mean_t, global_step)
                except Exception:
                    pass

                # Band ratios using configured edges, e.g., "850,900,950"
                try:
                    bands_raw = str(
                        getattr(args, "log_timestep_distribution_bands", "")
                    ).strip()
                    if bands_raw:
                        edges = [
                            int(x) for x in bands_raw.split(",") if str(x).strip() != ""
                        ]
                        # Allow 0 as a lower-bound sentinel; indices are clamped to [1,1000]
                        edges = sorted([e for e in edges if 0 <= e <= 1000])
                        if len(edges) >= 2:
                            prev = edges[0]
                            for e in edges[1:]:
                                mask = (agg >= prev) & (agg < e)
                                ratio = float(mask.float().mean().item())
                                writer.add_scalar(
                                    f"timestep/window/ratio_{prev}_{e}",
                                    ratio,
                                    global_step,
                                )
                                prev = e
                            # Tail bucket [prev, 1000]
                            mask = agg >= prev
                            ratio = float(mask.float().mean().item())
                            writer.add_scalar(
                                f"timestep/window/ratio_{prev}_1000", ratio, global_step
                            )
                except Exception:
                    pass

                # KL divergence to expected distribution simulated under current config
                try:
                    counts_w = _torch.zeros(1000, dtype=_torch.float32)
                    idx = agg.long().clamp_(1, 1000) - 1
                    counts_w.index_add_(
                        0, idx, _torch.ones_like(agg, dtype=_torch.float32)
                    )
                    p = counts_w / counts_w.sum().clamp_min(1.0)

                    samples = int(
                        getattr(args, "log_timestep_distribution_samples", 20000)
                    )
                    from scheduling.timestep_utils import (
                        _generate_timesteps_from_distribution as _gen,
                        _apply_timestep_constraints as _apply,
                    )

                    t_exp = _gen(args, samples, accelerator.device)
                    t_exp = _apply(t_exp, args, samples, accelerator.device)
                    idx_exp = (
                        (_torch.round(t_exp * 1000.0 + 1.0)).clamp_(1, 1000).long()
                    )
                    counts_e = _torch.zeros(1000, dtype=_torch.float32)
                    counts_e.index_add_(
                        0, idx_exp - 1, _torch.ones_like(idx_exp, dtype=_torch.float32)
                    )
                    q = counts_e / counts_e.sum().clamp_min(1.0)

                    eps = 1e-8
                    kl = float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())
                    writer.add_scalar("timestep/window/kl_to_expected", kl, global_step)
                except Exception:
                    pass

                # Optional PMF bar chart for rolling window
                try:
                    if bool(getattr(args, "log_timestep_distribution_pmf", False)):
                        import matplotlib.pyplot as plt
                        import numpy as _np

                        pmf = p
                        x = _np.arange(1, 1001, dtype=_np.int32)
                        y = pmf.numpy()
                        tmin = int(getattr(args, "min_timestep", 0))
                        tmax = int(getattr(args, "max_timestep", 1000))
                        lo = max(1, tmin)
                        hi = max(lo + 1, min(1000, tmax))

                        fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=120)
                        ax.bar(x[lo - 1 : hi], y[lo - 1 : hi], width=1.0)
                        ax.set_title("Rolling window PMF over indices")
                        ax.set_xlabel("timestep index [1..1000]")
                        ax.set_ylabel("probability")
                        fig.tight_layout()
                        writer.add_figure(
                            "timestep/used_pmf_window", fig, global_step=global_step
                        )
                        plt.close(fig)
                except Exception:
                    pass
    except Exception:
        # Never block training on logging failures
        pass

    # Chart image (normalized 0..1)
    if log_mode in ["chart", "both"]:
        try:
            import matplotlib.pyplot as plt

            ts_norm = (ts.numpy() / 1000.0).clip(0.0, 1.0)
            fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=120)
            ax.hist(ts_norm, bins=num_bins, range=(0.0, 1.0), density=True, alpha=0.8)
            ax.set_title("Live timestep distribution (used)")
            ax.set_xlabel("t in [0,1]")
            ax.set_ylabel("density")
            fig.tight_layout()
            writer.add_figure("timestep/used_chart", fig, global_step=global_step)
            plt.close(fig)
        except Exception:
            pass

        # Rolling-window chart
        try:
            import matplotlib.pyplot as plt
            import numpy as _np

            window_size = int(getattr(args, "log_timestep_distribution_window", 10000))
            if window_size > 0:
                if (
                    "_USED_INDICES_BUFFER" in globals()
                    and _USED_INDICES_BUFFER is not None
                ):
                    arr = _np.asarray(list(_USED_INDICES_BUFFER), dtype=_np.float32)
                    if arr.size > 0:
                        ts_norm_w = (arr / 1000.0).clip(0.0, 1.0)
                        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=120)
                        ax.hist(
                            ts_norm_w,
                            bins=num_bins,
                            range=(0.0, 1.0),
                            density=True,
                            alpha=0.8,
                        )
                        ax.set_title("Live timestep distribution (window)")
                        ax.set_xlabel("t in [0,1]")
                        ax.set_ylabel("density")
                        fig.tight_layout()
                        writer.add_figure(
                            "timestep/used_chart_window", fig, global_step=global_step
                        )
                        plt.close(fig)
        except Exception:
            pass


def log_loss_scatterplot(
    accelerator,
    args,
    timesteps,
    model_pred,
    target,
    global_step: int,
) -> None:
    """Log a scatter plot of per-sample loss vs timestep to TensorBoard.

    Safely no-ops if TensorBoard is not enabled, matplotlib is missing,
    or the feature is disabled in args.
    """
    writer = _get_tensorboard_writer(accelerator)
    if writer is None:
        return

    # Gate via config flag and interval
    enabled = bool(getattr(args, "log_loss_scatterplot", False))
    if not enabled:
        return
    interval = int(getattr(args, "log_loss_scatterplot_interval", 500))
    if interval <= 0 or (global_step % interval != 0):
        return

    try:
        import torch as _torch
        import matplotlib.pyplot as _plt

        # Compute per-sample velocity MSE matching extra-train-metrics convention
        with _torch.no_grad():
            mp = model_pred.float()
            tgt = target.float()
            # Expect shapes (B, C, F, H, W). Reduce all but batch
            per_sample = _torch.nn.functional.mse_loss(mp, tgt, reduction="none").mean(
                dim=[1, 2, 3, 4]
            )

            # Ensure timesteps alignment (1D of length B)
            ts = timesteps.detach().float().view(-1)
            if per_sample.shape[0] != ts.shape[0]:
                # Best-effort alignment: take min length
                n = int(min(per_sample.shape[0], ts.shape[0]))
                per_sample = per_sample[:n]
                ts = ts[:n]

            # Move to CPU
            ts_np = ts.cpu().numpy()
            loss_np = per_sample.cpu().numpy()

        # Plot
        fig, ax = _plt.subplots(figsize=(6.5, 4.5), dpi=120)
        ax.scatter(ts_np, loss_np, alpha=0.5, s=6)
        ax.set_yscale("log")
        ax.set_xlabel("Timestep (index)")
        ax.set_ylabel("Per-sample velocity loss (log)")
        ax.set_title(f"Loss vs. Timestep (Step: {global_step})")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        fig.tight_layout()

        writer.add_figure("timestep/loss_scatter", fig, global_step=global_step)
        _plt.close(fig)
    except Exception:
        # Never block training on logging failures
        return


def log_show_timesteps_figure_unconditional(
    accelerator: Any,
    args: Any,
    timestep_distribution: Any,
    noise_scheduler: Any,
) -> None:
    """Always log a two-subplot figure (sampled timesteps and loss weighting) to TensorBoard.

    Runs once per logging directory (guarded by a marker file) and does not depend on
    show_timesteps or log_timestep_distribution settings.
    """
    logdir = getattr(args, "logging_dir", None)
    if not logdir:
        return
    try:
        os.makedirs(logdir, exist_ok=True)
        marker_path = os.path.join(logdir, "timestep_show_timesteps_logged.marker")
        if os.path.exists(marker_path):
            return
    except Exception:
        marker_path = None

    writer = _get_tensorboard_writer(accelerator)
    created_writer = False
    if writer is None:
        try:
            try:
                from tensorboardX import SummaryWriter as _SummaryWriter  # type: ignore
            except ImportError:
                from torch.utils.tensorboard.writer import SummaryWriter as _SummaryWriter  # type: ignore
            writer = _SummaryWriter(log_dir=logdir)
            created_writer = True
        except Exception:
            writer = None
    if writer is None:
        return

    try:
        import torch
        import numpy as _np
        import matplotlib.pyplot as plt

        # Warn once when advanced sampling/regime features may cause mismatch
        global _WARNED_VIZ_MISMATCH
        if not _WARNED_VIZ_MISMATCH:
            try:
                features: list[str] = []
                if str(getattr(args, "timestep_sampling", "")).lower() == "fopp":
                    features.append("FoPP AR-Diffusion")
                if bool(getattr(args, "enable_fvdm", False)):
                    features.append("FVDM")
                if bool(getattr(args, "enable_dual_model_training", False)):
                    features.append("Dual-model regime switching")
                if features:
                    logger.warning(
                        "Timestep visualization may not fully reflect training due to: "
                        + ", ".join(features)
                    )
                    _WARNED_VIZ_MISMATCH = True
            except Exception:
                pass

        try:
            from scheduling.timestep_utils import (
                _generate_timesteps_from_distribution,
                _apply_timestep_constraints,
            )
            from scheduling.timestep_distribution import (
                should_use_precomputed_timesteps as _use_precomp,
            )

            samples = int(getattr(args, "log_timestep_distribution_samples", 20000))
            _device_any = getattr(accelerator, "device", "cpu")
            if isinstance(_device_any, str):
                _device = torch.device(_device_any)
            else:
                try:
                    _device = (
                        _device_any
                        if isinstance(_device_any, torch.device)
                        else torch.device("cpu")
                    )
                except Exception:
                    _device = torch.device("cpu")

            # Prefer precomputed buckets when enabled and initialized, to mirror training
            if (
                _use_precomp(args)
                and getattr(timestep_distribution, "is_initialized", False)
                and getattr(timestep_distribution, "distribution", None) is not None
            ):
                t = timestep_distribution.sample(samples, _device)
            else:
                method = str(getattr(args, "timestep_sampling", "")).lower()
                if method == "sigma":
                    from utils.train_utils import compute_density_for_timestep_sampling
                    import torch as _torch

                    _wm = str(getattr(args, "weighting_scheme", "none"))
                    _lm = float(getattr(args, "logit_mean", 0.0) or 0.0)
                    _ls = float(getattr(args, "logit_std", 1.0) or 1.0)
                    _ms = float(getattr(args, "mode_scale", 1.29) or 1.29)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=_wm,
                        batch_size=samples,
                        logit_mean=_lm,
                        logit_std=_ls,
                        mode_scale=_ms,
                    )
                    t_min = int(getattr(args, "min_timestep", 0) or 0)
                    t_max = int(getattr(args, "max_timestep", 1000) or 1000)
                    indices = (u * (t_max - t_min) + t_min).long()
                    t = (indices.to(dtype=_torch.float32) + 1.0) / 1000.0
                else:
                    t = _generate_timesteps_from_distribution(args, samples, _device)

            # Apply the same constraint utility used in training (respects skip flag)
            t = _apply_timestep_constraints(t, args, samples, _device)
        except Exception:
            t = torch.rand(int(20000))

        idx = (torch.clamp((t * 1000.0).round_(), 0, 999)).long().cpu()
        sampled_timesteps = torch.bincount(idx, minlength=1000).float().numpy()

        try:
            from utils.train_utils import compute_loss_weighting_for_sd3

            weights = []
            # Respect configured timestep limits by masking outside [min, max]
            t_min = int(getattr(args, "min_timestep", 0) or 0)
            t_max = int(getattr(args, "max_timestep", 1000) or 1000)
            t_min = max(0, min(999, t_min))
            t_max = max(0, min(1000, t_max))

            for i in range(1000):
                if i < t_min or i >= t_max:
                    weights.append(0.0)
                    continue
                ts = torch.tensor([i + 1], device="cpu")
                w = compute_loss_weighting_for_sd3(
                    getattr(args, "weighting_scheme", "none"),
                    noise_scheduler,
                    ts,
                    "cpu",
                    torch.float16,
                )
                if w is None or (torch.isinf(w).any() or torch.isnan(w).any()):
                    weights.append(1.0)
                else:
                    weights.append(float(w.item()))
            sampled_weighting = _np.array(weights, dtype=_np.float32)
        except Exception:
            sampled_weighting = _np.ones((1000,), dtype=_np.float32)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(_np.arange(1000), sampled_timesteps, width=1.0)
        ax1.set_title("Sampled timesteps")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Count")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(_np.arange(1000), sampled_weighting, width=1.0)
        ax2.set_title("Sampled loss weighting")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Weighting")
        fig.tight_layout()

        try:
            writer.add_figure("timestep/show_timesteps_chart", fig, global_step=0)
        finally:
            plt.close(fig)

        try:
            if marker_path is not None:
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write("logged\n")
        except Exception:
            pass
    finally:
        # Only close if we created a standalone writer that exposes close()
        if created_writer:
            try:
                # Close only if the dynamically created writer exposes close(); typing: ignore to satisfy protocol
                if hasattr(writer, "close"):
                    writer.close()  # type: ignore[attr-defined]
            except Exception:
                pass
