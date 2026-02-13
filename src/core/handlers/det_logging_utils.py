"""DeT-specific logging helpers for training handlers."""

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_DET_COMPONENT_LOG_KEYS: Tuple[Tuple[str, str], ...] = (
    ("det_temporal_kernel_loss", "loss/det_temporal_kernel"),
    ("det_dense_tracking_loss", "loss/det_dense_tracking"),
    ("det_external_tracking_loss", "loss/det_external_tracking"),
    ("det_external_tracking_active_samples", "det/external_active_samples"),
    ("det_external_tracking_active_ratio", "det/external_active_ratio"),
    ("det_external_tracking_active_points", "det/external_active_points"),
    ("det_high_frequency_loss", "loss/det_high_frequency"),
    ("det_nonlocal_fallback_loss", "loss/det_nonlocal_fallback"),
    ("det_nonlocal_fallback_blend_mean", "det/nonlocal_fallback_blend_mean"),
    ("det_nonlocal_fallback_active_depths", "det/nonlocal_fallback_active_depths"),
    ("det_controller_sync_enabled", "det/controller_sync_enabled"),
    ("det_controller_sync_applied", "det/controller_sync_applied"),
    ("det_controller_sync_world_size", "det/controller_sync_world_size"),
    ("det_optimizer_lr_scale", "det/optimizer_lr_scale"),
    ("det_optimizer_lr_modulation_active", "det/optimizer_lr_modulation_active"),
    ("det_locality_ratio", "det/locality_ratio"),
    ("det_locality_scale", "det/locality_scale"),
    ("det_attention_locality_ratio", "det/attention_locality_ratio"),
    ("det_attention_locality_scale", "det/attention_locality_scale"),
    ("det_attention_locality_policy_active", "det/attention_locality_policy_active"),
    ("det_unified_controller_scale", "det/unified_controller_scale"),
    ("det_unified_controller_locality_scale", "det/unified_controller_locality_scale"),
    ("det_unified_controller_stability_scale", "det/unified_controller_stability_scale"),
    ("det_unified_controller_spike_ratio", "det/unified_controller_spike_ratio"),
    ("det_unified_controller_cooldown_active", "det/unified_controller_cooldown_active"),
    ("det_per_depth_scale_mean", "det/per_depth_scale_mean"),
    ("det_per_depth_scale_min", "det/per_depth_scale_min"),
    ("det_per_depth_scale_max", "det/per_depth_scale_max"),
    ("det_per_depth_cooldown_active_count", "det/per_depth_cooldown_active_count"),
    ("det_per_depth_spike_ratio_max", "det/per_depth_spike_ratio_max"),
    ("det_schedule_temporal_factor", "det/schedule_temporal_factor"),
    ("det_schedule_tracking_factor", "det/schedule_tracking_factor"),
    ("det_schedule_external_factor", "det/schedule_external_factor"),
    ("det_schedule_nonlocal_factor", "det/schedule_nonlocal_factor"),
    ("det_schedule_hf_factor", "det/schedule_hf_factor"),
    ("det_auto_safeguard_active", "det/auto_safeguard_active"),
    ("det_auto_safeguard_risky_step", "det/auto_safeguard_risky_step"),
    ("det_auto_safeguard_bad_streak", "det/auto_safeguard_bad_streak"),
    ("det_auto_safeguard_good_streak", "det/auto_safeguard_good_streak"),
    ("det_auto_safeguard_risk_locality", "det/auto_safeguard_risk_locality"),
    ("det_auto_safeguard_risk_spike", "det/auto_safeguard_risk_spike"),
    ("det_auto_safeguard_risk_cooldown", "det/auto_safeguard_risk_cooldown"),
    ("det_auto_safeguard_local_scale_cap", "det/auto_safeguard_local_scale_cap"),
    ("det_auto_safeguard_nonlocal_boost", "det/auto_safeguard_nonlocal_boost"),
)


def attach_det_component_logs(logs: Dict[str, float], loss_components: Any) -> None:
    """Attach DeT scalar metrics from LossComponents into logging dict."""
    for field_name, log_name in _DET_COMPONENT_LOG_KEYS:
        value = getattr(loss_components, field_name, None)
        if value is None:
            continue
        logs[log_name] = float(value.item())


def log_det_locality_profile_visuals(
    accelerator: Any,
    args: argparse.Namespace,
    det_motion_helper: Optional[Any],
    global_step: int,
    logs: Dict[str, float],
) -> None:
    if det_motion_helper is None:
        return
    if not bool(getattr(args, "det_locality_profiler_enabled", False)):
        return

    try:
        profile = det_motion_helper.consume_locality_profile(step=global_step)
    except Exception:
        return
    if not isinstance(profile, dict):
        return

    hist_matrix = profile.get("hist_matrix")
    bin_centers = profile.get("bin_centers")
    mean_by_depth = profile.get("mean_by_depth")
    std_by_depth = profile.get("std_by_depth")
    sample_count_by_depth = profile.get("sample_count_by_depth")
    depths = profile.get("depths")
    if (
        not torch.is_tensor(hist_matrix)
        or not torch.is_tensor(bin_centers)
        or not torch.is_tensor(mean_by_depth)
        or not isinstance(depths, list)
    ):
        return

    hist_matrix = hist_matrix.detach().to(dtype=torch.float32, device="cpu")
    bin_centers = bin_centers.detach().to(dtype=torch.float32, device="cpu")
    mean_by_depth = mean_by_depth.detach().to(dtype=torch.float32, device="cpu")
    if torch.is_tensor(std_by_depth):
        std_by_depth = std_by_depth.detach().to(dtype=torch.float32, device="cpu")
    if torch.is_tensor(sample_count_by_depth):
        sample_count_by_depth = sample_count_by_depth.detach().to(
            dtype=torch.float32, device="cpu"
        )

    if hist_matrix.ndim != 2 or hist_matrix.shape[0] <= 0 or hist_matrix.shape[1] <= 0:
        return
    if mean_by_depth.numel() != hist_matrix.shape[0]:
        return
    if len(depths) != hist_matrix.shape[0]:
        return

    prefix = str(
        getattr(args, "det_locality_profiler_log_prefix", "det_locality_profile")
        or "det_locality_profile"
    )
    logs[f"{prefix}/active_depths"] = float(hist_matrix.shape[0])
    logs[f"{prefix}/mean_distance"] = float(mean_by_depth.mean().item())
    for row_idx, depth in enumerate(depths):
        depth_key = str(depth).replace("-", "neg")
        logs[f"{prefix}/depth_{depth_key}_mean_distance"] = float(
            mean_by_depth[row_idx].item()
        )
        if torch.is_tensor(std_by_depth) and row_idx < int(std_by_depth.numel()):
            logs[f"{prefix}/depth_{depth_key}_std_distance"] = float(
                std_by_depth[row_idx].item()
            )

    if not accelerator.is_main_process:
        return

    tb_tracker = None
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tb_tracker = tracker
            break

    max_plot_depths = max(
        1,
        int(getattr(args, "det_locality_profiler_max_depths_in_plot", 8) or 8),
    )
    selected_indices = list(range(hist_matrix.shape[0]))
    if len(selected_indices) > max_plot_depths:
        if torch.is_tensor(sample_count_by_depth) and sample_count_by_depth.numel() == hist_matrix.shape[0]:
            _, topk_idx = torch.topk(
                sample_count_by_depth,
                k=max_plot_depths,
                largest=True,
                sorted=True,
            )
            selected_indices = [int(v) for v in topk_idx.tolist()]
        else:
            selected_indices = selected_indices[:max_plot_depths]
        selected_indices = sorted(selected_indices)

    depth_labels = [str(depths[i]) for i in selected_indices]
    plot_matrix = hist_matrix.index_select(
        dim=0,
        index=torch.tensor(selected_indices, dtype=torch.long),
    )

    if tb_tracker is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_h = max(2.5, 0.4 * len(selected_indices) + 1.5)
            fig, ax = plt.subplots(1, 1, figsize=(8.0, fig_h))
            im = ax.imshow(
                plot_matrix.numpy(),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap="magma",
                vmin=0.0,
            )
            ax.set_xlabel("Normalized Displacement Bin")
            ax.set_ylabel("Depth")
            ax.set_yticks(list(range(len(depth_labels))))
            ax.set_yticklabels(depth_labels)
            x_ticks = torch.linspace(
                0,
                plot_matrix.shape[1] - 1,
                steps=min(6, plot_matrix.shape[1]),
            ).round().long()
            ax.set_xticks(x_ticks.tolist())
            ax.set_xticklabels(
                [f"{float(bin_centers[i]):.2f}" for i in x_ticks.tolist()]
            )
            ax.set_title("DeT Locality Profile (Depth x Displacement)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            tb_tracker.writer.add_figure(  # type: ignore[attr-defined]
                f"{prefix}/heatmap",
                fig,
                global_step=global_step,
            )
            plt.close(fig)

            fig2, ax2 = plt.subplots(1, 1, figsize=(8.0, 4.0))
            for row_idx, depth_label in enumerate(depth_labels):
                ax2.plot(
                    bin_centers.numpy(),
                    plot_matrix[row_idx].numpy(),
                    label=f"depth {depth_label}",
                    linewidth=1.2,
                )
            ax2.set_xlabel("Normalized Displacement")
            ax2.set_ylabel("Probability")
            ax2.set_ylim(bottom=0.0)
            ax2.set_title("DeT Locality Distribution by Depth")
            if len(depth_labels) <= 8:
                ax2.legend(loc="upper right", fontsize=8)
            fig2.tight_layout()
            tb_tracker.writer.add_figure(  # type: ignore[attr-defined]
                f"{prefix}/histograms",
                fig2,
                global_step=global_step,
            )
            plt.close(fig2)
        except Exception as exc:
            logger.warning(
                "Failed to log DeT locality profile figures at step %d: %s",
                global_step,
                exc,
            )

    if bool(getattr(args, "det_locality_profiler_export_artifacts", False)):
        try:
            export_dir = str(
                getattr(args, "det_locality_profiler_export_dir", "") or ""
            ).strip()
            if not export_dir:
                output_dir = str(getattr(args, "output_dir", "output") or "output")
                export_dir = os.path.join(output_dir, "det_locality_profiles")
            os.makedirs(export_dir, exist_ok=True)

            payload = {
                "step": int(global_step),
                "depths": [int(depths[i]) for i in selected_indices],
                "bin_centers": [float(v) for v in bin_centers.tolist()],
                "hist_matrix": plot_matrix.tolist(),
                "mean_by_depth": [float(v) for v in mean_by_depth.tolist()],
                "std_by_depth": (
                    [float(v) for v in std_by_depth.tolist()]
                    if torch.is_tensor(std_by_depth)
                    else None
                ),
                "sample_count_by_depth": (
                    [float(v) for v in sample_count_by_depth.tolist()]
                    if torch.is_tensor(sample_count_by_depth)
                    else None
                ),
            }
            out_path = os.path.join(
                export_dir,
                f"det_locality_profile_step{int(global_step):08d}.json",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning(
                "Failed to export DeT locality profile payload at step %d: %s",
                global_step,
                exc,
            )

