from __future__ import annotations

from typing import Any, Dict


def attach_attention_metrics_and_maybe_heatmap(
    accelerator: Any,
    args: Any,
    logs: Dict[str, float],
    global_step: int,
) -> None:
    """Attach scalar attention metrics and optionally log a small heatmap.

    - Scalars come from common.attention_metrics.get_and_clear_latest_metrics()
    - Heatmap (if enabled) is logged to TensorBoard under prefix
      args.attention_metrics_heatmap_log_prefix
    """
    try:
        from common import (
            attention_metrics as _attn_metrics,
        )  # local import to avoid cycles

        attn_logs = _attn_metrics.get_and_clear_latest_metrics()
        if attn_logs:
            logs.update(attn_logs)

        if getattr(args, "attention_metrics_log_heatmap", False):
            heat = _attn_metrics.get_and_clear_latest_heatmap()
            if heat is not None:
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        try:
                            import numpy as _np  # type: ignore
                            import matplotlib.pyplot as _plt  # type: ignore
                            from matplotlib.colors import LogNorm  # type: ignore

                            _data = heat.numpy()
                            # Floor very small values for LogNorm stability
                            _eps = 1e-8
                            if _data.size > 0:
                                _data = _np.maximum(_data, _eps)
                            # Read config preferences
                            _cmap = str(
                                getattr(args, "attention_metrics_heatmap_cmap", "magma")
                            )
                            _norm_mode = str(
                                getattr(args, "attention_metrics_heatmap_norm", "log")
                            ).lower()
                            _vmin_pct = float(
                                getattr(
                                    args, "attention_metrics_heatmap_vmin_pct", 60.0
                                )
                            )
                            _vmax_pct = float(
                                getattr(
                                    args, "attention_metrics_heatmap_vmax_pct", 99.5
                                )
                            )
                            _fig_w = float(
                                getattr(args, "attention_metrics_heatmap_fig_w", 6.0)
                            )
                            _fig_h = float(
                                getattr(args, "attention_metrics_heatmap_fig_h", 4.0)
                            )

                            # Percentile-based contrast stretching
                            try:
                                _vmin = float(_np.percentile(_data, _vmin_pct))
                                _vmax = float(_np.percentile(_data, _vmax_pct))
                                if not _np.isfinite(_vmin) or not _np.isfinite(_vmax):
                                    raise ValueError("non-finite percentiles")
                                if _vmax <= _vmin:
                                    _vmin = max(_eps, _vmin * 0.5)
                                    _vmax = max(_vmin + 1e-6, _vmax * 1.5)
                                if _norm_mode == "log":
                                    _norm = LogNorm(vmin=_vmin, vmax=_vmax)
                                else:
                                    _norm = None
                            except Exception:
                                _norm = None  # Fallback to Matplotlib default

                            _fig, _ax = _plt.subplots(figsize=(_fig_w, _fig_h))
                            _im = _ax.imshow(
                                _data,
                                aspect="auto",
                                cmap=_cmap,
                                norm=_norm,
                            )
                            _ax.set_xlabel("text tokens")
                            _ax.set_ylabel("queries")
                            _ax.set_title(
                                str(
                                    getattr(
                                        args,
                                        "attention_metrics_heatmap_log_prefix",
                                        "attn_hm",
                                    )
                                )
                            )
                            try:
                                _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
                            except Exception:
                                pass
                            _plt.tight_layout()
                            tracker.writer.add_figure(
                                f"{getattr(args, 'attention_metrics_heatmap_log_prefix', 'attn_hm')}/map",
                                _fig,
                                global_step,
                            )
                            _plt.close(_fig)
                        except Exception:
                            # Fallback: try add_image with normalized tensor [1,H,W]
                            try:
                                img = heat.clone()
                                if img.numel() > 0:
                                    mn = float(img.min().item())
                                    mx = float(img.max().item())
                                    if mx > mn:
                                        img = (img - mn) / (mx - mn)
                                tracker.writer.add_image(
                                    f"{getattr(args, 'attention_metrics_heatmap_log_prefix', 'attn_hm')}/map",
                                    img.unsqueeze(0),
                                    global_step,
                                )
                            except Exception:
                                pass
                        break
    except Exception:
        # Never fail logging path
        pass
