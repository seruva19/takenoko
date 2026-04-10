from __future__ import annotations

from typing import Any, Dict


def apply_diagnostic_metrics_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse opt-in diagnostic metrics settings onto args."""
    args.enable_diagnostic_metrics = bool(
        config.get("enable_diagnostic_metrics", False)
    )
    args.diagnostic_metrics_interval = int(config.get("diagnostic_metrics_interval", 50))
    args.diagnostic_metrics_enable_effective_batch_size = bool(
        config.get("diagnostic_metrics_enable_effective_batch_size", True)
    )
    args.diagnostic_metrics_enable_per_sample_loss_stats = bool(
        config.get("diagnostic_metrics_enable_per_sample_loss_stats", True)
    )
    args.diagnostic_metrics_enable_boundary_frame_loss = bool(
        config.get("diagnostic_metrics_enable_boundary_frame_loss", True)
    )
    args.diagnostic_metrics_enable_optical_flow_error = bool(
        config.get("diagnostic_metrics_enable_optical_flow_error", True)
    )
    args.diagnostic_metrics_hard_example_fraction = float(
        config.get("diagnostic_metrics_hard_example_fraction", 0.1)
    )

    if args.diagnostic_metrics_interval <= 0:
        raise ValueError("diagnostic_metrics_interval must be > 0")
    if not 0.0 < args.diagnostic_metrics_hard_example_fraction <= 1.0:
        raise ValueError("diagnostic_metrics_hard_example_fraction must be in (0, 1]")

    if args.enable_diagnostic_metrics:
        logger.info(
            "Diagnostic metrics enabled: interval=%d, effective_batch=%s, per_sample_loss_stats=%s, boundary_frame_loss=%s, optical_flow_error=%s, hard_example_fraction=%.3f",
            args.diagnostic_metrics_interval,
            str(args.diagnostic_metrics_enable_effective_batch_size).lower(),
            str(args.diagnostic_metrics_enable_per_sample_loss_stats).lower(),
            str(args.diagnostic_metrics_enable_boundary_frame_loss).lower(),
            str(args.diagnostic_metrics_enable_optical_flow_error).lower(),
            args.diagnostic_metrics_hard_example_fraction,
        )
