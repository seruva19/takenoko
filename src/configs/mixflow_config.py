"""MixFlow config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


def apply_mixflow_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.enable_mixflow = bool(config.get("enable_mixflow", False))
    args.mixflow_gamma = float(config.get("mixflow_gamma", 0.4))
    args.mixflow_beta_t_sampling = bool(config.get("mixflow_beta_t_sampling", False))
    args.mixflow_time_dist_shift_enabled = bool(
        config.get("mixflow_time_dist_shift_enabled", False)
    )
    args.mixflow_time_dist_shift = float(config.get("mixflow_time_dist_shift", 1.0))
    args.mixflow_log_every_n_steps = int(config.get("mixflow_log_every_n_steps", 0))

    if args.mixflow_gamma < 0.0 or args.mixflow_gamma > 1.0:
        raise ValueError("mixflow_gamma must be in [0.0, 1.0]")
    if args.mixflow_time_dist_shift <= 0.0:
        raise ValueError("mixflow_time_dist_shift must be > 0")
    if args.mixflow_log_every_n_steps < 0:
        raise ValueError("mixflow_log_every_n_steps must be >= 0")

    if args.enable_mixflow:
        logger.info(
            "MixFlow enabled (gamma=%.4f, beta_t_sampling=%s, time_dist_shift_enabled=%s, time_dist_shift=%.4f, log_every_n_steps=%d).",
            args.mixflow_gamma,
            args.mixflow_beta_t_sampling,
            args.mixflow_time_dist_shift_enabled,
            args.mixflow_time_dist_shift,
            args.mixflow_log_every_n_steps,
        )
