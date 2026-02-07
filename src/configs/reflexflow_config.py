from __future__ import annotations

from typing import Any, Dict


def apply_reflexflow_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse ReflexFlow settings onto args (train-only, default off)."""
    args.enable_reflexflow = bool(config.get("enable_reflexflow", False))
    args.reflexflow_alpha = float(config.get("reflexflow_alpha", 1.0))
    args.reflexflow_beta1 = float(config.get("reflexflow_beta1", 10.0))
    args.reflexflow_beta2 = float(config.get("reflexflow_beta2", 1.0))
    args.reflexflow_warmup_steps = int(config.get("reflexflow_warmup_steps", 0))
    args.reflexflow_apply_prob = float(config.get("reflexflow_apply_prob", 1.0))
    args.reflexflow_apply_prob_start = float(
        config.get("reflexflow_apply_prob_start", args.reflexflow_apply_prob)
    )
    args.reflexflow_apply_prob_end = float(
        config.get("reflexflow_apply_prob_end", args.reflexflow_apply_prob)
    )
    args.reflexflow_apply_prob_ramp_steps = int(
        config.get("reflexflow_apply_prob_ramp_steps", 0)
    )
    args.reflexflow_apply_prob_start_step = int(
        config.get("reflexflow_apply_prob_start_step", 0)
    )
    args.reflexflow_apply_prob_ramp_shape = str(
        config.get("reflexflow_apply_prob_ramp_shape", "linear")
    ).lower()
    args.reflexflow_shift = float(config.get("reflexflow_shift", 0.6))
    args.reflexflow_min_t = float(config.get("reflexflow_min_t", 0.05))
    args.reflexflow_max_t = float(config.get("reflexflow_max_t", 0.95))
    args.reflexflow_blend = float(config.get("reflexflow_blend", 1.0))
    args.reflexflow_history_start_frame = int(
        config.get("reflexflow_history_start_frame", 0)
    )
    args.reflexflow_history_exclude_tail_frames = int(
        config.get("reflexflow_history_exclude_tail_frames", 1)
    )
    args.reflexflow_log_interval = int(config.get("reflexflow_log_interval", 50))

    if args.reflexflow_alpha < 0.0:
        raise ValueError("reflexflow_alpha must be >= 0")
    if args.reflexflow_beta1 < 0.0:
        raise ValueError("reflexflow_beta1 must be >= 0")
    if args.reflexflow_beta2 < 0.0:
        raise ValueError("reflexflow_beta2 must be >= 0")
    if args.reflexflow_warmup_steps < 0:
        raise ValueError("reflexflow_warmup_steps must be >= 0")
    if not 0.0 <= args.reflexflow_apply_prob <= 1.0:
        raise ValueError("reflexflow_apply_prob must be between 0 and 1")
    if not 0.0 <= args.reflexflow_apply_prob_start <= 1.0:
        raise ValueError("reflexflow_apply_prob_start must be between 0 and 1")
    if not 0.0 <= args.reflexflow_apply_prob_end <= 1.0:
        raise ValueError("reflexflow_apply_prob_end must be between 0 and 1")
    if args.reflexflow_apply_prob_ramp_steps < 0:
        raise ValueError("reflexflow_apply_prob_ramp_steps must be >= 0")
    if args.reflexflow_apply_prob_start_step < 0:
        raise ValueError("reflexflow_apply_prob_start_step must be >= 0")
    if args.reflexflow_apply_prob_ramp_shape not in ("linear", "cosine"):
        raise ValueError(
            "reflexflow_apply_prob_ramp_shape must be one of: linear, cosine"
        )
    if args.reflexflow_shift <= 0.0:
        raise ValueError("reflexflow_shift must be > 0")
    if not 0.0 <= args.reflexflow_min_t < 1.0:
        raise ValueError("reflexflow_min_t must be in [0, 1)")
    if not 0.0 < args.reflexflow_max_t <= 1.0:
        raise ValueError("reflexflow_max_t must be in (0, 1]")
    if args.reflexflow_min_t >= args.reflexflow_max_t:
        raise ValueError("reflexflow_min_t must be < reflexflow_max_t")
    if not 0.0 <= args.reflexflow_blend <= 1.0:
        raise ValueError("reflexflow_blend must be between 0 and 1")
    if args.reflexflow_history_start_frame < 0:
        raise ValueError("reflexflow_history_start_frame must be >= 0")
    if args.reflexflow_history_exclude_tail_frames < 0:
        raise ValueError("reflexflow_history_exclude_tail_frames must be >= 0")
    if args.reflexflow_log_interval <= 0:
        raise ValueError("reflexflow_log_interval must be > 0")

    if args.enable_reflexflow:
        logger.info(
            "ReflexFlow enabled: alpha=%.3f beta1=%.3f beta2=%.3f warmup_steps=%d apply_prob=%.3f (start=%.3f end=%.3f ramp_steps=%d start_step=%d shape=%s) shift=%.3f t_range=[%.3f, %.3f] blend=%.3f",
            args.reflexflow_alpha,
            args.reflexflow_beta1,
            args.reflexflow_beta2,
            args.reflexflow_warmup_steps,
            args.reflexflow_apply_prob,
            args.reflexflow_apply_prob_start,
            args.reflexflow_apply_prob_end,
            args.reflexflow_apply_prob_ramp_steps,
            args.reflexflow_apply_prob_start_step,
            args.reflexflow_apply_prob_ramp_shape,
            args.reflexflow_shift,
            args.reflexflow_min_t,
            args.reflexflow_max_t,
            args.reflexflow_blend,
        )
