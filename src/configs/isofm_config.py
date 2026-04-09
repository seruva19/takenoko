from __future__ import annotations

from typing import Any, Dict


def apply_isofm_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Isokinetic Flow Matching settings (train-time only, default off)."""
    args.enable_isofm = bool(config.get("enable_isofm", False))
    args.isofm_lambda = float(config.get("isofm_lambda", 4.0))
    args.isofm_time_weight_exponent = float(
        config.get("isofm_time_weight_exponent", 2.0)
    )
    args.isofm_lookahead_eps_min = float(config.get("isofm_lookahead_eps_min", 0.02))
    args.isofm_lookahead_eps_max = float(config.get("isofm_lookahead_eps_max", 0.05))
    args.isofm_min_t = float(config.get("isofm_min_t", 0.0))
    args.isofm_max_t = float(config.get("isofm_max_t", 0.98))
    args.isofm_apply_prob = float(config.get("isofm_apply_prob", 1.0))
    args.isofm_warmup_steps = int(config.get("isofm_warmup_steps", 0))
    args.isofm_normalize_by_speed = bool(
        config.get("isofm_normalize_by_speed", True)
    )
    args.isofm_speed_epsilon = float(config.get("isofm_speed_epsilon", 1e-6))
    args.isofm_log_interval = int(config.get("isofm_log_interval", 100))

    if args.isofm_lambda < 0.0:
        raise ValueError("isofm_lambda must be >= 0")
    if args.isofm_time_weight_exponent < 0.0:
        raise ValueError("isofm_time_weight_exponent must be >= 0")
    if args.isofm_lookahead_eps_min <= 0.0:
        raise ValueError("isofm_lookahead_eps_min must be > 0")
    if args.isofm_lookahead_eps_max <= 0.0:
        raise ValueError("isofm_lookahead_eps_max must be > 0")
    if args.isofm_lookahead_eps_min > args.isofm_lookahead_eps_max:
        raise ValueError(
            "isofm_lookahead_eps_min must be <= isofm_lookahead_eps_max"
        )
    if not (0.0 <= args.isofm_min_t < 1.0):
        raise ValueError("isofm_min_t must be in [0, 1)")
    if not (0.0 < args.isofm_max_t <= 1.0):
        raise ValueError("isofm_max_t must be in (0, 1]")
    if args.isofm_min_t >= args.isofm_max_t:
        raise ValueError("isofm_min_t must be < isofm_max_t")
    if not (0.0 <= args.isofm_apply_prob <= 1.0):
        raise ValueError("isofm_apply_prob must be between 0 and 1")
    if args.isofm_warmup_steps < 0:
        raise ValueError("isofm_warmup_steps must be >= 0")
    if args.isofm_speed_epsilon <= 0.0:
        raise ValueError("isofm_speed_epsilon must be > 0")
    if args.isofm_log_interval <= 0:
        raise ValueError("isofm_log_interval must be > 0")

    if args.enable_isofm:
        logger.info(
            "IsoFM enabled: lambda=%.3f alpha=%.3f eps=[%.4f, %.4f] t_range=[%.3f, %.3f] "
            "apply_prob=%.3f warmup_steps=%d normalize_by_speed=%s speed_epsilon=%.1e log_interval=%d",
            args.isofm_lambda,
            args.isofm_time_weight_exponent,
            args.isofm_lookahead_eps_min,
            args.isofm_lookahead_eps_max,
            args.isofm_min_t,
            args.isofm_max_t,
            args.isofm_apply_prob,
            args.isofm_warmup_steps,
            str(args.isofm_normalize_by_speed).lower(),
            args.isofm_speed_epsilon,
            args.isofm_log_interval,
        )
