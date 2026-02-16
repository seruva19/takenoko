"""ShortFT config parsing for reward LoRA training (train-time only)."""

from __future__ import annotations

from typing import Any, Dict


def apply_shortft_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse ShortFT settings onto args with default-off gating."""
    args.enable_shortft = bool(config.get("enable_shortft", False))
    args.shortft_num_segments = int(config.get("shortft_num_segments", 4))
    args.shortft_stage_transition_steps = int(
        config.get("shortft_stage_transition_steps", 0)
    )
    args.shortft_start_step = int(config.get("shortft_start_step", 0))
    args.shortft_shortcut_anchor_count = int(
        config.get("shortft_shortcut_anchor_count", 1)
    )
    args.shortft_dense_backprop_mode = str(
        config.get("shortft_dense_backprop_mode", "all")
    ).lower()
    args.shortft_log_interval = int(config.get("shortft_log_interval", 100))

    if args.shortft_num_segments < 2:
        raise ValueError("shortft_num_segments must be >= 2")
    if args.shortft_stage_transition_steps < 0:
        raise ValueError("shortft_stage_transition_steps must be >= 0")
    if args.shortft_start_step < 0:
        raise ValueError("shortft_start_step must be >= 0")
    if args.shortft_shortcut_anchor_count <= 0:
        raise ValueError("shortft_shortcut_anchor_count must be > 0")
    if args.shortft_dense_backprop_mode not in ("all", "base"):
        raise ValueError("shortft_dense_backprop_mode must be one of: all, base")
    if args.shortft_log_interval < 0:
        raise ValueError("shortft_log_interval must be >= 0")

    if args.enable_shortft:
        if not bool(getattr(args, "enable_reward_lora", False)):
            logger.warning(
                "enable_shortft=true but enable_reward_lora=false; ShortFT will be inactive."
            )
        logger.info(
            "ShortFT enabled (segments=%d, stage_transition_steps=%d, start_step=%d, "
            "shortcut_anchor_count=%d, dense_backprop_mode=%s, log_interval=%d).",
            args.shortft_num_segments,
            args.shortft_stage_transition_steps,
            args.shortft_start_step,
            args.shortft_shortcut_anchor_count,
            args.shortft_dense_backprop_mode,
            args.shortft_log_interval,
        )
