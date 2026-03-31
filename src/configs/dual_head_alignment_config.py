"""Dual-head alignment config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


_ALLOWED_TEACHER_MODES = {
    "base_model",
    "base_model_multiplier",
    "base_model_disable_lora",
}
_ALLOWED_WINDOW_SAMPLING = {"all", "random"}


def apply_dual_head_alignment_config(
    args: Any, config: Dict[str, Any], logger: Any
) -> None:
    args.enable_dual_head_alignment = bool(
        config.get("enable_dual_head_alignment", False)
    )
    args.dual_head_alignment_global_weight = float(
        config.get("dual_head_alignment_global_weight", 1.0)
    )
    args.dual_head_alignment_local_weight = float(
        config.get("dual_head_alignment_local_weight", 1.0)
    )
    args.dual_head_alignment_window_frames = int(
        config.get("dual_head_alignment_window_frames", 0)
    )
    args.dual_head_alignment_window_stride = int(
        config.get("dual_head_alignment_window_stride", 0)
    )
    args.dual_head_alignment_local_recon_weight = float(
        config.get("dual_head_alignment_local_recon_weight", 0.0)
    )
    args.dual_head_alignment_local_behavior_weight = float(
        config.get("dual_head_alignment_local_behavior_weight", 1.0)
    )
    args.dual_head_alignment_local_kl_weight = float(
        config.get("dual_head_alignment_local_kl_weight", 0.0)
    )
    args.dual_head_alignment_local_div_forward_weight = float(
        config.get("dual_head_alignment_local_div_forward_weight", 0.0)
    )
    args.dual_head_alignment_local_div_reverse_weight = float(
        config.get("dual_head_alignment_local_div_reverse_weight", 1.0)
    )
    args.dual_head_alignment_temperature = float(
        config.get("dual_head_alignment_temperature", 1.0)
    )
    args.dual_head_alignment_teacher_mode = str(
        config.get("dual_head_alignment_teacher_mode", "base_model")
    ).lower()
    args.dual_head_alignment_head_lr_scale = float(
        config.get("dual_head_alignment_head_lr_scale", 1.0)
    )
    args.dual_head_alignment_start_step = int(
        config.get("dual_head_alignment_start_step", 0)
    )
    args.dual_head_alignment_teacher_interval_steps = int(
        config.get("dual_head_alignment_teacher_interval_steps", 1)
    )
    args.dual_head_alignment_local_weight_ramp_steps = int(
        config.get("dual_head_alignment_local_weight_ramp_steps", 0)
    )
    args.dual_head_alignment_window_sampling = str(
        config.get("dual_head_alignment_window_sampling", "all")
    ).lower()
    args.dual_head_alignment_max_windows = int(
        config.get("dual_head_alignment_max_windows", 0)
    )
    args.dual_head_alignment_random_seed = int(
        config.get("dual_head_alignment_random_seed", 42)
    )

    if not args.enable_dual_head_alignment:
        return

    if args.dual_head_alignment_global_weight < 0.0:
        raise ValueError("dual_head_alignment_global_weight must be >= 0")
    if args.dual_head_alignment_local_weight < 0.0:
        raise ValueError("dual_head_alignment_local_weight must be >= 0")
    if args.dual_head_alignment_window_frames < 0:
        raise ValueError("dual_head_alignment_window_frames must be >= 0")
    if args.dual_head_alignment_window_stride < 0:
        raise ValueError("dual_head_alignment_window_stride must be >= 0")
    if args.dual_head_alignment_local_recon_weight < 0.0:
        raise ValueError("dual_head_alignment_local_recon_weight must be >= 0")
    if args.dual_head_alignment_local_behavior_weight < 0.0:
        raise ValueError("dual_head_alignment_local_behavior_weight must be >= 0")
    if args.dual_head_alignment_local_kl_weight < 0.0:
        raise ValueError("dual_head_alignment_local_kl_weight must be >= 0")
    if args.dual_head_alignment_local_div_forward_weight < 0.0:
        raise ValueError(
            "dual_head_alignment_local_div_forward_weight must be >= 0"
        )
    if args.dual_head_alignment_local_div_reverse_weight < 0.0:
        raise ValueError(
            "dual_head_alignment_local_div_reverse_weight must be >= 0"
        )
    if args.dual_head_alignment_temperature <= 0.0:
        raise ValueError("dual_head_alignment_temperature must be > 0")
    if args.dual_head_alignment_head_lr_scale <= 0.0:
        raise ValueError("dual_head_alignment_head_lr_scale must be > 0")
    if args.dual_head_alignment_start_step < 0:
        raise ValueError("dual_head_alignment_start_step must be >= 0")
    if args.dual_head_alignment_teacher_interval_steps < 1:
        raise ValueError("dual_head_alignment_teacher_interval_steps must be >= 1")
    if args.dual_head_alignment_local_weight_ramp_steps < 0:
        raise ValueError("dual_head_alignment_local_weight_ramp_steps must be >= 0")
    if args.dual_head_alignment_window_sampling not in _ALLOWED_WINDOW_SAMPLING:
        raise ValueError(
            "dual_head_alignment_window_sampling must be one of "
            f"{sorted(_ALLOWED_WINDOW_SAMPLING)}"
        )
    if args.dual_head_alignment_max_windows < 0:
        raise ValueError("dual_head_alignment_max_windows must be >= 0")
    if args.dual_head_alignment_teacher_mode not in _ALLOWED_TEACHER_MODES:
        raise ValueError(
            "dual_head_alignment_teacher_mode must be one of "
            f"{sorted(_ALLOWED_TEACHER_MODES)}"
        )

    logger.info(
        (
            "Dual-head alignment helper enabled "
            "(global=%.3f, local=%.3f, window_frames=%d, window_stride=%d, "
            "teacher_mode=%s, teacher_interval=%d, start_step=%d, "
            "window_sampling=%s, max_windows=%d)"
        ),
        args.dual_head_alignment_global_weight,
        args.dual_head_alignment_local_weight,
        args.dual_head_alignment_window_frames,
        args.dual_head_alignment_window_stride,
        args.dual_head_alignment_teacher_mode,
        args.dual_head_alignment_teacher_interval_steps,
        args.dual_head_alignment_start_step,
        args.dual_head_alignment_window_sampling,
        args.dual_head_alignment_max_windows,
    )

