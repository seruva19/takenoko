from __future__ import annotations

from typing import Any, Dict


_MASK_RATIO_MODES = {"auto", "fixed"}
_LOSS_TYPES = {"negative_cosine", "one_minus_cosine"}


def apply_self_flow_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse and validate Self-Flow train-time settings."""
    args.enable_self_flow = bool(config.get("enable_self_flow", False))

    # Core method toggles.
    args.self_flow_enable_dual_timestep = bool(
        config.get("self_flow_enable_dual_timestep", True)
    )
    args.self_flow_enable_feature_alignment = bool(
        config.get("self_flow_enable_feature_alignment", True)
    )
    args.self_flow_strict_mode = bool(config.get("self_flow_strict_mode", True))

    # Dual-timestep masking controls.
    args.self_flow_mask_ratio_mode = str(
        config.get("self_flow_mask_ratio_mode", "auto")
    ).lower()
    args.self_flow_mask_ratio = float(config.get("self_flow_mask_ratio", 0.25))
    args.self_flow_mask_ratio_image = float(
        config.get("self_flow_mask_ratio_image", 0.25)
    )
    args.self_flow_mask_ratio_video = float(
        config.get("self_flow_mask_ratio_video", 0.10)
    )
    args.self_flow_mask_ratio_audio = float(
        config.get("self_flow_mask_ratio_audio", 0.50)
    )
    args.self_flow_force_tokenwise_timesteps = bool(
        config.get("self_flow_force_tokenwise_timesteps", True)
    )
    args.self_flow_timestep_upper_bound = float(
        config.get("self_flow_timestep_upper_bound", 0.999)
    )
    args.self_flow_timestep_lower_bound = float(
        config.get("self_flow_timestep_lower_bound", 0.001)
    )

    # Representation objective controls.
    args.self_flow_rep_loss_weight = float(config.get("self_flow_rep_loss_weight", 0.8))
    args.self_flow_rep_loss_type = str(
        config.get("self_flow_rep_loss_type", "negative_cosine")
    ).lower()
    args.self_flow_teacher_momentum = float(
        config.get("self_flow_teacher_momentum", 0.9999)
    )
    args.self_flow_teacher_update_interval = int(
        config.get("self_flow_teacher_update_interval", 1)
    )
    args.self_flow_student_layer_ratio = float(
        config.get("self_flow_student_layer_ratio", 0.3)
    )
    args.self_flow_teacher_layer_ratio = float(
        config.get("self_flow_teacher_layer_ratio", 0.7)
    )
    args.self_flow_student_layer_index = int(
        config.get("self_flow_student_layer_index", -1)
    )
    args.self_flow_teacher_layer_index = int(
        config.get("self_flow_teacher_layer_index", -1)
    )
    args.self_flow_projection_hidden_multiplier = int(
        config.get("self_flow_projection_hidden_multiplier", 1)
    )
    args.self_flow_teacher_use_ema = bool(config.get("self_flow_teacher_use_ema", True))

    if not args.enable_self_flow:
        return

    if not args.self_flow_enable_dual_timestep and not args.self_flow_enable_feature_alignment:
        raise ValueError(
            "enable_self_flow=true requires at least one of "
            "self_flow_enable_dual_timestep or self_flow_enable_feature_alignment."
        )

    if args.self_flow_mask_ratio_mode not in _MASK_RATIO_MODES:
        raise ValueError(
            f"self_flow_mask_ratio_mode must be one of {sorted(_MASK_RATIO_MODES)}"
        )

    if args.self_flow_rep_loss_type not in _LOSS_TYPES:
        raise ValueError(
            f"self_flow_rep_loss_type must be one of {sorted(_LOSS_TYPES)}"
        )

    if not (0.0 <= args.self_flow_mask_ratio <= 0.5):
        raise ValueError("self_flow_mask_ratio must be in [0.0, 0.5]")
    if not (0.0 <= args.self_flow_mask_ratio_image <= 0.5):
        raise ValueError("self_flow_mask_ratio_image must be in [0.0, 0.5]")
    if not (0.0 <= args.self_flow_mask_ratio_video <= 0.5):
        raise ValueError("self_flow_mask_ratio_video must be in [0.0, 0.5]")
    if not (0.0 <= args.self_flow_mask_ratio_audio <= 0.5):
        raise ValueError("self_flow_mask_ratio_audio must be in [0.0, 0.5]")
    if not (0.0 <= args.self_flow_timestep_lower_bound < 1.0):
        raise ValueError("self_flow_timestep_lower_bound must be in [0.0, 1.0)")
    if not (0.0 < args.self_flow_timestep_upper_bound <= 1.0):
        raise ValueError("self_flow_timestep_upper_bound must be in (0.0, 1.0]")
    if args.self_flow_timestep_lower_bound >= args.self_flow_timestep_upper_bound:
        raise ValueError(
            "self_flow_timestep_lower_bound must be < self_flow_timestep_upper_bound"
        )

    if args.self_flow_rep_loss_weight < 0.0:
        raise ValueError("self_flow_rep_loss_weight must be >= 0")
    if not (0.0 <= args.self_flow_teacher_momentum < 1.0):
        raise ValueError("self_flow_teacher_momentum must be in [0.0, 1.0)")
    if args.self_flow_teacher_update_interval <= 0:
        raise ValueError("self_flow_teacher_update_interval must be > 0")
    if not (0.0 < args.self_flow_student_layer_ratio <= 1.0):
        raise ValueError("self_flow_student_layer_ratio must be in (0.0, 1.0]")
    if not (0.0 < args.self_flow_teacher_layer_ratio <= 1.0):
        raise ValueError("self_flow_teacher_layer_ratio must be in (0.0, 1.0]")
    if args.self_flow_teacher_layer_ratio <= args.self_flow_student_layer_ratio:
        raise ValueError(
            "self_flow_teacher_layer_ratio must be > self_flow_student_layer_ratio"
        )
    if args.self_flow_projection_hidden_multiplier < 1:
        raise ValueError("self_flow_projection_hidden_multiplier must be >= 1")

    # Strict mode keeps method close to the paper and avoids known conflicts.
    if args.self_flow_strict_mode:
        if bool(config.get("enable_fvdm", False)):
            raise ValueError(
                "Self-Flow strict mode is incompatible with enable_fvdm=true."
            )
        if bool(config.get("enable_mixflow", False)):
            raise ValueError(
                "Self-Flow strict mode is incompatible with enable_mixflow=true."
            )
        if bool(config.get("enable_self_resampling", False)):
            raise ValueError(
                "Self-Flow strict mode is incompatible with enable_self_resampling=true."
            )
        if bool(config.get("enable_reflexflow", False)):
            raise ValueError(
                "Self-Flow strict mode is incompatible with enable_reflexflow=true."
            )
        if bool(config.get("enable_error_recycling", False)):
            raise ValueError(
                "Self-Flow strict mode is incompatible with enable_error_recycling=true."
            )
        if bool(config.get("broadcast_time_embed", False)):
            raise ValueError(
                "Self-Flow strict mode requires broadcast_time_embed=false "
                "to preserve tokenwise timestep conditioning."
            )

    logger.info(
        "Self-Flow enabled: dual_timestep=%s feature_alignment=%s mask_mode=%s "
        "mask_fixed=%.3f mask_video=%.3f rep_weight=%.3f rep_type=%s "
        "ema_teacher=%s momentum=%.4f layer_ratio=(%.2f->%.2f) strict_mode=%s",
        str(args.self_flow_enable_dual_timestep).lower(),
        str(args.self_flow_enable_feature_alignment).lower(),
        args.self_flow_mask_ratio_mode,
        args.self_flow_mask_ratio,
        args.self_flow_mask_ratio_video,
        args.self_flow_rep_loss_weight,
        args.self_flow_rep_loss_type,
        str(args.self_flow_teacher_use_ema).lower(),
        args.self_flow_teacher_momentum,
        args.self_flow_student_layer_ratio,
        args.self_flow_teacher_layer_ratio,
        str(args.self_flow_strict_mode).lower(),
    )
