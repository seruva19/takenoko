from __future__ import annotations

from typing import Any, Dict


_MASK_RATIO_MODES = {"auto", "fixed"}
_LOSS_TYPES = {"negative_cosine", "one_minus_cosine"}
_TEACHER_MODES = {"ema", "base", "partial_ema"}
_PROJECTOR_ACTIVATIONS = {"silu", "gelu"}
_PROJECTOR_DESIGNS = {"plain_mlp", "rmsnorm_gelu_mlp"}
_TEMPORAL_MODES = {"off", "frame", "delta", "hybrid"}
_TEMPORAL_GRANULARITIES = {"frame", "patch"}
_PATCH_MATCH_MODES = {"hard", "soft"}
_MOTION_WEIGHTING_MODES = {"none", "teacher_delta"}
_TEMPORAL_SCHEDULES = {"constant", "linear", "cosine"}


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
    args.self_flow_frame_level_mask = bool(
        config.get("self_flow_frame_level_mask", False)
    )

    # Representation objective controls.
    args.self_flow_rep_loss_weight = float(config.get("self_flow_rep_loss_weight", 0.8))
    args.self_flow_rep_loss_type = str(
        config.get("self_flow_rep_loss_type", "negative_cosine")
    ).lower()
    args.self_flow_mask_focus_loss = bool(
        config.get("self_flow_mask_focus_loss", False)
    )
    args.self_flow_max_loss = float(config.get("self_flow_max_loss", 0.0))
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
    args.self_flow_projector_activation = str(
        config.get("self_flow_projector_activation", "silu")
    ).lower()
    args.self_flow_projector_design = str(
        config.get("self_flow_projector_design", "plain_mlp")
    ).lower()
    projector_lr = config.get("self_flow_projector_lr", None)
    if projector_lr is None:
        args.self_flow_projector_lr = None
    else:
        projector_lr_value = float(projector_lr)
        args.self_flow_projector_lr = (
            None if projector_lr_value <= 0.0 else projector_lr_value
        )
    args.self_flow_teacher_use_ema = bool(config.get("self_flow_teacher_use_ema", True))
    args.self_flow_teacher_mode = str(
        config.get("self_flow_teacher_mode", "ema")
    ).lower()
    args.self_flow_student_layer_stochastic_range = int(
        config.get("self_flow_student_layer_stochastic_range", 0)
    )
    args.self_flow_offload_teacher_features = bool(
        config.get("self_flow_offload_teacher_features", False)
    )
    args.self_flow_offload_teacher_params = bool(
        config.get("self_flow_offload_teacher_params", False)
    )
    args.self_flow_temporal_mode = str(
        config.get("self_flow_temporal_mode", "off")
    ).lower()
    args.self_flow_lambda_temporal = float(
        config.get("self_flow_lambda_temporal", 0.0)
    )
    args.self_flow_lambda_delta = float(config.get("self_flow_lambda_delta", 0.0))
    args.self_flow_temporal_tau = float(config.get("self_flow_temporal_tau", 1.0))
    args.self_flow_num_neighbors = int(config.get("self_flow_num_neighbors", 2))
    args.self_flow_temporal_granularity = str(
        config.get("self_flow_temporal_granularity", "frame")
    ).lower()
    args.self_flow_patch_spatial_radius = int(
        config.get("self_flow_patch_spatial_radius", 0)
    )
    args.self_flow_patch_match_mode = str(
        config.get("self_flow_patch_match_mode", "hard")
    ).lower()
    args.self_flow_patch_match_temperature = float(
        config.get("self_flow_patch_match_temperature", 0.1)
    )
    args.self_flow_delta_num_steps = int(config.get("self_flow_delta_num_steps", 1))
    args.self_flow_motion_weighting = str(
        config.get("self_flow_motion_weighting", "none")
    ).lower()
    args.self_flow_motion_weight_strength = float(
        config.get("self_flow_motion_weight_strength", 0.0)
    )
    args.self_flow_temporal_schedule = str(
        config.get("self_flow_temporal_schedule", "constant")
    ).lower()
    args.self_flow_temporal_warmup_steps = int(
        config.get("self_flow_temporal_warmup_steps", 0)
    )
    args.self_flow_temporal_max_steps = int(
        config.get("self_flow_temporal_max_steps", 0)
    )

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
    if args.self_flow_teacher_mode not in _TEACHER_MODES:
        raise ValueError(
            f"self_flow_teacher_mode must be one of {sorted(_TEACHER_MODES)}"
        )
    if args.self_flow_projector_activation not in _PROJECTOR_ACTIVATIONS:
        raise ValueError(
            "self_flow_projector_activation must be one of "
            f"{sorted(_PROJECTOR_ACTIVATIONS)}"
        )
    if args.self_flow_projector_design not in _PROJECTOR_DESIGNS:
        raise ValueError(
            "self_flow_projector_design must be one of "
            f"{sorted(_PROJECTOR_DESIGNS)}"
        )
    if args.self_flow_temporal_mode not in _TEMPORAL_MODES:
        raise ValueError(
            f"self_flow_temporal_mode must be one of {sorted(_TEMPORAL_MODES)}"
        )
    if args.self_flow_temporal_granularity not in _TEMPORAL_GRANULARITIES:
        raise ValueError(
            "self_flow_temporal_granularity must be one of "
            f"{sorted(_TEMPORAL_GRANULARITIES)}"
        )
    if args.self_flow_patch_match_mode not in _PATCH_MATCH_MODES:
        raise ValueError(
            f"self_flow_patch_match_mode must be one of {sorted(_PATCH_MATCH_MODES)}"
        )
    if args.self_flow_motion_weighting not in _MOTION_WEIGHTING_MODES:
        raise ValueError(
            "self_flow_motion_weighting must be one of "
            f"{sorted(_MOTION_WEIGHTING_MODES)}"
        )
    if args.self_flow_temporal_schedule not in _TEMPORAL_SCHEDULES:
        raise ValueError(
            "self_flow_temporal_schedule must be one of "
            f"{sorted(_TEMPORAL_SCHEDULES)}"
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
    if args.self_flow_max_loss < 0.0:
        raise ValueError("self_flow_max_loss must be >= 0")
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
    if args.self_flow_student_layer_stochastic_range < 0:
        raise ValueError("self_flow_student_layer_stochastic_range must be >= 0")
    if args.self_flow_lambda_temporal < 0.0:
        raise ValueError("self_flow_lambda_temporal must be >= 0")
    if args.self_flow_lambda_delta < 0.0:
        raise ValueError("self_flow_lambda_delta must be >= 0")
    if args.self_flow_temporal_tau <= 0.0:
        raise ValueError("self_flow_temporal_tau must be > 0")
    if args.self_flow_num_neighbors < 0:
        raise ValueError("self_flow_num_neighbors must be >= 0")
    if args.self_flow_patch_spatial_radius < 0:
        raise ValueError("self_flow_patch_spatial_radius must be >= 0")
    if args.self_flow_patch_match_temperature <= 0.0:
        raise ValueError("self_flow_patch_match_temperature must be > 0")
    if args.self_flow_delta_num_steps < 1:
        raise ValueError("self_flow_delta_num_steps must be >= 1")
    if args.self_flow_motion_weight_strength < 0.0:
        raise ValueError("self_flow_motion_weight_strength must be >= 0")
    if args.self_flow_temporal_warmup_steps < 0:
        raise ValueError("self_flow_temporal_warmup_steps must be >= 0")
    if args.self_flow_temporal_max_steps < 0:
        raise ValueError("self_flow_temporal_max_steps must be >= 0")

    # Strict mode keeps Self-Flow on the default reference configuration and
    # avoids known conflicts.
    if args.self_flow_strict_mode:
        if not args.self_flow_enable_dual_timestep:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_enable_dual_timestep=true."
            )
        if not args.self_flow_enable_feature_alignment:
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_enable_feature_alignment=true."
            )
        if args.self_flow_mask_ratio_mode != "auto":
            raise ValueError(
                "Self-Flow strict mode requires self_flow_mask_ratio_mode='auto' "
                "to preserve the default per-modality mask ratios."
            )
        if abs(args.self_flow_mask_ratio_image - 0.25) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_mask_ratio_image=0.25."
            )
        if abs(args.self_flow_mask_ratio_video - 0.10) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_mask_ratio_video=0.10."
            )
        if abs(args.self_flow_mask_ratio_audio - 0.50) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_mask_ratio_audio=0.50."
            )
        if not args.self_flow_force_tokenwise_timesteps:
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_force_tokenwise_timesteps=true."
            )
        if abs(args.self_flow_rep_loss_weight - 0.8) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_rep_loss_weight=0.8."
            )
        if args.self_flow_rep_loss_type != "negative_cosine":
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_rep_loss_type='negative_cosine'."
            )
        if args.self_flow_mask_focus_loss:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_mask_focus_loss=false."
            )
        if args.self_flow_max_loss > 0.0:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_max_loss=0.0."
            )
        if args.self_flow_teacher_mode != "ema":
            raise ValueError(
                "Self-Flow strict mode requires self_flow_teacher_mode='ema'."
            )
        if not args.self_flow_teacher_use_ema:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_teacher_use_ema=true."
            )
        if abs(args.self_flow_teacher_momentum - 0.9999) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_teacher_momentum=0.9999."
            )
        if args.self_flow_teacher_update_interval != 1:
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_teacher_update_interval=1."
            )
        if abs(args.self_flow_student_layer_ratio - 0.3) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_student_layer_ratio=0.3."
            )
        if abs(args.self_flow_teacher_layer_ratio - 0.7) > 1e-8:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_teacher_layer_ratio=0.7."
            )
        if args.self_flow_student_layer_index >= 1:
            raise ValueError(
                "Self-Flow strict mode requires ratio-based student layer "
                "selection (self_flow_student_layer_index < 1)."
            )
        if args.self_flow_teacher_layer_index >= 1:
            raise ValueError(
                "Self-Flow strict mode requires ratio-based teacher layer "
                "selection (self_flow_teacher_layer_index < 1)."
            )
        if args.self_flow_frame_level_mask:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_frame_level_mask=false."
            )
        if args.self_flow_student_layer_stochastic_range != 0:
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_student_layer_stochastic_range=0."
            )
        if args.self_flow_projector_design != "plain_mlp":
            raise ValueError(
                "Self-Flow strict mode requires "
                "self_flow_projector_design='plain_mlp'."
            )
        if args.self_flow_temporal_mode != "off":
            raise ValueError(
                "Self-Flow strict mode requires self_flow_temporal_mode='off'."
            )
        if args.self_flow_lambda_temporal > 0.0:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_lambda_temporal=0.0."
            )
        if args.self_flow_lambda_delta > 0.0:
            raise ValueError(
                "Self-Flow strict mode requires self_flow_lambda_delta=0.0."
            )
        if args.self_flow_motion_weighting != "none":
            raise ValueError(
                "Self-Flow strict mode requires self_flow_motion_weighting='none'."
            )
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
        "mask_fixed=%.3f mask_video=%.3f frame_mask=%s rep_weight=%.3f rep_type=%s "
        "teacher_mode=%s ema_teacher=%s momentum=%.4f layer_ratio=(%.2f->%.2f) "
        "mask_focus=%s max_loss=%.3f stochastic_range=%d projector=%s projector_act=%s "
        "projector_lr=%s offload_features=%s offload_params=%s "
        "temporal_mode=%s lambda_temporal=%.3f lambda_delta=%.3f granularity=%s "
        "neighbors=%d schedule=%s strict_mode=%s strict_defaults=%s",
        str(args.self_flow_enable_dual_timestep).lower(),
        str(args.self_flow_enable_feature_alignment).lower(),
        args.self_flow_mask_ratio_mode,
        args.self_flow_mask_ratio,
        args.self_flow_mask_ratio_video,
        str(args.self_flow_frame_level_mask).lower(),
        args.self_flow_rep_loss_weight,
        args.self_flow_rep_loss_type,
        args.self_flow_teacher_mode,
        str(args.self_flow_teacher_use_ema).lower(),
        args.self_flow_teacher_momentum,
        args.self_flow_student_layer_ratio,
        args.self_flow_teacher_layer_ratio,
        str(args.self_flow_mask_focus_loss).lower(),
        args.self_flow_max_loss,
        args.self_flow_student_layer_stochastic_range,
        args.self_flow_projector_design,
        args.self_flow_projector_activation,
        args.self_flow_projector_lr,
        str(args.self_flow_offload_teacher_features).lower(),
        str(args.self_flow_offload_teacher_params).lower(),
        args.self_flow_temporal_mode,
        args.self_flow_lambda_temporal,
        args.self_flow_lambda_delta,
        args.self_flow_temporal_granularity,
        args.self_flow_num_neighbors,
        args.self_flow_temporal_schedule,
        str(args.self_flow_strict_mode).lower(),
        str(args.self_flow_strict_mode).lower(),
    )
