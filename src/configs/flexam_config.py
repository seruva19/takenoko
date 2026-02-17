from __future__ import annotations

from typing import Any, Dict


def apply_flexam_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse FlexAM-style train-time conditioning settings onto args."""
    args.enable_flexam_training = bool(config.get("enable_flexam_training", False))
    args.flexam_control_weight = float(config.get("flexam_control_weight", 1.0))
    args.flexam_mask_weight = float(config.get("flexam_mask_weight", 0.0))
    args.flexam_reference_weight = float(config.get("flexam_reference_weight", 0.0))
    args.flexam_reference_source = str(
        config.get("flexam_reference_source", "latents_first_frame")
    ).lower()
    args.flexam_reference_dropout_p = float(
        config.get("flexam_reference_dropout_p", 0.0)
    )
    args.flexam_additional_control_weight = float(
        config.get("flexam_additional_control_weight", 0.0)
    )
    args.flexam_additional_control_key = str(
        config.get("flexam_additional_control_key", "")
    )
    raw_flexam_additional_keys = config.get("flexam_additional_control_keys", [])
    if raw_flexam_additional_keys is None:
        raw_flexam_additional_keys = []
    if isinstance(raw_flexam_additional_keys, str):
        args.flexam_additional_control_keys = [
            part.strip()
            for part in raw_flexam_additional_keys.split(",")
            if part.strip()
        ]
    elif isinstance(raw_flexam_additional_keys, list):
        args.flexam_additional_control_keys = [
            str(part).strip() for part in raw_flexam_additional_keys if str(part).strip()
        ]
    else:
        raise ValueError(
            "flexam_additional_control_keys must be a list[str], a comma-separated string, or null"
        )
    args.flexam_additional_control_reduce = str(
        config.get("flexam_additional_control_reduce", "sum")
    ).lower()
    args.flexam_use_density_timestep_scaling = bool(
        config.get("flexam_use_density_timestep_scaling", False)
    )
    args.flexam_density_batch_key = str(
        config.get("flexam_density_batch_key", "density")
    )
    args.flexam_density_default = float(config.get("flexam_density_default", 1.0))
    args.flexam_density_min = float(config.get("flexam_density_min", 0.25))
    args.flexam_density_max = float(config.get("flexam_density_max", 4.0))
    args.flexam_weight_schedule = str(
        config.get("flexam_weight_schedule", "constant")
    ).lower()
    args.flexam_schedule_start_step = int(config.get("flexam_schedule_start_step", 0))
    args.flexam_schedule_end_step = int(config.get("flexam_schedule_end_step", 0))
    args.flexam_schedule_min_scale = float(config.get("flexam_schedule_min_scale", 0.0))
    args.flexam_schedule_max_scale = float(config.get("flexam_schedule_max_scale", 1.0))

    allowed_flexam_reference_sources = {
        "latents_first_frame",
        "noisy_first_frame",
        "control_first_frame",
    }
    if args.flexam_reference_source not in allowed_flexam_reference_sources:
        raise ValueError(
            "flexam_reference_source must be one of "
            f"{sorted(allowed_flexam_reference_sources)}, got {args.flexam_reference_source!r}"
        )
    if not 0.0 <= args.flexam_reference_dropout_p <= 1.0:
        raise ValueError("flexam_reference_dropout_p must be in [0, 1]")
    if args.flexam_density_min <= 0.0:
        raise ValueError("flexam_density_min must be > 0")
    if args.flexam_density_max < args.flexam_density_min:
        raise ValueError("flexam_density_max must be >= flexam_density_min")
    if args.flexam_additional_control_reduce not in {"sum", "mean"}:
        raise ValueError("flexam_additional_control_reduce must be 'sum' or 'mean'")
    if args.flexam_weight_schedule not in {"constant", "linear_ramp", "cosine_ramp"}:
        raise ValueError(
            "flexam_weight_schedule must be one of {'constant', 'linear_ramp', 'cosine_ramp'}"
        )
    if args.flexam_schedule_start_step < 0 or args.flexam_schedule_end_step < 0:
        raise ValueError(
            "flexam_schedule_start_step and flexam_schedule_end_step must be >= 0"
        )
    if args.flexam_schedule_max_scale < args.flexam_schedule_min_scale:
        raise ValueError(
            "flexam_schedule_max_scale must be >= flexam_schedule_min_scale"
        )
    if args.flexam_schedule_min_scale < 0.0:
        raise ValueError("flexam_schedule_min_scale must be >= 0")
    if args.enable_flexam_training and args.enable_control_lora:
        raise ValueError(
            "enable_flexam_training cannot be combined with enable_control_lora. "
            "Both features require the same control-concat path; use one mode at a time."
        )
    if args.enable_flexam_training and str(args.network_module) == "networks.ic_lora_wan":
        raise ValueError(
            "enable_flexam_training is incompatible with network_module='networks.ic_lora_wan'."
        )
    if args.enable_flexam_training:
        logger.info(
            "FlexAM training conditioning enabled (control_w=%.3f, mask_w=%.3f, ref_w=%.3f, ref_source=%s, density_scaling=%s, schedule=%s).",
            args.flexam_control_weight,
            args.flexam_mask_weight,
            args.flexam_reference_weight,
            args.flexam_reference_source,
            args.flexam_use_density_timestep_scaling,
            args.flexam_weight_schedule,
        )
