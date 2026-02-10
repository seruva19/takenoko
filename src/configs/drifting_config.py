"""Drifting loss config parsing and validation."""

from __future__ import annotations

import os
from typing import Any, Dict, List


def _parse_int_list(value: Any, key_name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        parsed: List[int] = []
        for item in value:
            parsed.append(int(item))
        return parsed
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        if not chunks:
            return []
        return [int(chunk) for chunk in chunks]
    raise ValueError(
        f"{key_name} must be an int, list of ints, or comma-separated string."
    )


def apply_drifting_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Drifting settings onto args (training-only, default off)."""
    args.enable_drifting = bool(config.get("enable_drifting", False))
    args.drifting_loss_weight = float(config.get("drifting_loss_weight", 0.05))
    args.drifting_temperature = float(config.get("drifting_temperature", 0.05))
    args.drifting_max_feature_dim = int(config.get("drifting_max_feature_dim", 1024))
    args.drifting_min_batch_size = int(config.get("drifting_min_batch_size", 2))
    args.drifting_use_dual_axis_normalization = bool(
        config.get("drifting_use_dual_axis_normalization", True)
    )
    args.drifting_multi_scale_enabled = bool(
        config.get("drifting_multi_scale_enabled", False)
    )
    args.drifting_multi_scale_factors = _parse_int_list(
        config.get("drifting_multi_scale_factors", [1, 2, 4]),
        "drifting_multi_scale_factors",
    )
    args.drifting_multi_scale_reduce = str(
        config.get("drifting_multi_scale_reduce", "mean")
    ).lower()

    args.drifting_queue_enabled = bool(config.get("drifting_queue_enabled", False))
    args.drifting_queue_size_per_label = int(
        config.get("drifting_queue_size_per_label", 128)
    )
    args.drifting_queue_size_global = int(config.get("drifting_queue_size_global", 1000))
    args.drifting_queue_neg_size_global = int(
        config.get("drifting_queue_neg_size_global", 1000)
    )
    args.drifting_queue_sample_per_step = int(
        config.get("drifting_queue_sample_per_step", 64)
    )
    args.drifting_queue_warmup_steps = int(config.get("drifting_queue_warmup_steps", 0))

    args.drifting_cfg_weighting_enabled = bool(
        config.get("drifting_cfg_weighting_enabled", False)
    )
    args.drifting_cfg_null_label = int(config.get("drifting_cfg_null_label", -1))
    args.drifting_cfg_conditional_weight = float(
        config.get("drifting_cfg_conditional_weight", 1.0)
    )
    args.drifting_cfg_unconditional_weight = float(
        config.get("drifting_cfg_unconditional_weight", 1.0)
    )
    args.drifting_cfg_apply_to_positive = bool(
        config.get("drifting_cfg_apply_to_positive", True)
    )
    args.drifting_cfg_apply_to_negative = bool(
        config.get("drifting_cfg_apply_to_negative", True)
    )
    args.drifting_cfg_use_batch_weight = bool(
        config.get("drifting_cfg_use_batch_weight", False)
    )
    args.drifting_feature_encoder_enabled = bool(
        config.get("drifting_feature_encoder_enabled", False)
    )
    args.drifting_feature_encoder_path = config.get(
        "drifting_feature_encoder_path", None
    )
    args.drifting_feature_encoder_input_size = int(
        config.get("drifting_feature_encoder_input_size", 224)
    )
    args.drifting_feature_encoder_frame_reduce = str(
        config.get("drifting_feature_encoder_frame_reduce", "mean")
    ).lower()
    args.drifting_feature_encoder_channel_mode = str(
        config.get("drifting_feature_encoder_channel_mode", "first3")
    ).lower()
    args.drifting_feature_encoder_use_fp16 = bool(
        config.get("drifting_feature_encoder_use_fp16", True)
    )
    args.drifting_feature_encoder_imagenet_norm = bool(
        config.get("drifting_feature_encoder_imagenet_norm", False)
    )
    args.drifting_feature_encoder_strict = bool(
        config.get("drifting_feature_encoder_strict", False)
    )

    if args.drifting_loss_weight < 0.0:
        raise ValueError("drifting_loss_weight must be >= 0")
    if args.drifting_temperature <= 0.0:
        raise ValueError("drifting_temperature must be > 0")
    if args.drifting_max_feature_dim < 1:
        raise ValueError("drifting_max_feature_dim must be >= 1")
    if args.drifting_min_batch_size < 2:
        raise ValueError("drifting_min_batch_size must be >= 2")
    if not args.drifting_multi_scale_factors:
        raise ValueError("drifting_multi_scale_factors must include at least one scale")
    if any(int(scale) < 1 for scale in args.drifting_multi_scale_factors):
        raise ValueError("drifting_multi_scale_factors values must be >= 1")
    args.drifting_multi_scale_factors = sorted(set(args.drifting_multi_scale_factors))
    if args.drifting_multi_scale_reduce not in ("mean", "sum"):
        raise ValueError("drifting_multi_scale_reduce must be one of: mean, sum")

    if args.drifting_queue_size_per_label < 0:
        raise ValueError("drifting_queue_size_per_label must be >= 0")
    if args.drifting_queue_size_global < 0:
        raise ValueError("drifting_queue_size_global must be >= 0")
    if args.drifting_queue_neg_size_global < 0:
        raise ValueError("drifting_queue_neg_size_global must be >= 0")
    if args.drifting_queue_sample_per_step < 0:
        raise ValueError("drifting_queue_sample_per_step must be >= 0")
    if args.drifting_queue_warmup_steps < 0:
        raise ValueError("drifting_queue_warmup_steps must be >= 0")

    if args.drifting_cfg_conditional_weight < 0.0:
        raise ValueError("drifting_cfg_conditional_weight must be >= 0")
    if args.drifting_cfg_unconditional_weight < 0.0:
        raise ValueError("drifting_cfg_unconditional_weight must be >= 0")
    if args.drifting_feature_encoder_input_size < 16:
        raise ValueError("drifting_feature_encoder_input_size must be >= 16")
    if args.drifting_feature_encoder_frame_reduce not in ("mean", "max", "middle"):
        raise ValueError(
            "drifting_feature_encoder_frame_reduce must be one of: mean, max, middle"
        )
    if args.drifting_feature_encoder_channel_mode not in ("first3", "mean_to_rgb"):
        raise ValueError(
            "drifting_feature_encoder_channel_mode must be one of: first3, mean_to_rgb"
        )

    if args.drifting_feature_encoder_enabled:
        if not args.drifting_feature_encoder_path:
            raise ValueError(
                "drifting_feature_encoder_path is required when drifting_feature_encoder_enabled=true"
            )
        if not os.path.exists(str(args.drifting_feature_encoder_path)):
            raise ValueError(
                "drifting_feature_encoder_path does not exist: "
                f"{args.drifting_feature_encoder_path}"
            )

    if args.enable_drifting and args.drifting_loss_weight == 0.0:
        logger.warning(
            "Drifting is enabled but drifting_loss_weight is 0.0; the auxiliary loss has no effect."
        )

    if args.enable_drifting:
        logger.info(
            "Drifting enabled (weight=%.6f, temperature=%.5f, max_feature_dim=%d, min_batch_size=%d, dual_axis_norm=%s, multi_scale=%s factors=%s reduce=%s, queue_enabled=%s queue_sample=%d warmup=%d, cfg_weighting=%s null_label=%d cond_w=%.3f uncond_w=%.3f).",
            args.drifting_loss_weight,
            args.drifting_temperature,
            args.drifting_max_feature_dim,
            args.drifting_min_batch_size,
            args.drifting_use_dual_axis_normalization,
            args.drifting_multi_scale_enabled,
            args.drifting_multi_scale_factors,
            args.drifting_multi_scale_reduce,
            args.drifting_queue_enabled,
            args.drifting_queue_sample_per_step,
            args.drifting_queue_warmup_steps,
            args.drifting_cfg_weighting_enabled,
            args.drifting_cfg_null_label,
            args.drifting_cfg_conditional_weight,
            args.drifting_cfg_unconditional_weight,
        )
        if args.drifting_feature_encoder_enabled:
            logger.info(
                "Drifting feature encoder enabled (path=%s, input_size=%d, frame_reduce=%s, channel_mode=%s, use_fp16=%s, imagenet_norm=%s, strict=%s).",
                args.drifting_feature_encoder_path,
                args.drifting_feature_encoder_input_size,
                args.drifting_feature_encoder_frame_reduce,
                args.drifting_feature_encoder_channel_mode,
                args.drifting_feature_encoder_use_fp16,
                args.drifting_feature_encoder_imagenet_norm,
                args.drifting_feature_encoder_strict,
            )
