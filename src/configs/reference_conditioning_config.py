"""Configuration parser for small reference-conditioning train-time features."""

from __future__ import annotations

from typing import Any, Dict

from enhancements.reference_conditioning.task_routing import (
    TRANSFER_TASK_TYPES_WITH_AUTO,
    normalize_transfer_task_type,
)

_ALLOWED_TPB_SCOPES = {"ic_lora"}


def apply_reference_conditioning_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse and validate small reference-conditioning feature flags."""

    args.enable_ic_lora_reference_positional_bias = bool(
        config.get("enable_ic_lora_reference_positional_bias", False)
    )
    args.ic_lora_reference_positional_bias_scope = str(
        config.get("ic_lora_reference_positional_bias_scope", "ic_lora")
    ).lower()
    if args.ic_lora_reference_positional_bias_scope not in _ALLOWED_TPB_SCOPES:
        raise ValueError(
            "ic_lora_reference_positional_bias_scope must be one of "
            f"{sorted(_ALLOWED_TPB_SCOPES)}, got "
            f"{args.ic_lora_reference_positional_bias_scope!r}"
        )

    raw_task_type = str(config.get("reference_conditioning_task_type", "auto")).lower()
    args.reference_conditioning_task_type = normalize_transfer_task_type(raw_task_type)
    if args.reference_conditioning_task_type not in TRANSFER_TASK_TYPES_WITH_AUTO:
        raise ValueError(
            "reference_conditioning_task_type must be one of "
            f"{sorted(TRANSFER_TASK_TYPES_WITH_AUTO)}, got "
            f"{args.reference_conditioning_task_type!r}"
        )

    args.ic_lora_reference_temporal_position_bias_scale = float(
        config.get("ic_lora_reference_temporal_position_bias_scale", 1.0)
    )
    if args.ic_lora_reference_temporal_position_bias_scale < 0:
        raise ValueError(
            "ic_lora_reference_temporal_position_bias_scale must be >= 0, got "
            f"{args.ic_lora_reference_temporal_position_bias_scale}"
        )

    args.ic_lora_reference_width_position_bias_scale = float(
        config.get("ic_lora_reference_width_position_bias_scale", 1.0)
    )
    if args.ic_lora_reference_width_position_bias_scale < 0:
        raise ValueError(
            "ic_lora_reference_width_position_bias_scale must be >= 0, got "
            f"{args.ic_lora_reference_width_position_bias_scale}"
        )

    args.enable_task_type_conditioning_tokens = bool(
        config.get("enable_task_type_conditioning_tokens", False)
    )
    args.task_type_conditioning_token_scale = float(
        config.get("task_type_conditioning_token_scale", 1.0)
    )
    if args.task_type_conditioning_token_scale < 0:
        raise ValueError(
            "task_type_conditioning_token_scale must be >= 0, got "
            f"{args.task_type_conditioning_token_scale}"
        )

    if (
        args.enable_ic_lora_reference_positional_bias
        and args.ic_lora_reference_positional_bias_scope == "ic_lora"
        and not bool(getattr(args, "enable_ic_lora", False))
    ):
        logger.warning(
            "enable_ic_lora_reference_positional_bias=true currently supports IC-LoRA only; "
            "disabling because enable_ic_lora is false."
        )
        args.enable_ic_lora_reference_positional_bias = False

    if args.enable_ic_lora_reference_positional_bias:
        logger.info(
            "IC-LoRA reference positional bias enabled (scope=%s, task_type=%s, temporal_scale=%.3f, width_scale=%.3f).",
            args.ic_lora_reference_positional_bias_scope,
            args.reference_conditioning_task_type,
            args.ic_lora_reference_temporal_position_bias_scale,
            args.ic_lora_reference_width_position_bias_scale,
        )

    if args.enable_task_type_conditioning_tokens:
        logger.info(
            "Task-type conditioning tokens enabled (task_type=%s, scale=%.3f).",
            args.reference_conditioning_task_type,
            args.task_type_conditioning_token_scale,
        )
