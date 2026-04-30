from __future__ import annotations

import math
from typing import Any, Dict


_REFERENCE_MODES = {"checkpoint", "zeros"}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def apply_neon_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Neon negative-extrapolation checkpoint export settings."""

    args.enable_neon_negative_extrapolation = _parse_bool(
        config.get("enable_neon_negative_extrapolation", False),
        False,
    )
    args.neon_reference_mode = str(
        config.get("neon_reference_mode", "checkpoint")
    ).strip().lower()
    args.neon_reference_lora_path = str(
        config.get("neon_reference_lora_path", "")
    ).strip()
    args.neon_extrapolation_weight = float(
        config.get("neon_extrapolation_weight", 1.0)
    )
    args.neon_min_match_ratio = float(config.get("neon_min_match_ratio", 0.8))
    args.neon_merge_alpha = _parse_bool(config.get("neon_merge_alpha", False), False)
    args.neon_allow_shape_mismatch = _parse_bool(
        config.get("neon_allow_shape_mismatch", False),
        False,
    )

    if args.neon_reference_mode not in _REFERENCE_MODES:
        raise ValueError(
            "neon_reference_mode must be one of "
            f"{sorted(_REFERENCE_MODES)}, got {args.neon_reference_mode!r}"
        )
    if not math.isfinite(args.neon_extrapolation_weight):
        raise ValueError("neon_extrapolation_weight must be finite")
    if args.neon_extrapolation_weight <= 0.0:
        raise ValueError("neon_extrapolation_weight must be > 0")
    if not (0.0 <= args.neon_min_match_ratio <= 1.0):
        raise ValueError("neon_min_match_ratio must be in [0.0, 1.0]")
    if (
        args.enable_neon_negative_extrapolation
        and args.neon_reference_mode == "checkpoint"
        and not args.neon_reference_lora_path
    ):
        raise ValueError(
            "neon_reference_lora_path is required when "
            "enable_neon_negative_extrapolation=true and "
            "neon_reference_mode='checkpoint'"
        )

    if args.enable_neon_negative_extrapolation:
        logger.info(
            "Neon negative extrapolation export enabled "
            "(mode=%s, weight=%.4f, min_match_ratio=%.2f, merge_alpha=%s).",
            args.neon_reference_mode,
            args.neon_extrapolation_weight,
            args.neon_min_match_ratio,
            args.neon_merge_alpha,
        )
