from __future__ import annotations

from typing import Any, Dict


def apply_dype_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse DyPE-style dynamic positional extrapolation settings."""
    args.enable_dynamic_positional_extrapolation = bool(
        config.get("enable_dynamic_positional_extrapolation", False)
    )
    args.dynamic_positional_extrapolation_mode = str(
        config.get("dynamic_positional_extrapolation_mode", "rope_scale")
    ).lower()
    args.dynamic_positional_base_resolution = int(
        config.get("dynamic_positional_base_resolution", 1024)
    )
    args.dynamic_positional_max_scale = float(
        config.get("dynamic_positional_max_scale", 4.0)
    )
    args.dynamic_positional_activate_above_frames = int(
        config.get("dynamic_positional_activate_above_frames", 0)
    )

    if args.dynamic_positional_extrapolation_mode not in {"rope_scale"}:
        raise ValueError(
            "dynamic_positional_extrapolation_mode must be 'rope_scale'"
        )
    if args.dynamic_positional_base_resolution <= 0:
        raise ValueError("dynamic_positional_base_resolution must be > 0")
    if args.dynamic_positional_max_scale < 1.0:
        raise ValueError("dynamic_positional_max_scale must be >= 1.0")
    if args.dynamic_positional_activate_above_frames < 0:
        raise ValueError("dynamic_positional_activate_above_frames must be >= 0")

    if args.enable_dynamic_positional_extrapolation:
        logger.info(
            "DyPE-style dynamic positional extrapolation enabled "
            "(mode=%s, base_resolution=%d, max_scale=%.3f, activate_above_frames=%d).",
            args.dynamic_positional_extrapolation_mode,
            args.dynamic_positional_base_resolution,
            args.dynamic_positional_max_scale,
            args.dynamic_positional_activate_above_frames,
        )
