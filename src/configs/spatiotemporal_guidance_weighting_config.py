from __future__ import annotations

from typing import Any, Dict


_ALLOWED_ST_GUIDANCE_LOSS_TYPES = {"mse", "l1", "smooth_l1", "sml1"}


def apply_spatiotemporal_guidance_weighting_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse URSA-inspired spatiotemporal guidance weighting settings."""
    args.enable_spatiotemporal_guidance_weighting = bool(
        config.get("enable_spatiotemporal_guidance_weighting", False)
    )
    args.spatiotemporal_guidance_scale = float(
        config.get("spatiotemporal_guidance_scale", 0.1)
    )
    args.spatiotemporal_guidance_anchor_weight = float(
        config.get("spatiotemporal_guidance_anchor_weight", 1.0)
    )
    args.spatiotemporal_guidance_temporal_weight = float(
        config.get("spatiotemporal_guidance_temporal_weight", 1.0)
    )
    args.spatiotemporal_guidance_loss_type = str(
        config.get("spatiotemporal_guidance_loss_type", "mse")
    ).lower()

    if args.spatiotemporal_guidance_scale < 0.0:
        raise ValueError("spatiotemporal_guidance_scale must be >= 0")
    if args.spatiotemporal_guidance_anchor_weight < 0.0:
        raise ValueError("spatiotemporal_guidance_anchor_weight must be >= 0")
    if args.spatiotemporal_guidance_temporal_weight < 0.0:
        raise ValueError("spatiotemporal_guidance_temporal_weight must be >= 0")
    if (
        args.spatiotemporal_guidance_anchor_weight <= 0.0
        and args.spatiotemporal_guidance_temporal_weight <= 0.0
    ):
        raise ValueError(
            "spatiotemporal_guidance_anchor_weight and "
            "spatiotemporal_guidance_temporal_weight cannot both be <= 0"
        )
    if (
        args.spatiotemporal_guidance_loss_type
        not in _ALLOWED_ST_GUIDANCE_LOSS_TYPES
    ):
        raise ValueError(
            "spatiotemporal_guidance_loss_type must be one of "
            f"{sorted(_ALLOWED_ST_GUIDANCE_LOSS_TYPES)}"
        )
    if (
        args.enable_spatiotemporal_guidance_weighting
        and args.spatiotemporal_guidance_scale <= 0.0
    ):
        raise ValueError(
            "spatiotemporal_guidance_scale must be > 0 when "
            "enable_spatiotemporal_guidance_weighting is true"
        )

    if args.enable_spatiotemporal_guidance_weighting:
        logger.info(
            "Spatiotemporal guidance weighting enabled: scale=%.4f, "
            "anchor_weight=%.3f, temporal_weight=%.3f, loss_type=%s",
            args.spatiotemporal_guidance_scale,
            args.spatiotemporal_guidance_anchor_weight,
            args.spatiotemporal_guidance_temporal_weight,
            args.spatiotemporal_guidance_loss_type,
        )
