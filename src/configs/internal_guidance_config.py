"""Internal Guidance (IG) config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


_ALLOWED_IG_LOSSES = {"sml1", "l1", "l2"}
_ALLOWED_IG_MODES = {"aux", "shift"}


def apply_internal_guidance_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.enable_internal_guidance = bool(
        config.get("enable_internal_guidance", False)
    )
    args.internal_guidance_weight = float(
        config.get("internal_guidance_weight", 0.5)
    )
    args.internal_guidance_intermediate_weight = float(
        config.get("internal_guidance_intermediate_weight", 1.0)
    )
    args.internal_guidance_target_block = int(
        config.get("internal_guidance_target_block", 8)
    )
    args.internal_guidance_mode = str(
        config.get("internal_guidance_mode", "shift")
    ).lower()
    args.internal_guidance_loss_type = str(
        config.get("internal_guidance_loss_type", "l2")
    ).lower()

    if not args.enable_internal_guidance:
        return

    if args.internal_guidance_weight <= 0.0:
        raise ValueError(
            "internal_guidance_weight must be > 0 when Internal Guidance is enabled"
        )
    if args.internal_guidance_intermediate_weight < 0.0:
        raise ValueError("internal_guidance_intermediate_weight must be >= 0")
    if args.internal_guidance_target_block < 0:
        raise ValueError("internal_guidance_target_block must be >= 0")
    if args.internal_guidance_mode not in _ALLOWED_IG_MODES:
        raise ValueError(
            f"internal_guidance_mode must be one of {_ALLOWED_IG_MODES}"
        )
    if args.internal_guidance_loss_type not in _ALLOWED_IG_LOSSES:
        raise ValueError(
            f"internal_guidance_loss_type must be one of {_ALLOWED_IG_LOSSES}"
        )

    logger.info(
        "Internal Guidance enabled (mode=%s, weight=%.4f, intermediate_weight=%.4f, target_block=%d, loss=%s)",
        args.internal_guidance_mode,
        args.internal_guidance_weight,
        args.internal_guidance_intermediate_weight,
        args.internal_guidance_target_block,
        args.internal_guidance_loss_type,
    )
