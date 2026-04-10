from __future__ import annotations

from typing import Any, Dict


def apply_vae_refinement_validation_profile(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Apply VAE refinement validation defaults without overriding explicit config."""

    profile = str(getattr(args, "vae_refinement_validation_profile", "none")).lower()
    if profile == "none":
        return
    if profile != "visual_quality":
        raise ValueError(
            f"Unsupported VAE refinement validation profile: {profile}"
        )

    if "load_val_pixels" not in config:
        args.load_val_pixels = True
    if "enable_lpips" not in config:
        args.enable_lpips = True
    if "lpips_network" not in config:
        args.lpips_network = "alex"
    if "lpips_frame_stride" not in config:
        args.lpips_frame_stride = 4
    if "enable_sensecraft_validation_metrics" not in config:
        args.enable_sensecraft_validation_metrics = True
    if "sensecraft_validation_metrics" not in config:
        args.sensecraft_validation_metrics = ["lpips", "ssim"]
    if "sensecraft_validation_lpips_network" not in config:
        args.sensecraft_validation_lpips_network = "alex"
    if "sensecraft_validation_frame_stride" not in config:
        args.sensecraft_validation_frame_stride = 4

    logger.info(
        "Applied VAE refinement validation profile '%s' (load_val_pixels=%s, enable_lpips=%s, enable_sensecraft_validation_metrics=%s, sensecraft_validation_metrics=%s).",
        profile,
        str(getattr(args, "load_val_pixels", False)).lower(),
        str(getattr(args, "enable_lpips", False)).lower(),
        str(getattr(args, "enable_sensecraft_validation_metrics", False)).lower(),
        getattr(args, "sensecraft_validation_metrics", []),
    )
