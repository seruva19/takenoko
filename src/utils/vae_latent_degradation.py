from __future__ import annotations

from typing import Literal

import torch

from criteria.hfato_loss import hfato_degrade


_VALID_MODES = ("bilinear", "bicubic", "area", "nearest")


def apply_vae_latent_degradation(
    latents: torch.Tensor,
    ratio: float = 0.5,
    mode: Literal["bilinear", "bicubic", "area", "nearest"] = "bilinear",
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Apply a lightweight refinement-time degradation proxy to VAE latents.

    The first-pass refinement proxy is intentionally simple:
    1. spatial downsample-upsample degradation using the existing HFATO helper
    2. optional additive Gaussian noise

    This keeps the latent tensor shape unchanged and is safe for decoder-only
    VAE refinement, where the encoder remains frozen.
    """
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"vae latent degradation ratio must be in (0, 1), got {ratio}")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"vae latent degradation mode must be one of {_VALID_MODES}, got {mode!r}"
        )
    if noise_std < 0.0:
        raise ValueError(
            f"vae latent degradation noise_std must be >= 0, got {noise_std}"
        )

    degraded = hfato_degrade(latents, ratio=ratio, mode=mode)
    if noise_std > 0.0:
        degraded = degraded + torch.randn_like(degraded) * float(noise_std)
    return degraded
