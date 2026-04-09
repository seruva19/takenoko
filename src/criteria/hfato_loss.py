"""
The High-Frequency-Awareness Training Objective (HFATO) replaces the clean
latent x_0 with a downsample-upsample degraded version DU(x_0) inside the
noisy model input, then supervises x_0 reconstruction against the ORIGINAL
undegraded latent:
    x_t   = (1 - σ_t) * DU(x_0) + σ_t * ε
    loss  = ‖x_0 − (x_t − σ_t * v_θ(x_t, t))‖²
forcing the network to recover high-frequency detail during denoising.

ViBe uses HFATO as Stage 2 of a two-stage "Relay LoRA" recipe:
  Stage 1: plain FM on images, rank 32, LoRA targets q/k/v/o/ffn.0/ffn.2.
            Saved adapter absorbs the image-mode drift; never loaded at
            inference. Config: configs/examples/vibe_stage1.toml.
  Stage 2: HFATO enabled, Stage 1 loaded as frozen_network_weights, new
            LoRA trains on top on higher-res images. Config:
            configs/examples/vibe_stage2.toml.
  Infer:   Load ONLY the Stage 2 adapter. Discarding Stage 1 at inference
            is the "relay" — the video base is untouched, Stage 2 adds
            spatial extrapolation without disturbing the motion prior.

Mathematically the relay is W_train = W_0 + ΔW1 + ΔW2 and
W_inference = W_0 + ΔW2. Takenoko's existing frozen_network_weights
mechanism provides the training-time composition; no merge step needed.

HFATO is applied after prepare_standard_training_inputs and is
silently overridden by FVDM / transition manager / error recycling /
self-resampling / glance / dual-model / EqM paths; the example configs
use the plain standard path.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F


_VALID_MODES = ("bilinear", "bicubic", "area", "nearest")


def hfato_degrade(
    latents: torch.Tensor,
    ratio: float = 0.5,
    mode: Literal["bilinear", "bicubic", "area", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Downsample-upsample spatial dims of a 4D or 5D latent."""
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"hfato_degrade ratio must be in (0, 1), got {ratio}")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"hfato_degrade mode must be one of {_VALID_MODES}, got {mode!r}"
        )

    if latents.dim() == 5:
        B, C, F_, H, W = latents.shape
        x = latents.permute(0, 2, 1, 3, 4).reshape(B * F_, C, H, W)
    elif latents.dim() == 4:
        B, C, H, W = latents.shape
        F_ = None
        x = latents
    else:
        raise ValueError(
            f"hfato_degrade expects 4D or 5D latent, got {latents.dim()}D"
        )

    h_down = max(1, int(round(H * ratio)))
    w_down = max(1, int(round(W * ratio)))

    align_corners: Optional[bool]
    if mode in ("nearest", "area"):
        align_corners = None
    else:
        align_corners = False

    down = F.interpolate(
        x, size=(h_down, w_down), mode=mode, align_corners=align_corners
    )
    up = F.interpolate(
        down, size=(H, W), mode=mode, align_corners=align_corners
    )

    if F_ is not None:
        up = up.reshape(B, F_, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    return up


def build_hfato_noisy_input(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    ratio: float = 0.5,
    mode: Literal["bilinear", "bicubic", "area", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Return x_t = (1 - σ_t)·DU(x_0) + σ_t·ε using the FM linear interpolant."""
    degraded = hfato_degrade(clean_latents, ratio=ratio, mode=mode)

    if sigmas.dim() < degraded.dim():
        view_shape = (-1,) + (1,) * (degraded.dim() - 1)
        sigmas_b = sigmas.view(view_shape)
    else:
        sigmas_b = sigmas

    sigmas_b = sigmas_b.to(device=degraded.device, dtype=degraded.dtype)
    noise_c = noise.to(device=degraded.device, dtype=degraded.dtype)

    return (1.0 - sigmas_b) * degraded + sigmas_b * noise_c


def compute_hfato_x0_reconstruction_loss(
    noisy_model_input: torch.Tensor,
    sigmas: torch.Tensor,
    model_pred: torch.Tensor,
    target_clean: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE(x_t - σ_t·v_pred, target_clean) — x0 reconstruction vs. undegraded."""
    if sigmas.dim() < noisy_model_input.dim():
        view_shape = (-1,) + (1,) * (noisy_model_input.dim() - 1)
        sigmas_b = sigmas.view(view_shape)
    else:
        sigmas_b = sigmas

    sigmas_b = sigmas_b.to(
        device=noisy_model_input.device, dtype=noisy_model_input.dtype
    )

    x0_pred = noisy_model_input - sigmas_b * model_pred.to(noisy_model_input)
    return F.mse_loss(
        x0_pred, target_clean.to(noisy_model_input), reduction=reduction
    )
