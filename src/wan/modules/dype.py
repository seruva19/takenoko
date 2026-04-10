from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


WAN_LATENT_DOWNSAMPLE = 8


def compute_dynamic_positional_extrapolation_scales(
    *,
    grid_sizes: torch.Tensor,
    patch_size: Tuple[int, int, int],
    base_resolution: int,
    max_scale: float,
    activate_above_frames: int,
    latent_downsample: int = WAN_LATENT_DOWNSAMPLE,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Compute per-sample temporal/height/width RoPE compression scales.

    Returned scales are >= 1.0. A value > 1 compresses the corresponding position
    indices so larger token grids reuse the base-frequency range.
    """
    if grid_sizes.numel() == 0:
        return None, {
            "dype/active": 0.0,
            "dype/scale_max": 1.0,
            "dype/scale_mean": 1.0,
        }

    token_base_h = max(1.0, float(base_resolution) / float(latent_downsample * patch_size[1]))
    token_base_w = max(1.0, float(base_resolution) / float(latent_downsample * patch_size[2]))
    device = grid_sizes.device
    scales = torch.ones(
        (grid_sizes.shape[0], 3),
        dtype=torch.float32,
        device=device,
    )

    f_vals = grid_sizes[:, 0].to(torch.float32)
    h_vals = grid_sizes[:, 1].to(torch.float32)
    w_vals = grid_sizes[:, 2].to(torch.float32)

    h_scale = torch.clamp(h_vals / token_base_h, min=1.0, max=max_scale)
    w_scale = torch.clamp(w_vals / token_base_w, min=1.0, max=max_scale)
    scales[:, 1] = h_scale
    scales[:, 2] = w_scale

    if activate_above_frames > 0:
        frame_base = max(1.0, float(activate_above_frames) / float(max(1, patch_size[0])))
        t_scale = torch.where(
            f_vals > frame_base,
            torch.clamp(f_vals / frame_base, min=1.0, max=max_scale),
            torch.ones_like(f_vals),
        )
        scales[:, 0] = t_scale

    active_mask = (scales > 1.0 + 1e-6).any(dim=1)
    metrics = {
        "dype/active": float(active_mask.any().item()),
        "dype/active_fraction": float(active_mask.to(torch.float32).mean().item()),
        "dype/scale_mean": float(scales.mean().item()),
        "dype/scale_max": float(scales.max().item()),
        "dype/temporal_scale_mean": float(scales[:, 0].mean().item()),
        "dype/height_scale_mean": float(scales[:, 1].mean().item()),
        "dype/width_scale_mean": float(scales[:, 2].mean().item()),
    }
    if not active_mask.any():
        return None, metrics
    return scales, metrics


def interpolate_frequency_axis(
    axis_freqs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation over a RoPE frequency axis for fractional positions."""
    max_index = axis_freqs.shape[0] - 1
    positions = positions.clamp(min=0.0, max=float(max_index))
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long)
    frac = (positions - lower.to(dtype=positions.dtype)).unsqueeze(-1)
    lower_vals = axis_freqs[lower]
    upper_vals = axis_freqs[upper]
    return lower_vals * (1.0 - frac) + upper_vals * frac
