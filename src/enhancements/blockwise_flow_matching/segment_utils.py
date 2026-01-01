from __future__ import annotations

from typing import Optional

import torch


def build_segment_boundaries(
    num_segments: int,
    min_t: float,
    max_t: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2.")
    if min_t < 0.0 or max_t > 1.0 or min_t >= max_t:
        raise ValueError("min_t/max_t must satisfy 0 <= min_t < max_t <= 1.")
    return torch.linspace(
        min_t, max_t, num_segments + 1, device=device, dtype=dtype
    )


def stratified_segment_timesteps(
    batch_size: int,
    boundaries: torch.Tensor,
    t_uniform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if batch_size <= 0:
        return torch.empty(0, device=boundaries.device, dtype=boundaries.dtype)
    num_segments = int(boundaries.numel() - 1)
    if num_segments <= 0:
        raise ValueError("boundaries must define at least one segment.")
    if t_uniform is None:
        t_uniform = torch.rand(
            batch_size, device=boundaries.device, dtype=boundaries.dtype
        )
    else:
        t_uniform = t_uniform.to(device=boundaries.device, dtype=boundaries.dtype)

    base = batch_size // num_segments
    remainder = batch_size % num_segments
    split_sizes = [
        base + 1 if idx < remainder else base for idx in range(num_segments)
    ]

    samples = []
    offset = 0
    for seg_idx, split_size in enumerate(split_sizes):
        if split_size == 0:
            continue
        segment_u = t_uniform[offset : offset + split_size]
        seg_min = boundaries[seg_idx]
        seg_max = boundaries[seg_idx + 1]
        samples.append(seg_min + segment_u * (seg_max - seg_min))
        offset += split_size

    if not samples:
        return torch.empty(0, device=boundaries.device, dtype=boundaries.dtype)
    return torch.cat(samples, dim=0)


def segment_index_for_timesteps(
    timesteps: torch.Tensor, boundaries: torch.Tensor
) -> torch.Tensor:
    if boundaries.numel() < 2:
        raise ValueError("boundaries must contain at least two values.")
    max_idx = boundaries.numel() - 2
    eps = torch.finfo(boundaries.dtype).eps
    t_clamped = timesteps.clamp(min=boundaries[0], max=boundaries[-1] - eps)
    indices = torch.bucketize(t_clamped, boundaries, right=False) - 1
    return indices.clamp(min=0, max=max_idx)


def normalize_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
    t = timesteps.float()
    if t.numel() == 0:
        return t
    if t.max() > 1.0:
        t = (t - 1.0) / 1000.0
    return t.clamp(0.0, 1.0)


def compute_blockwise_interpolant(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = normalize_timesteps(timesteps)
    segment_idx = segment_index_for_timesteps(t, boundaries)
    t_start = boundaries[segment_idx]
    t_end = boundaries[segment_idx + 1]
    eps = torch.finfo(t.dtype).eps
    denom = (t_end - t_start).clamp_min(eps)
    a = (t - t_start) / denom

    view_shape = (-1,) + (1,) * (latents.dim() - 1)
    t_start_b = t_start.view(view_shape)
    t_end_b = t_end.view(view_shape)
    a_b = a.view(view_shape)

    x_start = (1 - t_start_b) * latents + t_start_b * noise
    x_end = (1 - t_end_b) * latents + t_end_b * noise
    xt = (1 - a_b) * x_start + a_b * x_end
    target = (x_end - x_start) / denom.view(view_shape)
    return xt, target, segment_idx
