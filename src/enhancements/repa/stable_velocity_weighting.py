from __future__ import annotations

import math
from typing import Optional

import torch


def normalize_timesteps(
    timesteps: torch.Tensor,
    max_timestep: float,
) -> torch.Tensor:
    """Normalize timesteps to [0, 1] with robust handling of layout/scale."""
    t = timesteps.detach().float()
    if t.dim() > 1:
        t = t.view(t.shape[0], -1).mean(dim=1)

    scale = float(max_timestep)
    if scale > 1.0:
        t = t / scale
    elif t.numel() > 0 and float(t.max().item()) > 1.0:
        t = t / max(1.0, float(t.max().item()))
    return t.clamp(0.0, 1.0)


def align_weights_to_batch(weights: torch.Tensor, target_batch: int) -> torch.Tensor:
    """Broadcast/reduce 1D sample weights to match a target batch size."""
    if target_batch <= 0:
        return weights.new_zeros((0,))
    if weights.numel() == target_batch:
        return weights
    if target_batch % weights.numel() == 0:
        factor = target_batch // weights.numel()
        return weights.repeat_interleave(factor)
    if weights.numel() % target_batch == 0:
        factor = weights.numel() // target_batch
        return weights.view(target_batch, factor).mean(dim=1)
    if target_batch < weights.numel():
        return weights[:target_batch]
    repeats = int(math.ceil(target_batch / float(weights.numel())))
    return weights.repeat(repeats)[:target_batch]


def weighted_mean(
    per_sample_values: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute sample-weighted mean with safe fallback."""
    if per_sample_values.dim() > 1:
        per_sample_values = per_sample_values.reshape(
            per_sample_values.shape[0], -1
        ).mean(dim=1)
    if weights is None:
        return per_sample_values.mean()

    weights = align_weights_to_batch(weights, per_sample_values.shape[0]).to(
        device=per_sample_values.device,
        dtype=per_sample_values.dtype,
    )
    denom = weights.sum()
    if float(denom.item()) <= 1e-8:
        return per_sample_values.new_tensor(0.0)
    return (per_sample_values * weights).sum() / denom


def compute_stable_velocity_weights(
    t_norm: torch.Tensor,
    schedule: str,
    tau: float,
    k: float,
    path_type: str,
    min_weight: float,
) -> torch.Tensor:
    """Compute VA-REPA weighting schedules from StableVelocity."""
    t = t_norm.detach().float().clamp(0.0, 1.0)
    tau_clamped = max(min(float(tau), 1.0), 1e-6)
    schedule_name = str(schedule).lower()
    path = str(path_type).lower()

    if schedule_name == "hard":
        weights = (t < tau_clamped).float()
    elif schedule_name == "hard_high":
        weights = (t >= tau_clamped).float()
    elif schedule_name == "sigmoid":
        weights = torch.sigmoid(float(k) * (tau_clamped - t))
    elif schedule_name == "cosine":
        raw = 0.5 * (1.0 + torch.cos(math.pi * t / tau_clamped))
        weights = torch.where(t < tau_clamped, raw, torch.zeros_like(raw))
    elif schedule_name == "snr":
        t_snr = t.clamp(min=1e-6, max=1.0 - 1e-6)
        if path == "linear":
            snr_t = ((1.0 - t_snr) / t_snr) ** 2
            snr_limit = ((1.0 - tau_clamped) / tau_clamped) ** 2
        elif path == "cosine":
            angle_t = t_snr * math.pi / 2.0
            angle_tau = tau_clamped * math.pi / 2.0
            snr_t = (torch.cos(angle_t) / torch.sin(angle_t)) ** 2
            snr_limit = (math.cos(angle_tau) / math.sin(angle_tau)) ** 2
        else:
            raise ValueError(f"Unsupported StableVelocity path type: {path!r}")
        weights = snr_t / (snr_t + snr_limit)
    else:
        raise ValueError(f"Unsupported StableVelocity schedule: {schedule_name!r}")

    weights = weights.clamp(0.0, 1.0)
    if min_weight > 0.0:
        floor = max(0.0, min(1.0, float(min_weight)))
        weights = floor + (1.0 - floor) * weights
    return weights
