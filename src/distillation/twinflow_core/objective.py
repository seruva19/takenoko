"""TwinFlow objective helpers shared by the parallel runner and tests."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def match_time_shape(time_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while time_tensor.dim() < like.dim():
        time_tensor = time_tensor.unsqueeze(-1)
    return time_tensor


def sample_primary_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    time_dist_ctrl: list[float],
) -> torch.Tensor:
    alpha = max(float(time_dist_ctrl[0]), 1e-4)
    beta = max(float(time_dist_ctrl[1]), 1e-4)
    max_scale = max(float(time_dist_ctrl[2]), 1e-4)
    beta_dist = torch.distributions.Beta(alpha, beta)
    sigma = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
    return (sigma * max_scale).clamp_(0.0, 1.0)


def sample_target_time(sigmas: torch.Tensor, consistency_ratio: float = 1.0) -> torch.Tensor:
    ratio = max(0.0, float(consistency_ratio))
    tt = sigmas - torch.rand_like(sigmas) * ratio * sigmas
    eps = torch.finfo(sigmas.dtype).eps if torch.is_floating_point(sigmas) else 1e-4
    tt = torch.maximum(tt, torch.zeros_like(tt))
    tt = torch.minimum(tt, sigmas - eps)
    return tt


def build_enhancement_mask(
    sigmas: torch.Tensor,
    enhanced_range: list[float],
) -> torch.Tensor:
    sigma_flat = sigmas.view(sigmas.shape[0], -1)[:, 0] if sigmas.ndim > 1 else sigmas.view(-1)
    low = float(enhanced_range[0])
    high = float(enhanced_range[1])
    return (sigma_flat >= low) & (sigma_flat <= high)


def reconstruct_states(
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    flow_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_b = match_time_shape(sigma, x_t)
    gamma = 1.0 - sigma_b
    x_hat = x_t - sigma_b * flow_pred
    z_hat = x_t + gamma * flow_pred
    return x_hat, z_hat


def compute_rcgm_target(
    *,
    base_pred: torch.Tensor,
    target: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigma: torch.Tensor,
    tt: torch.Tensor,
    estimate_order: int,
    delta_t: float,
    clamp_target: float,
    teacher_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    sigma_b = match_time_shape(sigma, base_pred)
    tt_b = match_time_shape(tt, base_pred)
    steps = max(1, int(estimate_order))
    t_anchor = torch.maximum(tt_b, sigma_b - float(delta_t))
    pred_accum = torch.zeros_like(base_pred)
    x_t = noisy_latents
    t_prev = sigma_b

    schedule = []
    if steps == 1:
        schedule.append(tt_b)
    else:
        for idx in range(steps - 1):
            frac = float(idx + 1) / float(steps)
            schedule.append(t_anchor * frac + sigma_b * (1.0 - frac))
        schedule.append(tt_b)

    for t_next in schedule:
        flow = teacher_forward(x_t, t_prev)
        x_hat, z_hat = reconstruct_states(x_t, t_prev, flow)
        x_t = t_next * z_hat + (1.0 - t_next) * x_hat
        pred_accum = pred_accum + flow * (t_prev - t_next)
        t_prev = t_next

    base_detached = base_pred.detach()
    raw = base_detached - pred_accum - target
    return base_detached - raw.clamp(min=-float(clamp_target), max=float(clamp_target))
