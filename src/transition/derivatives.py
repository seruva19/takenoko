"""Derivative estimation utilities for transition training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


ModelForwardFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class DerivativeConfig:
    mode: str
    epsilon: float
    failover_mode: str


def compute_derivative(
    cfg: DerivativeConfig,
    forward_fn: ModelForwardFn,
    noisy_latents: Tensor,
    timesteps: Tensor,
) -> Tensor:
    mode = cfg.mode
    if mode == "none":
        return torch.zeros_like(noisy_latents)

    if mode == "auto":
        try:
            return _dde_estimate(cfg.epsilon, forward_fn, noisy_latents, timesteps)
        except Exception:
            fallback_cfg = cfg.failover_mode if cfg.failover_mode else "jvp"
            return compute_derivative(
                DerivativeConfig(
                    mode=fallback_cfg,
                    epsilon=cfg.epsilon,
                    failover_mode="none",
                ),
                forward_fn,
                noisy_latents,
                timesteps,
            )

    if mode == "jvp":
        try:
            return _jvp_estimate(forward_fn, noisy_latents, timesteps)
        except Exception:
            return _dde_estimate(cfg.epsilon, forward_fn, noisy_latents, timesteps)

    # Default to DDE
    return _dde_estimate(cfg.epsilon, forward_fn, noisy_latents, timesteps)


def _dde_estimate(
    epsilon: float,
    forward_fn: ModelForwardFn,
    noisy_latents: Tensor,
    timesteps: Tensor,
) -> Tensor:
    eps = max(epsilon, 1e-6)
    eps_tensor = torch.full_like(timesteps, eps)
    t_plus = timesteps + eps_tensor
    t_minus = torch.clamp(timesteps - eps_tensor, min=1.0)

    f_plus = forward_fn(noisy_latents, t_plus)
    f_minus = forward_fn(noisy_latents, t_minus)

    denom = (t_plus - t_minus).view(-1, *([1] * (noisy_latents.ndim - 1)))
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    derivative = (f_plus - f_minus) / denom
    return derivative


def _jvp_estimate(
    forward_fn: ModelForwardFn,
    noisy_latents: Tensor,
    timesteps: Tensor,
) -> Tensor:
    def func(latents: Tensor, times: Tensor) -> Tensor:
        return forward_fn(latents, times)

    tangents = (
        torch.zeros_like(noisy_latents),
        torch.ones_like(timesteps),
    )
    _, jvp = torch.autograd.functional.jvp(
        func,
        (noisy_latents.detach(), timesteps.detach()),
        tangents,
        create_graph=False,
        strict=False,
    )
    return jvp
