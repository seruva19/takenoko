"""Timestep scheduling utilities for transition training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from scheduling.timestep_utils import map_uniform_to_sampling


@dataclass
class TransitionSchedulerConfig:
    diffusion_ratio: float
    consistency_ratio: float
    weight_schedule: str
    use_tangent_weighting: bool
    use_adaptive_weighting: bool
    t_min: float
    t_max: float


@dataclass
class TransitionTimesteps:
    t: Tensor
    r: Tensor
    n_diffusion: int


class TransitionTimestepScheduler:
    """Samples timestep pairs (t, r) and computes per-sample weights."""

    def __init__(self, config: TransitionSchedulerConfig) -> None:
        self.config = config

    # ---- Sampling -----------------------------------------------------------

    def sample_pairs(
        self,
        args,
        latents: Tensor,
        device: torch.device,
        batch_size: int,
        transport,
    ) -> TransitionTimesteps:
        dtype = latents.dtype
        u1 = torch.rand(batch_size, device=device, dtype=dtype)
        u2 = torch.rand(batch_size, device=device, dtype=dtype)

        base_t1 = map_uniform_to_sampling(args, u1, latents)
        base_t2 = map_uniform_to_sampling(args, u2, latents)

        t1 = transport.to_native_time(base_t1)
        t2 = transport.to_native_time(base_t2)

        t = torch.maximum(t1, t2)
        r = torch.minimum(t1, t2)

        # Diffusion subset (t = r)
        n_diffusion = int(round(self.config.diffusion_ratio * batch_size))
        if n_diffusion > 0:
            idx = torch.arange(n_diffusion, device=device)
            r.index_copy_(0, idx, t.index_select(0, idx))

        # Consistency subset (r = t_min)
        n_consistency = int(round(self.config.consistency_ratio * batch_size))
        if n_consistency > 0:
            tail_idx = torch.arange(batch_size - n_consistency, batch_size, device=device)
            r.index_copy_(
                0, tail_idx, torch.full_like(r[tail_idx], transport.t_min)
            )

        t = t.clamp(transport.t_min, transport.t_max)
        r = r.clamp(transport.t_min, transport.t_max)

        return TransitionTimesteps(t=t, r=r, n_diffusion=n_diffusion)

    # ---- Weighting ----------------------------------------------------------

    def compute_weights(
        self,
        timesteps: TransitionTimesteps,
    ) -> Tensor:
        t = timesteps.t
        r = timesteps.r
        eps = 1e-6

        t_norm = self._normalize(t)
        r_norm = self._normalize(r)
        delta_norm = torch.clamp(t_norm - r_norm, min=0.0)

        if self.config.use_tangent_weighting:
            tau_t = torch.tan(torch.clamp(t_norm, 0.0, 1.0) * math.pi / 2.0)
            tau_r = torch.tan(torch.clamp(r_norm, 0.0, 1.0) * math.pi / 2.0)
            base = torch.clamp(tau_t - tau_r, min=0.0)
        else:
            base = delta_norm

        schedule = self.config.weight_schedule
        if schedule == "constant":
            weight = torch.ones_like(t)
        elif schedule == "tau_inverse":
            weight = 1.0 / (torch.abs(base) + eps)
        else:  # sqrt (default) or unknown
            weight = 1.0 / torch.sqrt(base + eps)

        if timesteps.n_diffusion > 0:
            weight[: timesteps.n_diffusion] = 1.0
        return torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)

    def adaptive_rescale(self, loss: Tensor, eps: float = 1e-6) -> Tensor:
        if not self.config.use_adaptive_weighting:
            return torch.ones_like(loss)
        return 1.0 / (loss.detach() + eps)

    def _normalize(self, t: Tensor) -> Tensor:
        denom = max(self.config.t_max - self.config.t_min, 1e-6)
        return (t - self.config.t_min) / denom
