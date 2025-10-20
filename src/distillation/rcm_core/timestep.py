"""Trigflow timestep sampling utilities for rCM distillation.

The routines in this module mirror the logic used in NVIDIA's rCM trainer
(`draw_training_time`, `draw_training_time_D`) while remaining lightweight and
framework-agnostic.  They will replace the heuristic conversion currently used
inside ``TrainingCore.call_dit`` as the integration progresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch


class TimestepDistribution(Protocol):
    """Protocol describing the minimal interface required for rCM SDE samplers."""

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ...


@dataclass(slots=True)
class RCMTimestepSamplers:
    """Container bundling the generator/critic timestep draw functions."""

    generator_times: torch.Tensor
    critic_times: torch.Tensor


class RCMSimpleEDMSDE:
    """Simplified EDM-SDE sampler mirroring upstream defaults."""

    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        p_mean: float,
        p_std: float,
    ) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, device=device, dtype=torch.float64)
        sigma = torch.exp(self.p_mean + self.p_std * noise)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        return sigma


def _ensure_double(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float64)


def draw_trigflow_time(
    *,
    batch_size: int,
    condition_modality: str,
    sde: TimestepDistribution,
    device: torch.device,
    video_noise_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample trigflow times for the student/generator step."""

    sigma = sde.sample_t(batch_size, device=device)
    sigma = _ensure_double(sigma)
    sigma = sigma.view(batch_size, 1)  # (B, 1)

    multiplier = video_noise_multiplier if condition_modality == "video" else 1.0
    sigma = sigma * multiplier
    time = torch.arctan(sigma)
    return time


def draw_trigflow_time_discriminator(
    *,
    batch_size: int,
    condition_modality: str,
    sde_d: Optional[TimestepDistribution],
    device: torch.device,
    video_noise_multiplier: float = 1.0,
    timestep_shift: float = 0.0,
) -> torch.Tensor:
    """Sample trigflow times for the fake-score/critic step."""

    if timestep_shift and timestep_shift > 0:
        sigma = torch.rand(batch_size, device=device, dtype=torch.float64)
        sigma = timestep_shift * sigma / (1 + (timestep_shift - 1) * sigma)
        sigma = sigma.view(batch_size, 1)
        time = torch.arctan(sigma / torch.clamp(1 - sigma, min=1e-6))
        return time

    if sde_d is None:
        raise ValueError("sde_d must be provided when timestep_shift is zero.")

    sigma = sde_d.sample_t(batch_size, device=device)
    sigma = _ensure_double(sigma)
    sigma = sigma.view(batch_size, 1)

    multiplier = video_noise_multiplier if condition_modality == "video" else 1.0
    sigma = sigma * multiplier
    time = torch.arctan(sigma)
    return time


def sample_trigflow_times(
    *,
    batch_size: int,
    condition_modality: str,
    sde: TimestepDistribution,
    sde_d: Optional[TimestepDistribution],
    device: torch.device,
    video_noise_multiplier: float = 1.0,
    timestep_shift: float = 0.0,
) -> RCMTimestepSamplers:
    """Convenience wrapper returning generator and critic times."""

    gen_time = draw_trigflow_time(
        batch_size=batch_size,
        condition_modality=condition_modality,
        sde=sde,
        device=device,
        video_noise_multiplier=video_noise_multiplier,
    )

    crit_time = draw_trigflow_time_discriminator(
        batch_size=batch_size,
        condition_modality=condition_modality,
        sde_d=sde_d,
        device=device,
        video_noise_multiplier=video_noise_multiplier,
        timestep_shift=timestep_shift,
    )
    return RCMTimestepSamplers(generator_times=gen_time, critic_times=crit_time)


def build_simple_edm_sde(
    *,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> RCMSimpleEDMSDE:
    """Factory helper used by the distillation runner."""

    return RCMSimpleEDMSDE(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
