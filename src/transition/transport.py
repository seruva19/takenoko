"""Transport implementations for TiM-style transition training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class TransportState:
    alpha_t: Tensor
    sigma_t: Tensor
    d_alpha_t: Tensor
    d_sigma_t: Tensor
    x_t: Tensor
    v_t: Tensor


class BaseTransport:
    """Base class for transports used by transition training."""

    def __init__(self, t_min: float, t_max: float) -> None:
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    # ---- Sampling utilities -------------------------------------------------

    def to_native_time(self, u: Tensor) -> Tensor:
        """Map unit interval samples to transport's native time domain."""
        return self.t_min + (self.t_max - self.t_min) * u

    def normalize(self, t: Tensor) -> Tensor:
        """Normalize time values to [0, 1] for weighting functions."""
        span = max(self.t_max - self.t_min, 1e-6)
        return (t - self.t_min) / span

    # ---- Interpolant --------------------------------------------------------

    def _interpolant_core(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def prepare_state(
        self, t: Tensor, latents: Tensor, noise: Tensor
    ) -> TransportState:
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._interpolant_core(t)
        view_shape = (-1,) + (1,) * (latents.ndim - 1)
        alpha_view = alpha_t.view(view_shape)
        sigma_view = sigma_t.view(view_shape)
        x_t = alpha_view * latents + sigma_view * noise
        d_alpha_view = d_alpha_t.view(view_shape)
        d_sigma_view = d_sigma_t.view(view_shape)
        v_t = d_alpha_view * latents + d_sigma_view * noise
        return TransportState(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
            x_t=x_t,
            v_t=v_t,
        )

    # ---- Target computation -------------------------------------------------

    def compute_target(
        self,
        state: TransportState,
        latents: Tensor,
        noise: Tensor,
        t: Tensor,
        r: Tensor,
        dF_dt: Tensor,
        teacher_pred: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError


class LinearTransport(BaseTransport):
    """Flow-matching with linear path (OT-FM)."""

    def __init__(self) -> None:
        super().__init__(t_min=0.0, t_max=1.0)

    def _interpolant_core(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha_t = 1.0 - t
        sigma_t = t
        d_alpha_t = t.new_full(t.shape, -1.0)
        d_sigma_t = t.new_ones(t.shape)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_target(
        self,
        state: TransportState,
        latents: Tensor,
        noise: Tensor,
        t: Tensor,
        r: Tensor,
        dF_dt: Tensor,
        teacher_pred: Optional[Tensor] = None,
    ) -> Tensor:
        view_shape = (-1,) + (1,) * (latents.ndim - 1)
        delta = (t - r).view(view_shape)
        v_t = state.v_t
        if teacher_pred is not None:
            # Blend teacher into v_t (assume teacher outputs same space as model)
            v_t = teacher_pred
        return v_t - delta * dF_dt


class TrigFlowTransport(BaseTransport):
    """TrigFlow formulation (consistency-inspired)."""

    def __init__(self) -> None:
        super().__init__(t_min=0.0, t_max=math.pi / 2)

    def _interpolant_core(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha_t = torch.cos(t)
        sigma_t = torch.sin(t)
        d_alpha_t = -torch.sin(t)
        d_sigma_t = torch.cos(t)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_target(
        self,
        state: TransportState,
        latents: Tensor,
        noise: Tensor,
        t: Tensor,
        r: Tensor,
        dF_dt: Tensor,
        teacher_pred: Optional[Tensor] = None,
    ) -> Tensor:
        v_t = state.v_t
        if teacher_pred is not None:
            v_t = teacher_pred
        angle = (t - r).clamp(min=0.0)
        view_shape = (-1,) + (1,) * (latents.ndim - 1)
        angle_view = angle.view(view_shape)
        return v_t - torch.tan(angle_view) * (state.x_t + dF_dt)


class VariancePreservingTransport(BaseTransport):
    """Variance-preserving SDE transport."""

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_d: float = 19.9,
        epsilon_t: float = 1e-5,
        T: float = 1000.0,
    ) -> None:
        super().__init__(t_min=epsilon_t, t_max=1.0)
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.epsilon_t = epsilon_t
        self.T = T

    def beta(self, t: Tensor) -> Tensor:
        return torch.sqrt((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1.0)

    def _interpolant_core(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        beta_t = self.beta(t)
        alpha_t = 1.0 / torch.sqrt(beta_t**2 + 1.0)
        sigma_t = beta_t / torch.sqrt(beta_t**2 + 1.0)
        d_alpha_t = -0.5 * (self.beta_d * t + self.beta_min) / torch.sqrt(beta_t**2 + 1.0)
        d_sigma_t = 0.5 * (self.beta_d * t + self.beta_min) / (
            beta_t * torch.sqrt(beta_t**2 + 1.0)
        )
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_target(
        self,
        state: TransportState,
        latents: Tensor,
        noise: Tensor,
        t: Tensor,
        r: Tensor,
        dF_dt: Tensor,
        teacher_pred: Optional[Tensor] = None,
    ) -> Tensor:
        beta_t = self.beta(t)
        beta_r = self.beta(r)
        d_beta_t = (self.beta_d * t + self.beta_min) * (beta_t**2 + 1.0) / (2.0 * beta_t)

        z = noise
        if teacher_pred is not None:
            z = teacher_pred

        view_shape = (-1,) + (1,) * (latents.ndim - 1)
        correction = (beta_t - beta_r).view(view_shape) / (d_beta_t.view(view_shape) + 1e-8)
        return z - dF_dt * correction


def create_transport(name: str) -> BaseTransport:
    if name == "trigflow":
        return TrigFlowTransport()
    if name == "vp":
        return VariancePreservingTransport()
    return LinearTransport()
