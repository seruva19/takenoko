"""Path planners used by EqM transport losses."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape 1D timesteps so they can broadcast across `x`."""
    dims = [1] * (x.ndim - 1)
    return t.view(t.size(0), *dims)


class ICPlan:
    """Linear interpolant path used in EqM."""

    def __init__(self, sigma: float = 0.0) -> None:
        self.sigma = sigma

    @staticmethod
    def compute_alpha_t(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return data coefficient α(t) and its derivative."""
        return t, torch.ones_like(t)

    @staticmethod
    def compute_sigma_t(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return noise coefficient σ(t) and its derivative."""
        return 1 - t, -torch.ones_like(t)

    @staticmethod
    def compute_d_alpha_alpha_ratio_t(t: torch.Tensor) -> torch.Tensor:
        """Return dα/α for numerical stability."""
        return 1 / t

    def compute_drift(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift and variance for the score-form SDE."""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        form: str = "constant",
        norm: float = 1.0,
    ) -> torch.Tensor:
        """Diffusion coefficient used in EqM sampling."""
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(math.pi * t) + 1) ** 2,
            # Preserve the EqM typo for compatibility while offering the corrected name.
            "increasing-decreasing": norm * torch.sin(math.pi * t) ** 2,
            "inccreasing-decreasing": norm * torch.sin(math.pi * t) ** 2,
        }
        if form not in choices:
            raise NotImplementedError(f"Diffusion form {form} not implemented.")
        diffusion = choices[form]

        if isinstance(diffusion, torch.Tensor):
            return diffusion
        return torch.tensor(diffusion, device=x.device, dtype=x.dtype)

    def get_score_from_velocity(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        mean = x
        return (reverse_alpha_ratio * velocity - mean) / variance

    def get_noise_from_velocity(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = reverse_alpha_ratio * d_sigma_t - sigma_t
        mean = x
        return (reverse_alpha_ratio * velocity - mean) / variance

    def get_velocity_from_noise(
        self, noise: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = reverse_alpha_ratio * d_sigma_t - sigma_t
        mean = x
        return (noise * variance + mean) / reverse_alpha_ratio

    def get_velocity_from_score(
        self, score: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x)
        drift, variance = self.compute_drift(x, t)
        return variance * score - drift

    def compute_mu_t(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        del xt  # unused
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """Variance-preserving path used by many diffusion models."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )

    def _d_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_log = self._log_mean_coeff(t)
        alpha_t = torch.exp(alpha_log)
        return alpha_t, alpha_t * self._d_log_mean_coeff(t)

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_sigma = 2 * self._log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(log_sigma))
        d_sigma_t = torch.exp(log_sigma) * (
            2 * self._d_log_mean_coeff(t)
        ) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        return self._d_log_mean_coeff(t)

    def compute_drift(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    """Geodesic variance-preserving path."""

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_t = torch.sin(t * math.pi / 2)
        d_alpha_t = (math.pi / 2) * torch.cos(t * math.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_t = torch.cos(t * math.pi / 2)
        d_sigma_t = -(math.pi / 2) * torch.sin(t * math.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        return math.pi / (2 * torch.tan(t * math.pi / 2))
