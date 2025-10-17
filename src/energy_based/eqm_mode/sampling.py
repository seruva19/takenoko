from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class AdaptiveStepController:
    """Simple adaptive step-size tracker for EqM gradient samplers."""

    initial_step: float
    min_step: float
    max_step: float
    growth: float
    shrink: float
    patience: int
    alignment_threshold: float = 0.0
    eps: float = 1e-8
    _step: float = field(init=False)
    _success_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._step = float(self.initial_step)
        self.min_step = max(float(self.min_step), self.eps)
        self.max_step = max(float(self.max_step), self.min_step)
        self.growth = max(float(self.growth), 1.0)
        self.shrink = min(float(self.shrink), 1.0)
        self.patience = max(int(self.patience), 1)

    @property
    def value(self) -> float:
        """Return the current step size."""
        return self._step

    def update(self, prev_update: torch.Tensor, current_update: torch.Tensor) -> bool:
        """Adjust step size based on alignment. Returns True when momentum should reset."""
        if prev_update is None:
            prev_norm = 0.0
        else:
            prev_norm = prev_update.norm().item()
        curr_norm = current_update.norm().item()

        if prev_norm < self.eps or curr_norm < self.eps:
            self._success_count = 0
            return False

        dot = torch.sum(prev_update * current_update).item()
        cos_sim = dot / (prev_norm * curr_norm + self.eps)

        if cos_sim < self.alignment_threshold:
            # Diverging: shrink step and request momentum reset
            self._step = max(self._step * self.shrink, self.min_step)
            self._success_count = 0
            return True

        self._success_count += 1
        if self._success_count >= self.patience:
            self._step = min(self._step * self.growth, self.max_step)
            self._success_count = 0
        return False


def gradient_descent_sampler(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    step_size: float = 1e-3,
    momentum: float = 0.0,
    num_steps: int = 50,
):
    """Return a callable that performs gradient-descent-style sampling."""

    def _sample(z0: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        z = z0.clone()
        velocity = torch.zeros_like(z)
        for _ in range(num_steps):
            grad = model_fn(z, t, **model_kwargs)
            if momentum != 0.0:
                velocity.mul_(momentum).add_(grad)
                z = z + step_size * velocity
            else:
                z = z + step_size * grad
            z = z
        return z

    return _sample
