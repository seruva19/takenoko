"""ReLoRA jagged cosine scheduler with restart warmups.

Uses initial warmup, cosine decay to a floor, and per-restart warmups that rise
toward the current decay curve.
"""

from __future__ import annotations

import math
from typing import List

import torch


class ReLoRAJaggedCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ReLoRA jagged cosine LR schedule.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate to schedule.
    first_warmup_steps : int
        Initial warmup length at the start of training.
    max_steps : int
        Total number of optimization steps.
    restart_warmup_steps : int
        Warmup length applied at each restart after the first cycle.
    restart_frequency : int
        Frequency of restarts in steps (usually relora_cycle_length).
    min_lr_ratio : float
        Minimum LR ratio relative to base LR, must be in (0, 1].
    last_epoch : int, default -1
        The index of the last epoch (step). Set to -1 to start scheduling at step 0.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        first_warmup_steps: int,
        max_steps: int,
        restart_warmup_steps: int,
        restart_frequency: int,
        min_lr_ratio: float,
        last_epoch: int = -1,
    ) -> None:
        if not (0.0 < float(min_lr_ratio) <= 1.0):
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if int(restart_warmup_steps) <= 0:
            raise ValueError("restart_warmup_steps must be > 0")
        if int(restart_frequency) <= 0:
            raise ValueError("restart_frequency must be > 0")
        if int(first_warmup_steps) <= 0:
            raise ValueError("first_warmup_steps must be > 0")
        if int(restart_frequency) % int(first_warmup_steps) != 0:
            raise ValueError(
                "restart_frequency must be divisible by first_warmup_steps"
            )

        self.first_warmup_steps = int(first_warmup_steps)
        self.max_steps = int(max_steps)
        self.restart_warmup_steps = int(restart_warmup_steps)
        self.restart_frequency = int(restart_frequency)
        self.min_lr_ratio = float(min_lr_ratio)

        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=last_epoch)

    @staticmethod
    def _get_cosine_decay(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _decay_ratio(self, step_idx: int) -> float:
        denom = max(1, self.max_steps - self.first_warmup_steps)
        progress = (step_idx - self.first_warmup_steps) / float(denom)
        decay = self._get_cosine_decay(progress)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay

    def get_lr(self) -> List[float]:
        step_idx = max(self.last_epoch, 0)

        # Initial warmup
        if step_idx < self.first_warmup_steps:
            warmup_ratio = step_idx / float(max(1, self.first_warmup_steps))
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]

        restart_step = step_idx % self.restart_frequency
        restart_number = step_idx // self.restart_frequency

        # Restart warmup phase after the first cycle
        if restart_step < self.restart_warmup_steps and step_idx >= self.restart_frequency:
            end_of_warmup_progress = (
                restart_number * self.restart_frequency
                + self.restart_warmup_steps
                - self.first_warmup_steps
            ) / float(max(1, self.max_steps - self.first_warmup_steps))
            decay_ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * self._get_cosine_decay(
                end_of_warmup_progress
            )
            warmup_ratio = restart_step / float(max(1, self.restart_warmup_steps))
            return [base_lr * warmup_ratio * decay_ratio for base_lr in self.base_lrs]

        # Standard cosine decay
        decay_ratio = self._decay_ratio(step_idx)
        return [base_lr * decay_ratio for base_lr in self.base_lrs]
