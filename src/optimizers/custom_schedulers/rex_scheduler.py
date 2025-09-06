"""Reflected Exponential (REX) learning rate scheduler.

The REX scheduler provides a smooth learning rate curve that starts and ends at a minimum
learning rate, with a peak in the middle. It's particularly effective for training
stability and convergence.

How to enable from TOML config (handled by optimizer_manager's alias map):

  lr_scheduler_type = "rex"
  lr_scheduler_args = [
    "max_lr=1e-4",
    "min_lr=1e-6",
    "num_steps=10000",
    "num_warmup_steps=100",
    "rex_alpha=0.1",
    "rex_beta=0.9",
  ]

Or use the short alias:
  lr_scheduler_type = "rex"
  lr_scheduler_args = [
    "min_lr_ratio=0.01",  # min_lr = max_lr * min_lr_ratio
  ]
"""

from __future__ import annotations

from typing import List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


class RexLR(_LRScheduler):
    """
    Reflected Exponential (REX) learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for
        max_lr (float): The maximum learning rate
        min_lr (float, optional): The minimum learning rate. If None, uses min_lr_ratio * max_lr
        min_lr_ratio (float, optional): Ratio of min_lr to max_lr. Defaults to 0.01
        num_steps (int): The total number of training steps
        num_warmup_steps (int, optional): The number of warmup steps. Defaults to 0
        rex_alpha (float): Constant added to the denominator of the REX factor;
            prevents division-by-zero and softens the initial decay (default: 0.1).
        rex_beta (float): Multiplier of z in the denominator of the REX factor;
            controls how quickly the decay flattens as z increases (default: 0.9).
        last_epoch (int, optional): The index of the last step. Defaults to -1
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: Optional[float] = None,
        min_lr_ratio: Optional[float] = 0.01,
        num_steps: int = 0,
        num_warmup_steps: int = 0,
        rex_alpha: float = 0.1,
        rex_beta: float = 0.9,
        last_epoch: int = -1,
    ):
        # Calculate min_lr if not provided
        if min_lr is None:
            if min_lr_ratio is None:
                min_lr_ratio = 0.01
            min_lr = max_lr * min_lr_ratio

        # Validate parameters
        if min_lr > max_lr:
            raise ValueError(
                f"Value of 'min_lr' should be less than value of 'max_lr'. "
                f"Got min_lr={min_lr} and max_lr={max_lr}"
            )
        if num_warmup_steps > num_steps:
            raise ValueError(
                f"num_warmup_steps ({num_warmup_steps}) must be less than "
                f"or equal to num_steps ({num_steps})"
            )

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.rex_alpha = rex_alpha
        self.rex_beta = rex_beta

        # Ensure each parameter group has an "initial_lr" key to avoid issues when resuming
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using the REX formula."""
        # Single warmup step
        if self.num_warmup_steps == 1 and self.last_epoch == 1:
            return [self.min_lr for _ in self.base_lrs]

        # Multiple warmup steps; increase lr linearly from min_lr to max_lr
        elif (
            self.num_warmup_steps > 1
            and self.last_epoch >= 1
            and self.last_epoch <= (self.num_warmup_steps - 1)
        ):
            return [
                self.min_lr
                + (self.max_lr - self.min_lr)
                * (self.last_epoch - 1)
                / (self.num_warmup_steps - 1)
                for _ in self.base_lrs
            ]

        # Post-warmup phase: adjust step relative to the end of warmup
        step_after = self.last_epoch - self.num_warmup_steps
        remaining_steps = self.num_steps - self.num_warmup_steps

        # Avoid LR spiking
        if step_after >= remaining_steps or step_after == -1 or remaining_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        # Calculate REX curve for current step
        rex_z = (remaining_steps - (step_after % remaining_steps)) / remaining_steps
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (
            rex_z / (self.rex_alpha + self.rex_beta * rex_z)
        )

        return [base_lr * rex_factor for base_lr in self.base_lrs]
