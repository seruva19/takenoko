"""Per-cycle warmup + optional stable plateau + cosine LR scheduler with a minimum floor.

This scheduler re-applies warmup at the start of every cycle, then cosine
decays to a floor learning rate. It is intended for very long or indefinite
training runs where you want periodic restarts without restarting the process
or losing optimizer state.

How to enable from TOML config (handled by optimizer_manager's hook):

  lr_scheduler_type = "per_cycle_cosine"
  lr_scheduler_args = [
    "cycle_steps=500",
    "warmup_steps=25",
    "stable_steps=0",      # optional plateau at base LR
    "decay_steps=475",
    "min_lr_ratio=0.05"    # or use eta_min instead of a ratio
  ]

Key behavior
------------
- Every cycle: warmup (linear) from floor_lr -> base_lr, optional stable
  plateau at base_lr, then cosine decay from base_lr -> floor_lr.
- If cycle_steps > warmup_steps + stable_steps + decay_steps, the remaining
  steps in the cycle stay at floor_lr.
- `eta_min` (absolute LR floor) takes precedence over `min_lr_ratio` when set.

Why it's useful for high-noise adapters
--------------------------------------
- Per-cycle warmup softens LR jumps at restarts and reduces gradient shocks.
- The LR floor prevents LR from collapsing to ~0, keeping learning active.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


class PerCycleWarmupCosineWithFloor(torch.optim.lr_scheduler._LRScheduler):
    """LR scheduler with per-cycle warmup + cosine decay and a minimum floor.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate to schedule.
    cycle_steps : int
        Total number of steps in one cycle (warmup + decay + optional floor tail).
    warmup_steps : int
        Linear warmup length at the start of each cycle.
    stable_steps : int
        Length of the constant plateau at base LR after warmup within each cycle.
    decay_steps : int
        Cosine decay length after the plateau within each cycle.
    min_lr_ratio : float, default 0.0
        Minimum LR as a ratio of the base LR (per param group). Ignored if
        `eta_min` is provided.
    eta_min : Optional[float]
        Absolute minimum LR. If provided, it overrides `min_lr_ratio`.
    last_epoch : int, default -1
        The index of the last epoch (step). Set to -1 to start scheduling at step 0.

    Behavior
    --------
    For each param group with base LR `base_lr` and floor `floor_lr` (derived from
    `eta_min` or `min_lr_ratio * base_lr`):
      - Warmup (0..warmup_steps-1): linear from floor_lr -> base_lr
      - Stable (warmup_steps..warmup_steps+stable_steps-1): constant at base_lr
      - Decay (warmup_steps+stable_steps..
        warmup_steps+stable_steps+decay_steps-1): cosine base_lr -> floor_lr
      - Tail (remaining steps in the cycle): constant at floor_lr
    This repeats every `cycle_steps` steps indefinitely.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        cycle_steps: int,
        warmup_steps: int,
        stable_steps: int = 0,
        decay_steps: int,
        min_lr_ratio: float = 0.0,
        eta_min: Optional[float] = None,
        last_epoch: int = -1,
    ) -> None:
        if cycle_steps <= 0:
            raise ValueError("cycle_steps must be > 0")
        if warmup_steps < 0 or stable_steps < 0 or decay_steps < 0:
            raise ValueError("warmup_steps, stable_steps and decay_steps must be >= 0")
        if warmup_steps + stable_steps + decay_steps > cycle_steps:
            raise ValueError(
                "warmup_steps + stable_steps + decay_steps must be <= cycle_steps"
            )
        if eta_min is None and (min_lr_ratio < 0.0 or not math.isfinite(min_lr_ratio)):
            raise ValueError("min_lr_ratio must be a finite value >= 0.0")

        self.cycle_steps: int = int(cycle_steps)
        self.warmup_steps: int = int(warmup_steps)
        self.stable_steps: int = int(stable_steps)
        self.decay_steps: int = int(decay_steps)
        self.min_lr_ratio: float = float(min_lr_ratio)
        self.eta_min: Optional[float] = float(eta_min) if eta_min is not None else None

        # Cache base LRs for each param group
        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=last_epoch)

    def _floor_lr(self, base_lr: float) -> float:
        if self.eta_min is not None:
            return float(self.eta_min)
        return float(base_lr * self.min_lr_ratio)

    def get_lr(self) -> List[float]:
        # PyTorch calls get_lr after updating last_epoch in step().
        step_idx = max(self.last_epoch, 0)
        pos_in_cycle = step_idx % self.cycle_steps

        lrs: List[float] = []
        for base_lr in self.base_lrs:
            floor_lr = self._floor_lr(base_lr)

            # Guardrails to avoid numerical issues
            if floor_lr > base_lr:
                floor_lr = base_lr

            warmup_end = self.warmup_steps
            stable_end = self.warmup_steps + self.stable_steps
            decay_end = self.warmup_steps + self.stable_steps + self.decay_steps

            if self.warmup_steps > 0 and pos_in_cycle < warmup_end:
                # Linear warmup from floor_lr -> base_lr
                progress = (pos_in_cycle + 1) / float(self.warmup_steps)
                lr = floor_lr + (base_lr - floor_lr) * progress
            elif self.stable_steps > 0 and pos_in_cycle < stable_end:
                # Stable plateau at base_lr
                lr = base_lr
            elif self.decay_steps > 0 and pos_in_cycle < decay_end:
                # Cosine decay from base_lr -> floor_lr
                t = (pos_in_cycle - self.warmup_steps - self.stable_steps + 1) / float(
                    self.decay_steps
                )
                t = min(max(t, 0.0), 1.0)
                lr = floor_lr + 0.5 * (base_lr - floor_lr) * (
                    1.0 + math.cos(math.pi * t)
                )
            else:
                # Tail: stay at floor
                lr = floor_lr

            lrs.append(float(lr))

        return lrs

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"PerCycleWarmupCosineWithFloor(cycle_steps={self.cycle_steps}, "
            f"warmup_steps={self.warmup_steps}, decay_steps={self.decay_steps}, "
            f"min_lr_ratio={self.min_lr_ratio}, eta_min={self.eta_min})"
        )
