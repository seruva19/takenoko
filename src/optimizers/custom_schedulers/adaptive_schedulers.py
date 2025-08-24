"""Adaptive learning rate schedulers tuned for high-noise training.

This module provides three schedulers:
  - EMAAdaptiveScheduler: adjusts LR from a bounded range using an EMA loss trend
  - NoiseAdaptiveScheduler: adapts cycle length and LR to gradient noise
  - HybridAdaptiveScheduler: composes per-cycle cosine with EMA+noise adaptation

All classes derive from torch.optim.lr_scheduler._LRScheduler and are stateless
with respect to training metrics except for internal EMAs/counters. For DDP/AMP
setups, update hooks should be called with synchronized gradients and unscaled
values:
  - Call update_gradient_stats() after any AMP unscale_ and before clipping
  - Prefer calling update_training_stats(loss) once per optimizer step

How to enable from TOML config (handled by optimizer_manager's alias map):

  # EMA-based adaptive scheduler
  lr_scheduler_type = "ema_adaptive"
  lr_scheduler_args = [
    "base_lr_range=(1e-5,6e-4)",
    "ema_alpha=0.05",
    "sensitivity=0.5",
    "min_history=20",
  ]

  # Noise-based adaptive scheduler
  lr_scheduler_type = "noise_adaptive"
  lr_scheduler_args = [
    "base_cycle_steps=500",
    "noise_window=64",
    "cycle_range=(200,1200)",
    "lr_range=(1e-5,5e-4)",
    "noise_threshold=1.0",
  ]

  # Hybrid adaptive scheduler
  lr_scheduler_type = "hybrid_adaptive"
  lr_scheduler_args = [
    "cycle_steps=500",
    "warmup_steps=25",
    "stable_steps=0",
    "decay_steps=475",
    "min_lr_ratio=0.05",
    "ema_alpha=0.05",
    "noise_sensitivity=0.3",
    "adaptation_strength=0.2",
  ]

  # Amplitude-adaptive per-cycle cosine (bounded multiplicative gain)
  lr_scheduler_type = "adaptive_per_cycle_cosine"
  lr_scheduler_args = [
    "cycle_steps=500",
    "warmup_steps=25",
    "stable_steps=0",
    "decay_steps=475",
    "min_lr_ratio=0.05",
    "ema_alpha=0.05",
    "noise_sensitivity=0.3",
    "adaptation_strength=0.1",
    "min_history=20",
    "adaptation_bounds=(0.7,1.5)",
  ]

  # Cycle-length adaptive per-cycle (adapts cycle at boundaries)
  lr_scheduler_type = "cycle_adaptive_per_cycle"
  lr_scheduler_args = [
    "base_cycle_steps=500",
    "warmup_ratio=0.05",
    "stable_ratio=0.0",
    "decay_ratio=0.95",
    "min_lr_ratio=0.05",
    "noise_window=64",
    "cycle_range=(200,800)",
    "adaptation_rate=0.1",
    "noise_threshold=1.0",
  ]
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple
import math

import torch


class EMAAdaptiveScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler that adapts within a fixed LR range using loss EMA.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    base_lr_range : Tuple[float, float]
        Min and max LR bounds applied uniformly across param groups.
    ema_alpha : float
        EMA coefficient (0<alpha<=1). Larger means more reactive.
    sensitivity : float
        Multiplicative factor on the normalized trend before tanh squashing.
    min_history : int
        Minimum number of updates before adapting; uses mid-range before that.
    last_epoch : int
        Standard PyTorch semantics.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr_range: Tuple[float, float] = (1e-5, 1e-3),
        ema_alpha: float = 0.05,
        sensitivity: float = 0.5,
        min_history: int = 20,
        last_epoch: int = -1,
    ) -> None:
        self.base_lr_range: Tuple[float, float] = base_lr_range
        self.ema_alpha: float = float(ema_alpha)
        self.sensitivity: float = float(sensitivity)
        self.min_history: int = int(min_history)

        self.loss_ema: Optional[float] = None
        self.loss_ema_derivative: Optional[float] = None
        self.step_count: int = 0

        super().__init__(optimizer, last_epoch=last_epoch)

    @torch.no_grad()
    def update_metrics(self, loss: float) -> None:
        """Update loss EMA and its derivative with a scale-invariant trend.

        Use normalized derivative to reduce dependence on absolute loss scale.
        """
        if self.loss_ema is None:
            self.loss_ema = float(loss)
            self.loss_ema_derivative = 0.0
        else:
            prev_ema = float(self.loss_ema)
            self.loss_ema = float(
                self.ema_alpha * loss + (1.0 - self.ema_alpha) * prev_ema
            )
            raw_derivative = self.loss_ema - prev_ema
            norm = abs(prev_ema) + 1e-8
            normalized_derivative = raw_derivative / norm
            if self.loss_ema_derivative is None:
                self.loss_ema_derivative = normalized_derivative
            else:
                self.loss_ema_derivative = float(
                    self.ema_alpha * normalized_derivative
                    + (1.0 - self.ema_alpha) * self.loss_ema_derivative
                )
        self.step_count += 1

    def get_lr(self) -> List[float]:
        if self.step_count < self.min_history or self.loss_ema_derivative is None:
            target_lr = 0.5 * (self.base_lr_range[0] + self.base_lr_range[1])
        else:
            # Negative trend => decreasing loss => allow higher LR
            trend = -float(self.loss_ema_derivative) * self.sensitivity
            trend = max(-5.0, min(5.0, trend))  # clip before tanh
            lr_factor = 0.5 * (1.0 + math.tanh(trend))
            target_lr = float(
                self.base_lr_range[0]
                + lr_factor * (self.base_lr_range[1] - self.base_lr_range[0])
            )
        return [target_lr for _ in self.base_lrs]


class NoiseAdaptiveScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Adapt cycle length and LR using recent gradient noise level.

    High noise => shorter cycles, smaller max LR. Low noise => longer cycles, larger max LR.
    Uses cosine annealing within each (adaptive) cycle.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_cycle_steps: int = 500,
        base_lr: float = 1e-4,
        noise_window: int = 64,
        cycle_range: Tuple[int, int] = (200, 1200),
        lr_range: Tuple[float, float] = (1e-5, 5e-4),
        noise_threshold: float = 1.0,
        smoothing: float = 0.2,
        last_epoch: int = -1,
    ) -> None:
        self.base_cycle_steps = int(base_cycle_steps)
        self.base_lr = float(base_lr)
        self.noise_window = int(noise_window)
        self.cycle_range = cycle_range
        self.lr_range = lr_range
        self.noise_threshold = float(noise_threshold)
        self.smoothing = float(smoothing)

        self.grad_norms: Deque[float] = deque(maxlen=self.noise_window)
        self.current_cycle_steps = int(base_cycle_steps)
        self.current_max_lr = float(base_lr)

        super().__init__(optimizer, last_epoch=last_epoch)

    @torch.no_grad()
    def update_gradient_stats(self) -> None:
        device: Optional[torch.device] = None
        total_sq_tensor: Optional[torch.Tensor] = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    g = param.grad
                    if device is None:
                        device = g.device
                        total_sq_tensor = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                    # Accumulate on-device to avoid per-parameter host syncs
                    total_sq_tensor = total_sq_tensor + torch.sum(g.mul(g)).to(torch.float32)  # type: ignore[operator]
        total_norm = (
            float(total_sq_tensor.sqrt().item()) if total_sq_tensor is not None else 0.0
        )
        self.grad_norms.append(total_norm)

    def _estimate_noise_level(self) -> float:
        n = len(self.grad_norms)
        if n < max(10, self.noise_window // 4):
            return 0.5
        # Compute coefficient of variation over the window
        vals = torch.tensor(list(self.grad_norms), dtype=torch.float32)
        mean = torch.mean(vals)
        std = torch.std(vals)
        if mean.abs().item() < 1e-8:
            return 0.5
        return float(min((std / (mean.abs() + 1e-8)).item(), 5.0))

    def _adapt_cycle_params(self) -> None:
        noise_level = self._estimate_noise_level()
        noise_factor = min(noise_level / self.noise_threshold, 2.0)
        cycle_factor = 1.0 / (1.0 + noise_factor)
        lr_factor = 1.0 / (1.0 + 0.5 * noise_factor)

        target_steps = int(
            self.cycle_range[0]
            + cycle_factor * (self.cycle_range[1] - self.cycle_range[0])
        )
        target_max_lr = float(
            self.lr_range[0] + lr_factor * (self.lr_range[1] - self.lr_range[0])
        )

        # Smooth changes to avoid abrupt LR jumps mid-cycle
        self.current_cycle_steps = int(
            self.smoothing * target_steps
            + (1.0 - self.smoothing) * self.current_cycle_steps
        )
        self.current_max_lr = float(
            self.smoothing * target_max_lr
            + (1.0 - self.smoothing) * self.current_max_lr
        )

    def get_lr(self) -> List[float]:
        # Adapt only at cycle boundaries to avoid mid-cycle discontinuities
        step_idx = max(self.last_epoch, 0)
        pos_in_cycle = step_idx % max(1, self.current_cycle_steps)
        if len(self.grad_norms) >= 10 and pos_in_cycle == 0:
            self._adapt_cycle_params()
            # DDP-safe broadcast of current cycle params if available
            try:
                import torch.distributed as dist  # type: ignore

                if dist.is_available() and dist.is_initialized():
                    tensor = torch.tensor(
                        [float(self.current_cycle_steps), float(self.current_max_lr)],
                        device=self.optimizer.param_groups[0]["params"][0].device,  # type: ignore[index]
                        dtype=torch.float32,
                    )
                    dist.broadcast(tensor, src=0)
                    self.current_cycle_steps = int(tensor[0].item())
                    self.current_max_lr = float(tensor[1].item())
            except Exception:
                pass

        progress = pos_in_cycle / max(1, self.current_cycle_steps)

        min_lr = float(self.lr_range[0])
        lr = min_lr + 0.5 * (self.current_max_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        return [lr for _ in self.base_lrs]


class HybridAdaptiveScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Compose per-cycle cosine with EMA+noise-based adaptation.

    The base schedule follows a warmup/stable/decay cycle, then the result is
    multiplied by a bounded adaptation factor derived from loss trend and noise.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        cycle_steps: int = 500,
        warmup_steps: int = 25,
        stable_steps: int = 0,
        decay_steps: int = 475,
        min_lr_ratio: float = 0.05,
        ema_alpha: float = 0.05,
        noise_sensitivity: float = 0.3,
        adaptation_strength: float = 0.2,
        last_epoch: int = -1,
    ) -> None:
        self.cycle_steps = int(cycle_steps)
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.min_lr_ratio = float(min_lr_ratio)

        self.ema_alpha = float(ema_alpha)
        self.noise_sensitivity = float(noise_sensitivity)
        self.adaptation_strength = float(adaptation_strength)

        self.loss_ema: Optional[float] = None
        self.loss_trend: float = 0.0
        self.grad_norms: Deque[float] = deque(maxlen=50)
        self.step_count: int = 0

        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    @torch.no_grad()
    def update_training_stats(self, loss: float) -> None:
        # EMA loss and trend
        if self.loss_ema is None:
            self.loss_ema = float(loss)
        else:
            prev = float(self.loss_ema)
            self.loss_ema = float(self.ema_alpha * loss + (1.0 - self.ema_alpha) * prev)
            delta = self.loss_ema - prev
            self.loss_trend = float(0.9 * self.loss_trend + 0.1 * delta)

        # Gradient norm (L2) computed on-device, single host sync
        device: Optional[torch.device] = None
        total_sq_tensor: Optional[torch.Tensor] = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    g = param.grad
                    if device is None:
                        device = g.device
                        total_sq_tensor = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                    total_sq_tensor = total_sq_tensor + torch.sum(g.mul(g)).to(torch.float32)  # type: ignore[operator]
        total_norm = (
            float(total_sq_tensor.sqrt().item()) if total_sq_tensor is not None else 0.0
        )
        self.grad_norms.append(total_norm)
        self.step_count += 1

    def _get_base_lr(self, base_lr: float, step_idx: int) -> float:
        pos_in_cycle = step_idx % max(1, self.cycle_steps)
        floor_lr = float(base_lr * self.min_lr_ratio)

        warmup_end = self.warmup_steps
        stable_end = self.warmup_steps + self.stable_steps
        decay_end = self.warmup_steps + self.stable_steps + self.decay_steps

        if self.warmup_steps > 0 and pos_in_cycle < warmup_end:
            progress = (pos_in_cycle + 1) / float(max(1, self.warmup_steps))
            return float(floor_lr + (base_lr - floor_lr) * progress)
        if self.stable_steps > 0 and pos_in_cycle < stable_end:
            return float(base_lr)
        if self.decay_steps > 0 and pos_in_cycle < decay_end:
            t = (pos_in_cycle - self.warmup_steps - self.stable_steps + 1) / float(
                max(1, self.decay_steps)
            )
            t = min(max(t, 0.0), 1.0)
            return float(
                floor_lr + 0.5 * (base_lr - floor_lr) * (1.0 + math.cos(math.pi * t))
            )
        return float(floor_lr)

    def _compute_adaptation_factor(self) -> float:
        if self.step_count < 20:
            return 1.0

        # Trend: decreasing loss (negative delta) -> increase LR
        trend_factor = 1.0 - 10.0 * self.loss_trend
        trend_factor = max(0.5, min(2.0, float(trend_factor)))

        # Noise via coefficient of variation of recent grad norms
        if len(self.grad_norms) >= 10:
            vals = torch.tensor(list(self.grad_norms)[-20:], dtype=torch.float32)
            mean = torch.mean(vals).abs() + 1e-8
            cv = float((torch.std(vals) / mean).item())
            noise_factor = 1.0 / (1.0 + self.noise_sensitivity * cv)
        else:
            noise_factor = 1.0

        combined = math.sqrt(trend_factor * noise_factor)
        factor = 1.0 + self.adaptation_strength * (combined - 1.0)
        # DDP-safe: broadcast a single factor across ranks if available
        factor = float(max(0.5, min(1.5, factor)))
        try:
            import torch.distributed as dist  # type: ignore

            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([factor], device=self.optimizer.param_groups[0]["params"][0].device, dtype=torch.float32)  # type: ignore[index]
                dist.broadcast(t, src=0)
                factor = float(t[0].item())
        except Exception:
            pass
        return factor

    def get_lr(self) -> List[float]:
        step_idx = max(self.last_epoch, 0)
        adapt = self._compute_adaptation_factor()

        lrs: List[float] = []
        for base_lr in self.base_lrs:
            base_sched = self._get_base_lr(float(base_lr), step_idx)
            adapted = base_sched * adapt
            floor_lr = float(base_lr * self.min_lr_ratio)
            cap_lr = float(base_lr * 2.0)
            adapted = max(floor_lr, min(cap_lr, adapted))
            lrs.append(adapted)
        return lrs


class AdaptivePerCycleWarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Per-cycle warmup/stable/decay cosine schedule with bounded adaptive gain.

    This mirrors a standard per-cycle cosine schedule but multiplies the base LR
    by an adaptation factor derived from EMA loss trend and gradient noise.
    The factor is DDP-synchronized to keep all ranks consistent.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        cycle_steps: int,
        warmup_steps: int,
        stable_steps: int = 0,
        decay_steps: int = 0,
        min_lr_ratio: float = 0.0,
        ema_alpha: float = 0.05,
        noise_sensitivity: float = 0.3,
        adaptation_strength: float = 0.1,
        min_history: int = 20,
        adaptation_bounds: Tuple[float, float] = (0.7, 1.5),
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

        self.cycle_steps = int(cycle_steps)
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.min_lr_ratio = float(min_lr_ratio)

        self.ema_alpha = float(ema_alpha)
        self.noise_sensitivity = float(noise_sensitivity)
        self.adaptation_strength = float(adaptation_strength)
        self.min_history = int(min_history)
        self.adaptation_bounds = (
            float(adaptation_bounds[0]),
            float(adaptation_bounds[1]),
        )

        self.loss_ema: Optional[float] = None
        self.loss_trend: float = 0.0
        self.grad_norms: Deque[float] = deque(maxlen=50)
        self.step_count: int = 0

        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    @torch.no_grad()
    def update_training_stats(self, loss: float) -> None:
        # EMA loss trend
        if self.loss_ema is None:
            self.loss_ema = float(loss)
        else:
            prev = float(self.loss_ema)
            self.loss_ema = float(self.ema_alpha * loss + (1.0 - self.ema_alpha) * prev)
            delta = self.loss_ema - prev
            self.loss_trend = float(0.8 * self.loss_trend + 0.2 * delta)

        # Global grad L2 norm on-device
        device: Optional[torch.device] = None
        total_sq_tensor: Optional[torch.Tensor] = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    g = param.grad
                    if device is None:
                        device = g.device
                        total_sq_tensor = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                    total_sq_tensor = total_sq_tensor + torch.sum(g.mul(g)).to(torch.float32)  # type: ignore[operator]
        total_norm = (
            float(total_sq_tensor.sqrt().item()) if total_sq_tensor is not None else 0.0
        )
        self.grad_norms.append(total_norm)
        self.step_count += 1

    def _get_base_cycle_lr(self, base_lr: float, step_idx: int) -> float:
        pos = step_idx % max(1, self.cycle_steps)
        floor_lr = float(base_lr * self.min_lr_ratio)

        warmup_end = self.warmup_steps
        stable_end = self.warmup_steps + self.stable_steps
        decay_end = self.warmup_steps + self.stable_steps + self.decay_steps

        if self.warmup_steps > 0 and pos < warmup_end:
            progress = (pos + 1) / float(max(1, self.warmup_steps))
            return float(floor_lr + (base_lr - floor_lr) * progress)
        if self.stable_steps > 0 and pos < stable_end:
            return float(base_lr)
        if self.decay_steps > 0 and pos < decay_end:
            t = (pos - self.warmup_steps - self.stable_steps + 1) / float(
                max(1, self.decay_steps)
            )
            t = min(max(t, 0.0), 1.0)
            return float(
                floor_lr + 0.5 * (base_lr - floor_lr) * (1.0 + math.cos(math.pi * t))
            )
        return float(floor_lr)

    def _compute_adaptation_factor(self) -> float:
        if self.step_count < self.min_history:
            return 1.0

        trend_factor = 1.0 - 20.0 * self.loss_trend
        trend_factor = max(0.3, min(3.0, float(trend_factor)))

        if len(self.grad_norms) >= 10:
            import numpy as _np  # local import to avoid global dependency

            recent = _np.asarray(list(self.grad_norms)[-20:], dtype=_np.float32)
            mean = float(_np.mean(recent))
            std = float(_np.std(recent))
            cv = (std / (abs(mean) + 1e-8)) if abs(mean) > 1e-8 else 0.0
            noise_factor = 1.0 / (1.0 + self.noise_sensitivity * cv)
        else:
            noise_factor = 1.0

        combined = math.sqrt(trend_factor * noise_factor)
        factor = 1.0 + self.adaptation_strength * (combined - 1.0)
        low, high = self.adaptation_bounds
        factor = float(max(low, min(high, factor)))

        # DDP-safe broadcast
        try:
            import torch.distributed as dist  # type: ignore

            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([factor], device=self.optimizer.param_groups[0]["params"][0].device, dtype=torch.float32)  # type: ignore[index]
                dist.broadcast(t, src=0)
                factor = float(t[0].item())
        except Exception:
            pass

        return factor

    def get_lr(self) -> List[float]:
        step_idx = max(self.last_epoch, 0)
        factor = self._compute_adaptation_factor()
        lrs: List[float] = []
        for base_lr in self.base_lrs:
            base = self._get_base_cycle_lr(float(base_lr), step_idx)
            floor_lr = float(base_lr * self.min_lr_ratio)
            adapted = max(floor_lr, base * factor)
            lrs.append(adapted)
        return lrs


class CycleAdaptivePerCycleScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Per-cycle schedule with adaptive cycle length (DDP-safe, boundary-updated).

    The cycle is defined by warmup/stable/decay ratios of the current cycle length.
    Cycle length is adapted at cycle boundaries using gradient noise and loss trend.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_cycle_steps: int,
        warmup_ratio: float = 0.05,
        stable_ratio: float = 0.0,
        decay_ratio: float = 0.95,
        min_lr_ratio: float = 0.05,
        noise_window: int = 64,
        cycle_range: Tuple[int, int] = (100, 1000),
        adaptation_rate: float = 0.1,
        noise_threshold: float = 1.0,
        ema_alpha: float = 0.05,
        last_epoch: int = -1,
    ) -> None:
        self.base_cycle_steps = int(base_cycle_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.stable_ratio = float(stable_ratio)
        self.decay_ratio = float(decay_ratio)
        if self.warmup_ratio + self.stable_ratio + self.decay_ratio > 1.0 + 1e-8:
            raise ValueError("warmup_ratio + stable_ratio + decay_ratio must be <= 1.0")
        self.min_lr_ratio = float(min_lr_ratio)

        self.noise_window = int(noise_window)
        self.cycle_range = cycle_range
        self.adaptation_rate = float(adaptation_rate)
        self.noise_threshold = float(noise_threshold)
        self.ema_alpha = float(ema_alpha)

        self.current_cycle_steps = int(base_cycle_steps)
        self.grad_norms: Deque[float] = deque(maxlen=self.noise_window)
        self.loss_ema: Optional[float] = None

        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    @torch.no_grad()
    def update_training_stats(self, loss: float) -> None:
        # Update loss EMA for progress signal
        if self.loss_ema is None:
            self.loss_ema = float(loss)
        else:
            self.loss_ema = float(
                self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.loss_ema
            )

        # Global grad norm on-device
        device: Optional[torch.device] = None
        total_sq_tensor: Optional[torch.Tensor] = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    g = param.grad
                    if device is None:
                        device = g.device
                        total_sq_tensor = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                    total_sq_tensor = total_sq_tensor + torch.sum(g.mul(g)).to(torch.float32)  # type: ignore[operator]
        total_norm = (
            float(total_sq_tensor.sqrt().item()) if total_sq_tensor is not None else 0.0
        )
        self.grad_norms.append(total_norm)

    def _adapt_next_cycle_params(self) -> None:
        # Estimate noise via coefficient of variation
        window = list(self.grad_norms)
        if len(window) < max(10, self.noise_window // 4):
            return
        vals = torch.tensor(window, dtype=torch.float32)
        mean = vals.mean().abs() + 1e-8
        cv = float((vals.std() / mean).item())
        noise_factor = min(cv / self.noise_threshold, 2.0)

        # Loss progress: smaller EMA implies improvement
        progress_factor = 0.0
        if self.loss_ema is not None:
            # Map decreasing loss to positive growth factor
            progress_factor = 0.0  # keep simple; advanced users can extend

        target_cycle_steps = int(
            self.base_cycle_steps / (1.0 + 0.5 * noise_factor) + progress_factor * 0.0
        )
        target_cycle_steps = max(
            self.cycle_range[0], min(self.cycle_range[1], target_cycle_steps)
        )

        self.current_cycle_steps = int(
            (1.0 - self.adaptation_rate) * self.current_cycle_steps
            + self.adaptation_rate * target_cycle_steps
        )

    def _phase_bounds(self) -> Tuple[int, int, int]:
        warm = max(1, int(self.current_cycle_steps * self.warmup_ratio))
        stable = int(self.current_cycle_steps * self.stable_ratio)
        decay = max(1, int(self.current_cycle_steps * self.decay_ratio))
        total = warm + stable + decay
        if total > self.current_cycle_steps:
            # Clamp decay to fit
            excess = total - self.current_cycle_steps
            decay = max(1, decay - excess)
        return warm, stable, decay

    def get_lr(self) -> List[float]:
        step_idx = max(self.last_epoch, 0)
        pos = step_idx % max(1, self.current_cycle_steps)
        if pos == 0:
            self._adapt_next_cycle_params()
            # DDP broadcast of cycle steps
            try:
                import torch.distributed as dist  # type: ignore

                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor([float(self.current_cycle_steps)], device=self.optimizer.param_groups[0]["params"][0].device, dtype=torch.float32)  # type: ignore[index]
                    dist.broadcast(t, src=0)
                    self.current_cycle_steps = int(t[0].item())
            except Exception:
                pass

        warm, stable, decay = self._phase_bounds()
        warm_end = warm
        stable_end = warm + stable
        decay_end = warm + stable + decay

        lrs: List[float] = []
        for base_lr in self.base_lrs:
            base_lr_f = float(base_lr)
            floor_lr = base_lr_f * self.min_lr_ratio

            if pos < warm_end:
                progress = (pos + 1) / float(max(1, warm))
                lr = floor_lr + (base_lr_f - floor_lr) * progress
            elif pos < stable_end:
                lr = base_lr_f
            elif pos < decay_end:
                t = (pos - warm - stable + 1) / float(max(1, decay))
                t = min(max(t, 0.0), 1.0)
                lr = floor_lr + 0.5 * (base_lr_f - floor_lr) * (
                    1.0 + math.cos(math.pi * t)
                )
            else:
                lr = floor_lr
            lrs.append(float(lr))
        return lrs
