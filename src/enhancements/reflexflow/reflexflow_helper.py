from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class ReflexFlowState:
    applied: bool
    warmup_active: bool
    simulation_timestep_mean: float
    simulation_timestep_std: float
    history_frame_start: int
    history_frame_end: int


class ReflexFlowHelper:
    """Standalone ReflexFlow scheduled-sampling helper (train-time, default off)."""

    def __init__(self, args: Any, noise_scheduler: Any) -> None:
        self.enabled = bool(getattr(args, "enable_reflexflow", False))
        self.warmup_steps = int(getattr(args, "reflexflow_warmup_steps", 0))
        self.apply_prob = float(getattr(args, "reflexflow_apply_prob", 1.0))
        self.apply_prob_start = float(
            getattr(args, "reflexflow_apply_prob_start", self.apply_prob)
        )
        self.apply_prob_end = float(
            getattr(args, "reflexflow_apply_prob_end", self.apply_prob)
        )
        self.apply_prob_ramp_steps = int(
            getattr(args, "reflexflow_apply_prob_ramp_steps", 0)
        )
        self.apply_prob_start_step = int(
            getattr(args, "reflexflow_apply_prob_start_step", 0)
        )
        self.apply_prob_ramp_shape = str(
            getattr(args, "reflexflow_apply_prob_ramp_shape", "linear")
        ).lower()
        self.shift = float(getattr(args, "reflexflow_shift", 0.6))
        self.min_t = float(getattr(args, "reflexflow_min_t", 0.05))
        self.max_t = float(getattr(args, "reflexflow_max_t", 0.95))
        self.blend = float(getattr(args, "reflexflow_blend", 1.0))
        self.history_start_frame = int(getattr(args, "reflexflow_history_start_frame", 0))
        self.history_exclude_tail_frames = int(
            getattr(args, "reflexflow_history_exclude_tail_frames", 1)
        )
        self.log_interval = int(getattr(args, "reflexflow_log_interval", 50))
        self.iteration_count = 0
        self._warned_invalid_shape = False

        self.num_train_timesteps = int(
            getattr(
                getattr(noise_scheduler, "config", object()),
                "num_train_timesteps",
                1000,
            )
            or 1000
        )

    def setup_hooks(self) -> None:
        """Reserved for parity with other enhancement helpers."""

    def remove_hooks(self) -> None:
        """Reserved for parity with other enhancement helpers."""

    def _sample_shifted_logit_normal(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = torch.sigmoid(
            torch.randn(batch_size, device=device, dtype=torch.float32)
        )
        shifted = self.shift * base / (1.0 + (self.shift - 1.0) * base)
        shifted = shifted.clamp(min=self.min_t, max=self.max_t)
        return shifted.to(dtype=dtype)

    def _resolve_apply_probability(self, step: int) -> float:
        prob = self.apply_prob
        if self.apply_prob_ramp_steps > 0:
            if step < self.apply_prob_start_step:
                prob = self.apply_prob_start
            else:
                progress = float(step - self.apply_prob_start_step) / float(
                    max(self.apply_prob_ramp_steps, 1)
                )
                progress = max(0.0, min(1.0, progress))
                if self.apply_prob_ramp_shape == "cosine":
                    progress = 0.5 - 0.5 * math.cos(math.pi * progress)
                prob = self.apply_prob_start + (
                    self.apply_prob_end - self.apply_prob_start
                ) * progress
        return float(max(0.0, min(1.0, prob)))

    def _resolve_history_slice(self, num_frames: int) -> Tuple[int, int]:
        start = max(0, self.history_start_frame)
        end = max(start, num_frames - self.history_exclude_tail_frames)
        end = min(end, num_frames)
        return start, end

    def _build_state(
        self,
        *,
        applied: bool,
        warmup_active: bool,
        simulation_timestep_mean: float,
        simulation_timestep_std: float,
        history_frame_start: int,
        history_frame_end: int,
    ) -> ReflexFlowState:
        return ReflexFlowState(
            applied=applied,
            warmup_active=warmup_active,
            simulation_timestep_mean=simulation_timestep_mean,
            simulation_timestep_std=simulation_timestep_std,
            history_frame_start=history_frame_start,
            history_frame_end=history_frame_end,
        )

    def apply_to_inputs(
        self,
        noisy_model_input: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        global_step: Optional[int],
    ) -> Tuple[torch.Tensor, Optional[ReflexFlowState]]:
        if not self.enabled:
            return noisy_model_input, None

        self.iteration_count += 1
        step = int(global_step or 0)
        effective_apply_prob = self._resolve_apply_probability(step)
        warmup_active = step < self.warmup_steps
        if warmup_active:
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=True,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
            )

        if noisy_model_input.dim() not in (4, 5) or latents.dim() != noisy_model_input.dim():
            if not self._warned_invalid_shape:
                logger.warning(
                    "ReflexFlow scheduled sampling skipped: expected 4D/5D tensors, got noisy=%s latents=%s noise=%s",
                    tuple(noisy_model_input.shape),
                    tuple(latents.shape),
                    tuple(noise.shape),
                )
                self._warned_invalid_shape = True
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
            )

        if noisy_model_input.shape != latents.shape or noisy_model_input.shape != noise.shape:
            if not self._warned_invalid_shape:
                logger.warning(
                    "ReflexFlow scheduled sampling skipped: mismatched shapes noisy=%s latents=%s noise=%s",
                    tuple(noisy_model_input.shape),
                    tuple(latents.shape),
                    tuple(noise.shape),
                )
                self._warned_invalid_shape = True
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
            )

        if effective_apply_prob <= 0.0:
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
            )

        if effective_apply_prob < 1.0 and (
            torch.rand((), device=noisy_model_input.device).item() > effective_apply_prob
        ):
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
            )

        batch_size = noisy_model_input.shape[0]
        ts = self._sample_shifted_logit_normal(
            batch_size=batch_size,
            device=noisy_model_input.device,
            dtype=noisy_model_input.dtype,
        )
        ts_view = ts.view(batch_size, *([1] * (noisy_model_input.dim() - 1)))
        resampled = ((1.0 - ts_view) * latents + ts_view * noise).detach()

        history_start = 0
        history_end = 1
        blended = noisy_model_input.clone()
        if noisy_model_input.dim() == 5:
            history_start, history_end = self._resolve_history_slice(noisy_model_input.shape[2])
            if history_end <= history_start:
                return noisy_model_input, self._build_state(
                    applied=False,
                    warmup_active=False,
                    simulation_timestep_mean=float(ts.mean().item()),
                    simulation_timestep_std=float(ts.std(unbiased=False).item()),
                    history_frame_start=history_start,
                    history_frame_end=history_end,
                )
            src_slice = noisy_model_input[:, :, history_start:history_end, ...]
            resampled_slice = resampled[:, :, history_start:history_end, ...]
            if self.blend >= 1.0:
                blended[:, :, history_start:history_end, ...] = resampled_slice
            else:
                blended[:, :, history_start:history_end, ...] = (
                    (1.0 - self.blend) * src_slice + self.blend * resampled_slice
                )
        else:
            if self.blend >= 1.0:
                blended = resampled
            else:
                blended = (1.0 - self.blend) * noisy_model_input + self.blend * resampled

        return blended, self._build_state(
            applied=True,
            warmup_active=False,
            simulation_timestep_mean=float(ts.mean().item()),
            simulation_timestep_std=float(ts.std(unbiased=False).item()),
            history_frame_start=history_start,
            history_frame_end=history_end,
        )

    def state_to_metrics(self, state: Optional[ReflexFlowState]) -> Dict[str, float]:
        if state is None:
            return {}
        history_span = max(0, state.history_frame_end - state.history_frame_start)
        return {
            "reflexflow/applied": 1.0 if state.applied else 0.0,
            "reflexflow/warmup_active": 1.0 if state.warmup_active else 0.0,
            "reflexflow/simulation_t_mean": state.simulation_timestep_mean,
            "reflexflow/simulation_t_std": state.simulation_timestep_std,
            "reflexflow/history_span_frames": float(history_span),
        }
