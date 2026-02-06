from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class SelfResamplingState:
    applied: bool
    warmup_active: bool
    simulation_timestep_mean: float
    simulation_timestep_std: float
    history_frame_start: int
    history_frame_end: int
    autoregressive_history_active: bool
    autoregressive_decay: float
    model_rollout_active: bool
    model_rollout_steps: int


class SelfResamplingHelper:
    """Train-time detached history resampling (LoRA-only, gated, default off)."""

    def __init__(self, args: Any, noise_scheduler: Any) -> None:
        self.enabled = bool(getattr(args, "enable_self_resampling", False))
        self.warmup_steps = int(getattr(args, "self_resampling_warmup_steps", 0))
        self.apply_prob = float(getattr(args, "self_resampling_apply_prob", 1.0))
        self.shift = float(getattr(args, "self_resampling_shift", 0.6))
        self.min_t = float(getattr(args, "self_resampling_min_t", 0.05))
        self.max_t = float(getattr(args, "self_resampling_max_t", 0.95))
        self.blend = float(getattr(args, "self_resampling_blend", 1.0))
        self.history_start_frame = int(
            getattr(args, "self_resampling_history_start_frame", 0)
        )
        self.history_exclude_tail_frames = int(
            getattr(args, "self_resampling_history_exclude_tail_frames", 1)
        )
        self.autoregressive_history = bool(
            getattr(args, "self_resampling_autoregressive_history", True)
        )
        self.autoregressive_decay = float(
            getattr(args, "self_resampling_autoregressive_decay", 0.5)
        )
        self.autoregressive_clip_multiplier = float(
            getattr(args, "self_resampling_autoregressive_clip_multiplier", 3.0)
        )
        self.model_rollout = bool(
            getattr(args, "self_resampling_model_rollout", False)
        )
        self.model_rollout_steps = int(
            getattr(args, "self_resampling_model_rollout_steps", 1)
        )
        self.log_interval = int(getattr(args, "self_resampling_log_interval", 50))
        self.iteration_count = 0
        self._warned_invalid_shape = False
        self._warned_missing_rollout_predictor = False
        self._warned_rollout_failure = False

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
        autoregressive_history_active: bool,
        model_rollout_active: bool,
    ) -> SelfResamplingState:
        return SelfResamplingState(
            applied=applied,
            warmup_active=warmup_active,
            simulation_timestep_mean=simulation_timestep_mean,
            simulation_timestep_std=simulation_timestep_std,
            history_frame_start=history_frame_start,
            history_frame_end=history_frame_end,
            autoregressive_history_active=autoregressive_history_active,
            autoregressive_decay=self.autoregressive_decay,
            model_rollout_active=model_rollout_active,
            model_rollout_steps=self.model_rollout_steps,
        )

    def _convert_rollout_timesteps(
        self,
        ts: torch.Tensor,
        timestep_reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if timestep_reference is None or not torch.is_tensor(timestep_reference):
            return ts * float(self.num_train_timesteps)

        ref = timestep_reference.detach().float()
        if ref.dim() == 0:
            ref = ref.view(1)
        while ref.dim() > 1:
            ref = ref.mean(dim=-1)

        if ref.numel() == ts.numel() and float(ref.max().item()) <= 1.5:
            return ts.to(dtype=timestep_reference.dtype, device=timestep_reference.device)

        scaled = ts * float(self.num_train_timesteps)
        return scaled.to(dtype=timestep_reference.dtype, device=timestep_reference.device)

    def _apply_autoregressive_error_accumulation(
        self,
        *,
        resampled: torch.Tensor,
        latents: torch.Tensor,
        history_start: int,
        history_end: int,
    ) -> Tuple[torch.Tensor, bool]:
        if not self.autoregressive_history or (history_end - history_start) <= 1:
            return resampled, False

        running_error = torch.zeros_like(
            resampled[:, :, history_start : history_start + 1, ...]
        )
        for frame_idx in range(history_start, history_end):
            frame_slice = slice(frame_idx, frame_idx + 1)
            clean_frame = latents[:, :, frame_slice, ...]
            sampled_frame = resampled[:, :, frame_slice, ...]
            frame_error = sampled_frame - clean_frame
            accum_error = frame_error + (self.autoregressive_decay * running_error)
            clip_bound = (
                frame_error.detach().abs().mean(dim=(1, 2, 3, 4), keepdim=True)
                * self.autoregressive_clip_multiplier
            )
            clip_bound = torch.clamp(clip_bound, min=1e-6)
            accum_error = torch.clamp(accum_error, -clip_bound, clip_bound)
            resampled[:, :, frame_slice, ...] = (clean_frame + accum_error).detach()
            running_error = accum_error.detach()

        return resampled, True

    def _rollout_history_with_model(
        self,
        *,
        noisy_model_input: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        ts: torch.Tensor,
        history_start: int,
        history_end: int,
        predict_velocity_fn: Callable[..., torch.Tensor],
        timestep_reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = noisy_model_input.shape[0]
        ts_view = ts.view(batch_size, 1, 1, 1, 1)
        x_ts = ((1.0 - ts_view) * latents + ts_view * noise).detach()
        probe_input = noisy_model_input.detach().clone()
        degraded_history = latents.detach().clone()

        predictor_cache: Dict[str, Any] = {}
        for frame_idx in range(history_start, history_end):
            frame_slice = slice(frame_idx, frame_idx + 1)
            if frame_idx > history_start:
                probe_input[:, :, history_start:frame_idx, ...] = degraded_history[
                    :, :, history_start:frame_idx, ...
                ]

            current_frame = x_ts[:, :, frame_slice, ...].clone()
            for rollout_step in range(self.model_rollout_steps):
                step_frac = 1.0 - (float(rollout_step) / float(self.model_rollout_steps))
                step_t = ts * step_frac
                model_t = self._convert_rollout_timesteps(step_t, timestep_reference)
                probe_input[:, :, frame_slice, ...] = current_frame
                predictor_cache["history_frame_count"] = max(
                    0, frame_idx - history_start
                )
                predictor_cache["frame_index"] = frame_idx
                predictor_cache["history_start"] = history_start
                predictor_cache["history_end"] = history_end
                predictor_cache["rollout_step"] = rollout_step
                predictor_cache["rollout_step_frac"] = step_frac
                try:
                    velocity = predict_velocity_fn(
                        probe_input,
                        model_t,
                        predictor_cache,
                    )
                except TypeError:
                    velocity = predict_velocity_fn(probe_input, model_t)
                frame_velocity = velocity[:, :, frame_slice, ...]
                current_frame = current_frame - (
                    ts_view / float(self.model_rollout_steps)
                ) * frame_velocity

            clean_frame = latents[:, :, frame_slice, ...]
            base_delta = x_ts[:, :, frame_slice, ...] - clean_frame
            delta = current_frame - clean_frame
            clip_bound = (
                base_delta.detach().abs().mean(dim=(1, 2, 3, 4), keepdim=True)
                * self.autoregressive_clip_multiplier
            )
            clip_bound = torch.clamp(clip_bound, min=1e-6)
            current_frame = clean_frame + torch.clamp(delta, -clip_bound, clip_bound)
            degraded_history[:, :, frame_slice, ...] = current_frame.detach()

        return degraded_history

    def apply_to_inputs(
        self,
        noisy_model_input: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        global_step: Optional[int],
        predict_velocity_fn: Optional[
            Callable[..., torch.Tensor]
        ] = None,
        timestep_reference: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[SelfResamplingState]]:
        if not self.enabled:
            return noisy_model_input, None

        self.iteration_count += 1
        step = int(global_step or 0)
        warmup_active = step < self.warmup_steps
        if warmup_active:
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=True,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
                autoregressive_history_active=False,
                model_rollout_active=False,
            )

        if noisy_model_input.dim() != 5 or latents.dim() != 5 or noise.dim() != 5:
            if not self._warned_invalid_shape:
                logger.warning(
                    "Self-resampling skipped: expected 5D tensors, got noisy=%s latents=%s noise=%s",
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
                autoregressive_history_active=False,
                model_rollout_active=False,
            )

        if (
            noisy_model_input.shape != latents.shape
            or noisy_model_input.shape != noise.shape
        ):
            if not self._warned_invalid_shape:
                logger.warning(
                    "Self-resampling skipped: mismatched shapes noisy=%s latents=%s noise=%s",
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
                autoregressive_history_active=False,
                model_rollout_active=False,
            )

        if self.apply_prob < 1.0 and (
            torch.rand((), device=noisy_model_input.device).item() > self.apply_prob
        ):
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=0.0,
                simulation_timestep_std=0.0,
                history_frame_start=0,
                history_frame_end=0,
                autoregressive_history_active=False,
                model_rollout_active=False,
            )

        batch_size = noisy_model_input.shape[0]
        ts = self._sample_shifted_logit_normal(
            batch_size=batch_size,
            device=noisy_model_input.device,
            dtype=noisy_model_input.dtype,
        )
        ts_view = ts.view(batch_size, 1, 1, 1, 1)
        history_start, history_end = self._resolve_history_slice(
            noisy_model_input.shape[2]
        )

        if history_end <= history_start:
            return noisy_model_input, self._build_state(
                applied=False,
                warmup_active=False,
                simulation_timestep_mean=float(ts.mean().item()),
                simulation_timestep_std=float(ts.std(unbiased=False).item()),
                history_frame_start=history_start,
                history_frame_end=history_end,
                autoregressive_history_active=False,
                model_rollout_active=False,
            )

        resampled = ((1.0 - ts_view) * latents + ts_view * noise).detach()
        autoregressive_active = False
        model_rollout_active = False

        if self.model_rollout:
            if predict_velocity_fn is None:
                if not self._warned_missing_rollout_predictor:
                    logger.warning(
                        "Self-resampling model rollout requested but no predictor callback was provided; falling back to detached corruption."
                    )
                    self._warned_missing_rollout_predictor = True
            else:
                try:
                    resampled = self._rollout_history_with_model(
                        noisy_model_input=noisy_model_input,
                        latents=latents,
                        noise=noise,
                        ts=ts,
                        history_start=history_start,
                        history_end=history_end,
                        predict_velocity_fn=predict_velocity_fn,
                        timestep_reference=timestep_reference,
                    )
                    autoregressive_active = True
                    model_rollout_active = True
                except Exception as exc:
                    if not self._warned_rollout_failure:
                        logger.warning(
                            "Self-resampling model rollout failed once (%s); falling back to detached corruption.",
                            exc,
                        )
                        self._warned_rollout_failure = True

        if not model_rollout_active:
            resampled, autoregressive_active = self._apply_autoregressive_error_accumulation(
                resampled=resampled,
                latents=latents,
                history_start=history_start,
                history_end=history_end,
            )

        blended = noisy_model_input.clone()
        src_slice = noisy_model_input[:, :, history_start:history_end, ...]
        resampled_slice = resampled[:, :, history_start:history_end, ...]
        if self.blend >= 1.0:
            blended[:, :, history_start:history_end, ...] = resampled_slice
        else:
            blended[:, :, history_start:history_end, ...] = (
                (1.0 - self.blend) * src_slice + self.blend * resampled_slice
            )

        return blended, self._build_state(
            applied=True,
            warmup_active=False,
            simulation_timestep_mean=float(ts.mean().item()),
            simulation_timestep_std=float(ts.std(unbiased=False).item()),
            history_frame_start=history_start,
            history_frame_end=history_end,
            autoregressive_history_active=autoregressive_active,
            model_rollout_active=model_rollout_active,
        )

    def state_to_metrics(
        self, state: Optional[SelfResamplingState]
    ) -> Dict[str, float]:
        if state is None:
            return {}
        history_span = max(0, state.history_frame_end - state.history_frame_start)
        return {
            "self_resampling/applied": 1.0 if state.applied else 0.0,
            "self_resampling/warmup_active": 1.0 if state.warmup_active else 0.0,
            "self_resampling/simulation_t_mean": state.simulation_timestep_mean,
            "self_resampling/simulation_t_std": state.simulation_timestep_std,
            "self_resampling/history_span_frames": float(history_span),
            "self_resampling/autoregressive_history_active": (
                1.0 if state.autoregressive_history_active else 0.0
            ),
            "self_resampling/autoregressive_decay": state.autoregressive_decay,
            "self_resampling/model_rollout_active": (
                1.0 if state.model_rollout_active else 0.0
            ),
            "self_resampling/model_rollout_steps": float(state.model_rollout_steps),
        }
