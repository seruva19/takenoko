"""Transition training manager coordinating scheduler, transport, and hooks."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.ema import ExponentialMovingAverage

from .scheduler import (
    TransitionSchedulerConfig,
    TransitionTimesteps,
    TransitionTimestepScheduler,
)
from .transport import BaseTransport, TransportState, create_transport
from .derivatives import DerivativeConfig, compute_derivative
from .lora_interval_adapter import IntervalModulationConfig, LoRAIntervalAdapter


@dataclass
class PreparedBatch:
    noisy_latents: Tensor
    timesteps: Tensor
    reference_timesteps: Tensor
    weights: Tensor
    need_directional: bool


@dataclass
class TransitionBatchState:
    latents: Tensor
    noise: Tensor
    t: Tensor
    r: Tensor
    n_diffusion: int
    weights: Tensor
    transport_state: TransportState


class TransitionTrainingManager:
    """Main entry-point for transition-training specific logic."""

    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

        self.transport: BaseTransport = create_transport(config.transport)
        self.scheduler = TransitionTimestepScheduler(
            TransitionSchedulerConfig(
                diffusion_ratio=config.diffusion_ratio,
                consistency_ratio=config.consistency_ratio,
                weight_schedule=config.weight_schedule,
                use_tangent_weighting=config.tangent_weighting,
                use_adaptive_weighting=config.adaptive_weighting,
                t_min=self.transport.t_min,
                t_max=self.transport.t_max,
            )
        )
        self.derivative_cfg = DerivativeConfig(
            mode=config.derivative_mode,
            epsilon=config.finite_difference_eps,
            failover_mode=config.derivative_failover,
        )
        self.lora_adapter = LoRAIntervalAdapter(
            IntervalModulationConfig(
                enabled=config.lora_interval_modulation,
                delta_attention=config.delta_attention,
                mlp_hidden=config.delta_mlp_hidden,
            )
        )

        self.directional_weight = config.directional_loss_weight
        self.use_ema_teacher = config.use_ema_teacher and config.teacher_mix > 0.0
        self.teacher_mix = config.teacher_mix if self.use_ema_teacher else 0.0
        self.teacher_decay = config.teacher_decay
        self.teacher_ema: Optional[ExponentialMovingAverage] = None

        self._last_state: Optional[TransitionBatchState] = None

    # ------------------------------------------------------------------ Teacher

    def attach_teacher(self, transformer) -> None:
        """Initialise EMA teacher on the provided transformer."""
        if not self.use_ema_teacher:
            return
        if self.teacher_ema is None:
            self.teacher_ema = ExponentialMovingAverage(
                parameters=[p for p in transformer.parameters() if p.requires_grad],
                decay=self.teacher_decay,
            )

    def update_teacher(self, transformer) -> None:
        if self.teacher_ema is None:
            return
        self.teacher_ema.update(
            [p for p in transformer.parameters() if p.requires_grad]
        )

    # ------------------------------------------------------------------ Helpers

    def should_request_intermediate(self) -> bool:
        return self.directional_weight > 0.0

    def _to_discrete(self, t: Tensor) -> Tensor:
        norm = self.transport.normalize(t).clamp(0.0, 1.0)
        return norm * 999.0 + 1.0

    # ------------------------------------------------------------------ Pipeline

    def prepare_batch(
        self,
        latents: Tensor,
        noise: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> PreparedBatch:
        timesteps = self.scheduler.sample_pairs(
            args=self.args,
            latents=latents,
            device=device,
            batch_size=batch_size,
            transport=self.transport,
        )

        transport_state = self.transport.prepare_state(
            timesteps.t, latents, noise
        )
        noisy_latents = transport_state.x_t
        weights = self.scheduler.compute_weights(timesteps)

        self._last_state = TransitionBatchState(
            latents=latents.detach(),
            noise=noise.detach(),
            t=timesteps.t.detach(),
            r=timesteps.r.detach(),
            n_diffusion=timesteps.n_diffusion,
            weights=weights.detach(),
            transport_state=transport_state,
        )

        discrete_t = self._to_discrete(timesteps.t).to(device=device, dtype=latents.dtype)
        discrete_r = self._to_discrete(timesteps.r).to(device=device, dtype=latents.dtype)

        return PreparedBatch(
            noisy_latents=noisy_latents,
            timesteps=discrete_t,
            reference_timesteps=discrete_r,
            weights=weights.to(device=device, dtype=latents.dtype),
            need_directional=self.should_request_intermediate(),
        )

    def finalize_batch(
        self,
        model_pred: Tensor,
        forward_fn,
        transformer,
        intermediate: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, float], Optional[Tensor]]:
        if self._last_state is None:
            raise RuntimeError("TransitionTrainingManager.finalize_batch called without prior state.")

        state = self._last_state
        derivative = compute_derivative(
            self.derivative_cfg,
            forward_fn=forward_fn,
            noisy_latents=state.transport_state.x_t,
            timesteps=self._to_discrete(state.t),
        )

        teacher_pred = None
        if self.teacher_ema is not None and self.teacher_mix > 0.0:
            with torch.no_grad():
                with self.teacher_ema.average_parameters(transformer.parameters()):
                    teacher_pred = forward_fn(state.transport_state.x_t, self._to_discrete(state.t))
            if teacher_pred is not None:
                teacher_pred = (1.0 - self.teacher_mix) * state.transport_state.v_t + self.teacher_mix * teacher_pred

        target = self.transport.compute_target(
            state.transport_state,
            state.latents,
            state.noise,
            state.t,
            state.r,
            derivative,
            teacher_pred=teacher_pred,
        )

        weights = state.weights
        metrics: Dict[str, float] = {
            "transition_training/diffusion_fraction": float(state.n_diffusion) / float(model_pred.size(0)),
            "transition_training/avg_delta_t": float((state.t - state.r).mean().item()),
        }
        mode_map = {"dde": 0.0, "jvp": 1.0, "none": 2.0, "auto": 3.0}
        metrics["transition_training/derivative_mode"] = mode_map.get(
            self.derivative_cfg.mode, 0.0
        )
        if self.teacher_ema is not None and self.teacher_mix > 0.0:
            metrics["transition_training/teacher_mix"] = float(self.teacher_mix)

        directional_loss = None
        if self.directional_weight > 0.0 and intermediate is not None:
            flattened_proj = intermediate.flatten(1)
            flattened_target = target.detach().flatten(1)
            cosine = F.cosine_similarity(flattened_proj, flattened_target, dim=1).clamp(-1.0, 1.0)
            directional_loss = 1.0 - cosine

        return target, weights, metrics, directional_loss

    def apply_adaptive_weighting(
        self, weights: Tensor, per_sample_loss: Optional[Tensor]
    ) -> Tensor:
        if per_sample_loss is None:
            return weights
        adaptive = self.scheduler.adaptive_rescale(per_sample_loss)
        return weights * adaptive.to(weights)

    def lora_modulation_context(
        self, network, primary_timesteps: Tensor, reference_timesteps: Tensor
    ):
        if not getattr(self.config, "enabled", False):
            return nullcontext()
        delta = (primary_timesteps - reference_timesteps) / 1000.0
        delta = delta.clamp(0.0, 1.0)
        return self.lora_adapter.maybe_modulate(network, delta)
