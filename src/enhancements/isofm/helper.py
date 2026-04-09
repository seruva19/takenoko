"""Isokinetic Flow Matching helper for train-time straightening regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from common.logger import get_logger
from enhancements.blockwise_flow_matching.segment_utils import normalize_timesteps

logger = get_logger(__name__)


@dataclass
class IsoFMResult:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class IsokineticFlowMatchingHelper:
    """Compute the Jacobian-free Iso-FM auxiliary loss with one lookahead forward."""

    def __init__(self, args: Any) -> None:
        self.enabled = bool(getattr(args, "enable_isofm", False))
        self.weight = float(getattr(args, "isofm_lambda", 4.0))
        self.time_weight_exponent = float(
            getattr(args, "isofm_time_weight_exponent", 2.0)
        )
        self.lookahead_eps_min = float(getattr(args, "isofm_lookahead_eps_min", 0.02))
        self.lookahead_eps_max = float(getattr(args, "isofm_lookahead_eps_max", 0.05))
        self.min_t = float(getattr(args, "isofm_min_t", 0.0))
        self.max_t = float(getattr(args, "isofm_max_t", 0.98))
        self.apply_prob = float(getattr(args, "isofm_apply_prob", 1.0))
        self.warmup_steps = int(getattr(args, "isofm_warmup_steps", 0))
        self.normalize_by_speed = bool(
            getattr(args, "isofm_normalize_by_speed", True)
        )
        self.speed_epsilon = float(getattr(args, "isofm_speed_epsilon", 1e-6))
        self.log_interval = int(getattr(args, "isofm_log_interval", 100))
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def setup_hooks(self) -> None:
        """Reserved for interface consistency with other helpers."""

    def remove_hooks(self) -> None:
        """Reserved for interface consistency with other helpers."""

    @staticmethod
    def _resolve_sample_values(timesteps: torch.Tensor) -> torch.Tensor:
        values = normalize_timesteps(timesteps.detach())
        if values.ndim <= 1:
            return values.view(-1)
        return values.reshape(values.shape[0], -1).mean(dim=1)

    @staticmethod
    def _expand_batch_values(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim <= 1:
            return values
        view_shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
        return values.view(view_shape)

    @staticmethod
    def _to_native_timestep_scale(
        normalized_timesteps: torch.Tensor,
        reference_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if reference_timesteps.numel() == 0:
            return normalized_timesteps.to(
                device=reference_timesteps.device,
                dtype=reference_timesteps.dtype,
            )
        if reference_timesteps.detach().float().max() > 1.0:
            native = 1.0 + normalized_timesteps * 1000.0
        else:
            native = normalized_timesteps
        return native.to(
            device=reference_timesteps.device,
            dtype=reference_timesteps.dtype,
        )

    def _sample_eps(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        if self.lookahead_eps_min == self.lookahead_eps_max:
            return torch.full(
                (batch_size,),
                self.lookahead_eps_min,
                device=device,
                dtype=torch.float32,
            )
        return torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(
            self.lookahead_eps_min,
            self.lookahead_eps_max,
        )

    def compute_loss(
        self,
        *,
        model_pred: torch.Tensor,
        noisy_model_input: torch.Tensor,
        model_timesteps: torch.Tensor,
        predict_lookahead: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        global_step: Optional[int] = None,
    ) -> Optional[IsoFMResult]:
        if not self.enabled or self.weight <= 0.0:
            return None
        if global_step is not None and global_step < self.warmup_steps:
            return None
        if model_pred.shape != noisy_model_input.shape:
            logger.debug(
                "IsoFM skipped because model_pred shape %s does not match noisy input shape %s.",
                tuple(model_pred.shape),
                tuple(noisy_model_input.shape),
            )
            return None

        sample_t = self._resolve_sample_values(model_timesteps).to(
            device=model_pred.device,
            dtype=torch.float32,
        )
        batch_size = int(sample_t.shape[0])
        sampled_eps = self._sample_eps(batch_size, device=model_pred.device)
        effective_eps = torch.minimum(sampled_eps, (1.0 - sample_t).clamp_min(0.0))

        active_mask = (sample_t >= self.min_t) & (sample_t <= self.max_t)
        active_mask = active_mask & (effective_eps > 0.0)
        if self.apply_prob < 1.0:
            prob_mask = (
                torch.rand(batch_size, device=model_pred.device) <= self.apply_prob
            )
            active_mask = active_mask & prob_mask
        if not bool(active_mask.any().item()):
            return None

        eps_for_input = self._expand_batch_values(effective_eps, noisy_model_input).to(
            dtype=noisy_model_input.dtype
        )
        lookahead_noisy_input = noisy_model_input + eps_for_input * model_pred.detach().to(
            dtype=noisy_model_input.dtype
        )

        eps_for_timestep = self._expand_batch_values(effective_eps, model_timesteps)
        lookahead_t_normalized = (
            normalize_timesteps(model_timesteps.detach()).to(
                device=model_pred.device,
                dtype=torch.float32,
            )
            + eps_for_timestep.to(device=model_pred.device, dtype=torch.float32)
        ).clamp(0.0, 1.0)
        lookahead_timesteps = self._to_native_timestep_scale(
            lookahead_t_normalized,
            model_timesteps,
        )

        with torch.no_grad():
            next_pred = predict_lookahead(lookahead_noisy_input, lookahead_timesteps)
        if next_pred.shape != model_pred.shape:
            logger.warning(
                "IsoFM lookahead prediction shape mismatch: expected %s, got %s.",
                tuple(model_pred.shape),
                tuple(next_pred.shape),
            )
            return None

        delta = model_pred.to(dtype=torch.float32) - next_pred.detach().to(
            device=model_pred.device,
            dtype=torch.float32,
        )
        if self.normalize_by_speed:
            speed = model_pred.detach().to(dtype=torch.float32).flatten(1).norm(dim=1)
            speed = speed.clamp_min(self.speed_epsilon)
            delta = delta / speed.view((-1,) + (1,) * (delta.ndim - 1))
        else:
            speed = torch.ones(batch_size, device=model_pred.device, dtype=torch.float32)

        per_sample_mse = delta.flatten(1).pow(2).mean(dim=1)
        time_weight = ((1.0 - sample_t).clamp_min(0.0) ** self.time_weight_exponent) / (
            effective_eps + self.speed_epsilon
        )
        weighted_loss = per_sample_mse * time_weight
        loss = weighted_loss[active_mask].mean()

        active_float = active_mask.float()
        self.last_metrics = {
            "isofm/weight_mean": time_weight[active_mask].mean().detach(),
            "isofm/eps_mean": effective_eps[active_mask].mean().detach(),
            "isofm/speed_mean": speed[active_mask].mean().detach(),
            "isofm/active_ratio": active_float.mean().detach(),
            "isofm/timestep_mean": sample_t[active_mask].mean().detach(),
        }
        return IsoFMResult(loss=loss, metrics=dict(self.last_metrics))
