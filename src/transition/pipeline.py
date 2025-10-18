"""Convenience helpers for transition training pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .manager import TransitionTrainingManager, PreparedBatch


def prepare_transition_batch(
    manager: Optional[TransitionTrainingManager],
    accelerator,
    latents: Tensor,
    noise: Tensor,
) -> Optional[PreparedBatch]:
    """Return prepared batch from manager when transition training is enabled."""
    if manager is None:
        return None
    return manager.prepare_batch(
        latents=latents,
        noise=noise,
        batch_size=latents.shape[0],
        device=accelerator.device,
    )


def finalize_batch_if_enabled(
    manager: Optional[TransitionTrainingManager],
    accelerator,
    model_pred: Tensor,
    forward_fn,
    transformer,
    need_intermediate: bool,
    intermediate_z: Optional[Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor], Dict[str, float], Optional[Tensor], Optional[Tensor]]:
    """Return target, weights, metrics, directional loss, per-sample loss if enabled."""
    if manager is None:
        return None, None, {}, None, None

    target, weights, metrics, directional_loss = manager.finalize_batch(
        model_pred=model_pred,
        forward_fn=forward_fn,
        transformer=transformer,
        intermediate=intermediate_z if need_intermediate else None,
    )

    per_sample_loss = (
        (model_pred.detach().to(torch.float32) - target.detach().to(torch.float32))
        .view(model_pred.shape[0], -1)
        .mean(dim=1)
    )

    adjusted_weights = manager.apply_adaptive_weighting(weights, per_sample_loss)

    return (
        target.to(device=accelerator.device, dtype=model_pred.dtype),
        adjusted_weights.to(device=accelerator.device, dtype=model_pred.dtype),
        metrics,
        directional_loss,
        per_sample_loss,
    )


def update_teacher_if_needed(
    manager: Optional[TransitionTrainingManager],
    accelerator,
    transformer,
) -> None:
    if manager is None:
        return
    try:
        manager.update_teacher(accelerator.unwrap_model(transformer))
    except Exception:
        pass
