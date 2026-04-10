from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SpatiotemporalGuidanceResult:
    scaled_loss: Optional[torch.Tensor]
    raw_loss: Optional[torch.Tensor]
    anchor_loss: Optional[torch.Tensor]
    temporal_loss: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]
    anchor_weight: Optional[torch.Tensor]
    temporal_weight: Optional[torch.Tensor]


def _elementwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction="none")
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction="none")
    if loss_type in {"smooth_l1", "sml1"}:
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unsupported spatiotemporal guidance loss type: {loss_type}")


def compute_spatiotemporal_guidance_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
    anchor_weight: float,
    temporal_weight: float,
    loss_type: str,
) -> SpatiotemporalGuidanceResult:
    """Compute URSA-inspired anchor and temporal-delta guidance loss."""
    if model_pred.dim() != 5 or target.dim() != 5:
        return SpatiotemporalGuidanceResult(
            scaled_loss=None,
            raw_loss=None,
            anchor_loss=None,
            temporal_loss=None,
            scale=None,
            anchor_weight=None,
            temporal_weight=None,
        )

    anchor_loss = _elementwise_loss(
        model_pred[:, :, :1, :, :],
        target[:, :, :1, :, :],
        loss_type,
    ).mean()

    temporal_loss: Optional[torch.Tensor] = None
    used_temporal_weight = 0.0
    if model_pred.shape[2] > 1:
        pred_delta = model_pred[:, :, 1:, :, :] - model_pred[:, :, :-1, :, :]
        target_delta = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        temporal_loss = _elementwise_loss(
            pred_delta,
            target_delta,
            loss_type,
        ).mean()
        used_temporal_weight = max(0.0, float(temporal_weight))

    used_anchor_weight = max(0.0, float(anchor_weight))
    if used_anchor_weight <= 0.0 and used_temporal_weight <= 0.0:
        used_anchor_weight = 1.0

    denom = used_anchor_weight + used_temporal_weight
    raw_loss = anchor_loss * used_anchor_weight
    if temporal_loss is not None and used_temporal_weight > 0.0:
        raw_loss = raw_loss + (temporal_loss * used_temporal_weight)
    raw_loss = raw_loss / max(denom, 1e-8)
    scaled_loss = raw_loss * float(scale)

    return SpatiotemporalGuidanceResult(
        scaled_loss=scaled_loss,
        raw_loss=raw_loss.detach(),
        anchor_loss=anchor_loss.detach(),
        temporal_loss=temporal_loss.detach() if temporal_loss is not None else None,
        scale=torch.tensor(float(scale), device=model_pred.device, dtype=torch.float32),
        anchor_weight=torch.tensor(
            float(used_anchor_weight),
            device=model_pred.device,
            dtype=torch.float32,
        ),
        temporal_weight=torch.tensor(
            float(used_temporal_weight),
            device=model_pred.device,
            dtype=torch.float32,
        ),
    )
