from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SplitVideoLossResult:
    reduced_loss: torch.Tensor
    anchor_loss: Optional[torch.Tensor]
    temporal_loss: Optional[torch.Tensor]
    delta: Optional[torch.Tensor]
    anchor_weight: Optional[torch.Tensor]
    temporal_weight: Optional[torch.Tensor]


def reduce_split_video_loss(
    loss_tensor: torch.Tensor,
    anchor_weight: float,
    temporal_weight: float,
) -> SplitVideoLossResult:
    """Reduce a video loss tensor with separate anchor and continuation weights.

    The input is expected to be shaped like (B, C, F, H, W). When both weights are
    set to 1.0, this reduction matches the standard full-tensor mean.
    """
    if loss_tensor.dim() != 5:
        reduced = loss_tensor.mean()
        return SplitVideoLossResult(
            reduced_loss=reduced,
            anchor_loss=None,
            temporal_loss=None,
            delta=None,
            anchor_weight=None,
            temporal_weight=None,
        )

    frame_count = int(loss_tensor.shape[2])
    if frame_count <= 1:
        reduced = loss_tensor.mean()
        anchor_loss = loss_tensor.mean().detach()
        return SplitVideoLossResult(
            reduced_loss=reduced,
            anchor_loss=anchor_loss,
            temporal_loss=None,
            delta=None,
            anchor_weight=torch.tensor(
                float(anchor_weight),
                device=loss_tensor.device,
                dtype=torch.float32,
            ),
            temporal_weight=None,
        )

    anchor_loss = loss_tensor[:, :, :1, :, :].mean(dim=(1, 2, 3, 4))
    temporal_loss = loss_tensor[:, :, 1:, :, :].mean(dim=(1, 2, 3, 4))

    anchor_frames = 1.0
    temporal_frames = float(frame_count - 1)
    denom = (anchor_weight * anchor_frames) + (temporal_weight * temporal_frames)
    if denom <= 0.0:
        raise ValueError("split video loss weights must produce a positive denominator")

    reduced_per_sample = (
        (anchor_loss * (anchor_weight * anchor_frames))
        + (temporal_loss * (temporal_weight * temporal_frames))
    ) / denom
    reduced_loss = reduced_per_sample.mean()

    anchor_scalar = anchor_loss.detach().mean()
    temporal_scalar = temporal_loss.detach().mean()
    return SplitVideoLossResult(
        reduced_loss=reduced_loss,
        anchor_loss=anchor_scalar,
        temporal_loss=temporal_scalar,
        delta=temporal_scalar - anchor_scalar,
        anchor_weight=torch.tensor(
            float(anchor_weight),
            device=loss_tensor.device,
            dtype=torch.float32,
        ),
        temporal_weight=torch.tensor(
            float(temporal_weight),
            device=loss_tensor.device,
            dtype=torch.float32,
        ),
    )
