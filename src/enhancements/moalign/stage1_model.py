"""Stage-1 MOALIGN teacher components.

This module contains the motion projector and flow predictor used to learn a
motion-centric subspace from frozen encoder tokens.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interpolate_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token sequence length to target token count."""
    bf, num_tokens, dim = tokens.shape
    if num_tokens == target_tokens:
        return tokens
    src_side = int(math.isqrt(num_tokens))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == num_tokens and tgt_side * tgt_side == target_tokens:
        x = tokens.permute(0, 2, 1).reshape(bf, dim, src_side, src_side)
        x = F.interpolate(
            x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False
        )
        return x.reshape(bf, dim, target_tokens).permute(0, 2, 1)
    x = tokens.permute(0, 2, 1)
    x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


class Stage1MotionProjector(nn.Module):
    """Project encoder tokens into a motion-centric subspace."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        self.temporal_conv = nn.Conv3d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.pointwise_conv = nn.Conv3d(
            self.hidden_dim, self.output_dim, kernel_size=(1, 1, 1)
        )
        self.act = nn.SiLU()

    def forward_tokens(
        self, encoder_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from encoder tokens.

        Args:
            encoder_tokens: [B, F, N, D]
        Returns:
            motion_tokens: [B, F, N', C]
            motion_maps: [B, C, F, H, W]
        """
        if encoder_tokens.dim() != 4:
            raise ValueError(
                f"encoder_tokens must be [B,F,N,D], got {tuple(encoder_tokens.shape)}"
            )
        bsz, frames, num_tokens, dim = encoder_tokens.shape
        if dim != self.input_dim:
            raise ValueError(
                f"encoder token dim mismatch: expected {self.input_dim}, got {dim}"
            )
        if frames < 2:
            raise ValueError("Stage-1 MOALIGN requires at least 2 frames per sample")

        side = max(1, int(round(math.sqrt(num_tokens))))
        grid_tokens = side * side
        tokens_2d = encoder_tokens.reshape(bsz * frames, num_tokens, dim)
        if grid_tokens != num_tokens:
            tokens_2d = _interpolate_tokens(tokens_2d, grid_tokens)

        x = tokens_2d.permute(0, 2, 1).reshape(bsz, frames, dim, side, side)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, D, F, H, W]
        x = self.act(self.temporal_conv(x))
        x = self.act(self.pointwise_conv(x))

        motion_tokens = x.permute(0, 2, 3, 4, 1).reshape(
            bsz, frames, grid_tokens, self.output_dim
        )
        return motion_tokens, x


class Stage1FlowPredictor(nn.Module):
    """Predict optical flow from motion-centric maps."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.conv1 = nn.Conv3d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(self.hidden_dim, 2, kernel_size=1)

    def forward(
        self,
        motion_maps: torch.Tensor,
        target_height: int,
        target_width: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Predict flow in [B, F-1, 2, H, W]."""
        if motion_maps.dim() != 5:
            raise ValueError(
                f"motion_maps must be [B,C,F,H,W], got {tuple(motion_maps.shape)}"
            )
        if motion_maps.shape[2] < 2:
            raise ValueError("Flow prediction requires at least 2 frames")

        # Motion is represented as temporal change of projected features.
        delta = motion_maps[:, :, 1:] - motion_maps[:, :, :-1]
        x = F.relu(self.conv1(delta))
        x = self.conv2(x)  # [B, 2, F-1, h, w]
        if (
            x.shape[2] != target_frames
            or x.shape[3] != target_height
            or x.shape[4] != target_width
        ):
            x = F.interpolate(
                x,
                size=(target_frames, target_height, target_width),
                mode="trilinear",
                align_corners=False,
            )
        return x.permute(0, 2, 1, 3, 4).contiguous()
