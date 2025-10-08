"""
Video-specific reward models for SRPO training.

This module implements video-aware reward metrics beyond single-frame image quality:
1. Temporal Consistency - Frame-to-frame similarity to penalize flickering
2. Optical Flow - Motion smoothness and quality
3. Motion Quality - Overall motion coherence

These complement standard image rewards (HPS, PickScore, Aesthetic) for better
video generation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalConsistencyReward(nn.Module):
    """
    Reward based on frame-to-frame consistency.

    Penalizes large differences between consecutive frames (flickering/instability).
    Higher reward = more temporally stable video.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute_reward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency reward from video frames.

        Args:
            frames: Tensor of shape [B, C, F, H, W] or [B, F, C, H, W]

        Returns:
            Reward tensor [B] - higher is more consistent
        """
        # Ensure shape is [B, C, F, H, W]
        if frames.dim() == 5:
            if frames.shape[2] > frames.shape[1]:  # [B, C, F, H, W]
                pass
            else:  # [B, F, C, H, W]
                frames = frames.permute(0, 2, 1, 3, 4)

        B, C, F, H, W = frames.shape

        if F < 2:
            # Can't compute temporal consistency with single frame
            return torch.ones(B, device=self.device, dtype=self.dtype)

        # Compute pairwise frame differences
        frame_diffs = []
        for i in range(F - 1):
            frame_current = frames[:, :, i, :, :]
            frame_next = frames[:, :, i + 1, :, :]

            # L2 distance between consecutive frames
            diff = torch.norm(frame_current - frame_next, p=2, dim=(1, 2, 3))
            frame_diffs.append(diff)

        # Stack all differences [B, F-1]
        frame_diffs = torch.stack(frame_diffs, dim=1)

        # Reward is inverse of average frame difference
        # Normalize by image size for scale invariance
        avg_diff = frame_diffs.mean(dim=1) / (C * H * W) ** 0.5

        # Convert to reward (lower diff = higher reward)
        # Use exponential to make it positive and bounded
        reward = torch.exp(-avg_diff * 10.0)  # Scale factor controls sensitivity

        return reward


class OpticalFlowReward(nn.Module):
    """
    Reward based on optical flow smoothness.

    Measures motion quality by analyzing optical flow between frames.
    Smooth, coherent flow = high reward. Chaotic/discontinuous flow = low reward.

    Note: This is a simplified version. For production, consider using
    pre-trained optical flow models like RAFT or FlowNet.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute_reward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow-based reward.

        Args:
            frames: Tensor of shape [B, C, F, H, W]

        Returns:
            Reward tensor [B]
        """
        # Ensure shape is [B, C, F, H, W]
        if frames.dim() == 5:
            if frames.shape[2] > frames.shape[1]:
                pass
            else:
                frames = frames.permute(0, 2, 1, 3, 4)

        B, C, F, H, W = frames.shape

        if F < 2:
            return torch.ones(B, device=self.device, dtype=self.dtype)

        # Compute gradient-based motion estimation (simplified optical flow)
        motion_scores = []

        for i in range(F - 1):
            frame_current = frames[:, :, i, :, :]
            frame_next = frames[:, :, i + 1, :, :]

            # Compute spatial gradients
            grad_x_curr = frame_current[:, :, :, 1:] - frame_current[:, :, :, :-1]
            grad_y_curr = frame_current[:, :, 1:, :] - frame_current[:, :, :-1, :]

            grad_x_next = frame_next[:, :, :, 1:] - frame_next[:, :, :, :-1]
            grad_y_next = frame_next[:, :, 1:, :] - frame_next[:, :, :-1, :]

            # Motion as gradient difference
            motion_x = torch.abs(grad_x_next - grad_x_curr)
            motion_y = torch.abs(grad_y_next - grad_y_curr)

            # Smoothness: variance of motion across spatial locations
            # Lower variance = smoother, more coherent motion
            motion_variance = motion_x.var(dim=(1, 2, 3)) + motion_y.var(dim=(1, 2, 3))
            motion_scores.append(motion_variance)

        # Average across all frame pairs
        avg_motion_variance = torch.stack(motion_scores, dim=1).mean(dim=1)

        # Reward is inverse of variance (smoother = higher reward)
        reward = torch.exp(-avg_motion_variance * 5.0)

        return reward


class MotionQualityReward(nn.Module):
    """
    Overall motion quality reward combining multiple factors.

    Evaluates:
    - Motion magnitude (not too static, not too chaotic)
    - Motion distribution (balanced across frame)
    - Temporal smoothness
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute_reward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute motion quality reward.

        Args:
            frames: Tensor of shape [B, C, F, H, W]

        Returns:
            Reward tensor [B]
        """
        # Ensure shape is [B, C, F, H, W]
        if frames.dim() == 5:
            if frames.shape[2] > frames.shape[1]:
                pass
            else:
                frames = frames.permute(0, 2, 1, 3, 4)

        B, C, F, H, W = frames.shape

        if F < 3:
            return torch.ones(B, device=self.device, dtype=self.dtype)

        # Compute frame differences
        frame_diffs = []
        for i in range(F - 1):
            diff = frames[:, :, i + 1, :, :] - frames[:, :, i, :, :]
            frame_diffs.append(diff)

        frame_diffs = torch.stack(frame_diffs, dim=2)  # [B, C, F-1, H, W]

        # 1. Motion magnitude: penalize too static or too chaotic
        motion_magnitude = torch.norm(frame_diffs, p=2, dim=(1, 2, 3, 4))
        normalized_magnitude = motion_magnitude / (C * (F - 1) * H * W) ** 0.5

        # Optimal motion is in middle range (not 0, not too large)
        target_motion = 0.1  # Target normalized motion
        magnitude_score = torch.exp(
            -((normalized_magnitude - target_motion) ** 2) / 0.01
        )

        # 2. Motion smoothness across time
        motion_smoothness = frame_diffs.var(dim=2).mean(dim=(1, 2, 3))
        smoothness_score = torch.exp(-motion_smoothness * 10.0)

        # 3. Spatial distribution of motion
        spatial_motion = frame_diffs.mean(dim=2).abs()  # [B, C, H, W]
        motion_uniformity = spatial_motion.std(dim=(1, 2, 3))
        uniformity_score = torch.exp(-motion_uniformity * 5.0)

        # Combine scores
        reward = (magnitude_score + smoothness_score + uniformity_score) / 3.0

        return reward


class VideoRewardAggregator(nn.Module):
    """
    Aggregates multiple video-specific rewards.

    Combines temporal consistency, optical flow, and motion quality
    with configurable weights.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        temporal_consistency_weight: float = 0.0,
        optical_flow_weight: float = 0.0,
        motion_quality_weight: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Initialize reward modules if weights > 0
        self.temporal_consistency_weight = temporal_consistency_weight
        self.optical_flow_weight = optical_flow_weight
        self.motion_quality_weight = motion_quality_weight

        if temporal_consistency_weight > 0:
            self.temporal_consistency = TemporalConsistencyReward(device, dtype)

        if optical_flow_weight > 0:
            self.optical_flow = OpticalFlowReward(device, dtype)

        if motion_quality_weight > 0:
            self.motion_quality = MotionQualityReward(device, dtype)

    def compute_reward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated video reward.

        Args:
            frames: Tensor of shape [B, C, F, H, W]

        Returns:
            Reward tensor [B]
        """
        total_reward = torch.zeros(
            frames.shape[0], device=self.device, dtype=self.dtype
        )
        total_weight = 0.0

        if self.temporal_consistency_weight > 0:
            reward = self.temporal_consistency.compute_reward(frames)
            total_reward += self.temporal_consistency_weight * reward
            total_weight += self.temporal_consistency_weight

        if self.optical_flow_weight > 0:
            reward = self.optical_flow.compute_reward(frames)
            total_reward += self.optical_flow_weight * reward
            total_weight += self.optical_flow_weight

        if self.motion_quality_weight > 0:
            reward = self.motion_quality.compute_reward(frames)
            total_reward += self.motion_quality_weight * reward
            total_weight += self.motion_quality_weight

        # Normalize by total weight if any rewards are active
        if total_weight > 0:
            total_reward = total_reward / total_weight

        return total_reward
