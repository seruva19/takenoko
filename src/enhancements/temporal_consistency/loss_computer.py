import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from .config import TemporalConsistencyConfig
from .frequency_analyzer import FrequencyAnalyzer

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TemporalConsistencyLossComputer:
    """Computes frequency-domain temporal consistency losses for video training."""

    def __init__(self, config: TemporalConsistencyConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cpu")

        # Initialize frequency analyzer
        self.freq_analyzer = FrequencyAnalyzer(
            device=self.device, enable_caching=self.config.freq_temporal_enable_caching
        )

        # Performance tracking
        self.total_computations = 0
        self.total_computation_time = 0.0

        logger.info(
            f"TemporalConsistencyLossComputer initialized with config: {self.config}"
        )

    def is_video_batch(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is a video batch [B, F, C, H, W]."""
        return (
            len(tensor.shape) == 5
            and tensor.shape[1] >= self.config.freq_temporal_min_frames
        )

    def compute_structural_consistency_loss(
        self, video_tensor: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Compute structural consistency loss using low-frequency components.

        This implements the core insight from Ouroboros-Diffusion: low-frequency
        components preserve structural consistency across frames.

        Args:
            video_tensor: Video batch [B, F, C, H, W]
            step: Current training step

        Returns:
            Structural consistency loss tensor
        """
        if not self.config.enable_frequency_domain_temporal_consistency:
            return torch.tensor(0.0, device=video_tensor.device)

        batch_size, num_frames, channels, height, width = video_tensor.shape

        if num_frames < 2:
            return torch.tensor(0.0, device=video_tensor.device)

        # Memory safety: limit frames processed per batch
        max_frames = min(num_frames, self.config.freq_temporal_max_frames_per_batch)
        if max_frames < num_frames:
            logger.debug(
                f"Limiting frames from {num_frames} to {max_frames} for memory safety"
            )

        # Get temporal weights based on frame distance
        temporal_weights = self.config.get_temporal_weights(max_frames)
        max_distance = min(self.config.freq_temporal_max_distance, max_frames - 1)

        total_loss = torch.tensor(0.0, device=video_tensor.device)
        loss_count = 0

        # Compute consistency loss between frame pairs
        for b in range(batch_size):
            for f in range(max_frames - 1):
                # Only process up to max_temporal_distance
                if f >= max_distance:
                    break

                current_frame = video_tensor[b, f]  # [C, H, W]
                next_frame = video_tensor[b, f + 1]  # [C, H, W]

                # Extract structural (low-frequency) components
                current_structure = self.freq_analyzer.extract_structural_component(
                    current_frame, threshold=self.config.freq_temporal_low_threshold
                )
                next_structure = self.freq_analyzer.extract_structural_component(
                    next_frame, threshold=self.config.freq_temporal_low_threshold
                )

                # Compute consistency loss between structural components
                frame_consistency_loss = F.mse_loss(
                    current_structure, next_structure, reduction="mean"
                )

                # Apply temporal weighting (closer frames weighted more)
                weight = (
                    temporal_weights[f]
                    if f < len(temporal_weights)
                    else temporal_weights[-1]
                )
                weighted_loss = weight * frame_consistency_loss

                total_loss += weighted_loss
                loss_count += 1

        # Average over all frame pairs and batches
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, device=video_tensor.device)

        return avg_loss

    def compute_motion_coherence_loss(
        self, video_tensor: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Compute motion coherence loss using frequency domain motion analysis.

        This ensures that motion patterns (high-frequency changes) are coherent
        while preserving detail.

        Args:
            video_tensor: Video batch [B, F, C, H, W]
            step: Current training step

        Returns:
            Motion coherence loss tensor
        """
        if not self.config.freq_temporal_enable_motion_coherence:
            return torch.tensor(0.0, device=video_tensor.device)

        batch_size, num_frames, channels, height, width = video_tensor.shape

        if num_frames < 3:  # Need at least 3 frames to analyze motion coherence
            return torch.tensor(0.0, device=video_tensor.device)

        total_loss = torch.tensor(0.0, device=video_tensor.device)
        loss_count = 0

        # Analyze motion coherence across consecutive frame triplets
        # Use softer motion coherence that allows natural acceleration/deceleration
        for b in range(batch_size):
            for f in range(num_frames - 2):
                frame1 = video_tensor[b, f]
                frame2 = video_tensor[b, f + 1]
                frame3 = video_tensor[b, f + 2]

                # Compute motion vectors in frequency domain
                motion1_2 = self.freq_analyzer.compute_temporal_frequency_difference(
                    frame1, frame2, component="high"
                )
                motion2_3 = self.freq_analyzer.compute_temporal_frequency_difference(
                    frame2, frame3, component="high"
                )

                # Softer motion coherence: allow some variation but penalize extreme changes
                motion_diff = motion2_3 - motion1_2
                # Use L1 loss for softer constraint and apply threshold
                motion_magnitude = torch.mean(torch.abs(motion_diff))
                # Only penalize if motion change exceeds configurable threshold
                motion_coherence_loss = F.relu(
                    motion_magnitude - self.config.freq_temporal_motion_threshold
                )

                # Apply temporal decay weighting (closer frames weighted more)
                weight = self.config.freq_temporal_decay_factor ** (
                    f
                )  # Fixed: closer frames higher weight
                total_loss += weight * motion_coherence_loss
                loss_count += 1

        # Average over all motion triplets
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, device=video_tensor.device)

        return avg_loss

    def compute_frequency_temporal_loss(
        self, pred_video: torch.Tensor, target_video: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Compute frequency-domain temporal loss between prediction and target.

        This ensures that the model learns to predict temporally consistent
        frequency components across frames by comparing actual frequency components.

        Args:
            pred_video: Predicted video [B, F, C, H, W]
            target_video: Target video [B, F, C, H, W]
            step: Current training step

        Returns:
            Frequency-domain temporal loss
        """
        if not self.config.freq_temporal_enable_prediction_loss:
            return torch.tensor(0.0, device=pred_video.device)

        batch_size, num_frames, channels, height, width = pred_video.shape

        if num_frames < 2:
            return torch.tensor(0.0, device=pred_video.device)

        total_loss = torch.tensor(0.0, device=pred_video.device)
        loss_count = 0

        # Compare frequency components directly between prediction and target
        for b in range(batch_size):
            for f in range(num_frames):
                pred_frame = pred_video[b, f]
                target_frame = target_video[b, f]

                # Extract low-frequency components (structural)
                pred_low_freq = self.freq_analyzer.extract_structural_component(
                    pred_frame, threshold=self.config.freq_temporal_low_threshold
                )
                target_low_freq = self.freq_analyzer.extract_structural_component(
                    target_frame, threshold=self.config.freq_temporal_low_threshold
                )

                # Compare frequency components directly
                freq_loss = F.mse_loss(pred_low_freq, target_low_freq, reduction="mean")
                total_loss += freq_loss
                loss_count += 1

        # Average over all frames
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, device=pred_video.device)

        return avg_loss

    def compute_total_temporal_consistency_loss(
        self, pred_video: torch.Tensor, target_video: Optional[torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total temporal consistency loss with all components.

        Args:
            pred_video: Predicted video batch [B, F, C, H, W]
            target_video: Target video batch [B, F, C, H, W] (optional)
            step: Current training step

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if not self.config.should_apply_at_step(step):
            return torch.tensor(0.0, device=pred_video.device), {}

        if not self.is_video_batch(pred_video):
            return torch.tensor(0.0, device=pred_video.device), {}

        import time

        start_time = time.time()

        # Compute loss weight multiplier (includes warmup)
        weight_multiplier = self.config.get_loss_weight_multiplier(step)

        if weight_multiplier == 0.0:
            return torch.tensor(0.0, device=pred_video.device), {}

        # Compute individual loss components
        structural_loss = self.compute_structural_consistency_loss(pred_video, step)
        motion_loss = self.compute_motion_coherence_loss(pred_video, step)

        # Compute frequency temporal loss if target is available
        frequency_temporal_loss = torch.tensor(0.0, device=pred_video.device)
        if (
            target_video is not None
            and self.config.freq_temporal_enable_prediction_loss
        ):
            frequency_temporal_loss = self.compute_frequency_temporal_loss(
                pred_video, target_video, step
            )

        # Combine losses with configured weights
        total_loss = (
            self.config.freq_temporal_consistency_weight * structural_loss
            + self.config.freq_temporal_motion_weight * motion_loss
            + self.config.freq_temporal_prediction_weight * frequency_temporal_loss
        )

        # Apply weight multiplier (warmup/scheduling)
        total_loss = weight_multiplier * total_loss

        # Update performance tracking
        self.total_computations += 1
        self.total_computation_time += time.time() - start_time

        # Prepare loss components for logging
        loss_components = {
            "structural_loss": structural_loss.item(),
            "motion_coherence_loss": motion_loss.item(),
            "frequency_temporal_loss": frequency_temporal_loss.item(),
            "total_temporal_loss": total_loss.item(),
            "weight_multiplier": weight_multiplier,
        }

        return total_loss, loss_components

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        avg_time = (
            self.total_computation_time / max(self.total_computations, 1)
            if self.total_computations > 0
            else 0.0
        )

        stats = {
            "total_computations": self.total_computations,
            "total_computation_time": self.total_computation_time,
            "average_computation_time_ms": avg_time * 1000,
            "config_enabled_features": {
                "temporal_consistency": self.config.enable_frequency_domain_temporal_consistency,
                "motion_coherence_loss": self.config.freq_temporal_enable_motion_coherence,
                "frequency_temporal_loss": self.config.freq_temporal_enable_prediction_loss,
            },
            "frequency_analyzer_cache": self.freq_analyzer.get_cache_stats(),
        }

        return stats

    def cleanup(self):
        """Cleanup resources and caches."""
        self.freq_analyzer.clear_cache()

        logger.info(
            f"TemporalConsistencyLossComputer cleanup completed. "
            f"Processed {self.total_computations} computations in "
            f"{self.total_computation_time:.2f}s"
        )
