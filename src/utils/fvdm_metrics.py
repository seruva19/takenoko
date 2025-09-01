"""
FVDM-specific training metrics and evaluation tools.

This module provides metrics to evaluate the effectiveness of FVDM training,
including temporal consistency, frame diversity, and async/sync sampling statistics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FVDMTrainingMetrics:
    """Metrics specifically for FVDM training evaluation and monitoring."""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        """
        Initialize FVDM training metrics.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.async_steps = 0
        self.sync_steps = 0
        self.temporal_consistency_losses = []
        self.frame_diversity_scores = []
        self.ptss_probabilities = []
        self.timestep_variance_scores = []
    
    def compute_temporal_consistency_loss(
        self, 
        frames: torch.Tensor, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Measure temporal consistency between adjacent frames.
        Lower values indicate better temporal consistency.
        
        Args:
            frames: [B, C, F, H, W] video tensor
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            Temporal consistency loss value
        """
        if frames.size(2) < 2:  # Need at least 2 frames
            return torch.tensor(0.0, device=frames.device)
            
        # Compute frame-to-frame differences
        frame_diff = frames[:, :, 1:] - frames[:, :, :-1]
        
        # MSE of differences (lower = more consistent)
        if reduction == 'none':
            consistency_loss = F.mse_loss(
                frame_diff, 
                torch.zeros_like(frame_diff), 
                reduction='none'
            )
        else:
            consistency_loss = F.mse_loss(
                frame_diff, 
                torch.zeros_like(frame_diff), 
                reduction=reduction
            )
        
        return consistency_loss
    
    def compute_frame_diversity_score(
        self, 
        frames: torch.Tensor,
        sample_pairs: int = 10
    ) -> torch.Tensor:
        """
        Measure diversity across frames to avoid temporal collapse.
        Higher scores indicate more diverse frames (good for avoiding mode collapse).
        
        Args:
            frames: [B, C, F, H, W] video tensor
            sample_pairs: Number of frame pairs to sample for efficiency
            
        Returns:
            Frame diversity score (higher = more diverse)
        """
        B, C, F, H, W = frames.shape
        
        if F < 2:
            return torch.tensor(0.0, device=frames.device)
        
        # Flatten spatial dimensions for easier computation
        frames_flat = frames.view(B, C * H * W, F)
        
        # Sample frame pairs to avoid O(F^2) computation
        max_pairs = F * (F - 1) // 2
        num_pairs = min(sample_pairs, max_pairs)
        
        # Compute pairwise distances between frames
        distances = []
        pairs_sampled = 0
        
        for i in range(F):
            for j in range(i + 1, F):
                if pairs_sampled >= num_pairs:
                    break
                    
                # MSE distance between frames
                dist = F.mse_loss(
                    frames_flat[:, :, i], 
                    frames_flat[:, :, j], 
                    reduction='mean'
                )
                distances.append(dist)
                pairs_sampled += 1
            
            if pairs_sampled >= num_pairs:
                break
        
        if not distances:
            return torch.tensor(0.0, device=frames.device)
            
        # Average distance across all sampled pairs
        diversity_score = torch.stack(distances).mean()
        return diversity_score
    
    def compute_timestep_variance(
        self, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure variance in timesteps across frames.
        Higher variance indicates more asynchronous sampling.
        
        Args:
            timesteps: [B, F] timestep tensor
            
        Returns:
            Timestep variance score
        """
        if timesteps.numel() == 0:
            return torch.tensor(0.0, device=timesteps.device)
            
        # Compute variance across the frame dimension
        if timesteps.dim() == 2:  # [B, F]
            variance = torch.var(timesteps.float(), dim=1).mean()
        else:  # [B*F] or other shapes
            variance = torch.var(timesteps.float())
            
        return variance
    
    def update_sampling_stats(
        self, 
        is_async: bool, 
        ptss_probability: Optional[float] = None
    ):
        """
        Track async vs sync sampling statistics.
        
        Args:
            is_async: Whether this step used async sampling
            ptss_probability: The PTSS probability used
        """
        if is_async:
            self.async_steps += 1
        else:
            self.sync_steps += 1
            
        if ptss_probability is not None:
            self.ptss_probabilities.append(ptss_probability)
    
    def record_metrics(
        self, 
        frames: torch.Tensor,
        timesteps: torch.Tensor,
        is_async: bool,
        ptss_probability: Optional[float] = None
    ):
        """
        Record all metrics for a training step.
        
        Args:
            frames: [B, C, F, H, W] video frames
            timesteps: [B, F] timesteps used
            is_async: Whether async sampling was used
            ptss_probability: PTSS probability used
        """
        try:
            # Compute and store metrics
            temporal_loss = self.compute_temporal_consistency_loss(frames)
            diversity_score = self.compute_frame_diversity_score(frames)
            timestep_variance = self.compute_timestep_variance(timesteps)
            
            self.temporal_consistency_losses.append(temporal_loss.item())
            self.frame_diversity_scores.append(diversity_score.item())
            self.timestep_variance_scores.append(timestep_variance.item())
            
            # Update sampling stats
            self.update_sampling_stats(is_async, ptss_probability)
            
        except Exception as e:
            logger.warning(f"Failed to record FVDM metrics: {e}")
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of current metrics and statistics
        """
        total_steps = self.async_steps + self.sync_steps
        
        stats = {
            'fvdm/async_ratio': self.async_steps / max(total_steps, 1),
            'fvdm/sync_ratio': self.sync_steps / max(total_steps, 1),
            'fvdm/total_steps': total_steps,
        }
        
        if self.temporal_consistency_losses:
            stats['fvdm/avg_temporal_consistency'] = sum(self.temporal_consistency_losses) / len(self.temporal_consistency_losses)
            stats['fvdm/recent_temporal_consistency'] = sum(self.temporal_consistency_losses[-10:]) / min(10, len(self.temporal_consistency_losses))
            
        if self.frame_diversity_scores:
            stats['fvdm/avg_frame_diversity'] = sum(self.frame_diversity_scores) / len(self.frame_diversity_scores)
            stats['fvdm/recent_frame_diversity'] = sum(self.frame_diversity_scores[-10:]) / min(10, len(self.frame_diversity_scores))
            
        if self.timestep_variance_scores:
            stats['fvdm/avg_timestep_variance'] = sum(self.timestep_variance_scores) / len(self.timestep_variance_scores)
            stats['fvdm/recent_timestep_variance'] = sum(self.timestep_variance_scores[-10:]) / min(10, len(self.timestep_variance_scores))
            
        if self.ptss_probabilities:
            stats['fvdm/avg_ptss_probability'] = sum(self.ptss_probabilities) / len(self.ptss_probabilities)
            stats['fvdm/recent_ptss_probability'] = sum(self.ptss_probabilities[-10:]) / min(10, len(self.ptss_probabilities))
        
        return stats
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get statistics for recent training steps.
        
        Args:
            window: Number of recent steps to consider
            
        Returns:
            Dictionary of recent metrics
        """
        recent_stats = {}
        
        if self.temporal_consistency_losses:
            recent_losses = self.temporal_consistency_losses[-window:]
            recent_stats['fvdm/recent_temporal_consistency'] = sum(recent_losses) / len(recent_losses)
            
        if self.frame_diversity_scores:
            recent_diversity = self.frame_diversity_scores[-window:]
            recent_stats['fvdm/recent_frame_diversity'] = sum(recent_diversity) / len(recent_diversity)
            
        if self.timestep_variance_scores:
            recent_variance = self.timestep_variance_scores[-window:]
            recent_stats['fvdm/recent_timestep_variance'] = sum(recent_variance) / len(recent_variance)
            
        # Recent async ratio
        recent_async = sum(1 for i in range(max(0, total_steps - window), total_steps) 
                          if i < self.async_steps) if hasattr(self, 'async_steps') else 0
        recent_total = min(window, self.async_steps + self.sync_steps)
        if recent_total > 0:
            recent_stats['fvdm/recent_async_ratio'] = recent_async / recent_total
            
        return recent_stats


def create_fvdm_metrics(device: torch.device) -> FVDMTrainingMetrics:
    """
    Factory function to create FVDM metrics tracker.
    
    Args:
        device: Device to run computations on
        
    Returns:
        Configured FVDMTrainingMetrics instance
    """
    return FVDMTrainingMetrics(device=device)


# Utility functions for integration with existing systems
def compute_fvdm_loss_components(
    frames: torch.Tensor,
    timesteps: torch.Tensor,
    temporal_weight: float = 0.1,
    diversity_weight: float = 0.05
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute FVDM-specific loss components that can be added to main training loss.
    
    Args:
        frames: [B, C, F, H, W] video frames
        timesteps: [B, F] timesteps
        temporal_weight: Weight for temporal consistency loss
        diversity_weight: Weight for frame diversity loss (negative to encourage diversity)
        
    Returns:
        Tuple of (total_additional_loss, individual_components)
    """
    metrics = FVDMTrainingMetrics(device=frames.device)
    
    # Compute components
    temporal_loss = metrics.compute_temporal_consistency_loss(frames)
    diversity_score = metrics.compute_frame_diversity_score(frames)
    
    # Combine losses
    # Temporal consistency: lower is better (minimize)
    # Frame diversity: higher is better (maximize, so subtract)
    total_loss = (temporal_weight * temporal_loss - 
                  diversity_weight * diversity_score)
    
    components = {
        'temporal_consistency': temporal_loss,
        'frame_diversity': diversity_score,
        'timestep_variance': metrics.compute_timestep_variance(timesteps)
    }
    
    return total_loss, components