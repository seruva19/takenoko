from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch


@dataclass
class TemporalConsistencyConfig:
    """Configuration for frequency-domain temporal consistency enhancement."""

    # Main feature flags
    enable_frequency_domain_temporal_consistency: bool = False
    freq_temporal_enable_motion_coherence: bool = False
    freq_temporal_enable_prediction_loss: bool = False

    # Low-frequency preservation settings
    freq_temporal_low_threshold: float = 0.25  # As used in paper
    freq_temporal_high_threshold: float = 0.7  # For motion components

    # Temporal consistency loss weights
    freq_temporal_consistency_weight: float = 0.1
    freq_temporal_motion_weight: float = 0.05
    freq_temporal_prediction_weight: float = 0.08

    # Frame processing settings
    freq_temporal_max_distance: int = 4  # Maximum frame distance for consistency
    freq_temporal_decay_factor: float = 0.8  # Weight decay for distant frames
    freq_temporal_min_frames: int = 4  # Minimum frames needed to apply enhancement
    freq_temporal_motion_threshold: float = (
        0.1  # Motion variation threshold for coherence loss
    )

    # Frequency analysis settings
    freq_temporal_preserve_dc: bool = True  # Preserve DC (average) component
    freq_temporal_adaptive_threshold: bool = False
    freq_temporal_adaptive_range: Tuple[float, float] = (0.15, 0.35)

    # Loss scheduling
    freq_temporal_start_step: int = 0
    freq_temporal_end_step: Optional[int] = None
    freq_temporal_warmup_steps: int = 100

    # Performance settings
    freq_temporal_enable_caching: bool = True
    freq_temporal_cache_size: int = 500
    freq_temporal_batch_parallel: bool = True
    freq_temporal_apply_every_n_steps: int = (
        1  # Apply enhancement every N steps for performance
    )
    freq_temporal_max_frames_per_batch: int = 16  # Limit frames processed to avoid OOM

    # Logging settings
    freq_temporal_log_every_steps: int = 500  # Console log interval (INFO)
    freq_temporal_tb_log_every_steps: int = 500  # TensorBoard log interval

    # Advanced settings
    freq_temporal_weight_strategy: str = (
        "exponential"  # "linear", "exponential", "uniform"
    )
    freq_temporal_loss_reduction: str = "mean"  # "mean", "sum", "none"
    freq_temporal_apply_to_latent: bool = True  # Apply to latent space vs pixel space

    @classmethod
    def from_args(cls, args) -> "TemporalConsistencyConfig":
        """Create config from command line arguments."""
        return cls(
            enable_frequency_domain_temporal_consistency=getattr(
                args, "enable_frequency_domain_temporal_consistency", False
            ),
            freq_temporal_enable_motion_coherence=getattr(
                args, "freq_temporal_enable_motion_coherence", False
            ),
            freq_temporal_enable_prediction_loss=getattr(
                args, "freq_temporal_enable_prediction_loss", False
            ),
            freq_temporal_low_threshold=getattr(
                args, "freq_temporal_low_threshold", 0.25
            ),
            freq_temporal_high_threshold=getattr(
                args, "freq_temporal_high_threshold", 0.7
            ),
            freq_temporal_consistency_weight=getattr(
                args, "freq_temporal_consistency_weight", 0.1
            ),
            freq_temporal_motion_weight=getattr(
                args, "freq_temporal_motion_weight", 0.05
            ),
            freq_temporal_prediction_weight=getattr(
                args, "freq_temporal_prediction_weight", 0.08
            ),
            freq_temporal_max_distance=getattr(args, "freq_temporal_max_distance", 4),
            freq_temporal_decay_factor=getattr(args, "freq_temporal_decay_factor", 0.8),
            freq_temporal_min_frames=getattr(args, "freq_temporal_min_frames", 4),
            freq_temporal_motion_threshold=getattr(
                args, "freq_temporal_motion_threshold", 0.1
            ),
            freq_temporal_preserve_dc=getattr(args, "freq_temporal_preserve_dc", True),
            freq_temporal_adaptive_threshold=getattr(
                args, "freq_temporal_adaptive_threshold", False
            ),
            freq_temporal_adaptive_range=getattr(
                args, "freq_temporal_adaptive_range", (0.15, 0.35)
            ),
            freq_temporal_start_step=getattr(args, "freq_temporal_start_step", 0),
            freq_temporal_end_step=getattr(args, "freq_temporal_end_step", None),
            freq_temporal_warmup_steps=getattr(args, "freq_temporal_warmup_steps", 100),
            freq_temporal_enable_caching=getattr(
                args, "freq_temporal_enable_caching", True
            ),
            freq_temporal_cache_size=getattr(args, "freq_temporal_cache_size", 500),
            freq_temporal_batch_parallel=getattr(
                args, "freq_temporal_batch_parallel", True
            ),
            freq_temporal_apply_every_n_steps=getattr(
                args, "freq_temporal_apply_every_n_steps", 1
            ),
            freq_temporal_max_frames_per_batch=getattr(
                args, "freq_temporal_max_frames_per_batch", 16
            ),
            freq_temporal_log_every_steps=getattr(
                args, "freq_temporal_log_every_steps", 500
            ),
            freq_temporal_tb_log_every_steps=getattr(
                args, "freq_temporal_tb_log_every_steps", 500
            ),
            freq_temporal_weight_strategy=getattr(
                args, "freq_temporal_weight_strategy", "exponential"
            ),
            freq_temporal_loss_reduction=getattr(
                args, "freq_temporal_loss_reduction", "mean"
            ),
            freq_temporal_apply_to_latent=getattr(
                args, "freq_temporal_apply_to_latent", True
            ),
        )

    def is_enabled(self) -> bool:
        """Check if any temporal consistency features are enabled."""
        return (
            self.enable_frequency_domain_temporal_consistency
            or self.freq_temporal_enable_motion_coherence
            or self.freq_temporal_enable_prediction_loss
        )

    def get_temporal_weights(self, num_frames: int) -> List[float]:
        """Generate temporal weights based on frame distance.

        Closer frames get higher weights for better temporal consistency.
        """
        if self.freq_temporal_weight_strategy == "uniform":
            return [1.0] * (num_frames - 1)
        elif self.freq_temporal_weight_strategy == "linear":
            # Linear decay: closer frames (lower i) get higher weights
            return [1.0 - (i / (num_frames - 1)) * 0.5 for i in range(num_frames - 1)]
        elif self.freq_temporal_weight_strategy == "exponential":
            # Exponential decay: closer frames get exponentially higher weights
            return [self.freq_temporal_decay_factor**i for i in range(num_frames - 1)]
        else:
            return [1.0] * (num_frames - 1)

    def should_apply_at_step(self, step: int) -> bool:
        """Check if temporal consistency should be applied at this training step."""
        if not self.is_enabled():
            return False

        if step < self.freq_temporal_start_step:
            return False

        if (
            self.freq_temporal_end_step is not None
            and step > self.freq_temporal_end_step
        ):
            return False

        # Apply only every N steps for performance optimization
        if step % self.freq_temporal_apply_every_n_steps != 0:
            return False

        return True

    def get_loss_weight_multiplier(self, step: int) -> float:
        """Get loss weight multiplier with warmup scheduling."""
        if not self.should_apply_at_step(step):
            return 0.0

        # Apply warmup
        if step < self.freq_temporal_start_step + self.freq_temporal_warmup_steps:
            warmup_progress = (
                step - self.freq_temporal_start_step
            ) / self.freq_temporal_warmup_steps
            return min(1.0, warmup_progress)

        return 1.0
