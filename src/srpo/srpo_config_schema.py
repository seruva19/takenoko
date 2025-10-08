"""
SRPO configuration schema with validation.

This module defines the SRPOConfig dataclass that encapsulates all SRPO-related
hyperparameters for Semantic Relative Preference Optimization.
It is instantiated from argparse.Namespace during config loading.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SRPOConfig:
    """
    Configuration for SRPO (Semantic Relative Preference Optimization) training.

    All parameters are loaded from the TOML config file under root-level `srpo_*` keys.
    This dataclass provides type safety and validation.
    """

    # Reward model configuration
    srpo_reward_model_name: str = "hps"  # Options: "hps", "pickscore", "aesthetic"
    srpo_reward_model_dtype: str = (
        "float32"  # Options: "float32", "bfloat16", "float16"
    )
    srpo_srp_control_weight: float = 1.0  # SRP k parameter (r = (1+k)*r_pos - r_neg)
    srpo_srp_positive_words: Optional[List[str]] = (
        None  # Positive control words for SRP
    )
    srpo_srp_negative_words: Optional[List[str]] = (
        None  # Negative control words for SRP
    )

    # Direct-Align algorithm parameters
    srpo_sigma_interpolation_method: str = "linear"  # Options: "linear", "cosine"
    srpo_sigma_interpolation_min: float = 0.0  # Minimum sigma for interpolation
    srpo_sigma_interpolation_max: float = 1.0  # Maximum sigma for interpolation
    srpo_num_inference_steps: int = 50  # Number of Euler steps for online rollout
    srpo_guidance_scale: float = (
        1.0  # CFG guidance scale (DEPRECATED - not used by WAN)
    )
    srpo_enable_sd3_time_shift: bool = (
        True  # Enable SD3-style time shift in sigma schedule
    )
    srpo_sd3_time_shift_value: float = 3.0  # Shift parameter for SD3 time shift

    # Discount schedule for gradient backpropagation
    srpo_discount_denoise_min: float = 0.0  # Minimum discount for denoise branch
    srpo_discount_denoise_max: float = 1.0  # Maximum discount for denoise branch
    srpo_discount_inversion_start: float = 1.0  # Starting discount for inversion branch
    srpo_discount_inversion_end: float = 0.0  # Ending discount for inversion branch

    # Training hyperparameters
    srpo_batch_size: int = 1  # Number of videos per SRPO iteration
    srpo_gradient_accumulation_steps: int = 4  # Gradient accumulation for SRPO
    srpo_num_training_steps: int = 500  # Total SRPO training steps

    # Validation parameters
    srpo_validation_prompts: Optional[List[str]] = None  # Prompts for validation
    srpo_validation_frequency: int = 50  # Validate every N steps
    srpo_save_validation_videos: bool = True  # Save validation videos to disk
    srpo_save_validation_as_images: bool = False  # Save as PNG images instead of videos

    # WAN-specific parameters (VERIFIED against WAN architecture)
    srpo_vae_scale_factor: int = 8  # WAN uses 8 (not FLUX's 16)
    srpo_latent_channels: int = 16  # WAN latent channels

    # Multi-frame reward parameters
    srpo_reward_frame_strategy: str = (
        "first"  # Options: "first", "uniform", "all", "boundary"
    )
    srpo_reward_num_frames: int = 1  # Number of frames to sample for reward computation
    srpo_reward_aggregation: str = "mean"  # Options: "mean", "min", "max", "weighted"

    # Video-specific reward parameters
    srpo_enable_video_rewards: bool = False  # Enable video-specific reward models
    srpo_temporal_consistency_weight: float = (
        0.0  # Weight for temporal consistency reward
    )
    srpo_optical_flow_weight: float = 0.0  # Weight for optical flow smoothness reward
    srpo_motion_quality_weight: float = 0.0  # Weight for motion quality reward

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate reward model
        valid_reward_models = ["hps", "pickscore", "aesthetic"]
        if self.srpo_reward_model_name not in valid_reward_models:
            raise ValueError(
                f"Invalid srpo_reward_model_name='{self.srpo_reward_model_name}'. "
                f"Must be one of {valid_reward_models}"
            )

        # Validate SRP control words
        if self.srpo_srp_positive_words is None:
            self.srpo_srp_positive_words = [
                "beautiful",
                "stunning",
                "gorgeous",
                "masterpiece",
                "professional",
                "high quality",
                "detailed",
                "elegant",
                "vibrant",
                "exquisite",
                "breathtaking",
                "magnificent",
                "pristine",
                "refined",
                "impeccable",
                "award-winning",
                "cinematic",
            ]
            logger.info(
                f"No srpo_srp_positive_words provided, using {len(self.srpo_srp_positive_words)} defaults"
            )

        if self.srpo_srp_negative_words is None:
            self.srpo_srp_negative_words = [
                "ugly",
                "blurry",
                "low quality",
                "distorted",
                "amateur",
                "poor",
                "bad",
                "grainy",
                "noisy",
                "artifacts",
                "watermark",
                "text overlay",
                "oversaturated",
                "underexposed",
                "overexposed",
                "pixelated",
                "dull",
            ]
            logger.info(
                f"No srpo_srp_negative_words provided, using {len(self.srpo_srp_negative_words)} defaults"
            )

        # Validate sigma interpolation
        if (
            self.srpo_sigma_interpolation_min < 0.0
            or self.srpo_sigma_interpolation_min > 1.0
        ):
            raise ValueError(
                f"srpo_sigma_interpolation_min={self.srpo_sigma_interpolation_min} must be in [0, 1]"
            )
        if (
            self.srpo_sigma_interpolation_max < 0.0
            or self.srpo_sigma_interpolation_max > 1.0
        ):
            raise ValueError(
                f"srpo_sigma_interpolation_max={self.srpo_sigma_interpolation_max} must be in [0, 1]"
            )
        if self.srpo_sigma_interpolation_min >= self.srpo_sigma_interpolation_max:
            raise ValueError(
                f"srpo_sigma_interpolation_min={self.srpo_sigma_interpolation_min} must be < "
                f"srpo_sigma_interpolation_max={self.srpo_sigma_interpolation_max}"
            )

        # Validate discount schedules
        if self.srpo_discount_denoise_min < 0.0 or self.srpo_discount_denoise_min > 1.0:
            raise ValueError(
                f"srpo_discount_denoise_min={self.srpo_discount_denoise_min} must be in [0, 1]"
            )
        if self.srpo_discount_denoise_max < 0.0 or self.srpo_discount_denoise_max > 1.0:
            raise ValueError(
                f"srpo_discount_denoise_max={self.srpo_discount_denoise_max} must be in [0, 1]"
            )
        if (
            self.srpo_discount_inversion_start < 0.0
            or self.srpo_discount_inversion_start > 1.0
        ):
            raise ValueError(
                f"srpo_discount_inversion_start={self.srpo_discount_inversion_start} must be in [0, 1]"
            )
        if (
            self.srpo_discount_inversion_end < 0.0
            or self.srpo_discount_inversion_end > 1.0
        ):
            raise ValueError(
                f"srpo_discount_inversion_end={self.srpo_discount_inversion_end} must be in [0, 1]"
            )

        # Validate training parameters
        if self.srpo_batch_size < 1:
            raise ValueError(f"srpo_batch_size={self.srpo_batch_size} must be >= 1")
        if self.srpo_gradient_accumulation_steps < 1:
            raise ValueError(
                f"srpo_gradient_accumulation_steps={self.srpo_gradient_accumulation_steps} must be >= 1"
            )
        if self.srpo_num_training_steps < 1:
            raise ValueError(
                f"srpo_num_training_steps={self.srpo_num_training_steps} must be >= 1"
            )

        # Validate validation parameters
        if self.srpo_validation_prompts is None:
            self.srpo_validation_prompts = [
                "a beautiful sunset over mountains",
                "a cat playing with a ball of yarn",
                "a futuristic cityscape at night",
            ]
            logger.info(
                f"No srpo_validation_prompts provided, using {len(self.srpo_validation_prompts)} defaults"
            )

        # Validate multi-frame reward parameters
        valid_strategies = ["first", "uniform", "all", "boundary"]
        if self.srpo_reward_frame_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid srpo_reward_frame_strategy='{self.srpo_reward_frame_strategy}'. "
                f"Must be one of {valid_strategies}"
            )

        valid_aggregations = ["mean", "min", "max", "weighted"]
        if self.srpo_reward_aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid srpo_reward_aggregation='{self.srpo_reward_aggregation}'. "
                f"Must be one of {valid_aggregations}"
            )

        if self.srpo_reward_num_frames < 1:
            raise ValueError(
                f"srpo_reward_num_frames must be >= 1, got {self.srpo_reward_num_frames}"
            )

        # Log warning if incompatible settings are detected
        incompatible_flags = []
        if self.srpo_guidance_scale != 1.0:
            incompatible_flags.append(
                f"srpo_guidance_scale={self.srpo_guidance_scale} (WAN doesn't use CFG during training)"
            )

        if incompatible_flags:
            logger.warning(
                f"SRPO training enabled with potentially incompatible settings: {', '.join(incompatible_flags)}"
            )

        logger.info("✓ SRPO configuration validated successfully")


def validate_srpo_config(config: SRPOConfig) -> None:
    """
    Additional validation for SRPO configuration.

    This function is called after the config is fully loaded to perform
    cross-parameter validation.

    Args:
        config: SRPOConfig instance to validate
    """
    # Check if SRP control words have overlap
    positive_set = set(config.srpo_srp_positive_words)
    negative_set = set(config.srpo_srp_negative_words)
    overlap = positive_set & negative_set

    if overlap:
        logger.warning(
            f"Found {len(overlap)} words in both positive and negative control lists: {overlap}. "
            "This may reduce SRP effectiveness."
        )

    # Warn if control word lists are very small
    if len(config.srpo_srp_positive_words) < 3:
        logger.warning(
            f"Only {len(config.srpo_srp_positive_words)} positive control words provided. "
            "Consider using more words for better SRP diversity."
        )
    if len(config.srpo_srp_negative_words) < 3:
        logger.warning(
            f"Only {len(config.srpo_srp_negative_words)} negative control words provided. "
            "Consider using more words for better SRP diversity."
        )

    logger.info(
        f"SRPO config summary: {config.srpo_reward_model_name} reward model, "
        f"{config.srpo_num_training_steps} steps, batch_size={config.srpo_batch_size}"
    )
