"""
SRPO configuration schema with validation.

This module defines the SRPOConfig dataclass that encapsulates all SRPO-related
hyperparameters for Semantic Relative Preference Optimization.
It is instantiated from argparse.Namespace during config loading.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging
import math

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

    # Self-Paced GRPO-inspired co-evolving reward mixing (default off)
    srpo_enable_self_paced_reward: bool = False
    srpo_self_paced_visual_threshold: float = 0.75
    srpo_self_paced_temporal_threshold: float = 0.75
    srpo_self_paced_semantic_threshold: float = 0.75
    srpo_self_paced_softmax_beta: float = 5.0
    srpo_self_paced_sparsity_lambda: float = 0.5
    srpo_self_paced_sigmoid_scale: float = 8.0
    srpo_self_paced_visual_scale: float = 1.0
    srpo_self_paced_temporal_scale: float = 1.0
    srpo_self_paced_semantic_scale: float = 1.0
    srpo_self_paced_enable_advantage_norm: bool = True
    srpo_self_paced_advantage_norm_eps: float = 1e-6
    srpo_self_paced_use_grpo_clip_objective: bool = False
    srpo_self_paced_grpo_clip_eps: float = 1e-4
    srpo_self_paced_policy_std: float = 1.0
    srpo_self_paced_apply_discount_to_grpo: bool = True
    srpo_self_paced_use_lagged_old_policy: bool = True
    srpo_self_paced_old_policy_update_interval: int = 1
    srpo_self_paced_old_policy_offload_cpu: bool = True
    srpo_self_paced_logprob_mode: str = "action_gaussian"
    srpo_self_paced_enable_offline_threshold_calibration: bool = False
    srpo_self_paced_offline_calibration_steps: int = 12
    srpo_self_paced_offline_calibration_batch_size: int = 0
    srpo_self_paced_offline_calibration_update_network: bool = False
    srpo_self_paced_enable_prompt_curriculum: bool = False
    srpo_self_paced_stage1_visual_prompt_suffix: str = ""
    srpo_self_paced_stage3_semantic_prompt_suffix: str = ""
    srpo_self_paced_auto_calibrate_thresholds: bool = False
    srpo_self_paced_threshold_calibration_factor: float = 0.7
    srpo_self_paced_threshold_calibration_warmup_steps: int = 50
    srpo_self_paced_threshold_calibration_momentum: float = 0.9

    # Euphonium-inspired SRPO enhancement (default off for behavior preservation)
    srpo_enable_euphonium: bool = False
    srpo_euphonium_process_reward_guidance_enabled: bool = False
    srpo_euphonium_process_reward_model_type: str = "none"
    srpo_euphonium_process_reward_model_path: str = ""
    srpo_euphonium_process_reward_model_entry: str = ""
    srpo_euphonium_process_reward_model_dtype: str = "float32"
    srpo_euphonium_process_reward_allow_proxy_fallback: bool = True
    srpo_euphonium_process_reward_gradient_mode: str = "autograd"
    srpo_euphonium_process_reward_spsa_sigma: float = 0.01
    srpo_euphonium_process_reward_spsa_num_samples: int = 1
    srpo_euphonium_process_reward_guidance_scale: float = 0.1
    srpo_euphonium_process_reward_guidance_kl_beta: float = 0.1
    srpo_euphonium_process_reward_guidance_eta: float = 1.0
    srpo_euphonium_process_reward_start_step: int = 0
    srpo_euphonium_process_reward_end_step: int = -1
    srpo_euphonium_process_reward_interval: int = 1
    srpo_euphonium_process_reward_normalize_gradient: bool = True
    srpo_euphonium_use_delta_t_for_guidance: bool = False
    srpo_euphonium_process_reward_apply_in_recovery: bool = False
    srpo_euphonium_process_reward_detach_target: bool = True
    srpo_euphonium_dual_reward_advantage_mode: str = "none"
    srpo_euphonium_process_reward_advantage_coef: float = 1.0
    srpo_euphonium_outcome_reward_advantage_coef: float = 1.0
    srpo_euphonium_log_interval: int = 50

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

        # Validate self-paced reward mixing settings
        if self.srpo_enable_self_paced_reward:
            if self.srpo_self_paced_softmax_beta <= 0.0:
                raise ValueError(
                    "srpo_self_paced_softmax_beta must be > 0, got "
                    f"{self.srpo_self_paced_softmax_beta}"
                )
            if self.srpo_self_paced_sigmoid_scale <= 0.0:
                raise ValueError(
                    "srpo_self_paced_sigmoid_scale must be > 0, got "
                    f"{self.srpo_self_paced_sigmoid_scale}"
                )
            if self.srpo_self_paced_advantage_norm_eps <= 0.0:
                raise ValueError(
                    "srpo_self_paced_advantage_norm_eps must be > 0, got "
                    f"{self.srpo_self_paced_advantage_norm_eps}"
                )
            if not (0.0 < self.srpo_self_paced_grpo_clip_eps <= 1.0):
                raise ValueError(
                    "srpo_self_paced_grpo_clip_eps must be in (0, 1], got "
                    f"{self.srpo_self_paced_grpo_clip_eps}"
                )
            if self.srpo_self_paced_policy_std <= 0.0:
                raise ValueError(
                    "srpo_self_paced_policy_std must be > 0, got "
                    f"{self.srpo_self_paced_policy_std}"
                )
            if self.srpo_self_paced_old_policy_update_interval < 1:
                raise ValueError(
                    "srpo_self_paced_old_policy_update_interval must be >= 1, got "
                    f"{self.srpo_self_paced_old_policy_update_interval}"
                )
            if self.srpo_self_paced_logprob_mode not in (
                "action_gaussian",
                "latent_error_proxy",
            ):
                raise ValueError(
                    "srpo_self_paced_logprob_mode must be 'action_gaussian' or "
                    f"'latent_error_proxy', got {self.srpo_self_paced_logprob_mode!r}"
                )
            if self.srpo_self_paced_offline_calibration_steps < 1:
                raise ValueError(
                    "srpo_self_paced_offline_calibration_steps must be >= 1, got "
                    f"{self.srpo_self_paced_offline_calibration_steps}"
                )
            if self.srpo_self_paced_offline_calibration_batch_size < 0:
                raise ValueError(
                    "srpo_self_paced_offline_calibration_batch_size must be >= 0, got "
                    f"{self.srpo_self_paced_offline_calibration_batch_size}"
                )
            if self.srpo_self_paced_visual_scale < 0.0:
                raise ValueError(
                    "srpo_self_paced_visual_scale must be >= 0, got "
                    f"{self.srpo_self_paced_visual_scale}"
                )
            if self.srpo_self_paced_temporal_scale < 0.0:
                raise ValueError(
                    "srpo_self_paced_temporal_scale must be >= 0, got "
                    f"{self.srpo_self_paced_temporal_scale}"
                )
            if self.srpo_self_paced_semantic_scale < 0.0:
                raise ValueError(
                    "srpo_self_paced_semantic_scale must be >= 0, got "
                    f"{self.srpo_self_paced_semantic_scale}"
                )
            if (
                self.srpo_self_paced_visual_scale
                + self.srpo_self_paced_temporal_scale
                + self.srpo_self_paced_semantic_scale
                <= 0.0
            ):
                raise ValueError(
                    "At least one self-paced reward scale must be > 0 "
                    "(srpo_self_paced_visual_scale / temporal_scale / semantic_scale)."
                )
            threshold_values = [
                self.srpo_self_paced_visual_threshold,
                self.srpo_self_paced_temporal_threshold,
                self.srpo_self_paced_semantic_threshold,
            ]
            for threshold_name, threshold_value in zip(
                (
                    "srpo_self_paced_visual_threshold",
                    "srpo_self_paced_temporal_threshold",
                    "srpo_self_paced_semantic_threshold",
                ),
                threshold_values,
            ):
                if not math.isfinite(float(threshold_value)):
                    raise ValueError(f"{threshold_name} must be finite, got {threshold_value}")
            if self.srpo_batch_size < 2:
                logger.warning(
                    "Self-Paced SRPO is enabled with srpo_batch_size=%d. "
                    "Batch-size 1 has weak group-relative statistics.",
                    self.srpo_batch_size,
                )
                if self.srpo_self_paced_enable_advantage_norm:
                    logger.warning(
                        "Self-Paced SRPO advantage normalization is enabled with batch_size=1; "
                        "normalization may be skipped when variance is near zero."
                    )
            if self.srpo_self_paced_threshold_calibration_warmup_steps < 1:
                raise ValueError(
                    "srpo_self_paced_threshold_calibration_warmup_steps must be >= 1, got "
                    f"{self.srpo_self_paced_threshold_calibration_warmup_steps}"
                )
            if not (
                0.0 <= self.srpo_self_paced_threshold_calibration_momentum < 1.0
            ):
                raise ValueError(
                    "srpo_self_paced_threshold_calibration_momentum must be in [0,1), got "
                    f"{self.srpo_self_paced_threshold_calibration_momentum}"
                )

        # Validate Euphonium integration settings
        valid_dual_reward_modes = ["none", "only", "both"]
        valid_process_model_types = ["none", "torchscript", "python_callable"]
        valid_process_model_dtypes = ["float32", "bfloat16", "float16"]
        valid_process_gradient_modes = ["autograd", "spsa"]
        if self.srpo_euphonium_dual_reward_advantage_mode not in valid_dual_reward_modes:
            raise ValueError(
                "Invalid srpo_euphonium_dual_reward_advantage_mode="
                f"{self.srpo_euphonium_dual_reward_advantage_mode!r}. Must be one of "
                f"{valid_dual_reward_modes}"
            )
        if (
            self.srpo_euphonium_process_reward_model_type
            not in valid_process_model_types
        ):
            raise ValueError(
                "Invalid srpo_euphonium_process_reward_model_type="
                f"{self.srpo_euphonium_process_reward_model_type!r}. Must be one of "
                f"{valid_process_model_types}"
            )
        if (
            self.srpo_euphonium_process_reward_model_dtype
            not in valid_process_model_dtypes
        ):
            raise ValueError(
                "Invalid srpo_euphonium_process_reward_model_dtype="
                f"{self.srpo_euphonium_process_reward_model_dtype!r}. Must be one of "
                f"{valid_process_model_dtypes}"
            )
        if (
            self.srpo_euphonium_process_reward_gradient_mode
            not in valid_process_gradient_modes
        ):
            raise ValueError(
                "Invalid srpo_euphonium_process_reward_gradient_mode="
                f"{self.srpo_euphonium_process_reward_gradient_mode!r}. Must be one of "
                f"{valid_process_gradient_modes}"
            )
        if self.srpo_euphonium_process_reward_spsa_sigma <= 0.0:
            raise ValueError(
                "srpo_euphonium_process_reward_spsa_sigma must be > 0, got "
                f"{self.srpo_euphonium_process_reward_spsa_sigma}"
            )
        if self.srpo_euphonium_process_reward_spsa_num_samples <= 0:
            raise ValueError(
                "srpo_euphonium_process_reward_spsa_num_samples must be > 0, got "
                f"{self.srpo_euphonium_process_reward_spsa_num_samples}"
            )
        if (
            self.srpo_euphonium_process_reward_model_type == "torchscript"
            and self.srpo_euphonium_process_reward_model_path.strip() == ""
        ):
            raise ValueError(
                "srpo_euphonium_process_reward_model_path must be set when "
                "srpo_euphonium_process_reward_model_type='torchscript'."
            )
        if (
            self.srpo_euphonium_process_reward_model_type == "python_callable"
            and self.srpo_euphonium_process_reward_model_entry.strip() == ""
        ):
            raise ValueError(
                "srpo_euphonium_process_reward_model_entry must be set when "
                "srpo_euphonium_process_reward_model_type='python_callable'."
            )
        if self.srpo_euphonium_process_reward_guidance_kl_beta <= 0.0:
            raise ValueError(
                "srpo_euphonium_process_reward_guidance_kl_beta must be > 0, got "
                f"{self.srpo_euphonium_process_reward_guidance_kl_beta}"
            )
        if self.srpo_euphonium_process_reward_guidance_eta < 0.0:
            raise ValueError(
                "srpo_euphonium_process_reward_guidance_eta must be >= 0, got "
                f"{self.srpo_euphonium_process_reward_guidance_eta}"
            )
        if self.srpo_euphonium_process_reward_start_step < 0:
            raise ValueError(
                "srpo_euphonium_process_reward_start_step must be >= 0, got "
                f"{self.srpo_euphonium_process_reward_start_step}"
            )
        if (
            self.srpo_euphonium_process_reward_end_step != -1
            and self.srpo_euphonium_process_reward_end_step
            < self.srpo_euphonium_process_reward_start_step
        ):
            raise ValueError(
                "srpo_euphonium_process_reward_end_step must be -1 or >= "
                f"srpo_euphonium_process_reward_start_step ({self.srpo_euphonium_process_reward_start_step})"
            )
        if self.srpo_euphonium_process_reward_interval <= 0:
            raise ValueError(
                "srpo_euphonium_process_reward_interval must be > 0, got "
                f"{self.srpo_euphonium_process_reward_interval}"
            )
        if self.srpo_euphonium_process_reward_advantage_coef < 0.0:
            raise ValueError(
                "srpo_euphonium_process_reward_advantage_coef must be >= 0, got "
                f"{self.srpo_euphonium_process_reward_advantage_coef}"
            )
        if self.srpo_euphonium_outcome_reward_advantage_coef < 0.0:
            raise ValueError(
                "srpo_euphonium_outcome_reward_advantage_coef must be >= 0, got "
                f"{self.srpo_euphonium_outcome_reward_advantage_coef}"
            )
        if self.srpo_euphonium_log_interval <= 0:
            raise ValueError(
                f"srpo_euphonium_log_interval must be > 0, got {self.srpo_euphonium_log_interval}"
            )
        needs_process_signal = (
            self.srpo_euphonium_process_reward_guidance_enabled
            or self.srpo_euphonium_dual_reward_advantage_mode in {"only", "both"}
        )
        if (
            needs_process_signal
            and self.srpo_euphonium_process_reward_model_type == "none"
            and not self.srpo_euphonium_process_reward_allow_proxy_fallback
        ):
            raise ValueError(
                "Euphonium process reward guidance/dual mode requires either "
                "a process reward model backend or "
                "srpo_euphonium_process_reward_allow_proxy_fallback=true."
            )
        if (
            needs_process_signal
            and self.srpo_euphonium_process_reward_gradient_mode == "spsa"
            and self.srpo_euphonium_process_reward_model_type == "none"
            and self.srpo_euphonium_process_reward_allow_proxy_fallback
        ):
            logger.warning(
                "srpo_euphonium_process_reward_gradient_mode='spsa' is enabled without "
                "an external process reward model; proxy guidance path will ignore SPSA and use legacy proxy gradient."
            )
        if self.srpo_enable_euphonium and (
            self.srpo_euphonium_dual_reward_advantage_mode in {"only", "both"}
            and self.srpo_batch_size < 2
        ):
            logger.warning(
                "SRPO Euphonium dual reward mode '%s' works best with srpo_batch_size >= 2 "
                "(current: %d).",
                self.srpo_euphonium_dual_reward_advantage_mode,
                self.srpo_batch_size,
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
    if config.srpo_enable_self_paced_reward:
        logger.info(
            "SRPO Self-Paced reward enabled: thresholds=(%.3f, %.3f, %.3f), beta=%.3f, lambda=%.3f, "
            "sigmoid_scale=%.3f, scales=(%.3f, %.3f, %.3f), advantage_norm=%s, eps=%.2e, "
            "grpo_clip=%s, clip_eps=%.6f, policy_std=%.4f, lagged_old=%s, old_update=%d, offload_cpu=%s, "
            "logprob_mode=%s, prompt_curriculum=%s, auto_calibrate=%s, offline_calibration=%s/%d batch=%d update=%s",
            config.srpo_self_paced_visual_threshold,
            config.srpo_self_paced_temporal_threshold,
            config.srpo_self_paced_semantic_threshold,
            config.srpo_self_paced_softmax_beta,
            config.srpo_self_paced_sparsity_lambda,
            config.srpo_self_paced_sigmoid_scale,
            config.srpo_self_paced_visual_scale,
            config.srpo_self_paced_temporal_scale,
            config.srpo_self_paced_semantic_scale,
            config.srpo_self_paced_enable_advantage_norm,
            config.srpo_self_paced_advantage_norm_eps,
            config.srpo_self_paced_use_grpo_clip_objective,
            config.srpo_self_paced_grpo_clip_eps,
            config.srpo_self_paced_policy_std,
            config.srpo_self_paced_use_lagged_old_policy,
            config.srpo_self_paced_old_policy_update_interval,
            config.srpo_self_paced_old_policy_offload_cpu,
            config.srpo_self_paced_logprob_mode,
            config.srpo_self_paced_enable_prompt_curriculum,
            config.srpo_self_paced_auto_calibrate_thresholds,
            config.srpo_self_paced_enable_offline_threshold_calibration,
            config.srpo_self_paced_offline_calibration_steps,
            config.srpo_self_paced_offline_calibration_batch_size,
            config.srpo_self_paced_offline_calibration_update_network,
        )
    if config.srpo_enable_euphonium:
        logger.info(
            "SRPO Euphonium enabled: guidance=%s process_model=%s dtype=%s grad_mode=%s spsa_sigma=%.6f spsa_samples=%d scale=%.4f kl_beta=%.4f eta=%.4f recovery_guidance=%s mode=%s process_coef=%.3f outcome_coef=%.3f",
            config.srpo_euphonium_process_reward_guidance_enabled,
            config.srpo_euphonium_process_reward_model_type,
            config.srpo_euphonium_process_reward_model_dtype,
            config.srpo_euphonium_process_reward_gradient_mode,
            config.srpo_euphonium_process_reward_spsa_sigma,
            config.srpo_euphonium_process_reward_spsa_num_samples,
            config.srpo_euphonium_process_reward_guidance_scale,
            config.srpo_euphonium_process_reward_guidance_kl_beta,
            config.srpo_euphonium_process_reward_guidance_eta,
            config.srpo_euphonium_process_reward_apply_in_recovery,
            config.srpo_euphonium_dual_reward_advantage_mode,
            config.srpo_euphonium_process_reward_advantage_coef,
            config.srpo_euphonium_outcome_reward_advantage_coef,
        )
