"""
SRPO setup and initialization utilities.

This module provides helper functions to set up SRPO training mode,
keeping the main trainer code clean and maintainable.
"""

import logging
from typing import Any, Dict
import torch.nn as nn

from srpo.srpo_config_schema import SRPOConfig, validate_srpo_config
from srpo.srpo_training_core import SRPOTrainingCore

logger = logging.getLogger(__name__)


def setup_srpo_training(
    args,
    accelerator,
    transformer: nn.Module,
    network: nn.Module,
    optimizer,
    vae: nn.Module,
    model_config: Dict[str, Any],
) -> SRPOTrainingCore:
    """
    Set up and initialize SRPO training mode.

    This function:
    1. Validates that required components (VAE, T5) are available
    2. Loads T5 text encoder
    3. Creates SRPOConfig from args
    4. Validates the config
    5. Initializes and returns SRPOTrainingCore

    Args:
        args: Training arguments from config
        accelerator: Accelerator instance
        transformer: WAN transformer model
        network: LoRA network
        optimizer: Optimizer instance
        vae: VAE model for latent decoding
        model_config: Model configuration dictionary

    Returns:
        Initialized SRPOTrainingCore instance

    Raises:
        ValueError: If required components are missing
    """
    logger.info(
        "🎯 Setting up SRPO (Semantic Relative Preference Optimization) training"
    )

    # Ensure VAE is available
    if vae is None:
        raise ValueError(
            "SRPO training requires a VAE checkpoint (set 'vae' in config)"
        )

    # Load T5 text encoder (required for on-the-fly prompt encoding)
    if not hasattr(args, "t5") or args.t5 is None:
        raise ValueError("SRPO training requires a T5 encoder (set 't5' in config)")

    logger.info(f"Loading T5 encoder from {args.t5}")
    from wan.modules.t5 import T5EncoderModel

    text_encoder = T5EncoderModel(
        text_len=model_config.get("text_len", 512),
        dtype=model_config.get("t5_dtype", accelerator.state.mixed_precision_dtype),
        device=accelerator.device,
        weight_path=args.t5,
        fp8=getattr(args, "fp8_t5", False),
    )
    text_encoder.eval()  # Freeze T5 for SRPO training
    logger.info("✓ T5 encoder loaded successfully")

    # Create SRPO config from args
    srpo_config = SRPOConfig(
        srpo_reward_model_name=args.srpo_reward_model_name,
        srpo_reward_model_dtype=args.srpo_reward_model_dtype,
        srpo_srp_control_weight=args.srpo_srp_control_weight,
        srpo_srp_positive_words=args.srpo_srp_positive_words,
        srpo_srp_negative_words=args.srpo_srp_negative_words,
        srpo_sigma_interpolation_method=args.srpo_sigma_interpolation_method,
        srpo_sigma_interpolation_min=args.srpo_sigma_interpolation_min,
        srpo_sigma_interpolation_max=args.srpo_sigma_interpolation_max,
        srpo_num_inference_steps=args.srpo_num_inference_steps,
        srpo_guidance_scale=args.srpo_guidance_scale,
        srpo_enable_sd3_time_shift=args.srpo_enable_sd3_time_shift,
        srpo_sd3_time_shift_value=args.srpo_sd3_time_shift_value,
        srpo_discount_denoise_min=args.srpo_discount_denoise_min,
        srpo_discount_denoise_max=args.srpo_discount_denoise_max,
        srpo_discount_inversion_start=args.srpo_discount_inversion_start,
        srpo_discount_inversion_end=args.srpo_discount_inversion_end,
        srpo_batch_size=args.srpo_batch_size,
        srpo_gradient_accumulation_steps=args.srpo_gradient_accumulation_steps,
        srpo_num_training_steps=args.srpo_num_training_steps,
        srpo_validation_prompts=args.srpo_validation_prompts,
        srpo_validation_frequency=args.srpo_validation_frequency,
        srpo_save_validation_videos=args.srpo_save_validation_videos,
        srpo_save_validation_as_images=args.srpo_save_validation_as_images,
        srpo_vae_scale_factor=args.srpo_vae_scale_factor,
        srpo_latent_channels=args.srpo_latent_channels,
        srpo_reward_frame_strategy=args.srpo_reward_frame_strategy,
        srpo_reward_num_frames=args.srpo_reward_num_frames,
        srpo_reward_aggregation=args.srpo_reward_aggregation,
        srpo_enable_video_rewards=args.srpo_enable_video_rewards,
        srpo_temporal_consistency_weight=args.srpo_temporal_consistency_weight,
        srpo_optical_flow_weight=args.srpo_optical_flow_weight,
        srpo_motion_quality_weight=args.srpo_motion_quality_weight,
        srpo_enable_euphonium=args.srpo_enable_euphonium,
        srpo_euphonium_process_reward_guidance_enabled=args.srpo_euphonium_process_reward_guidance_enabled,
        srpo_euphonium_process_reward_model_type=args.srpo_euphonium_process_reward_model_type,
        srpo_euphonium_process_reward_model_path=args.srpo_euphonium_process_reward_model_path,
        srpo_euphonium_process_reward_model_entry=args.srpo_euphonium_process_reward_model_entry,
        srpo_euphonium_process_reward_model_dtype=args.srpo_euphonium_process_reward_model_dtype,
        srpo_euphonium_process_reward_allow_proxy_fallback=args.srpo_euphonium_process_reward_allow_proxy_fallback,
        srpo_euphonium_process_reward_gradient_mode=args.srpo_euphonium_process_reward_gradient_mode,
        srpo_euphonium_process_reward_spsa_sigma=args.srpo_euphonium_process_reward_spsa_sigma,
        srpo_euphonium_process_reward_spsa_num_samples=args.srpo_euphonium_process_reward_spsa_num_samples,
        srpo_euphonium_process_reward_guidance_scale=args.srpo_euphonium_process_reward_guidance_scale,
        srpo_euphonium_process_reward_guidance_kl_beta=args.srpo_euphonium_process_reward_guidance_kl_beta,
        srpo_euphonium_process_reward_guidance_eta=args.srpo_euphonium_process_reward_guidance_eta,
        srpo_euphonium_process_reward_start_step=args.srpo_euphonium_process_reward_start_step,
        srpo_euphonium_process_reward_end_step=args.srpo_euphonium_process_reward_end_step,
        srpo_euphonium_process_reward_interval=args.srpo_euphonium_process_reward_interval,
        srpo_euphonium_process_reward_normalize_gradient=args.srpo_euphonium_process_reward_normalize_gradient,
        srpo_euphonium_use_delta_t_for_guidance=args.srpo_euphonium_use_delta_t_for_guidance,
        srpo_euphonium_process_reward_apply_in_recovery=args.srpo_euphonium_process_reward_apply_in_recovery,
        srpo_euphonium_process_reward_detach_target=args.srpo_euphonium_process_reward_detach_target,
        srpo_euphonium_dual_reward_advantage_mode=args.srpo_euphonium_dual_reward_advantage_mode,
        srpo_euphonium_process_reward_advantage_coef=args.srpo_euphonium_process_reward_advantage_coef,
        srpo_euphonium_outcome_reward_advantage_coef=args.srpo_euphonium_outcome_reward_advantage_coef,
        srpo_euphonium_log_interval=args.srpo_euphonium_log_interval,
    )

    # Validate config
    validate_srpo_config(srpo_config)

    # Initialize SRPO training core
    srpo_trainer = SRPOTrainingCore(
        srpo_config=srpo_config,
        model_config=model_config,
        accelerator=accelerator,
        transformer=transformer,
        network=network,
        optimizer=optimizer,
        vae=vae,
        text_encoder=text_encoder,
        args=args,
    )

    logger.info("✓ SRPO training setup complete")
    return srpo_trainer
