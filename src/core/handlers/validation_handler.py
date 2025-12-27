"""Validation logic handler for training loop."""

import argparse
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def handle_step_validation(
    should_validating: bool,
    validation_core: Any,
    val_dataloader: Optional[Any],
    val_epoch_step_sync: Any,
    current_epoch: Optional[Any],
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    accelerator: Any,
    transformer: Any,
    noise_scheduler: Any,
    control_signal_processor: Any,
    vae: Optional[Any],
    sampling_manager: Optional[Any],
    warned_no_val_pixels_for_perceptual: bool,
    last_validated_step: int,
    timestep_distribution: Optional[Any] = None,
    should_fast_validating: bool = False,
    last_fast_validated_step: int = -1,
) -> Tuple[int, bool, int]:
    """Handle validation during training step.

    Args:
        should_validating: Whether validation should occur
        validation_core: The validation core instance
        val_dataloader: Validation dataloader
        val_epoch_step_sync: Epoch step sync object
        current_epoch: Current epoch counter
        epoch: Current epoch (0-indexed)
        global_step: Current global step
        args: Training arguments
        accelerator: Accelerator instance
        transformer: Transformer model
        noise_scheduler: Noise scheduler
        control_signal_processor: Control signal processor
        vae: VAE model (may be None)
        sampling_manager: Sampling manager instance
        warned_no_val_pixels_for_perceptual: Warning flag state
        last_validated_step: Last step when validation occurred
        timestep_distribution: Timestep distribution (optional)
        should_fast_validating: Whether fast validation should occur
        last_fast_validated_step: Last step when fast validation occurred

    Returns:
        Tuple of (new_last_validated_step, new_warned_flag, new_last_fast_validated_step)
    """
    if not should_validating and not should_fast_validating:
        return last_validated_step, warned_no_val_pixels_for_perceptual, last_fast_validated_step

    # Sync validation datasets before validation runs
    validation_core.sync_validation_epoch(
        val_dataloader,
        val_epoch_step_sync,
        current_epoch.value if current_epoch else epoch + 1,
        global_step,
    )

    # Determine if validation metrics require a VAE and lazily load if needed
    metrics_enabled = any(
        [
            bool(getattr(args, "enable_perceptual_snr", False)),
            bool(getattr(args, "enable_temporal_ssim", False)),
            bool(getattr(args, "enable_temporal_lpips", False)),
            bool(getattr(args, "enable_flow_warped_ssim", False)),
            bool(getattr(args, "enable_fvd", False)),
            bool(getattr(args, "enable_vmaf", False)),
        ]
    )
    # Only consider loading a VAE if validation batches include pixels
    requires_vae_for_val = (
        bool(getattr(args, "load_val_pixels", False)) and metrics_enabled
    )

    # If metrics are enabled but pixels are not loaded, warn once and skip VAE loading
    new_warned_flag = warned_no_val_pixels_for_perceptual
    if (
        metrics_enabled
        and not getattr(args, "load_val_pixels", False)
        and accelerator.is_main_process
        and not warned_no_val_pixels_for_perceptual
    ):
        logger.warning(
            "Perceptual/temporal validation metrics are enabled but load_val_pixels=false; skipping these metrics and not loading a VAE. Set load_val_pixels=true to enable them."
        )
        new_warned_flag = True

    temp_val_vae = None
    val_vae_to_use = vae
    if val_vae_to_use is None and requires_vae_for_val:
        try:
            if sampling_manager is not None:
                if accelerator.is_main_process:
                    logger.info("🔄 Loading VAE temporarily for validation metrics...")
                temp_val_vae = sampling_manager._load_vae_lazy()  # type: ignore[attr-defined]
                val_vae_to_use = temp_val_vae
            else:
                if accelerator.is_main_process:
                    logger.warning(
                        "Validation metrics requiring a VAE are enabled but no SamplingManager is available to lazy-load one. Metrics may be skipped."
                    )
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(
                    (
                        "Failed to load VAE for validation metrics: %s. "
                        "Ensure a valid 'vae' checkpoint path is set in the config, network access is available to download it, "
                        "and that its dtype (vae_dtype) is compatible with the current device."
                    ),
                    e,
                )

    # Determine subset_fraction and random_subset for fast validation
    subset_fraction = None
    random_subset = False
    if should_fast_validating:
        subset_fraction = getattr(args, "validation_fast_subset_fraction", 0.1)
        random_subset = getattr(args, "validation_fast_random_subset", False)
        if accelerator.is_main_process:
            sampling_type = "randomly sampled" if random_subset else "first"
            logger.info(f"⚡ Running fast validation with {subset_fraction*100:.0f}% of validation data ({sampling_type})")

    val_loss = validation_core.validate(
        accelerator,
        transformer,
        val_dataloader,
        noise_scheduler,
        args,
        control_signal_processor,
        val_vae_to_use,
        global_step,
        timestep_distribution=timestep_distribution,
        subset_fraction=subset_fraction,
        random_subset=random_subset,
    )
    validation_core.log_validation_results(accelerator, val_loss, global_step)

    # Unload temporary VAE if it was loaded for validation
    if temp_val_vae is not None and sampling_manager is not None:
        try:
            sampling_manager._unload_vae(temp_val_vae)  # type: ignore[attr-defined]
            if accelerator.is_main_process:
                logger.info("🧹 Unloaded temporary VAE after validation")
        except Exception as e:
            if accelerator.is_main_process:
                logger.debug(f"Failed to unload temporary VAE after validation: {e}")

    # Track that validation occurred at this step
    new_last_validated_step = global_step if should_validating else last_validated_step
    new_last_fast_validated_step = global_step if should_fast_validating else last_fast_validated_step
    return new_last_validated_step, new_warned_flag, new_last_fast_validated_step


def handle_epoch_end_validation(
    should_validate_on_epoch_end: bool,
    val_dataloader: Optional[Any],
    last_validated_step: int,
    global_step: int,
    validation_core: Any,
    val_epoch_step_sync: Any,
    current_epoch: Optional[Any],
    epoch: int,
    args: argparse.Namespace,
    accelerator: Any,
    transformer: Any,
    noise_scheduler: Any,
    control_signal_processor: Any,
    vae: Optional[Any],
) -> None:
    """Handle validation at end of epoch.

    Args:
        should_validate_on_epoch_end: Whether epoch-end validation should occur
        val_dataloader: Validation dataloader
        last_validated_step: Last step when validation occurred
        global_step: Current global step
        validation_core: The validation core instance
        val_epoch_step_sync: Epoch step sync object
        current_epoch: Current epoch counter
        epoch: Current epoch (0-indexed)
        args: Training arguments
        accelerator: Accelerator instance
        transformer: Transformer model
        noise_scheduler: Noise scheduler
        control_signal_processor: Control signal processor
        vae: VAE model (may be None)
    """
    if not (
        val_dataloader is not None
        and last_validated_step != global_step
        and should_validate_on_epoch_end
    ):
        if val_dataloader is None:
            accelerator.print(f"\n[Epoch {epoch+1}] No validation dataset configured")
        elif last_validated_step == global_step:
            accelerator.print(
                f"\n[Epoch {epoch+1}] Validation already performed at step {global_step}"
            )
        return

    # Sync validation datasets before validation runs
    validation_core.sync_validation_epoch(
        val_dataloader,
        val_epoch_step_sync,
        current_epoch.value if current_epoch else epoch + 1,
        global_step,
    )

    val_loss = validation_core.validate(
        accelerator,
        transformer,
        val_dataloader,
        noise_scheduler,
        args,
        control_signal_processor,
        vae,
        global_step,
    )
    validation_core.log_validation_results(
        accelerator, val_loss, global_step, epoch + 1
    )
