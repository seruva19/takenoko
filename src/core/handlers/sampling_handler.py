"""Sampling logic handler for training loop."""

import argparse
from typing import Any, Optional


def handle_training_sampling(
    should_sampling: bool,
    sampling_manager: Optional[Any],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    vae: Any,
    transformer: Any,
    sample_parameters: Any,
    dit_dtype: Any,
    last_sampled_step: int,
) -> int:
    """Handle image sampling during training if conditions are met.
    
    Args:
        should_sampling: Whether sampling should occur
        sampling_manager: The sampling manager instance
        args: Training arguments
        epoch: Current epoch (0-indexed)
        global_step: Current global step
        vae: VAE model
        transformer: Transformer model
        sample_parameters: Sampling parameters
        dit_dtype: Model dtype
        last_sampled_step: Last step when sampling occurred
        
    Returns:
        Updated last_sampled_step value
    """
    if not should_sampling or not sampling_manager:
        return last_sampled_step
    
    # Use epoch-based naming only if sampling was triggered by epoch, not steps
    # This prevents filename conflicts when resuming training in the same epoch
    epoch_for_naming = None
    if (
        args.sample_every_n_epochs is not None
        and args.sample_every_n_epochs > 0
        and (epoch + 1) % args.sample_every_n_epochs == 0
    ):
        # This sampling was triggered by epoch boundary
        epoch_for_naming = epoch + 1
    # Otherwise, leave epoch_for_naming as None to use step-based naming

    sampling_manager.sample_images(
        None,  # accelerator - will be passed by caller
        args,
        epoch_for_naming,  # Use None for step-based sampling
        global_step,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    )

    # Track that sampling occurred at this step
    return global_step


def handle_epoch_end_sampling(
    should_sample_at_epoch_end: bool,
    last_sampled_step: int,
    global_step: int,
    sampling_manager: Optional[Any],
    args: argparse.Namespace,
    epoch: int,
    vae: Any,
    transformer: Any,
    sample_parameters: Any,
    dit_dtype: Any,
) -> None:
    """Handle epoch-end sampling if conditions are met.
    
    Args:
        should_sample_at_epoch_end: Whether epoch-end sampling should occur
        last_sampled_step: Last step when sampling occurred
        global_step: Current global step
        sampling_manager: The sampling manager instance
        args: Training arguments
        epoch: Current epoch (0-indexed) 
        vae: VAE model
        transformer: Transformer model
        sample_parameters: Sampling parameters
        dit_dtype: Model dtype
    """
    # Only sample if epoch-based sampling is enabled AND we haven't already sampled at this step
    if not (should_sample_at_epoch_end and last_sampled_step != global_step and sampling_manager):
        return
    
    sampling_manager.sample_images(
        None,  # accelerator - will be passed by caller
        args,
        epoch + 1,
        global_step,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    )


# Wrapper functions that include accelerator parameter for compatibility
def handle_training_sampling_with_accelerator(
    should_sampling: bool,
    sampling_manager: Optional[Any],
    args: argparse.Namespace,
    accelerator: Any,
    epoch: int,
    global_step: int,
    vae: Any,
    transformer: Any,
    sample_parameters: Any,
    dit_dtype: Any,
    last_sampled_step: int,
) -> int:
    """Handle training sampling with accelerator parameter."""
    if not should_sampling or not sampling_manager:
        return last_sampled_step
    
    # Use epoch-based naming only if sampling was triggered by epoch, not steps
    epoch_for_naming = None
    if (
        args.sample_every_n_epochs is not None
        and args.sample_every_n_epochs > 0
        and (epoch + 1) % args.sample_every_n_epochs == 0
    ):
        epoch_for_naming = epoch + 1

    sampling_manager.sample_images(
        accelerator,
        args,
        epoch_for_naming,
        global_step,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    )

    return global_step


def handle_epoch_end_sampling_with_accelerator(
    should_sample_at_epoch_end: bool,
    last_sampled_step: int,
    global_step: int,
    sampling_manager: Optional[Any],
    args: argparse.Namespace,
    accelerator: Any,
    epoch: int,
    vae: Any,
    transformer: Any,
    sample_parameters: Any,
    dit_dtype: Any,
) -> None:
    """Handle epoch-end sampling with accelerator parameter."""
    if not (should_sample_at_epoch_end and last_sampled_step != global_step and sampling_manager):
        return
    
    sampling_manager.sample_images(
        accelerator,
        args,
        epoch + 1,
        global_step,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    )
