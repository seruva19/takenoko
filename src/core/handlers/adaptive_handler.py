"""Adaptive timestep sampling handler for training loop."""

import logging
import torch
import torch.nn.functional as F
from typing import Any, Optional

logger = logging.getLogger(__name__)


def handle_adaptive_timestep_sampling(
    adaptive_manager: Optional[Any],
    accelerator: Any,
    training_model: Any,
    latents: torch.Tensor,
    noise_scheduler: Any,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
) -> None:
    """Handle adaptive timestep sampling processing during training.
    
    Args:
        adaptive_manager: The adaptive timestep manager instance
        accelerator: Accelerator instance
        training_model: The diffusion model for training
        latents: Clean latents (x_0)
        noise_scheduler: The noise scheduler/diffusion process
        model_pred: Model prediction tensor
        target: Target tensor
        timesteps: Current timesteps tensor
    """
    if not (adaptive_manager and adaptive_manager.enabled):
        return
    
    try:
        # Check if research mode is enabled
        if (
            hasattr(adaptive_manager, "research_mode_enabled")
            and adaptive_manager.research_mode_enabled
        ):
            # Use full research algorithm (Algorithm 1 & 2)
            research_result = adaptive_manager.research_training_step(
                model=training_model,  # The diffusion model
                x_0=latents,  # Clean latents
                diffusion=noise_scheduler,  # The noise scheduler
                all_timesteps=list(
                    range(0, 1000, 10)
                ),  # Sample every 10 timesteps for efficiency
            )

            if research_result and research_result.get("research_mode"):
                # Log research statistics
                if accelerator.is_main_process:
                    logger.info(
                        f"🔬 Research step: t={research_result.get('sampled_timestep')}, "
                        f"Δ_t_k={research_result.get('delta_t_k', 0):.6f}"
                    )

                    if research_result.get("selected_features"):
                        logger.debug(
                            f"   Selected features: {research_result.get('selected_features')[:5]}..."
                        )

        else:
            # Use simplified approach (current implementation)
            with torch.no_grad():
                # Compute individual losses per timestep
                per_timestep_losses = F.mse_loss(
                    model_pred.detach().float(),
                    target.detach().float(),
                    reduction="none",
                )

                # Reduce spatial and channel dimensions, keep batch
                if per_timestep_losses.ndim == 5:  # Video: [B, C, F, H, W]
                    per_timestep_losses = per_timestep_losses.mean(dim=[1, 2, 3, 4])
                elif per_timestep_losses.ndim == 4:  # Image: [B, C, H, W]
                    per_timestep_losses = per_timestep_losses.mean(dim=[1, 2, 3])
                else:
                    per_timestep_losses = per_timestep_losses.mean(
                        dim=list(range(1, per_timestep_losses.ndim))
                    )

                # Convert timesteps to 0-1 range for recording
                timesteps_normalized = timesteps.float() / 1000.0

                # Record losses for analysis
                adaptive_manager.record_timestep_loss(
                    timesteps_normalized, per_timestep_losses
                )

                # Update important timesteps if needed
                adaptive_manager.update_important_timesteps()

    except Exception as e:
        logger.debug(f"Error in adaptive timestep processing: {e}")
        # Continue training without adaptive sampling