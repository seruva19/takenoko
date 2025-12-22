# Slider Training Manager - Fully encapsulated slider training functionality
# Handles all slider-specific logic without polluting existing code

from typing import Optional, Dict, Any
import torch
from torch import Tensor
import argparse

from enhancements.slider.slider_training_core import SliderTrainingCore
from enhancements.slider.slider_config import SliderConfig

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SliderTrainingManager:
    """
    Fully encapsulated manager for slider training functionality.

    This class handles all slider training logic and provides a clean interface
    for integration with Takenoko's training system without polluting existing code.
    """

    def __init__(self):
        self.slider_core: Optional[SliderTrainingCore] = None
        self.is_slider_enabled = False

    def initialize_from_args(self, args: argparse.Namespace) -> None:
        """
        Initialize slider training if the network module indicates slider training.

        Args:
            args: Training arguments from config
        """
        # Check if slider training is enabled by network module
        network_module = getattr(args, "network_module", "")
        self.is_slider_enabled = "slider" in network_module.lower()

        if not self.is_slider_enabled:
            logger.debug(
                "Slider training not enabled - network module does not contain 'slider'"
            )
            return

        # Validate slider configuration
        try:
            slider_config = SliderConfig(
                guidance_strength=getattr(args, "slider_guidance_strength", 3.0),
                anchor_strength=getattr(args, "slider_anchor_strength", 1.0),
                guidance_scale=getattr(args, "slider_guidance_scale", 1.0),
                guidance_embedding_scale=getattr(args, "slider_guidance_embedding_scale", 1.0),
                target_guidance_scale=getattr(args, "slider_target_guidance_scale", 1.0),
                positive_prompt=getattr(args, "slider_positive_prompt", ""),
                negative_prompt=getattr(args, "slider_negative_prompt", ""),
                target_class=getattr(args, "slider_target_class", ""),
                anchor_class=getattr(args, "slider_anchor_class", None),
                slider_learning_rate_multiplier=getattr(
                    args, "slider_learning_rate_multiplier", 1.0
                ),
                slider_cache_embeddings=getattr(args, "slider_cache_embeddings", True),
                slider_t5_device=getattr(args, "slider_t5_device", "cpu"),
                slider_cache_on_init=getattr(args, "slider_cache_on_init", True),
            )

            # Validate required fields
            slider_config.validate()

            # Initialize slider core
            self.slider_core = SliderTrainingCore(
                guidance_strength=slider_config.guidance_strength,
                anchor_strength=slider_config.anchor_strength,
                guidance_scale=slider_config.guidance_scale,
                guidance_embedding_scale=slider_config.guidance_embedding_scale,
                target_guidance_scale=slider_config.target_guidance_scale,
                positive_prompt=slider_config.positive_prompt,
                negative_prompt=slider_config.negative_prompt,
                target_class=slider_config.target_class,
                anchor_class=slider_config.anchor_class,
                t5_device=slider_config.slider_t5_device,
                cache_on_init=slider_config.slider_cache_on_init,
            )

            # Initialize embeddings with args for T5 loading
            self.slider_core.initialize_embeddings(args=args)

            logger.info("✅ Slider training manager initialized successfully")

        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize slider training: {e}")
            self.is_slider_enabled = False
            self.slider_core = None

    def set_text_encoder(self, text_encoder) -> None:
        """
        Set the text encoder for real embedding computation.
        This should be called after the T5 encoder is loaded.

        Args:
            text_encoder: T5 text encoder model from Takenoko
        """
        if self.slider_core is not None:
            logger.info("🔧 Setting real T5 text encoder for slider training")
            # Re-initialize embeddings with the real text encoder
            self.slider_core.text_encoder = text_encoder
            self.slider_core.is_initialized = False  # Force re-initialization
            self.slider_core.initialize_embeddings(text_encoder=text_encoder)

    def should_use_slider_loss(self, network=None) -> bool:
        """Check if slider training should be used."""
        # Check if slider is enabled and configured
        if not (self.is_slider_enabled and self.slider_core is not None):
            return False

        # Additionally check if the network itself is a slider network
        if network is not None and hasattr(network, "is_slider_network"):
            try:
                return network.is_slider_network()
            except:
                pass

        return True

    def compute_loss_if_enabled(
        self,
        loss_computer,
        transformer: torch.nn.Module,
        network,
        noisy_latents: Tensor,
        timesteps: Tensor,
        batch: Dict[str, Any],
        noise: Tensor,
        noise_scheduler,
        args,
        accelerator,
        **kwargs,
    ) -> Any:
        """
        Compute loss - either slider guided loss or delegate to normal loss computer.

        Returns:
            Either slider loss tensor (if slider enabled) or normal loss_components
        """
        if not self.should_use_slider_loss(network):
            # Not slider training - delegate to normal loss computation
            return loss_computer.compute_training_loss(
                args=args,
                accelerator=accelerator,
                latents=kwargs.get("latents", noisy_latents),
                noise=noise,
                noisy_model_input=noisy_latents,
                timesteps=timesteps,
                network_dtype=kwargs.get("network_dtype"),
                model_pred=kwargs.get("model_pred"),
                target=kwargs.get("target"),
                weighting=kwargs.get("weighting"),
                batch=batch,
                intermediate_z=kwargs.get("intermediate_z"),
                vae=kwargs.get("vae"),
                transformer=transformer,
                network=network,
                control_signal_processor=kwargs.get("control_signal_processor"),
                repa_helper=kwargs.get("repa_helper"),
                sara_helper=kwargs.get("sara_helper"),
                layer_sync_helper=kwargs.get("layer_sync_helper"),
                crepa_helper=kwargs.get("crepa_helper"),
                raft=kwargs.get("raft"),
                warp_fn=kwargs.get("warp_fn"),
                adaptive_manager=kwargs.get("adaptive_manager"),
                transition_loss_context=kwargs.get("transition_loss_context"),
            )

        # Slider training - compute guided loss
        try:
            slider_loss = self.slider_core.compute_guided_loss(
                transformer=transformer,
                network=network,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                batch=batch,
                noise=noise,
                noise_scheduler=noise_scheduler,
                args=args,
                accelerator=accelerator,
                **kwargs,
            )

            # Return a simple object that behaves like loss_components for compatibility
            class SliderLossComponents:
                def __init__(self, total_loss):
                    self.total_loss = total_loss
                    self.is_slider_loss = True

            return SliderLossComponents(slider_loss)

        except Exception as e:
            logger.warning(
                f"⚠️ Slider loss computation failed, falling back to normal loss: {e}"
            )
            # Fallback to normal loss computation
            return loss_computer.compute_training_loss(
                args=args,
                accelerator=accelerator,
                latents=kwargs.get("latents", noisy_latents),
                noise=noise,
                noisy_model_input=noisy_latents,
                timesteps=timesteps,
                network_dtype=kwargs.get("network_dtype"),
                model_pred=kwargs.get("model_pred"),
                target=kwargs.get("target"),
                weighting=kwargs.get("weighting"),
                batch=batch,
                intermediate_z=kwargs.get("intermediate_z"),
                vae=kwargs.get("vae"),
                transformer=transformer,
                network=network,
                control_signal_processor=kwargs.get("control_signal_processor"),
                repa_helper=kwargs.get("repa_helper"),
                sara_helper=kwargs.get("sara_helper"),
                layer_sync_helper=kwargs.get("layer_sync_helper"),
                crepa_helper=kwargs.get("crepa_helper"),
                raft=kwargs.get("raft"),
                warp_fn=kwargs.get("warp_fn"),
                adaptive_manager=kwargs.get("adaptive_manager"),
            )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get slider configuration for logging/saving."""
        if not self.should_use_slider_loss():
            return {}

        return {
            "slider_enabled": True,
            "slider_config": (
                self.slider_core.get_config_dict() if self.slider_core else {}
            ),
        }
