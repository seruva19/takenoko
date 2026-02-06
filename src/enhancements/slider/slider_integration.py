# Slider Training Integration - Clean interface for training_core
# Encapsulates all slider functionality with minimal training_core pollution

import argparse
from typing import Optional, Any, Dict
import torch
from torch import Tensor

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SliderIntegration:
    """
    Clean slider training integration for Takenoko.

    Provides a minimal interface to training_core while encapsulating
    all slider-specific functionality in slider modules.
    """

    def __init__(self):
        self.slider_manager = None
        self._initialized = False

    def initialize(self, args: argparse.Namespace) -> None:
        """
        Initialize slider training if needed based on network module.

        Args:
            args: Training arguments
        """
        if self._initialized:
            return

        # Check if slider training should be enabled
        if self._should_enable_slider(args):
            try:
                from enhancements.slider.slider_training_manager import SliderTrainingManager

                self.slider_manager = SliderTrainingManager()
                self.slider_manager.initialize_from_args(args)
                logger.info("✅ Slider training integration enabled")

            except ImportError:
                logger.debug("Slider training not available (modules not found)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize slider training: {e}")
                # Ensure slider_manager remains None on failure
                self.slider_manager = None

        self._initialized = True

    def _should_enable_slider(self, args: argparse.Namespace) -> bool:
        """
        Check if slider training should be enabled based on network module.

        Args:
            args: Training arguments

        Returns:
            True if slider training should be enabled
        """
        network_module = getattr(args, "network_module", "")
        return "slider" in network_module.lower()

    def set_text_encoder(self, text_encoder) -> None:
        """
        Set text encoder for slider training if enabled.

        Args:
            text_encoder: T5 text encoder model
        """
        if self.slider_manager is not None:
            self.slider_manager.set_text_encoder(text_encoder)

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

        This is the main integration point for training_core.

        Returns:
            Either slider loss components (if slider enabled) or normal loss_components
        """
        if self.slider_manager is not None:
            # Slider training enabled - delegate to slider manager
            return self.slider_manager.compute_loss_if_enabled(
                loss_computer=loss_computer,
                transformer=transformer,
                network=network,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                batch=batch,
                noise=noise,
                noise_scheduler=noise_scheduler,
                args=args,
                accelerator=accelerator,
                transition_loss_context=kwargs.get("transition_loss_context"),
                **kwargs,
            )
        else:
            # No slider training - delegate to normal loss computation
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
                sft_alignment_helper=kwargs.get("sft_alignment_helper"),
                moalign_helper=kwargs.get("moalign_helper"),
                semfeat_helper=kwargs.get("semfeat_helper"),
                bfm_conditioning_helper=kwargs.get("bfm_conditioning_helper"),
                reg_helper=kwargs.get("reg_helper"),
                reg_cls_pred=kwargs.get("reg_cls_pred"),
                reg_cls_target=kwargs.get("reg_cls_target"),
                sara_helper=kwargs.get("sara_helper"),
                layer_sync_helper=kwargs.get("layer_sync_helper"),
                crepa_helper=kwargs.get("crepa_helper"),
                self_transcendence_helper=kwargs.get("self_transcendence_helper"),
                haste_helper=kwargs.get("haste_helper"),
                contrastive_attention_helper=kwargs.get(
                    "contrastive_attention_helper"
                ),
                raft=kwargs.get("raft"),
                warp_fn=kwargs.get("warp_fn"),
                adaptive_manager=kwargs.get("adaptive_manager"),
                transition_loss_context=kwargs.get("transition_loss_context"),
                noise_scheduler=noise_scheduler,
                global_step=kwargs.get("global_step"),
                current_epoch=kwargs.get("current_epoch"),
            )

    def is_enabled(self) -> bool:
        """Check if slider training is enabled."""
        return self.slider_manager is not None

    def get_config_dict(self) -> Dict[str, Any]:
        """Get slider configuration for logging/saving."""
        if self.slider_manager is not None:
            return self.slider_manager.get_config_dict()
        return {}


# Global slider integration instance
_slider_integration: Optional[SliderIntegration] = None


def get_slider_integration() -> SliderIntegration:
    """Get or create the global slider integration instance."""
    global _slider_integration
    if _slider_integration is None:
        _slider_integration = SliderIntegration()
    return _slider_integration


def initialize_slider_integration(args: argparse.Namespace) -> None:
    """Initialize slider integration from training args."""
    integration = get_slider_integration()
    integration.initialize(args)


def set_text_encoder_for_slider(text_encoder) -> None:
    """Set text encoder for slider training if enabled."""
    integration = get_slider_integration()
    integration.set_text_encoder(text_encoder)


def compute_slider_loss_if_enabled(
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
    """Compute loss with slider integration if enabled."""
    integration = get_slider_integration()
    return integration.compute_loss_if_enabled(
        loss_computer=loss_computer,
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
