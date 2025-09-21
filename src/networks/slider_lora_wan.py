# Slider LoRA network module for concept editing training

import math
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn

from networks.lora_wan import LoRANetwork, LoRAModule
from utils.lora_utils import create_network_from_weights
import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SliderLoRANetwork(LoRANetwork):
    """
    Slider LoRA Network for concept editing training.

    This network extends the standard LoRA implementation to support
    slider training where the same LoRA is used with positive and negative
    multipliers to enhance or suppress concepts.

    The presence of this network type automatically enables slider training.
    """

    def __init__(
        self,
        unet,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        module_class: str = "LoRAModule",
        varbvs_session=None,
        # Slider-specific parameters (extracted from network_args)
        guidance_strength: float = 3.0,
        anchor_strength: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Slider LoRA Network.

        Args:
            unet: The transformer/unet model to attach LoRA to
            rank: LoRA rank/dimension
            multiplier: Base LoRA multiplier
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate for LoRA layers
            module_class: LoRA module class to use
            varbvs_session: Optional session for variable selection
            guidance_strength: Strength of concept guidance (default: 3.0)
            anchor_strength: Strength of anchor class preservation (default: 1.0)
        """
        # Initialize base LoRA network
        super().__init__(
            unet=unet,
            rank=rank,
            multiplier=multiplier,
            alpha=alpha,
            dropout=dropout,
            module_class=module_class,
            varbvs_session=varbvs_session,
            **kwargs,
        )

        # Slider-specific configuration
        self.guidance_strength = guidance_strength
        self.anchor_strength = anchor_strength
        self.is_slider_training = True

        # Track current multiplier for dual-polarity training
        self._current_multiplier = multiplier

        logger.info(f"✅ Slider LoRA Network initialized:")
        logger.info(f"   • Guidance strength: {guidance_strength}")
        logger.info(f"   • Anchor strength: {anchor_strength}")
        logger.info(f"   • Base rank: {rank}, alpha: {alpha}")

    def is_slider_network(self) -> bool:
        """Return True since this is always a slider network."""
        return True

    def set_multiplier(self, multiplier: float) -> None:
        """
        Set the LoRA multiplier for dual-polarity training.

        This is called during slider training to switch between
        positive and negative concept enhancement.

        Args:
            multiplier: New multiplier value (can be negative)
        """
        self._current_multiplier = multiplier
        # Update all LoRA modules with new multiplier
        for lora in self.unet_loras:
            if hasattr(lora, "multiplier"):
                lora.multiplier = multiplier

    def get_multiplier(self) -> float:
        """Get current LoRA multiplier."""
        return self._current_multiplier

    def prepare_optimizer_params(
        self, unet_lr: float, input_lr_scale: float = 1.0, **kwargs
    ) -> tuple[List[Dict], List[str]]:
        """
        Prepare optimizer parameters for slider training.

        Args:
            unet_lr: Learning rate for transformer/unet parameters
            input_lr_scale: Learning rate scale for input layers

        Returns:
            Tuple of (parameter_groups, descriptions)
        """
        # Use parent implementation - slider training doesn't change optimizer setup
        return super().prepare_optimizer_params(unet_lr, input_lr_scale, **kwargs)

    def prepare_grad_etc(self, unet) -> None:
        """Prepare gradients and other training setup."""
        # Use parent implementation
        super().prepare_grad_etc(unet)

    def apply_to(self, multiplier: Optional[float] = None, **kwargs) -> None:
        """
        Apply LoRA to the model with optional multiplier override.

        Args:
            multiplier: Optional multiplier override
        """
        if multiplier is not None:
            self.set_multiplier(multiplier)
        super().apply_to(**kwargs)

    def is_mergeable(self) -> bool:
        """Check if this network can be merged with others."""
        # Slider networks should not be merged as they require dynamic multipliers
        return False

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving."""
        config = super().get_config_dict()
        config.update(
            {
                "guidance_strength": self.guidance_strength,
                "anchor_strength": self.anchor_strength,
                "is_slider_training": True,
                "network_type": "slider_lora",
            }
        )
        return config

    @classmethod
    def create_network_from_weights(
        cls, multiplier: float, file: str, unet, **kwargs
    ) -> "SliderLoRANetwork":
        """
        Create slider network from saved weights.

        Args:
            multiplier: LoRA multiplier
            file: Path to weights file
            unet: Model to attach to

        Returns:
            Loaded SliderLoRANetwork instance
        """
        # Load base network first
        network = create_network_from_weights(multiplier, file, unet, **kwargs)

        # Convert to slider network if it isn't already
        if not isinstance(network, SliderLoRANetwork):
            # Create new slider network with same parameters
            slider_network = cls(
                unet=unet,
                rank=getattr(network, "rank", 4),
                multiplier=multiplier,
                alpha=getattr(network, "alpha", 1.0),
                **kwargs,
            )

            # Copy weights from loaded network
            slider_network.load_state_dict(network.state_dict(), strict=False)
            return slider_network

        return network


def create_network(
    multiplier: float,
    network_dim: Optional[int] = None,
    network_alpha: Optional[float] = None,
    unet=None,
    **kwargs,
) -> SliderLoRANetwork:
    """
    Factory function to create a Slider LoRA network.

    Args:
        multiplier: LoRA multiplier
        network_dim: LoRA rank/dimension
        network_alpha: LoRA alpha scaling
        unet: Model to attach to
        **kwargs: Additional arguments

    Returns:
        SliderLoRANetwork instance
    """
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = network_dim

    # Extract slider-specific parameters
    guidance_strength = kwargs.pop("guidance_strength", 3.0)
    anchor_strength = kwargs.pop("anchor_strength", 1.0)

    network = SliderLoRANetwork(
        unet=unet,
        rank=network_dim,
        multiplier=multiplier,
        alpha=network_alpha,
        guidance_strength=guidance_strength,
        anchor_strength=anchor_strength,
        **kwargs,
    )

    return network
