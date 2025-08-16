"""VAE training network for WAN models.

This module enables direct training of the VAE model, useful for:
- Fine-tuning VAE on specific datasets
- Domain adaptation
- Improving reconstruction quality
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from wan.modules.vae import WanVAE

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class VaeWanNetwork(torch.nn.Module):
    """Network module for training VAE directly."""

    def __init__(
        self,
        vae: Any,  # WanVAE, but using Any to avoid linter issues
        multiplier: float = 1.0,
        training_mode: str = "full",  # "full", "decoder_only", "encoder_only"
        **kwargs,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.training_mode = training_mode
        self.vae = vae

        # Freeze/unfreeze based on training mode
        self._setup_training_mode()

        logger.info(f"VAE training mode: {training_mode}")
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def _setup_training_mode(self):
        """Setup which parts of VAE to train based on training_mode."""
        # First freeze everything
        for param in self.vae.parameters():
            param.requires_grad = False

        if self.training_mode == "full":
            # Train entire VAE
            for param in self.vae.parameters():
                param.requires_grad = True
        elif self.training_mode == "decoder_only":
            # Only train decoder
            if hasattr(self.vae, "decoder"):
                for param in self.vae.decoder.parameters():
                    param.requires_grad = True
        elif self.training_mode == "encoder_only":
            # Only train encoder
            if hasattr(self.vae, "encoder"):
                for param in self.vae.encoder.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def prepare_optimizer_params(
        self, unet_lr: float, input_lr_scale: float = 1.0, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Prepare optimizer parameters for VAE training."""
        params = []
        lr_descriptions = []

        # Get trainable parameters
        trainable_params = [p for p in self.vae.parameters() if p.requires_grad]

        if trainable_params:
            params.append(
                {
                    "params": trainable_params,
                    "lr": unet_lr * self.multiplier,
                }
            )
            lr_descriptions.append(f"vae_{self.training_mode}")

        logger.info(f"VAE optimizer: {len(trainable_params):,} trainable parameters")
        return params, lr_descriptions

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.vae.parameters() if p.requires_grad]

    def apply_max_norm_regularization(
        self, max_norm_value: float, device: torch.device
    ) -> Tuple[int, float, float]:
        """Apply max norm regularization to trainable parameters."""
        if max_norm_value <= 0:
            return 0, 0.0, 0.0

        params = self.get_trainable_params()
        if not params:
            return 0, 0.0, 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.data) for p in params])
        ).item()
        max_norm = max_norm_value

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in params:
                p.data.mul_(clip_coef)
            return len(params), total_norm, max_norm

        return 0, total_norm, total_norm

    def on_epoch_start(self, transformer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_step_start(self) -> None:
        """Called at the start of each step."""
        pass

    def forward(self, *args, **kwargs):
        """Forward pass - not used in VAE training."""
        raise NotImplementedError(
            "VAE network doesn't use forward pass during training"
        )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: Any,  # WanVAE, but using Any to avoid linter issues
    transformer: Optional[Any] = None,
    **kwargs,
) -> VaeWanNetwork:
    """Create VAE network for training."""

    # Parse training mode from network_args if provided
    training_mode = kwargs.get("training_mode", "full")

    logger.info(f"Creating VAE network with training_mode={training_mode}")

    network = VaeWanNetwork(
        vae=vae,
        multiplier=multiplier,
        training_mode=training_mode,
        **kwargs,
    )

    return network


def create_network_from_weights(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: Any,  # WanVAE, but using Any to avoid linter issues
    weights_sd: Optional[Dict[str, torch.Tensor]] = None,
    transformer: Optional[Any] = None,
    **kwargs,
) -> VaeWanNetwork:
    """Create VAE network and load weights."""

    network = create_network(
        multiplier, network_dim, network_alpha, vae, transformer, **kwargs
    )

    if weights_sd is not None:
        # Load VAE weights
        missing_keys, unexpected_keys = network.vae.load_state_dict(
            weights_sd, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading VAE weights: {missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading VAE weights: {unexpected_keys}"
            )

    return network
