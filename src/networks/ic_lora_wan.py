"""IC-LoRA WAN network module.

This module provides a dedicated `network_module` entrypoint for IC-LoRA
experiments while preserving baseline LoRA behavior. It intentionally wraps
`networks.lora_wan` so existing training behavior remains unchanged until
IC-LoRA-specific logic is added behind explicit config flags.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from networks.lora_wan import (
    LoRAInfModule,
    LoRAModule,
    LoRANetwork,
    WAN_TARGET_REPLACE_MODULES,
    create_arch_network as _base_create_arch_network,
    create_arch_network_from_weights as _base_create_arch_network_from_weights,
    create_network as _base_create_network,
    create_network_from_weights as _base_create_network_from_weights,
)


def _mark_ic_lora(network: LoRANetwork) -> LoRANetwork:
    """Attach a marker so downstream code can detect IC-LoRA modules safely."""
    setattr(network, "is_ic_lora_network", True)
    return network


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRANetwork:
    network = _base_create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    return _mark_ic_lora(network)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRANetwork:
    network = _base_create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    return _mark_ic_lora(network)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    network = _base_create_arch_network_from_weights(
        multiplier,
        weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )
    return _mark_ic_lora(network)


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    network = _base_create_network_from_weights(
        target_replace_modules,
        multiplier,
        weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )
    return _mark_ic_lora(network)

