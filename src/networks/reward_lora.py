## Based on https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1_fun/train_reward_lora.py (Apache)

"""Reward LoRA network for WAN.

This module intentionally reuses the standard LoRA implementation for WAN
(`lora_wan.LoRANetwork`) because the architectural changes for Reward LoRA
are in the training loop (reward-based objective), not in the adapter blocks
themselves. We provide a thin wrapper class and factory functions to align
with the project's network loading conventions.

Factory functions exported:
- create_arch_network
- create_arch_network_from_weights
- create_network_from_weights

These mirror the signatures used across Takenoko for consistency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.logger import get_logger

# Reuse the base WAN LoRA implementation
from .lora_wan import (
    LoRANetwork,
    LoRAModule,
    LoRAInfModule,
    create_arch_network as _base_create_arch_network,
    create_arch_network_from_weights as _base_create_arch_network_from_weights,
    create_network_from_weights as _base_create_network_from_weights,
    WAN_TARGET_REPLACE_MODULES,
)


logger = get_logger(__name__)


class RewardLoRANetwork(LoRANetwork):
    """Thin wrapper around LoRANetwork for naming and future extensions."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Optional[List[nn.Module]],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: type = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            target_replace_modules,
            prefix,
            text_encoders,  # type: ignore[arg-type]
            unet,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            conv_lora_dim,
            conv_alpha,
            module_class,
            modules_dim,
            modules_alpha,
            exclude_patterns,
            include_patterns,
            verbose,
        )

    # No behavior change; we keep this class for clearer logs and future tweaks.


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: Optional[nn.Module],
    text_encoders: Optional[List[nn.Module]],
    transformer: nn.Module,
    neuron_dropout: Optional[float] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> RewardLoRANetwork:
    """Create a RewardLoRANetwork with standard WAN target modules.

    Parameters
    - multiplier: float — LoRA scale multiplier
    - network_dim: Optional[int] — LoRA rank
    - network_alpha: Optional[float] — LoRA alpha
    - vae: unused (kept for interface compatibility)
    - text_encoders: Optional[List[nn.Module]] — not applied for WAN
    - transformer: nn.Module — the WAN DiT model to adapt
    - neuron_dropout: Optional[float] — LoRA neuron dropout
    - verbose: bool — log extra info
    - kwargs: Any — propagated for compatibility
    """
    _ = vae  # unused - kept for signature compatibility

    lora_dim = int(network_dim) if network_dim is not None else 4
    alpha = float(network_alpha) if network_alpha is not None else float(lora_dim)

    # Use the base factory to ensure target module discovery remains unified.
    # Ensure text_encoders is a list for the base factory typing
    te_list: List[nn.Module] = (
        list(text_encoders) if isinstance(text_encoders, list) else []
    )

    base_net: LoRANetwork = _base_create_arch_network(
        multiplier,
        lora_dim,
        alpha,
        vae,  # type: ignore[arg-type]
        te_list,
        transformer,
        neuron_dropout=neuron_dropout,
        verbose=verbose,
        **kwargs,
    )

    # Re-wrap to label as RewardLoRANetwork for clearer logs
    reward_net = RewardLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix=base_net.prefix,
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=transformer,
        multiplier=multiplier,
        lora_dim=lora_dim,
        alpha=alpha,
        dropout=neuron_dropout,
        rank_dropout=getattr(base_net, "rank_dropout", None),
        module_dropout=getattr(base_net, "module_dropout", None),
        conv_lora_dim=getattr(base_net, "conv_lora_dim", None),
        conv_alpha=getattr(base_net, "conv_alpha", None),
        modules_dim=getattr(base_net, "modules_dim", None),
        modules_alpha=getattr(base_net, "modules_alpha", None),
        exclude_patterns=getattr(base_net, "exclude_patterns", None),
        include_patterns=getattr(base_net, "include_patterns", None),
        verbose=verbose,
    )

    # Copy created submodules from base network
    for name, module in base_net.named_children():
        reward_net.add_module(name, module)

    logger.info(
        "Created RewardLoRANetwork (WAN) with rank=%s alpha=%s", lora_dim, alpha
    )
    return reward_net


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> Tuple[RewardLoRANetwork, Dict[str, Any]]:
    """Create RewardLoRANetwork from weights (dim/alpha inferred from state dict)."""
    # Ensure text_encoders is a list for the base factory typing
    te_list: List[nn.Module] = (
        list(text_encoders) if isinstance(text_encoders, list) else []
    )

    base_net: LoRANetwork = _base_create_arch_network_from_weights(
        multiplier,
        weights_sd,
        text_encoders=te_list,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )

    # Wrap base into RewardLoRANetwork for consistent type naming
    reward_net = RewardLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix=base_net.prefix,
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet if isinstance(unet, nn.Module) else None,  # type: ignore[arg-type]
        multiplier=multiplier,
        lora_dim=getattr(base_net, "lora_dim", 4),
        alpha=getattr(base_net, "alpha", 1.0),
        dropout=getattr(base_net, "dropout", None),
        rank_dropout=getattr(base_net, "rank_dropout", None),
        module_dropout=getattr(base_net, "module_dropout", None),
        conv_lora_dim=getattr(base_net, "conv_lora_dim", None),
        conv_alpha=getattr(base_net, "conv_alpha", None),
        modules_dim=getattr(base_net, "modules_dim", None),
        modules_alpha=getattr(base_net, "modules_alpha", None),
        exclude_patterns=getattr(base_net, "exclude_patterns", None),
        include_patterns=getattr(base_net, "include_patterns", None),
        verbose=getattr(base_net, "verbose", False),
    )
    for name, module in base_net.named_children():
        reward_net.add_module(name, module)

    info: Dict[str, Any] = {"inferred_dim_alpha": True}
    return reward_net, info


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> RewardLoRANetwork:
    base_net: LoRANetwork = _base_create_network_from_weights(
        target_replace_modules,
        multiplier,
        weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )

    reward_net = RewardLoRANetwork(
        target_replace_modules=target_replace_modules,
        prefix=base_net.prefix,
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet if isinstance(unet, nn.Module) else None,  # type: ignore[arg-type]
        multiplier=multiplier,
        lora_dim=getattr(base_net, "lora_dim", 4),
        alpha=getattr(base_net, "alpha", 1.0),
        dropout=getattr(base_net, "dropout", None),
        rank_dropout=getattr(base_net, "rank_dropout", None),
        module_dropout=getattr(base_net, "module_dropout", None),
        conv_lora_dim=getattr(base_net, "conv_lora_dim", None),
        conv_alpha=getattr(base_net, "conv_alpha", None),
        modules_dim=getattr(base_net, "modules_dim", None),
        modules_alpha=getattr(base_net, "modules_alpha", None),
        exclude_patterns=getattr(base_net, "exclude_patterns", None),
        include_patterns=getattr(base_net, "include_patterns", None),
        verbose=getattr(base_net, "verbose", False),
    )
    for name, module in base_net.named_children():
        reward_net.add_module(name, module)
    return reward_net
