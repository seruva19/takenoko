"""Scion optimizer creation helpers for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch


def create_scion_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using Scion optimizer | {optimizer_kwargs}")

    from optimizers.scion import Scion

    # Default parameters
    momentum = optimizer_kwargs.get("momentum", 0.1)
    scale = optimizer_kwargs.get("scale", 1.0)
    norm = optimizer_kwargs.get("norm", "Auto")
    norm_kwargs = optimizer_kwargs.get("norm_kwargs", {})
    unconstrained = optimizer_kwargs.get("unconstrained", False)

    # Check if user provided parameter groups with different norms
    # If trainable_params is already a list of dicts with 'norm' key, use it directly
    if (
        isinstance(trainable_params, list)
        and len(trainable_params) > 0
        and isinstance(trainable_params[0], dict)
        and "norm" in trainable_params[0]
    ):
        logger.info("Using custom parameter groups for Scion")
        optimizer_class = Scion
        optimizer = optimizer_class(trainable_params, lr=lr, momentum=momentum)
    else:
        # Single parameter group with specified norm
        logger.info(
            f"Scion config: norm={norm}, scale={scale}, momentum={momentum}, unconstrained={unconstrained}"
        )
        optimizer_class = Scion
        optimizer = optimizer_class(
            trainable_params,
            lr=lr,
            momentum=momentum,
            norm=norm,
            norm_kwargs=norm_kwargs,
            scale=scale,
            unconstrained=unconstrained,
        )

    return optimizer_class, optimizer


def create_scionlight_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using ScionLight optimizer (memory-efficient) | {optimizer_kwargs}")

    from optimizers.scion import ScionLight

    # Default parameters
    momentum = optimizer_kwargs.get("momentum", 0.1)
    scale = optimizer_kwargs.get("scale", 1.0)
    norm = optimizer_kwargs.get("norm", "Auto")
    norm_kwargs = optimizer_kwargs.get("norm_kwargs", {})
    unconstrained = optimizer_kwargs.get("unconstrained", False)

    # Check if user provided parameter groups with different norms
    if (
        isinstance(trainable_params, list)
        and len(trainable_params) > 0
        and isinstance(trainable_params[0], dict)
        and "norm" in trainable_params[0]
    ):
        logger.info("Using custom parameter groups for ScionLight")
        optimizer_class = ScionLight
        optimizer = optimizer_class(trainable_params, lr=lr, momentum=momentum)
    else:
        # Single parameter group with specified norm
        logger.info(
            f"ScionLight config: norm={norm}, scale={scale}, momentum={momentum}, unconstrained={unconstrained}"
        )
        optimizer_class = ScionLight
        optimizer = optimizer_class(
            trainable_params,
            lr=lr,
            momentum=momentum,
            norm=norm,
            norm_kwargs=norm_kwargs,
            scale=scale,
            unconstrained=unconstrained,
        )

    return optimizer_class, optimizer
