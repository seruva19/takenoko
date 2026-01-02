"""Bitsandbytes-backed optimizer creation helpers for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch


def create_adamw8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is not installed. Please install bitsandbytes to use 8-bit optimizers."
        )

    logger.info(f"using AdamW8bit optimizer | {optimizer_kwargs}")
    optimizer_class = bnb.optim.AdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_came8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    try:
        from optimizers.sana_optimizer import CAME8BitWrapper

        optimizer_class = CAME8BitWrapper
        logger.info(
            "using CamE8Bit optimizer (SANA implementation) | %s",
            optimizer_kwargs,
        )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        return optimizer_class, optimizer
    except Exception as err:
        logger.warning(
            "⚠️ Failed to import CamE8Bit implementation (%s).",
            err,
        )
        raise ImportError("CamE8Bit implementation could not be used") from err


def create_adamw8bitkahan_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using AdamW8bitKahan optimizer | {optimizer_kwargs}")

    from optimizers.adamw_8bit_kahan import AdamW8bitKahan

    optimizer_class = AdamW8bitKahan
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_temporal_adamw8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using TemporalAdamW8bit optimizer | {optimizer_kwargs}")

    try:
        from optimizers.temporal_adamw_8bit import TemporalAdamW8bit
    except Exception as err:
        raise ImportError(
            "TemporalAdamW8bit requires bitsandbytes. Please install it."
        ) from err

    optimizer_class = TemporalAdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_raven_adamw8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using RavenAdamW8bit optimizer | {optimizer_kwargs}")

    try:
        from optimizers.raven_8bit import RavenAdamW8bit
    except Exception as err:
        raise ImportError(
            "RavenAdamW8bit requires bitsandbytes. Please install it:\n"
            "pip install bitsandbytes"
        ) from err

    optimizer_class = RavenAdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer
