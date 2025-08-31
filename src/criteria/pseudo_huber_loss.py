"""Pseudo-Huber Loss Implementation for Takenoko

This module provides a gated, optional, non-intrusive integration of Pseudo-Huber loss
into the Takenoko training pipeline.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def pseudo_huber_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    huber_c: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Pseudo-Huber loss.

    The Pseudo-Huber loss is defined as:
    L(x, y) = c * (sqrt((x - y)^2 + c^2) - c)

    This is a smooth approximation to the Huber loss that is differentiable everywhere.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        huber_c: Huber parameter c (controls the transition point)
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Computed loss tensor
    """
    # Ensure tensors are in float32 for numerical stability
    model_pred = model_pred.float()
    target = target.float()

    # Compute squared difference
    diff_sq = (model_pred - target) ** 2

    # Pseudo-Huber formula: c * (sqrt(diff^2 + c^2) - c)
    loss = huber_c * (torch.sqrt(diff_sq + huber_c**2) - huber_c)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def scheduled_pseudo_huber_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    huber_c: float = 0.5,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    schedule_type: str = "linear",
    c_min: float = 0.1,
    c_max: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Scheduled Pseudo-Huber loss with adaptive c parameter.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        huber_c: Base Huber parameter c (used if scheduling is disabled)
        current_step: Current training step (for scheduling)
        total_steps: Total training steps (for scheduling)
        schedule_type: Scheduling type ("linear", "cosine", "exponential")
        c_min: Minimum c value for scheduling
        c_max: Maximum c value for scheduling
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Computed loss tensor
    """
    # Use scheduled c if step information is provided
    if current_step is not None and total_steps is not None and total_steps > 0:
        progress = min(current_step / total_steps, 1.0)

        if schedule_type == "linear":
            # Linear decay from c_max to c_min
            scheduled_c = c_max - progress * (c_max - c_min)
        elif schedule_type == "cosine":
            # Cosine decay from c_max to c_min
            scheduled_c = c_min + 0.5 * (c_max - c_min) * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )
            scheduled_c = float(scheduled_c)
        elif schedule_type == "exponential":
            # Exponential decay from c_max to c_min
            decay_rate = -torch.log(torch.tensor(c_min / c_max))
            scheduled_c = c_max * torch.exp(-decay_rate * progress)
            scheduled_c = float(scheduled_c)
        else:
            logger.warning(
                f"Unknown schedule type '{schedule_type}', using base huber_c"
            )
            scheduled_c = huber_c
    else:
        scheduled_c = huber_c

    return pseudo_huber_loss(model_pred, target, scheduled_c, reduction)


def conditional_loss_with_pseudo_huber(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_c: float = 0.5,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    schedule_type: str = "linear",
    c_min: float = 0.1,
    c_max: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Conditional loss function supporting Pseudo-Huber loss.

    This function provides a drop-in replacement for standard loss functions
    with optional Pseudo-Huber loss support.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        loss_type: Loss type ("mse", "l1", "huber", "pseudo_huber", "pseudo_huber_scheduled")
        huber_c: Huber parameter c
        current_step: Current training step (for scheduled variants)
        total_steps: Total training steps (for scheduled variants)
        schedule_type: Scheduling type for scheduled variants
        c_min: Minimum c value for scheduling
        c_max: Maximum c value for scheduling
        reduction: Reduction method

    Returns:
        Computed loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        return F.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        return F.smooth_l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "pure_huber":
        return F.huber_loss(
            model_pred.float(), target.float(), reduction=reduction, delta=huber_c
        )
    elif loss_type == "pseudo_huber":
        return pseudo_huber_loss(model_pred, target, huber_c, reduction)
    elif loss_type == "pseudo_huber_scheduled":
        return scheduled_pseudo_huber_loss(
            model_pred,
            target,
            huber_c,
            current_step,
            total_steps,
            schedule_type,
            c_min,
            c_max,
            reduction,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def log_loss_type_info(args) -> None:
    """Log loss type information at training start.

    Args:
        args: Training arguments containing loss configuration
    """
    loss_type = getattr(args, "loss_type", "mse")
    if loss_type == "pseudo_huber":
        pseudo_huber_c_min = getattr(args, "pseudo_huber_c_min", 0.01)
        pseudo_huber_c_max = getattr(args, "pseudo_huber_c_max", 1.0)
        schedule_type = getattr(args, "pseudo_huber_schedule_type", "linear")
        logger.info(
            f"📊 Loss function: Pseudo-Huber (c_min={pseudo_huber_c_min}, c_max={pseudo_huber_c_max}, schedule={schedule_type})"
        )
    else:
        logger.info(f"📊 Loss function: {loss_type.upper()}")
