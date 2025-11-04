"""EMA (Exponential Moving Average) utilities for training."""

import argparse
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_ema_beta(beta: float) -> None:
    """Validate EMA beta parameter."""
    if not 0.0 < beta < 1.0:
        raise ValueError("EMA beta must be between 0.0 and 1.0")


def configure_ema_from_args(args: argparse.Namespace) -> Tuple[Optional[float], Optional[int]]:
    """Configure EMA hyperparameters from args if provided.
    
    Returns:
        Tuple of (ema_beta, ema_bias_warmup_steps) or None values if not configured.
    """
    ema_beta = None
    ema_bias_warmup_steps = None
    
    try:
        if hasattr(args, "ema_loss_beta"):
            ema_beta = float(args.ema_loss_beta)
            validate_ema_beta(ema_beta)
        if hasattr(args, "ema_loss_bias_warmup_steps"):
            ema_bias_warmup_steps = int(args.ema_loss_bias_warmup_steps)
    except Exception:
        pass
    
    return ema_beta, ema_bias_warmup_steps


def update_ema_loss(
    current_loss: float,
    ema_loss: Optional[float],
    ema_step_count: int,
    ema_beta: float,
    ema_bias_warmup_steps: int,
    zero_started_ema_loss: Optional[float],
) -> Tuple[float, float, float, int, float]:
    """Update EMA loss with warm-start behaviour and optional warmup gating.
    
    Args:
        current_loss: Current loss value
        ema_loss: Current EMA loss (None for initialization)
        ema_step_count: Current step count
        ema_beta: EMA smoothing factor
        ema_bias_warmup_steps: Steps to defer reporting the EMA
        zero_started_ema_loss: EMA accumulator initialized at zero for bias correction
        
    Returns:
        Tuple of (reported_ema_loss, new_ema_loss, new_zero_started_ema, new_step_count, debiased_ema)
    """
    new_step_count = ema_step_count + 1
    
    if ema_loss is None:
        # Warm-start from current loss to avoid huge initial spike
        new_ema_loss = float(current_loss)
    else:
        new_ema_loss = ema_beta * ema_loss + (1 - ema_beta) * float(current_loss)

    if zero_started_ema_loss is None:
        new_zero_started_ema = (1 - ema_beta) * float(current_loss)
    else:
        new_zero_started_ema = (
            ema_beta * zero_started_ema_loss + (1 - ema_beta) * float(current_loss)
        )

    denominator = 1 - ema_beta**max(new_step_count, 1)
    debiased_ema = (
        float(new_zero_started_ema / denominator) if denominator != 0 else float(current_loss)
    )

    # During warmup we still just report the raw EMA value for the warm-start metric
    reported_ema = float(new_ema_loss)

    return reported_ema, new_ema_loss, new_zero_started_ema, new_step_count, debiased_ema


def update_iter_time_ema(
    last_iter_seconds: float,
    iter_time_ema_sec: Optional[float],
    iter_time_ema_beta: float
) -> float:
    """Update EMA of iteration time in seconds.

    Args:
        last_iter_seconds: Duration of last iteration in seconds (>0)
        iter_time_ema_sec: Current EMA time (None for initialization)
        iter_time_ema_beta: EMA beta for iteration time

    Returns:
        The updated EMA value in seconds.
    """
    if last_iter_seconds <= 0:
        return float(iter_time_ema_sec) if iter_time_ema_sec else 0.0
    
    if iter_time_ema_sec is None:
        return float(last_iter_seconds)
    else:
        return iter_time_ema_beta * float(iter_time_ema_sec) + (1 - iter_time_ema_beta) * float(last_iter_seconds)
