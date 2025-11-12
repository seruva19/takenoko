"""Core helper functions for Differential Guidance enhancement.

This module provides the target transformation logic that amplifies the
difference between model predictions and training targets.
"""

import torch
from typing import Optional, Dict

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def apply_differential_guidance(
    target: torch.Tensor,
    model_pred: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply differential guidance to training target.

    Computes: new_target = model_pred + scale * (target - model_pred)

    This amplifies the difference between prediction and target,
    causing the model to "overshoot" and potentially converge faster.

    Args:
        target: Original training target tensor
        model_pred: Model's prediction tensor
        scale: Amplification factor for the difference

    Returns:
        Transformed target tensor

    Raises:
        ValueError: If tensors have mismatched shapes
    """
    if target.shape != model_pred.shape:
        raise ValueError(
            f"Target shape {target.shape} must match model_pred shape {model_pred.shape}"
        )

    with torch.no_grad():
        difference = target - model_pred
        guided_target = model_pred + scale * difference

    return guided_target


def compute_guidance_metrics(
    target: torch.Tensor,
    model_pred: torch.Tensor,
    guided_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics for TensorBoard logging.

    Args:
        target: Original training target
        model_pred: Model's prediction
        guided_target: Transformed target after guidance

    Returns:
        Dictionary of metric names to values
    """
    with torch.no_grad():
        # Compute difference magnitudes
        original_diff = target - model_pred
        guidance_diff = guided_target - target

        metrics = {
            "differential_guidance/mean_original_diff": original_diff.abs()
            .mean()
            .item(),
            "differential_guidance/mean_guidance_diff": guidance_diff.abs()
            .mean()
            .item(),
            "differential_guidance/target_mean": target.mean().item(),
            "differential_guidance/guided_target_mean": guided_target.mean().item(),
            "differential_guidance/target_std": target.std().item(),
            "differential_guidance/guided_target_std": guided_target.std().item(),
        }

    return metrics
