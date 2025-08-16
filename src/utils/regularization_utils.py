"""
Regularization utilities for training with regularization datasets.

This module provides utilities for handling regularization datasets and loss weighting
in the training process. It follows the battle-tested implementation from kohya-ss/sd-scripts.
"""

import logging
from typing import Dict, Any, Optional
import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def apply_regularization_weights(
    loss: torch.Tensor, batch: Dict[str, Any], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Apply regularization weights to the loss tensor.

    Args:
        loss: The computed loss tensor
        batch: The batch dictionary containing weights
        device: Target device for the weights
        dtype: Target dtype for the weights

    Returns:
        The weighted loss tensor
    """
    # Get the per-sample weights from the batch
    sample_weights = batch.get("weight", None)

    if sample_weights is not None:
        sample_weights = sample_weights.to(device=device, dtype=dtype)

        # Reshape weights to broadcast with loss
        while sample_weights.dim() < loss.dim():
            sample_weights = sample_weights.unsqueeze(-1)

        # Apply weights to loss
        loss = loss * sample_weights

        # Log regularization usage if weights are not all 1.0
        if torch.any(sample_weights != 1.0):
            reg_count = torch.sum(sample_weights != 1.0).item()
            total_count = sample_weights.numel()
            logger.debug(
                f"Applied regularization weights to {reg_count}/{total_count} samples"
            )

    return loss


def log_regularization_info(dataset_group: Any) -> None:
    """
    Log information about regularization datasets in the dataset group.

    Args:
        dataset_group: The dataset group to analyze
    """
    if not hasattr(dataset_group, "datasets"):
        return

    reg_datasets = []
    normal_datasets = []

    for dataset in dataset_group.datasets:
        if hasattr(dataset, "is_reg") and dataset.is_reg:
            reg_datasets.append(dataset)
        else:
            normal_datasets.append(dataset)

    if reg_datasets:
        logger.info("📊 Regularization Dataset Configuration:")
        logger.info(f"   🎯 Regularization datasets: {len(reg_datasets)}")
        logger.info(f"   📚 Normal datasets: {len(normal_datasets)}")

        for i, dataset in enumerate(reg_datasets):
            dataset_id = getattr(
                dataset, "get_dataset_identifier", lambda: f"Dataset_{i}"
            )()
            logger.info(f"      • {dataset_id} (regularization)")

        for i, dataset in enumerate(normal_datasets):
            dataset_id = getattr(
                dataset, "get_dataset_identifier", lambda: f"Dataset_{i}"
            )()
            logger.info(f"      • {dataset_id} (normal)")
    else:
        logger.info("📊 No regularization datasets found - using standard training")


def validate_regularization_config(args: Any) -> None:
    """
    Validate regularization configuration.

    Args:
        args: Training arguments containing regularization settings
    """
    prior_loss_weight = getattr(args, "prior_loss_weight", 1.0)

    if prior_loss_weight != 1.0:
        logger.info(f"🔧 Regularization weight: {prior_loss_weight}")

        if prior_loss_weight < 0:
            logger.warning(
                "⚠️  Negative regularization weight detected - this may cause training instability"
            )
        elif prior_loss_weight > 10:
            logger.warning(
                "⚠️  High regularization weight detected - this may cause over-regularization"
            )
    else:
        logger.info("🔧 Using default regularization weight (1.0)")
