"""Helpers for ReLoRA merge/reset operations."""

from __future__ import annotations

from functools import partial
from typing import Iterable, List

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer

from common.logger import get_logger


logger = get_logger(__name__)


@torch.no_grad()
def random_pruning_(tensor: torch.Tensor, prune_ratio: float) -> None:
    """Randomly zero elements in-place according to prune_ratio."""
    if prune_ratio <= 0.0:
        return
    mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(mask)


@torch.no_grad()
def magnitude_pruning_(tensor: torch.Tensor, prune_ratio: float) -> None:
    """Zero elements by magnitude threshold in-place according to prune_ratio."""
    if prune_ratio <= 0.0:
        return
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(
        tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio
    ).to(dtype=tensor.dtype)
    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


def optimizer_reset(
    optimizer: torch.optim.Optimizer,
    *,
    reset_params: Iterable[torch.nn.Parameter],
    optimizer_state_keys: List[str],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
) -> None:
    """Reset or prune optimizer states for selected parameters."""
    n_reset_types = (
        int(bool(reset_optimizer_on_relora))
        + int(bool(optimizer_random_pruning))
        + int(bool(optimizer_magnitude_pruning))
    )
    if n_reset_types != 1:
        raise ValueError(
            "Exactly one of reset_optimizer_on_relora, optimizer_random_pruning, "
            "optimizer_magnitude_pruning must be set"
        )

    if reset_optimizer_on_relora:
        logger.info("ReLoRA: resetting optimizer states")
        pruning_fn = None
    elif optimizer_random_pruning:
        logger.info(
            "ReLoRA: random pruning optimizer states (%s)",
            optimizer_random_pruning,
        )
        pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
    else:
        logger.info(
            "ReLoRA: magnitude pruning optimizer states (%s)",
            optimizer_magnitude_pruning,
        )
        pruning_fn = partial(
            magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning
        )

    optimizer_state = optimizer.state
    is_zero = isinstance(optimizer, ZeroRedundancyOptimizer)
    if is_zero:
        optimizer_state = optimizer.optim.state

    n_zeros = 0
    n_total = 0
    for param in reset_params:
        if param not in optimizer_state:
            continue
        param_state = optimizer_state[param]
        if len(param_state) == 0:
            continue
        for key in optimizer_state_keys:
            if key not in param_state:
                continue
            state_tensor = param_state[key]
            if not isinstance(state_tensor, torch.Tensor):
                continue
            if reset_optimizer_on_relora:
                if is_zero:
                    random_pruning_(state_tensor, prune_ratio=0.999)
                else:
                    state_tensor.zero_()
            else:
                assert pruning_fn is not None
                pruning_fn(state_tensor)
            n_total += state_tensor.numel()
            n_zeros += torch.sum(state_tensor == 0).item()

    if n_total > 0:
        zeroed = n_zeros / (1e-7 + n_total) * 100.0
        logger.info("ReLoRA: optimizer states zeroed %.2f%%", zeroed)
