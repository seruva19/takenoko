"""Helpers for ReLoRA merge/reset operations."""

from __future__ import annotations

from functools import partial
from typing import Iterable, List, Optional, Tuple

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


def _is_deepspeed_zero_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    name = optimizer.__class__.__name__
    inner = getattr(optimizer, "optimizer", None)
    inner_name = inner.__class__.__name__ if inner is not None else ""
    return "DeepSpeedZero" in name or "DeepSpeedZero" in inner_name


def _deepspeed_zero_reset(
    optimizer: torch.optim.Optimizer,
    *,
    named_reset_params: List[Tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: List[str],
    reset_prune_ratio: float,
) -> Optional[float]:
    ds_optimizer = getattr(optimizer, "optimizer", None) or optimizer
    if not hasattr(ds_optimizer, "params_in_partition"):
        return None
    if not hasattr(ds_optimizer, "_param_slice_mappings"):
        return None
    if not ds_optimizer.state:
        return None

    state_dict = next(iter(ds_optimizer.state.values()))
    params_in_partition = set(ds_optimizer.params_in_partition[0])
    slice_mappings = ds_optimizer._param_slice_mappings[0]

    reset_prune_ratio = float(reset_prune_ratio or 0.0)
    non_zero_sum = 0
    zeroed = 0

    for key in optimizer_state_keys:
        state_tensor = state_dict.get(key)
        if not isinstance(state_tensor, torch.Tensor):
            continue
        for name, param in named_reset_params:
            if param not in params_in_partition:
                continue
            fixed_name = name.split(".module.")[-1]
            param_slice_map = slice_mappings.get(fixed_name)
            if param_slice_map is None:
                continue
            if isinstance(param_slice_map, slice):
                param_slice = param_slice_map
                param_size = param_slice.stop - param_slice.start
            else:
                if not hasattr(param_slice_map, "start") or not hasattr(
                    param_slice_map, "numel"
                ):
                    continue
                param_size = param_slice_map.numel
                param_slice = slice(
                    param_slice_map.start, param_slice_map.start + param_size
                )

            before = torch.count_nonzero(state_tensor[param_slice]).item()
            if reset_prune_ratio > 0.0:
                mask = (
                    torch.rand(param_size, device=state_tensor.device)
                    > reset_prune_ratio
                )
                state_tensor[param_slice].mul_(mask)
            else:
                state_tensor[param_slice].zero_()
            after = torch.count_nonzero(state_tensor[param_slice]).item()
            non_zero_sum += before
            zeroed += before - after

    if non_zero_sum <= 0:
        return None
    return zeroed / non_zero_sum


def optimizer_reset(
    optimizer: torch.optim.Optimizer,
    *,
    reset_params: Iterable[torch.nn.Parameter],
    optimizer_state_keys: List[str],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
    reset_prune_ratio: float,
    enable_deepspeed_zero_reset: bool = False,
    named_reset_params: Optional[List[Tuple[str, torch.nn.Parameter]]] = None,
) -> Optional[float]:
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

    zero_reset_prune_ratio = float(reset_prune_ratio or 0.0)
    if reset_optimizer_on_relora:
        if zero_reset_prune_ratio > 0.0:
            logger.info(
                "ReLoRA: resetting optimizer states with random pruning (%s)",
                zero_reset_prune_ratio,
            )
            pruning_fn = partial(random_pruning_, prune_ratio=zero_reset_prune_ratio)
        else:
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

    if (
        reset_optimizer_on_relora
        and enable_deepspeed_zero_reset
        and named_reset_params
        and _is_deepspeed_zero_optimizer(optimizer)
    ):
        zeroed_ratio = _deepspeed_zero_reset(
            optimizer,
            named_reset_params=named_reset_params,
            optimizer_state_keys=optimizer_state_keys,
            reset_prune_ratio=zero_reset_prune_ratio,
        )
        if zeroed_ratio is not None:
            logger.info("ReLoRA: optimizer states zeroed %.2f%%", zeroed_ratio * 100.0)
        return zeroed_ratio

    optimizer_state = optimizer.state
    is_zero = isinstance(optimizer, ZeroRedundancyOptimizer)
    if is_zero:
        optimizer_state = optimizer.optim.state

    non_zero_sum = 0
    zeroed = 0
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
            non_zero_before = torch.count_nonzero(state_tensor).item()
            if reset_optimizer_on_relora:
                if is_zero and zero_reset_prune_ratio <= 0.0:
                    random_pruning_(state_tensor, prune_ratio=0.999)
                elif pruning_fn is None:
                    state_tensor.zero_()
                else:
                    pruning_fn(state_tensor)
            else:
                assert pruning_fn is not None
                pruning_fn(state_tensor)
            non_zero_after = torch.count_nonzero(state_tensor).item()
            non_zero_sum += non_zero_before
            zeroed += non_zero_before - non_zero_after

    if non_zero_sum > 0:
        zeroed_ratio = zeroed / non_zero_sum
        logger.info("ReLoRA: optimizer states zeroed %.2f%%", zeroed_ratio * 100.0)
        return zeroed_ratio
    return None
