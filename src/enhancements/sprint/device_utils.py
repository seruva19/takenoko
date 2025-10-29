"""
Device validation utilities for Sprint module.

This module provides device compatibility checks, memory estimation,
and safe device operations for Sprint components.
"""

import gc
import warnings
from typing import Dict, Optional, Any, Tuple
import torch

from .exceptions import SprintDeviceError, SprintMemoryError


def validate_device_consistency(tensors: Dict[str, torch.Tensor],
                              expected_device: Optional[torch.device] = None) -> torch.device:
    """
    Validate that all tensors are on the same device.

    Args:
        tensors: Dictionary of tensor names to tensor objects
        expected_device: Expected device (if None, uses first tensor's device)

    Returns:
        The common device all tensors should be on

    Raises:
        SprintDeviceError: If tensors are on different devices
    """
    if not tensors:
        return torch.device('cpu')

    # Determine expected device
    if expected_device is None:
        # Use first tensor's device as reference
        first_tensor = next(iter(tensors.values()))
        expected_device = first_tensor.device

    # Validate all tensors
    for name, tensor in tensors.items():
        if tensor.device != expected_device:
            raise SprintDeviceError(
                f"Device mismatch: {name} on {tensor.device}, expected {expected_device}",
                device=str(expected_device),
                tensor_name=name
            )

    return expected_device


def safe_to_device(tensor: torch.Tensor, device: torch.device,
                   name: str = "tensor") -> torch.Tensor:
    """
    Safely move tensor to device with validation.

    Args:
        tensor: Tensor to move
        device: Target device
        name: Tensor name for error messages

    Returns:
        Tensor on target device

    Raises:
        SprintDeviceError: If device transfer fails
    """
    if tensor.device == device:
        return tensor

    try:
        return tensor.to(device=device, non_blocking=True)
    except Exception as e:
        raise SprintDeviceError(
            f"Failed to move {name} from {tensor.device} to {device}: {e}",
            device=str(device),
            tensor_name=name
        ) from e


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, int]:
    """
    Get GPU memory information.

    Args:
        device: Device to check (if None, uses current CUDA device)

    Returns:
        Dictionary with memory info in MB
    """
    if not torch.cuda.is_available():
        return {'total': 0, 'free': 0, 'used': 0}

    if device is None:
        device = torch.device('cuda')

    if not device.type == 'cuda':
        return {'total': 0, 'free': 0, 'used': 0}

    device_idx = device.index if device.index is not None else torch.cuda.current_device()

    try:
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_idx)
        allocated_memory = torch.cuda.memory_allocated(device_idx)

        free_memory = total_memory - reserved_memory
        used_memory = allocated_memory

        return {
            'total': total_memory // (1024 * 1024),  # Convert to MB
            'free': free_memory // (1024 * 1024),
            'used': used_memory // (1024 * 1024)
        }
    except Exception as e:
        warnings.warn(f"Failed to get GPU memory info: {e}")
        return {'total': 0, 'free': 0, 'used': 0}


def estimate_sprint_memory_usage(config: Dict[str, Any],
                                model_config: Dict[str, Any]) -> Dict[str, int]:
    """
    Estimate memory usage for Sprint configuration.

    Args:
        config: Sprint configuration
        model_config: Model configuration

    Returns:
        Dictionary with memory estimates in MB
    """
    # Base model memory (rough estimate)
    hidden_size = model_config.get('hidden_size', 1024)
    num_blocks = model_config.get('num_blocks', 24)

    # Sprint fusion memory overhead
    encoder_layers = config.get('encoder_layers', 6)
    middle_layers = config.get('middle_layers', 6)

    # Estimate memory for fusion projections
    fusion_memory_mb = (encoder_layers + middle_layers) * hidden_size * hidden_size * 4 // (1024 * 1024)

    # Estimate memory for token sampling
    sampling_memory_mb = 50  # Rough estimate for sampling buffers

    # Estimate activation memory reduction
    reduction_ratio = config.get('token_drop_ratio', 0.5)
    sequence_length = model_config.get('max_sequence_length', 4096)
    batch_size = model_config.get('batch_size', 1)

    activation_memory_saved_mb = (
        num_blocks * hidden_size * sequence_length * batch_size * reduction_ratio * 4 // (1024 * 1024)
    )

    return {
        'fusion_overhead_mb': fusion_memory_mb,
        'sampling_overhead_mb': sampling_memory_mb,
        'activation_memory_saved_mb': activation_memory_saved_mb,
        'total_overhead_mb': fusion_memory_mb + sampling_memory_mb,
        'net_memory_change_mb': fusion_memory_mb + sampling_memory_mb - activation_memory_saved_mb
    }


def validate_sprint_memory_requirements(config: Dict[str, Any],
                                     model_config: Dict[str, Any],
                                     device: Optional[torch.device] = None) -> None:
    """
    Validate that sufficient memory is available for Sprint.

    Args:
        config: Sprint configuration
        model_config: Model configuration
        device: Device to check memory on

    Raises:
        SprintMemoryError: If insufficient memory
    """
    if not torch.cuda.is_available() or (device and device.type != 'cuda'):
        # Can't validate memory for CPU/CPU offload cases
        return

    memory_info = get_gpu_memory_info(device)
    memory_estimate = estimate_sprint_memory_usage(config, model_config)

    required_mb = memory_estimate['total_overhead_mb']
    available_mb = memory_info['free']

    # Use 80% of available memory as safety margin
    safe_available_mb = int(available_mb * 0.8)

    if required_mb > safe_available_mb:
        raise SprintMemoryError(
            f"Insufficient GPU memory for Sprint. Required: ~{required_mb}MB, "
            f"Available: ~{available_mb}MB (safe: ~{safe_available_mb}MB)",
            required_mb=required_mb,
            available_mb=available_mb
        )

    # Warning if memory usage will be high
    if required_mb > safe_available_mb * 0.7:
        warnings.warn(
            f"Sprint will use significant memory (~{required_mb}MB of {available_mb}MB available). "
            "Consider reducing token_drop_ratio or using gradient checkpointing."
        )


def validate_sprint_device_compatibility(device: torch.device) -> None:
    """
    Validate device compatibility with Sprint requirements.

    Args:
        device: Device to validate

    Raises:
        SprintDeviceError: If device is incompatible
    """
    # Sprint requires CUDA for performance
    if device.type == 'cpu':
        warnings.warn(
            "Sprint on CPU may be very slow. Consider using CUDA for better performance."
        )

    # Check CUDA availability
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SprintDeviceError(
            f"CUDA requested but not available. Device: {device}",
            device=str(device)
        )

    # Check specific CUDA device
    if device.type == 'cuda':
        try:
            device_idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.get_device_properties(device_idx)
        except Exception as e:
            raise SprintDeviceError(
                f"Invalid CUDA device {device}: {e}",
                device=str(device)
            ) from e


def optimize_memory_for_sprint(device: Optional[torch.device] = None) -> None:
    """
    Optimize memory usage before Sprint operations.

    Args:
        device: Device to optimize
    """
    if device and device.type == 'cuda':
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_tensor_device_safety(tensor: torch.Tensor,
                             expected_device: torch.device,
                             operation: str = "operation") -> None:
    """
    Check if tensor is safe for the current operation.

    Args:
        tensor: Tensor to check
        expected_device: Expected device
        operation: Operation description for error messages

    Raises:
        SprintDeviceError: If tensor device is unsafe
    """
    if tensor.device != expected_device:
        raise SprintDeviceError(
            f"Tensor device mismatch during {operation}. "
            f"Expected: {expected_device}, Got: {tensor.device}",
            device=str(expected_device),
            tensor_name="input_tensor"
        )

    # Check for NaN/Inf if it's a floating point tensor
    if tensor.is_floating_point():
        if torch.isnan(tensor).any():
            raise SprintDeviceError(
                f"NaN detected in tensor during {operation}",
                device=str(expected_device),
                tensor_name="input_tensor"
            )
        if torch.isinf(tensor).any():
            raise SprintDeviceError(
                f"Inf detected in tensor during {operation}",
                device=str(expected_device),
                tensor_name="input_tensor"
            )