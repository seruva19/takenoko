"""Universal safetensors loader that can be optimized or standard based on config."""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch
import safetensors
import safetensors.torch

# This module provides drop-in replacements for safetensors functions
# The behavior switches based on the global memory optimization config


# Global reference to the memory manager (set during initialization)
_memory_manager: Optional[Any] = None


def initialize_loader(memory_manager: Optional[Any] = None) -> None:
    """Initialize the global safetensors loader with optional memory manager."""
    global _memory_manager
    _memory_manager = memory_manager


def load_file(
    path: Union[str, Path], device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Drop-in replacement for safetensors.torch.load_file()

    Uses optimized loading if memory manager is enabled, otherwise standard loading.
    This function signature exactly matches safetensors.torch.load_file()
    """
    global _memory_manager

    if (
        _memory_manager
        and hasattr(_memory_manager, "enabled")
        and _memory_manager.enabled
        and hasattr(_memory_manager, "optimized_loading_enabled")
        and _memory_manager.optimized_loading_enabled
    ):
        # Use optimized loading
        return _memory_manager.load_safetensors_optimized(path, device)
    else:
        # Use standard loading - exact same behavior as safetensors.torch.load_file
        if device is not None:
            return safetensors.torch.load_file(path, device=device)
        else:
            return safetensors.torch.load_file(path)


def safe_open(
    path: Union[str, Path], framework: str = "pt", device: Optional[str] = None
):
    """
    Drop-in replacement for safe_open()

    For now, this passes through to standard safe_open since our optimization
    is focused on load_file(). Could be extended in the future.
    """
    if device is not None:
        return safetensors.safe_open(path, framework=framework, device=device)
    else:
        return safetensors.safe_open(path, framework=framework)


# For compatibility, also expose the torch submodule functions
class TorchModule:
    """Torch submodule replacement for safetensors.torch"""

    @staticmethod
    def load_file(
        path: Union[str, Path], device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Replacement for safetensors.torch.load_file"""
        return load_file(path, device)

    @staticmethod
    def save_file(
        tensors: Dict[str, torch.Tensor],
        filename: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Passthrough to safetensors.torch.save_file - no optimization needed for saving"""
        return safetensors.torch.save_file(tensors, filename, metadata)


torch = TorchModule()
