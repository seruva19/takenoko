# TorchAO FP8 Quantization Integration for Takenoko
# This module provides TorchAO-based FP8 quantization as an alternative backend

from typing import Optional, Union, Iterable, Callable, List
import torch
import torch.nn as nn
import logging
import time

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Optional TorchAO imports with graceful fallback
try:
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
    TORCHAO_AVAILABLE = True
    logger.info("🧮 TorchAO available for FP8 quantization")
except ImportError:
    TORCHAO_AVAILABLE = False
    logger.warning("⚠️  TorchAO not available. Enhanced FP8 quantization will fall back to native implementation")


def _make_filter_fn(
    target_layer_keys: Optional[Iterable[str]],
    exclude_layer_keys: Optional[Iterable[str]]
) -> Callable[[nn.Module, str], bool]:
    """
    Create a filter function for TorchAO quantization.

    Quantizes layers whose FQN contains any of target_layer_keys (if not None)
    and does not contain any of exclude_layer_keys (if not None).

    Args:
        target_layer_keys: Keys to include in quantization
        exclude_layer_keys: Keys to exclude from quantization

    Returns:
        Filter function for TorchAO quantize_
    """

    def _is_target(name: str) -> bool:
        if target_layer_keys is not None and not any(p in name for p in target_layer_keys):
            return False
        if exclude_layer_keys is not None and any(p in name for p in exclude_layer_keys):
            return False
        return True

    def _filter(mod: nn.Module, name: str) -> bool:
        return isinstance(mod, nn.Linear) and _is_target(name)

    return _filter


@torch.no_grad()
def quantize_model_weights_fp8_torchao(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    target_layer_keys: Optional[Iterable[str]] = None,
    exclude_layer_keys: Optional[Iterable[str]] = None,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
) -> nn.Module:
    """
    Quantize model weights for Linear layers to FP8 using TorchAO.

    This uses TorchAO's Float8WeightOnlyConfig which provides:
    - Per-channel (row-wise) symmetric quantization
    - Scale derived from amax (max(abs)) of each row
    - No need to replace forward methods manually

    Args:
        model: Model to be quantized (inplace update)
        device: Device to place the quantized model (e.g. "cuda"). If None, don't move.
        target_layer_keys: Filter by FQN - include these patterns
        exclude_layer_keys: Filter by FQN - exclude these patterns
        weight_dtype: torch.float8_e4m3fn or torch.float8_e5m2

    Returns:
        model: Quantized model (same instance as input)

    Raises:
        ImportError: If TorchAO is not available
        ValueError: If weight_dtype is not supported
    """
    if not TORCHAO_AVAILABLE:
        raise ImportError(
            "TorchAO is not available. Please install torchao or use the native FP8 implementation."
        )

    if weight_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"Unsupported weight_dtype: {weight_dtype}. Supported: float8_e4m3fn, float8_e5m2")

    logger.info(f"🧮 TorchAO FP8 starting - {weight_dtype} format")
    start_time = time.perf_counter()

    # Create TorchAO configuration
    cfg = Float8WeightOnlyConfig(weight_dtype=weight_dtype)

    # Create filter function
    filter_fn = _make_filter_fn(target_layer_keys, exclude_layer_keys)

    # Count layers before quantization for reporting
    total_linear_layers = sum(1 for name, module in model.named_modules()
                             if isinstance(module, nn.Linear))
    targeted_layers = sum(1 for name, module in model.named_modules()
                         if filter_fn(module, name))

    logger.info(f"🧮 TorchAO targets: {targeted_layers}/{total_linear_layers} Linear layers")

    # Apply TorchAO quantization
    quantize_(model, cfg, filter_fn=filter_fn, device=device)

    end_time = time.perf_counter()
    logger.info(f"🧮 TorchAO FP8 quantization completed: {targeted_layers} layers quantized ({end_time - start_time:.2f}s)")

    return model


def is_torchao_available() -> bool:
    """
    Check if TorchAO is available for FP8 quantization.

    Returns:
        bool: True if TorchAO is available and can be used
    """
    return TORCHAO_AVAILABLE


def get_torchao_supported_dtypes() -> List[torch.dtype]:
    """
    Get list of FP8 dtypes supported by TorchAO.

    Returns:
        List of supported torch dtypes
    """
    if not TORCHAO_AVAILABLE:
        return []

    return [torch.float8_e4m3fn, torch.float8_e5m2]


class TorchAOFP8Config:
    """
    Configuration class for TorchAO FP8 quantization.

    This provides a unified interface for TorchAO-specific settings
    that integrates with Takenoko's configuration system.
    """

    def __init__(
        self,
        enabled: bool = False,
        weight_dtype: str = "e4m3fn",
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.weight_dtype = weight_dtype
        self.target_modules = target_modules or []
        self.exclude_modules = exclude_modules or []

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate TorchAO configuration parameters."""
        if not self.enabled:
            return

        if not TORCHAO_AVAILABLE:
            logger.warning("⚠️  TorchAO not available but torchao_fp8_enabled=True")
            self.enabled = False
            return

        valid_dtypes = ["e4m3fn", "e5m2"]
        if self.weight_dtype not in valid_dtypes:
            raise ValueError(f"Invalid weight_dtype: {self.weight_dtype}. Valid: {valid_dtypes}")

    def get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype from string configuration."""
        if self.weight_dtype == "e4m3fn":
            return torch.float8_e4m3fn
        elif self.weight_dtype == "e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported weight_dtype: {self.weight_dtype}")

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "weight_dtype": self.weight_dtype,
            "target_modules": self.target_modules,
            "exclude_modules": self.exclude_modules,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TorchAOFP8Config":
        """Create config from dictionary."""
        return cls(**config_dict)


def apply_torchao_fp8_quantization(
    model: nn.Module,
    config: TorchAOFP8Config,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Apply TorchAO FP8 quantization to a model using configuration.

    Args:
        model: Model to quantize
        config: TorchAO configuration
        device: Target device

    Returns:
        Quantized model (same instance)

    Raises:
        RuntimeError: If TorchAO is not available but config.enabled=True
    """
    if not config.enabled:
        logger.info("🧮 TorchAO FP8 quantization disabled")
        return model

    if not TORCHAO_AVAILABLE:
        raise RuntimeError(
            "TorchAO FP8 quantization enabled but TorchAO is not available. "
            "Please install torchao or disable torchao_fp8_enabled."
        )

    return quantize_model_weights_fp8_torchao(
        model=model,
        device=device,
        target_layer_keys=config.target_modules if config.target_modules else None,
        exclude_layer_keys=config.exclude_modules if config.exclude_modules else None,
        weight_dtype=config.get_torch_dtype(),
    )