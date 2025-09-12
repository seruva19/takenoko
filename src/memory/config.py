"""Memory optimization configuration parsing."""

import argparse
import logging
from common.logger import get_logger
from typing import Any, Dict

logger = get_logger(__name__, level=logging.INFO)


def parse_memory_optimization_config(
    args: argparse.Namespace, config: Dict[str, Any]
) -> argparse.Namespace:
    """Parse training-safe memory optimization configuration (root-level keys only)."""

    # Training-safe defaults
    defaults = {
        # Master switch - controls all features (no prefix for master switch)
        "safe_memory_optimization_enabled": False,
        # Sub-feature switches (with unique prefix)
        "memory_opt_loading_enabled": True,  # Enable memory-efficient SafeTensors loading
        "memory_opt_monitoring_enabled": True,  # Enable VRAM/RAM monitoring
        # Direct tunable parameters (with unique prefix)
        "memory_opt_gc_threshold_ratio": 0.85,  # GC trigger threshold (85% VRAM usage)
        "memory_opt_gc_interval_steps": 100,  # Trigger GC every N steps
        "memory_opt_monitor_interval_steps": 50,  # Monitor memory every N steps
        # IMPORTANT: Explicitly disable training-incompatible features (with unique prefix)
        "memory_opt_unsafe_quantization": False,  # Would break LORA gradients
        "memory_opt_inference_model_hooks": False,  # Would interfere with training
        "memory_opt_inference_optimizations": False,  # Incompatible with training
        # Memory Tracking Configuration (new)
        "memory_tracking_enabled": False,  # Master switch for memory tracking
        "memory_tracking_max_records": 10000,  # Maximum allocation records to keep
        "memory_tracking_stack_traces": False,  # Capture stack traces (expensive)
        "memory_tracking_tensor_tracking": True,  # Track individual tensor references
        "memory_tracking_auto_snapshot_interval": 100,  # Auto-snapshot every N allocations
        "memory_tracking_verbosity": "standard",  # "minimal", "standard", "debug"
        "memory_tracking_export_on_oom": True,  # Export diagnostics on OOM errors
        "memory_tracking_export_directory": "logs",  # Directory for diagnostic exports
        # CUDA configuration defaults (handled by memory_utils.py but tracked here)
        # All optimizations DISABLED by default - no tuning applied unless explicitly enabled
        "cuda_allocator_enable": False,
        "cuda_allocator_max_split_size_mb": 512,  # Only used when allocator enabled
        "cuda_allocator_expandable_segments": True,  # Only used when allocator enabled
        "cuda_launch_blocking": False,  # Default: async execution (faster)
        "cuda_managed_force_device_alloc": False,  # Default: no unified memory
        "cuda_visible_devices": None,  # Default: use all available GPUs
        "cuda_memory_fraction": None,  # Default: use PyTorch default (no artificial limit)
        "cuda_empty_cache": False,  # Default: don't clear cache at startup
        "cuda_flash_sdp_enabled": None,  # Default: use PyTorch default
    }

    # Root-level overrides for known keys only (section is ignored by design)
    memory_config: Dict[str, Any] = dict(defaults)
    for key in defaults.keys():
        if key in config:
            memory_config[key] = config[key]

    # CRITICAL: Master switch validation - disable all features if master switch is off
    if not memory_config.get("safe_memory_optimization_enabled", False):
        # Force disable all sub-features when master switch is off
        feature_keys = ["memory_opt_loading_enabled", "memory_opt_monitoring_enabled"]
        for feature_key in feature_keys:
            if memory_config.get(feature_key, False):
                logger.info(
                    f"Feature '{feature_key}' disabled (safe_memory_optimization_enabled=False)"
                )
                memory_config[feature_key] = False
    else:
        # Master switch is enabled - activate safe defaults for sub-features if not explicitly set
        safe_sub_defaults = {
            "memory_opt_loading_enabled": True,
            "memory_opt_monitoring_enabled": True,
        }
        for key, safe_default in safe_sub_defaults.items():
            if key not in memory_config or memory_config[key] is None:
                memory_config[key] = safe_default

    # Validation: Ensure incompatible features are disabled
    incompatible_features = [
        "memory_opt_unsafe_quantization",
        "memory_opt_inference_model_hooks",
        "memory_opt_inference_optimizations",
    ]
    for feature in incompatible_features:
        if memory_config.get(feature, False):
            logger.warning(
                f"Training-incompatible feature '{feature}' detected and disabled for safety"
            )
            memory_config[feature] = False

    # Add all memory config to args
    for key, value in memory_config.items():
        setattr(args, key, value)

    return args
