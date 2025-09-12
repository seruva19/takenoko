from __future__ import annotations

import os
import logging
import torch
from typing import Any, Dict, Optional


def configure_cuda_from_config(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Configure PyTorch CUDA settings and environment variables from config.

    Reads top-level keys (recommended) usually placed under the OPTIMIZATION section
    of the TOML. Legacy [cuda_allocator] table is no longer supported.

    Supported keys (top-level):
      CUDA Allocator Settings:
      - cuda_allocator_enable: bool
      - cuda_allocator_max_split_size_mb: int > 0
      - cuda_allocator_expandable_segments: bool

      CUDA Environment Settings:
      - cuda_launch_blocking: bool
      - cuda_managed_force_device_alloc: bool
      - cuda_visible_devices: str

      CUDA Runtime Settings:
      - cuda_memory_fraction: float (0.0-1.0)
      - cuda_empty_cache: bool
      - cuda_flash_sdp_enabled: bool

    All settings are logged with success/failure status.
    """

    log = logger or logging.getLogger(__name__)

    log.info("Configuring CUDA settings from config...")

    # Configure CUDA environment variables first (must be set before importing torch)
    _configure_cuda_environment_variables(config, log)

    # Configure CUDA allocator settings
    _configure_cuda_allocator(config, log)

    # Configure CUDA runtime settings (after torch is available)
    _configure_cuda_runtime_settings(config, log)

    log.info("✅ CUDA configuration completed")


def _configure_cuda_environment_variables(
    config: Dict[str, Any], log: logging.Logger
) -> None:
    """Configure CUDA environment variables that must be set before torch operations."""

    log.info("Configuring CUDA environment variables...")

    # CUDA_LAUNCH_BLOCKING
    cuda_launch_blocking = config.get("cuda_launch_blocking")
    if cuda_launch_blocking is not None:
        try:
            value = "1" if bool(cuda_launch_blocking) else "0"
            if "CUDA_LAUNCH_BLOCKING" not in os.environ:
                os.environ["CUDA_LAUNCH_BLOCKING"] = value
                log.info(f"Set CUDA_LAUNCH_BLOCKING={value}")
            else:
                log.info(
                    f"Using existing CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}"
                )
        except Exception as e:
            log.error(f"⚠️ Failed to set CUDA_LAUNCH_BLOCKING: {e}")

    # CUDA_MANAGED_FORCE_DEVICE_ALLOC
    cuda_managed_force = config.get("cuda_managed_force_device_alloc")
    if cuda_managed_force is not None:
        try:
            value = "1" if bool(cuda_managed_force) else "0"
            if "CUDA_MANAGED_FORCE_DEVICE_ALLOC" not in os.environ:
                os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = value
                log.info(f"Set CUDA_MANAGED_FORCE_DEVICE_ALLOC={value}")
            else:
                log.info(
                    f"💾 Using existing CUDA_MANAGED_FORCE_DEVICE_ALLOC={os.environ['CUDA_MANAGED_FORCE_DEVICE_ALLOC']}"
                )
        except Exception as e:
            log.error(f"⚠️ Failed to set CUDA_MANAGED_FORCE_DEVICE_ALLOC: {e}")

    # CUDA_VISIBLE_DEVICES
    cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        try:
            value = str(cuda_visible_devices)
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = value
                log.info(f"Set CUDA_VISIBLE_DEVICES={value}")
            else:
                log.info(
                    f"💾 Using existing CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
                )
        except Exception as e:
            log.error(f"⚠️ Failed to set CUDA_VISIBLE_DEVICES: {e}")


def _configure_cuda_allocator(config: Dict[str, Any], log: logging.Logger) -> None:
    """Configure PyTorch CUDA caching allocator via environment variable."""

    log.info("Configuring CUDA allocator...")

    # Respect existing environment
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        log.info(
            f"Using existing PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
        )
        return

    # Check if allocator is enabled
    enable = bool(config.get("cuda_allocator_enable", False))
    if not enable:
        log.info("⏭️  CUDA allocator configuration disabled")
        return

    options: list[str] = []

    # max_split_size_mb (only when allocator is enabled)
    max_split_raw = config.get("cuda_allocator_max_split_size_mb")
    if max_split_raw is not None:
        try:
            msz = int(max_split_raw)
            if msz > 0:
                options.append(f"max_split_size_mb:{msz}")
                log.info(f"Added max_split_size_mb:{msz}")
            else:
                log.warning(
                    f"⚠️ Invalid max_split_size_mb value (must be > 0): {max_split_raw}"
                )
        except Exception as e:
            log.error(
                f"⚠️ Failed to parse max_split_size_mb value: {max_split_raw}, error: {e}"
            )

    # expandable_segments (only when allocator is enabled)
    expandable_raw = config.get("cuda_allocator_expandable_segments")
    if expandable_raw is not None:
        try:
            expandable_val = bool(expandable_raw)
            options.append(
                f"expandable_segments:{'True' if expandable_val else 'False'}"
            )
            log.info(
                f"✅ Added expandable_segments:{'True' if expandable_val else 'False'}"
            )
        except Exception as e:
            log.error(
                f"❌ Failed to parse expandable_segments value: {expandable_raw}, error: {e}"
            )

    if not options:
        log.warning("⚠️  No valid CUDA allocator options configured")
        return

    try:
        env_value = ",".join(options)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_value
        log.info(f"Set PYTORCH_CUDA_ALLOC_CONF='{env_value}'")
    except Exception as e:
        log.error(f"⚠️ Failed to set PYTORCH_CUDA_ALLOC_CONF: {e}")


def _configure_cuda_runtime_settings(
    config: Dict[str, Any], log: logging.Logger
) -> None:
    """Configure CUDA runtime settings that require torch to be available."""

    log.info("Configuring CUDA runtime settings...")

    if not torch.cuda.is_available():
        log.warning("⚠️ CUDA not available, skipping runtime configuration")
        return

    # Empty CUDA cache (only if explicitly enabled)
    cuda_empty_cache = config.get("cuda_empty_cache", False)  # Default: disabled
    if cuda_empty_cache:
        try:
            torch.cuda.empty_cache()
            log.info("CUDA cache emptied")
        except Exception as e:
            log.error(f"⚠️ Failed to empty CUDA cache: {e}")

    # Set per-process memory fraction (only if explicitly configured)
    cuda_memory_fraction = config.get(
        "cuda_memory_fraction"
    )  # Default: None (use PyTorch default)
    if cuda_memory_fraction is not None:
        try:
            fraction = float(cuda_memory_fraction)
            if 0.0 < fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction)
                log.info(f"Set CUDA memory fraction to {fraction:.2%}")
            else:
                log.error(f"⚠️ Invalid memory fraction (must be 0.0-1.0): {fraction}")
        except Exception as e:
            log.error(f"⚠️ Failed to set CUDA memory fraction: {e}")

    # Configure flash SDP
    cuda_flash_sdp_enabled = config.get("cuda_flash_sdp_enabled")
    if cuda_flash_sdp_enabled is not None:
        try:
            enabled = bool(cuda_flash_sdp_enabled)
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(enabled)
                log.info(f"Set flash SDP enabled: {enabled}")
            else:
                log.warning(
                    "⚠️ torch.backends.cuda.enable_flash_sdp not available in this PyTorch version"
                )
        except Exception as e:
            log.error(f"⚠️ Failed to configure flash SDP: {e}")
