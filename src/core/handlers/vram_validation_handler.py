"""
VRAM Estimation Validation Handler

Handles VRAM estimation at training initialization and automatic validation
of estimates against actual usage after first training step.

Also integrates Windows-specific shared GPU memory detection.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def estimate_and_store_vram(
    args,
    config: Dict[str, Any],
) -> None:
    """
    Estimate VRAM usage and store in args for later validation.

    Only runs if validation is enabled (log_vram_validation config flag).
    Stores estimate in args._vram_estimate_gb and args._vram_estimate_details.

    Args:
        args: Training arguments to store estimate in
        config: Configuration dictionary
    """
    # Check if validation is disabled
    if not getattr(args, "log_vram_validation", False):
        return

    try:
        from common.vram_estimator import estimate_peak_vram_gb_from_config

        estimated_gb, details = estimate_peak_vram_gb_from_config(config)
        args._vram_estimate_gb = estimated_gb
        args._vram_estimate_details = details

        logger.info(
            f"🧠 Estimated peak VRAM: {estimated_gb:.2f} GB "
            f"(will compare with actual after first step)"
        )

    except Exception as e:
        logger.debug(f"Failed to estimate VRAM: {e}")
        args._vram_estimate_gb = None
        args._vram_estimate_details = None


def handle_vram_validation_if_enabled(
    args,
    global_step: int,
    accelerator,
) -> None:
    """
    Validate VRAM estimate vs actual usage after first training step.

    Only runs if:
    - It's the first step (global_step == 1)
    - Main process
    - Validation is enabled (log_vram_validation config flag)
    - VRAM estimate was stored during initialization

    Args:
        args: Training arguments containing config flags and stored estimate
        global_step: Current training step
        accelerator: Accelerator instance
    """
    # Check if validation is disabled
    if not getattr(args, "log_vram_validation", True):
        return

    # Only validate on first step
    if global_step != 1:
        return

    # Only on main process
    if not accelerator.is_main_process:
        return

    # Only if CUDA is available
    if not torch.cuda.is_available():
        return

    try:
        from common.vram_estimator import log_vram_comparison

        # Check if estimate was stored during initialization
        if not hasattr(args, "_vram_estimate_gb") or args._vram_estimate_gb is None:
            logger.debug("VRAM estimate not available, skipping validation")
            return

        estimated_gb = args._vram_estimate_gb
        details = getattr(args, "_vram_estimate_details", None)

        # Get actual peak VRAM
        actual_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # Log comparison
        log_vram_comparison(estimated_gb, actual_gb, details, logger)

    except Exception as e:
        logger.debug(f"Failed to validate VRAM estimate: {e}")


def handle_windows_shared_memory_check(
    args,
    global_step: int,
    accelerator,
) -> None:
    """
    Check for Windows shared GPU memory usage at training step.

    This function is called periodically during training to detect when
    Windows starts using system RAM as shared GPU memory (VRAM overflow).

    Only runs if:
    - Windows VRAM monitoring is enabled
    - Main process
    - CUDA is available

    Args:
        args: Training arguments containing config flags
        global_step: Current training step
        accelerator: Accelerator instance
    """
    # Only on main process
    if not accelerator.is_main_process:
        return

    # Only if CUDA is available
    if not torch.cuda.is_available():
        return

    try:
        from memory.windows_vram_monitor import check_shared_memory_at_step

        # Check at this step (monitor handles internal interval logic)
        check_shared_memory_at_step(global_step)

    except Exception as e:
        logger.debug(f"Windows shared memory check failed: {e}")
