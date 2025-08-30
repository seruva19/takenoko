"""Adaptive timestep sampling configuration utilities."""

import argparse
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Adaptive timestep sampling
try:
    from scheduling.adaptive_timestep_manager import (
        AdaptiveTimestepManager,
        create_adaptive_timestep_manager,
    )
except ImportError as e:
    logger.warning(f"Could not import AdaptiveTimestepManager: {e}")
    AdaptiveTimestepManager = None
    create_adaptive_timestep_manager = None


def initialize_adaptive_timestep_sampling(args: argparse.Namespace) -> Optional[Any]:
    """Initialize adaptive timestep sampling if enabled.
    
    Returns:
        The initialized adaptive manager or None if disabled/unavailable.
    """
    try:
        if create_adaptive_timestep_manager is None:
            if getattr(args, "enable_adaptive_timestep_sampling", False):
                logger.warning(
                    "Adaptive timestep sampling requested but AdaptiveTimestepManager not available"
                )
            return None

        adaptive_manager = create_adaptive_timestep_manager(args)

        if adaptive_manager and adaptive_manager.enabled:
            logger.info("🎯 Adaptive Timestep Sampling initialized successfully")
            logger.info(
                f"   Boundary range: [{adaptive_manager.min_timestep}, {adaptive_manager.max_timestep}]"
            )
            logger.info(
                f"   Focus strength: {adaptive_manager.focus_strength}"
            )
            logger.info(f"   Warmup steps: {adaptive_manager.warmup_steps}")
            logger.info(
                f"   Video-specific categories: {adaptive_manager.video_specific_categories}"
            )
            return adaptive_manager
        else:
            logger.debug("Adaptive Timestep Sampling is disabled")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize Adaptive Timestep Sampling: {e}")
        logger.warning("Continuing with standard timestep sampling")
        return None