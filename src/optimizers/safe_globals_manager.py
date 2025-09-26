"""Safe globals manager for custom optimizer classes.

This module provides a centralized way to add custom optimizer classes to PyTorch's
safe globals list for state loading compatibility with PyTorch 2.6+.
"""

import logging
import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SafeGlobalsManager:
    """Manages PyTorch safe globals for custom optimizer classes."""

    @staticmethod
    def add_custom_optimizer_safe_globals():
        """Add custom optimizer classes to PyTorch's safe globals list for state loading."""
        try:
            # Import Fira classes that need to be in the safe globals list
            # Use vendored implementation
            from vendor.fira.gradient_projection import GradientProjector
            from vendor.fira.fira_adamw import FiraAdamW

            # Import custom classes from other optimizers
            from optimizers.optimizer_utils import Auto8bitTensor
            from optimizers.hina_adaptive import (
                EnhancedBufferPool,
                CompactStateDict,
                CompressedRelationships,
                AsyncComputeManager,
            )

            # Add to PyTorch's safe globals
            torch.serialization.add_safe_globals(
                [
                    # Fira optimizer classes
                    GradientProjector,
                    FiraAdamW,
                    # Automagic optimizer classes
                    Auto8bitTensor,
                    # HinaAdaptive optimizer classes
                    EnhancedBufferPool,
                    CompactStateDict,
                    CompressedRelationships,
                    AsyncComputeManager,
                ]
            )
            logger.info("✅ Added custom optimizer classes to PyTorch safe globals")
        except ImportError:
            logger.warning(
                "⚠️  Some optimizer packages not available, skipping safe globals addition"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to add custom optimizer safe globals: {e}")
