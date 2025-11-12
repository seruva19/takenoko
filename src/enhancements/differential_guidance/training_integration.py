"""Training Integration for Differential Guidance Enhancement.

This module provides a single-point integration for training_core to minimize clutter.
All differential guidance logic is encapsulated here.
"""

import torch
import argparse
from typing import Optional, Dict

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class DifferentialGuidanceIntegration:
    """Single-point integration for differential guidance in training_core."""

    def __init__(self):
        self.config = None
        self._initialized = False
        self._log_cadence = 500  # Log metrics every N steps
        self._last_logged_step = -1

    @classmethod
    def initialize_and_create(
        cls, args: argparse.Namespace
    ) -> "DifferentialGuidanceIntegration":
        """Create and initialize differential guidance integration.

        This encapsulates ALL initialization logic.

        Args:
            args: Command line arguments or config namespace

        Returns:
            DifferentialGuidanceIntegration instance
        """
        integration = cls()
        integration.initialize_from_args(args)
        return integration

    def initialize_from_args(self, args: argparse.Namespace) -> None:
        """Initialize differential guidance enhancement from args."""
        try:
            from .config import DifferentialGuidanceConfig

            self.config = DifferentialGuidanceConfig.from_args(args)

            if self.config.enable_differential_guidance:
                self._initialized = True
                logger.info(
                    "✅ Differential Guidance enabled: scale=%.2f, start_step=%d, end_step=%s",
                    self.config.differential_guidance_scale,
                    self.config.differential_guidance_start_step,
                    (
                        self.config.differential_guidance_end_step
                        if self.config.differential_guidance_end_step is not None
                        else "None"
                    ),
                )
            else:
                logger.debug("Differential Guidance disabled")

        except ImportError as e:
            logger.warning(
                f"Cannot initialize differential guidance - import error: {e}"
            )
        except ValueError as e:
            logger.error(f"Invalid differential guidance configuration: {e}")
            self.config = None
        except Exception as e:
            logger.error(f"Failed to initialize differential guidance: {e}")
            self.config = None

    def transform_target(
        self,
        target: torch.Tensor,
        model_pred: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """Single-call method to transform training target.

        Args:
            target: Original training target
            model_pred: Model predictions
            step: Current training step

        Returns:
            Transformed target (or original target if enhancement disabled/failed)
        """
        if not self._initialized or self.config is None:
            return target

        if not self.config.is_enabled_for_step(step):
            return target

        try:
            from .helper import apply_differential_guidance, compute_guidance_metrics

            guided_target = apply_differential_guidance(
                target=target,
                model_pred=model_pred,
                scale=self.config.differential_guidance_scale,
            )

            # Periodic logging for monitoring
            if (
                self._log_cadence > 0
                and step % self._log_cadence == 0
                and step != self._last_logged_step
            ):
                metrics = compute_guidance_metrics(target, model_pred, guided_target)
                logger.debug(
                    f"Differential Guidance @ step {step}: "
                    f"mean_diff={metrics['differential_guidance/mean_original_diff']:.4f}"
                )
                self._last_logged_step = step

            return guided_target

        except Exception as e:
            logger.error(
                f"Failed to apply differential guidance at step {step}: {e}, "
                f"falling back to original target"
            )
            return target

    def is_enabled(self) -> bool:
        """Check if differential guidance is enabled and working."""
        return (
            self._initialized
            and self.config is not None
            and self.config.enable_differential_guidance
        )

    def is_enabled_for_step(self, step: int) -> bool:
        """Check if differential guidance should be applied at this step."""
        if not self.is_enabled():
            return False
        return self.config.is_enabled_for_step(step)

    def get_config_summary(self) -> Dict[str, any]:
        """Get configuration summary for logging."""
        if not self._initialized or self.config is None:
            return {"differential_guidance": "disabled"}

        return {
            "differential_guidance": "enabled",
            "scale": self.config.differential_guidance_scale,
            "start_step": self.config.differential_guidance_start_step,
            "end_step": self.config.differential_guidance_end_step,
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.config = None
        self._initialized = False


# Global convenience functions for even simpler integration
def create_differential_guidance_integration(
    args: argparse.Namespace,
) -> DifferentialGuidanceIntegration:
    return DifferentialGuidanceIntegration.initialize_and_create(args)


def transform_target_with_differential_guidance(
    integration: Optional[DifferentialGuidanceIntegration],
    target: torch.Tensor,
    model_pred: torch.Tensor,
    step: int = 0,
) -> torch.Tensor:
    if integration is None or not integration.is_enabled():
        return target
    return integration.transform_target(target, model_pred, step)
