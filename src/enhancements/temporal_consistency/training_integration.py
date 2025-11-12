"""
Training Integration for Temporal Consistency Enhancement

This module provides a single-point integration for training_core to minimize clutter.
All temporal consistency logic is encapsulated here.
"""

import torch
import argparse
from typing import Optional

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TemporalConsistencyTrainingIntegration:
    """Single-point integration for temporal consistency in training_core."""

    def __init__(self):
        self.enhancer = None
        self.monitor = None
        self._initialized = False
        # TensorBoard logging callback (logs: Dict[str, float], step: int) -> None
        self._tb_logger = None
        self._tb_every_steps: int = 500

    @classmethod
    def initialize_and_create(
        cls, args: argparse.Namespace, device: Optional[torch.device] = None
    ) -> "TemporalConsistencyTrainingIntegration":
        """Create and initialize temporal consistency integration.

        This encapsulates ALL initialization logic.

        Args:
            args: Command line arguments or config namespace

        Returns:
            TemporalConsistencyTrainingIntegration instance
        """
        integration = cls()
        integration.initialize_from_args(args, device)
        return integration

    def initialize_from_args(
        self, args: argparse.Namespace, device: Optional[torch.device] = None
    ) -> None:
        """Initialize temporal consistency enhancement from args."""
        try:
            from .temporal_enhancer import TemporalConsistencyEnhancer
            from .utils import TemporalConsistencyMonitor
            from .config import TemporalConsistencyConfig
            from .config import TemporalConsistencyConfig

            _device = device or getattr(args, "device", torch.device("cpu"))
            self.enhancer = TemporalConsistencyEnhancer.initialize_from_args(
                args, _device
            )

            if self.enhancer is not None:
                self.monitor = TemporalConsistencyMonitor(history_size=1000)
                self._initialized = True
                # Emit a clear one-time INFO log confirming configuration
                cfg = TemporalConsistencyConfig.from_args(args)
                enabled = []
                if cfg.enable_frequency_domain_temporal_consistency:
                    enabled.append(
                        f"structural(low_thr={cfg.freq_temporal_low_threshold:.2f}, w={cfg.freq_temporal_consistency_weight:.3f})"
                    )
                if cfg.freq_temporal_enable_motion_coherence:
                    enabled.append(
                        f"motion(w={cfg.freq_temporal_motion_weight:.3f}, thr={cfg.freq_temporal_motion_threshold:.3f})"
                    )
                if cfg.freq_temporal_enable_prediction_loss:
                    enabled.append(
                        f"prediction(w={cfg.freq_temporal_prediction_weight:.3f})"
                    )
                logger.info(
                    "✅ Temporal Consistency enabled: %s",
                    ", ".join(enabled) if enabled else "none",
                )
                # Initialize TB cadence from config (caller can override via setter)
                try:
                    self._tb_every_steps = max(
                        1,
                        int(
                            getattr(
                                args,
                                "freq_temporal_tb_log_every_steps",
                                cfg.freq_temporal_tb_log_every_steps,
                            )
                        ),
                    )
                except Exception:
                    self._tb_every_steps = 500

        except ImportError as e:
            logger.warning(
                f"Cannot initialize temporal consistency - import error: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize temporal consistency: {e}")
            self.enhancer = None
            self.monitor = None

    def enhance_loss(
        self,
        base_loss: torch.Tensor,
        model_pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """Single-call method to enhance training loss.

        Args:
            base_loss: Original training loss
            model_pred: Model predictions
            target: Target values (optional)
            step: Current training step

        Returns:
            Enhanced loss (or original loss if enhancement disabled/failed)
        """
        if not self._initialized or self.enhancer is None:
            return base_loss

        enhanced_loss = self.enhancer.enhance_training_loss_with_monitoring(
            base_loss=base_loss,
            model_pred=model_pred,
            target=target,
            step=step,
            monitor=self.monitor,
        )

        return enhanced_loss

    def is_enabled(self) -> bool:
        """Check if temporal consistency is enabled and working."""
        return (
            self._initialized
            and self.enhancer is not None
            and self.enhancer.is_enabled()
        )

    def get_performance_stats(self) -> dict:
        """Get performance statistics (safe call)."""
        if not self._initialized or self.enhancer is None:
            return {"temporal_consistency": "disabled"}

        try:
            stats = self.enhancer.get_performance_stats()
            if self.monitor is not None:
                stats.update({"monitor": self.monitor.get_summary_report()})
            return stats
        except Exception as e:
            return {"error": str(e)}

    # Public API to enable TB logging from the host training loop
    def set_tensorboard_logger(
        self, logger_fn, every_steps: Optional[int] = None
    ) -> None:
        """Register a TensorBoard logging callback.

        Args:
            logger_fn: Callable(logs: Dict[str, float], step: int) -> None
            every_steps: Optional cadence override; defaults to config if None
        """
        self._tb_logger = logger_fn
        if self.enhancer is not None:
            self.enhancer.set_tensorboard_logger(logger_fn, every_steps)
        if every_steps is not None:
            try:
                self._tb_every_steps = max(1, int(every_steps))
            except Exception:
                self._tb_every_steps = max(1, self._tb_every_steps)

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.enhancer is not None:
            try:
                self.enhancer.cleanup()
            except Exception:
                pass

        if self.monitor is not None:
            try:
                self.monitor.clear_history()
            except Exception:
                pass

        self.enhancer = None
        self.monitor = None
        self._initialized = False


# Global convenience functions for even simpler integration
def create_temporal_consistency_integration(
    args: argparse.Namespace,
) -> TemporalConsistencyTrainingIntegration:
    """Create temporal consistency integration in one call."""
    return TemporalConsistencyTrainingIntegration.initialize_and_create(args)


def enhance_loss_with_temporal_consistency(
    integration: Optional[TemporalConsistencyTrainingIntegration],
    base_loss: torch.Tensor,
    model_pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    step: int = 0,
) -> torch.Tensor:
    """Global function for loss enhancement - ultimate single call."""
    if integration is None:
        return base_loss
    return integration.enhance_loss(base_loss, model_pred, target, step)
