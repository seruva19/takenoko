"""Activation Statistics Tracking for Training Diagnostics

This module provides utilities for monitoring activation values during the forward
pass of DiT/transformer models. It helps detect potential training instabilities
like activation explosion before they cause NaN/Inf errors.

Key metrics tracked:
- Min/max activation values per layer
- Mean/std of activations
- Detection of extreme values (potential NaN precursors)

The tracking is designed to have zero overhead when disabled.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class ActivationStatsTracker:
    """Tracks activation statistics during forward pass using hooks.

    Designed for zero overhead when disabled - hooks are only registered
    when explicitly enabled.
    """

    def __init__(
        self,
        log_interval: int = 100,
        max_layers: int = 8,
        warn_threshold: float = 1000.0,
        critical_threshold: float = 10000.0,
    ):
        """Initialize the activation stats tracker.

        Args:
            log_interval: Steps between logging (to reduce overhead)
            max_layers: Maximum number of layers to track (first N blocks)
            warn_threshold: Activation magnitude that triggers a warning
            critical_threshold: Activation magnitude that triggers critical alert
        """
        self.log_interval = log_interval
        self.max_layers = max_layers
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold

        self._enabled = False
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._stats: Dict[str, Dict[str, float]] = {}
        self._should_collect = False
        self._layer_names: List[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_log(self, global_step: int) -> bool:
        """Check if we should log at this step."""
        return (
            self._enabled and global_step > 0 and global_step % self.log_interval == 0
        )

    def _create_hook(self, layer_name: str) -> Callable:
        """Create a forward hook for a specific layer."""

        def hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
            if not self._should_collect:
                return

            # Handle tuple outputs (some layers return multiple tensors)
            if isinstance(output, tuple):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                return

            try:
                # Compute stats without gradient tracking
                with torch.no_grad():
                    flat = output.detach().float()
                    self._stats[layer_name] = {
                        "min": flat.min().item(),
                        "max": flat.max().item(),
                        "mean": flat.mean().item(),
                        "std": flat.std().item(),
                        "abs_max": flat.abs().max().item(),
                    }
            except Exception:
                pass

        return hook

    def register_hooks(
        self, model: nn.Module, layer_class_name: str = "WanAttentionBlock"
    ) -> int:
        """Register forward hooks on target layers.

        Args:
            model: The model to register hooks on
            layer_class_name: Class name of layers to hook (default: WanAttentionBlock)

        Returns:
            Number of hooks registered
        """
        if self._enabled:
            return len(self._hooks)

        # Handle accelerate-wrapped models
        if hasattr(model, "module"):
            model = model.module

        count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == layer_class_name:
                if count >= self.max_layers:
                    break

                layer_name = f"block_{count}"
                hook = module.register_forward_hook(self._create_hook(layer_name))
                self._hooks.append(hook)
                self._layer_names.append(layer_name)
                count += 1

        if count > 0:
            self._enabled = True
            logger.info(
                f"📊 Activation stats tracking enabled: "
                f"{count} layers, interval={self.log_interval}"
            )

        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_names.clear()
        self._stats.clear()
        self._enabled = False

    def start_collection(self) -> None:
        """Enable stats collection for the current forward pass."""
        self._should_collect = True
        self._stats.clear()

    def stop_collection(self) -> None:
        """Disable stats collection."""
        self._should_collect = False

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get collected statistics from the last forward pass."""
        return self._stats.copy()

    def get_scalar_metrics(self) -> Dict[str, float]:
        """Get flattened scalar metrics for logging."""
        metrics = {}

        for layer_name, stats in self._stats.items():
            prefix = f"activation/{layer_name}"
            metrics[f"{prefix}/min"] = stats["min"]
            metrics[f"{prefix}/max"] = stats["max"]
            metrics[f"{prefix}/mean"] = stats["mean"]
            metrics[f"{prefix}/std"] = stats["std"]
            metrics[f"{prefix}/abs_max"] = stats["abs_max"]

        # Aggregate stats across all layers
        if self._stats:
            all_abs_max = [s["abs_max"] for s in self._stats.values()]
            all_means = [s["mean"] for s in self._stats.values()]
            all_stds = [s["std"] for s in self._stats.values()]

            metrics["activation/global_abs_max"] = max(all_abs_max)
            metrics["activation/global_mean"] = sum(all_means) / len(all_means)
            metrics["activation/global_std"] = sum(all_stds) / len(all_stds)

        return metrics

    def check_for_anomalies(self) -> List[str]:
        """Check for anomalous activation values.

        Returns:
            List of warning messages (empty if no anomalies)
        """
        warnings = []

        for layer_name, stats in self._stats.items():
            abs_max = stats["abs_max"]

            if abs_max >= self.critical_threshold:
                warnings.append(
                    f"🚨 CRITICAL: {layer_name} activation abs_max={abs_max:.1f} "
                    f"exceeds critical threshold ({self.critical_threshold})"
                )
            elif abs_max >= self.warn_threshold:
                warnings.append(
                    f"⚠️ WARNING: {layer_name} activation abs_max={abs_max:.1f} "
                    f"exceeds warning threshold ({self.warn_threshold})"
                )

        return warnings


# Global instance
_activation_tracker: Optional[ActivationStatsTracker] = None


def get_activation_tracker() -> Optional[ActivationStatsTracker]:
    """Get the global activation tracker instance."""
    return _activation_tracker


def initialize_activation_tracker(
    log_interval: int = 100,
    max_layers: int = 8,
    warn_threshold: float = 1000.0,
    critical_threshold: float = 10000.0,
) -> ActivationStatsTracker:
    """Initialize the global activation tracker.

    Args:
        log_interval: Steps between logging
        max_layers: Maximum number of layers to track
        warn_threshold: Activation magnitude that triggers a warning
        critical_threshold: Activation magnitude that triggers critical alert

    Returns:
        The initialized tracker
    """
    global _activation_tracker
    _activation_tracker = ActivationStatsTracker(
        log_interval=log_interval,
        max_layers=max_layers,
        warn_threshold=warn_threshold,
        critical_threshold=critical_threshold,
    )
    return _activation_tracker


def setup_activation_hooks(model: nn.Module) -> int:
    """Set up activation hooks on the model.

    Args:
        model: The DiT/transformer model

    Returns:
        Number of hooks registered
    """
    if _activation_tracker is None:
        return 0
    return _activation_tracker.register_hooks(model)


def collect_activation_stats(global_step: int) -> Optional[Dict[str, float]]:
    """Collect activation stats if it's time to log.

    This should be called BEFORE the forward pass to enable collection,
    and the stats should be retrieved AFTER the forward pass.

    Args:
        global_step: Current training step

    Returns:
        Dict of metrics if logging, None otherwise
    """
    if _activation_tracker is None:
        return None

    if not _activation_tracker.should_log(global_step):
        return None

    # Enable collection for this step
    _activation_tracker.start_collection()
    return None  # Stats will be collected during forward pass


def get_activation_metrics_after_forward() -> Optional[Dict[str, float]]:
    """Get activation metrics after forward pass completes.

    Returns:
        Dict of metrics, or None if not collecting
    """
    if _activation_tracker is None or not _activation_tracker._should_collect:
        return None

    _activation_tracker.stop_collection()

    # Check for anomalies and log warnings
    warnings = _activation_tracker.check_for_anomalies()
    for warning in warnings:
        logger.warning(warning)

    return _activation_tracker.get_scalar_metrics()


def cleanup_activation_tracker() -> None:
    """Clean up the activation tracker and remove hooks."""
    global _activation_tracker
    if _activation_tracker is not None:
        _activation_tracker.remove_hooks()
        _activation_tracker = None
