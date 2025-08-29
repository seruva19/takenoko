"""Enhanced optimizer logging functionality.

This module provides enhanced logging capabilities for specific optimizers that have
internal state or adaptive learning rates, such as Prodigy and Automagic optimizers.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.optim import Optimizer

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class EnhancedOptimizerLogger:
    """Enhanced logging functionality for specific optimizers.

    This class provides additional metrics and visualizations for optimizers that have
    internal state or adaptive learning rates, such as Prodigy and Automagic optimizers.
    """

    def __init__(self):
        """Initialize the enhanced optimizer logger."""
        self.supported_optimizers = {
            "Prodigy": self._log_prodigy_metrics,
            "Prodigy8bit": self._log_prodigy_metrics,
            "Automagic": self._log_automagic_metrics,
        }

    def get_prodigy_d(self, optimizer: Optimizer) -> float:
        """Extract the 'd' parameter from Prodigy optimizer for logging.

        Args:
            optimizer: The optimizer instance

        Returns:
            Average 'd' value across all parameter groups
        """
        try:
            if not hasattr(optimizer, "param_groups"):
                return 0.0

            d_values = []
            for group in optimizer.param_groups:
                if "d" in group:
                    d_values.append(group["d"])

            if not d_values:
                return 0.0

            return sum(d_values) / len(d_values)
        except Exception as e:
            logger.debug(f"Failed to extract Prodigy 'd' parameter: {e}")
            return 0.0

    def get_automagic_lrs(self, optimizer: Optimizer) -> Tuple[torch.Tensor, float]:
        """Extract ALL individual, per-element learning rates from Automagic optimizer.

        Args:
            optimizer: The Automagic optimizer instance

        Returns:
            Tuple of (all_learning_rates_tensor, average_learning_rate)
        """
        try:
            # Handle Accelerate-wrapped optimizers
            if hasattr(optimizer, "optimizer"):
                # This is an AcceleratedOptimizer, get the underlying optimizer
                actual_optimizer = optimizer.optimizer  # type: ignore
            else:
                actual_optimizer = optimizer

            all_lrs_tensors = []
            total_params = 0
            params_with_state = 0
            params_with_lr_mask = 0

            for group in actual_optimizer.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        total_params += 1

                        # The critical check: does this parameter have a state and the lr_mask?
                        if p in actual_optimizer.state:
                            params_with_state += 1
                            state = actual_optimizer.state[p]

                            if "lr_mask" in state:
                                params_with_lr_mask += 1
                                lr_mask = state["lr_mask"]

                                # The lr_mask is a custom Auto8bitTensor. We need to get the float tensor.
                                # Based on the optimizer code, it has a `.to(torch.float32)` method.
                                if hasattr(lr_mask, "to"):
                                    float_lrs = lr_mask.to(torch.float32)
                                else:
                                    # Fallback just in case
                                    float_lrs = lr_mask

                                # Add the full tensor for this parameter
                                all_lrs_tensors.append(float_lrs.flatten())

            # If we didn't find any parameters with an lr_mask, return empty.
            if not all_lrs_tensors:
                return torch.tensor([]), 0.0

            # Concatenate all learning rate tensors into a single flat tensor
            full_lrs_tensor = torch.cat(all_lrs_tensors)

            if full_lrs_tensor.numel() == 0:
                return torch.tensor([]), 0.0

            # Calculate the true average from the full tensor
            avg_lr = full_lrs_tensor.mean().item()

            return full_lrs_tensor, avg_lr
        except Exception as e:
            print(f"[DEBUG] Error extracting Automagic learning rates: {e}")
            logger.error(
                f"Error extracting Automagic learning rates: {e}", exc_info=True
            )
            return torch.tensor([]), 0.0

    def _log_prodigy_metrics(self, optimizer: Optimizer) -> Dict[str, float]:
        """Log Prodigy optimizer metrics.

        Args:
            optimizer: The Prodigy optimizer instance

        Returns:
            Dict containing Prodigy-specific metrics
        """
        metrics = {}
        try:
            prodigy_d = self.get_prodigy_d(optimizer)
            metrics["train/prodigy_d"] = prodigy_d
        except Exception as e:
            logger.debug(f"Failed to log Prodigy metrics: {e}")
        return metrics

    def _log_automagic_metrics(self, optimizer: Optimizer) -> Dict[str, float]:
        """Log Automagic optimizer metrics.

        Args:
            optimizer: The Automagic optimizer instance

        Returns:
            Dict containing Automagic-specific metrics
        """
        metrics = {}
        try:
            lrs_tensor, avg_lr = self.get_automagic_lrs(optimizer)
            if len(lrs_tensor) > 0:
                metrics["train/automagic_avg_lr"] = avg_lr
                logger.debug(
                    f"Automagic logging: {len(lrs_tensor)} parameters, avg_lr={avg_lr:.2e}"
                )
            else:
                logger.debug("Automagic logging: No learning rates found")
        except Exception as e:
            logger.debug(f"Failed to log Automagic metrics: {e}")
        return metrics

    def get_enhanced_metrics(self, optimizer: Optimizer) -> Dict[str, float]:
        """Get enhanced metrics for the given optimizer.

        Args:
            optimizer: The optimizer instance

        Returns:
            Dict containing enhanced metrics for supported optimizers
        """
        # Handle Accelerate-wrapped optimizers
        if hasattr(optimizer, "optimizer"):
            # This is an AcceleratedOptimizer, get the underlying optimizer
            actual_optimizer = optimizer.optimizer  # type: ignore
            optimizer_name = actual_optimizer.__class__.__name__
        else:
            optimizer_name = optimizer.__class__.__name__

        if optimizer_name in self.supported_optimizers:
            return self.supported_optimizers[optimizer_name](optimizer)

        return {}

    def get_histogram_data(
        self, optimizer: Optimizer
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Get histogram data for optimizers that support it.

        Args:
            optimizer: The optimizer instance

        Returns:
            Tuple of (metric_name, tensor_data) if supported, None otherwise
        """
        # Handle Accelerate-wrapped optimizers
        if hasattr(optimizer, "optimizer"):
            # This is an AcceleratedOptimizer, get the underlying optimizer
            actual_optimizer = optimizer.optimizer  # type: ignore
            optimizer_name = actual_optimizer.__class__.__name__
        else:
            optimizer_name = optimizer.__class__.__name__

        if optimizer_name == "Automagic":
            try:
                lrs_tensor, _ = self.get_automagic_lrs(optimizer)
                if len(lrs_tensor) > 0:
                    return ("train/automagic_lrs_histogram", lrs_tensor)
            except Exception as e:
                logger.debug(f"Failed to get Automagic histogram data: {e}")

        return None

    def is_supported(self, optimizer: Optimizer) -> bool:
        """Check if the optimizer supports enhanced logging.

        Args:
            optimizer: The optimizer instance

        Returns:
            True if enhanced logging is supported, False otherwise
        """
        # Handle Accelerate-wrapped optimizers
        if hasattr(optimizer, "optimizer"):
            # This is an AcceleratedOptimizer, get the underlying optimizer
            actual_optimizer = optimizer.optimizer  # type: ignore
            optimizer_name = actual_optimizer.__class__.__name__
        else:
            optimizer_name = optimizer.__class__.__name__

        return optimizer_name in self.supported_optimizers

    def get_supported_optimizers(self) -> List[str]:
        """Get list of supported optimizer names.

        Returns:
            List of optimizer class names that support enhanced logging
        """
        return list(self.supported_optimizers.keys())

    def test_enhanced_logging(self, optimizer: Optimizer) -> Dict[str, Any]:
        """Test the enhanced optimizer logging functionality.

        Args:
            optimizer: The optimizer to test

        Returns:
            Dict with test results
        """
        test_results = {}

        try:
            optimizer_name = optimizer.__class__.__name__
            test_results["optimizer_name"] = optimizer_name
            test_results["is_supported"] = self.is_supported(optimizer)

            if not self.is_supported(optimizer):
                test_results["message"] = (
                    f"Optimizer {optimizer_name} not supported for enhanced logging"
                )
                return test_results

            # Get enhanced metrics
            metrics = self.get_enhanced_metrics(optimizer)
            test_results["metrics"] = metrics

            # Get histogram data if available
            histogram_data = self.get_histogram_data(optimizer)
            if histogram_data:
                metric_name, tensor_data = histogram_data
                test_results["histogram_metric"] = metric_name
                test_results["histogram_data_shape"] = list(tensor_data.shape)
                test_results["histogram_data_count"] = len(tensor_data)

            # Log test results
            if optimizer_name in ["Prodigy", "Prodigy8bit"]:
                prodigy_d = self.get_prodigy_d(optimizer)
                logger.info(f"Prodigy 'd' parameter: {prodigy_d}")

            elif optimizer_name == "Automagic":
                lrs_tensor, avg_lr = self.get_automagic_lrs(optimizer)
                logger.info(
                    f"Automagic learning rates: {len(lrs_tensor)} parameters, avg: {avg_lr:.2e}"
                )

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Error testing enhanced optimizer logging: {e}")

        return test_results


# Global instance for easy access
enhanced_logger = EnhancedOptimizerLogger()


def get_prodigy_d(optimizer: Optimizer) -> float:
    """Extract the 'd' parameter from Prodigy optimizer for logging.

    Args:
        optimizer: The optimizer instance

    Returns:
        Average 'd' value across all parameter groups
    """
    return enhanced_logger.get_prodigy_d(optimizer)


def get_automagic_lrs(optimizer: Optimizer) -> Tuple[torch.Tensor, float]:
    """Extract learning rates from Automagic optimizer for logging.

    Args:
        optimizer: The Automagic optimizer instance

    Returns:
        Tuple of (learning_rates_tensor, average_learning_rate)
    """
    return enhanced_logger.get_automagic_lrs(optimizer)


def get_enhanced_metrics(optimizer: Optimizer) -> Dict[str, float]:
    """Get enhanced metrics for the given optimizer.

    Args:
        optimizer: The optimizer instance

    Returns:
        Dict containing enhanced metrics for supported optimizers
    """
    return enhanced_logger.get_enhanced_metrics(optimizer)


def get_histogram_data(optimizer: Optimizer) -> Optional[Tuple[str, torch.Tensor]]:
    """Get histogram data for optimizers that support it.

    Args:
        optimizer: The optimizer instance

    Returns:
        Tuple of (metric_name, tensor_data) if supported, None otherwise
    """
    return enhanced_logger.get_histogram_data(optimizer)


def is_supported(optimizer: Optimizer) -> bool:
    """Check if the optimizer supports enhanced logging.

    Args:
        optimizer: The optimizer instance

    Returns:
        True if enhanced logging is supported, False otherwise
    """
    return enhanced_logger.is_supported(optimizer)
