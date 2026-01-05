"""LoRA Weight Statistics Tracking

This module provides utilities for tracking and logging LoRA weight distributions
during training. It enables visualization of weight histograms in TensorBoard,
helping detect training issues like weight explosion, collapse, or drift.

Key metrics tracked:
- Weight value distribution (histogram)
- Mean, std, min, max of weights
- Separate tracking for lora_down and lora_up matrices
"""

import torch
from typing import Any, Dict, List, Optional, Tuple
import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class LoRAWeightStatsTracker:
    """Tracks and logs LoRA weight statistics during training.

    Collects weight values from LoRA modules and provides:
    - Histogram data for TensorBoard visualization
    - Scalar statistics (mean, std, min, max)
    - Separate tracking for down/up projection matrices
    """

    def __init__(
        self,
        log_interval: int = 100,
        log_separate_matrices: bool = False,
        log_per_module: bool = False,
    ):
        """Initialize the LoRA weight stats tracker.

        Args:
            log_interval: Steps between logging (histogram logging is expensive)
            log_separate_matrices: Log lora_down and lora_up separately
            log_per_module: Log per-module statistics (very verbose)
        """
        self.log_interval = log_interval
        self.log_separate_matrices = log_separate_matrices
        self.log_per_module = log_per_module
        self._initialized = False
        self._initial_stats: Dict[str, Dict[str, float]] = {}

    def should_log(self, global_step: int) -> bool:
        """Check if we should log at this step."""
        return global_step > 0 and global_step % self.log_interval == 0

    def collect_weight_tensors(
        self, network: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect all LoRA weight values from the network.

        Args:
            network: The LoRA network (LoRANetwork instance)

        Returns:
            Tuple of (all_weights, down_weights, up_weights) as flattened tensors
        """
        all_weights: List[torch.Tensor] = []
        down_weights: List[torch.Tensor] = []
        up_weights: List[torch.Tensor] = []

        # Handle accelerate-wrapped models
        if hasattr(network, "module"):
            network = network.module

        # Get all LoRA modules
        lora_modules = []
        if hasattr(network, "unet_loras"):
            lora_modules.extend(network.unet_loras)
        if hasattr(network, "text_encoder_loras"):
            lora_modules.extend(network.text_encoder_loras)

        for lora_module in lora_modules:
            # Handle both single and split (ModuleList) cases
            if hasattr(lora_module, "lora_down"):
                down = lora_module.lora_down
                up = lora_module.lora_up

                if isinstance(down, torch.nn.ModuleList):
                    # Split QKV case
                    for d, u in zip(down, up):
                        down_weights.append(d.weight.data.detach().float().flatten())
                        up_weights.append(u.weight.data.detach().float().flatten())
                else:
                    # Standard case
                    down_weights.append(down.weight.data.detach().float().flatten())
                    up_weights.append(up.weight.data.detach().float().flatten())

        if not down_weights:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        down_tensor = torch.cat(down_weights)
        up_tensor = torch.cat(up_weights)
        all_tensor = torch.cat([down_tensor, up_tensor])

        return all_tensor, down_tensor, up_tensor

    def compute_statistics(self, weights: torch.Tensor) -> Dict[str, float]:
        """Compute scalar statistics for a weight tensor.

        Args:
            weights: Flattened weight tensor

        Returns:
            Dict with mean, std, min, max, abs_mean
        """
        if weights.numel() == 0:
            return {}

        return {
            "mean": weights.mean().item(),
            "std": weights.std().item(),
            "min": weights.min().item(),
            "max": weights.max().item(),
            "abs_mean": weights.abs().mean().item(),
            "count": weights.numel(),
        }

    def initialize_baseline(self, network: Any) -> None:
        """Store initial weight statistics for comparison.

        Args:
            network: The LoRA network
        """
        all_weights, down_weights, up_weights = self.collect_weight_tensors(network)

        if all_weights.numel() > 0:
            self._initial_stats = {
                "all": self.compute_statistics(all_weights),
                "down": self.compute_statistics(down_weights),
                "up": self.compute_statistics(up_weights),
            }
            self._initialized = True

            logger.info(
                f"📊 LoRA Weight Stats Initialized: "
                f"{self._initial_stats['all']['count']:,} parameters, "
                f"mean={self._initial_stats['all']['mean']:.6f}, "
                f"std={self._initial_stats['all']['std']:.6f}"
            )

    def get_histogram_data(self, network: Any) -> List[Tuple[str, torch.Tensor]]:
        """Get histogram data for TensorBoard logging.

        Args:
            network: The LoRA network

        Returns:
            List of (metric_name, tensor) tuples for histogram logging
        """
        all_weights, down_weights, up_weights = self.collect_weight_tensors(network)

        histograms = []

        if all_weights.numel() > 0:
            histograms.append(("lora_weights/all_weights", all_weights.cpu()))

            if self.log_separate_matrices:
                if down_weights.numel() > 0:
                    histograms.append(("lora_weights/down_weights", down_weights.cpu()))
                if up_weights.numel() > 0:
                    histograms.append(("lora_weights/up_weights", up_weights.cpu()))

        return histograms

    def compute_spectral_stats(self, network: Any) -> Dict[str, float]:
        """Compute spectral statistics (singular values) for LoRA matrices.

        Returns:
            Dict with mean/max spectral norms for down/up matrices
        """
        down_norms = []
        up_norms = []

        # Handle accelerate-wrapped models
        if hasattr(network, "module"):
            network = network.module

        # Get all LoRA modules
        lora_modules = []
        if hasattr(network, "unet_loras"):
            lora_modules.extend(network.unet_loras)
        if hasattr(network, "text_encoder_loras"):
            lora_modules.extend(network.text_encoder_loras)

        for lora_module in lora_modules:
            if hasattr(lora_module, "lora_down"):
                down = lora_module.lora_down
                up = lora_module.lora_up

                if isinstance(down, torch.nn.ModuleList):
                    # Split QKV case
                    for d, u in zip(down, up):
                        if hasattr(d, "weight") and hasattr(u, "weight"):
                            # Compute spectral norm (largest singular value)
                            # detach() and float() to ensure compatibility and no graph retention
                            try:
                                down_norms.append(
                                    torch.linalg.norm(
                                        d.weight.data.float(), ord=2
                                    ).item()
                                )
                                up_norms.append(
                                    torch.linalg.norm(
                                        u.weight.data.float(), ord=2
                                    ).item()
                                )
                            except Exception:
                                pass
                else:
                    if hasattr(down, "weight") and hasattr(up, "weight"):
                        try:
                            down_norms.append(
                                torch.linalg.norm(
                                    down.weight.data.float(), ord=2
                                ).item()
                            )
                            up_norms.append(
                                torch.linalg.norm(up.weight.data.float(), ord=2).item()
                            )
                        except Exception:
                            pass

        stats = {}
        if down_norms:
            stats["lora_weights/spectral_norm_down_mean"] = sum(down_norms) / len(
                down_norms
            )
            stats["lora_weights/spectral_norm_down_max"] = max(down_norms)

        if up_norms:
            stats["lora_weights/spectral_norm_up_mean"] = sum(up_norms) / len(up_norms)
            stats["lora_weights/spectral_norm_up_max"] = max(up_norms)

        return stats

    def get_scalar_metrics(self, network: Any) -> Dict[str, float]:
        """Get scalar metrics for standard logging.

        Args:
            network: The LoRA network

        Returns:
            Dict of metric_name -> value
        """
        all_weights, down_weights, up_weights = self.collect_weight_tensors(network)

        metrics = {}

        if all_weights.numel() > 0:
            stats = self.compute_statistics(all_weights)
            metrics["lora_weights/mean"] = stats["mean"]
            metrics["lora_weights/std"] = stats["std"]
            metrics["lora_weights/abs_mean"] = stats["abs_mean"]
            metrics["lora_weights/min"] = stats["min"]
            metrics["lora_weights/max"] = stats["max"]

            if self.log_separate_matrices:
                down_stats = self.compute_statistics(down_weights)
                up_stats = self.compute_statistics(up_weights)

                metrics["lora_weights/down_mean"] = down_stats["mean"]
                metrics["lora_weights/down_std"] = down_stats["std"]
                metrics["lora_weights/up_mean"] = up_stats["mean"]
                metrics["lora_weights/up_std"] = up_stats["std"]

            # Compute drift from initial if available
            if self._initialized and "all" in self._initial_stats:
                initial = self._initial_stats["all"]
                metrics["lora_weights/mean_drift"] = abs(
                    stats["mean"] - initial["mean"]
                )
                metrics["lora_weights/std_change"] = (stats["std"] - initial["std"]) / (
                    initial["std"] + 1e-8
                )

            # Add spectral stats (always computed as they are important for Muon/AdaMuon)
            spectral_stats = self.compute_spectral_stats(network)
            metrics.update(spectral_stats)

        return metrics


# Global instance for easy access
_lora_stats_tracker: Optional[LoRAWeightStatsTracker] = None


def get_lora_stats_tracker() -> Optional[LoRAWeightStatsTracker]:
    """Get the global LoRA stats tracker instance."""
    return _lora_stats_tracker


def initialize_lora_stats_tracker(
    log_interval: int = 100,
    log_separate_matrices: bool = False,
    log_per_module: bool = False,
) -> LoRAWeightStatsTracker:
    """Initialize the global LoRA stats tracker.

    Args:
        log_interval: Steps between histogram logging
        log_separate_matrices: Log lora_down and lora_up separately
        log_per_module: Log per-module statistics

    Returns:
        The initialized tracker
    """
    global _lora_stats_tracker
    _lora_stats_tracker = LoRAWeightStatsTracker(
        log_interval=log_interval,
        log_separate_matrices=log_separate_matrices,
        log_per_module=log_per_module,
    )
    return _lora_stats_tracker


def log_lora_weight_histograms(
    accelerator: Any,
    network: Any,
    global_step: int,
    tracker: Optional[LoRAWeightStatsTracker] = None,
) -> None:
    """Log LoRA weight histograms to TensorBoard.

    Args:
        accelerator: The Accelerate accelerator
        network: The LoRA network
        global_step: Current training step
        tracker: Optional tracker instance (uses global if None)
    """
    if tracker is None:
        tracker = _lora_stats_tracker

    if tracker is None or not tracker.should_log(global_step):
        return

    if not accelerator.is_main_process:
        return

    try:
        histogram_data = tracker.get_histogram_data(network)

        for tracker_obj in accelerator.trackers:
            if tracker_obj.name == "tensorboard":
                for metric_name, tensor_data in histogram_data:
                    if tensor_data.numel() > 0:
                        tracker_obj.writer.add_histogram(
                            metric_name, tensor_data, global_step
                        )
                break
    except Exception as e:
        logger.debug(f"Failed to log LoRA weight histograms: {e}")


def get_lora_weight_metrics(
    network: Any,
    tracker: Optional[LoRAWeightStatsTracker] = None,
) -> Dict[str, float]:
    """Get LoRA weight scalar metrics for logging.

    Args:
        network: The LoRA network
        tracker: Optional tracker instance (uses global if None)

    Returns:
        Dict of metric_name -> value
    """
    if tracker is None:
        tracker = _lora_stats_tracker

    if tracker is None:
        return {}

    try:
        return tracker.get_scalar_metrics(network)
    except Exception as e:
        logger.debug(f"Failed to get LoRA weight metrics: {e}")
        return {}
