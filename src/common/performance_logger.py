"""Performance logging functionality for training diagnostics.

This module provides comprehensive logging capabilities for:
- Performance timing analysis (data loading, forward/backward pass, optimizer step)
- Model output and target tensor statistics
- Enhanced hardware utilization metrics

Supports configurable verbosity levels:
- minimal: Essential metrics only (timing, convergence, stability)
- standard: Standard metrics + essential (default)
- debug: All metrics including detailed breakdowns
"""

import time
import math
import logging
from typing import Any, Dict, Optional, Tuple, Literal
import torch
from accelerate import Accelerator

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Verbosity levels for metric collection
VerbosityLevel = Literal["minimal", "standard", "debug"]


class PerformanceLogger:
    """Comprehensive performance logging for training diagnostics.

    This class provides detailed logging capabilities to help diagnose training
    performance bottlenecks, monitor model stability, and track hardware utilization.
    Supports configurable verbosity levels to balance insight vs. monitoring overhead.
    """

    def __init__(self, verbosity: VerbosityLevel = "standard"):
        """Initialize the performance logger.

        Args:
            verbosity: Metric collection level ("minimal", "standard", "debug")
        """
        self.verbosity = verbosity
        self.loop_end_time = None
        self.step_start_time = None
        self.forward_pass_start_time = None
        self.forward_pass_end_time = None
        self.backward_pass_start_time = None
        self.backward_pass_end_time = None
        self.optimizer_step_start_time = None
        self.optimizer_step_end_time = None
        self.last_metrics: Dict[str, float] = {}

    def start_step_timing(self) -> None:
        """Start timing for the current training step."""
        self.step_start_time = time.perf_counter()
        self.data_loading_time = self.step_start_time - (
            self.loop_end_time or self.step_start_time
        )

    def start_forward_pass_timing(self) -> None:
        """Start timing for the forward pass."""
        self.forward_pass_start_time = time.perf_counter()

    def end_forward_pass_timing(self) -> None:
        """End timing for the forward pass."""
        self.forward_pass_end_time = time.perf_counter()

    def start_backward_pass_timing(self) -> None:
        """Start timing for the backward pass."""
        self.backward_pass_start_time = time.perf_counter()

    def end_backward_pass_timing(self) -> None:
        """End timing for the backward pass."""
        self.backward_pass_end_time = time.perf_counter()

    def start_optimizer_step_timing(self) -> None:
        """Start timing for the optimizer step."""
        self.optimizer_step_start_time = time.perf_counter()

    def end_optimizer_step_timing(self) -> None:
        """End timing for the optimizer step."""
        self.optimizer_step_end_time = time.perf_counter()

    def end_step_timing(self) -> None:
        """End timing for the current training step.

        Stores a snapshot of timing metrics for access in the next step.
        """
        self.loop_end_time = time.perf_counter()
        try:
            self.last_metrics = self.get_timing_metrics() or {}
        except Exception:
            self.last_metrics = {}

    def get_timing_metrics(self) -> Dict[str, float]:
        """Get timing metrics for the current step.

        Returns:
            Dict containing timing metrics in milliseconds
        """
        if not all(
            [
                self.step_start_time,
                self.forward_pass_start_time,
                self.forward_pass_end_time,
                self.backward_pass_start_time,
                self.backward_pass_end_time,
                self.optimizer_step_start_time,
                self.optimizer_step_end_time,
                self.loop_end_time,
            ]
        ):
            return {}

        try:
            timings = {
                "timing/data_loading_ms": self.data_loading_time * 1000,
                "timing/forward_pass_ms": (
                    self.forward_pass_end_time - self.forward_pass_start_time
                )  # type: ignore
                * 1000,
                "timing/backward_pass_ms": (
                    self.backward_pass_end_time - self.backward_pass_start_time
                )  # type: ignore
                * 1000,
                "timing/optimizer_step_ms": (
                    self.optimizer_step_end_time - self.optimizer_step_start_time
                )  # type: ignore
                * 1000,
            }

            step_total_time = self.loop_end_time - self.step_start_time  # type: ignore
            timings["timing/total_step_ms"] = step_total_time * 1000

            # Actual train iteration time (excludes validation/sampling overhead):
            # forward + backward + optimizer
            try:
                last_train_iter_ms = (
                    timings["timing/forward_pass_ms"]
                    + timings["timing/backward_pass_ms"]
                    + timings["timing/optimizer_step_ms"]
                )
                timings["timing/last_train_iter_ms"] = last_train_iter_ms
            except Exception:
                pass

            return timings
        except Exception as e:
            logger.debug(f"Failed to calculate timing metrics: {e}")
            return {}

    def get_last_train_iter_ms(self) -> float:
        """Return the last measured training-iteration duration in milliseconds.

        This is computed as forward+backward+optimizer of the previous completed step.
        """
        try:
            if self.last_metrics and "timing/last_train_iter_ms" in self.last_metrics:
                return float(self.last_metrics["timing/last_train_iter_ms"])
        except Exception:
            pass
        return 0.0

    def get_last_total_step_ms(self) -> float:
        """Return the last completed step total wall time in milliseconds."""
        try:
            if self.last_metrics and "timing/total_step_ms" in self.last_metrics:
                return float(self.last_metrics["timing/total_step_ms"])
        except Exception:
            pass
        return 0.0

    def get_model_statistics(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        is_main_process: bool = True,
        timesteps: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Get meaningful model output and target tensor statistics.

        Args:
            model_pred: Model predictions tensor
            target: Target tensor
            is_main_process: Whether this is the main process (to avoid redundant calculations)
            timesteps: Optional timesteps tensor for timestep-specific analysis
            global_step: Optional global step for convergence tracking

        Returns:
            Dict containing meaningful model and target statistics based on verbosity level
        """
        if not is_main_process:
            return {}

        try:
            with torch.no_grad():
                model_pred_f = model_pred.float()
                target_f = target.float()

                stats = {}

                # ESSENTIAL METRICS (all verbosity levels)
                essential_metrics = self._compute_essential_metrics(
                    model_pred_f, target_f
                )
                stats.update(essential_metrics)

                # STANDARD METRICS (standard and debug verbosity)
                if self.verbosity in ["standard", "debug"]:
                    standard_metrics = self._compute_standard_metrics(
                        model_pred_f, target_f, timesteps
                    )
                    stats.update(standard_metrics)

                # DEBUG METRICS (debug verbosity only)
                if self.verbosity == "debug":
                    debug_metrics = self._compute_debug_metrics(
                        model_pred_f, target_f, timesteps
                    )
                    stats.update(debug_metrics)

                return stats
        except Exception as e:
            logger.debug(f"Failed to calculate model statistics: {e}")
            return {}

    def _compute_essential_metrics(
        self, model_pred_f: torch.Tensor, target_f: torch.Tensor
    ) -> Dict[str, float]:
        """Compute essential metrics for all verbosity levels."""
        metrics = {}

        # 1. Convergence Analysis (Essential)
        pred_target_diff = (model_pred_f - target_f).abs()
        metrics.update(
            {
                "prediction/mean_absolute_error": pred_target_diff.mean().item(),
                "prediction/mean_squared_error": (pred_target_diff**2).mean().item(),
                "prediction/max_error": pred_target_diff.max().item(),
            }
        )

        # 2. Numerical Stability (Essential)
        nan_count = torch.isnan(model_pred_f).sum().item()
        inf_count = torch.isinf(model_pred_f).sum().item()
        metrics.update(
            {
                "prediction/nan_count": float(nan_count),
                "prediction/inf_count": float(inf_count),
                "prediction/numerical_stability": self._compute_numerical_stability(
                    model_pred_f
                ),
            }
        )

        return metrics

    def _compute_standard_metrics(
        self,
        model_pred_f: torch.Tensor,
        target_f: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute standard metrics for standard and debug verbosity."""
        metrics = {}

        # 1. Consolidated Scale Alignment (instead of 3 separate ratios)
        scale_alignment = self._compute_scale_alignment_score(model_pred_f, target_f)
        metrics["prediction/scale_alignment"] = scale_alignment

        # 2. Distribution Similarity
        distribution_similarity = self._compute_distribution_similarity(
            model_pred_f, target_f
        )
        metrics["prediction/distribution_similarity"] = distribution_similarity

        # 3. Timestep Analysis (summarized, not per-timestep)
        if timesteps is not None and timesteps.numel() > 0:
            timestep_summary = self._compute_timestep_summary(
                model_pred_f, target_f, timesteps
            )
            metrics.update(timestep_summary)

        # 4. Gradient Flow Indicators (when available)
        # Only check gradients on leaf tensors to avoid PyTorch warnings
        if model_pred_f.is_leaf and model_pred_f.grad is not None:
            gradient_metrics = self._compute_gradient_metrics(model_pred_f)
            metrics.update(gradient_metrics)

        return metrics

    def _compute_debug_metrics(
        self,
        model_pred_f: torch.Tensor,
        target_f: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute debug metrics for debug verbosity only."""
        metrics = {}

        # 1. Basic Statistics (Debug only)
        metrics.update(
            {
                "prediction/mean": model_pred_f.mean().item(),
                "prediction/std": model_pred_f.std().item(),
                "prediction/max": model_pred_f.max().item(),
                "prediction/min": model_pred_f.min().item(),
                "target_gt/mean": target_f.mean().item(),
                "target_gt/std": target_f.std().item(),
                "target_gt/max": target_f.max().item(),
                "target_gt/min": target_f.min().item(),
            }
        )

        # 2. Detailed Scale Analysis (Debug only)
        detailed_scale = self._compute_detailed_scale_analysis(model_pred_f, target_f)
        metrics.update(detailed_scale)

        # 3. Per-Timestep Breakdown (Debug only)
        if timesteps is not None and timesteps.numel() > 0:
            per_timestep_metrics = self._compute_per_timestep_metrics(
                model_pred_f, target_f, timesteps
            )
            metrics.update(per_timestep_metrics)

        return metrics

    def _compute_scale_alignment_score(
        self, model_pred_f: torch.Tensor, target_f: torch.Tensor
    ) -> float:
        """Compute consolidated scale alignment score (0 = no alignment, 1 = perfect alignment)."""
        try:
            # Compute range, std, and mean ratios
            pred_range = model_pred_f.max() - model_pred_f.min()
            target_range = target_f.max() - target_f.min()
            pred_std = model_pred_f.std()
            target_std = target_f.std()
            pred_mean = model_pred_f.mean()
            target_mean = target_f.mean()

            # Compute individual alignment scores
            range_ratio = (
                (pred_range / target_range).item() if target_range > 0 else 0.0
            )
            std_ratio = (pred_std / target_std).item() if target_std > 0 else 0.0
            mean_ratio = (pred_mean / target_mean).item() if target_mean != 0 else 0.0

            # Convert ratios to alignment scores (closer to 1.0 = better alignment)
            range_alignment = 1.0 - min(abs(range_ratio - 1.0), 1.0)
            std_alignment = 1.0 - min(abs(std_ratio - 1.0), 1.0)
            mean_alignment = 1.0 - min(abs(mean_ratio - 1.0), 1.0)

            # Weighted average (std alignment is most important)
            scale_alignment = (
                0.5 * std_alignment + 0.3 * range_alignment + 0.2 * mean_alignment
            )
            return scale_alignment
        except Exception:
            return 0.0

    def _compute_distribution_similarity(
        self, model_pred_f: torch.Tensor, target_f: torch.Tensor
    ) -> float:
        """Compute distribution similarity using histogram intersection."""
        try:
            # Normalize both to [0,1] for comparison
            pred_norm = (model_pred_f - model_pred_f.min()) / (
                model_pred_f.max() - model_pred_f.min() + 1e-8
            )
            target_norm = (target_f - target_f.min()) / (
                target_f.max() - target_f.min() + 1e-8
            )

            # Histogram-based similarity
            pred_hist = torch.histc(pred_norm, bins=10, min=0, max=1)
            target_hist = torch.histc(target_norm, bins=10, min=0, max=1)

            # Normalize histograms
            pred_sum = pred_hist.sum()
            target_sum = target_hist.sum()
            pred_hist = pred_hist / (pred_sum + 1e-8) if pred_sum > 0 else pred_hist
            target_hist = (
                target_hist / (target_sum + 1e-8) if target_sum > 0 else target_hist
            )

            # Histogram intersection similarity
            hist_intersection = torch.minimum(pred_hist, target_hist).sum().item()
            return hist_intersection
        except Exception:
            return 0.0

    def _compute_timestep_summary(
        self,
        model_pred_f: torch.Tensor,
        target_f: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute summarized timestep metrics (percentiles instead of per-timestep)."""
        metrics = {}
        try:
            # Group by timestep and compute errors
            unique_timesteps = torch.unique(timesteps)
            if len(unique_timesteps) > 1:
                timestep_errors = []
                for t in unique_timesteps:
                    mask = timesteps == t
                    if mask.sum() > 0:
                        pred_t = model_pred_f[mask]
                        target_t = target_f[mask]
                        mae_t = (pred_t - target_t).abs().mean().item()
                        timestep_errors.append(mae_t)

                if len(timestep_errors) > 1:
                    timestep_errors_tensor = torch.tensor(timestep_errors)

                    # Compute percentiles with bounds checking
                    sorted_errors = torch.sort(timestep_errors_tensor)[0]
                    n_errors = len(sorted_errors)

                    # Ensure indices are within bounds
                    p50_idx = min(n_errors // 2, n_errors - 1)
                    p90_idx = min(int(n_errors * 0.9), n_errors - 1)
                    p99_idx = min(int(n_errors * 0.99), n_errors - 1)

                    metrics.update(
                        {
                            "prediction/mae_p50": sorted_errors[p50_idx].item(),
                            "prediction/mae_p90": sorted_errors[p90_idx].item(),
                            "prediction/mae_p99": sorted_errors[p99_idx].item(),
                            "prediction/mae_min": sorted_errors[0].item(),
                            "prediction/mae_max": sorted_errors[-1].item(),
                        }
                    )

                    # Overall correlation
                    timestep_values = unique_timesteps.float()
                    correlation = torch.corrcoef(
                        torch.stack([timestep_values, timestep_errors_tensor])
                    )[0, 1].item()
                    if not math.isnan(correlation):
                        metrics["prediction/timestep_error_correlation"] = correlation

        except Exception as e:
            logger.debug(f"Failed to compute timestep summary: {e}")

        return metrics

    def _compute_gradient_metrics(self, model_pred_f: torch.Tensor) -> Dict[str, float]:
        """Compute gradient flow indicators."""
        metrics = {}
        try:
            # Only compute gradient metrics for leaf tensors with gradients
            if model_pred_f.is_leaf and model_pred_f.grad is not None:
                grad_norm = model_pred_f.grad.norm().item()
                metrics.update(
                    {
                        "prediction/gradient_norm": grad_norm,
                        "prediction/gradient_mean": model_pred_f.grad.mean().item(),
                        "prediction/gradient_std": model_pred_f.grad.std().item(),
                    }
                )
        except Exception:
            pass
        return metrics

    def _compute_detailed_scale_analysis(
        self, model_pred_f: torch.Tensor, target_f: torch.Tensor
    ) -> Dict[str, float]:
        """Compute detailed scale analysis (debug only)."""
        metrics = {}
        try:
            pred_range = model_pred_f.max() - model_pred_f.min()
            target_range = target_f.max() - target_f.min()
            pred_std = model_pred_f.std()
            target_std = target_f.std()

            metrics.update(
                {
                    "prediction/range_ratio": (
                        (pred_range / target_range).item() if target_range > 0 else 0.0
                    ),
                    "prediction/std_ratio": (
                        (pred_std / target_std).item() if target_std > 0 else 0.0
                    ),
                    "prediction/mean_ratio": (
                        (model_pred_f.mean() / target_f.mean()).item()
                        if target_f.mean() != 0
                        else 0.0
                    ),
                }
            )
        except Exception:
            pass
        return metrics

    def _compute_per_timestep_metrics(
        self,
        model_pred_f: torch.Tensor,
        target_f: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-timestep metrics (debug only)."""
        metrics = {}
        try:
            unique_timesteps = torch.unique(timesteps)
            if len(unique_timesteps) > 1:
                for t in unique_timesteps:
                    mask = timesteps == t
                    if mask.sum() > 0:
                        pred_t = model_pred_f[mask]
                        target_t = target_f[mask]

                        mae_t = (pred_t - target_t).abs().mean().item()
                        mse_t = ((pred_t - target_t) ** 2).mean().item()

                        metrics[f"prediction/mae_t{t.item():.0f}"] = mae_t
                        metrics[f"prediction/mse_t{t.item():.0f}"] = mse_t

                        pred_std_t = pred_t.std().item()
                        target_std_t = target_t.std().item()
                        if target_std_t > 0:
                            scale_ratio_t = pred_std_t / target_std_t
                            metrics[f"prediction/scale_ratio_t{t.item():.0f}"] = (
                                scale_ratio_t
                            )
        except Exception as e:
            logger.debug(f"Failed to compute per-timestep metrics: {e}")
        return metrics

    def _compute_numerical_stability(self, tensor: torch.Tensor) -> float:
        """Compute numerical stability score (0 = very unstable, 1 = very stable)."""
        try:
            # Check for extreme values
            abs_tensor = tensor.abs()
            max_val = abs_tensor.max().item()

            if max_val == 0:
                return 1.0  # Perfect stability (all zeros)

            # Normalize by max value
            normalized = abs_tensor / max_val

            # Stability score based on how close values are to reasonable range
            # Values close to 1.0 are stable, very large or very small values reduce stability
            stability_score = 1.0 - torch.clamp(normalized - 1.0, min=0).mean().item()

            return max(0.0, min(1.0, stability_score))
        except Exception:
            return 0.0

    def get_hardware_metrics(self) -> Dict[str, str]:
        """Get enhanced hardware utilization metrics.

        Returns:
            Dict containing hardware metrics for progress bar display
        """
        metrics = {}

        try:
            # Basic CUDA memory metrics
            if torch.cuda.is_available():
                device = torch.device("cuda")
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)

                if peak_allocated > 0.1:
                    metrics["peak"] = f"{peak_allocated:.2f} GiB"

            # Enhanced GPU metrics with pynvml
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Add VRAM usage
                vram_used = float(meminfo.used) / 1024**3
                metrics["peak"] = f"{vram_used:.2f} GiB"

                # Add GPU utilization
                metrics["util"] = f"{utilization.gpu}%"

                # Add memory utilization
                memory_util = (meminfo.used / meminfo.total) * 100  # type: ignore
                metrics["mem_util"] = f"{memory_util:.1f}%"

                pynvml.nvmlShutdown()

            except Exception as e:
                logger.debug(f"Failed to get pynvml metrics: {e}")
                # Fallback to basic CUDA metrics
                pass

        except Exception as e:
            logger.debug(f"Failed to get hardware metrics: {e}")

        return metrics

    def get_comprehensive_metrics(
        self,
        model_pred: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        is_main_process: bool = True,
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics.

        Args:
            model_pred: Model predictions tensor (optional)
            target: Target tensor (optional)
            is_main_process: Whether this is the main process

        Returns:
            Dict containing all performance metrics
        """
        metrics = {}

        # Add timing metrics
        timing_metrics = self.get_timing_metrics()
        metrics.update(timing_metrics)

        # Add model statistics if tensors are provided
        if model_pred is not None and target is not None:
            model_stats = self.get_model_statistics(model_pred, target, is_main_process)
            metrics.update(model_stats)

        # Add hardware metrics
        hardware_metrics = self.get_hardware_metrics()
        metrics.update(hardware_metrics)

        return metrics

    def log_performance_summary(
        self, step: int, timing_metrics: Dict[str, float]
    ) -> None:
        """Log a performance summary for the current step.

        Args:
            step: Current training step
            timing_metrics: Timing metrics dictionary
        """
        if not timing_metrics:
            return

        try:
            total_time = timing_metrics.get("timing/total_step_ms", 0)
            forward_time = timing_metrics.get("timing/forward_pass_ms", 0)
            backward_time = timing_metrics.get("timing/backward_pass_ms", 0)
            optimizer_time = timing_metrics.get("timing/optimizer_step_ms", 0)
            data_loading_time = timing_metrics.get("timing/data_loading_ms", 0)

            if total_time > 0:
                forward_pct = (forward_time / total_time) * 100
                backward_pct = (backward_time / total_time) * 100
                optimizer_pct = (optimizer_time / total_time) * 100
                data_loading_pct = (data_loading_time / total_time) * 100

                logger.info(
                    f"Step {step} Performance Summary: "
                    f"Total={total_time:.1f}ms "
                    f"(Forward: {forward_pct:.1f}%, "
                    f"Backward: {backward_pct:.1f}%, "
                    f"Optimizer: {optimizer_pct:.1f}%, "
                    f"Data: {data_loading_pct:.1f}%)"
                )

        except Exception as e:
            logger.debug(f"Failed to log performance summary: {e}")


# Global instance for easy access
performance_logger = PerformanceLogger(verbosity="standard")


def start_step_timing() -> None:
    """Start timing for the current training step."""
    performance_logger.start_step_timing()


def start_forward_pass_timing() -> None:
    """Start timing for the forward pass."""
    performance_logger.start_forward_pass_timing()


def end_forward_pass_timing() -> None:
    """End timing for the forward pass."""
    performance_logger.end_forward_pass_timing()


def start_backward_pass_timing() -> None:
    """Start timing for the backward pass."""
    performance_logger.start_backward_pass_timing()


def end_backward_pass_timing() -> None:
    """End timing for the backward pass."""
    performance_logger.end_backward_pass_timing()


def start_optimizer_step_timing() -> None:
    """Start timing for the optimizer step."""
    performance_logger.start_optimizer_step_timing()


def end_optimizer_step_timing() -> None:
    """End timing for the optimizer step."""
    performance_logger.end_optimizer_step_timing()


def end_step_timing() -> None:
    """End timing for the current training step."""
    performance_logger.end_step_timing()


def get_timing_metrics() -> Dict[str, float]:
    """Get timing metrics for the current step."""
    return performance_logger.get_timing_metrics()


def get_last_train_iter_ms() -> float:
    """Get last measured training-iteration duration (ms)."""
    return performance_logger.get_last_train_iter_ms()


def get_last_total_step_ms() -> float:
    """Get last completed step total duration (ms)."""
    return performance_logger.get_last_total_step_ms()


def configure_verbosity(verbosity: VerbosityLevel) -> None:
    """Configure the verbosity level for metric collection.

    Args:
        verbosity: Metric collection level ("minimal", "standard", "debug")
    """
    global performance_logger
    performance_logger = PerformanceLogger(verbosity=verbosity)
    logger.info(f"Performance logger verbosity set to: {verbosity}")


def get_model_statistics(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    is_main_process: bool = True,
    timesteps: Optional[torch.Tensor] = None,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    """Get meaningful model output and target tensor statistics."""
    return performance_logger.get_model_statistics(
        model_pred, target, is_main_process, timesteps, global_step
    )


def get_hardware_metrics() -> Dict[str, str]:
    """Get enhanced hardware utilization metrics."""
    return performance_logger.get_hardware_metrics()


def get_comprehensive_metrics(
    model_pred: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    is_main_process: bool = True,
) -> Dict[str, Any]:
    """Get comprehensive performance metrics."""
    return performance_logger.get_comprehensive_metrics(
        model_pred, target, is_main_process
    )


def log_performance_summary(step: int, timing_metrics: Dict[str, float]) -> None:
    """Log a performance summary for the current step."""
    performance_logger.log_performance_summary(step, timing_metrics)


# --- GPU memory tracing utilities ---
from typing import Dict as _Dict  # alias to avoid shadowing above imports


def snapshot_gpu_memory(tag: str = "mem") -> _Dict[str, float]:
    """Capture a GPU memory snapshot using both PyTorch and NVML if available.

    Args:
        tag: Label to prefix metrics with for easier correlation

    Returns:
        Dictionary of memory metrics in GiB
    """
    stats: _Dict[str, float] = {}
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                stats.update(
                    {
                        f"{tag}/torch_allocated_gb": float(allocated),
                        f"{tag}/torch_reserved_gb": float(reserved),
                        f"{tag}/torch_free_gb": float(free_gb),
                        f"{tag}/torch_total_gb": float(total_gb),
                    }
                )
            except Exception:
                pass

            # NVML snapshot
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats.update(
                    {
                        f"{tag}/nvml_used_gb": float(meminfo.used) / (1024**3),
                        f"{tag}/nvml_free_gb": float(meminfo.free) / (1024**3),
                        f"{tag}/nvml_total_gb": float(meminfo.total) / (1024**3),
                    }
                )
            except Exception:
                # NVML optional
                pass
            finally:
                try:
                    # Safe shutdown if initialized
                    import pynvml  # type: ignore

                    if hasattr(pynvml, "nvmlShutdown"):
                        pynvml.nvmlShutdown()
                except Exception:
                    pass

        # Log rounded values for readability
        logger.info(
            "GPU MEM SNAPSHOT %s: %s",
            tag,
            {k: round(v, 3) for k, v in stats.items()},
        )
    except Exception as e:
        logger.warning("snapshot_gpu_memory failed for %s: %s", tag, e)
    return stats


def force_cuda_cleanup(tag: str = "cleanup") -> None:
    """Aggressively attempt to release CUDA memory from this process.

    This captures a post-cleanup snapshot for verification.

    Args:
        tag: Label used in the post-cleanup snapshot
    """
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        import gc as _gc

        _gc.collect()
    except Exception as e:
        logger.warning("force_cuda_cleanup encountered an error: %s", e)
    finally:
        snapshot_gpu_memory(f"{tag}/after")
