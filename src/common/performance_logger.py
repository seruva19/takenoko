"""Performance logging functionality for training diagnostics.

This module provides comprehensive logging capabilities for:
- Performance timing analysis (data loading, forward/backward pass, optimizer step)
- Model output and target tensor statistics
- Enhanced hardware utilization metrics
"""

import time
import logging
from typing import Any, Dict, Optional, Tuple
import torch
from accelerate import Accelerator

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class PerformanceLogger:
    """Comprehensive performance logging for training diagnostics.

    This class provides detailed logging capabilities to help diagnose training
    performance bottlenecks, monitor model stability, and track hardware utilization.
    """

    def __init__(self):
        """Initialize the performance logger."""
        self.loop_end_time = None
        self.step_start_time = None
        self.forward_pass_start_time = None
        self.forward_pass_end_time = None
        self.backward_pass_start_time = None
        self.backward_pass_end_time = None
        self.optimizer_step_start_time = None
        self.optimizer_step_end_time = None

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
        """End timing for the current training step."""
        self.loop_end_time = time.perf_counter()

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

            return timings
        except Exception as e:
            logger.debug(f"Failed to calculate timing metrics: {e}")
            return {}

    def get_model_statistics(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        is_main_process: bool = True,
    ) -> Dict[str, float]:
        """Get model output and target tensor statistics.

        Args:
            model_pred: Model predictions tensor
            target: Target tensor
            is_main_process: Whether this is the main process (to avoid redundant calculations)

        Returns:
            Dict containing model and target statistics
        """
        if not is_main_process:
            return {}

        try:
            with torch.no_grad():
                model_pred_f = model_pred.float()
                target_f = target.float()

                model_stats = {
                    "prediction/mean": model_pred_f.mean().item(),
                    "prediction/std": model_pred_f.std().item(),
                    "prediction/max": model_pred_f.max().item(),
                    "prediction/min": model_pred_f.min().item(),
                }
                target_stats = {
                    "target_gt/mean": target_f.mean().item(),
                    "target_gt/std": target_f.std().item(),
                    "target_gt/max": target_f.max().item(),
                    "target_gt/min": target_f.min().item(),
                }

                # Combine stats
                stats = {}
                stats.update(model_stats)
                stats.update(target_stats)

                return stats
        except Exception as e:
            logger.debug(f"Failed to calculate model statistics: {e}")
            return {}

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
performance_logger = PerformanceLogger()


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


def get_model_statistics(
    model_pred: torch.Tensor, target: torch.Tensor, is_main_process: bool = True
) -> Dict[str, float]:
    """Get model output and target tensor statistics."""
    return performance_logger.get_model_statistics(model_pred, target, is_main_process)


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
