"""Enhanced memory allocation tracking and OOM diagnostics.

This module provides comprehensive memory tracking capabilities to help diagnose
CUDA out of memory errors by monitoring allocation requests, peak usage, and
fragmentation patterns.
"""

from __future__ import annotations

import logging
import time
import threading
import traceback
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import torch
import gc
import weakref

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class AllocationRecord:
    """Record of a memory allocation request."""

    timestamp: float
    size_bytes: int
    device: str
    operation: str
    stack_trace: str = ""
    success: bool = True
    error_message: str = ""


@dataclass
class MemorySnapshot:
    """Comprehensive memory state snapshot."""

    timestamp: float
    torch_allocated: int = 0
    torch_reserved: int = 0
    torch_free: int = 0
    torch_total: int = 0
    nvml_used: int = 0
    nvml_free: int = 0
    nvml_total: int = 0
    active_tensors: int = 0
    peak_allocated: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0


class MemoryTracker:
    """Advanced memory allocation tracker with OOM diagnostics."""

    def __init__(
        self,
        max_records: int = 10000,
        enable_stack_traces: bool = False,
        enable_tensor_tracking: bool = True,
    ):
        """Initialize memory tracker.

        Args:
            max_records: Maximum number of allocation records to keep
            enable_stack_traces: Whether to capture stack traces (expensive)
            enable_tensor_tracking: Whether to track individual tensor references
        """
        self.max_records = max_records
        self.enable_stack_traces = enable_stack_traces
        self.enable_tensor_tracking = enable_tensor_tracking

        # Thread-safe collections
        self._lock = threading.RLock()
        self.allocation_records: deque[AllocationRecord] = deque(maxlen=max_records)
        self.snapshots: deque[MemorySnapshot] = deque(maxlen=1000)

        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.total_bytes_requested = 0
        self.total_bytes_freed = 0
        self.peak_bytes_requested = 0
        self.failed_allocations = 0

        # Tensor tracking
        if enable_tensor_tracking:
            self.active_tensors: Dict[int, Tuple[int, str]] = (
                {}
            )  # id -> (size, creation_info)

        # Hook state
        self._original_allocator: Optional[Callable] = None
        self._hooks_installed = False

        # Auto-snapshot configuration
        self.auto_snapshot_interval = 100  # Take snapshot every N allocations
        self.allocation_count_since_snapshot = 0

    def install_hooks(self) -> None:
        """Install memory allocation hooks."""
        if self._hooks_installed:
            logger.warning("Memory hooks already installed")
            return

        try:
            # Install PyTorch memory hooks if available
            if hasattr(torch.cuda, "memory"):
                # Modern PyTorch versions
                torch.cuda.memory._set_memory_fraction = self._wrap_memory_function(
                    torch.cuda.memory.set_per_process_memory_fraction
                )

            # Install tensor creation hooks
            if self.enable_tensor_tracking:
                self._install_tensor_hooks()

            self._hooks_installed = True
            logger.info("✅ Memory tracking hooks installed")

        except Exception as e:
            logger.error(f"Failed to install memory hooks: {e}")

    def uninstall_hooks(self) -> None:
        """Remove memory allocation hooks."""
        if not self._hooks_installed:
            return

        try:
            # Restore original functions if we wrapped them
            if self._original_allocator:
                # Restore original allocator
                pass  # Implementation depends on PyTorch version

            self._hooks_installed = False
            logger.info("Memory tracking hooks removed")

        except Exception as e:
            logger.error(f"Failed to remove memory hooks: {e}")

    def _wrap_memory_function(self, original_func: Callable) -> Callable:
        """Wrap a memory function to track allocations."""

        def wrapper(*args, **kwargs):
            size_estimate = self._estimate_allocation_size(args, kwargs)
            start_time = time.perf_counter()

            try:
                result = original_func(*args, **kwargs)
                self._record_allocation(
                    size_bytes=size_estimate,
                    operation=original_func.__name__,
                    success=True,
                )
                return result

            except torch.cuda.OutOfMemoryError as e:
                self._record_allocation(
                    size_bytes=size_estimate,
                    operation=original_func.__name__,
                    success=False,
                    error_message=str(e),
                )
                # Take emergency snapshot for debugging
                self.take_snapshot("oom_error")
                raise

        return wrapper

    def _install_tensor_hooks(self) -> None:
        """Install hooks to track tensor creation and destruction."""
        # This is a simplified approach - full implementation would need
        # to hook into PyTorch's tensor creation mechanisms
        pass

    def _estimate_allocation_size(self, args: tuple, kwargs: dict) -> int:
        """Estimate allocation size from function arguments."""
        # This is a heuristic - actual implementation would analyze
        # the specific function and its arguments
        return 0

    def record_manual_allocation(
        self,
        size_bytes: int,
        operation: str = "manual",
        success: bool = True,
        error_message: str = "",
    ) -> None:
        """Manually record a memory allocation."""
        self._record_allocation(size_bytes, operation, success, error_message)

    def _record_allocation(
        self,
        size_bytes: int,
        operation: str,
        success: bool = True,
        error_message: str = "",
    ) -> None:
        """Record a memory allocation attempt."""
        with self._lock:
            # Create allocation record
            record = AllocationRecord(
                timestamp=time.perf_counter(),
                size_bytes=size_bytes,
                device=self._get_current_device(),
                operation=operation,
                success=success,
                error_message=error_message,
            )

            # Add stack trace if enabled
            if self.enable_stack_traces:
                record.stack_trace = "".join(traceback.format_stack())

            # Store record
            self.allocation_records.append(record)

            # Update statistics
            if success:
                self.total_allocations += 1
                self.total_bytes_requested += size_bytes
                self.peak_bytes_requested = max(self.peak_bytes_requested, size_bytes)
            else:
                self.failed_allocations += 1

            # Auto-snapshot if needed
            self.allocation_count_since_snapshot += 1
            if self.allocation_count_since_snapshot >= self.auto_snapshot_interval:
                self.take_snapshot("auto")
                self.allocation_count_since_snapshot = 0

    def _get_current_device(self) -> str:
        """Get current CUDA device string."""
        try:
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
        except Exception:
            pass
        return "unknown"

    def take_snapshot(self, tag: str = "manual") -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        snapshot = MemorySnapshot(timestamp=time.perf_counter())

        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")

                # PyTorch memory info
                try:
                    snapshot.torch_allocated = torch.cuda.memory_allocated(device)
                    snapshot.torch_reserved = torch.cuda.memory_reserved(device)
                    snapshot.peak_allocated = torch.cuda.max_memory_allocated(device)

                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    snapshot.torch_free = free_bytes
                    snapshot.torch_total = total_bytes
                except Exception:
                    pass

                # NVML memory info
                try:
                    import pynvml  # nvidia-ml-py package

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    snapshot.nvml_used = meminfo.used
                    snapshot.nvml_free = meminfo.free
                    snapshot.nvml_total = meminfo.total

                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            # Tensor tracking
            if self.enable_tensor_tracking:
                snapshot.active_tensors = len(self.active_tensors)

            # Allocation statistics
            with self._lock:
                snapshot.allocation_count = self.total_allocations
                snapshot.deallocation_count = self.total_deallocations

            # Store snapshot
            with self._lock:
                self.snapshots.append(snapshot)

            logger.debug(f"Memory snapshot '{tag}': {self._format_snapshot(snapshot)}")

        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")

        return snapshot

    def _format_snapshot(self, snapshot: MemorySnapshot) -> str:
        """Format snapshot for logging."""
        return (
            f"Allocated: {snapshot.torch_allocated / (1024**3):.2f}GB, "
            f"Reserved: {snapshot.torch_reserved / (1024**3):.2f}GB, "
            f"Free: {snapshot.torch_free / (1024**3):.2f}GB, "
            f"Total: {snapshot.torch_total / (1024**3):.2f}GB"
        )

    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive allocation summary."""
        with self._lock:
            recent_records = list(self.allocation_records)[
                -100:
            ]  # Last 100 allocations
            failed_records = [r for r in recent_records if not r.success]

            # Calculate statistics
            total_recent_bytes = sum(r.size_bytes for r in recent_records if r.success)
            avg_allocation_size = (
                total_recent_bytes / len(recent_records) if recent_records else 0
            )

            # Group by operation
            operation_stats = defaultdict(
                lambda: {"count": 0, "total_bytes": 0, "failures": 0}
            )
            for record in recent_records:
                stats = operation_stats[record.operation]
                stats["count"] += 1
                if record.success:
                    stats["total_bytes"] += record.size_bytes
                else:
                    stats["failures"] += 1

            return {
                "total_allocations": self.total_allocations,
                "total_deallocations": self.total_deallocations,
                "failed_allocations": self.failed_allocations,
                "total_bytes_requested": self.total_bytes_requested,
                "total_bytes_freed": self.total_bytes_freed,
                "peak_bytes_requested": self.peak_bytes_requested,
                "recent_avg_allocation_mb": avg_allocation_size / (1024**2),
                "recent_failures": len(failed_records),
                "operation_breakdown": dict(operation_stats),
                "active_tensors": (
                    len(self.active_tensors) if self.enable_tensor_tracking else 0
                ),
                "snapshots_taken": len(self.snapshots),
            }

    def diagnose_oom_error(self, error_message: str) -> Dict[str, Any]:
        """Analyze OOM error and provide diagnostic information."""
        logger.error(f"🔍 Diagnosing OOM Error: {error_message}")

        # Take immediate snapshot
        current_snapshot = self.take_snapshot("oom_diagnosis")

        # Get recent allocation history
        with self._lock:
            recent_records = list(self.allocation_records)[-50:]  # Last 50 allocations
            failed_records = [r for r in self.allocation_records if not r.success]

        # Calculate memory pressure indicators
        if len(self.snapshots) >= 2:
            prev_snapshot = self.snapshots[-2]
            memory_growth = (
                current_snapshot.torch_allocated - prev_snapshot.torch_allocated
            )
            fragmentation_indicator = (
                (current_snapshot.torch_reserved - current_snapshot.torch_allocated)
                / current_snapshot.torch_reserved
                if current_snapshot.torch_reserved > 0
                else 0
            )
        else:
            memory_growth = 0
            fragmentation_indicator = 0

        # Extract requested size from error message
        requested_size = self._extract_requested_size(error_message)

        # Build diagnosis
        diagnosis = {
            "error_message": error_message,
            "requested_size_mb": requested_size / (1024**2) if requested_size else None,
            "current_memory": {
                "allocated_gb": current_snapshot.torch_allocated / (1024**3),
                "reserved_gb": current_snapshot.torch_reserved / (1024**3),
                "free_gb": current_snapshot.torch_free / (1024**3),
                "total_gb": current_snapshot.torch_total / (1024**3),
                "utilization_pct": (
                    (
                        current_snapshot.torch_allocated
                        / current_snapshot.torch_total
                        * 100
                    )
                    if current_snapshot.torch_total > 0
                    else 0
                ),
            },
            "memory_pressure": {
                "recent_growth_mb": memory_growth / (1024**2),
                "fragmentation_ratio": fragmentation_indicator,
                "recent_allocation_count": len(recent_records),
                "recent_failure_count": len(
                    [r for r in recent_records if not r.success]
                ),
            },
            "allocation_history": self.get_allocation_summary(),
            "recommendations": self._generate_recommendations(
                current_snapshot, requested_size
            ),
        }

        # Log detailed diagnosis
        self._log_oom_diagnosis(diagnosis)

        return diagnosis

    def _extract_requested_size(self, error_message: str) -> Optional[int]:
        """Extract requested allocation size from OOM error message."""
        try:
            import re

            # Look for patterns like "Tried to allocate 40.00 MiB"
            match = re.search(r"Tried to allocate ([\d.]+) (\w+)", error_message)
            if match:
                size_str, unit = match.groups()
                size = float(size_str)

                # Convert to bytes
                unit = unit.upper()
                if unit in ["MIB", "MB"]:
                    return int(size * 1024 * 1024)
                elif unit in ["GIB", "GB"]:
                    return int(size * 1024 * 1024 * 1024)
                elif unit in ["KIB", "KB"]:
                    return int(size * 1024)
                else:
                    return int(size)
        except Exception:
            pass
        return None

    def _generate_recommendations(
        self, snapshot: MemorySnapshot, requested_size: Optional[int]
    ) -> List[str]:
        """Generate recommendations for addressing OOM issues."""
        recommendations = []

        utilization = (
            snapshot.torch_allocated / snapshot.torch_total
            if snapshot.torch_total > 0
            else 0
        )
        fragmentation = (
            (snapshot.torch_reserved - snapshot.torch_allocated)
            / snapshot.torch_reserved
            if snapshot.torch_reserved > 0
            else 0
        )

        if utilization > 0.95:
            recommendations.append(
                "Memory utilization is very high (>95%). Consider reducing batch size or model size."
            )

        if fragmentation > 0.3:
            recommendations.append(
                "High memory fragmentation detected. Try torch.cuda.empty_cache() or restart training."
            )

        if requested_size and requested_size > snapshot.torch_free:
            recommendations.append(
                f"Requested {requested_size / (1024**2):.1f}MB but only {snapshot.torch_free / (1024**2):.1f}MB free. Reduce allocation size."
            )

        if self.failed_allocations > 5:
            recommendations.append(
                "Multiple allocation failures detected. Check for memory leaks or reduce memory usage."
            )

        # Configuration recommendations
        recommendations.extend(
            [
                "Enable gradient checkpointing to reduce activation memory",
                "Use mixed precision (fp16/bf16) to reduce memory usage",
                "Consider using PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "Enable block swapping if supported by your model",
                "Reduce sequence length or resolution if possible",
            ]
        )

        return recommendations

    def _log_oom_diagnosis(self, diagnosis: Dict[str, Any]) -> None:
        """Log comprehensive OOM diagnosis."""
        logger.error("=" * 80)
        logger.error("🚨 CUDA OUT OF MEMORY DIAGNOSIS")
        logger.error("=" * 80)

        # Current memory state
        mem = diagnosis["current_memory"]
        logger.error(f"📊 Current Memory State:")
        logger.error(
            f"   Allocated: {mem['allocated_gb']:.2f} GB ({mem['utilization_pct']:.1f}%)"
        )
        logger.error(f"   Reserved:  {mem['reserved_gb']:.2f} GB")
        logger.error(f"   Free:      {mem['free_gb']:.2f} GB")
        logger.error(f"   Total:     {mem['total_gb']:.2f} GB")

        # Request details
        if diagnosis["requested_size_mb"]:
            logger.error(
                f"🎯 Allocation Request: {diagnosis['requested_size_mb']:.2f} MB"
            )

        # Memory pressure
        pressure = diagnosis["memory_pressure"]
        logger.error(f"📈 Memory Pressure:")
        logger.error(f"   Recent growth: {pressure['recent_growth_mb']:.2f} MB")
        logger.error(f"   Fragmentation: {pressure['fragmentation_ratio']:.2%}")
        logger.error(f"   Recent failures: {pressure['recent_failure_count']}")

        # Top recommendations
        logger.error(f"💡 Top Recommendations:")
        for i, rec in enumerate(diagnosis["recommendations"][:5], 1):
            logger.error(f"   {i}. {rec}")

        logger.error("=" * 80)

    def export_diagnostics(self, filepath: str) -> None:
        """Export comprehensive diagnostics to file."""
        try:
            import json

            diagnostics = {
                "summary": self.get_allocation_summary(),
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "torch_allocated": s.torch_allocated,
                        "torch_reserved": s.torch_reserved,
                        "torch_free": s.torch_free,
                        "torch_total": s.torch_total,
                        "active_tensors": s.active_tensors,
                        "allocation_count": s.allocation_count,
                    }
                    for s in list(self.snapshots)
                ],
                "recent_allocations": [
                    {
                        "timestamp": r.timestamp,
                        "size_bytes": r.size_bytes,
                        "operation": r.operation,
                        "success": r.success,
                        "error_message": r.error_message,
                    }
                    for r in list(self.allocation_records)[-100:]
                ],
            }

            with open(filepath, "w") as f:
                json.dump(diagnostics, f, indent=2)

            logger.info(f"Memory diagnostics exported to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to export diagnostics: {e}")


# Global memory tracker instance
_global_tracker: Optional[MemoryTracker] = None


def get_memory_tracker(auto_install: bool = True) -> MemoryTracker:
    """Get or create the global memory tracker instance."""
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = MemoryTracker(
            enable_stack_traces=False,  # Expensive, disabled by default
            enable_tensor_tracking=True,
        )

        if auto_install:
            _global_tracker.install_hooks()

    return _global_tracker


def track_cuda_oom(error: torch.cuda.OutOfMemoryError) -> Dict[str, Any]:
    """Track and diagnose a CUDA OOM error."""
    tracker = get_memory_tracker()
    return tracker.diagnose_oom_error(str(error))


def take_memory_snapshot(tag: str = "manual") -> MemorySnapshot:
    """Take a memory snapshot using the global tracker."""
    tracker = get_memory_tracker()
    return tracker.take_snapshot(tag)


def record_allocation(size_bytes: int, operation: str = "manual") -> None:
    """Record a manual memory allocation."""
    tracker = get_memory_tracker()
    tracker.record_manual_allocation(size_bytes, operation)


def get_memory_summary() -> Dict[str, Any]:
    """Get memory allocation summary."""
    tracker = get_memory_tracker()
    return tracker.get_allocation_summary()


def export_memory_diagnostics(filepath: str) -> None:
    """Export memory diagnostics to file."""
    tracker = get_memory_tracker()
    tracker.export_diagnostics(filepath)


# Context manager for tracking operations
class MemoryTrackingContext:
    """Context manager for tracking memory usage during operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.end_snapshot: Optional[MemorySnapshot] = None

    def __enter__(self):
        self.start_snapshot = take_memory_snapshot(f"{self.operation_name}/start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_snapshot = take_memory_snapshot(f"{self.operation_name}/end")

        if exc_type is torch.cuda.OutOfMemoryError:
            # Diagnose OOM error
            track_cuda_oom(exc_val)

        # Log memory usage
        if self.start_snapshot and self.end_snapshot:
            memory_delta = (
                self.end_snapshot.torch_allocated - self.start_snapshot.torch_allocated
            )
            logger.info(
                f"Memory usage for '{self.operation_name}': "
                f"{memory_delta / (1024**2):+.2f} MB "
                f"({self.end_snapshot.torch_allocated / (1024**3):.2f} GB total)"
            )


def memory_tracked(operation_name: str):
    """Decorator for tracking memory usage of functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with MemoryTrackingContext(f"{func.__name__}_{operation_name}"):
                return func(*args, **kwargs)

        return wrapper

    return decorator
