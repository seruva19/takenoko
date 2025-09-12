"""Memory tracking manager for coordinating memory diagnostics across the application.

This module provides a centralized manager for memory tracking functionality,
including initialization, configuration, and diagnostics display.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, Optional
import torch
import argparse

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Import memory tracking utilities
try:
    from utils.memory_tracker import get_memory_tracker, MemoryTracker
    from common.performance_logger import handle_cuda_oom_error, track_memory_usage

    MEMORY_TRACKING_AVAILABLE = True
except ImportError:
    MEMORY_TRACKING_AVAILABLE = False
    logger.debug("Memory tracking utilities not available")


class MemoryTrackingManager:
    """Centralized manager for memory tracking functionality."""

    def __init__(self, args: argparse.Namespace):
        """Initialize memory tracking manager.

        Args:
            args: Configuration arguments containing memory tracking settings
        """
        self.args = args
        self.tracker: Optional[MemoryTracker] = None
        self.is_initialized = False

    def setup_memory_tracking(self) -> None:
        """Setup memory tracking based on configuration."""
        if not MEMORY_TRACKING_AVAILABLE:
            logger.debug("Memory tracking not available - skipping setup")
            return

        # Check if memory tracking is enabled
        memory_tracking_enabled = getattr(self.args, "memory_tracking_enabled", False)
        if not memory_tracking_enabled:
            logger.debug("Memory tracking disabled in configuration")
            return

        try:
            # Get configuration parameters
            max_records = getattr(self.args, "memory_tracking_max_records", 10000)
            enable_stack_traces = getattr(
                self.args, "memory_tracking_stack_traces", False
            )
            enable_tensor_tracking = getattr(
                self.args, "memory_tracking_tensor_tracking", True
            )
            auto_snapshot_interval = getattr(
                self.args, "memory_tracking_auto_snapshot_interval", 100
            )

            # Initialize memory tracker with configuration
            self.tracker = get_memory_tracker(auto_install=True)
            self.tracker.max_records = max_records
            self.tracker.enable_stack_traces = enable_stack_traces
            self.tracker.enable_tensor_tracking = enable_tensor_tracking
            self.tracker.auto_snapshot_interval = auto_snapshot_interval

            logger.info("🧠 Memory tracking initialized")
            logger.info(f"   Max records: {max_records}")
            logger.info(f"   Stack traces: {enable_stack_traces}")
            logger.info(f"   Tensor tracking: {enable_tensor_tracking}")
            logger.info(f"   Auto-snapshot interval: {auto_snapshot_interval}")

            # Take initial snapshot
            self.tracker.take_snapshot("initialization")
            self.is_initialized = True

        except Exception as e:
            logger.warning(f"Failed to setup memory tracking: {e}")

    def is_available(self) -> bool:
        """Check if memory tracking is available and initialized."""
        return MEMORY_TRACKING_AVAILABLE and self.is_initialized

    def show_memory_diagnostics(self) -> None:
        """Display comprehensive memory diagnostics."""
        logger.info("🧠 Memory Diagnostics")
        logger.info("=" * 60)

        if not MEMORY_TRACKING_AVAILABLE:
            logger.warning("Memory tracking not available")
            self._show_basic_diagnostics()
            return

        if not self.is_initialized:
            logger.warning("Memory tracking not initialized")
            self._show_basic_diagnostics()
            return

        try:
            # Get memory tracker
            tracker = self.tracker or get_memory_tracker(auto_install=False)

            # Take current snapshot
            current_snapshot = tracker.take_snapshot("diagnostics")

            # Get allocation summary
            summary = tracker.get_allocation_summary()

            # Display comprehensive diagnostics
            self._display_current_memory_state(current_snapshot)
            self._display_allocation_statistics(summary)
            self._display_operation_breakdown(summary)
            self._display_memory_pressure(tracker, current_snapshot)
            self._display_fragmentation_analysis(current_snapshot)
            self._export_diagnostics(tracker)

        except Exception as e:
            logger.error(f"Failed to get memory diagnostics: {e}")
            self._show_basic_diagnostics()

    def _show_basic_diagnostics(self) -> None:
        """Show basic memory diagnostics as fallback."""
        try:
            from common.performance_logger import snapshot_gpu_memory

            snapshot_gpu_memory("diagnostics")

            if torch.cuda.is_available():
                device = torch.device("cuda")
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)

                logger.info(f"📊 Basic Memory State:")
                logger.info(f"   Current Allocated: {allocated:.2f} GB")
                logger.info(f"   Current Reserved:  {reserved:.2f} GB")
                logger.info(f"   Peak Allocated:    {peak:.2f} GB")
                logger.info(f"   Available Free:    {free_gb:.2f} GB")
                logger.info(f"   Total GPU Memory:  {total_gb:.2f} GB")
                logger.info(f"   Utilization:       {(allocated/total_gb)*100:.1f}%")

        except Exception as e:
            logger.error(f"Failed to get basic memory diagnostics: {e}")

    def _display_current_memory_state(self, snapshot) -> None:
        """Display current memory state information."""
        logger.info("📊 Current Memory State:")
        logger.info(f"   Allocated:     {snapshot.torch_allocated / (1024**3):.2f} GB")
        logger.info(f"   Reserved:      {snapshot.torch_reserved / (1024**3):.2f} GB")
        logger.info(f"   Free:          {snapshot.torch_free / (1024**3):.2f} GB")
        logger.info(f"   Total:         {snapshot.torch_total / (1024**3):.2f} GB")
        logger.info(f"   Peak:          {snapshot.peak_allocated / (1024**3):.2f} GB")
        logger.info(f"   Active Tensors: {snapshot.active_tensors}")

        if snapshot.torch_total > 0:
            utilization = (snapshot.torch_allocated / snapshot.torch_total) * 100
            logger.info(f"   Utilization:   {utilization:.1f}%")

    def _display_allocation_statistics(self, summary: Dict[str, Any]) -> None:
        """Display allocation statistics."""
        logger.info("\n📈 Allocation Statistics:")
        logger.info(f"   Total Allocations:    {summary['total_allocations']:,}")
        logger.info(f"   Failed Allocations:   {summary['failed_allocations']:,}")
        logger.info(
            f"   Total Bytes Requested: {summary['total_bytes_requested'] / (1024**3):.2f} GB"
        )
        logger.info(
            f"   Peak Request Size:    {summary['peak_bytes_requested'] / (1024**2):.2f} MB"
        )
        logger.info(
            f"   Recent Avg Alloc:     {summary['recent_avg_allocation_mb']:.2f} MB"
        )

        if summary["failed_allocations"] > 0:
            logger.warning(
                f"⚠️  {summary['failed_allocations']} allocation failures detected!"
            )

    def _display_operation_breakdown(self, summary: Dict[str, Any]) -> None:
        """Display operation breakdown if available."""
        if summary["operation_breakdown"]:
            logger.info("\n🔍 Allocation Breakdown by Operation:")
            for op, stats in summary["operation_breakdown"].items():
                logger.info(
                    f"   {op}: {stats['count']} calls, {stats['total_bytes'] / (1024**2):.1f} MB, {stats['failures']} failures"
                )

    def _display_memory_pressure(self, tracker, current_snapshot) -> None:
        """Display memory pressure indicators."""
        if len(tracker.snapshots) >= 2:
            prev_snapshot = tracker.snapshots[-2]
            growth = current_snapshot.torch_allocated - prev_snapshot.torch_allocated
            if growth > 0:
                logger.info(
                    f"\n📈 Memory Growth: +{growth / (1024**2):.2f} MB since last snapshot"
                )
            elif growth < 0:
                logger.info(
                    f"\n📉 Memory Freed: {abs(growth) / (1024**2):.2f} MB since last snapshot"
                )

    def _display_fragmentation_analysis(self, snapshot) -> None:
        """Display memory fragmentation analysis."""
        if snapshot.torch_reserved > snapshot.torch_allocated:
            fragmentation = snapshot.torch_reserved - snapshot.torch_allocated
            frag_ratio = (
                fragmentation / snapshot.torch_reserved
                if snapshot.torch_reserved > 0
                else 0
            )
            logger.info(f"\n🧩 Memory Fragmentation:")
            logger.info(
                f"   Fragmented: {fragmentation / (1024**3):.2f} GB ({frag_ratio:.1%})"
            )

            if frag_ratio > 0.3:
                logger.warning(
                    "⚠️  High fragmentation detected! Consider torch.cuda.empty_cache()"
                )

    def _export_diagnostics(self, tracker) -> None:
        """Export diagnostics to file."""
        export_dir = getattr(self.args, "memory_tracking_export_directory", "logs")
        timestamp = int(time.time())
        export_path = f"{export_dir}/memory_diagnostics_{timestamp}.json"

        try:
            os.makedirs(export_dir, exist_ok=True)
            tracker.export_diagnostics(export_path)
            logger.info(f"\n💾 Diagnostics exported to: {export_path}")
        except Exception as e:
            logger.warning(f"Failed to export diagnostics: {e}")

    def take_snapshot(self, tag: str) -> Optional[Any]:
        """Take a memory snapshot if tracking is available.

        Args:
            tag: Tag for the snapshot

        Returns:
            Memory snapshot or None if not available
        """
        if not self.is_available():
            return None

        try:
            tracker = self.tracker or get_memory_tracker(auto_install=False)
            return tracker.take_snapshot(tag)
        except Exception as e:
            logger.debug(f"Failed to take memory snapshot: {e}")
            return None

    def record_allocation(self, size_bytes: int, operation: str) -> None:
        """Record a manual allocation if tracking is available.

        Args:
            size_bytes: Size of allocation in bytes
            operation: Operation name
        """
        if not self.is_available():
            return

        try:
            tracker = self.tracker or get_memory_tracker(auto_install=False)
            tracker.record_manual_allocation(size_bytes, operation)
        except Exception as e:
            logger.debug(f"Failed to record allocation: {e}")

    def get_memory_summary(self) -> Optional[Dict[str, Any]]:
        """Get memory allocation summary if available.

        Returns:
            Memory summary dictionary or None if not available
        """
        if not self.is_available():
            return None

        try:
            tracker = self.tracker or get_memory_tracker(auto_install=False)
            return tracker.get_allocation_summary()
        except Exception as e:
            logger.debug(f"Failed to get memory summary: {e}")
            return None

    def handle_oom_error(
        self, error: torch.cuda.OutOfMemoryError, context: str = "unknown"
    ) -> None:
        """Handle OOM error with diagnostics.

        Args:
            error: The CUDA OOM error
            context: Context where error occurred
        """
        if MEMORY_TRACKING_AVAILABLE:
            try:
                handle_cuda_oom_error(error, context)
                return
            except Exception as e:
                logger.debug(f"Enhanced OOM handling failed: {e}")

        # Fallback handling
        logger.error(f"🚨 CUDA OUT OF MEMORY ERROR in {context}")
        logger.error(f"Error message: {error}")
        self._show_basic_diagnostics()


# Global memory tracking manager instance
_global_manager: Optional[MemoryTrackingManager] = None


def initialize_memory_tracking(args: argparse.Namespace) -> MemoryTrackingManager:
    """Initialize the global memory tracking manager.

    Args:
        args: Configuration arguments

    Returns:
        Initialized memory tracking manager
    """
    global _global_manager

    _global_manager = MemoryTrackingManager(args)
    _global_manager.setup_memory_tracking()

    return _global_manager


def get_memory_tracking_manager() -> Optional[MemoryTrackingManager]:
    """Get the global memory tracking manager instance.

    Returns:
        Memory tracking manager or None if not initialized
    """
    return _global_manager


def is_memory_tracking_available() -> bool:
    """Check if memory tracking is available and initialized.

    Returns:
        True if memory tracking is available
    """
    return _global_manager is not None and _global_manager.is_available()


def show_memory_diagnostics() -> None:
    """Show memory diagnostics using the global manager."""
    if _global_manager:
        _global_manager.show_memory_diagnostics()
    else:
        logger.warning("Memory tracking manager not initialized")
        # Show basic diagnostics as fallback
        _show_basic_fallback_diagnostics()


def _show_basic_fallback_diagnostics() -> None:
    """Show basic memory diagnostics when manager is not available."""
    try:
        from common.performance_logger import snapshot_gpu_memory

        snapshot_gpu_memory("diagnostics")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)

            logger.info("🧠 Basic Memory Diagnostics")
            logger.info("=" * 60)
            logger.info(f"📊 Basic Memory State:")
            logger.info(f"   Current Allocated: {allocated:.2f} GB")
            logger.info(f"   Current Reserved:  {reserved:.2f} GB")
            logger.info(f"   Peak Allocated:    {peak:.2f} GB")
            logger.info(f"   Available Free:    {free_gb:.2f} GB")
            logger.info(f"   Total GPU Memory:  {total_gb:.2f} GB")
            logger.info(f"   Utilization:       {(allocated/total_gb)*100:.1f}%")

    except Exception as e:
        logger.error(f"Failed to get basic memory diagnostics: {e}")


def take_memory_snapshot(tag: str) -> Optional[Any]:
    """Take a memory snapshot using the global manager.

    Args:
        tag: Tag for the snapshot

    Returns:
        Memory snapshot or None
    """
    if _global_manager:
        return _global_manager.take_snapshot(tag)
    return None


def record_memory_allocation(size_bytes: int, operation: str) -> None:
    """Record a memory allocation using the global manager.

    Args:
        size_bytes: Size of allocation in bytes
        operation: Operation name
    """
    if _global_manager:
        _global_manager.record_allocation(size_bytes, operation)


def handle_oom_with_diagnostics(
    error: torch.cuda.OutOfMemoryError, context: str = "unknown"
) -> None:
    """Handle OOM error with comprehensive diagnostics.

    Args:
        error: The CUDA OOM error
        context: Context where error occurred
    """
    if _global_manager:
        _global_manager.handle_oom_error(error, context)
    else:
        # Basic fallback
        logger.error(f"🚨 CUDA OUT OF MEMORY ERROR in {context}")
        logger.error(f"Error message: {error}")
