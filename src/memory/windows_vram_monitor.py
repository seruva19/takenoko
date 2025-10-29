"""Windows-specific shared GPU memory detection and monitoring.

This module provides lightweight detection of when Windows uses system RAM
as shared GPU memory (overflow from dedicated VRAM). It monitors VRAM usage
and provides warnings when performance degradation is likely due to shared
memory usage.

Key Features:
- Asynchronous monitoring with configurable check intervals
- Detection of shared memory usage via NVML + PyTorch APIs
- Performance recommendations when shared memory is detected
- Minimal overhead on training performance
"""

from __future__ import annotations

import argparse
import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class VRAMSnapshot:
    """Snapshot of VRAM and shared memory state."""

    timestamp: float
    # PyTorch CUDA API metrics
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    torch_free_gb: float = 0.0
    torch_total_gb: float = 0.0
    # NVML hardware metrics
    nvml_used_gb: float = 0.0
    nvml_free_gb: float = 0.0
    nvml_total_gb: float = 0.0
    # Shared memory detection
    shared_memory_gb: float = 0.0
    using_shared_memory: bool = False
    confidence_level: str = "none"  # "none", "low", "medium", "high"


@dataclass
class SharedMemoryDetectionResult:
    """Result of shared memory detection analysis."""

    using_shared_memory: bool
    shared_memory_gb: float
    confidence: str  # "none", "low", "medium", "high"
    signals: List[str]
    recommendations: List[str]


class WindowsVRAMMonitor:
    """Lightweight Windows shared GPU memory monitor.

    Monitors VRAM usage and detects when Windows starts using system RAM
    as shared GPU memory. Provides warnings and recommendations when detected.

    Design principles:
    - Minimal overhead: Async monitoring with configurable intervals
    - Peak tracking: Uses torch.cuda.max_memory_allocated/reserved() to catch
      transient spikes between checks (validation, sampling, checkpointing)
    - Windows-only: Automatically disables on non-Windows platforms
    - Non-blocking: Checks only at specified step intervals
    - Safe: Graceful degradation if NVML unavailable

    Key feature: Even with 50-step intervals, catches ALL VRAM spikes via peak tracking!
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize Windows VRAM monitor.

        Args:
            args: Namespace containing configuration parameters
        """
        self.enabled = getattr(args, "enable_windows_vram_monitor", False)
        self.check_interval = getattr(args, "windows_vram_check_interval", 50)
        self.warning_threshold_gb = getattr(
            args, "windows_vram_warning_threshold_gb", 0.5
        )
        self.detailed_logging = getattr(args, "windows_vram_detailed_logging", False)
        self.auto_suggest = getattr(
            args, "windows_vram_auto_suggest_optimizations", True
        )

        # Internal state
        self._is_windows = platform.system() == "Windows"
        self._nvml_available = False
        self._monitoring_active = False
        self._step_counter = 0
        self._last_check_step = -1
        self._lock = threading.RLock()
        self._last_snapshot: Optional[VRAMSnapshot] = None
        self._shared_memory_detected = False
        self._warning_logged = False

        # Initialize if enabled
        if self.enabled and self._is_windows:
            self._initialize_nvml()
            if not self._nvml_available:
                logger.warning(
                    "⚠️  Windows VRAM monitor enabled but NVML (pynvml) not available. "
                    "Install nvidia-ml-py for full functionality: pip install nvidia-ml-py"
                )
                logger.info(
                    "   Falling back to PyTorch-only detection (reduced accuracy)"
                )
        elif self.enabled and not self._is_windows:
            logger.info(
                f"Windows VRAM monitor disabled (platform: {platform.system()})"
            )
            self.enabled = False

    def _initialize_nvml(self) -> None:
        """Initialize NVML library if available."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_available = True
            logger.info("✅ Windows VRAM monitor initialized (NVML available)")
        except ImportError:
            logger.debug("NVML (pynvml) not available, using PyTorch-only detection")
        except Exception as e:
            logger.debug(f"Failed to initialize NVML: {e}")

    def check_step(self, global_step: int) -> None:
        """Check for shared memory usage at training step.

        This is the main entry point called from training loop.
        Only performs check every N steps based on check_interval.

        Args:
            global_step: Current training step number
        """
        if not self.enabled or not self._is_windows:
            return

        self._step_counter = global_step

        # Check if it's time for periodic monitoring
        if (global_step - self._last_check_step) >= self.check_interval:
            self._perform_check()
            self._last_check_step = global_step

    def _perform_check(self) -> None:
        """Perform actual shared memory detection check.

        Uses peak memory tracking to detect spikes that occurred since last check,
        ensuring we don't miss transient VRAM overflows that happen between checks
        (e.g., during validation, sampling, or checkpoint saving).
        """
        try:
            # Take VRAM snapshot (uses peak values to catch spikes)
            snapshot = self._take_vram_snapshot()

            # Analyze for shared memory usage
            detection_result = self._analyze_shared_memory(snapshot)

            # Store results
            with self._lock:
                self._last_snapshot = snapshot
                self._shared_memory_detected = detection_result.using_shared_memory

            # Log warnings if threshold exceeded
            if detection_result.using_shared_memory:
                self._handle_shared_memory_detection(detection_result, snapshot)

            # Note: We intentionally DON'T reset peak memory stats here
            # PyTorch resets them automatically at key points, and keeping
            # the running peak helps us detect if we've EVER hit shared memory
            # during training, not just at specific checkpoints

        except Exception as e:
            logger.debug(f"Windows VRAM monitor check failed: {e}")

    def _take_vram_snapshot(self) -> VRAMSnapshot:
        """Take snapshot of current VRAM state.

        Uses peak memory tracking to detect spikes between checks.
        """
        snapshot = VRAMSnapshot(timestamp=time.perf_counter())

        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")

                # PyTorch CUDA API metrics - use PEAK values to catch spikes
                # This captures the maximum memory used since last reset/check
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)

                # Use peak values instead of current to detect spikes
                snapshot.torch_allocated_gb = peak_allocated / (1024**3)
                snapshot.torch_reserved_gb = peak_reserved / (1024**3)

                free_bytes, total_bytes = torch.cuda.mem_get_info()
                snapshot.torch_free_gb = free_bytes / (1024**3)
                snapshot.torch_total_gb = total_bytes / (1024**3)

                # NVML hardware metrics (if available)
                if self._nvml_available:
                    try:
                        import pynvml

                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        snapshot.nvml_used_gb = meminfo.used / (1024**3)
                        snapshot.nvml_free_gb = meminfo.free / (1024**3)
                        snapshot.nvml_total_gb = meminfo.total / (1024**3)

                    except Exception as e:
                        logger.debug(f"NVML query failed: {e}")

        except Exception as e:
            logger.debug(f"Failed to take VRAM snapshot: {e}")

        return snapshot

    def _analyze_shared_memory(
        self, snapshot: VRAMSnapshot
    ) -> SharedMemoryDetectionResult:
        """Analyze snapshot for shared memory usage indicators.

        Args:
            snapshot: VRAM snapshot to analyze

        Returns:
            Detection result with confidence level and recommendations
        """
        signals = []
        recommendations = []
        confidence = "none"
        using_shared = False
        shared_gb = 0.0

        # Signal 1: NVML available - check if peak reserved > total VRAM
        if self._nvml_available and snapshot.nvml_total_gb > 0:
            if snapshot.torch_reserved_gb > snapshot.nvml_total_gb:
                # Peak reserved memory exceeds physical VRAM - definitely using shared
                shared_gb = snapshot.torch_reserved_gb - snapshot.nvml_total_gb
                using_shared = True
                confidence = "high"
                signals.append(
                    f"Peak Reserved ({snapshot.torch_reserved_gb:.2f}GB) > Physical VRAM ({snapshot.nvml_total_gb:.2f}GB)"
                )

            # Check VRAM utilization
            vram_usage_pct = (snapshot.nvml_used_gb / snapshot.nvml_total_gb) * 100
            if vram_usage_pct > 98:
                signals.append(f"VRAM at {vram_usage_pct:.1f}% capacity")
                if confidence == "none":
                    confidence = "medium"

        # Signal 2: PyTorch-only fallback (less accurate)
        elif snapshot.torch_total_gb > 0:
            # If allocated is very close to total, likely at capacity
            usage_pct = (snapshot.torch_allocated_gb / snapshot.torch_total_gb) * 100
            if usage_pct > 95:
                signals.append(f"Allocated memory at {usage_pct:.1f}% of total")
                confidence = "low"
                # Can't definitively detect shared memory without NVML
                signals.append(
                    "Install nvidia-ml-py for accurate shared memory detection"
                )

        # Generate recommendations if using shared memory
        if using_shared or confidence != "none":
            recommendations = self._generate_recommendations(snapshot, shared_gb)

        return SharedMemoryDetectionResult(
            using_shared_memory=using_shared,
            shared_memory_gb=shared_gb,
            confidence=confidence,
            signals=signals,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, snapshot: VRAMSnapshot, shared_gb: float
    ) -> List[str]:
        """Generate actionable recommendations for reducing memory usage.

        Args:
            snapshot: Current VRAM snapshot
            shared_gb: Amount of shared memory being used

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if shared_gb > 1.0:
            recommendations.append(
                "Reduce batch size significantly (>1GB shared memory)"
            )
        elif shared_gb > 0.5:
            recommendations.append("Reduce batch size or enable gradient checkpointing")

        recommendations.extend(
            [
                "Enable gradient checkpointing if not already active",
                "Use lower precision (fp16/bf16) if not already enabled",
                "Consider using gradient accumulation with smaller batch size",
                "Enable safe memory optimizations (safe_memory_optimization_enabled=true)",
                "Reduce sequence length or video frame count if applicable",
            ]
        )

        if not self._nvml_available:
            recommendations.append(
                "Install nvidia-ml-py for accurate monitoring: pip install nvidia-ml-py"
            )

        return recommendations

    def _handle_shared_memory_detection(
        self, result: SharedMemoryDetectionResult, snapshot: VRAMSnapshot
    ) -> None:
        """Handle detected shared memory usage with appropriate logging.

        Args:
            result: Detection result
            snapshot: Current VRAM snapshot
        """
        # Only log warning if exceeds threshold and not already warned
        if result.shared_memory_gb >= self.warning_threshold_gb:
            if not self._warning_logged:
                logger.warning("=" * 80)
                logger.warning("⚠️  WINDOWS SHARED GPU MEMORY DETECTED")
                logger.warning("=" * 80)
                logger.warning(
                    f"📊 Peak Shared Memory Usage: {result.shared_memory_gb:.2f} GB "
                    f"(Confidence: {result.confidence})"
                )
                logger.warning(
                    f"   (Detected via peak memory tracking - spike occurred between steps "
                    f"{self._last_check_step - self.check_interval} and {self._last_check_step})"
                )

                if self.detailed_logging:
                    logger.warning(
                        f"   Dedicated VRAM: {snapshot.nvml_total_gb:.2f} GB"
                    )
                    logger.warning(
                        f"   Peak Reserved: {snapshot.torch_reserved_gb:.2f} GB"
                    )
                    logger.warning(
                        f"   Peak Allocated: {snapshot.torch_allocated_gb:.2f} GB"
                    )

                logger.warning("🔍 Detection Signals:")
                for signal in result.signals:
                    logger.warning(f"   • {signal}")

                logger.warning(
                    "⚡ Performance Impact: System RAM is slower than VRAM - expect reduced training speed"
                )

                if self.auto_suggest and result.recommendations:
                    logger.warning("💡 Recommendations:")
                    for i, rec in enumerate(result.recommendations[:5], 1):
                        logger.warning(f"   {i}. {rec}")

                logger.warning("=" * 80)

                self._warning_logged = True

        # Always log detailed info if detailed logging enabled
        elif self.detailed_logging and result.confidence != "none":
            logger.info(
                f"Windows VRAM Monitor: {result.confidence} confidence of shared memory usage"
            )
            if result.signals:
                logger.info(f"  Signals: {', '.join(result.signals)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Dictionary with current status information
        """
        with self._lock:
            status = {
                "enabled": self.enabled,
                "is_windows": self._is_windows,
                "nvml_available": self._nvml_available,
                "shared_memory_detected": self._shared_memory_detected,
                "last_check_step": self._last_check_step,
                "check_interval": self.check_interval,
            }

            if self._last_snapshot:
                status["last_snapshot"] = {
                    "torch_allocated_gb": self._last_snapshot.torch_allocated_gb,
                    "torch_reserved_gb": self._last_snapshot.torch_reserved_gb,
                    "nvml_used_gb": self._last_snapshot.nvml_used_gb,
                    "nvml_total_gb": self._last_snapshot.nvml_total_gb,
                    "shared_memory_gb": self._last_snapshot.shared_memory_gb,
                }

            return status

    def shutdown(self) -> None:
        """Cleanup and shutdown monitor."""
        if self._nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

        self._monitoring_active = False
        logger.debug("Windows VRAM monitor shut down")


# Singleton instance for global access
_global_monitor: Optional[WindowsVRAMMonitor] = None


def initialize_windows_vram_monitor(args: argparse.Namespace) -> WindowsVRAMMonitor:
    """Initialize global Windows VRAM monitor instance.

    Args:
        args: Configuration namespace

    Returns:
        Initialized monitor instance
    """
    global _global_monitor
    _global_monitor = WindowsVRAMMonitor(args)
    return _global_monitor


def get_windows_vram_monitor() -> Optional[WindowsVRAMMonitor]:
    """Get global Windows VRAM monitor instance.

    Returns:
        Monitor instance if initialized, None otherwise
    """
    return _global_monitor


def check_shared_memory_at_step(global_step: int) -> None:
    """Convenience function to check shared memory at training step.

    Args:
        global_step: Current training step
    """
    monitor = get_windows_vram_monitor()
    if monitor:
        monitor.check_step(global_step)
