"""Training-safe memory optimization manager for Takenoko."""

import gc
import logging
from common.logger import get_logger
import mmap
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Union, Optional, Callable
import safetensors
import safetensors.torch

logger = get_logger(__name__, level=logging.INFO)


class SafeMemoryManager:
    """Training-safe memory optimization manager with direct config parameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with direct config parameters."""
        self.enabled = config.get("safe_memory_optimization_enabled", False)
        self.optimized_loading_enabled = config.get("memory_opt_loading_enabled", False)
        self.memory_monitoring_enabled = config.get(
            "memory_opt_monitoring_enabled", False
        )
        self.gc_threshold_ratio = config.get("memory_opt_gc_threshold_ratio", 0.85)
        self.gc_interval_steps = config.get("memory_opt_gc_interval_steps", 100)
        self.monitor_interval_steps = config.get(
            "memory_opt_monitor_interval_steps", 50
        )

        # Force disable all sub-features if master switch is off
        if not self.enabled:
            self.optimized_loading_enabled = False
            self.memory_monitoring_enabled = False
            logger.info("Safe memory optimization disabled - all sub-features disabled")
        else:
            logger.info(
                f"🧠 Safe memory optimization enabled — loading: {self.optimized_loading_enabled}, monitoring: {self.memory_monitoring_enabled}"
            )

        self._step_counter = 0
        self._last_gc_step = 0
        self._last_monitor_step = 0

        # Initialize the universal safetensors loader
        self._initialize_universal_loader()

    def load_safetensors_optimized(
        self,
        path: Union[str, Path],
        device: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        SINGLE METHOD INTERFACE: Drop-in replacement for safetensors.torch.load_file()

        This method handles all memory optimizations internally:
        1. Optimized loading with memory mapping (if enabled)
        2. Memory monitoring and cleanup (if enabled)
        3. Falls back to standard loading if optimizations disabled
        """
        if step is not None:
            self._step_counter = step
        else:
            self._step_counter += 1

        # Memory monitoring before loading
        if self.memory_monitoring_enabled and self._should_monitor():
            self._monitor_and_cleanup()

        # Load tensors with or without optimization
        if self.optimized_loading_enabled:
            try:
                tensors = self._load_with_mmap(path, device)
                logger.debug(f"🧠📦 Loaded {path} with memory mapping optimization")
            except Exception as e:
                logger.warning(
                    f"🧠⚠️ Optimized loading failed, falling back to standard: {e}"
                )
                # Match original safetensors.torch.load_file behavior exactly
                if device is not None:
                    tensors = safetensors.torch.load_file(path, device=device)
                else:
                    tensors = safetensors.torch.load_file(path)
        else:
            # Match original safetensors.torch.load_file behavior exactly
            if device is not None:
                tensors = safetensors.torch.load_file(path, device=device)
            else:
                tensors = safetensors.torch.load_file(path)

        # Memory monitoring after loading
        if self.memory_monitoring_enabled and self._should_trigger_gc():
            self._trigger_gc()

        return tensors

    def _load_with_mmap(
        self, path: Union[str, Path], device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Load tensors using memory mapping for efficient loading."""
        path = Path(path)

        try:
            # Use safetensors built-in memory mapping support
            # This avoids loading the entire file into RAM
            tensors = {}

            # Handle device parameter like original load_file
            device_param = device if device is not None else "cpu"
            with safetensors.safe_open(path, framework="pt", device=device_param) as f:
                for key in f.keys():
                    # This loads tensors efficiently using safetensors' internal optimization
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor

            return tensors

        except Exception as e:
            logger.warning(
                f"🧠⚠️ Memory-mapped loading failed for {path}: {e}, falling back to standard loading"
            )
            # Match original behavior exactly
            if device is not None:
                return safetensors.torch.load_file(path, device=device)
            else:
                return safetensors.torch.load_file(path)

    def _should_monitor(self) -> bool:
        """Check if we should run memory monitoring this step."""
        return (
            self._step_counter - self._last_monitor_step
        ) >= self.monitor_interval_steps

    def _should_trigger_gc(self) -> bool:
        """Check if we should trigger garbage collection this step."""
        return (self._step_counter - self._last_gc_step) >= self.gc_interval_steps

    def _monitor_and_cleanup(self):
        """Monitor memory usage and log statistics."""
        try:
            # CPU Memory
            cpu_memory = psutil.virtual_memory()
            cpu_percent = cpu_memory.percent

            # GPU Memory
            gpu_stats = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = (
                        gpu_memory.get("allocated_bytes.all.current", 0) / 1024**3
                    )
                    reserved = gpu_memory.get("reserved_bytes.all.current", 0) / 1024**3
                    gpu_stats[f"gpu_{i}"] = {
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "utilization": allocated / reserved if reserved > 0 else 0,
                    }

            # Check if cleanup needed
            cleanup_needed = cpu_percent > (self.gc_threshold_ratio * 100)
            for gpu_id, stats in gpu_stats.items():
                if stats["utilization"] > self.gc_threshold_ratio:
                    cleanup_needed = True
                    break

            if cleanup_needed:
                logger.info(
                    f"🧠🧹 Memory cleanup triggered — CPU: {cpu_percent:.1f}%, GPU utilization > {self.gc_threshold_ratio}"
                )
                self._trigger_gc()

            self._last_monitor_step = self._step_counter

        except Exception as e:
            logger.warning(f"🧠⚠️ Memory monitoring failed: {e}")

    def _trigger_gc(self):
        """Trigger garbage collection and CUDA cache cleanup."""
        try:
            # Python garbage collection
            collected = gc.collect()

            # CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._last_gc_step = self._step_counter
            logger.debug(
                f"🧠♻️ Garbage collection completed — collected {collected} objects"
            )

        except Exception as e:
            logger.warning(f"🧠⚠️ Garbage collection failed: {e}")

    def _initialize_universal_loader(self) -> None:
        """Initialize the universal safetensors loader with this memory manager."""
        from . import safetensors_loader

        if self.enabled and self.optimized_loading_enabled:
            # Initialize with this memory manager for optimization
            safetensors_loader.initialize_loader(self)
            logger.debug(
                "🧠⚙️ Initialized universal safetensors loader with optimization"
            )
        else:
            # Initialize without memory manager (standard loading)
            safetensors_loader.initialize_loader(None)
            logger.debug(
                "🧠⚙️ Initialized universal safetensors loader with standard loading"
            )
