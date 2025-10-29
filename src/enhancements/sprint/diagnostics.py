"""
Sprint Diagnostics and TensorBoard Logging

Provides detailed metrics and logging for Sprint training monitoring.
"""

import time
import threading
from collections import deque, defaultdict
from typing import Optional, Dict, Any, Callable, List, Tuple
import torch

from .exceptions import SprintMemoryError


class SprintDiagnostics:
    """
    Tracks and logs Sprint training metrics for monitoring and debugging.

    Metrics tracked:
    - Token counts (kept/dropped per stage)
    - Drop ratio per step
    - Stage information (pretrain/finetune)
    - Sprint usage (active/disabled)
    - Performance timing
    """

    def __init__(self, enabled: bool = True, max_history: int = 10000):
        """
        Initialize Sprint diagnostics.

        Args:
            enabled: Whether diagnostics are enabled
            max_history: Maximum number of metrics to store in memory
        """
        self.enabled = enabled
        self.max_history = max_history
        self._tensorboard_logger: Optional[Callable] = None
        self._log_every_n_steps: int = 10

        # Thread safety
        self._lock = threading.RLock()

        # Cumulative metrics
        self._step_count = 0
        self._total_tokens_kept = 0
        self._total_tokens_dropped = 0
        self._sprint_active_count = 0
        self._sprint_disabled_count = 0

        # Bounded metrics history (prevents memory leaks)
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._recent_drop_ratios: deque = deque(maxlen=max_history)
        self._recent_token_counts: deque = deque(maxlen=max_history)

        # Timing
        self._last_forward_time: Optional[float] = None
        self._timing_history: deque = deque(maxlen=max_history)

    def set_tensorboard_logger(
        self,
        logger_fn: Callable[[Dict[str, Any], int], None],
        log_every_n_steps: int = 10
    ):
        """
        Set TensorBoard logging function.

        Args:
            logger_fn: Function that takes (metrics_dict, step) and logs to TensorBoard
            log_every_n_steps: Log metrics every N steps

        Example:
            def log_fn(metrics, step):
                accelerator.log(metrics, step=step)

            diagnostics.set_tensorboard_logger(log_fn, log_every_n_steps=10)
        """
        self._tensorboard_logger = logger_fn
        self._log_every_n_steps = log_every_n_steps

    def record_forward_pass(
        self,
        sprint_active: bool,
        drop_ratio: float,
        original_tokens: int,
        kept_tokens: int,
        stage_name: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        """
        Record metrics from a forward pass (thread-safe).

        Args:
            sprint_active: Whether Sprint was used for this forward pass
            drop_ratio: Token drop ratio used (0.0 = no dropping)
            original_tokens: Total number of tokens before dropping
            kept_tokens: Number of tokens kept (for sparse stage)
            stage_name: Training stage name (e.g., "pretrain", "finetune")
            global_step: Current global training step
        """
        if not self.enabled:
            return

        with self._lock:
            self._step_count += 1

            if sprint_active:
                self._sprint_active_count += 1
                dropped_tokens = original_tokens - kept_tokens
                self._total_tokens_kept += kept_tokens
                self._total_tokens_dropped += dropped_tokens
            else:
                self._sprint_disabled_count += 1

            # Store metrics in bounded history
            if global_step is not None:
                self._metrics_history['drop_ratio'].append((global_step, drop_ratio))
                self._metrics_history['original_tokens'].append((global_step, original_tokens))
                self._metrics_history['kept_tokens'].append((global_step, kept_tokens))
                self._metrics_history['sprint_active'].append((global_step, int(sprint_active)))

                if stage_name:
                    self._metrics_history['stage'].append((global_step, stage_name))

            self._recent_drop_ratios.append(drop_ratio)
            self._recent_token_counts.append((original_tokens, kept_tokens))

        # Log to TensorBoard if configured (outside lock to avoid holding it during I/O)
        if (self._tensorboard_logger and
            global_step is not None and
            global_step % self._log_every_n_steps == 0):

            metrics = {
                "sprint/active": 1.0 if sprint_active else 0.0,
                "sprint/drop_ratio": drop_ratio,
            }

            if sprint_active:
                metrics.update({
                    "sprint/tokens_kept": kept_tokens,
                    "sprint/tokens_dropped": original_tokens - kept_tokens,
                    "sprint/keep_ratio": kept_tokens / original_tokens if original_tokens > 0 else 0.0,
                })

            if stage_name:
                # Encode stage as number for plotting
                stage_mapping = {"warmup": 0, "pretrain": 1, "transition": 2, "finetune": 3}
                metrics["sprint/stage"] = stage_mapping.get(stage_name, -1)

            self._tensorboard_logger(metrics, global_step)

    def record_timing(self, forward_time_ms: float, global_step: Optional[int] = None):
        """
        Record forward pass timing.

        Args:
            forward_time_ms: Forward pass time in milliseconds
            global_step: Current global training step
        """
        if not self.enabled:
            return

        self._last_forward_time = forward_time_ms

        if (self._tensorboard_logger and
            global_step is not None and
            global_step % self._log_every_n_steps == 0):

            self._tensorboard_logger({
                "sprint/forward_time_ms": forward_time_ms
            }, global_step)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary of summary metrics
        """
        total_steps = self._step_count

        return {
            "total_steps": total_steps,
            "sprint_active_steps": self._sprint_active_count,
            "sprint_disabled_steps": self._sprint_disabled_count,
            "sprint_usage_ratio": (
                self._sprint_active_count / total_steps
                if total_steps > 0 else 0.0
            ),
            "total_tokens_kept": self._total_tokens_kept,
            "total_tokens_dropped": self._total_tokens_dropped,
            "avg_tokens_per_step": (
                (self._total_tokens_kept + self._total_tokens_dropped) / self._sprint_active_count
                if self._sprint_active_count > 0 else 0
            ),
        }

    def log_summary(self, logger_fn: Optional[Callable] = None):
        """
        Log summary statistics.

        Args:
            logger_fn: Optional custom logger function. Uses print if not provided.
        """
        if not self.enabled:
            return

        summary = self.get_summary()

        log = logger_fn if logger_fn else print
        log("=" * 60)
        log("Sprint Training Summary")
        log("=" * 60)
        log(f"Total steps: {summary['total_steps']}")
        log(f"Sprint active: {summary['sprint_active_steps']} steps "
            f"({summary['sprint_usage_ratio']:.1%})")
        log(f"Sprint disabled: {summary['sprint_disabled_steps']} steps")

        if summary['sprint_active_steps'] > 0:
            log(f"Tokens kept: {summary['total_tokens_kept']:,}")
            log(f"Tokens dropped: {summary['total_tokens_dropped']:,}")
            log(f"Avg tokens/step: {summary['avg_tokens_per_step']:.0f}")

        log("=" * 60)

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for monitoring.

        Returns:
            Dictionary with memory usage information
        """
        with self._lock:
            import sys

            total_metrics = sum(len(history) for history in self._metrics_history.values())

            # Estimate memory usage (rough approximation)
            estimated_memory_bytes = 0
            for history in self._metrics_history.values():
                # Each entry is roughly (step, value) tuple
                estimated_memory_bytes += sys.getsizeof(history) + len(history) * 64  # 64 bytes per entry estimate

            estimated_memory_bytes += sys.getsizeof(self._recent_drop_ratios) + len(self._recent_drop_ratios) * 8
            estimated_memory_bytes += sys.getsizeof(self._recent_token_counts) + len(self._recent_token_counts) * 16
            estimated_memory_bytes += sys.getsizeof(self._timing_history) + len(self._timing_history) * 16

            return {
                'total_metrics_stored': total_metrics,
                'max_history': self.max_history,
                'history_utilization': total_metrics / (len(self._metrics_history) * self.max_history) if self._metrics_history else 0,
                'estimated_memory_mb': estimated_memory_bytes / (1024 * 1024),
                'drop_ratio_history_size': len(self._recent_drop_ratios),
                'token_count_history_size': len(self._recent_token_counts),
                'timing_history_size': len(self._timing_history)
            }

    def cleanup_old_metrics(self, keep_last_n: Optional[int] = None) -> None:
        """
        Clean up old metrics to prevent memory leaks.

        Args:
            keep_last_n: Number of recent metrics to keep (uses max_history if None)
        """
        with self._lock:
            keep_n = keep_last_n if keep_last_n is not None else self.max_history

            for name, history in self._metrics_history.items():
                if len(history) > keep_n:
                    # Keep only the last keep_n entries
                    recent_entries = list(history)[-keep_n:]
                    self._metrics_history[name] = deque(recent_entries, maxlen=self.max_history)

            # Clean up other deques
            if len(self._recent_drop_ratios) > keep_n:
                recent_drop_ratios = list(self._recent_drop_ratios)[-keep_n:]
                self._recent_drop_ratios = deque(recent_drop_ratios, maxlen=self.max_history)

            if len(self._recent_token_counts) > keep_n:
                recent_token_counts = list(self._recent_token_counts)[-keep_n:]
                self._recent_token_counts = deque(recent_token_counts, maxlen=self.max_history)

            if len(self._timing_history) > keep_n:
                recent_timing = list(self._timing_history)[-keep_n:]
                self._timing_history = deque(recent_timing, maxlen=self.max_history)

    def get_metrics_history(self, metric_name: str) -> List[Tuple[int, Any]]:
        """
        Get history for a specific metric (thread-safe copy).

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            List of (step, value) tuples
        """
        with self._lock:
            return list(self._metrics_history.get(metric_name, []))

    def clear_all_metrics(self) -> None:
        """Clear all stored metrics (reset to initial state)."""
        with self._lock:
            self._metrics_history.clear()
            self._recent_drop_ratios.clear()
            self._recent_token_counts.clear()
            self._timing_history.clear()

            # Reset cumulative metrics
            self._step_count = 0
            self._total_tokens_kept = 0
            self._total_tokens_dropped = 0
            self._sprint_active_count = 0
            self._sprint_disabled_count = 0
            self._last_forward_time = None


# Thread-safe global diagnostics management
_global_diagnostics: Optional[SprintDiagnostics] = None
_diagnostics_lock = threading.Lock()


def get_diagnostics() -> Optional[SprintDiagnostics]:
    """Get global Sprint diagnostics instance (thread-safe)."""
    with _diagnostics_lock:
        return _global_diagnostics


def initialize_diagnostics(enabled: bool = True, max_history: int = 10000) -> SprintDiagnostics:
    """
    Initialize global Sprint diagnostics (thread-safe).

    Args:
        enabled: Whether diagnostics are enabled
        max_history: Maximum number of metrics to store in memory

    Returns:
        SprintDiagnostics instance
    """
    global _global_diagnostics
    with _diagnostics_lock:
        _global_diagnostics = SprintDiagnostics(enabled=enabled, max_history=max_history)
        return _global_diagnostics


def record_sprint_step_threadsafe(
    sprint_active: bool,
    drop_ratio: float,
    seq_lens: torch.Tensor,
    sparse_seq_lens: Optional[torch.Tensor] = None,
    stage_name: Optional[str] = None,
    global_step: Optional[int] = None,
):
    """
    Thread-safe version of record_sprint_step.

    Args:
        sprint_active: Whether Sprint was used
        drop_ratio: Token drop ratio
        seq_lens: Original sequence lengths [B]
        sparse_seq_lens: Sparse sequence lengths [B] (if Sprint active)
        stage_name: Training stage name
        global_step: Current global step
    """
    diagnostics = get_diagnostics()
    if diagnostics and diagnostics.enabled:
        # Convert tensors to CPU to avoid CUDA context issues in multi-threading
        seq_lens_cpu = seq_lens.cpu() if seq_lens.is_cuda else seq_lens
        sparse_seq_lens_cpu = sparse_seq_lens.cpu() if sparse_seq_lens is not None and sparse_seq_lens.is_cuda else None

        original_tokens = seq_lens_cpu.sum().item()
        kept_tokens = sparse_seq_lens_cpu.sum().item() if sparse_seq_lens_cpu is not None else original_tokens

        diagnostics.record_forward_pass(
            sprint_active=sprint_active,
            drop_ratio=drop_ratio,
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            stage_name=stage_name,
            global_step=global_step,
        )


def record_sprint_step(
    sprint_active: bool,
    drop_ratio: float,
    seq_lens: torch.Tensor,
    sparse_seq_lens: Optional[torch.Tensor] = None,
    stage_name: Optional[str] = None,
    global_step: Optional[int] = None,
):
    """
    Convenience function to record Sprint step metrics (backward compatibility).

    Note: For thread safety, consider using record_sprint_step_threadsafe() instead.

    Args:
        sprint_active: Whether Sprint was used
        drop_ratio: Token drop ratio
        seq_lens: Original sequence lengths [B]
        sparse_seq_lens: Sparse sequence lengths [B] (if Sprint active)
        stage_name: Training stage name
        global_step: Current global step
    """
    # Use the thread-safe version for consistency
    record_sprint_step_threadsafe(
        sprint_active=sprint_active,
        drop_ratio=drop_ratio,
        seq_lens=seq_lens,
        sparse_seq_lens=sparse_seq_lens,
        stage_name=stage_name,
        global_step=global_step,
    )
