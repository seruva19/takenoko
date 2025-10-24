"""Advanced metrics tracking for comprehensive training diagnostics.

Provides optional tracking of:
- Gradient stability with moving averages
- HIGH/LOW noise loss split (for WAN dual LoRA)
- Convergence trend analysis (R² correlation)
- Loss distribution histograms
- Oscillation bounds detection

All features are opt-in and have zero overhead when disabled.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import torch

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LossRecord:
    """Single record for comprehensive loss tracking.

    Attributes:
        step: Global training step
        loss: Loss value for this step
        gradient_norm: Gradient norm (if available)
        is_high_noise: Whether this is high-noise regime (for dual LoRA, optional)
    """
    step: int
    loss: float
    gradient_norm: Optional[float] = None
    is_high_noise: Optional[bool] = None


class AdvancedMetricsTracker:
    """Advanced metrics tracking with minimal overhead when disabled.

    This tracker provides comprehensive training diagnostics similar to
    those shown in the WAN 2.2 Reddit training reports, including:
    - Gradient stability monitoring with watch thresholds
    - HIGH/LOW noise loss split tracking (for dual LoRA training)
    - Multi-scale convergence analysis using R² correlation
    - Oscillation bounds detection

    All features are optional and can be selectively enabled.
    When disabled (default), this has zero computational overhead.
    """

    def __init__(
        self,
        enabled: bool = False,
        features: Optional[Set[str]] = None,
        max_history: int = 10000,
        dual_model_manager: Optional[Any] = None,
        gradient_watch_threshold: float = 0.5,
        gradient_stability_window: int = 10,
        convergence_window_sizes: Optional[List[int]] = None,
    ):
        """Initialize the advanced metrics tracker.

        Args:
            enabled: Master enable switch. If False, all tracking is disabled.
            features: Set of enabled features. Options:
                - "gradient_stability": Track gradient norms with moving averages
                - "convergence": Multi-scale R² convergence analysis
                - "noise_split": HIGH/LOW noise loss split (requires dual_model_manager)
                - "oscillation_bounds": Track upper/lower loss bounds
                If None, all features are enabled when enabled=True.
            max_history: Maximum number of records to retain (prevents unbounded memory)
            dual_model_manager: Reference to DualModelManager (required for noise_split)
            gradient_watch_threshold: Alert threshold for gradient norm warnings
            gradient_stability_window: Moving average window size for gradient stability
            convergence_window_sizes: Window sizes for multi-scale convergence analysis
        """
        self.enabled = enabled

        # Early exit if disabled - no initialization needed
        if not enabled:
            return

        # Default to all features if not specified
        all_features = {
            "gradient_stability",
            "convergence",
            "noise_split",
            "oscillation_bounds",
        }
        self.features = features if features is not None else all_features

        # Store references and configuration
        self.dual_model_manager = dual_model_manager
        self.gradient_threshold = gradient_watch_threshold
        self.gradient_window = gradient_stability_window
        self.convergence_windows = convergence_window_sizes or [10, 25, 50, 100]

        # Validate features and disable if requirements not met
        self._validate_features()

        # Single source of truth for all data (bounded to prevent memory leaks)
        self.records: deque = deque(maxlen=max_history)

        # Step counter
        self.step_count = 0

        logger.info(
            f"AdvancedMetricsTracker initialized with features: {sorted(self.features)}"
        )

    def _validate_features(self) -> None:
        """Validate feature requirements and disable features if needed."""
        if "noise_split" in self.features:
            if self.dual_model_manager is None:
                logger.warning(
                    "noise_split tracking requires DualModelManager but it's not available. "
                    "Disabling noise_split feature."
                )
                self.features.discard("noise_split")

    def track_step(
        self,
        step: int,
        loss: float,
        gradient_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Track metrics for the current training step.

        Args:
            step: Current global training step
            loss: Loss value for this step
            gradient_norm: Gradient norm (if available)

        Returns:
            Dict of metrics to log. Empty dict if tracking is disabled.
        """
        # Early exit before ANY computation if disabled
        if not self.enabled:
            return {}

        self.step_count += 1

        # Determine noise level from DualModelManager if available
        is_high_noise = None
        if "noise_split" in self.features and self.dual_model_manager is not None:
            try:
                is_high_noise = self.dual_model_manager.current_model_is_high_noise
            except Exception as e:
                logger.debug(f"Failed to get noise level from DualModelManager: {e}")

        # Store single record (bounded deque prevents memory leaks)
        record = LossRecord(
            step=step,
            loss=loss,
            gradient_norm=gradient_norm,
            is_high_noise=is_high_noise,
        )
        self.records.append(record)

        # Compute and return metrics (single dict allocation)
        return self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute all enabled metrics efficiently.

        Returns:
            Dict of metrics with feature-specific prefixes
        """
        metrics = {}

        # Gradient stability metrics
        if "gradient_stability" in self.features:
            grad_metrics = self._compute_gradient_metrics()
            metrics.update(grad_metrics)

        # Convergence analysis metrics
        if "convergence" in self.features:
            conv_metrics = self._compute_convergence_metrics()
            metrics.update(conv_metrics)

        # Noise split metrics (HIGH/LOW for dual LoRA)
        if "noise_split" in self.features:
            noise_metrics = self._compute_noise_split_metrics()
            metrics.update(noise_metrics)

        # Oscillation bounds metrics
        if "oscillation_bounds" in self.features:
            osc_metrics = self._compute_oscillation_metrics()
            metrics.update(osc_metrics)

        return metrics

    def _compute_gradient_metrics(self) -> Dict[str, float]:
        """Compute gradient stability metrics from recent records.

        Returns:
            Dict with gradient/* metrics
        """
        # Filter records with gradient norms
        recent_grads = [
            r.gradient_norm
            for r in list(self.records)[-self.gradient_window:]
            if r.gradient_norm is not None
        ]

        if len(recent_grads) < 2:
            return {}

        metrics = {}
        avg_grad = sum(recent_grads) / len(recent_grads)
        metrics['gradient/stability_avg'] = avg_grad
        metrics['gradient/stability_max'] = max(recent_grads)
        metrics['gradient/stability_min'] = min(recent_grads)

        # Alert if current gradient exceeds watch threshold
        if recent_grads[-1] > self.gradient_threshold:
            metrics['gradient/threshold_exceeded'] = 1.0
        else:
            metrics['gradient/threshold_exceeded'] = 0.0

        return metrics

    def _compute_noise_split_metrics(self) -> Dict[str, float]:
        """Compute HIGH/LOW noise split metrics from records.

        This is useful for WAN dual LoRA training where the model switches
        between HIGH noise (timesteps >= boundary) and LOW noise regimes.

        Returns:
            Dict with loss/high_noise_* and loss/low_noise_* metrics
        """
        # Separate by noise level
        high_losses = [r.loss for r in self.records if r.is_high_noise is True]
        low_losses = [r.loss for r in self.records if r.is_high_noise is False]

        metrics = {}

        if high_losses:
            # Use recent 100 for smoother metrics
            recent_high = high_losses[-100:]
            metrics['loss/high_noise_avg'] = sum(recent_high) / len(recent_high)
            metrics['loss/high_noise_count'] = float(len(high_losses))

            # Distribution stats
            if len(recent_high) > 1:
                import statistics
                metrics['loss/high_noise_std'] = statistics.stdev(recent_high)

        if low_losses:
            recent_low = low_losses[-100:]
            metrics['loss/low_noise_avg'] = sum(recent_low) / len(recent_low)
            metrics['loss/low_noise_count'] = float(len(low_losses))

            # Distribution stats
            if len(recent_low) > 1:
                import statistics
                metrics['loss/low_noise_std'] = statistics.stdev(recent_low)

        # Current batch classification (for visualization)
        if self.records and self.records[-1].is_high_noise is not None:
            metrics['loss/current_is_high_noise'] = float(self.records[-1].is_high_noise)

        return metrics

    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute R² convergence trends for multiple time windows.

        This analyzes convergence at different scales to detect both
        short-term and long-term training trends.

        Returns:
            Dict with convergence/r2_* metrics for each window size
        """
        metrics = {}

        for window in self.convergence_windows:
            if len(self.records) >= window:
                recent_losses = [r.loss for r in list(self.records)[-window:]]
                r_squared = self._compute_r_squared(recent_losses)
                metrics[f'convergence/r2_{window}step'] = r_squared

        return metrics

    def _compute_oscillation_metrics(self) -> Dict[str, float]:
        """Compute oscillation bounds from recent history.

        Tracks upper/lower bounds and oscillation range to detect
        training stability patterns.

        Returns:
            Dict with loss/upper_bound, loss/lower_bound, loss/oscillation_range
        """
        window = 50
        if len(self.records) < window:
            return {}

        recent_losses = [r.loss for r in list(self.records)[-window:]]

        return {
            'loss/upper_bound': max(recent_losses),
            'loss/lower_bound': min(recent_losses),
            'loss/oscillation_range': max(recent_losses) - min(recent_losses),
        }

    @staticmethod
    def _compute_r_squared(data: List[float]) -> float:
        """Compute R² coefficient for linear trend in data.

        R² near 0: No trend (random walk)
        R² near 1: Strong trend (converging or diverging)
        Negative R² indicates correlation is negative (downward trend)

        Args:
            data: List of loss values

        Returns:
            R² coefficient (0.0 if computation fails or insufficient data)
        """
        if len(data) < 2:
            return 0.0

        try:
            x = torch.arange(len(data), dtype=torch.float32)
            y = torch.tensor(data, dtype=torch.float32)

            # Compute correlation matrix
            corr_matrix = torch.corrcoef(torch.stack([x, y]))
            correlation = corr_matrix[0, 1]

            # R² is square of correlation coefficient
            r_squared = (correlation ** 2).item()

            return r_squared
        except Exception as e:
            logger.debug(f"R² computation failed: {e}")
            return 0.0

    def get_visualization_data(self) -> Dict[str, any]:
        """Get data for dashboard generation.

        Extracts data from records in a format suitable for creating
        comprehensive training visualizations (similar to Reddit report).

        Returns:
            Dict containing:
                - steps: List of step numbers
                - losses: List of loss values
                - gradient_norms: List of gradient norms (sparse)
                - high_noise_losses: List of (step, loss) tuples for HIGH noise
                - low_noise_losses: List of (step, loss) tuples for LOW noise
                - total_records: Total number of records stored
        """
        if not self.enabled:
            return {}

        # Extract data from records efficiently
        steps = [r.step for r in self.records]
        losses = [r.loss for r in self.records]
        gradient_norms = [
            (r.step, r.gradient_norm)
            for r in self.records
            if r.gradient_norm is not None
        ]

        high_noise_losses = [
            (r.step, r.loss)
            for r in self.records
            if r.is_high_noise is True
        ]
        low_noise_losses = [
            (r.step, r.loss)
            for r in self.records
            if r.is_high_noise is False
        ]

        return {
            'steps': steps,
            'losses': losses,
            'gradient_norms': gradient_norms,
            'high_noise_losses': high_noise_losses,
            'low_noise_losses': low_noise_losses,
            'total_records': len(self.records),
        }

    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for the current training session.

        Returns:
            Dict with summary statistics and feature status
        """
        if not self.enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'total_steps': self.step_count,
            'total_records': len(self.records),
            'features': sorted(self.features),
            'max_history': self.records.maxlen,
            'has_noise_split': "noise_split" in self.features,
        }
