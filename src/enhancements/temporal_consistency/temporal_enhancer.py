import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import time

from .utils import TemporalConsistencyMonitor

from .config import TemporalConsistencyConfig
from .frequency_analyzer import FrequencyAnalyzer
from .loss_computer import TemporalConsistencyLossComputer

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TemporalConsistencyEnhancer:
    """Main enhancement engine for frequency-domain temporal consistency in training."""

    def __init__(self, config: TemporalConsistencyConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cpu")

        # Initialize components
        self.freq_analyzer = FrequencyAnalyzer(
            device=self.device, enable_caching=self.config.freq_temporal_enable_caching
        )
        self.loss_computer = TemporalConsistencyLossComputer(config, device)

        # Performance and monitoring
        self.total_enhanced_batches = 0
        self.total_enhancement_time = 0.0
        self.loss_history = []
        self._has_logged_activation = False
        # TensorBoard logging
        self._tb_logger = None
        try:
            self._tb_every_steps = max(
                1, int(self.config.freq_temporal_tb_log_every_steps)
            )
        except Exception:
            self._tb_every_steps = 500

        logger.info(
            f"TemporalConsistencyEnhancer initialized with config: {self.config}"
        )

    def is_enabled(self) -> bool:
        """Check if temporal consistency enhancement is enabled."""
        return self.config.is_enabled()

    def should_apply_to_batch(self, tensor: torch.Tensor, step: int) -> bool:
        """Check if enhancement should be applied to this batch at this step."""
        if not self.config.should_apply_at_step(step):
            return False

        # Must be video batch with sufficient frames
        if not self.loss_computer.is_video_batch(tensor):
            return False

        return True

    def enhance_training_loss(
        self,
        base_loss: torch.Tensor,
        model_pred: torch.Tensor,
        target: Optional[torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Enhance training loss with temporal consistency components.

        This is the main integration point for the training loop.

        Args:
            base_loss: Original training loss
            model_pred: Model predictions
            target: Target values (optional)
            step: Current training step

        Returns:
            Tuple of (enhanced_loss, enhancement_info_dict)
        """
        if not self.should_apply_to_batch(model_pred, step):
            return base_loss, {"temporal_enhancement_applied": False}

        start_time = time.time()

        try:
            # Compute temporal consistency loss
            temporal_loss, loss_components = (
                self.loss_computer.compute_total_temporal_consistency_loss(
                    pred_video=model_pred, target_video=target, step=step
                )
            )

            # Combine with base loss
            enhanced_loss = base_loss + temporal_loss

            # Update performance tracking
            self.total_enhanced_batches += 1
            self.total_enhancement_time += time.time() - start_time

            # Store loss history for analysis
            self.loss_history.append(
                {
                    "step": step,
                    "base_loss": base_loss.item(),
                    "temporal_loss": temporal_loss.item(),
                    "enhanced_loss": enhanced_loss.item(),
                    **loss_components,
                }
            )

            # Prepare enhancement info
            enhancement_info = {
                "temporal_enhancement_applied": True,
                "temporal_loss_magnitude": temporal_loss.item(),
                "base_loss_magnitude": base_loss.item(),
                "enhancement_ratio": temporal_loss.item() / max(base_loss.item(), 1e-8),
                "processing_time_ms": (time.time() - start_time) * 1000,
                **loss_components,
            }

            # One-time INFO confirmation when enhancement first activates
            if not self._has_logged_activation and enhancement_info.get(
                "temporal_enhancement_applied", False
            ):
                logger.info(
                    (
                        "🌀 Temporal Consistency ACTIVE at step %d | "
                        "structural=%s (w=%.3f), motion=%s (w=%.3f), prediction=%s (w=%.3f) | "
                        "temporal_loss=%.6f, ratio=%.3f"
                    ),
                    step,
                    self.config.enable_frequency_domain_temporal_consistency,
                    self.config.freq_temporal_consistency_weight,
                    self.config.freq_temporal_enable_motion_coherence,
                    self.config.freq_temporal_motion_weight,
                    self.config.freq_temporal_enable_prediction_loss,
                    self.config.freq_temporal_prediction_weight,
                    temporal_loss.item(),
                    enhancement_info["enhancement_ratio"],
                )
                self._has_logged_activation = True

            # Log periodically
            if step % max(1, int(self.config.freq_temporal_log_every_steps)) == 0:
                logger.info(
                    f"🌀 TC step {step}: base={base_loss.item():.6f}, temporal={temporal_loss.item():.6f}, total={enhanced_loss.item():.6f}"
                )

            # TensorBoard logging if configured
            try:
                if (
                    self._tb_logger is not None
                    and step % max(1, int(self._tb_every_steps)) == 0
                ):
                    stats = self.get_performance_stats()
                    logs: dict = {}

                    def _add_tc_metric(key: str, value):
                        if isinstance(value, (int, float)):
                            logs[key] = float(value)

                    _add_tc_metric(
                        "tc/total_enhanced_batches", stats.get("total_enhanced_batches")
                    )
                    _add_tc_metric(
                        "tc/avg_enhancement_time_ms",
                        stats.get("average_enhancement_time_ms"),
                    )
                    _add_tc_metric(
                        "tc/recent_avg_temporal_loss",
                        stats.get("recent_avg_temporal_loss"),
                    )
                    _add_tc_metric(
                        "tc/recent_avg_enhancement_ratio",
                        stats.get("recent_avg_enhancement_ratio"),
                    )
                    _add_tc_metric(
                        "tc/loss_history_length", stats.get("loss_history_length")
                    )

                    lcs = stats.get("loss_computer_stats", {}) or {}
                    _add_tc_metric(
                        "tc/loss_computer/avg_comp_ms",
                        lcs.get("average_computation_time_ms"),
                    )
                    _add_tc_metric(
                        "tc/loss_computer/total_computations",
                        lcs.get("total_computations"),
                    )

                    fac = stats.get("frequency_analyzer_stats", {}) or {}
                    _add_tc_metric(
                        "tc/freq_cache/cached_masks", fac.get("cached_masks")
                    )

                    mon = stats.get("monitor", {}) or {}
                    _add_tc_metric("tc/monitor/success_rate", mon.get("success_rate"))
                    _add_tc_metric(
                        "tc/monitor/average_enhancement_ratio",
                        mon.get("average_enhancement_ratio"),
                    )
                    _add_tc_metric(
                        "tc/monitor/average_processing_time_ms",
                        mon.get("average_processing_time_ms"),
                    )

                    if logs:
                        self._tb_logger(logs, step)
            except Exception:
                pass

            return enhanced_loss, enhancement_info

        except Exception as e:
            logger.error(f"Error in temporal consistency enhancement: {e}")
            return base_loss, {"temporal_enhancement_applied": False, "error": str(e)}

    def analyze_video_temporal_consistency(
        self, video_tensor: torch.Tensor, step: int
    ) -> Dict[str, Any]:
        """Analyze temporal consistency of video for debugging/monitoring.

        Args:
            video_tensor: Video batch to analyze
            step: Current training step

        Returns:
            Dictionary with temporal analysis results
        """
        if not self.loss_computer.is_video_batch(video_tensor):
            return {"error": "Not a valid video batch"}

        try:
            # Use frequency analyzer for detailed analysis
            analysis = self.freq_analyzer.analyze_temporal_consistency(video_tensor)

            # Add step information
            analysis.update(
                {
                    "analysis_step": step,
                    "video_shape": list(video_tensor.shape),
                    "enhancement_enabled": self.is_enabled(),
                }
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in temporal consistency analysis: {e}")
            return {"error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_time = (
            self.total_enhancement_time / max(self.total_enhanced_batches, 1)
            if self.total_enhanced_batches > 0
            else 0.0
        )

        # Loss statistics from history
        loss_stats = {}
        if self.loss_history:
            recent_losses = self.loss_history[-100:]  # Last 100 entries

            loss_stats = {
                "recent_avg_base_loss": sum(l["base_loss"] for l in recent_losses)
                / len(recent_losses),
                "recent_avg_temporal_loss": sum(
                    l["temporal_loss"] for l in recent_losses
                )
                / len(recent_losses),
                "recent_avg_enhancement_ratio": sum(
                    l.get("enhancement_ratio", 0) for l in recent_losses
                )
                / len(recent_losses),
                "loss_history_length": len(self.loss_history),
            }

        base_stats = {
            "total_enhanced_batches": self.total_enhanced_batches,
            "total_enhancement_time": self.total_enhancement_time,
            "average_enhancement_time_ms": avg_time * 1000,
            "config_summary": {
                "temporal_consistency_weight": self.config.freq_temporal_consistency_weight,
                "motion_coherence_weight": self.config.freq_temporal_motion_weight,
                "frequency_temporal_weight": self.config.freq_temporal_prediction_weight,
                "low_freq_threshold": self.config.freq_temporal_low_threshold,
                "max_temporal_distance": self.config.freq_temporal_max_distance,
            },
        }

        # Combine all stats
        all_stats = {**base_stats, **loss_stats}

        # Add component stats
        all_stats["frequency_analyzer_stats"] = self.freq_analyzer.get_cache_stats()
        all_stats["loss_computer_stats"] = self.loss_computer.get_performance_stats()

        return all_stats

    def get_recent_loss_trends(self, num_recent: int = 50) -> Dict[str, Any]:
        """Get recent loss trends for monitoring."""
        if len(self.loss_history) < num_recent:
            recent_history = self.loss_history
        else:
            recent_history = self.loss_history[-num_recent:]

        if not recent_history:
            return {"error": "No loss history available"}

        # Calculate trends
        base_losses = [l["base_loss"] for l in recent_history]
        temporal_losses = [l["temporal_loss"] for l in recent_history]
        enhanced_losses = [l["enhanced_loss"] for l in recent_history]

        return {
            "num_samples": len(recent_history),
            "base_loss_trend": {
                "mean": sum(base_losses) / len(base_losses),
                "min": min(base_losses),
                "max": max(base_losses),
            },
            "temporal_loss_trend": {
                "mean": sum(temporal_losses) / len(temporal_losses),
                "min": min(temporal_losses),
                "max": max(temporal_losses),
            },
            "enhanced_loss_trend": {
                "mean": sum(enhanced_losses) / len(enhanced_losses),
                "min": min(enhanced_losses),
                "max": max(enhanced_losses),
            },
            "recent_steps": [l["step"] for l in recent_history[-5:]],  # Last 5 steps
        }

    def cleanup(self):
        """Cleanup resources and prepare for shutdown."""
        self.freq_analyzer.clear_cache()
        self.loss_computer.cleanup()

        # Clear loss history to free memory
        self.loss_history.clear()

        logger.info(
            f"TemporalConsistencyEnhancer cleanup completed. "
            f"Enhanced {self.total_enhanced_batches} batches in "
            f"{self.total_enhancement_time:.2f}s"
        )

    # Public API: allow host to register TB logger and cadence
    def set_tensorboard_logger(
        self, logger_fn, every_steps: Optional[int] = None
    ) -> None:
        self._tb_logger = logger_fn
        if every_steps is not None:
            try:
                self._tb_every_steps = max(1, int(every_steps))
            except Exception:
                self._tb_every_steps = max(1, self._tb_every_steps)

    @classmethod
    def initialize_from_args(
        cls, args, device=None
    ) -> Optional["TemporalConsistencyEnhancer"]:
        """Initialize temporal consistency enhancement from args if enabled.

        This encapsulates the initialization logic to keep training_core clean.

        Args:
            args: Command line arguments or config namespace
            device: Device to run operations on

        Returns:
            TemporalConsistencyEnhancer instance if enabled, None otherwise
        """
        try:
            from .main import is_temporal_consistency_available
            from .config import TemporalConsistencyConfig

            if not is_temporal_consistency_available():
                logger.warning(
                    "Temporal consistency enhancement not available - missing torch.fft support"
                )
                return None

            config = TemporalConsistencyConfig.from_args(args)
            if config.is_enabled():
                device = device or getattr(args, "device", torch.device("cpu"))
                enhancer = cls(config, device)

                logger.info("✅ Temporal consistency enhancement initialized")

                # Log enabled features
                enabled_features = []
                if config.enable_frequency_domain_temporal_consistency:
                    enabled_features.append(
                        f"structural_consistency(threshold={config.freq_temporal_low_threshold:.2f})"
                    )
                if config.freq_temporal_enable_motion_coherence:
                    enabled_features.append(
                        f"motion_coherence(weight={config.freq_temporal_motion_weight:.3f})"
                    )
                if config.freq_temporal_enable_prediction_loss:
                    enabled_features.append(
                        f"freq_temporal(weight={config.freq_temporal_prediction_weight:.3f})"
                    )

                logger.info(f"🎯 Enabled features: {', '.join(enabled_features)}")
                return enhancer
            else:
                logger.info("Temporal consistency enhancement disabled in config")
                return None

        except ImportError as e:
            logger.warning(
                f"Cannot initialize temporal consistency enhancement - import error: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize temporal consistency enhancement: {e}")
            return None

    def enhance_training_loss_with_monitoring(
        self,
        base_loss: torch.Tensor,
        model_pred: torch.Tensor,
        target: Optional[torch.Tensor],
        step: int,
        monitor: Optional["TemporalConsistencyMonitor"] = None,
    ) -> torch.Tensor:
        """Enhanced version with built-in monitoring.

        This combines enhancement and monitoring to keep training_core integration simple.

        Args:
            base_loss: Original training loss
            model_pred: Model predictions
            target: Target values (optional)
            step: Current training step
            monitor: Optional monitoring instance

        Returns:
            Enhanced loss with temporal consistency components
        """
        if not self.is_enabled():
            return base_loss

        try:
            enhanced_loss, enhancement_info = self.enhance_training_loss(
                base_loss=base_loss, model_pred=model_pred, target=target, step=step
            )

            # Log enhancement information if monitor is available
            if monitor is not None:
                monitor.log_enhancement(step, enhancement_info)

            return enhanced_loss

        except Exception as e:
            logger.warning(f"Error in temporal consistency enhancement: {e}")
            return base_loss
