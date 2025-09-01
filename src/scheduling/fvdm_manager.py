"""
Unified FVDM Manager - Clean interface for all FVDM functionality.

This manager encapsulates all FVDM-related logic including:
- Adaptive PTSS scheduling
- Integration with AdaptiveTimestepManager
- Training metrics and evaluation
- Loss components
- Logging and statistics

Usage:
    fvdm_manager = FVDMManager(args, device, adaptive_manager)

    # In training loop:
    noisy_input, timesteps, sigmas = fvdm_manager.get_noisy_input_and_timesteps(...)
    fvdm_manager.record_training_step(...)
    additional_loss = fvdm_manager.get_additional_loss(...)
"""

import argparse
import torch
import logging
from typing import Tuple, Any, Optional, Dict
from utils.fvdm_metrics import FVDMTrainingMetrics, compute_fvdm_loss_components
import scheduling.fvdm as fvdm_core

logger = logging.getLogger(__name__)


class FVDMManager:
    """
    Unified manager for all FVDM functionality.

    Handles FVDM training, metrics, and integration in a clean, encapsulated way.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        adaptive_manager: Optional[Any] = None,
    ):
        """
        Initialize FVDM manager.

        Args:
            args: Training arguments
            device: Device for computations
            adaptive_manager: Optional AdaptiveTimestepManager for integration
        """
        self.args = args
        self.device = device
        self.adaptive_manager = adaptive_manager

        # Check if FVDM is enabled
        self.enabled = getattr(args, "enable_fvdm", False)

        if not self.enabled:
            # Disabled - create no-op manager
            self.metrics = None
            self._log_disabled()
            return

        # FVDM is enabled - initialize all components
        self._initialize_components()
        self._log_initialization()

    def _initialize_components(self):
        """Initialize FVDM components when enabled."""
        # Initialize metrics
        self.metrics = FVDMTrainingMetrics(device=self.device)

        # Cache frequently accessed settings
        self.eval_temporal_metrics = getattr(
            self.args, "fvdm_eval_temporal_metrics", False
        )
        self.eval_frequency = getattr(self.args, "fvdm_eval_frequency", 1000)
        self.temporal_loss_weight = getattr(
            self.args, "fvdm_temporal_consistency_weight", 0.0
        )
        self.diversity_loss_weight = getattr(
            self.args, "fvdm_frame_diversity_weight", 0.0
        )
        self.integrate_adaptive = getattr(
            self.args, "fvdm_integrate_adaptive_timesteps", False
        )

        # Track current step for adaptive features
        self.current_step = 0

    def _log_disabled(self):
        """Log that FVDM is disabled."""
        logger.debug("FVDM Manager initialized in disabled mode")

    def _log_initialization(self):
        """Log FVDM initialization details."""
        logger.info("🎬 FVDM Manager initialized with enhanced features:")
        ptss_mode = (
            "Adaptive" if getattr(self.args, "fvdm_adaptive_ptss", False) else "Fixed"
        )
        logger.info(f"   PTSS Mode: {ptss_mode}")
        if ptss_mode == "Adaptive":
            logger.info(
                f"   PTSS Adaptive: initial={getattr(self.args, 'fvdm_ptss_initial', 0.3)}, "
                f"final={getattr(self.args, 'fvdm_ptss_final', 0.1)}, "
                f"warmup={getattr(self.args, 'fvdm_ptss_warmup', 1000)}"
            )
        else:
            logger.info(
                f"   PTSS Probability (fixed): {getattr(self.args, 'fvdm_ptss_p', 0.2)}"
            )
        logger.info(
            f"   Timestep bounds: [{getattr(self.args, 'min_timestep', 0)}, "
            f"{getattr(self.args, 'max_timestep', 1000)}] out of [0, 1000]"
        )
        logger.info(f"   Adaptive Integration: {self.integrate_adaptive}")
        logger.info(f"   Temporal Metrics: {self.eval_temporal_metrics}")
        logger.info(
            f"   Additional Loss Components: {self.temporal_loss_weight > 0 or self.diversity_loss_weight > 0}"
        )

    def get_noisy_input_and_timesteps(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        noise_scheduler: Any,
        dtype: torch.dtype,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get noisy input and timesteps using FVDM or fallback to standard method.

        Args:
            noise: Noise tensor
            latents: Clean latents
            noise_scheduler: Noise scheduler
            dtype: Data type
            step: Current training step

        Returns:
            Tuple of (noisy_model_input, timesteps, sigmas)
        """
        if not self.enabled:
            # Not enabled - this should not be called, but provide safe fallback
            logger.warning(
                "FVDM Manager called when disabled - this indicates a code issue"
            )
            return (
                latents,
                torch.zeros(latents.shape[0], device=self.device),
                torch.zeros_like(latents),
            )

        # Update current step for adaptive features
        self.current_step = step

        # Call enhanced FVDM function
        return fvdm_core.get_noisy_model_input_and_timesteps_fvdm(
            self.args,
            noise,
            latents,
            noise_scheduler,
            self.device,
            dtype,
            current_step=step,
            adaptive_manager=self.adaptive_manager if self.integrate_adaptive else None,
        )

    def record_training_step(
        self,
        frames: torch.Tensor,
        timesteps: torch.Tensor,
        loss: torch.Tensor,
        step: int,
    ):
        """
        Record metrics for a training step.

        Args:
            frames: Video frames [B, C, F, H, W]
            timesteps: Used timesteps [B, F]
            loss: Training loss value
            step: Current step number
        """
        if not self.enabled or self.metrics is None:
            return

        try:
            # Determine if this was async sampling (heuristic based on timestep variance)
            timestep_variance = torch.var(timesteps.float(), dim=-1).mean()
            is_async = (
                timestep_variance.item() > 0.01
            )  # Threshold for considering it async

            # Get current PTSS probability
            if getattr(self.args, "fvdm_adaptive_ptss", False):
                ptss_p = fvdm_core.get_adaptive_ptss_p(
                    step,
                    getattr(self.args, "max_train_steps", 100000),
                    getattr(self.args, "fvdm_ptss_initial", 0.3),
                    getattr(self.args, "fvdm_ptss_final", 0.1),
                    getattr(self.args, "fvdm_ptss_warmup", 1000),
                )
            else:
                ptss_p = getattr(self.args, "fvdm_ptss_p", 0.2)

            # Record all metrics
            self.metrics.record_metrics(frames, timesteps, is_async, ptss_p)

        except Exception as e:
            logger.warning(f"Failed to record FVDM metrics: {e}")

    def get_additional_loss(
        self, frames: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute additional FVDM loss components.

        Args:
            frames: Video frames [B, C, F, H, W]
            timesteps: Used timesteps [B, F]

        Returns:
            Tuple of (additional_loss, loss_components_dict)
        """
        if not self.enabled or (
            self.temporal_loss_weight == 0 and self.diversity_loss_weight == 0
        ):
            # No additional loss needed
            return torch.tensor(0.0, device=self.device), {}

        try:
            return compute_fvdm_loss_components(
                frames,
                timesteps,
                temporal_weight=self.temporal_loss_weight,
                diversity_weight=self.diversity_loss_weight,
            )
        except Exception as e:
            logger.warning(f"Failed to compute FVDM loss components: {e}")
            return torch.tensor(0.0, device=self.device), {}

    def should_log_metrics(self, step: int) -> bool:
        """Check if we should log FVDM metrics at this step."""
        return (
            self.enabled
            and self.eval_temporal_metrics
            and step > 0
            and step % self.eval_frequency == 0
        )

    def get_metrics_for_logging(self, step: int) -> Dict[str, float]:
        """Get current FVDM metrics for logging."""
        if not self.enabled or self.metrics is None:
            return {}

        try:
            stats = self.metrics.get_stats()
            recent_stats = self.metrics.get_recent_stats(window=min(100, step))

            # Combine stats
            all_stats = {**stats, **recent_stats}

            # Add current PTSS probability
            if getattr(self.args, "fvdm_adaptive_ptss", False):
                current_ptss_p = fvdm_core.get_adaptive_ptss_p(
                    step,
                    getattr(self.args, "max_train_steps", 100000),
                    getattr(self.args, "fvdm_ptss_initial", 0.3),
                    getattr(self.args, "fvdm_ptss_final", 0.1),
                    getattr(self.args, "fvdm_ptss_warmup", 1000),
                )
                all_stats["fvdm/current_ptss_p"] = current_ptss_p

            return all_stats

        except Exception as e:
            logger.warning(f"Failed to get FVDM metrics for logging: {e}")
            return {}

    def get_status_summary(self) -> str:
        """Get a brief status summary for logging."""
        if not self.enabled:
            return "FVDM: Disabled"

        if self.metrics is None:
            return "FVDM: Enabled (metrics unavailable)"

        stats = self.metrics.get_stats()
        async_ratio = stats.get("fvdm/async_ratio", 0.0)
        total_steps = stats.get("fvdm/total_steps", 0)

        return f"FVDM: Enabled | {total_steps} steps | {async_ratio:.1%} async"

    def reset_metrics(self):
        """Reset accumulated metrics (useful for validation phases)."""
        if self.enabled and self.metrics is not None:
            self.metrics.reset()


def create_fvdm_manager(
    args: argparse.Namespace,
    device: torch.device,
    adaptive_manager: Optional[Any] = None,
) -> FVDMManager:
    """
    Factory function to create FVDM manager.

    Args:
        args: Training arguments
        device: Device for computations
        adaptive_manager: Optional adaptive timestep manager

    Returns:
        Configured FVDMManager instance
    """
    return FVDMManager(args, device, adaptive_manager)
