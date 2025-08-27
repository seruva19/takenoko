## Based on https://arxiv.org/abs/2411.09998 and https://github.com/ku-dmlab/Adaptive-Timestep-Sampler (MIT)

"""Adaptive Timestep Sampling implementation for diffusion model training.

This module implements adaptive timestep sampling, a technique that identifies and focuses training
on the most critical timesteps for video generation quality. Unlike standard uniform timestep
sampling, adaptive timestep sampling dynamically adapts to focus on timesteps that have the
highest impact on model performance based on loss analysis.

Designed specifically for WanVideo LoRA training in Takenoko with full boundary respect.
"""

import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Try to import scikit-learn for statistical feature selection
try:
    from sklearn.feature_selection import SelectKBest, f_regression

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning(
        "scikit-learn not available. Statistical feature selection will be disabled."
    )
    SelectKBest = None
    f_regression = None
    SKLEARN_AVAILABLE = False


class AdaptiveTimestepManager:
    """Manages adaptive timestep sampling for diffusion model training.

    This class implements the core adaptive timestep sampling algorithm which:
    1. Analyzes loss patterns across timesteps during training
    2. Identifies timesteps that are most critical for generation quality
    3. Dynamically adjusts timestep sampling to focus on these critical points
    4. Provides adaptive loss weighting based on timestep importance

    The system is designed to be gated (disabled by default) and respects existing
    timestep boundaries in Takenoko's training pipeline.
    """

    def __init__(
        self,
        enabled: bool = False,
        analysis_window: int = 1000,
        importance_detection_threshold: float = 1.5,
        focus_strength: float = 2.0,
        min_important_timesteps: int = 50,
        max_important_timesteps: int = 200,
        warmup_steps: int = 500,
        update_frequency: int = 100,
        video_specific_categories: bool = True,
        motion_weight: float = 1.0,
        detail_weight: float = 1.0,
        temporal_weight: float = 1.0,
        min_timestep: Optional[int] = None,
        max_timestep: Optional[int] = None,
        # Research paper alignment parameters
        use_beta_sampler: bool = False,
        feature_selection_size: int = 3,
        sampler_update_frequency: int = 40,
        use_neural_sampler: bool = False,
        beta_alpha_init: float = 1.0,
        beta_beta_init: float = 1.0,
        neural_hidden_size: int = 64,
        # Complementary approach parameters
        use_importance_weighting: bool = True,
        use_kl_reward_learning: bool = False,
        use_replay_buffer: bool = False,
        use_statistical_features: bool = False,
        weight_combination: str = "fallback",
        replay_buffer_size: int = 100,
        rl_learning_rate: float = 1e-4,
        entropy_coefficient: float = 0.01,
        kl_update_frequency: int = 20,
    ):
        """Initialize the AdaptiveTimestepManager.

        Args:
            enabled: Whether adaptive timestep sampling is enabled (gated by default)
            analysis_window: Number of recent loss samples to analyze
            importance_detection_threshold: Threshold multiplier for detecting important timesteps
            focus_strength: How much to focus sampling on important timesteps
            min_important_timesteps: Minimum number of important timesteps to maintain
            max_important_timesteps: Maximum number of important timesteps to maintain
            warmup_steps: Steps before importance analysis begins
            update_frequency: How often to update important timesteps
            video_specific_categories: Enable video-specific timestep categories
            motion_weight: Weight for motion consistency timesteps (early range)
            detail_weight: Weight for detail preservation timesteps (middle range)
            temporal_weight: Weight for temporal coherence timesteps (late range)
            min_timestep: Minimum timestep boundary (0-1000), respects existing boundaries
            max_timestep: Maximum timestep boundary (0-1000), respects existing boundaries
            use_beta_sampler: Enable Beta distribution sampling (paper methodology)
            feature_selection_size: |S| for feature selection approximation
            sampler_update_frequency: f_S frequency for sampler parameter updates
            use_neural_sampler: Enable separate neural network for timestep sampling
            beta_alpha_init: Initial Alpha parameter for Beta distribution
            beta_beta_init: Initial Beta parameter for Beta distribution
            neural_hidden_size: Hidden layer size for neural sampler network
            use_importance_weighting: Enable loss-variance importance weighting (stable)
            use_kl_reward_learning: Enable KL divergence RL learning (paper exact)
            use_replay_buffer: Enable replay buffer for historical learning
            use_statistical_features: Enable SelectKBest statistical feature selection
            weight_combination: How to combine approaches ("fallback", "ensemble", "best")
            replay_buffer_size: Size of replay buffer for KL differences
            rl_learning_rate: Learning rate for RL policy updates
            entropy_coefficient: Entropy regularization coefficient for RL
            kl_update_frequency: How often to update via KL reward signal
        """
        self.enabled = enabled
        self.analysis_window = analysis_window
        self.importance_detection_threshold = importance_detection_threshold
        self.focus_strength = focus_strength
        self.min_important_timesteps = min_important_timesteps
        self.max_important_timesteps = max_important_timesteps
        self.warmup_steps = warmup_steps
        self.update_frequency = update_frequency
        self.video_specific_categories = video_specific_categories
        self.motion_weight = motion_weight
        self.detail_weight = detail_weight
        self.temporal_weight = temporal_weight

        # Research paper alignment parameters
        self.use_beta_sampler = use_beta_sampler
        self.feature_selection_size = feature_selection_size
        self.sampler_update_frequency = sampler_update_frequency
        self.use_neural_sampler = use_neural_sampler
        self.beta_alpha_init = beta_alpha_init
        self.beta_beta_init = beta_beta_init
        self.neural_hidden_size = neural_hidden_size

        # Complementary approach parameters
        self.use_importance_weighting = use_importance_weighting
        self.use_kl_reward_learning = use_kl_reward_learning
        self.use_replay_buffer = use_replay_buffer
        self.use_statistical_features = use_statistical_features
        self.weight_combination = weight_combination
        self.replay_buffer_size = replay_buffer_size
        self.rl_learning_rate = rl_learning_rate
        self.entropy_coefficient = entropy_coefficient
        self.kl_update_frequency = kl_update_frequency

        # Boundary constraints - respect existing Takenoko timestep boundaries
        self.min_timestep = min_timestep if min_timestep is not None else 0
        self.max_timestep = max_timestep if max_timestep is not None else 1000
        self.min_t_normalized = self.min_timestep / 1000.0  # Convert to 0-1 range
        self.max_t_normalized = self.max_timestep / 1000.0

        # Internal state
        self.step_count = 0
        self.loss_history: Dict[int, deque] = {}  # timestep -> loss values
        self.important_timesteps: List[int] = []
        self.timestep_weights: Dict[int, float] = {}
        self.timestep_importance: Dict[int, float] = {}
        self.last_update_step = 0

        # Video-specific timestep categories (within boundaries)
        self.motion_timesteps: List[int] = []  # Early timesteps within boundary
        self.detail_timesteps: List[int] = []  # Middle timesteps within boundary
        self.temporal_timesteps: List[int] = []  # Late timesteps within boundary

        # Beta sampler state (paper implementation)
        self.beta_alpha = beta_alpha_init
        self.beta_beta = beta_beta_init
        self.feature_selection_step = 0
        self.selected_features = []  # Current |S| = feature_selection_size features
        self.neural_sampler = None

        # Initialize neural sampler if enabled
        if self.use_neural_sampler and self.enabled:
            self._initialize_neural_sampler()

        # Complementary approach state
        self.replay_buffer = None
        self.kl_history = (
            deque(maxlen=self.replay_buffer_size) if self.use_replay_buffer else None
        )
        self.rl_optimizer = None
        self.last_kl_update_step = 0
        self.statistical_selector = None

        # Initialize complementary components
        if self.enabled:
            self._initialize_complementary_components()

        # Statistics tracking
        self.stats = {
            "total_important_detected": 0,
            "importance_updates": 0,
            "avg_importance": 0.0,
            "timestep_coverage": 0.0,
            "beta_alpha": self.beta_alpha,
            "beta_beta": self.beta_beta,
        }

        if self.enabled:
            logger.info("🎯 Adaptive Timestep Sampling enabled")
            logger.info(
                f"   Boundary range: [{self.min_timestep}, {self.max_timestep}] (0-1000 scale)"
            )
            logger.info(f"   Analysis window: {self.analysis_window}")
            logger.info(
                f"   Importance detection threshold: {self.importance_detection_threshold}"
            )
            logger.info(f"   Focus strength: {self.focus_strength}")
            logger.info(f"   Warmup steps: {self.warmup_steps}")
            logger.info(
                f"   Video-specific categories: {self.video_specific_categories}"
            )
        else:
            logger.debug("Adaptive Timestep Sampling is disabled")

    def record_timestep_loss(
        self, timesteps: torch.Tensor, losses: torch.Tensor
    ) -> None:
        """Record loss values for given timesteps (only within boundaries).

        Args:
            timesteps: Timestep values (0-1 normalized range)
            losses: Corresponding loss values
        """
        if not self.enabled:
            return

        self.step_count += 1

        # Convert normalized timesteps to integers for indexing
        timestep_ints = (timesteps * 1000).long().cpu().numpy()
        loss_values = losses.detach().cpu().numpy()

        # Record losses for each timestep, but only within boundaries
        for timestep_int, loss_value in zip(timestep_ints, loss_values):
            timestep_int = int(timestep_int)

            # Respect boundary constraints - only record within allowed range
            if not (self.min_timestep <= timestep_int <= self.max_timestep):
                continue

            if timestep_int not in self.loss_history:
                self.loss_history[timestep_int] = deque(maxlen=self.analysis_window)

            self.loss_history[timestep_int].append(float(loss_value))

    def should_update_importance(self) -> bool:
        """Check if important timesteps should be updated.

        Returns:
            True if importance should be updated
        """
        if not self.enabled:
            return False

        # Wait for warmup period
        if self.step_count < self.warmup_steps:
            return False

        # Update at specified frequency
        return (self.step_count - self.last_update_step) >= self.update_frequency

    def update_important_timesteps(self) -> None:
        """Update the important timesteps based on accumulated loss data."""
        if not self.enabled or not self.should_update_importance():
            return

        if len(self.loss_history) < 10:  # Need minimum data
            return

        logger.debug(f"🎯 Updating important timesteps at step {self.step_count}")

        # Calculate importance scores for timesteps within boundaries
        self.timestep_importance = self._calculate_timestep_importance()

        if not self.timestep_importance:
            logger.warning("No timestep importance data available for update")
            return

        # Detect important timesteps based on importance within boundaries
        self.important_timesteps = self._detect_important_timesteps()

        # Calculate timestep weights
        self.timestep_weights = self._calculate_timestep_weights()

        # Update video-specific timestep categories if enabled
        if self.video_specific_categories:
            self._update_video_specific_timesteps()

        self.last_update_step = self.step_count
        self.stats["importance_updates"] += 1

        # Log update results
        self._log_importance_update()

    def _calculate_timestep_importance(self) -> Dict[int, float]:
        """Calculate importance scores for each timestep within boundaries.

        Importance is based on:
        1. Loss variance (higher variance = more critical)
        2. Recent loss trends (increasing losses = more critical)
        3. Statistical significance (sufficient data points)

        Returns:
            Dictionary mapping timestep to importance score (only within boundaries)
        """
        importance_scores = {}

        for timestep, loss_values in self.loss_history.items():
            # Double-check boundary constraints
            if not (self.min_timestep <= timestep <= self.max_timestep):
                continue

            if len(loss_values) < 10:  # Need minimum samples
                continue

            losses = np.array(list(loss_values))

            # Calculate various importance metrics
            loss_mean = np.mean(losses)
            loss_var = np.var(losses)
            loss_std = np.std(losses)

            # Recent trend analysis (last 25% of samples)
            recent_size = max(len(losses) // 4, 5)
            recent_losses = losses[-recent_size:]
            recent_mean = np.mean(recent_losses)

            # Trend score (positive if losses are increasing)
            trend_score = (recent_mean - loss_mean) / max(loss_std, 1e-6)

            # Variance-based importance (normalized)
            variance_importance = loss_var / max(loss_mean, 1e-6)

            # Combine metrics into importance score
            importance = variance_importance + max(trend_score, 0) * 0.5

            # Bonus for having sufficient data
            data_bonus = min(len(losses) / self.analysis_window, 1.0) * 0.1
            importance += data_bonus

            importance_scores[timestep] = importance

        return importance_scores

    def _detect_important_timesteps(self) -> List[int]:
        """Detect important timesteps based on importance scores within boundaries.

        Returns:
            List of timesteps identified as important (only within boundaries)
        """
        if not self.timestep_importance:
            return []

        # Calculate threshold for importance detection
        importance_values = list(self.timestep_importance.values())
        mean_importance = np.mean(importance_values)
        std_importance = np.std(importance_values)

        threshold = (
            mean_importance + self.importance_detection_threshold * std_importance
        )

        # Find timesteps above threshold (already within boundaries from calculation)
        candidate_important = [
            timestep
            for timestep, importance in self.timestep_importance.items()
            if importance >= threshold
        ]

        # Sort by importance (descending)
        candidate_important.sort(
            key=lambda t: self.timestep_importance[t], reverse=True
        )

        # Apply min/max constraints
        num_important = max(
            self.min_important_timesteps,
            min(len(candidate_important), self.max_important_timesteps),
        )

        return candidate_important[:num_important]

    def _calculate_timestep_weights(self) -> Dict[int, float]:
        """Calculate sampling weights for important timesteps.

        Returns:
            Dictionary mapping timestep to sampling weight
        """
        if not self.important_timesteps or not self.timestep_importance:
            return {}

        weights = {}
        max_importance = max(
            self.timestep_importance.get(t, 0) for t in self.important_timesteps
        )

        for timestep in self.important_timesteps:
            importance = self.timestep_importance.get(timestep, 0)
            # Normalize and apply focus strength
            normalized_importance = importance / max(max_importance, 1e-6)
            weight = 1.0 + (self.focus_strength - 1.0) * normalized_importance
            weights[timestep] = weight

        return weights

    def _update_video_specific_timesteps(self) -> None:
        """Update video-specific timestep categories within boundaries."""
        if not self.important_timesteps:
            return

        # Clear existing categories
        self.motion_timesteps.clear()
        self.detail_timesteps.clear()
        self.temporal_timesteps.clear()

        # Calculate category boundaries within the constrained range
        boundary_range = self.max_timestep - self.min_timestep
        motion_threshold = self.min_timestep + boundary_range * 0.33
        detail_threshold = self.min_timestep + boundary_range * 0.66

        # Categorize important timesteps by ranges within boundaries
        for timestep in self.important_timesteps:
            if timestep <= motion_threshold:  # Early timesteps - motion consistency
                self.motion_timesteps.append(timestep)
            elif timestep <= detail_threshold:  # Middle timesteps - detail preservation
                self.detail_timesteps.append(timestep)
            else:  # Late timesteps - temporal coherence
                self.temporal_timesteps.append(timestep)

    def get_adaptive_sampling_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sampling weights for given timesteps based on importance analysis.

        Args:
            timesteps: Timestep values (0-1 normalized range)

        Returns:
            Weight multipliers for each timestep
        """
        if not self.enabled:
            return torch.ones_like(timesteps)

        # Use new complementary approach if any advanced methods are enabled
        if self.use_kl_reward_learning or (
            self.use_importance_weighting
            and (self.use_beta_sampler or self.use_statistical_features)
        ):
            return self.get_combined_sampling_weights(timesteps)

        # Legacy single-method approach for backward compatibility
        # Try Beta sampler first if enabled
        if self.use_beta_sampler:
            beta_weights = self.get_beta_sampling_weights(timesteps)
            if beta_weights is not None:
                return beta_weights

        # Fallback to importance weighting
        return self._get_importance_weights_only(timesteps)  # type: ignore

    def get_adaptive_timestep_distribution(
        self, base_distribution: torch.Tensor
    ) -> torch.Tensor:
        """Modify a timestep distribution to focus on important timesteps within boundaries.

        Args:
            base_distribution: Original timestep distribution

        Returns:
            Modified distribution that focuses on important timesteps
        """
        if not self.enabled or not self.important_timesteps:
            return base_distribution

        # Create importance-focused distribution
        focused_distribution = base_distribution.clone()

        # Only modify timesteps within boundaries
        boundary_mask = (base_distribution >= self.min_t_normalized) & (
            base_distribution <= self.max_t_normalized
        )

        # Boost probability around important timesteps
        for important_timestep in self.important_timesteps:
            important_t = important_timestep / 1000.0  # Convert to 0-1 range

            # Find indices near this important timestep within boundaries
            distances = torch.abs(base_distribution - important_t)
            close_indices = (
                distances < 0.05
            ) & boundary_mask  # Within 5% and within boundaries

            # Boost probability for nearby timesteps
            weight = self.timestep_weights.get(important_timestep, 1.0)
            focused_distribution[close_indices] *= weight

        return focused_distribution

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about adaptive timestep sampling performance.

        Returns:
            Dictionary containing adaptive timestep sampling statistics
        """
        if not self.enabled:
            return {"enabled": False}

        # Update statistics
        self.stats["total_important_detected"] = len(self.important_timesteps)

        if self.timestep_importance:
            important_importances = [
                self.timestep_importance.get(t, 0) for t in self.important_timesteps
            ]
            self.stats["avg_importance"] = (
                np.mean(important_importances) if important_importances else 0.0
            )

        # Timestep coverage within boundaries
        if self.important_timesteps:
            boundary_range = self.max_timestep - self.min_timestep
            if boundary_range > 0:
                timestep_span = max(self.important_timesteps) - min(
                    self.important_timesteps
                )
                self.stats["timestep_coverage"] = timestep_span / boundary_range

        return {
            "enabled": True,
            "boundary_range": f"[{self.min_timestep}, {self.max_timestep}]",
            "step_count": self.step_count,
            "total_important": len(self.important_timesteps),
            "motion_timesteps": len(self.motion_timesteps),
            "detail_timesteps": len(self.detail_timesteps),
            "temporal_timesteps": len(self.temporal_timesteps),
            "importance_updates": self.stats["importance_updates"],
            "avg_importance": self.stats["avg_importance"],
            "timestep_coverage": self.stats["timestep_coverage"],
            "warmup_remaining": max(0, self.warmup_steps - self.step_count),
        }  # type: ignore

    def _log_importance_update(self) -> None:
        """Log information about the latest importance update."""
        stats = self.get_stats()

        logger.info(f"🎯 Importance update #{stats['importance_updates']}:")
        logger.info(f"   Boundary: {stats['boundary_range']}")
        logger.info(f"   Total important: {stats['total_important']}")

        if self.video_specific_categories:
            logger.info(
                f"   Motion timesteps: {stats['motion_timesteps']} (early range)"
            )
            logger.info(
                f"   Detail timesteps: {stats['detail_timesteps']} (middle range)"
            )
            logger.info(
                f"   Temporal timesteps: {stats['temporal_timesteps']} (late range)"
            )

        logger.info(f"   Avg importance: {stats['avg_importance']:.4f}")
        logger.info(f"   Coverage: {stats['timestep_coverage']:.1%} of boundary range")

        # Log top important timesteps
        if self.important_timesteps:
            top_important = self.important_timesteps[:5]  # Show top 5
            importance_info = []
            for timestep in top_important:
                importance = self.timestep_importance.get(timestep, 0)
                weight = self.timestep_weights.get(timestep, 1.0)
                importance_info.append(
                    f"{timestep}(imp={importance:.3f},w={weight:.2f})"
                )

            logger.info(f"   Top important: {', '.join(importance_info)}")

    def _initialize_neural_sampler(self):
        """Initialize the neural network for Beta distribution parameter prediction."""
        try:
            self.neural_sampler = BetaSamplerNetwork(
                input_size=self.feature_selection_size,
                hidden_size=self.neural_hidden_size,
            )
            logger.info(
                f"🧠 Neural Beta sampler initialized with {self.neural_hidden_size} hidden units"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize neural sampler: {e}")
            self.use_neural_sampler = False

    def _update_beta_parameters(self, features: torch.Tensor):
        """Update Beta distribution parameters using neural network or simple heuristics."""
        if self.use_neural_sampler and self.neural_sampler is not None:
            try:
                with torch.no_grad():
                    alpha, beta = self.neural_sampler(features)
                    self.beta_alpha = float(alpha.item())
                    self.beta_beta = float(beta.item())

                    # Update stats
                    self.stats["beta_alpha"] = self.beta_alpha
                    self.stats["beta_beta"] = self.beta_beta
            except Exception as e:
                logger.warning(f"Neural sampler failed, using heuristic update: {e}")
                self._update_beta_parameters_heuristic(features)
        else:
            self._update_beta_parameters_heuristic(features)

    def _update_beta_parameters_heuristic(self, features: torch.Tensor):
        """Simple heuristic-based Beta parameter updates when neural sampler unavailable."""
        if features.numel() > 0:
            # Simple heuristic: adjust based on feature variance
            feature_var = torch.var(features)
            if feature_var > 0.5:  # High variance -> more uniform sampling
                self.beta_alpha = max(0.5, self.beta_alpha - 0.1)
                self.beta_beta = max(0.5, self.beta_beta - 0.1)
            else:  # Low variance -> more concentrated sampling
                self.beta_alpha = min(3.0, self.beta_alpha + 0.1)
                self.beta_beta = min(3.0, self.beta_beta + 0.1)

            # Update stats
            self.stats["beta_alpha"] = self.beta_alpha
            self.stats["beta_beta"] = self.beta_beta

    def _beta_sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample timesteps using Beta distribution (research paper methodology)."""
        try:
            # Create Beta distribution
            alpha = torch.tensor(self.beta_alpha, device=device)
            beta = torch.tensor(self.beta_beta, device=device)
            beta_dist = torch.distributions.Beta(alpha, beta)

            # Sample from Beta and convert to timestep range
            beta_samples = beta_dist.sample((batch_size,))

            # Map from [0,1] to boundary range [min_t, max_t] normalized
            timesteps = self.min_t_normalized + beta_samples * (
                self.max_t_normalized - self.min_t_normalized
            )

            return timesteps

        except Exception as e:
            logger.warning(f"Beta sampling failed, falling back to uniform: {e}")
            # Fallback to uniform sampling within boundaries
            return (
                torch.rand(batch_size, device=device)
                * (self.max_t_normalized - self.min_t_normalized)
                + self.min_t_normalized
            )

    def _update_feature_selection(self):
        """Update selected features every f_S steps (paper methodology)."""
        if self.feature_selection_step % self.sampler_update_frequency == 0:
            # Select top |S| most important timesteps as features
            if len(self.important_timesteps) >= self.feature_selection_size:
                self.selected_features = self.important_timesteps[
                    : self.feature_selection_size
                ]
            else:
                # Pad with random timesteps if not enough important ones
                remaining = self.feature_selection_size - len(self.important_timesteps)
                random_timesteps = torch.randint(
                    self.min_timestep, self.max_timestep + 1, (remaining,)
                ).tolist()
                self.selected_features = self.important_timesteps + random_timesteps

            logger.debug(f"Updated feature selection: {self.selected_features}")

        self.feature_selection_step += 1

    def get_beta_sampling_weights(
        self, timesteps: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get sampling weights using Beta distribution approach."""
        if not self.use_beta_sampler or not self.enabled:
            return None

        try:
            # Update feature selection periodically
            self._update_feature_selection()

            # Create feature vector from selected timesteps
            if self.selected_features:
                feature_values = []
                for timestep in self.selected_features:
                    importance = self.timestep_importance.get(timestep, 1.0)
                    feature_values.append(importance)

                features = torch.tensor(feature_values, device=timesteps.device)

                # Update Beta parameters based on features
                self._update_beta_parameters(features)

            # Sample new timesteps using Beta distribution
            beta_timesteps = self._beta_sample_timesteps(
                len(timesteps), timesteps.device
            )

            # Compute importance weights based on proximity to Beta samples
            weights = torch.ones_like(timesteps, dtype=torch.float32)

            for i, t in enumerate(timesteps):
                t_norm = t.float() / 1000.0  # Normalize to [0,1]
                # Find closest Beta sample
                distances = torch.abs(beta_timesteps - t_norm)
                min_dist = torch.min(distances)

                # Apply exponential weighting based on distance
                weight = torch.exp(-min_dist * 10.0)  # Closer samples get higher weight
                weights[i] = 1.0 + weight * (self.focus_strength - 1.0)

            return weights

        except Exception as e:
            logger.warning(f"Beta sampling weights failed: {e}")
            return None

    def _initialize_complementary_components(self):
        """Initialize complementary approach components."""
        try:
            # Initialize replay buffer for KL learning
            if self.use_replay_buffer and self.use_kl_reward_learning:
                self.replay_buffer = AdaptiveReplayBuffer(
                    capacity=self.replay_buffer_size,
                    timestep_range=(self.min_timestep, self.max_timestep),
                )
                logger.info(
                    f"📊 Replay buffer initialized with capacity {self.replay_buffer_size}"
                )

            # Initialize RL optimizer for policy updates
            if self.use_kl_reward_learning and self.neural_sampler is not None:
                self.rl_optimizer = optim.Adam(
                    self.neural_sampler.parameters(), lr=self.rl_learning_rate
                )
                logger.info(
                    f"🎯 RL optimizer initialized with lr={self.rl_learning_rate}"
                )

            # Initialize statistical feature selector
            if (
                self.use_statistical_features
                and SKLEARN_AVAILABLE
                and f_regression is not None
            ):
                self.statistical_selector = SelectKBest(
                    score_func=f_regression, k=self.feature_selection_size
                )  # type: ignore
                logger.info("📈 Statistical feature selector initialized")
            elif self.use_statistical_features and not SKLEARN_AVAILABLE:
                logger.warning(
                    "Statistical features requested but sklearn not available"
                )
                self.use_statistical_features = False

        except Exception as e:
            logger.warning(f"Failed to initialize complementary components: {e}")

    def compute_kl_divergence(self, model, x_0, x_t, t):
        """Compute KL divergence between true and predicted distributions."""
        try:
            # This requires access to the diffusion model components
            # For now, use MSE loss as a proxy for KL divergence
            with torch.no_grad():
                # Predict noise
                predicted_noise = model(x_t, t)

                # Compute MSE loss per sample
                mse_per_sample = F.mse_loss(predicted_noise, x_0, reduction="none")
                mse_per_sample = mse_per_sample.view(mse_per_sample.shape[0], -1).mean(
                    dim=1
                )

                # Convert MSE to KL-like metric (proxy)
                kl_proxy = torch.log(1.0 + mse_per_sample)

                return kl_proxy

        except Exception as e:
            logger.warning(f"KL divergence computation failed: {e}")
            return torch.zeros(x_0.shape[0], device=x_0.device)

    def record_kl_improvement(self, timesteps, kl_before, kl_after):
        """Record KL divergence improvements for replay buffer."""
        if not self.use_replay_buffer or self.replay_buffer is None:
            return

        try:
            kl_diff = kl_before - kl_after
            kl_improvement = kl_diff.sum().item()

            # Store in replay buffer
            self.replay_buffer.add_experience(
                timesteps=timesteps.cpu().numpy(),
                kl_improvement=kl_improvement,
                kl_diffs=kl_diff.cpu().numpy(),
            )

            # Update KL history
            if self.kl_history is not None:
                self.kl_history.append(
                    {
                        "timesteps": timesteps.clone(),
                        "kl_improvement": kl_improvement,
                        "step": self.step_count,
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to record KL improvement: {e}")

    def update_via_kl_reward(self, timesteps, rewards):
        """Update policy using KL divergence rewards (RL approach)."""
        if not self.use_kl_reward_learning or self.neural_sampler is None:
            return

        if self.step_count - self.last_kl_update_step < self.kl_update_frequency:
            return

        try:
            # Sample from Beta distribution to get log probabilities
            if len(self.selected_features) >= self.feature_selection_size:
                features = torch.tensor(
                    [
                        self.timestep_importance.get(ts, 1.0)
                        for ts in self.selected_features[: self.feature_selection_size]
                    ],
                    device=timesteps.device,
                )

                # Get current alpha, beta parameters
                alpha, beta = self.neural_sampler(features.unsqueeze(0))

                # Create Beta distribution
                beta_dist = torch.distributions.Beta(alpha, beta)

                # Convert timesteps to [0,1] range for Beta distribution
                timesteps_norm = (timesteps.float() - self.min_timestep) / (
                    self.max_timestep - self.min_timestep
                )

                # Compute log probabilities
                log_probs = beta_dist.log_prob(timesteps_norm)

                # Compute entropy for regularization
                entropy = beta_dist.entropy()

                # Compute policy gradient loss
                policy_loss = -(log_probs * rewards).mean()
                entropy_bonus = -self.entropy_coefficient * entropy

                total_loss = policy_loss + entropy_bonus

                # Update policy
                self.rl_optimizer.zero_grad()  # type: ignore
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.neural_sampler.parameters(), max_norm=1.0
                )
                self.rl_optimizer.step()  # type: ignore

                self.last_kl_update_step = self.step_count

                logger.debug(
                    f"RL update: policy_loss={policy_loss:.6f}, entropy={entropy:.6f}"
                )

        except Exception as e:
            logger.warning(f"KL reward update failed: {e}")

    def get_statistical_features(self):
        """Get statistically selected important timesteps."""
        if not self.use_statistical_features or self.replay_buffer is None:
            return []

        if not hasattr(self.replay_buffer, "get_feature_data"):
            return []

        try:
            X, y = self.replay_buffer.get_feature_data()
            if X.shape[0] < 10:  # Need sufficient data
                return []

            # Use SelectKBest to find important features
            self.statistical_selector.fit(X, y)  # type: ignore
            selected_indices = self.statistical_selector.get_support(indices=True)  # type: ignore

            # Convert indices back to timesteps
            timestep_range = np.arange(self.min_timestep, self.max_timestep + 1)
            selected_timesteps = timestep_range[selected_indices]

            return selected_timesteps.tolist()

        except Exception as e:
            logger.warning(f"Statistical feature selection failed: {e}")
            return []

    def record_training_step(
        self, model, x_0, x_t, timesteps, loss_before=None, loss_after=None
    ):
        """Record a training step for KL learning and replay buffer updates."""
        if not self.enabled:
            return

        try:
            # Compute KL divergence for RL learning
            if self.use_kl_reward_learning and self.replay_buffer is not None:
                if loss_before is not None and loss_after is not None:
                    # Use provided losses
                    kl_improvement = (loss_before - loss_after).mean()
                else:
                    # Compute KL proxy from model
                    kl_before = self.compute_kl_divergence(model, x_0, x_t, timesteps)
                    # For after, we'd need the updated model - skip for now
                    kl_improvement = kl_before.mean()

                # Record in replay buffer
                self.record_kl_improvement(
                    timesteps,
                    kl_before if "kl_before" in locals() else loss_before,
                    kl_improvement.unsqueeze(0) if loss_after is None else loss_after,
                )

                # Update via RL if it's time
                if kl_improvement > 0:  # Only reward actual improvements
                    self.update_via_kl_reward(
                        timesteps, kl_improvement.expand(len(timesteps))
                    )

            # Update statistical features
            if self.use_statistical_features:
                statistical_features = self.get_statistical_features()
                if statistical_features:
                    # Merge with existing important timesteps
                    combined_important = list(
                        set(self.important_timesteps + statistical_features)
                    )
                    combined_important.sort()
                    self.important_timesteps = combined_important[
                        : self.max_important_timesteps
                    ]

                    logger.debug(
                        f"Statistical features added: {statistical_features[:5]}..."
                    )

        except Exception as e:
            logger.warning(f"Failed to record training step: {e}")

    def get_combined_sampling_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get combined weights from multiple approaches."""
        weights_list = []
        method_names = []

        # Get importance weighting approach
        if self.use_importance_weighting:
            importance_weights = self._get_importance_weights_only(timesteps)
            if importance_weights is not None:
                weights_list.append(importance_weights)
                method_names.append("importance")

        # Get KL/RL approach weights
        if self.use_kl_reward_learning:
            kl_weights = self.get_beta_sampling_weights(timesteps)
            if kl_weights is not None:
                weights_list.append(kl_weights)
                method_names.append("kl_rl")

        # Combine weights based on strategy
        if len(weights_list) == 0:
            return torch.ones_like(timesteps)
        elif len(weights_list) == 1:
            return weights_list[0]
        else:
            return self._combine_weight_methods(weights_list, method_names, timesteps)

    def _get_importance_weights_only(
        self, timesteps: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get weights using only importance weighting (our stable approach)."""
        if not self.timestep_weights:
            return torch.ones_like(timesteps)

        timestep_ints = (timesteps * 1000).long()
        weights = torch.ones_like(timesteps)

        for i, timestep_int in enumerate(timestep_ints):
            timestep_key = int(timestep_int.item())

            # Only apply weights within boundaries
            if not (self.min_timestep <= timestep_key <= self.max_timestep):
                continue

            if timestep_key in self.timestep_weights:
                adaptive_weight = self.timestep_weights[timestep_key]

                # Apply video-specific weighting if enabled
                if self.video_specific_categories:
                    if timestep_key in self.motion_timesteps:
                        adaptive_weight *= self.motion_weight
                    elif timestep_key in self.detail_timesteps:
                        adaptive_weight *= self.detail_weight
                    elif timestep_key in self.temporal_timesteps:
                        adaptive_weight *= self.temporal_weight

                weights[i] = adaptive_weight

        return weights

    def _combine_weight_methods(
        self,
        weights_list: List[torch.Tensor],
        method_names: List[str],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Combine multiple weighting approaches."""
        if self.weight_combination == "fallback":
            # Use first successful method (preference order)
            return weights_list[0]

        elif self.weight_combination == "ensemble":
            # Average all methods
            combined = torch.stack(weights_list).mean(dim=0)
            return combined

        elif self.weight_combination == "best":
            # Choose method with highest variance (most focused)
            best_weights = weights_list[0]
            best_variance = torch.var(weights_list[0])

            for weights in weights_list[1:]:
                variance = torch.var(weights)
                if variance > best_variance:
                    best_weights = weights
                    best_variance = variance

            return best_weights

        else:
            # Default to ensemble
            combined = torch.stack(weights_list).mean(dim=0)
            return combined


class AdaptiveReplayBuffer:
    """Replay buffer for storing KL divergence improvements and feature data."""

    def __init__(
        self, capacity: int = 100, timestep_range: Tuple[int, int] = (0, 1000)
    ):
        self.capacity = capacity
        self.min_timestep, self.max_timestep = timestep_range
        self.buffer = deque(maxlen=capacity)
        self.timestep_count = self.max_timestep - self.min_timestep + 1

    def add_experience(
        self, timesteps: np.ndarray, kl_improvement: float, kl_diffs: np.ndarray
    ):
        """Add experience to replay buffer."""
        experience = {
            "timesteps": timesteps.copy(),
            "kl_improvement": kl_improvement,
            "kl_diffs": kl_diffs.copy(),
            "timestamp": len(self.buffer),
        }
        self.buffer.append(experience)

    def get_feature_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get feature matrix X and target vector y for statistical learning."""
        if len(self.buffer) == 0:
            return np.array([]), np.array([])

        # Create feature matrix: each row represents timestep activations for one experience
        X = np.zeros((len(self.buffer), self.timestep_count))
        y = np.zeros(len(self.buffer))

        for i, exp in enumerate(self.buffer):
            # Create one-hot encoding of active timesteps
            for timestep in exp["timesteps"]:
                if self.min_timestep <= timestep <= self.max_timestep:
                    idx = timestep - self.min_timestep
                    X[i, idx] = 1.0

            y[i] = exp["kl_improvement"]

        return X, y

    def size(self) -> int:
        return len(self.buffer)


class BetaSamplerNetwork(nn.Module):
    """Neural network for predicting Beta distribution parameters."""

    def __init__(self, input_size: int = 3, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # Output: [alpha, beta]
            nn.Softplus(),  # Ensure positive parameters
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict Alpha and Beta parameters."""
        if features.dim() == 1:
            features = features.unsqueeze(0)

        params = self.net(features)
        alpha = params[:, 0] + 0.1  # Minimum value to avoid degenerate distributions
        beta = params[:, 1] + 0.1

        return alpha.squeeze(), beta.squeeze()


def create_adaptive_timestep_manager(
    args: argparse.Namespace,
) -> AdaptiveTimestepManager:
    """Factory function to create an AdaptiveTimestepManager from command line arguments.

    Args:
        args: Command line arguments containing adaptive timestep sampling configuration

    Returns:
        Configured AdaptiveTimestepManager instance with boundary respect
    """
    return AdaptiveTimestepManager(
        enabled=getattr(args, "enable_adaptive_timestep_sampling", False),
        analysis_window=getattr(args, "adaptive_analysis_window", 1000),
        importance_detection_threshold=getattr(
            args, "adaptive_importance_threshold", 1.5
        ),
        focus_strength=getattr(args, "adaptive_focus_strength", 2.0),
        min_important_timesteps=getattr(args, "adaptive_min_timesteps", 50),
        max_important_timesteps=getattr(args, "adaptive_max_timesteps", 200),
        warmup_steps=getattr(args, "adaptive_warmup_steps", 500),
        update_frequency=getattr(args, "adaptive_update_frequency", 100),
        video_specific_categories=getattr(args, "adaptive_video_specific", True),
        motion_weight=getattr(args, "adaptive_motion_weight", 1.0),
        detail_weight=getattr(args, "adaptive_detail_weight", 1.0),
        temporal_weight=getattr(args, "adaptive_temporal_weight", 1.0),
        min_timestep=getattr(args, "min_timestep", None),  # Respect existing boundaries
        max_timestep=getattr(args, "max_timestep", None),  # Respect existing boundaries
        # Research paper alignment parameters
        use_beta_sampler=getattr(args, "adaptive_use_beta_sampler", False),
        feature_selection_size=getattr(args, "adaptive_feature_selection_size", 3),
        sampler_update_frequency=getattr(args, "adaptive_sampler_update_frequency", 40),
        use_neural_sampler=getattr(args, "adaptive_use_neural_sampler", False),
        beta_alpha_init=getattr(args, "adaptive_beta_alpha_init", 1.0),
        beta_beta_init=getattr(args, "adaptive_beta_beta_init", 1.0),
        neural_hidden_size=getattr(args, "adaptive_neural_hidden_size", 64),
        # Complementary approach parameters
        use_importance_weighting=getattr(
            args, "adaptive_use_importance_weighting", True
        ),
        use_kl_reward_learning=getattr(args, "adaptive_use_kl_reward_learning", False),
        use_replay_buffer=getattr(args, "adaptive_use_replay_buffer", False),
        use_statistical_features=getattr(
            args, "adaptive_use_statistical_features", False
        ),
        weight_combination=getattr(args, "adaptive_weight_combination", "fallback"),
        replay_buffer_size=getattr(args, "adaptive_replay_buffer_size", 100),
        rl_learning_rate=getattr(args, "adaptive_rl_learning_rate", 1e-4),
        entropy_coefficient=getattr(args, "adaptive_entropy_coefficient", 0.01),
        kl_update_frequency=getattr(args, "adaptive_kl_update_frequency", 20),
    )
