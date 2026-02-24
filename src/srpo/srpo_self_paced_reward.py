"""
Self-paced reward mixing utilities for SRPO.

Implements a lightweight, train-time-only approximation of the competence-aware
co-evolving reward mechanism from Self-Paced GRPO (arXiv:2511.19356v1).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


class SelfPacedRewardMixer:
    """Mix visual/temporal/semantic rewards with competence-aware soft weights."""

    def __init__(
        self,
        *,
        visual_threshold: float = 0.75,
        temporal_threshold: float = 0.75,
        semantic_threshold: float = 0.75,
        softmax_beta: float = 5.0,
        sparsity_lambda: float = 0.5,
        sigmoid_scale: float = 8.0,
        enable_advantage_norm: bool = True,
        advantage_norm_eps: float = 1e-6,
        auto_calibrate_thresholds: bool = False,
        threshold_calibration_factor: float = 0.7,
        threshold_calibration_warmup_steps: int = 50,
        threshold_calibration_momentum: float = 0.9,
    ) -> None:
        self.thresholds = [
            float(visual_threshold),
            float(temporal_threshold),
            float(semantic_threshold),
        ]
        self.softmax_beta = float(softmax_beta)
        self.sparsity_lambda = float(sparsity_lambda)
        self.sigmoid_scale = float(sigmoid_scale)
        self.enable_advantage_norm = bool(enable_advantage_norm)
        self.advantage_norm_eps = float(advantage_norm_eps)
        self.auto_calibrate_thresholds = bool(auto_calibrate_thresholds)
        self.threshold_calibration_factor = float(threshold_calibration_factor)
        self.threshold_calibration_warmup_steps = int(threshold_calibration_warmup_steps)
        self.threshold_calibration_momentum = float(threshold_calibration_momentum)
        self._calibration_steps = 0
        self._baseline_means: torch.Tensor | None = None
        self._delta_ema: torch.Tensor | None = None

    def _hoyer_sparsity(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute Hoyer sparsity in [0, 1] for a 1D reward vector."""
        flattened = scores.reshape(-1).to(dtype=torch.float32)
        n = flattened.numel()
        if n <= 1:
            return flattened.new_tensor(0.0)
        l1 = flattened.abs().sum()
        l2 = flattened.pow(2).sum().sqrt().clamp_min(self.advantage_norm_eps)
        sqrt_n = math.sqrt(float(n))
        sparsity = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0)
        return sparsity.clamp(0.0, 1.0)

    def mix(
        self,
        *,
        visual_rewards: torch.Tensor,
        temporal_rewards: torch.Tensor,
        semantic_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return mixed reward signal and diagnostics."""
        components = torch.stack(
            [
                visual_rewards.to(dtype=torch.float32),
                temporal_rewards.to(dtype=torch.float32),
                semantic_rewards.to(dtype=torch.float32),
            ],
            dim=1,
        )  # [B, 3]

        means = components.mean(dim=0)
        sparsities = torch.stack(
            [
                self._hoyer_sparsity(components[:, 0]),
                self._hoyer_sparsity(components[:, 1]),
                self._hoyer_sparsity(components[:, 2]),
            ]
        )

        logits = []
        for idx in range(3):
            competence = torch.sigmoid(
                self.sigmoid_scale * (means[idx] - float(self.thresholds[idx]))
            )
            sparsity_term = means.new_tensor(0.0)
            if idx > 0:
                sparsity_term = self.sparsity_lambda * (
                    sparsities[idx - 1] - sparsities[idx]
                )
            logits.append(self.softmax_beta * (competence + sparsity_term))

        weight_logits = torch.stack(logits)
        weights = torch.softmax(weight_logits, dim=0)  # [3]
        mixed_rewards = (components * weights.unsqueeze(0)).sum(dim=1)  # [B]

        normalized_applied = 0.0
        mixed_signal = mixed_rewards
        mixed_mean = mixed_rewards.mean()
        mixed_std = mixed_rewards.std(unbiased=False)
        if (
            self.enable_advantage_norm
            and mixed_rewards.numel() > 1
            and float(mixed_std.item()) > self.advantage_norm_eps
        ):
            mixed_signal = (mixed_rewards - mixed_mean) / mixed_std.clamp_min(
                self.advantage_norm_eps
            )
            normalized_applied = 1.0

        metrics = {
            "self_paced_weight_visual": float(weights[0].item()),
            "self_paced_weight_temporal": float(weights[1].item()),
            "self_paced_weight_semantic": float(weights[2].item()),
            "self_paced_logit_visual": float(weight_logits[0].item()),
            "self_paced_logit_temporal": float(weight_logits[1].item()),
            "self_paced_logit_semantic": float(weight_logits[2].item()),
            "self_paced_mean_visual": float(means[0].item()),
            "self_paced_mean_temporal": float(means[1].item()),
            "self_paced_mean_semantic": float(means[2].item()),
            "self_paced_sparsity_visual": float(sparsities[0].item()),
            "self_paced_sparsity_temporal": float(sparsities[1].item()),
            "self_paced_sparsity_semantic": float(sparsities[2].item()),
            "self_paced_mixed_reward_mean": float(mixed_mean.item()),
            "self_paced_mixed_reward_std": float(mixed_std.item()),
            "self_paced_advantage_norm_applied": normalized_applied,
            "self_paced_threshold_visual": float(self.thresholds[0]),
            "self_paced_threshold_temporal": float(self.thresholds[1]),
            "self_paced_threshold_semantic": float(self.thresholds[2]),
        }
        return mixed_signal, metrics

    def update_thresholds(self, *, means: torch.Tensor) -> Dict[str, float]:
        """
        Optionally calibrate thresholds from observed reward improvements.

        Uses an online approximation of the paper's threshold calibration idea:
        threshold = baseline + factor * EMA(max(mean - baseline, 0)).
        """
        if not self.auto_calibrate_thresholds:
            return {
                "self_paced_threshold_update_applied": 0.0,
                "self_paced_threshold_calibration_step": float(self._calibration_steps),
            }

        means = means.detach().to(dtype=torch.float32).reshape(3)
        if self._baseline_means is None:
            self._baseline_means = means.clone()
            self._delta_ema = torch.zeros_like(means)

        assert self._delta_ema is not None
        delta = torch.relu(means - self._baseline_means)
        momentum = self.threshold_calibration_momentum
        self._delta_ema = momentum * self._delta_ema + (1.0 - momentum) * delta
        self._calibration_steps += 1

        applied = 0.0
        if self._calibration_steps >= self.threshold_calibration_warmup_steps:
            calibrated = self._baseline_means + (
                self.threshold_calibration_factor * self._delta_ema
            )
            self.thresholds = [
                float(calibrated[0].item()),
                float(calibrated[1].item()),
                float(calibrated[2].item()),
            ]
            applied = 1.0

        return {
            "self_paced_threshold_update_applied": applied,
            "self_paced_threshold_calibration_step": float(self._calibration_steps),
            "self_paced_threshold_visual": float(self.thresholds[0]),
            "self_paced_threshold_temporal": float(self.thresholds[1]),
            "self_paced_threshold_semantic": float(self.thresholds[2]),
        }


def compute_grpo_clipped_objective(
    *,
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO/GRPO-style clipped objective value per sample."""
    log_ratio = (current_log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    metrics = {
        "self_paced_grpo_ratio_mean": float(ratio.mean().item()),
        "self_paced_grpo_ratio_std": float(ratio.std(unbiased=False).item()),
        "self_paced_grpo_clip_eps": float(clip_eps),
    }
    return objective, metrics
