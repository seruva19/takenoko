"""LoRA interval modulation utilities."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class IntervalModulationConfig:
    enabled: bool
    delta_attention: bool
    mlp_hidden: int


class LoRAIntervalAdapter:
    """Applies Δt-aware modulation to LoRA networks."""

    def __init__(self, config: IntervalModulationConfig) -> None:
        self.config = config

    @contextmanager
    def maybe_modulate(self, network, delta_t: Tensor):
        if not self.config.enabled or network is None:
            yield
            return

        base_multiplier: Optional[float] = None
        applied_rank_mask = False

        try:
            if hasattr(network, "multiplier"):
                base_multiplier = float(network.multiplier)
                network.multiplier = base_multiplier * self._compute_scale(delta_t)

            if (
                self.config.delta_attention
                and hasattr(network, "update_rank_mask_from_timesteps")
            ):
                # Use delta as proxy for timesteps (scaled to [0, 1000])
                pseudo_timesteps = (delta_t.clamp(0.0, 1.0) * 1000.0).to(
                    device=delta_t.device, dtype=torch.float32
                )
                try:
                    network.update_rank_mask_from_timesteps(
                        pseudo_timesteps,
                        max_timestep=1000,
                    )
                    applied_rank_mask = True
                except Exception:
                    applied_rank_mask = False
            yield
        finally:
            if base_multiplier is not None and hasattr(network, "multiplier"):
                network.multiplier = base_multiplier
            if applied_rank_mask and hasattr(network, "clear_rank_mask"):
                try:
                    network.clear_rank_mask()
                except Exception:
                    pass

    def _compute_scale(self, delta_t: Tensor) -> float:
        if delta_t.numel() == 0:
            return 1.0
        delta_mean = float(delta_t.mean().detach().cpu().item())
        hidden = max(self.config.mlp_hidden, 1)
        # Deterministic feature-based modulation
        features = []
        for idx in range(hidden):
            freq = (idx + 1) / hidden
            features.append(math.sin(math.pi * freq * delta_mean))
            features.append(math.cos(math.pi * freq * delta_mean))
        modulation = sum(features) / max(len(features), 1)
        scale = 1.0 + 0.25 * modulation
        return max(0.1, min(2.0, scale))
