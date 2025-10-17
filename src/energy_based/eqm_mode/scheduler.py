from __future__ import annotations

from typing import Optional, Any

import torch


class EqMSchedulerAdapter:
    """Maps EqM continuous timesteps onto the discrete scheduler domain."""

    def __init__(self, noise_scheduler: Any):
        self._scheduler = noise_scheduler
        timesteps = getattr(noise_scheduler, "timesteps", None)
        if timesteps is None:
            raise AttributeError("Noise scheduler must expose `timesteps`.")
        self._timesteps = timesteps.to(torch.float32)
        self._max_index = self._timesteps.numel() - 1

    def map_timesteps(
        self,
        t_continuous: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert [0,1] EqM timesteps into discrete scheduler timesteps."""
        if device is None:
            device = t_continuous.device
        if dtype is None:
            dtype = t_continuous.dtype

        indices = torch.clamp(t_continuous, 0.0, 1.0) * self._max_index
        indices = torch.round(indices).long()
        mapped = self._timesteps[indices]
        return mapped.to(device=device, dtype=dtype)
