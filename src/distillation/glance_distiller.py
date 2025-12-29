"""Glance-style distillation helpers for WAN training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from common.logger import get_logger
from utils.train_utils import compute_loss_weighting_for_sd3, get_sigmas

logger = get_logger(__name__)


DEFAULT_SLOW_TIMESTEPS: List[float] = [
    1000.0,
    979.1915,
    957.5157,
    934.9171,
    911.3354,
]
DEFAULT_FAST_TIMESTEPS: List[float] = [
    886.7053,
    745.0728,
    562.9505,
    320.0802,
    20.0,
]


@dataclass
class GlanceConfig:
    enabled: bool
    mode: str
    timesteps: List[float]


class GlanceDistiller:
    """Provide Glance-style timestep sampling and noisy input generation."""

    def __init__(
        self,
        config: GlanceConfig,
        *,
        min_timestep: float = 0.0,
        max_timestep: float = 1000.0,
    ) -> None:
        self.config = config
        self.min_timestep = float(min_timestep)
        self.max_timestep = float(max_timestep)
        self._validate_timesteps(self.config.timesteps)

    @classmethod
    def from_args(cls, args) -> "GlanceDistiller":
        enabled = bool(getattr(args, "glance_enabled", False))
        mode = str(getattr(args, "glance_mode", "slow")).lower()
        custom = getattr(args, "glance_timesteps", None)
        min_timestep = float(getattr(args, "min_timestep", 0) or 0)
        max_timestep = float(getattr(args, "max_timestep", 1000) or 1000)

        if mode not in {"slow", "fast", "custom"}:
            raise ValueError(
                f"glance_mode must be one of slow|fast|custom, got '{mode}'"
            )

        if mode == "custom":
            if not custom:
                raise ValueError(
                    "glance_mode=custom requires glance_timesteps to be set"
                )
            timesteps = list(custom)
        elif mode == "fast":
            timesteps = list(DEFAULT_FAST_TIMESTEPS)
        else:
            timesteps = list(DEFAULT_SLOW_TIMESTEPS)

        config = GlanceConfig(enabled=enabled, mode=mode, timesteps=timesteps)
        return cls(
            config,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _validate_timesteps(self, timesteps: Iterable[float]) -> None:
        steps = [float(t) for t in timesteps]
        if not steps:
            raise ValueError("Glance timesteps cannot be empty.")
        for t in steps:
            if t < self.min_timestep or t > self.max_timestep:
                raise ValueError(
                    "Glance timesteps must be within "
                    f"[{self.min_timestep}, {self.max_timestep}], got {t}."
                )

    def _sample_timesteps(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        steps = torch.tensor(
            self.config.timesteps, device=device, dtype=dtype
        )
        indices = torch.randint(
            0,
            steps.numel(),
            (batch_size,),
            device=device,
        )
        return steps[indices]

    def prepare_training_inputs(
        self,
        *,
        args,
        accelerator,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dtype: torch.dtype,
        batch,
        cdc_gamma_b=None,
        item_info=None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size = noise.shape[0]
        timesteps = self._sample_timesteps(
            batch_size=batch_size,
            device=accelerator.device,
            dtype=dtype,
        )
        sigmas = get_sigmas(
            noise_scheduler,
            timesteps,
            accelerator.device,
            n_dim=latents.ndim,
            dtype=dtype,
            source="glance",
        )
        if getattr(args, "enable_cdc_fm", False) and cdc_gamma_b is not None:
            try:
                from enhancements.cdc.cdc_fm import apply_cdc_noise_transformation

                num_ts = float(
                    getattr(
                        getattr(noise_scheduler, "config", object()),
                        "num_train_timesteps",
                        1000,
                    )
                )
                noise = apply_cdc_noise_transformation(
                    noise,
                    timesteps.float() / max(num_ts, 1.0),
                    cdc_gamma_b,
                    item_info,
                )
            except Exception as exc:
                logger.warning("CDC-FM noise transform failed: %s", exc)

        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme,
            noise_scheduler,
            timesteps,
            accelerator.device,
            dtype,
        )
        return noisy_model_input, timesteps, sigmas, weighting
