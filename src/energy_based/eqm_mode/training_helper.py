from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from easydict import EasyDict

from .integration import (
    EqMModeContext,
    EqMStepResult,
    compute_eqm_step,
    setup_eqm_mode,
)
from .scheduler import EqMSchedulerAdapter
from .weighting import compute_weight_scale


@dataclass
class EqMForwardResult:
    """Prepared outputs from an EqM forward pass."""

    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    noisy_latents: torch.Tensor
    intermediate: Optional[torch.Tensor]
    loss_components: EasyDict
    metrics: Dict[str, float]


class EqMTrainingHelper:
    """Encapsulates EqM training state (context, scheduler adapter, weighting)."""

    def __init__(self, context: EqMModeContext, adapter: EqMSchedulerAdapter):
        self._context = context
        self._adapter = adapter

    @property
    def context(self) -> EqMModeContext:
        return self._context

    @classmethod
    def maybe_create(
        cls, args: argparse.Namespace, noise_scheduler: Any
    ) -> Optional["EqMTrainingHelper"]:
        """Instantiate the helper if EqM mode is enabled; otherwise return None."""
        if not getattr(args, "enable_eqm_mode", False):
            return None
        context = setup_eqm_mode(args)
        adapter = EqMSchedulerAdapter(noise_scheduler)
        return cls(context, adapter)

    def _apply_weighting(
        self, loss_components: EasyDict, *, global_step: int
    ) -> None:
        """Apply optional EqM loss weighting schedule in-place."""
        weight = compute_weight_scale(
            schedule=self._context.config.weighting_schedule,
            total_steps=self._context.config.weighting_steps,
            current_step=global_step,
        )
        if weight is None:
            return

        loss_tensor = loss_components["loss"]
        weight_tensor = torch.tensor(
            weight, device=loss_tensor.device, dtype=loss_tensor.dtype
        )

        scaled_loss = loss_tensor * weight_tensor
        loss_components["loss"] = scaled_loss
        loss_components.loss = scaled_loss  # type: ignore[attr-defined]

        if "total_loss" in loss_components:
            total_scaled = loss_components["total_loss"] * weight_tensor
            loss_components["total_loss"] = total_scaled
            loss_components.total_loss = total_scaled  # type: ignore[attr-defined]

        loss_components["eqm_weight_scale"] = weight_tensor.detach()

    def prepare_forward(
        self,
        *,
        transformer: Any,
        latents: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        args: argparse.Namespace,
        accelerator: Accelerator,
        network_dtype: torch.dtype,
        patch_size: Any,
        global_step: int,
    ) -> EqMForwardResult:
        """Run the EqM forward pass and return prepared tensors/metrics."""
        eqm_step: EqMStepResult = compute_eqm_step(
            self._context,
            transformer=transformer,
            latents=latents,
            batch=batch,
            args=args,
            accelerator=accelerator,
            network_dtype=network_dtype,
            patch_size=patch_size,
        )

        mapped_timesteps = self._adapter.map_timesteps(
            eqm_step.timesteps,
            device=accelerator.device,
            dtype=network_dtype,
        )

        loss_components = eqm_step.loss_components
        if eqm_step.energy is not None:
            loss_components["eqm_energy_mean"] = eqm_step.energy.mean().detach()

        self._apply_weighting(loss_components, global_step=global_step)

        metrics: Dict[str, float] = {}
        if "eqm_weight_scale" in loss_components:
            metrics["train/eqm_weight_scale"] = float(
                loss_components["eqm_weight_scale"].item()
            )
        if "eqm_energy_mean" in loss_components:
            metrics["train/eqm_energy_mean"] = float(
                loss_components["eqm_energy_mean"].item()
            )

        return EqMForwardResult(
            model_pred=eqm_step.model_pred,
            target=eqm_step.target,
            timesteps=mapped_timesteps,
            noise=eqm_step.noise,
            noisy_latents=eqm_step.noisy_latents,
            intermediate=eqm_step.intermediate,
            loss_components=loss_components,
            metrics=metrics,
        )

    def log_startup(self) -> Dict[str, str]:
        """Return metadata announcing EqM mode for logging/telemetry."""
        return {
            "prediction": self._context.config.prediction,
            "path_type": self._context.config.path_type,
        }
