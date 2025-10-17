from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from .integration import EqMModeContext, predict_velocity
from .sampling import AdaptiveStepController
from .sampling_eqm import sample_eqm_latents


@dataclass
class EqMSamplingResult:
    latent: torch.Tensor
    likelihood: Optional[Dict[str, torch.Tensor]]


def run_eqm_sampling(
    *,
    eqm_context: EqMModeContext,
    accelerator: Accelerator,
    transformer: Any,
    initial_latent: torch.Tensor,
    context_tokens: torch.Tensor,
    negative_context: Optional[torch.Tensor],
    sample_parameter: Dict[str, Any],
    sample_steps: int,
    cfg_scale: float,
    device: torch.device,
    network_dtype: torch.dtype,
    patch_size: Tuple[int, int, int],
    args: argparse.Namespace,
) -> EqMSamplingResult:
    """Execute EqM sampling (GD/NGD or integrator-based) and return latent state."""

    base_step_size = float(
        sample_parameter.get(
            "eqm_step_size", getattr(args, "eqm_step_size", 1.0 / max(sample_steps, 1))
        )
    )
    momentum = float(
        sample_parameter.get("eqm_momentum", getattr(args, "eqm_momentum", 0.0))
    )
    sampler_type = str(
        sample_parameter.get("eqm_sampler", getattr(args, "eqm_sampler", "gd"))
    ).lower()
    use_adaptive = bool(
        sample_parameter.get(
            "eqm_use_adaptive_sampler", getattr(args, "eqm_use_adaptive_sampler", False)
        )
    )

    adaptive_controller: Optional[AdaptiveStepController] = None
    if use_adaptive:
        adaptive_controller = AdaptiveStepController(
            initial_step=base_step_size,
            min_step=float(
                sample_parameter.get(
                    "eqm_adaptive_step_min",
                    getattr(args, "eqm_adaptive_step_min", base_step_size),
                )
            ),
            max_step=float(
                sample_parameter.get(
                    "eqm_adaptive_step_max",
                    getattr(args, "eqm_adaptive_step_max", base_step_size),
                )
            ),
            growth=float(
                sample_parameter.get(
                    "eqm_adaptive_growth", getattr(args, "eqm_adaptive_growth", 1.05)
                )
            ),
            shrink=float(
                sample_parameter.get(
                    "eqm_adaptive_shrink", getattr(args, "eqm_adaptive_shrink", 0.5)
                )
            ),
            patience=int(
                sample_parameter.get(
                    "eqm_adaptive_restart_patience",
                    getattr(args, "eqm_adaptive_restart_patience", 4),
                )
            ),
            alignment_threshold=float(
                sample_parameter.get(
                    "eqm_adaptive_alignment_threshold",
                    getattr(args, "eqm_adaptive_alignment_threshold", 0.0),
                )
            ),
        )
        current_step_size = adaptive_controller.value
    else:
        current_step_size = base_step_size

    if sampler_type in {"ode", "sde", "ode_likelihood"}:
        latent_state, likelihood = sample_eqm_latents(
            eqm_context=eqm_context,
            accelerator=accelerator,
            transformer=transformer,
            initial_latent=initial_latent,
            context_tokens=context_tokens,
            negative_context=negative_context,
            sample_parameter=sample_parameter,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            device=device,
            network_dtype=network_dtype,
            patch_size=patch_size,
            args=args,
        )
        return EqMSamplingResult(latent=latent_state, likelihood=likelihood)

    latent = initial_latent.to(device=device, dtype=network_dtype)
    prev_update = torch.zeros_like(latent)
    t_value = torch.zeros((), device=device, dtype=network_dtype)

    cond_tokens = [context_tokens] if context_tokens is not None else []
    if not cond_tokens:
        raise ValueError("EqM sampling requires conditioning tokens.")
    null_tokens = [negative_context] if negative_context is not None else []

    with torch.no_grad():
        for _ in range(sample_steps):
            t_tensor = t_value.clone().unsqueeze(0)

            eval_latent = latent
            if sampler_type == "ngd" and momentum != 0.0:
                eval_latent = latent + current_step_size * momentum * prev_update

            preds_cond = predict_velocity(
                eqm_context,
                transformer=transformer,
                latents=eval_latent.unsqueeze(0),
                t=t_tensor,
                context_tokens=cond_tokens,
                accelerator=accelerator,
                network_dtype=network_dtype,
                patch_size=patch_size,
            )[0]

            if null_tokens:
                preds_uncond = predict_velocity(
                    eqm_context,
                    transformer=transformer,
                    latents=eval_latent.unsqueeze(0),
                    t=t_tensor,
                    context_tokens=null_tokens,
                    accelerator=accelerator,
                    network_dtype=network_dtype,
                    patch_size=patch_size,
                )[0]
                preds = preds_uncond + cfg_scale * (preds_cond - preds_uncond)
            else:
                preds = preds_cond

            latent = latent + current_step_size * preds
            t_value = t_value + current_step_size

            if adaptive_controller is not None:
                reset_momentum = adaptive_controller.update(prev_update, preds)
                current_step_size = adaptive_controller.value
                if reset_momentum:
                    prev_update = torch.zeros_like(prev_update)
                    continue

            prev_update = preds

    return EqMSamplingResult(latent=latent, likelihood=None)
