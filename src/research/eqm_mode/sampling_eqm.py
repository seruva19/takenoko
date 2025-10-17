from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from .integration import EqMModeContext, predict_velocity
from ..eqm_transport.transport import Sampler as EqMTransportSampler


def _prepare_tensors(values: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = values.to(device=device, dtype=dtype)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    return values


def _guided_velocity(
    *,
    eqm_context: EqMModeContext,
    accelerator: Accelerator,
    transformer: Any,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    context_tokens: list[torch.Tensor],
    negative_tokens: list[torch.Tensor],
    cfg_scale: float,
    network_dtype: torch.dtype,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    preds_cond = predict_velocity(
        eqm_context,
        transformer=transformer,
        latents=latents,
        t=timesteps,
        context_tokens=context_tokens,
        accelerator=accelerator,
        network_dtype=network_dtype,
        patch_size=patch_size,
    )

    if negative_tokens:
        preds_uncond = predict_velocity(
            eqm_context,
            transformer=transformer,
            latents=latents,
            t=timesteps,
            context_tokens=negative_tokens,
            accelerator=accelerator,
            network_dtype=network_dtype,
            patch_size=patch_size,
        )
        return preds_uncond + cfg_scale * (preds_cond - preds_uncond)
    return preds_cond


def sample_eqm_latents(
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
    args: Any,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    cond_tokens = [context_tokens] if context_tokens is not None else []
    if not cond_tokens:
        raise ValueError("EqM sampling requires conditioning tokens.")
    null_tokens = [negative_context] if negative_context is not None else []

    def model_fn(latents: torch.Tensor, timestep: torch.Tensor, **_unused) -> torch.Tensor:
        t_tensor = _prepare_tensors(values=timestep, device=device, dtype=network_dtype)
        return _guided_velocity(
            eqm_context=eqm_context,
            accelerator=accelerator,
            transformer=transformer,
            latents=latents,
            timesteps=t_tensor,
            context_tokens=cond_tokens,
            negative_tokens=null_tokens,
            cfg_scale=cfg_scale,
            network_dtype=network_dtype,
            patch_size=patch_size,
        )

    sampler = EqMTransportSampler(eqm_context.transport)
    state = initial_latent.to(device=device, dtype=network_dtype).unsqueeze(0)

    sampler_type = str(
        sample_parameter.get("eqm_sampler", getattr(args, "eqm_sampler", "gd"))
    ).lower()

    if sampler_type == "ode":
        ode_method = str(
            sample_parameter.get("eqm_ode_method", getattr(args, "eqm_ode_method", "dopri5"))
        ).lower()
        ode_steps = int(
            sample_parameter.get(
                "eqm_ode_steps",
                getattr(args, "eqm_ode_steps", sample_steps if sample_steps > 0 else 50),
            )
        )
        ode_atol = float(
            sample_parameter.get("eqm_ode_atol", getattr(args, "eqm_ode_atol", 1e-6))
        )
        ode_rtol = float(
            sample_parameter.get("eqm_ode_rtol", getattr(args, "eqm_ode_rtol", 1e-3))
        )
        ode_reverse = bool(
            sample_parameter.get("eqm_ode_reverse", getattr(args, "eqm_ode_reverse", False))
        )
        ode_sampler = sampler.sample_ode(
            sampling_method=ode_method,
            num_steps=max(ode_steps, 2),
            atol=ode_atol,
            rtol=ode_rtol,
            reverse=ode_reverse,
        )
        trajectory = ode_sampler(state, model_fn)
        return trajectory[-1].squeeze(0), None

    if sampler_type == "ode_likelihood":
        ode_method = str(
            sample_parameter.get("eqm_ode_method", getattr(args, "eqm_ode_method", "dopri5"))
        ).lower()
        ode_steps = int(
            sample_parameter.get(
                "eqm_ode_steps",
                getattr(args, "eqm_ode_steps", sample_steps if sample_steps > 0 else 50),
            )
        )
        ode_atol = float(
            sample_parameter.get(
                "eqm_ode_likelihood_atol",
                getattr(args, "eqm_ode_likelihood_atol", getattr(args, "eqm_ode_atol", 1e-6)),
            )
        )
        ode_rtol = float(
            sample_parameter.get(
                "eqm_ode_likelihood_rtol",
                getattr(args, "eqm_ode_likelihood_rtol", getattr(args, "eqm_ode_rtol", 1e-3)),
            )
        )
        trace_samples = int(
            sample_parameter.get(
                "eqm_ode_likelihood_trace_samples",
                getattr(args, "eqm_ode_likelihood_trace_samples", 1),
            )
        )
        likelihood_sampler = sampler.sample_ode_likelihood(
            sampling_method=ode_method,
            num_steps=max(ode_steps, 2),
            atol=ode_atol,
            rtol=ode_rtol,
            trace_samples=max(trace_samples, 1),
        )
        result = likelihood_sampler(state, model_fn)
        likelihood_metrics = {
            "logp": result.logp.detach(),
            "prior_logp": result.prior_logp.detach(),
            "delta_logp": result.delta_logp[-1].detach(),
            "delta_path": [entry.detach() for entry in result.delta_logp],
        }
        return result.final_state.squeeze(0), likelihood_metrics

    sde_method = str(
        sample_parameter.get("eqm_sde_method", getattr(args, "eqm_sde_method", "Euler"))
    )
    sde_steps = int(
        sample_parameter.get(
            "eqm_sde_steps",
            getattr(args, "eqm_sde_steps", sample_steps if sample_steps > 0 else 250),
        )
    )
    sde_last_step = sample_parameter.get(
        "eqm_sde_last_step", getattr(args, "eqm_sde_last_step", "Mean")
    )
    sde_last_step_size = float(
        sample_parameter.get(
            "eqm_sde_last_step_size",
            getattr(args, "eqm_sde_last_step_size", 0.04),
        )
    )
    sde_diff_form = sample_parameter.get(
        "eqm_sde_diffusion_form", getattr(args, "eqm_sde_diffusion_form", "SBDM")
    )
    sde_diff_norm = float(
        sample_parameter.get(
            "eqm_sde_diffusion_norm",
            getattr(args, "eqm_sde_diffusion_norm", 1.0),
        )
    )
    sde_sampler = sampler.sample_sde(
        sampling_method=sde_method,
        diffusion_form=sde_diff_form,
        diffusion_norm=sde_diff_norm,
        last_step=sde_last_step,
        last_step_size=sde_last_step_size,
        num_steps=max(sde_steps, 2),
    )
    trajectory = sde_sampler(state, model_fn)
    return trajectory[-1].squeeze(0), None
