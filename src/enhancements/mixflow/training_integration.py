"""MixFlow training integration helpers.

This module implements slowed interpolation mixture for flow-matching training.
The feature is training-only and fully config-gated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from utils.train_utils import get_sigmas


logger = get_logger(__name__)


def _resolve_timestep_bounds(
    noise_scheduler: Any, timesteps: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    schedule_timesteps = getattr(noise_scheduler, "timesteps", None)
    if schedule_timesteps is None:
        return None
    schedule_timesteps = schedule_timesteps.to(
        device=timesteps.device, dtype=torch.float32
    )
    t_min = torch.min(schedule_timesteps)
    t_max = torch.max(schedule_timesteps)
    span = t_max - t_min
    if float(span.abs().item()) < 1e-12:
        return None
    return schedule_timesteps, t_min, t_max


def maybe_apply_mixflow_beta_t_sampling(
    *,
    args: Any,
    noise_scheduler: Any,
    timesteps: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """Optionally resample timesteps using the MixFlow Beta(2,1)-style rule."""
    if not bool(getattr(args, "enable_mixflow", False)):
        return timesteps, None
    if not bool(getattr(args, "mixflow_beta_t_sampling", False)):
        return timesteps, None
    if noise_scheduler is None:
        logger.warning("MixFlow beta timestep sampling enabled but scheduler is unavailable; skipping.")
        return timesteps, None

    bounds = _resolve_timestep_bounds(noise_scheduler, timesteps)
    if bounds is None:
        logger.warning("MixFlow beta timestep sampling enabled but scheduler timestep range is invalid; skipping.")
        return timesteps, None
    _, t_min, t_max = bounds
    span = t_max - t_min

    t_norm_original = ((timesteps.to(torch.float32) - t_min) / span).clamp(0.0, 1.0)
    # Beta(2, 1) style from reference code: t = 1 - sqrt(u), u ~ Uniform(0, 1).
    u = torch.rand_like(t_norm_original, dtype=torch.float32)
    t_norm_beta = 1.0 - torch.sqrt(u.clamp(0.0, 1.0))
    t_norm_final = t_norm_beta

    # Optional parity path from MixFlow appendix code:
    # t <- shift * t / (1 + (shift - 1) * t)
    use_shift = bool(getattr(args, "mixflow_time_dist_shift_enabled", False))
    shift = float(getattr(args, "mixflow_time_dist_shift", 1.0))
    if use_shift and abs(shift - 1.0) > 1e-12:
        denom = 1.0 + (shift - 1.0) * t_norm_beta
        denom = torch.clamp(denom, min=1e-8)
        t_norm_final = (shift * t_norm_beta) / denom
        t_norm_final = t_norm_final.clamp(0.0, 1.0)

    beta_timesteps = (t_norm_final * span + t_min).to(
        device=timesteps.device, dtype=timesteps.dtype
    )
    stats = {
        "mixflow/t_original_mean": float(t_norm_original.detach().mean().item()),
        "mixflow/t_beta_mean": float(t_norm_beta.detach().mean().item()),
        "mixflow/t_final_mean": float(t_norm_final.detach().mean().item()),
    }
    return beta_timesteps, stats


def apply_mixflow_slowed_interpolation(
    *,
    args: Any,
    noise_scheduler: Any,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """Build MixFlow slowed interpolation inputs for model conditioning.

    Returns:
        slowed_noisy_input:
            Interpolated noisy latents sampled at slowed timesteps.
        stats:
            Optional scalar stats for logging/debugging.
    """
    if not bool(getattr(args, "enable_mixflow", False)):
        return noisy_model_input, None
    if noise_scheduler is None:
        logger.warning("MixFlow enabled but noise_scheduler is unavailable; skipping.")
        return noisy_model_input, None

    gamma = float(getattr(args, "mixflow_gamma", 0.4))
    if gamma <= 0.0:
        return noisy_model_input, None

    bounds = _resolve_timestep_bounds(noise_scheduler, timesteps)
    if bounds is None:
        logger.warning("MixFlow enabled but scheduler has no timesteps; skipping.")
        return noisy_model_input, None
    _, t_min, t_max = bounds
    span = t_max - t_min

    t_norm = ((timesteps.to(torch.float32) - t_min) / span).clamp(0.0, 1.0)
    rand_u = torch.rand_like(t_norm, dtype=torch.float32)
    mt_norm = t_norm + rand_u * gamma * (1.0 - t_norm)
    slowed_timesteps = (mt_norm * span + t_min).to(device=timesteps.device)

    layout = "per_frame" if timesteps.dim() > 1 else "per_sample"
    sigmas = get_sigmas(
        noise_scheduler,
        slowed_timesteps.to(dtype=timesteps.dtype),
        device=noise.device,
        n_dim=noise.dim(),
        dtype=noise.dtype,
        timestep_layout=layout,
        source="mixflow/slowed_interpolation",
    )

    slowed_noisy_input = sigmas * noise + (1.0 - sigmas) * latents.to(
        device=noise.device, dtype=noise.dtype
    )
    stats = {
        "mixflow/t_mean": float(t_norm.detach().mean().item()),
        "mixflow/mt_mean": float(mt_norm.detach().mean().item()),
        "mixflow/mean_delta": float((mt_norm - t_norm).detach().mean().item()),
    }
    return slowed_noisy_input, stats
