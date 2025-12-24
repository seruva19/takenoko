"""Helpers for preparing standard training inputs."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from scheduling.timestep_utils import get_noisy_model_input_and_timesteps
from utils.train_utils import compute_loss_weighting_for_sd3


def prepare_standard_training_inputs(
    args,
    accelerator,
    latents: Tensor,
    noise: Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    timestep_distribution,
    dual_model_manager,
    batch: Dict[str, Tensor],
    cdc_gamma_b=None,
    item_info=None,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """
    Prepare noisy latents / timesteps / sigmas / weighting using the legacy logic
    (dual-model manager, optional pre-sampled timesteps, standard weighting).
    """
    if dual_model_manager is not None:
        noisy_model_input, timesteps, sigmas = (
            dual_model_manager.determine_and_prepare_batch(
                args=args,
                noise=noise,
                latents=latents,
                noise_scheduler=noise_scheduler,
                device=accelerator.device,
                dtype=dit_dtype,
                timestep_distribution=timestep_distribution,
                presampled_uniform=(
                    None
                    if (
                        hasattr(args, "use_precomputed_timesteps")
                        and getattr(args, "use_precomputed_timesteps", False)
                    )
                    else batch.get("timesteps", None)
                ),
            )
        )
        try:
            dual_model_manager.swap_if_needed(accelerator)
        except Exception as err:
            # Preserve previous behaviour: warn but continue
            from common.logger import get_logger

            get_logger(__name__).warning(f"DualModelManager swap failed: {err}")

        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme,
            noise_scheduler,
            timesteps,
            accelerator.device,
            dit_dtype,
        )
        return noisy_model_input, timesteps, sigmas, weighting

    # If dataset provided per-batch pre-sampled uniform t values and
    # precomputed timesteps are NOT enabled, map them through the
    # selected sampling strategy. Otherwise, ignore and use current path.
    batch_timesteps_uniform = None
    try:
        if hasattr(args, "use_precomputed_timesteps") and getattr(
            args, "use_precomputed_timesteps", False
        ):
            batch_timesteps_uniform = None
        else:
            bt = batch.get("timesteps", None)
            if bt is not None:
                batch_timesteps_uniform = torch.tensor(
                    bt,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
    except Exception:
        batch_timesteps_uniform = None

    noisy_model_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        args,
        noise,
        latents,
        noise_scheduler,
        accelerator.device,
        dit_dtype,
        timestep_distribution,
        batch_timesteps_uniform,
        cdc_gamma_b=cdc_gamma_b,
        item_info=item_info,
    )
    weighting = compute_loss_weighting_for_sd3(
        args.weighting_scheme,
        noise_scheduler,
        timesteps,
        accelerator.device,
        dit_dtype,
    )
    return noisy_model_input, timesteps, sigmas, weighting
