import argparse
import torch
from typing import Tuple, Any


def get_noisy_model_input_and_timesteps_fvdm(
    args: argparse.Namespace,
    noise: torch.Tensor,
    latents: torch.Tensor,
    noise_scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate FVDM/PUSA vectorized noisy inputs and timesteps for a Flow Matching trainer.

    This function implements the Rectified Flow noising process:
    x_t = (1-t) * x_0 + t * x_1
    where x_0 is the clean latent and x_1 is the noise.
    The model's objective will be to predict the velocity v = x_1 - x_0.
    """
    if latents.ndim != 5:
        raise ValueError(
            f"FVDM requires 5D latents (B, C, F, H, W), but got {latents.ndim}D"
        )

    B, C, F, H, W = latents.shape
    T = noise_scheduler.num_train_timesteps

    # 1. Probabilistic Timestep Sampling Strategy (PTSS)
    if torch.rand((), device=device) < args.fvdm_ptss_p:
        # Asynchronous: Sample a different continuous time t for each frame
        t_cont = torch.rand((B, F), device=device, dtype=dtype)
    else:
        # Synchronous: Sample one continuous time t for the whole clip
        t_cont = torch.rand((B, 1), device=device, dtype=dtype).expand(B, F)

    # 2. Apply min/max timestep constraints
    # These are normalized to the [0, 1] range of t_cont
    t_min = getattr(args, "min_timestep", 0.0) / T
    t_max = getattr(args, "max_timestep", T) / T
    t_cont = t_cont * (t_max - t_min) + t_min

    # 3. Add noise using the Flow Matching equation
    # For Rectified Flow, the noise level `sigma` is equivalent to the time `t`.
    sigma_cont = t_cont

    # Reshape sigma for broadcasting: (B, F) -> (B, 1, F, 1, 1)
    sigma_broadcast = sigma_cont.view(B, 1, F, 1, 1)

    noisy_model_input = (1.0 - sigma_broadcast) * latents + sigma_broadcast * noise

    # 4. Prepare discrete timesteps for the model's embedding layer
    # WanModel expects timesteps in the integer range [0, 999]
    timesteps_discrete = (sigma_cont * (T - 1)).round().long().clamp(0, T - 1)

    return noisy_model_input, timesteps_discrete, sigma_cont
