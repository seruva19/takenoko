import argparse
import logging
import torch
from typing import Tuple, Any, Optional
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)
# One-time emission flag for PTSS parameter logging
_ptss_params_logged_once = False


def get_adaptive_ptss_p(
    current_step: int,
    total_steps: int,
    initial_p: float = 0.3,
    final_p: float = 0.1,
    warmup_steps: int = 1000,
) -> float:
    """
    Adaptive PTSS probability that decreases during training.
    Early training: Higher async probability for exploration
    Late training: Lower async probability for stability

    Args:
        current_step: Current training step
        total_steps: Total training steps
        initial_p: Starting probability (higher for exploration)
        final_p: Ending probability (lower for stability)
        warmup_steps: Steps before adaptation begins

    Returns:
        Adaptive PTSS probability
    """
    if current_step < warmup_steps:
        return initial_p

    progress = min(
        1.0, (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    )
    return initial_p + (final_p - initial_p) * progress


def get_noisy_model_input_and_timesteps_fvdm(
    args: argparse.Namespace,
    noise: torch.Tensor,
    latents: torch.Tensor,
    noise_scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    current_step: int = 0,
    adaptive_manager: Optional[Any] = None,
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
    T = noise_scheduler.config.num_train_timesteps

    # 1. Enhanced Probabilistic Timestep Sampling Strategy (PTSS)
    # Determine PTSS probability (adaptive or fixed)
    if getattr(args, "fvdm_adaptive_ptss", False):
        ptss_p = get_adaptive_ptss_p(
            current_step,
            getattr(args, "max_train_steps", 100000),
            getattr(args, "fvdm_ptss_initial", 0.3),
            getattr(args, "fvdm_ptss_final", 0.1),
            getattr(args, "fvdm_ptss_warmup", 1000),
        )
    else:
        ptss_p = getattr(args, "fvdm_ptss_p", 0.2)  # Default to paper-recommended value

    # Apply PTSS sampling
    is_async = torch.rand((), device=device) < ptss_p

    # Sample uniform values first
    if is_async:
        # Asynchronous: Sample a different uniform value for each frame
        t_uniform = torch.rand((B, F), device=device, dtype=dtype)
    else:
        # Synchronous: Sample one uniform value for the whole clip
        t_uniform = torch.rand((B, 1), device=device, dtype=dtype).expand(B, F)

    # 2. Apply timestep_sampling distribution (LogSNR, uniform, etc.)
    # Import here to avoid circular dependency
    from scheduling.timestep_utils import map_uniform_to_sampling

    # Transform uniform samples through the configured distribution
    # This gives us t in [0, 1] range following the configured distribution
    t_cont = map_uniform_to_sampling(args, t_uniform, latents)

    # 3. Apply min/max timestep constraints
    # Scale the [0,1] distribution to [min_timestep, max_timestep] range
    # This preserves the shape of the distribution (LogSNR, etc.)
    original_min_timestep = int(getattr(args, "min_timestep", 0))
    original_max_timestep = int(getattr(args, "max_timestep", T))
    t_min = original_min_timestep / T
    t_max = original_max_timestep / T

    # Scale from [0,1] to [t_min, t_max] while preserving distribution shape
    t_cont = t_cont * (t_max - t_min) + t_min

    # 2.5. Optional: Integrate with AdaptiveTimestepManager for importance-based sampling
    resample_count_applied = 0
    adaptive_integration_enabled = (
        adaptive_manager is not None
        and getattr(args, "fvdm_integrate_adaptive_timesteps", False)
        and hasattr(adaptive_manager, "enabled")
        and adaptive_manager.enabled
    )

    if adaptive_integration_enabled:
        try:
            # Convert to discrete timesteps for importance checking
            timesteps_discrete_check = (t_cont * (T - 1)).round().long().clamp(0, T - 1)

            # Get importance weights
            importance_weights = adaptive_manager.get_adaptive_sampling_weights(
                timesteps_discrete_check
            )

            # Occasionally bias toward important timesteps (30% of the time)
            if torch.rand((), device=device) < 0.3:
                # Find frames with low importance
                low_importance_mask = importance_weights < 1.2
                if low_importance_mask.any():
                    # Resample some low-importance frames to potentially hit important timesteps
                    resample_count = min(
                        low_importance_mask.sum().item(), max(1, F // 4)
                    )
                    if resample_count > 0:
                        resample_indices = torch.nonzero(
                            low_importance_mask, as_tuple=True
                        )[0][:resample_count]
                        # Resample these frames
                        new_t_values = (
                            torch.rand(resample_count, device=device, dtype=dtype)
                            * (t_max - t_min)
                            + t_min
                        )
                        t_cont.view(-1)[resample_indices] = new_t_values
                        resample_count_applied = int(resample_count)
        except Exception:
            # Graceful fallback - continue with original t_cont if integration fails
            pass

    # Explicit one-time INFO log of PTSS scheduling parameters
    global _ptss_params_logged_once
    if not _ptss_params_logged_once:
        try:
            ptss_mode = (
                "adaptive" if getattr(args, "fvdm_adaptive_ptss", False) else "fixed"
            )
            sampling_mode = "async" if is_async else "sync"
            timestep_sampling = getattr(args, "timestep_sampling", "uniform")
            logger.info(
                (
                    "⏰ FVDM PTSS applied | mode=%s | ptss_p=%.4f | sampling=%s | "
                    "timestep_dist=%s | t_min_norm=%.4f | t_max_norm=%.4f | "
                    "t_min_idx=%d | t_max_idx=%d | T=%d | step=%d | "
                    "adaptive_integration=%s | resample_count=%d"
                ),
                ptss_mode,
                float(ptss_p),
                sampling_mode,
                timestep_sampling,
                float(t_min),
                float(t_max),
                int(original_min_timestep),
                int(original_max_timestep),
                int(T),
                int(current_step),
                bool(adaptive_integration_enabled),
                int(resample_count_applied),
            )
            _ptss_params_logged_once = True
        except Exception:
            # Logging must not break training if any attribute is missing
            _ptss_params_logged_once = True
            pass

    # 3. Add noise using the Flow Matching equation
    # For Rectified Flow, the noise level `sigma` is equivalent to the time `t`.
    sigma_cont = t_cont

    # Reshape sigma for broadcasting: (B, F) -> (B, 1, F, 1, 1)
    sigma_broadcast = sigma_cont.view(B, 1, F, 1, 1)

    noisy_model_input = (1.0 - sigma_broadcast) * latents + sigma_broadcast * noise

    # 4. Prepare discrete timesteps for the model's embedding layer
    # WanModel expects timesteps in the integer range [1, 1000] to match standard implementation
    timesteps_discrete = (sigma_cont * T).round().long().clamp(1, T)

    return noisy_model_input, timesteps_discrete, sigma_cont
