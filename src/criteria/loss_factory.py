"""Pseudo-Huber Loss Implementation for Takenoko

This module provides a gated, optional, non-intrusive integration of Pseudo-Huber loss
into the Takenoko training pipeline.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Any

from common.logger import get_logger

logger = get_logger(__name__)


def pseudo_huber_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    huber_c: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Pseudo-Huber loss.

    The Pseudo-Huber loss is defined as:
    L(x, y) = c * (sqrt((x - y)^2 + c^2) - c)

    This is a smooth approximation to the Huber loss that is differentiable everywhere.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        huber_c: Huber parameter c (controls the transition point)
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Computed loss tensor
    """
    # Ensure tensors are in float32 for numerical stability
    model_pred = model_pred.float()
    target = target.float()

    # Compute squared difference
    diff_sq = (model_pred - target) ** 2

    # Pseudo-Huber formula: c * (sqrt(diff^2 + c^2) - c)
    loss = huber_c * (torch.sqrt(diff_sq + huber_c**2) - huber_c)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def scheduled_pseudo_huber_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    huber_c: float = 0.5,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    schedule_type: str = "linear",
    c_min: float = 0.1,
    c_max: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Scheduled Pseudo-Huber loss with adaptive c parameter.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        huber_c: Base Huber parameter c (used if scheduling is disabled)
        current_step: Current training step (for scheduling)
        total_steps: Total training steps (for scheduling)
        schedule_type: Scheduling type ("linear", "cosine", "exponential")
        c_min: Minimum c value for scheduling
        c_max: Maximum c value for scheduling
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Computed loss tensor
    """
    # Use scheduled c if step information is provided
    if current_step is not None and total_steps is not None and total_steps > 0:
        progress = min(current_step / total_steps, 1.0)

        if schedule_type == "linear":
            # Linear decay from c_max to c_min
            scheduled_c = c_max - progress * (c_max - c_min)
        elif schedule_type == "cosine":
            # Cosine decay from c_max to c_min
            scheduled_c = c_min + 0.5 * (c_max - c_min) * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )
            scheduled_c = float(scheduled_c)
        elif schedule_type == "exponential":
            # Exponential decay from c_max to c_min
            decay_rate = -torch.log(torch.tensor(c_min / c_max))
            scheduled_c = c_max * torch.exp(-decay_rate * progress)
            scheduled_c = float(scheduled_c)
        else:
            logger.warning(
                f"Unknown schedule type '{schedule_type}', using base huber_c"
            )
            scheduled_c = huber_c
    else:
        scheduled_c = huber_c

    return pseudo_huber_loss(model_pred, target, scheduled_c, reduction)


def stepped_loss(
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    step_size: int = 50,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute stepped loss with forward stepping and image reconstruction.

    This loss function performs a forward step in the diffusion process and then
    reconstructs the original image, computing loss on the reconstructed result.
    This approach aims to reduce errors at high-noise timesteps and create
    smoother flow dynamics.

    Args:
        model_pred: Model prediction tensor (predicted noise)
        latents: Clean latent tensor (target for reconstruction)
        noise: Original noise tensor
        noisy_latents: Noisy latent representation at current timestep
        timesteps: Current timesteps
        noise_scheduler: Diffusion noise scheduler with sigmas
        step_size: Number of timestep indices to step forward (default: 50)
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Computed stepped loss tensor
    """
    # Ensure inputs are in float32 for numerical stability
    model_pred = model_pred.float()
    noise = noise.float()
    noisy_latents = noisy_latents.float()

    # Get batch size
    batch_size = model_pred.shape[0]

    # Check if scheduler has sigmas attribute
    if not hasattr(noise_scheduler, 'sigmas'):
        logger.warning("Stepped loss requires noise scheduler with 'sigmas' attribute. Falling back to MSE loss.")
        # For flow matching: target = noise - latents, so use noise as prediction target for fallback
        return F.mse_loss(model_pred, noise, reduction=reduction)

    # Split tensors by batch dimension for individual processing
    noise_pred_chunks = torch.chunk(model_pred, batch_size)
    timestep_chunks = torch.chunk(timesteps, batch_size)
    noisy_latent_chunks = torch.chunk(noisy_latents, batch_size)
    noise_chunks = torch.chunk(noise, batch_size)

    x0_pred_chunks = []

    for idx in range(batch_size):
        model_output = noise_pred_chunks[idx]  # predicted noise (same shape as latent)
        timestep = timestep_chunks[idx]  # scalar tensor per sample (e.g., [t])
        sample = noisy_latent_chunks[idx].to(torch.float32)
        noise_i = noise_chunks[idx].to(sample.dtype).to(sample.device)

        # Initialize scheduler step index for this sample
        noise_scheduler._step_index = None
        noise_scheduler._init_step_index(timestep)

        # ---- Step +50 indices (or to the end) in sigma-space ----
        sigma = noise_scheduler.sigmas[noise_scheduler.step_index]
        target_idx = min(noise_scheduler.step_index + step_size, len(noise_scheduler.sigmas) - 1)
        sigma_next = noise_scheduler.sigmas[target_idx]

        # One-step update along the model-predicted direction
        stepped = sample + (sigma_next - sigma) * model_output

        # ---- Inverse-Gaussian recovery at the target timestep ----
        t_01 = (
            (noise_scheduler.sigmas[target_idx]).to(stepped.device).to(stepped.dtype)
        )
        original_samples = (stepped - t_01 * noise_i) / (1.0 - t_01)
        x0_pred_chunks.append(original_samples)

    # Reconstruct predicted images and target images
    predicted_images = torch.cat(x0_pred_chunks, dim=0)

    # Compare predicted images with original clean latents (as in original patch)
    loss = F.mse_loss(
        predicted_images.float(),
        latents.float().to(device=predicted_images.device),
        reduction="none",
    )

    return loss


def conditional_loss_with_pseudo_huber(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_c: float = 0.5,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    schedule_type: str = "linear",
    c_min: float = 0.1,
    c_max: float = 1.0,
    reduction: str = "mean",
    # Additional parameters for advanced loss types
    timesteps: Optional[torch.Tensor] = None,
    alphas_cumprod: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    # Stepped loss specific parameters
    noisy_latents: Optional[torch.Tensor] = None,
    clean_latents: Optional[torch.Tensor] = None,
    noise_scheduler: Optional[Any] = None,
    # For future extensibility
    batch: Optional[dict] = None,
    **kwargs,
) -> torch.Tensor:
    """Conditional loss function supporting multiple loss types.

    This function provides a unified interface for different loss functions including
    standard losses (MSE, L1, Huber), Pseudo-Huber variants, and advanced loss types
    (Fourier, DWT, Clustered MSE, EW, Stepped).

    Advanced loss types automatically fall back to simpler alternatives if required
    parameters are missing or imports fail.

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        loss_type: Loss type. Supported options:
            - Standard: "mse", "l1"/"mae", "huber", "pure_huber"
            - Pseudo-Huber: "pseudo_huber", "pseudo_huber_scheduled"
            - Advanced: "fourier", "dwt"/"wavelet", "clustered_mse", "ew", "stepped"
            Advanced types fall back to simpler alternatives if requirements not met.
        huber_c: Huber parameter c
        current_step: Current training step (for scheduled variants)
        total_steps: Total training steps (for scheduled variants)
        schedule_type: Scheduling type for scheduled variants
        c_min: Minimum c value for scheduling
        c_max: Maximum c value for scheduling
        reduction: Reduction method
        timesteps: Timestep tensor (required for advanced loss types)
        alphas_cumprod: Alpha cumulative product tensor (for EW loss)
        noise: Noise tensor (required for stepped loss)
        noisy_latents: Noisy latent tensor (required for stepped loss)
        clean_latents: Clean latent tensor (required for stepped loss)
        noise_scheduler: Noise scheduler with sigmas (required for stepped loss)

    Returns:
        Computed loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        return F.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        return F.smooth_l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "pure_huber":
        # Use huber_delta from kwargs if available, otherwise fall back to huber_c
        delta = kwargs.get("huber_delta", huber_c)
        return F.huber_loss(
            model_pred.float(), target.float(), reduction=reduction, delta=delta
        )
    elif loss_type == "pseudo_huber":
        return pseudo_huber_loss(model_pred, target, huber_c, reduction)
    elif loss_type == "pseudo_huber_scheduled":
        return scheduled_pseudo_huber_loss(
            model_pred,
            target,
            huber_c,
            current_step,
            total_steps,
            schedule_type,
            c_min,
            c_max,
            reduction,
        )
    elif loss_type == "fourier":
        # Import here to avoid circular imports
        try:
            from criteria.fourier_loss import (
                fourier_latent_loss_basic,
                fourier_latent_loss_weighted,
                fourier_latent_loss_multiscale,
                fourier_latent_loss_adaptive,
            )
        except ImportError as e:
            logger.warning(
                f"Fourier loss requires fourier_loss module: {e}. Falling back to MSE loss."
            )
            return F.mse_loss(model_pred, target, reduction=reduction)

        # Get fourier parameters from kwargs
        fourier_mode = kwargs.get("fourier_mode", "weighted")
        fourier_dims = kwargs.get("fourier_dims", (-2, -1))
        fourier_eps = kwargs.get("fourier_eps", 1e-8)
        fourier_norm = kwargs.get("fourier_norm", "l2")

        # Note: Fourier loss functions don't support reduction parameter, so we apply it ourselves
        if fourier_mode == "basic":
            loss = fourier_latent_loss_basic(
                model_pred,
                target,
                norm_type=fourier_norm,
                dims=fourier_dims,
                eps=fourier_eps,
            )
        elif fourier_mode == "weighted":
            high_freq_weight = kwargs.get("fourier_high_freq_weight", 2.0)
            loss = fourier_latent_loss_weighted(
                model_pred,
                target,
                high_freq_weight=high_freq_weight,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
            )
        elif fourier_mode == "multiscale":
            factors = kwargs.get("fourier_multiscale_factors", [1, 2, 4])
            loss = fourier_latent_loss_multiscale(
                model_pred,
                target,
                scales=factors,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
            )
        elif fourier_mode == "adaptive":
            # Get current_step and total_steps for adaptive mode
            current_step = kwargs.get("current_step", 0)
            total_steps = kwargs.get("total_steps", 1)
            max_weight = kwargs.get("fourier_adaptive_alpha", 2.0)
            min_weight = kwargs.get("fourier_adaptive_threshold", 0.5)
            loss = fourier_latent_loss_adaptive(
                model_pred,
                target,
                current_step=current_step,
                total_steps=total_steps,
                max_weight=max_weight,
                min_weight=min_weight,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
            )
        else:
            high_freq_weight = kwargs.get("fourier_high_freq_weight", 2.0)
            loss = fourier_latent_loss_weighted(
                model_pred,
                target,
                high_freq_weight=high_freq_weight,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
            )

        # Apply reduction manually since fourier functions don't support it
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # reduction == "none"
            return loss
    elif loss_type == "dwt" or loss_type == "wavelet":
        # Import here to avoid circular imports
        try:
            from criteria.dwt_loss import dwt_loss

            return dwt_loss(target, model_pred, reduction=reduction)
        except ImportError as e:
            # Fall back to MSE if pytorch-wavelets is not installed
            logger.warning(
                f"DWT/Wavelet loss requires pytorch-wavelets package: {e}. Falling back to MSE loss."
            )
            return F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "clustered_mse":
        # Import here to avoid circular imports
        from criteria.clustered_mse_loss import adaptive_clustered_mse_loss

        if timesteps is not None:
            try:
                # Get parameters from kwargs with safe bounds
                base_clusters = kwargs.get("clustered_mse_num_clusters", 8)
                min_clusters = max(
                    4, base_clusters // 2
                )  # At least 4, default function minimum
                max_clusters = max(
                    min_clusters + 1, base_clusters * 10
                )  # Ensure max > min

                # Create a simple loss_map if not provided (dictionary mapping timesteps to loss factors)
                # Use varying loss factors to avoid division by zero in clustered MSE loss
                unique_timesteps = torch.unique(timesteps)
                loss_map = {}
                if len(unique_timesteps) == 1:
                    # Special case: only one timestep, create minimal variation
                    loss_map[unique_timesteps[0].item()] = 1.0
                    loss_map[unique_timesteps[0].item() + 0.001] = (
                        1.1  # Add dummy entry
                    )
                else:
                    for i, t in enumerate(unique_timesteps):
                        # Create variation from 0.9 to 1.1 across timesteps
                        factor = 0.9 + 0.2 * (i / (len(unique_timesteps) - 1))
                        loss_map[t.item()] = factor

                return adaptive_clustered_mse_loss(
                    model_pred,
                    target,
                    timesteps.to(model_pred.device),
                    loss_map,
                    reduction=reduction,
                    min_clusters=min_clusters,
                    max_clusters=max_clusters,
                )
            except Exception as e:
                logger.warning(
                    f"Clustered MSE loss failed: {e}. Falling back to MSE loss."
                )
                return F.mse_loss(model_pred, target, reduction=reduction)
        else:
            logger.warning(
                "Clustered MSE loss requires timesteps parameter. Falling back to MSE loss."
            )
            return F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "ew":
        # Import here to avoid circular imports
        from criteria.ew_loss import exponential_weighted_loss

        if timesteps is not None:
            # Try to get or create alphas_cumprod if not provided
            if alphas_cumprod is None:
                # Create a simple alphas_cumprod schedule if not available
                # This is a fallback - real training should provide this
                device = model_pred.device
                # Assume 1000 timesteps is a reasonable default for most diffusion models
                max_timesteps = (
                    max(1000, timesteps.max().item() + 1)
                    if timesteps.numel() > 0
                    else 1000
                )
                t_range = torch.arange(
                    max_timesteps, device=device, dtype=torch.float32
                )
                t_normalized = t_range / max_timesteps
                alphas_cumprod = (
                    torch.cos((t_normalized + 0.008) / 1.008 * torch.pi / 2) ** 2
                )

        if timesteps is not None:
            try:
                # Create a simple loss_map if not provided (dictionary mapping timesteps to loss factors)
                # Use varying loss factors to avoid division by zero in EW loss
                unique_timesteps = torch.unique(timesteps)
                loss_map = {}
                if len(unique_timesteps) == 1:
                    # Special case: only one timestep, create minimal variation
                    loss_map[unique_timesteps[0].item()] = 1.0
                    loss_map[unique_timesteps[0].item() + 0.001] = (
                        1.1  # Add dummy entry
                    )
                else:
                    for i, t in enumerate(unique_timesteps):
                        # Create variation from 0.9 to 1.1 across timesteps
                        factor = 0.9 + 0.2 * (i / (len(unique_timesteps) - 1))
                        loss_map[t.item()] = factor

                # Get boundary shift parameter
                boundary_shift = kwargs.get("ew_boundary_shift", 0.0)

                return exponential_weighted_loss(
                    model_pred,
                    target,
                    alphas_cumprod.to(model_pred.device),
                    timesteps.to(model_pred.device),
                    loss_map,
                    reduction=reduction,
                    boundary_shift=boundary_shift,
                )
            except Exception as e:
                logger.warning(f"EW loss failed: {e}. Falling back to L1 loss.")
                return F.l1_loss(model_pred, target, reduction=reduction)
        else:
            logger.warning(
                "EW loss requires timesteps parameter. Falling back to L1 loss."
            )
            return F.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "stepped":
        # Stepped loss with forward stepping and image reconstruction
        if (
            noise is not None
            and noisy_latents is not None
            and clean_latents is not None
            and timesteps is not None
            and noise_scheduler is not None
        ):
            try:
                # Get stepped loss configuration parameters
                stepped_step_size = kwargs.get("stepped_step_size", 50)
                stepped_multiplier = kwargs.get("stepped_multiplier", 1.0)

                step_loss = stepped_loss(
                    model_pred=model_pred,
                    latents=clean_latents,
                    noise=noise,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    noise_scheduler=noise_scheduler,
                    step_size=stepped_step_size,
                    reduction=reduction,
                )

                # Apply multiplier to adjust loss magnitude (as in original patch)
                return step_loss * stepped_multiplier

            except Exception as e:
                logger.warning(f"Stepped loss failed: {e}. Falling back to MSE loss.")
                return F.mse_loss(model_pred, target, reduction=reduction)
        else:
            logger.warning(
                "Stepped loss requires noise, noisy_latents, clean_latents, timesteps, and noise_scheduler parameters. Falling back to MSE loss."
            )
            return F.mse_loss(model_pred, target, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def log_loss_type_info(args) -> None:
    """Log loss type information at training start.

    Args:
        args: Training arguments containing loss configuration
    """
    loss_type = getattr(args, "loss_type", "mse")
    if loss_type == "pseudo_huber":
        pseudo_huber_c_min = getattr(args, "pseudo_huber_c_min", 0.01)
        pseudo_huber_c_max = getattr(args, "pseudo_huber_c_max", 1.0)
        schedule_type = getattr(args, "pseudo_huber_schedule_type", "linear")
        logger.info(
            f"📊 Loss function: Pseudo-Huber (c_min={pseudo_huber_c_min}, c_max={pseudo_huber_c_max}, schedule={schedule_type})"
        )
    elif loss_type == "stepped":
        stepped_step_size = getattr(args, "stepped_step_size", 50)
        stepped_multiplier = getattr(args, "stepped_multiplier", 1.0)
        logger.info(
            f"📊 Loss function: Stepped Recovery (step_size={stepped_step_size}, multiplier={stepped_multiplier})"
        )
    else:
        logger.info(f"📊 Loss function: {loss_type.upper()}")
