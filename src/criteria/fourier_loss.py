"""
Fourier loss function module

This module provides various Fourier domain loss functions to enhance the frequency domain representation capability of deep learning models.
Main features include:
- Basic Fourier loss
- Frequency-weighted Fourier loss
- Multi-scale Fourier loss
- Adaptive Fourier loss
- Composite loss functions
- Settings and utility tools
"""

import torch
import logging
from typing import Optional, List, Dict, Any, Tuple
import json
from torch.fft import fftfreq, fftn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def create_frequency_weight_mask(
    height: int,
    width: int,
    high_freq_weight: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create a frequency weight mask, giving higher weights to high-frequency components

    Args:
        height: Height dimension
        width: Width dimension
        high_freq_weight: High-frequency weight multiplier
        device: Tensor device
        dtype: Tensor data type

    Returns:
        Frequency weight mask tensor
    """
    # Limit weight range to prevent excessive amplification
    high_freq_weight = max(1.0, min(high_freq_weight, 3.0))

    # Create frequency coordinates
    freq_h_kwargs = {}
    freq_w_kwargs = {}
    if device is not None:
        freq_h_kwargs["device"] = device
        freq_w_kwargs["device"] = device
    if dtype is not None:
        freq_h_kwargs["dtype"] = dtype
        freq_w_kwargs["dtype"] = dtype
    freq_h = fftfreq(height, **freq_h_kwargs)
    freq_w = fftfreq(width, **freq_w_kwargs)

    # Create 2D frequency grid
    freq_h_grid, freq_w_grid = torch.meshgrid(freq_h, freq_w, indexing="ij")

    # Calculate frequency magnitude
    freq_magnitude = torch.sqrt(freq_h_grid**2 + freq_w_grid**2)

    # Normalize to [0, 1] range, add smoothing
    max_freq = freq_magnitude.max()
    if max_freq > 0:
        freq_magnitude = freq_magnitude / max_freq
    else:
        freq_magnitude = torch.zeros_like(freq_magnitude)

    # Use a smoother weighting (sigmoid function instead of linear)
    # This reduces extreme weight values
    sigmoid_factor = 4.0  # Controls the steepness of the transition
    freq_sigmoid = torch.sigmoid(sigmoid_factor * (freq_magnitude - 0.5))

    # Create weights: low frequency is 1.0, high frequency gradually increases to high_freq_weight
    weight_mask = 1.0 + (high_freq_weight - 1.0) * freq_sigmoid

    return weight_mask


def compute_fourier_magnitude_spectrum(
    tensor: torch.Tensor,
    dims: tuple = (-2, -1),
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute the Fourier magnitude spectrum of a tensor

    Args:
        tensor: Input tensor
        dims: Dimensions for FFT
        eps: Numerical stability constant
        normalize: Whether to normalize the magnitude spectrum

    Returns:
        Fourier magnitude spectrum
    """
    # Compute multi-dimensional FFT
    fft_result = fftn(tensor, dim=dims)

    # Compute magnitude spectrum and add numerical stability
    magnitude = torch.abs(fft_result) + eps

    if normalize:
        # Normalize: divide by the square root of tensor size and max value
        tensor_numel = 1
        for dim in dims:
            tensor_numel *= tensor.shape[dim]

        # Normalize by tensor size (similar to FFTW normalization)
        magnitude = magnitude / (tensor_numel**0.5)

        # Further normalize by the input tensor's value range
        input_scale = torch.std(tensor) + eps
        magnitude = magnitude / input_scale

    return magnitude


def fourier_latent_loss_basic(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    norm_type: str = "l2",
    dims: tuple = (-2, -1),
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Basic Fourier latent loss calculation

    Args:
        model_pred: Model prediction (z_SR)
        target: Target (z_HR)
        norm_type: Loss norm type ("l1" or "l2")
        dims: FFT calculation dimensions
        eps: Numerical stability constant

    Returns:
        Fourier feature loss
    """
    # Compute Fourier magnitude spectrum (normalized)
    mag_pred = compute_fourier_magnitude_spectrum(model_pred, dims, eps, normalize=True)
    mag_target = compute_fourier_magnitude_spectrum(target, dims, eps, normalize=True)

    # Compute loss
    if norm_type == "l1":
        loss = torch.mean(torch.abs(mag_target - mag_pred))
    elif norm_type == "l2":
        loss = torch.mean((mag_target - mag_pred) ** 2)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    # Further constrain loss range to prevent outliers
    loss = torch.clamp(
        loss, max=torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
    )

    return loss


def fourier_latent_loss_weighted(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    high_freq_weight: float = 2.0,
    dims: tuple = (-2, -1),
    norm_type: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Frequency-weighted Fourier latent loss

    Args:
        model_pred: Model prediction (z_SR)
        target: Target (z_HR)
        high_freq_weight: High-frequency component weight multiplier
        dims: FFT calculation dimensions
        norm_type: Loss norm type
        eps: Numerical stability constant

    Returns:
        Weighted Fourier feature loss
    """
    # Limit high-frequency weight range to prevent excessive amplification
    high_freq_weight = torch.clamp(
        torch.tensor(high_freq_weight), min=1.0, max=3.0
    ).item()

    # Compute Fourier magnitude spectrum (normalized)
    mag_pred = compute_fourier_magnitude_spectrum(model_pred, dims, eps, normalize=True)
    mag_target = compute_fourier_magnitude_spectrum(target, dims, eps, normalize=True)

    # Create frequency weight mask
    height, width = model_pred.shape[dims[0]], model_pred.shape[dims[1]]
    weight_mask = create_frequency_weight_mask(
        height,
        width,
        high_freq_weight,
        device=model_pred.device,
        dtype=model_pred.dtype,
    )

    # Expand weight mask to match tensor shape
    while weight_mask.dim() < mag_pred.dim():
        weight_mask = weight_mask.unsqueeze(0)

    # Compute weighted difference
    diff = mag_target - mag_pred
    if norm_type == "l1":
        weighted_diff = torch.abs(diff) * weight_mask
    elif norm_type == "l2":
        weighted_diff = (diff**2) * weight_mask
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    # Compute weighted average loss
    loss = torch.mean(weighted_diff)

    # Further constrain loss range to prevent outliers
    loss = torch.clamp(
        loss, max=torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
    )

    return loss


def fourier_latent_loss_multiscale(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    scales: Optional[List[int]] = None,
    scale_weights: Optional[List[float]] = None,
    dims: tuple = (-2, -1),
    norm_type: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-scale Fourier latent loss

    Args:
        model_pred: Model prediction (z_SR)
        target: Target (z_HR)
        scales: Multiple scaling factors
        scale_weights: Weights for each scale (if None, calculated automatically)
        dims: FFT calculation dimensions
        norm_type: Loss norm type
        eps: Numerical stability constant

    Returns:
        Multi-scale Fourier feature loss
    """
    if scales is None:
        scales = [1, 2]

    if scale_weights is None:
        scale_weights = [1.0 / scale for scale in scales]

    if len(scale_weights) != len(scales):
        raise ValueError("scale_weights length must match scales length")

    total_loss = 0.0
    total_weight = 0.0

    for scale, weight in zip(scales, scale_weights):
        if scale == 1:
            pred_scaled = model_pred
            target_scaled = target
        else:
            # Check if tensor dimensions are sufficient for pooling
            if (
                model_pred.dim() >= 4
                and model_pred.shape[-1] >= scale
                and model_pred.shape[-2] >= scale
            ):
                # Use average pooling for downsampling
                pred_scaled = F.avg_pool2d(model_pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
            else:
                # Skip invalid scale
                continue

        # Compute Fourier loss at this scale
        scale_loss = fourier_latent_loss_basic(
            pred_scaled, target_scaled, norm_type, dims, eps
        )

        total_loss += weight * scale_loss
        total_weight += weight

    # Prevent division by zero
    if total_weight == 0:
        return torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)

    # Constrain final loss value
    final_loss = total_loss / total_weight
    if not isinstance(final_loss, torch.Tensor):
        final_loss = torch.tensor(
            final_loss, device=model_pred.device, dtype=model_pred.dtype
        )
    final_loss = torch.clamp(
        final_loss,
        max=torch.tensor(10.0, device=final_loss.device, dtype=final_loss.dtype),
    )

    return final_loss


def fourier_latent_loss_adaptive(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    current_step: int,
    total_steps: int,
    max_weight: float = 2.0,
    min_weight: float = 0.5,
    dims: tuple = (-2, -1),
    norm_type: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Adaptive Fourier latent loss (weight adjusts with training progress)

    Args:
        model_pred: Model prediction (z_SR)
        target: Target (z_HR)
        current_step: Current training step
        total_steps: Total training steps
        max_weight: Maximum high-frequency weight
        min_weight: Minimum high-frequency weight
        dims: FFT calculation dimensions
        norm_type: Loss norm type
        eps: Numerical stability constant

    Returns:
        Adaptive Fourier feature loss
    """
    # Limit weight range
    max_weight = max(1.0, min(max_weight, 3.0))
    min_weight = max(0.5, min(min_weight, max_weight))

    # Calculate training progress (0.0 to 1.0)
    progress = min(current_step / max(total_steps, 1), 1.0)

    # Early training emphasizes high frequency, later gradually balances
    high_freq_weight = max_weight - (max_weight - min_weight) * progress

    return fourier_latent_loss_weighted(
        model_pred, target, high_freq_weight, dims, norm_type, eps
    )


def conditional_loss_with_fourier(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
    current_step: int = 0,
    total_steps: int = 1000,
    # Fourier feature loss parameters
    fourier_weight: float = 0.05,
    fourier_mode: str = "weighted",  # "basic", "weighted", "multiscale", "adaptive"
    fourier_norm: str = "l2",
    fourier_dims: tuple = (-2, -1),
    fourier_high_freq_weight: float = 2.0,
    fourier_scales: Optional[List[int]] = None,
    fourier_scale_weights: Optional[List[float]] = None,
    fourier_adaptive_max_weight: float = 2.0,
    fourier_adaptive_min_weight: float = 0.5,
    fourier_eps: float = 1e-8,
    fourier_warmup_steps: int = 200,
) -> torch.Tensor:
    """
    Enhanced conditional_loss supporting Fourier feature loss

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        loss_type: Base loss type ("fourier")
        reduction: Loss reduction method ("mean", "sum", "none")
        huber_c: Huber loss parameter
        current_step: Current training step
        total_steps: Total training steps

        # Fourier feature loss parameters
        fourier_weight: Fourier loss weight
        fourier_mode: Fourier loss mode
        fourier_norm: Fourier loss norm ("l1" or "l2")
        fourier_dims: FFT calculation dimensions
        fourier_high_freq_weight: High-frequency weight multiplier (weighted mode)
        fourier_scales: Multi-scale list (multiscale mode)
        fourier_scale_weights: Scale weight list (multiscale mode)
        fourier_adaptive_max_weight: Adaptive max weight (adaptive mode)
        fourier_adaptive_min_weight: Adaptive min weight (adaptive mode)
        fourier_eps: Numerical stability constant
        fourier_warmup_steps: Fourier loss warmup steps

    Returns:
        Composite loss value
    """

    # Compute base loss
    if fourier_norm == "l1":
        base_loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    else:
        base_loss = torch.nn.functional.mse_loss(
            model_pred, target, reduction=reduction
        )

    # If not fourier loss or weight is 0, return base loss directly
    if loss_type != "fourier" or fourier_weight <= 0.0:
        return base_loss

    # If weight is 0, return base loss directly
    if fourier_weight <= 0.0:
        return base_loss

    # If within warmup period, return base loss directly
    if current_step < fourier_warmup_steps:
        return base_loss

    # Check if tensor dimensions are sufficient
    if model_pred.dim() < 3 or target.dim() < 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Fourier loss requires at least 3D tensors, got {model_pred.dim()}D and {target.dim()}D tensors, skipping Fourier loss calculation"
        )
        return base_loss

    # Ensure tensor shapes match
    if model_pred.shape != target.shape:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"model_pred and target shapes do not match: {model_pred.shape} vs {target.shape}, skipping Fourier loss calculation"
        )
        return base_loss

    try:
        # Compute Fourier loss based on mode
        if fourier_mode == "basic":
            fourier_loss = fourier_latent_loss_basic(
                model_pred, target, fourier_norm, fourier_dims, fourier_eps
            )
        elif fourier_mode == "weighted":
            fourier_loss = fourier_latent_loss_weighted(
                model_pred,
                target,
                fourier_high_freq_weight,
                fourier_dims,
                fourier_norm,
                fourier_eps,
            )
        elif fourier_mode == "multiscale":
            if fourier_scales is None:
                fourier_scales = [1, 2]
            fourier_loss = fourier_latent_loss_multiscale(
                model_pred,
                target,
                fourier_scales,
                fourier_scale_weights,
                fourier_dims,
                fourier_norm,
                fourier_eps,
            )
        elif fourier_mode == "adaptive":
            fourier_loss = fourier_latent_loss_adaptive(
                model_pred,
                target,
                current_step,
                total_steps,
                fourier_adaptive_max_weight,
                fourier_adaptive_min_weight,
                fourier_dims,
                fourier_norm,
                fourier_eps,
            )
        elif fourier_mode == "unified":
            # Use unified mode, support extra parameters
            unified_kwargs = {}
            if fourier_scales is not None:
                unified_kwargs["scales"] = fourier_scales
            if fourier_scale_weights is not None:
                unified_kwargs["scale_weights"] = fourier_scale_weights

            fourier_loss = fourier_latent_loss_unified(
                model_pred,
                target,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
                high_freq_weight=fourier_high_freq_weight,
                current_step=current_step,
                total_steps=total_steps,
                max_weight=fourier_adaptive_max_weight,
                min_weight=fourier_adaptive_min_weight,
                **unified_kwargs,
            )
        elif fourier_mode in [
            "unified_basic",
            "unified_balanced",
            "unified_detail",
            "unified_adaptive",
        ]:
            # Use simplified unified mode
            mode_map = {
                "unified_basic": "basic",
                "unified_balanced": "balanced",
                "unified_detail": "detail",
                "unified_adaptive": "adaptive",
            }
            fourier_loss = fourier_latent_loss_unified_simple(
                model_pred,
                target,
                mode=mode_map[fourier_mode],
                current_step=current_step,
                total_steps=total_steps,
            )
        else:
            raise ValueError(f"Unsupported fourier_mode: {fourier_mode}")

        # Dynamically adjust Fourier loss weight to avoid large gap with base loss
        # Safely convert loss to scalar for comparison
        try:
            # Ensure base loss is scalar
            if base_loss.numel() > 1:
                base_loss_magnitude = base_loss.detach().mean().item()
            else:
                base_loss_magnitude = base_loss.detach().item()

            # Ensure Fourier loss is scalar
            if fourier_loss.numel() > 1:
                fourier_loss_magnitude = fourier_loss.detach().mean().item()
            else:
                fourier_loss_magnitude = fourier_loss.detach().item()

            # Compute adaptive weight to ensure Fourier loss does not overwhelm base loss
            adaptive_weight = fourier_weight
            if (
                fourier_loss_magnitude > 0
                and base_loss_magnitude > 0
                and not (
                    torch.isnan(torch.tensor(fourier_loss_magnitude))
                    or torch.isnan(torch.tensor(base_loss_magnitude))
                )
            ):
                ratio = fourier_loss_magnitude / base_loss_magnitude
                if ratio > 10.0:  # If Fourier loss is too large, reduce weight
                    adaptive_weight = fourier_weight / (ratio / 10.0)
                    adaptive_weight = max(adaptive_weight, fourier_weight * 0.1)
        except (RuntimeError, ValueError, AttributeError) as e:
            # If unable to get scalar value, use original weight
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Unable to compute adaptive weight, using original weight: {e}"
            )
            adaptive_weight = fourier_weight

        # Combine base loss and Fourier loss
        total_loss = base_loss + adaptive_weight * fourier_loss

        return total_loss

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Fourier loss calculation failed: {e}, falling back to base loss"
        )
        return base_loss


def fourier_latent_loss_unified(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    # Basic parameters
    dims: tuple = (-2, -1),
    norm_type: str = "l2",
    eps: float = 1e-8,
    # Multi-scale parameters
    scales: Optional[List[int]] = None,
    scale_weights: Optional[List[float]] = None,
    enable_multiscale: bool = True,
    # Frequency weighting parameters
    enable_frequency_weighting: bool = True,
    high_freq_weight: float = 2.0,
    freq_weight_per_scale: Optional[List[float]] = None,  # Frequency weight per scale
    # Adaptive parameters
    enable_adaptive: bool = True,
    current_step: int = 0,
    total_steps: int = 1000,
    adaptive_mode: str = "linear",  # "linear", "cosine", "exponential"
    max_weight: float = 2.5,
    min_weight: float = 0.8,
    # Integration strategy parameters
    multiscale_weight: float = 0.6,  # Multi-scale loss weight
    weighted_weight: float = 0.4,  # Weighted loss weight
    adaptive_scaling: bool = True,  # Whether to adaptively scale weights
) -> torch.Tensor:
    """
    Unified Fourier latent loss calculation

    Combines multi-scale, frequency weighting, and adaptive strategies in a unified implementation

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        dims: FFT calculation dimensions
        norm_type: Loss norm type
        eps: Numerical stability constant

        # Multi-scale parameters
        scales: Multi-scale list
        scale_weights: Weights for each scale
        enable_multiscale: Whether to enable multi-scale

        # Frequency weighting parameters
        enable_frequency_weighting: Whether to enable frequency weighting
        high_freq_weight: Base high-frequency weight
        freq_weight_per_scale: Frequency weight override per scale

        # Adaptive parameters
        enable_adaptive: Whether to enable adaptation
        current_step: Current step
        total_steps: Total steps
        adaptive_mode: Adaptive mode
        max_weight: Maximum weight
        min_weight: Minimum weight

        # Integration strategy parameters
        multiscale_weight: Multi-scale component weight
        weighted_weight: Weighted component weight
        adaptive_scaling: Whether to adaptively scale combined weights

    Returns:
        Integrated Fourier loss
    """

    # Parameter validation and default settings
    if scales is None:
        scales = [1, 2, 4] if enable_multiscale else [1]

    if scale_weights is None:
        # Dynamically calculate scale weights, larger scales get smaller weights
        scale_weights = [1.0 / (scale**0.5) for scale in scales]
        # Normalize weights
        total_scale_weight = sum(scale_weights)
        scale_weights = [w / total_scale_weight for w in scale_weights]

    if freq_weight_per_scale is None:
        freq_weight_per_scale = [high_freq_weight] * len(scales)
    elif len(freq_weight_per_scale) != len(scales):
        # Extend or truncate to correct length
        freq_weight_per_scale = (
            freq_weight_per_scale + [high_freq_weight] * len(scales)
        )[: len(scales)]

    # Compute adaptive weight factor
    adaptive_factor = 1.0
    if enable_adaptive:
        # Calculate training progress
        progress = min(current_step / max(total_steps, 1), 1.0)

        # Compute adaptive factor based on different modes
        if adaptive_mode == "linear":
            # Linear decay: from max_weight to min_weight
            adaptive_factor = max_weight - (max_weight - min_weight) * progress
        elif adaptive_mode == "cosine":
            # Cosine decay: smoother transition
            import math

            adaptive_factor = min_weight + (max_weight - min_weight) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        elif adaptive_mode == "exponential":
            # Exponential decay: fast drop early, slow later
            import math

            adaptive_factor = min_weight + (max_weight - min_weight) * math.exp(
                -5 * progress
            )
        else:
            raise ValueError(f"Unsupported adaptive_mode: {adaptive_mode}")

    # Compute multi-scale loss component
    multiscale_loss = 0.0
    if enable_multiscale and len(scales) > 1:
        total_loss = 0.0
        total_weight = 0.0

        for i, (scale, scale_weight) in enumerate(zip(scales, scale_weights)):
            # Get tensor at this scale
            if scale == 1:
                pred_scaled = model_pred
                target_scaled = target
            else:
                # Check tensor dimensions
                if (
                    model_pred.dim() >= 4
                    and model_pred.shape[-1] >= scale
                    and model_pred.shape[-2] >= scale
                ):
                    pred_scaled = F.avg_pool2d(model_pred, scale)
                    target_scaled = F.avg_pool2d(target, scale)
                else:
                    continue

            # Compute frequency-weighted loss at this scale
            if enable_frequency_weighting:
                current_freq_weight = freq_weight_per_scale[i] * adaptive_factor
                scale_loss = fourier_latent_loss_weighted(
                    pred_scaled,
                    target_scaled,
                    current_freq_weight,
                    dims,
                    norm_type,
                    eps,
                )
            else:
                scale_loss = fourier_latent_loss_basic(
                    pred_scaled, target_scaled, norm_type, dims, eps
                )

            total_loss += scale_weight * scale_loss
            total_weight += scale_weight

        if total_weight > 0:
            multiscale_loss = total_loss / total_weight

    # Compute single-scale weighted loss component (base scale)
    weighted_loss = 0.0
    if enable_frequency_weighting:
        current_freq_weight = high_freq_weight * adaptive_factor
        weighted_loss = fourier_latent_loss_weighted(
            model_pred, target, current_freq_weight, dims, norm_type, eps
        )
    else:
        weighted_loss = fourier_latent_loss_basic(
            model_pred, target, norm_type, dims, eps
        )

    # Combine losses
    if enable_multiscale and len(scales) > 1:
        # Adaptively adjust combined weights
        if adaptive_scaling:
            # Adjust multi-scale and weighted loss ratio based on training progress
            progress = min(current_step / max(total_steps, 1), 1.0)
            # Early: emphasize multi-scale, later: emphasize detail
            current_multiscale_weight = multiscale_weight * (
                1.0 + 0.5 * (1.0 - progress)
            )
            current_weighted_weight = weighted_weight * (1.0 + 0.5 * progress)

            # Normalize weights
            total_weight = current_multiscale_weight + current_weighted_weight
            current_multiscale_weight /= total_weight
            current_weighted_weight /= total_weight
        else:
            current_multiscale_weight = multiscale_weight
            current_weighted_weight = weighted_weight

        final_loss = (
            current_multiscale_weight * multiscale_loss
            + current_weighted_weight * weighted_loss
        )
    else:
        # If no multi-scale, use only weighted loss
        final_loss = weighted_loss

    # Final constraint
    final_loss = torch.clamp(
        final_loss,
        max=torch.tensor(10.0, device=final_loss.device, dtype=final_loss.dtype),
    )

    return final_loss


def get_fourier_loss_unified_config(mode: str = "balanced") -> Dict[str, Any]:
    """
    Get default unified Fourier loss settings
    """
    # Default configs
    configs = {
        "basic": {
            "enable_multiscale": False,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "high_freq_weight": 1.5,
            "adaptive_mode": "linear",
            "max_weight": 2.0,
            "min_weight": 1.0,
        },
        "balanced": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2],
            "high_freq_weight": 2.0,
            "adaptive_mode": "linear",
            "max_weight": 2.5,
            "min_weight": 0.8,
            "multiscale_weight": 0.6,
            "weighted_weight": 0.4,
        },
        "detail": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2, 4],
            "high_freq_weight": 2.5,
            "freq_weight_per_scale": [2.0, 2.5, 3.0],
            "adaptive_mode": "cosine",
            "max_weight": 3.0,
            "min_weight": 1.0,
            "multiscale_weight": 0.7,
            "weighted_weight": 0.3,
        },
        "adaptive": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2],
            "adaptive_mode": "exponential",
            "max_weight": 2.8,
            "min_weight": 0.5,
            "adaptive_scaling": True,
        },
    }

    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")

    return configs[mode]


def fourier_latent_loss_unified_simple(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "balanced",
    current_step: int = 0,
    total_steps: int = 1000,
    **kwargs,
) -> torch.Tensor:
    """
    Simplified unified Fourier loss with default configs

    Args:
        model_pred: Model prediction tensor
        target: Target tensor
        mode: Default mode
            - "basic": Basic mode, mainly uses single-scale weighting
            - "balanced": Balanced mode, combines multi-scale and weighting
            - "detail": Detail mode, emphasizes high frequency and multi-scale
            - "adaptive": Adaptive mode, emphasizes dynamic adjustment
        current_step: Current step
        total_steps: Total steps
        **kwargs: Other parameter overrides

    Returns:
        Fourier loss
    """

    # Merge config and user parameters
    config = get_fourier_loss_unified_config(mode).copy()
    config.update(kwargs)

    return fourier_latent_loss_unified(
        model_pred, target, current_step=current_step, total_steps=total_steps, **config
    )


# Convenience function: default settings


def get_fourier_loss_config(mode: str = "balanced") -> Dict[str, Any]:
    """
    Get default Fourier loss settings

    Args:
        mode: Setting mode
            - "conservative": Conservative setting, smaller Fourier weight
            - "balanced": Balanced setting, medium Fourier weight
            - "aggressive": Aggressive setting, larger Fourier weight
            - "super_resolution": For super-resolution tasks
            - "fine_detail": Focus on detail enhancement

    Returns:
        Settings dictionary
    """
    configs = {
        "conservative": {
            "fourier_weight": 0.01,
            "fourier_mode": "basic",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 1.5,
            "fourier_warmup_steps": 500,
        },
        "balanced": {
            "fourier_weight": 0.05,
            "fourier_mode": "weighted",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 2.0,
            "fourier_warmup_steps": 300,
        },
        "aggressive": {
            "fourier_weight": 0.1,
            "fourier_mode": "multiscale",
            "fourier_norm": "l1",
            "fourier_scales": [1, 2, 4],
            "fourier_warmup_steps": 200,
        },
        "super_resolution": {
            "fourier_weight": 0.08,
            "fourier_mode": "adaptive",
            "fourier_norm": "l2",
            "fourier_adaptive_max_weight": 3.0,
            "fourier_adaptive_min_weight": 1.0,
            "fourier_warmup_steps": 400,
        },
        "fine_detail": {
            "fourier_weight": 0.12,
            "fourier_mode": "weighted",
            "fourier_norm": "l1",
            "fourier_high_freq_weight": 2.5,
            "fourier_warmup_steps": 100,
        },
        "unified_balanced": {
            "fourier_weight": 0.06,
            "fourier_mode": "unified_balanced",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 250,
        },
        "unified_detail": {
            "fourier_weight": 0.08,
            "fourier_mode": "unified_detail",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 200,
        },
        "unified_adaptive": {
            "fourier_weight": 0.07,
            "fourier_mode": "unified_adaptive",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 300,
        },
        "unified_custom": {
            "fourier_weight": 0.05,
            "fourier_mode": "unified",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 2.0,
            "fourier_scales": [1, 2, 4],
            "fourier_adaptive_max_weight": 2.5,
            "fourier_adaptive_min_weight": 0.8,
            "fourier_warmup_steps": 250,
        },
    }

    if mode not in configs:
        raise ValueError(
            f"Unknown mode: {mode}. Available modes: {list(configs.keys())}"
        )

    return configs[mode]


# Convenience function for training scripts


def apply_fourier_loss_to_args(args, quick_mode: str = "balanced"):
    """
    Apply Fourier loss settings to training arguments

    Args:
        args: Training arguments object
        quick_mode: Setting mode
    """
    quick_mode = (
        quick_mode
        if quick_mode
        in [
            "conservative",
            "balanced",
            "aggressive",
            "super_resolution",
            "fine_detail",
            "unified_balanced",
            "unified_detail",
            "unified_adaptive",
            "unified_custom",
        ]
        else "balanced"
    )

    config = get_fourier_loss_config(quick_mode)

    # Set loss type to fourier
    args.loss_type = "fourier"

    # Set Fourier-related parameters
    for key, value in config.items():
        setattr(args, key, value)

    # Set to default if parameter does not exist
    if hasattr(args, "fourier_weight") is False:
        args.fourier_weight = 0.05
    if hasattr(args, "fourier_mode") is False:
        args.fourier_mode = "weighted"
    if hasattr(args, "fourier_norm") is False:
        args.fourier_norm = "l2"
    if hasattr(args, "fourier_dims") is False:
        args.fourier_dims = (-2, -1)
    if hasattr(args, "fourier_high_freq_weight") is False:
        args.fourier_high_freq_weight = 2.0
    if hasattr(args, "fourier_scales") is False:
        args.fourier_scales = None
    if hasattr(args, "fourier_scale_weights") is False:
        args.fourier_scale_weights = None
    if hasattr(args, "fourier_adaptive_max_weight") is False:
        args.fourier_adaptive_max_weight = 2.0
    if hasattr(args, "fourier_adaptive_min_weight") is False:
        args.fourier_adaptive_min_weight = 0.5
    if hasattr(args, "fourier_eps") is False:
        args.fourier_eps = 1e-8
    if hasattr(args, "fourier_warmup_steps") is False:
        args.fourier_warmup_steps = 300

    if args.fourier_mode in [
        "unified_basic",
        "unified_balanced",
        "unified_detail",
        "unified_adaptive",
    ]:
        # Use simplified unified mode
        mode_map = {
            "unified_basic": "basic",
            "unified_balanced": "balanced",
            "unified_detail": "detail",
            "unified_adaptive": "adaptive",
        }
        args.fourier_unified_config = json.dumps(
            get_fourier_loss_unified_config(mode_map[args.fourier_mode])
        )

    return args
