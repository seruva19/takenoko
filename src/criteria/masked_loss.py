"""
Masked training loss utilities.

"""

import torch
import torch.nn.functional as F
from typing import Optional
from common.logger import get_logger

logger = get_logger(__name__)


def masked_losses_with_prior(
    losses: torch.Tensor,
    prior_losses: Optional[torch.Tensor],
    mask: torch.Tensor,
    unmasked_weight: float,
    normalize_masked_area_loss: bool,
    masked_prior_preservation_weight: float,
) -> torch.Tensor:
    """
    Apply masking with optional prior preservation.

    Args:
        losses: Per-pixel losses with spatial dimensions (B, C, H, W) or (B, C, F, H, W)
        prior_losses: Prior model losses for preservation (same shape as losses)
        mask: Mask tensor with compatible shape, values [0, 1]
        unmasked_weight: Minimum weight for unmasked regions [0, 1]
        normalize_masked_area_loss: Whether to normalize by masked area
        masked_prior_preservation_weight: Weight for prior preservation [0, inf]

    Returns:
        Masked losses with same spatial dimensions

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values
    """
    # Validate inputs
    if not isinstance(losses, torch.Tensor) or losses.dim() < 3:
        raise ValueError(
            f"losses must be a tensor with at least 3 dimensions, got {losses.dim()}"
        )

    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a tensor")

    if not (0.0 <= unmasked_weight <= 1.0):
        logger.warning(
            f"unmasked_weight should be in [0, 1], got {unmasked_weight}, clamping"
        )
        unmasked_weight = max(0.0, min(1.0, unmasked_weight))

    if masked_prior_preservation_weight < 0.0:
        logger.warning(
            f"masked_prior_preservation_weight should be >= 0, got {masked_prior_preservation_weight}, using 0"
        )
        masked_prior_preservation_weight = 0.0

    # Validate shapes compatibility
    if prior_losses is not None:
        if prior_losses.shape != losses.shape:
            raise ValueError(
                f"prior_losses shape {prior_losses.shape} doesn't match losses shape {losses.shape}"
            )

    # Ensure mask can broadcast with losses
    try:
        _ = mask + torch.zeros_like(losses)
    except RuntimeError as e:
        raise ValueError(
            f"mask shape {mask.shape} is not compatible with losses shape {losses.shape}: {e}"
        )

    clamped_mask = torch.clamp(mask, unmasked_weight, 1.0)

    # Apply mask to main losses
    losses = losses * clamped_mask

    # Normalize by masked area if enabled
    if normalize_masked_area_loss:
        # Use dynamic dimension calculation for both 4D and 5D support
        spatial_dims = list(range(1, clamped_mask.dim()))
        mask_mean = clamped_mask.mean(dim=spatial_dims, keepdim=True)

        # Prevent division by zero and ensure numerical stability
        mask_mean = torch.clamp(mask_mean, min=1e-8)
        losses = losses / mask_mean

    # Prior preservation in unmasked areas
    if masked_prior_preservation_weight > 0.0 and prior_losses is not None:

        # Invert mask for unmasked regions
        # Modifies clamped_mask in-place
        inverted_mask = 1.0 - clamped_mask  # Get inverted mask
        prior_losses = prior_losses * inverted_mask * masked_prior_preservation_weight

        # Normalize prior losses if enabled
        if normalize_masked_area_loss:
            # Use dynamic dimension calculation for both 4D and 5D support
            spatial_dims = list(range(1, inverted_mask.dim()))
            prior_mask_mean = inverted_mask.mean(dim=spatial_dims, keepdim=True)

            # Prevent division by zero and ensure numerical stability
            prior_mask_mean = torch.clamp(prior_mask_mean, min=1e-8)
            prior_losses = prior_losses / prior_mask_mean

        # Combine losses
        losses = losses + prior_losses

    return losses


def prepare_mask_for_loss(
    mask: torch.Tensor, loss_shape: torch.Size, interpolation_mode: str = "area"
) -> torch.Tensor:
    """
    Prepare mask for loss computation.

    Args:
        mask: Raw mask tensor
        loss_shape: Target shape to match (B, C, H, W) or (B, C, F, H, W)
        interpolation_mode: Interpolation mode for resizing

    Returns:
        Processed mask ready for loss computation
    """
    original_mask = mask

    # Handle video tensors (5D)
    if len(loss_shape) == 5:  # Video: (B, C, F, H, W)
        target_spatial = loss_shape[3:]  # (H, W)

        # Ensure mask has proper dimensions for video
        if mask.dim() == 2:  # (H, W) -> (B, 1, F, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            mask = mask.expand(loss_shape[0], 1, loss_shape[2], -1, -1)
        elif mask.dim() == 3:  # (B, H, W) -> (B, 1, F, H, W)
            mask = mask.unsqueeze(1).unsqueeze(1)
            mask = mask.expand(-1, 1, loss_shape[2], -1, -1)
        elif mask.dim() == 4:  # (B, 1, H, W) -> (B, 1, F, H, W)
            mask = mask.unsqueeze(2)
            mask = mask.expand(-1, -1, loss_shape[2], -1, -1)
        elif mask.dim() == 5:  # (B, 1, F, H, W) - already correct
            pass
        else:
            raise ValueError(f"Unsupported mask dimensions for video: {mask.dim()}")

    else:  # Image: (B, C, H, W)
        target_spatial = loss_shape[2:]  # (H, W)

        # Ensure mask has batch dimension
        if mask.dim() == 2:  # (H, W) -> (1, 1, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:  # (B, H, W) -> (B, 1, H, W)
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4:  # (B, 1, H, W) - already correct
            pass
        else:
            raise ValueError(f"Unsupported mask dimensions for image: {mask.dim()}")

    # Resize mask to match loss spatial dimensions
    if mask.shape[-2:] != target_spatial:
        if len(loss_shape) == 5:  # Video - handle temporal dimension
            b, c, f, h, w = mask.shape
            # Reshape to (B*F, C, H, W) for interpolation
            mask_reshaped = mask.view(b * f, c, h, w)
            mask_reshaped = F.interpolate(
                mask_reshaped,
                size=target_spatial,
                mode=interpolation_mode,
                align_corners=False if interpolation_mode != "area" else None,
            )
            # Reshape back to (B, C, F, H, W)
            mask = mask_reshaped.view(b, c, f, target_spatial[0], target_spatial[1])
        else:  # Image
            mask = F.interpolate(
                mask,
                size=target_spatial,
                mode=interpolation_mode,
                align_corners=False if interpolation_mode != "area" else None,
            )

    # Normalize mask - handle different input ranges more robustly
    mask_min, mask_max = mask.min(), mask.max()

    if mask_min >= -1.1 and mask_max <= 1.1:  # Likely [-1, 1] range
        # Normalize from [-1, 1] to [0, 1]
        mask = mask / 2.0 + 0.5
    elif mask_min >= -0.1 and mask_max <= 1.1:  # Likely [0, 1] range
        # Already in correct range, just ensure bounds
        pass
    elif mask_min >= -0.1 and mask_max <= 255.1:  # Likely [0, 255] range
        # Normalize from [0, 255] to [0, 1]
        mask = mask / 255.0
    else:
        # Unknown range - normalize to [0, 1]
        mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)

    # Final clamp to ensure valid range
    mask = torch.clamp(mask, 0.0, 1.0)

    return mask
