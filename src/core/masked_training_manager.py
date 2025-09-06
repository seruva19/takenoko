"""
Masked Training Manager - Encapsulated masked training functionality.

This class provides a complete masked training solution that can be enabled/disabled
via configuration without affecting existing code paths.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from common.logger import get_logger
from criteria.masked_loss import masked_losses_with_prior, prepare_mask_for_loss
from criteria.pseudo_huber_loss import conditional_loss_with_pseudo_huber

logger = get_logger(__name__)


@dataclass
class MaskedTrainingConfig:
    """Configuration for masked training."""

    unmasked_probability: float = 0.1
    unmasked_weight: float = 0.1
    masked_prior_preservation_weight: float = 0.0
    normalize_masked_area_loss: bool = False

    mask_interpolation_mode: str = "area"

    enable_prior_computation: bool = True
    prior_computation_method: str = "lora_disabled"  # "lora_disabled" or "cached"

    # Video-specific parameters
    temporal_consistency_weight: float = 0.0  # Weight for temporal consistency loss
    frame_consistency_mode: str = "adjacent"  # "adjacent" or "all_pairs"


class MaskedTrainingManager:
    """
    Manages masked training with prior preservation.

    Combines prior preservation with loss computation flow.
    Designed for maximum encapsulation with minimal
    integration points.
    """

    def __init__(self, config: MaskedTrainingConfig):
        self.config = config
        logger.info("Initialized MaskedTrainingManager with config: %s", config)

    def should_remove_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Determine which samples should have masks removed.

        Returns:
            Boolean tensor (batch_size,) indicating which masks to remove
        """
        if self.config.unmasked_probability <= 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)

        return torch.rand(batch_size, device=device) < self.config.unmasked_probability

    def compute_masked_loss_with_prior(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        prior_pred: Optional[torch.Tensor] = None,
        loss_type: str = "mse",
        **loss_kwargs,
    ) -> torch.Tensor:
        """
        Main method: Compute masked loss with optional prior preservation.

        Args:
            model_pred: Model predictions (B, C, H, W) or (B, C, F, H, W)
            target: Target values (B, C, H, W) or (B, C, F, H, W)
            mask: Mask tensor (any reasonable shape)
            prior_pred: Prior model predictions for preservation
            loss_type: Loss function type
            **loss_kwargs: Additional arguments for loss computation

        Returns:
            Scalar loss value

        Raises:
            ValueError: If inputs have incompatible shapes
            RuntimeError: If loss computation fails
        """
        # Validate inputs
        if not isinstance(model_pred, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise ValueError("model_pred and target must be tensors")

        if model_pred.shape != target.shape:
            raise ValueError(
                f"model_pred shape {model_pred.shape} != target shape {target.shape}"
            )

        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask must be a tensor")

        # Validate prior prediction if provided
        if prior_pred is not None:
            if not isinstance(prior_pred, torch.Tensor):
                raise ValueError("prior_pred must be a tensor when provided")
            if prior_pred.shape != model_pred.shape:
                raise ValueError(
                    f"prior_pred shape {prior_pred.shape} != model_pred shape {model_pred.shape}"
                )

        try:
            batch_size = model_pred.shape[0]

            # Handle random mask removal
            mask_removal = self.should_remove_mask(batch_size, model_pred.device)
            if mask_removal.any():
                # Create a copy to avoid modifying original mask (handle read-only tensors)
                mask = mask.clone().detach()
                # Ensure mask is writable and on correct device/dtype
                mask = mask.to(device=model_pred.device, dtype=model_pred.dtype)
                mask[mask_removal] = 1.0  # Remove mask for selected samples

            # Compute per-pixel loss FIRST (keep spatial dimensions)
            loss = self._compute_pixel_loss(
                model_pred, target, loss_type, **loss_kwargs
            )

            # Prepare mask for loss computation
            processed_mask = prepare_mask_for_loss(
                mask, loss.shape, self.config.mask_interpolation_mode
            )

            # Compute prior loss if needed
            # Prior loss measures how well prior prediction matches target
            prior_loss = None
            if (
                self.config.masked_prior_preservation_weight > 0.0
                and prior_pred is not None
            ):
                prior_loss = self._compute_pixel_loss(
                    prior_pred, target, loss_type, **loss_kwargs
                )

            # Apply masking with prior preservation
            masked_loss = masked_losses_with_prior(
                losses=loss,
                prior_losses=prior_loss,
                mask=processed_mask,
                unmasked_weight=self.config.unmasked_weight,
                normalize_masked_area_loss=self.config.normalize_masked_area_loss,
                masked_prior_preservation_weight=self.config.masked_prior_preservation_weight,
            )

            # Add temporal consistency loss for video
            if (
                self.config.temporal_consistency_weight > 0.0 and masked_loss.dim() == 5
            ):  # Video tensor
                temporal_loss = self._compute_temporal_consistency_loss(
                    masked_loss, processed_mask
                )
                masked_loss = (
                    masked_loss
                    + self.config.temporal_consistency_weight * temporal_loss
                )

            # Reduce dimensions AFTER masking
            return self._reduce_loss(masked_loss)

        except Exception as e:
            logger.error(f"Error in masked loss computation: {e}")
            logger.error(
                f"Shapes - model_pred: {model_pred.shape}, target: {target.shape}, mask: {mask.shape}"
            )
            if prior_pred is not None:
                logger.error(f"prior_pred shape: {prior_pred.shape}")
            # Fallback to basic loss computation to prevent training crash
            logger.warning("Falling back to unmasked loss computation")
            basic_loss = self._compute_pixel_loss(
                model_pred, target, loss_type, **loss_kwargs
            )
            return self._reduce_loss(basic_loss)

    def _compute_pixel_loss(
        self, pred: torch.Tensor, target: torch.Tensor, loss_type: str, **kwargs
    ) -> torch.Tensor:
        """Compute per-pixel loss without reduction, adapted for flow matching."""
        # Handle both 4D (image) and 5D (video) tensors
        if pred.dim() == 5:  # Video: (B, C, F, H, W)
            # Reshape to (B*F, C, H, W) for loss computation
            b, c, f, h, w = pred.shape
            pred_reshaped = pred.view(b * f, c, h, w)
            target_reshaped = target.view(b * f, c, h, w)
        else:
            pred_reshaped = pred
            target_reshaped = target

        # Compute loss based on type
        if loss_type == "mse":
            loss = F.mse_loss(
                pred_reshaped.float(), target_reshaped.float(), reduction="none"
            )
        elif loss_type == "mae" or loss_type == "l1":
            loss = F.l1_loss(
                pred_reshaped.float(), target_reshaped.float(), reduction="none"
            )
        elif loss_type == "pseudo_huber":
            loss = conditional_loss_with_pseudo_huber(
                pred_reshaped, target_reshaped, reduction="none", **kwargs
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Reshape back to original dimensions if needed
        if pred.dim() == 5:
            loss = loss.view(b, c, f, h, w)

        return loss

    def _compute_temporal_consistency_loss(
        self, masked_loss: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss for video masks.

        Args:
            masked_loss: Loss tensor with shape (B, C, F, H, W)
            mask: Mask tensor with shape (B, 1, F, H, W)

        Returns:
            Temporal consistency loss with same shape as input
        """
        if masked_loss.dim() != 5:
            return torch.zeros_like(masked_loss)

        b, c, f, h, w = masked_loss.shape

        if f < 2:  # Need at least 2 frames
            return torch.zeros_like(masked_loss)

        consistency_loss = torch.zeros_like(masked_loss)

        if self.config.frame_consistency_mode == "adjacent":
            # Compute consistency between adjacent frames
            for t in range(f - 1):
                frame_t = masked_loss[:, :, t]  # (B, C, H, W)
                frame_t1 = masked_loss[:, :, t + 1]  # (B, C, H, W)
                mask_t = mask[:, :, t]  # (B, 1, H, W)
                mask_t1 = mask[:, :, t + 1]  # (B, 1, H, W)

                # Only penalize differences where both frames are masked
                combined_mask = mask_t * mask_t1  # (B, 1, H, W)

                # Compute L2 difference
                frame_diff = (frame_t - frame_t1) ** 2  # (B, C, H, W)
                weighted_diff = frame_diff * combined_mask

                # Add to both frames
                consistency_loss[:, :, t] += weighted_diff
                consistency_loss[:, :, t + 1] += weighted_diff

        elif self.config.frame_consistency_mode == "all_pairs":
            # Compute consistency between all frame pairs (more expensive)
            for t1 in range(f):
                for t2 in range(t1 + 1, f):
                    frame_t1 = masked_loss[:, :, t1]
                    frame_t2 = masked_loss[:, :, t2]
                    mask_t1 = mask[:, :, t1]
                    mask_t2 = mask[:, :, t2]

                    combined_mask = mask_t1 * mask_t2
                    frame_diff = (frame_t1 - frame_t2) ** 2
                    weighted_diff = frame_diff * combined_mask

                    # Weight by temporal distance
                    temporal_weight = 1.0 / (abs(t2 - t1) + 1.0)
                    weighted_diff *= temporal_weight

                    consistency_loss[:, :, t1] += weighted_diff
                    consistency_loss[:, :, t2] += weighted_diff

        return consistency_loss

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss from spatial dimensions to scalar."""
        # Reduce all dimensions except batch
        spatial_dims = list(range(1, loss.dim()))
        reduced_loss = loss.mean(dim=spatial_dims)  # (B,)
        return reduced_loss.mean()  # Scalar


def create_masked_training_manager(args) -> Optional[MaskedTrainingManager]:
    """
    Factory function to create masked training manager from args.

    Returns None if masked training is disabled.
    """
    if not getattr(args, "use_masked_training_with_prior", False):
        return None

    config = MaskedTrainingConfig(
        unmasked_probability=getattr(args, "unmasked_probability", 0.1),
        unmasked_weight=getattr(args, "unmasked_weight", 0.1),
        masked_prior_preservation_weight=getattr(
            args, "masked_prior_preservation_weight", 0.0
        ),
        normalize_masked_area_loss=getattr(args, "normalize_masked_area_loss", False),
        mask_interpolation_mode=getattr(args, "mask_interpolation_mode", "area"),
        enable_prior_computation=getattr(args, "enable_prior_computation", True),
        prior_computation_method=getattr(
            args, "prior_computation_method", "lora_disabled"
        ),
        temporal_consistency_weight=getattr(args, "temporal_consistency_weight", 0.0),
        frame_consistency_mode=getattr(args, "frame_consistency_mode", "adjacent"),
    )

    return MaskedTrainingManager(config)
