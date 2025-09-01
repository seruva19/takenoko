"""
Utility functions for enhanced Contrastive Flow Matching implementation.
Based on improvements from DeltaFM implementation.
"""

import torch
from typing import Optional, Dict, Any, Tuple


def class_conditioned_negative_sampling(
    labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Sample negative examples ensuring they come from different classes.
    
    Args:
        labels: Tensor of class labels, shape [batch_size]
        device: Device to put result on
        
    Returns:
        indices: Indices for negative samples, shape [batch_size]
        
    Raises:
        ValueError: If labels tensor is invalid or empty
    """
    if labels is None or labels.numel() == 0:
        raise ValueError("Labels tensor cannot be None or empty")
        
    if labels.dim() != 1:
        raise ValueError(f"Labels must be 1D tensor, got shape {labels.shape}")
        
    batch_size = labels.shape[0]
    
    if batch_size < 2:
        raise ValueError(f"Batch size must be >= 2 for contrastive learning, got {batch_size}")
    
    # Create a mask where True indicates different classes
    mask = ~(labels[None, :] == labels[:, None])
    
    # Convert mask to weights for sampling
    weights = mask.float()
    weights_sum = weights.sum(dim=1, keepdim=True)
    
    # Handle edge case where no different classes exist
    if (weights_sum == 0).any():
        # Fallback to random sampling if no class diversity
        choices = torch.randperm(batch_size, device=device)
        # Ensure no self-sampling by shifting indices
        choices = (choices + 1) % batch_size
    else:
        # Normalize weights to create valid probability distribution
        weights = weights / weights_sum.clamp(min=1)
        # Sample from available choices based on weights
        choices = torch.multinomial(weights, 1).squeeze(1)
    
    # Validation: ensure no self-sampling occurred
    self_sampling_mask = (choices == torch.arange(batch_size, device=device))
    if self_sampling_mask.any():
        # Fix self-sampling by replacing with random valid choice
        for idx in torch.where(self_sampling_mask)[0]:
            available = torch.where(mask[idx])[0]
            if len(available) > 0:
                choices[idx] = available[torch.randint(0, len(available), (1,))]
            else:
                # Last resort: just shift by 1
                choices[idx] = (idx + 1) % batch_size
    
    return choices


def compute_enhanced_contrastive_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    negative_target: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    null_class_idx: Optional[int] = None,
    dont_contrast_on_unconditional: bool = False,
    lambda_val: float = 0.05,
    loss_fn = None,
    **loss_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute enhanced contrastive flow matching loss with better handling.
    
    Args:
        model_pred: Model predictions
        target: Positive targets  
        negative_target: Negative targets
        labels: Class labels for conditional handling
        null_class_idx: Index representing unconditional/null class
        dont_contrast_on_unconditional: Whether to skip contrastive loss on unconditional samples
        lambda_val: Contrastive loss weight
        loss_fn: Loss function to use
        **loss_kwargs: Additional arguments for loss function
        
    Returns:
        Dictionary with loss components
    """
    # Compute positive (flow matching) loss
    flow_loss = loss_fn(model_pred, target, **loss_kwargs)
    
    # Compute negative (contrastive) loss
    contrastive_loss_raw = loss_fn(model_pred, negative_target, **loss_kwargs)
    
    # Handle unconditional samples if specified
    if dont_contrast_on_unconditional and labels is not None and null_class_idx is not None:
        # Mask out unconditional samples from contrastive loss
        non_null_mask = (labels != null_class_idx)
        batch_size = labels.shape[0]
        
        if non_null_mask.any():
            # Scale contrastive loss to account for masked samples
            contrastive_loss = (
                contrastive_loss_raw * non_null_mask.unsqueeze(-1).expand_as(contrastive_loss_raw)
            )
            # Rescale to maintain proper magnitude
            scale_factor = batch_size / non_null_mask.sum().float()
            contrastive_loss = contrastive_loss * scale_factor
        else:
            # All samples are unconditional, skip contrastive loss
            contrastive_loss = torch.zeros_like(contrastive_loss_raw)
    else:
        contrastive_loss = contrastive_loss_raw
    
    # Combine losses: positive - lambda * negative  
    total_loss = flow_loss - lambda_val * contrastive_loss
    
    return {
        'total_loss': total_loss,
        'flow_loss': flow_loss,
        'contrastive_loss': contrastive_loss,
        'lambda_val': lambda_val
    }


def extract_class_labels_from_batch(
    batch: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """
    Extract class labels from batch data if available.
    
    Args:
        batch: Batch dictionary containing various data
        
    Returns:
        Class labels tensor if found, None otherwise
    """
    # Common keys where class labels might be stored
    potential_keys = [
        'class_labels', 'labels', 'class_ids', 'categories',
        'y', 'class', 'target', 'caption_ids'  # Some datasets use caption IDs as class proxies
    ]
    
    for key in potential_keys:
        if key in batch and batch[key] is not None:
            labels = batch[key]
            # Ensure it's a tensor and has the right shape
            if isinstance(labels, torch.Tensor) and labels.numel() > 0:
                # Flatten if needed to ensure it's [batch_size]
                if labels.dim() > 1:
                    labels = labels.flatten()
                return labels
    
    return None