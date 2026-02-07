from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ReflexFlowLossTerms:
    adr_loss: Optional[torch.Tensor]
    fc_loss: Optional[torch.Tensor]
    exposure_bias_mean: Optional[torch.Tensor]
    weight_mean: Optional[torch.Tensor]


def _safe_vector_norm(
    tensor: torch.Tensor, *, dim: int, keepdim: bool, eps: float
) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim).clamp_min(
        eps
    )


def compute_reflexflow_adr_loss(
    *,
    model_pred: torch.Tensor,
    noisy_model_input: torch.Tensor,
    latents: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Compute ADR term using normalized direction alignment."""
    if model_pred.shape != noisy_model_input.shape or latents.shape != noisy_model_input.shape:
        return None

    pred = model_pred.to(torch.float32)
    anti_drift_target = (noisy_model_input - latents).to(torch.float32)

    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = anti_drift_target.view(anti_drift_target.shape[0], -1)
    pred_dir = F.normalize(pred_flat, dim=1, eps=eps)
    target_dir = F.normalize(target_flat, dim=1, eps=eps)
    return (pred_dir - target_dir).pow(2).sum(dim=1).mean()


def compute_reflexflow_fc_terms(
    *,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    clean_pred: Optional[torch.Tensor],
    alpha: float,
    eps: float = 1e-6,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute FC term and diagnostics from exposure bias."""
    if clean_pred is None:
        return None, None, None
    if clean_pred.shape != model_pred.shape or target.shape != model_pred.shape:
        return None, None, None

    # Exposure bias map from biased-vs-clean prediction discrepancy.
    delta = (model_pred.detach() - clean_pred.detach()).to(torch.float32)
    if delta.dim() < 2:
        return None, None, None

    # Collapse channels into a non-negative saliency map per sample.
    if delta.dim() == 2:
        delta_energy = delta.abs()
    else:
        delta_energy = _safe_vector_norm(delta, dim=1, keepdim=True, eps=eps)
    reduce_dims = tuple(range(1, delta_energy.dim()))
    denom = delta_energy.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    weight = 1.0 + float(alpha) * (delta_energy / denom)

    residual = (model_pred - target).to(torch.float32)
    fc_loss = (weight * residual).pow(2).mean()
    return fc_loss, delta_energy.mean().detach(), weight.mean().detach()


def compute_reflexflow_loss_terms(
    *,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    noisy_model_input: torch.Tensor,
    latents: torch.Tensor,
    clean_pred: Optional[torch.Tensor],
    alpha: float,
) -> ReflexFlowLossTerms:
    adr_loss = compute_reflexflow_adr_loss(
        model_pred=model_pred,
        noisy_model_input=noisy_model_input,
        latents=latents,
    )
    fc_loss, exposure_bias_mean, weight_mean = compute_reflexflow_fc_terms(
        model_pred=model_pred,
        target=target,
        clean_pred=clean_pred,
        alpha=alpha,
    )
    return ReflexFlowLossTerms(
        adr_loss=adr_loss,
        fc_loss=fc_loss,
        exposure_bias_mean=exposure_bias_mean,
        weight_mean=weight_mean,
    )
