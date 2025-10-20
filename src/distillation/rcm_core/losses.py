"""Loss builders used by the RCM distillation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ReconstructionLossConfig:
    """Controls reconstruction loss hyperparameters."""

    weight: float = 1.0
    reduction: str = "mean"  # "mean" | "sum"


@dataclass(slots=True)
class BehaviorCloningLossConfig:
    """Controls action imitation loss hyperparameters."""

    weight: float = 1.0
    label_smoothing: float = 0.0
    reduction: str = "mean"


@dataclass(slots=True)
class LogitsKLLossConfig:
    """KL divergence between teacher and student logits."""

    weight: float = 0.0
    temperature: float = 1.0
    reduction: str = "batchmean"  # "batchmean" | "mean" | "sum"


@dataclass(slots=True)
class DivergenceLossConfig:
    """Forward / reverse divergence weighting."""

    forward_weight: float = 0.0
    reverse_weight: float = 0.0
    temperature: float = 1.0
    reduction: str = "batchmean"


@dataclass(slots=True)
class DistillationLossOutput:
    """Structured output for downstream logging."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    behavior_clone: torch.Tensor
    kl_divergence: torch.Tensor
    forward_divergence: torch.Tensor
    reverse_divergence: torch.Tensor
    metrics: Dict[str, float]


def _apply_reduction(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return tensor.sum()
    if reduction == "mean":
        return tensor.mean()
    raise ValueError(f"Unsupported reduction mode: {reduction}")


def _per_sample_weight(weighting: Optional[torch.Tensor], batch_size: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if weighting is None:
        return None
    weight = weighting
    if weight.ndim > 1:
        weight = weight.view(batch_size, -1).mean(dim=1)
    if weight.shape[0] != batch_size:
        weight = weight.expand(batch_size)
    return weight.to(dtype=dtype)


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(tensor.size(0), -1)


def compute_distillation_losses(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    flow_target: torch.Tensor,
    weighting: torch.Tensor | None,
    recon_cfg: ReconstructionLossConfig,
    bc_cfg: BehaviorCloningLossConfig,
    kl_cfg: LogitsKLLossConfig | None = None,
    div_cfg: DivergenceLossConfig | None = None,
) -> DistillationLossOutput:
    """Calculate WAN-style distillation losses using flow-matching predictions."""

    if student_pred.shape != flow_target.shape:
        raise ValueError("student_pred and flow_target must have identical shapes.")
    if teacher_pred.shape != student_pred.shape:
        raise ValueError("teacher_pred must match student_pred shape.")

    batch_size = student_pred.size(0)
    per_sample_weight = _per_sample_weight(weighting, batch_size, student_pred.dtype)

    # Reconstruction (student -> flow target)
    recon_error = (student_pred - flow_target.detach()).pow(2)
    recon_per_sample = recon_error.view(batch_size, -1).mean(dim=1)
    if per_sample_weight is not None:
        recon_per_sample = recon_per_sample * per_sample_weight
    reconstruction_loss = _apply_reduction(recon_per_sample, recon_cfg.reduction)

    # Behaviour cloning (student -> teacher)
    bc_error = (student_pred - teacher_pred.detach()).pow(2)
    bc_per_sample = bc_error.view(batch_size, -1).mean(dim=1)
    if bc_cfg.label_smoothing > 0:
        smooth = float(bc_cfg.label_smoothing)
        bc_per_sample = (1 - smooth) * bc_per_sample + smooth * recon_per_sample
    behavior_loss = _apply_reduction(bc_per_sample, bc_cfg.reduction)

    # KL divergence on flattened logits
    device = student_pred.device
    kl_loss = torch.tensor(0.0, device=device)
    if kl_cfg is not None and kl_cfg.weight > 0:
        temperature = max(float(kl_cfg.temperature), 1e-6)
        student_logits = _flatten_features(student_pred) / temperature
        teacher_logits = _flatten_features(teacher_pred.detach()) / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction=kl_cfg.reduction) * (
            temperature**2
        )

    forward_div = torch.tensor(0.0, device=device)
    reverse_div = torch.tensor(0.0, device=device)
    if div_cfg is not None and (div_cfg.forward_weight > 0 or div_cfg.reverse_weight > 0):
        temperature = max(float(div_cfg.temperature), 1e-6)
        student_logits = _flatten_features(student_pred) / temperature
        teacher_logits = _flatten_features(teacher_pred.detach()) / temperature
        target_logits = _flatten_features(flow_target.detach()) / temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()
        target_probs = target_log_probs.exp()

        reduction = div_cfg.reduction

        def _reduce(values: torch.Tensor) -> torch.Tensor:
            if reduction == "batchmean":
                return values.mean(dim=1).mean()
            if reduction == "mean":
                return values.mean()
            if reduction == "sum":
                return values.sum()
            raise ValueError(f"Unsupported divergence reduction: {reduction}")

        if div_cfg.forward_weight > 0:
            # Forward divergence: student vs target
            forward_vals = F.kl_div(student_log_probs, target_probs, reduction="none") * (
                temperature**2
            )
            forward_div = _reduce(forward_vals)

        if div_cfg.reverse_weight > 0:
            # Reverse divergence: teacher vs student
            reverse_vals = F.kl_div(teacher_log_probs, student_probs, reduction="none") * (
                temperature**2
            )
            reverse_div = _reduce(reverse_vals)

    total = (
        recon_cfg.weight * reconstruction_loss
        + bc_cfg.weight * behavior_loss
        + (kl_cfg.weight * kl_loss if kl_cfg is not None else 0.0)
        + (div_cfg.forward_weight * forward_div if div_cfg is not None else 0.0)
        + (div_cfg.reverse_weight * reverse_div if div_cfg is not None else 0.0)
    )
    metrics = {
        "loss/reconstruction": float(reconstruction_loss.detach().item()),
        "loss/behavior_clone": float(behavior_loss.detach().item()),
        "loss/kl": float(kl_loss.detach().item()),
        "loss/div_forward": float(forward_div.detach().item()),
        "loss/div_reverse": float(reverse_div.detach().item()),
        "loss/total": float(total.detach().item()),
    }

    return DistillationLossOutput(
        total=total,
        reconstruction=reconstruction_loss,
        behavior_clone=behavior_loss,
        kl_divergence=kl_loss,
        forward_divergence=forward_div,
        reverse_divergence=reverse_div,
        metrics=metrics,
    )
