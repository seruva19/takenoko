"""Generator / critic step helpers for the rCM training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .forward import RCMDenoiseResult
from .losses import DistillationLossOutput, compute_distillation_losses


@dataclass(slots=True)
class RCMStepMetrics:
    """Captures aggregate metrics for student / critic steps."""

    loss_output: DistillationLossOutput
    fake_loss: Optional[torch.Tensor]
    metrics: Dict[str, float]
    fake_loss_active: bool


def execute_student_step(
    *,
    ctx,
    student_outputs: torch.Tensor,
    teacher_outputs: torch.Tensor,
    teacher_target: torch.Tensor,
    weighting: Optional[torch.Tensor],
    recon_cfg,
    bc_cfg,
    kl_cfg,
    div_cfg,
    student_rcm: Optional[RCMDenoiseResult],
    teacher_rcm: Optional[RCMDenoiseResult],
) -> RCMStepMetrics:
    """Compute student losses (flow + optional fake-score contribution)."""

    loss_output = compute_distillation_losses(
        student_pred=student_outputs,
        teacher_pred=teacher_outputs.detach(),
        flow_target=teacher_target.detach(),
        weighting=weighting,
        recon_cfg=recon_cfg,
        bc_cfg=bc_cfg,
        kl_cfg=kl_cfg,
        div_cfg=div_cfg,
    )

    fake_loss = None
    fake_loss_active = False
    metrics = dict(loss_output.metrics)

    if (
        ctx.bundle.fake_score is not None
        and ctx.fake_score_optimizer is not None
        and student_rcm is not None
        and teacher_rcm is not None
    ):
        fake_input = student_rcm.prediction.x0.detach()
        fake_target = teacher_rcm.prediction.x0.detach()
        fake_pred = ctx.bundle.fake_score(fake_input)
        fake_loss = F.mse_loss(fake_pred, fake_target)
        fake_weight = float(ctx.config.extra_args.get("rcm_fake_score_loss_weight", 1.0))
        loss_output.total = loss_output.total + fake_weight * fake_loss
        metrics["loss/fake_score"] = fake_loss.item()
        fake_loss_active = True

    return RCMStepMetrics(
        loss_output=loss_output,
        fake_loss=fake_loss,
        metrics=metrics,
        fake_loss_active=fake_loss_active,
    )


def execute_fake_score_step(
    *,
    ctx,
    loss_output: DistillationLossOutput,
    fake_loss_active: bool,
) -> None:
    """Optional post-processing hook after the generator step."""

    if fake_loss_active and ctx.fake_score_optimizer is not None:
        ctx.fake_score_optimizer.step()
        if ctx.fake_score_scheduler is not None:
            ctx.fake_score_scheduler.step()


def update_ema_if_needed(ctx) -> None:
    """Synchronise EMA weights with the current student network if enabled."""

    if ctx.ema_helper is None:
        return

    ctx.ema_helper.update(ctx.bundle.student)
    if ctx.bundle.teacher_ema is not None:
        try:
            ctx.bundle.teacher_ema.load_state_dict(ctx.ema_helper.shadow, strict=False)
        except Exception as exc:  # pragma: no cover - guard against mismatch
            ctx.accelerator.print(f"[RCM] EMA teacher sync failed: {exc}")
