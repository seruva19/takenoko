"""Dual-head alignment helper handler for training-loop maintainability."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from enhancements.memflow_guidance.training_integration import suspend_memflow_guidance

logger = get_logger(__name__)


def _resolve_teacher_mode(dual_head_alignment_helper: Any) -> str:
    try:
        teacher_mode = str(
            getattr(dual_head_alignment_helper, "config", None).teacher_mode
        )
    except Exception:
        teacher_mode = "base_model"
    if teacher_mode == "base_model":
        teacher_mode = "base_model_multiplier"
    return teacher_mode


def _should_apply_dual_head_step(dual_head_alignment_helper: Any, global_step: int) -> bool:
    try:
        return bool(dual_head_alignment_helper.should_apply_step(global_step))
    except Exception:
        return True


def compute_dual_head_alignment_loss(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    dual_head_alignment_helper: Any,
    global_step: int,
    unwrapped_net: Any,
    active_transformer: Any,
    latents_for_dit: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    network_dtype: torch.dtype,
    control_signal_processor: Optional[Any],
    controlnet: Optional[Any],
    model_pred: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor],
    call_dit_fn: Callable[..., Any],
    semanticgen_state: Any,
    warned_no_multiplier: bool,
    warned_teacher_failed: bool,
    last_mixflow_stats: Optional[Dict[str, float]],
) -> Tuple[
    Optional[torch.Tensor],
    Dict[str, float],
    bool,
    bool,
    Optional[Dict[str, float]],
]:
    """Compute and return optional dual-head alignment loss and updated runtime flags."""

    dual_loss: Optional[torch.Tensor] = None
    dual_metrics: Dict[str, float] = {}
    teacher_pred: Optional[torch.Tensor] = None

    should_apply_dual_head = _should_apply_dual_head_step(
        dual_head_alignment_helper, global_step
    )
    teacher_mode = _resolve_teacher_mode(dual_head_alignment_helper)

    if should_apply_dual_head and teacher_mode in (
        "base_model_multiplier",
        "base_model_disable_lora",
    ):
        can_switch_multiplier = hasattr(unwrapped_net, "set_multiplier")
        can_switch_enabled = hasattr(unwrapped_net, "set_enabled")
        if can_switch_multiplier or can_switch_enabled:
            prev_multiplier = float(getattr(unwrapped_net, "multiplier", 1.0))
            prev_enabled = bool(getattr(unwrapped_net, "enabled", True))
            saved_semantic_kl = semanticgen_state.kl_loss
            saved_semantic_tokens = semanticgen_state.tokens_for_alignment
            saved_mixflow_stats = last_mixflow_stats
            try:
                with torch.no_grad():
                    if teacher_mode == "base_model_disable_lora" and can_switch_enabled:
                        unwrapped_net.set_enabled(False)
                    elif can_switch_multiplier:
                        unwrapped_net.set_multiplier(0.0)
                    with suspend_memflow_guidance():
                        teacher_result = call_dit_fn(
                            args,
                            accelerator,
                            active_transformer,
                            latents_for_dit,
                            batch,
                            noise,
                            noisy_model_input,
                            timesteps,
                            network_dtype,
                            control_signal_processor,
                            controlnet,
                            global_step=global_step,
                            reg_cls_token=None,
                            apply_stable_velocity_target=False,
                        )
                if (
                    isinstance(teacher_result, (tuple, list))
                    and len(teacher_result) > 0
                    and torch.is_tensor(teacher_result[0])
                ):
                    teacher_pred = teacher_result[0].detach()
            except Exception as exc:
                if not warned_teacher_failed:
                    logger.warning(
                        "Dual-head teacher forward failed once and will be skipped: %s",
                        exc,
                    )
                    warned_teacher_failed = True
            finally:
                try:
                    if can_switch_enabled:
                        unwrapped_net.set_enabled(prev_enabled)
                except Exception:
                    pass
                try:
                    if can_switch_multiplier:
                        unwrapped_net.set_multiplier(prev_multiplier)
                except Exception:
                    pass
                semanticgen_state.kl_loss = saved_semantic_kl
                semanticgen_state.tokens_for_alignment = saved_semantic_tokens
                last_mixflow_stats = saved_mixflow_stats
        elif not warned_no_multiplier:
            logger.warning(
                "Dual-head alignment helper requires network.set_multiplier or network.set_enabled for base-model teacher mode; skipping auxiliary terms."
            )
            warned_no_multiplier = True

    try:
        dual_loss, dual_metrics = dual_head_alignment_helper.compute_loss(
            student_pred=model_pred,
            target=target,
            teacher_pred=teacher_pred,
            weighting=weighting,
            global_step=global_step,
        )
    except Exception as exc:
        if not warned_teacher_failed:
            logger.warning(
                "Dual-head alignment helper loss failed once and will be skipped: %s",
                exc,
            )
            warned_teacher_failed = True

    return (
        dual_loss,
        dual_metrics,
        warned_no_multiplier,
        warned_teacher_failed,
        last_mixflow_stats,
    )
