"""Training-loop helper for merged-state manifold consensus."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from enhancements.memflow_guidance.training_integration import suspend_memflow_guidance
from utils.train_utils import get_sigmas

logger = get_logger(__name__)


def compute_manifold_consensus_loss(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    manifold_consensus_helper: Any,
    global_step: int,
    active_transformer: Any,
    latents_for_dit: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    network_dtype: torch.dtype,
    control_signal_processor: Optional[Any],
    controlnet: Optional[Any],
    noise_scheduler: Any,
    model_pred: torch.Tensor,
    weighting: Optional[torch.Tensor],
    call_dit_fn: Callable[..., Any],
    semanticgen_state: Any,
    last_mixflow_stats: Optional[Dict[str, float]],
    model_timesteps_override: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float], Optional[Dict[str, float]]]:
    """Compute a merged-state continuation loss from extra stochastic views."""

    if manifold_consensus_helper is None:
        return None, {}, last_mixflow_stats
    if not bool(manifold_consensus_helper.should_collect_extra_views()):
        return None, {}, last_mixflow_stats

    sigmas = get_sigmas(
        noise_scheduler,
        timesteps,
        latents_for_dit.device,
        n_dim=latents_for_dit.dim(),
        dtype=noisy_model_input.dtype,
        source="manifold_consensus",
    )

    saved_semantic_kl = semanticgen_state.kl_loss
    saved_semantic_tokens = semanticgen_state.tokens_for_alignment
    saved_mixflow_stats = last_mixflow_stats

    num_extra_views = max(0, int(manifold_consensus_helper.config.num_views) - 1)
    for _ in range(num_extra_views):
        alt_noise = torch.randn_like(latents_for_dit)
        alt_noisy_model_input = sigmas * alt_noise + (1.0 - sigmas) * latents_for_dit
        try:
            manifold_consensus_helper.begin_extra_view_capture()
            with torch.no_grad():
                with suspend_memflow_guidance():
                    _ = call_dit_fn(
                        args,
                        accelerator,
                        active_transformer,
                        latents_for_dit,
                        batch,
                        alt_noise,
                        alt_noisy_model_input,
                        timesteps,
                        network_dtype,
                        control_signal_processor,
                        controlnet,
                        global_step=global_step,
                        reg_cls_token=None,
                        model_timesteps_override=model_timesteps_override,
                        apply_stable_velocity_target=False,
                    )
        except Exception as exc:
            logger.warning(
                "Manifold consensus auxiliary view failed once and will be skipped for this step: %s",
                exc,
            )
            manifold_consensus_helper.reset_step_state()
            semanticgen_state.kl_loss = saved_semantic_kl
            semanticgen_state.tokens_for_alignment = saved_semantic_tokens
            return None, {}, saved_mixflow_stats
        finally:
            manifold_consensus_helper.finish_extra_view_capture()
            semanticgen_state.kl_loss = saved_semantic_kl
            semanticgen_state.tokens_for_alignment = saved_semantic_tokens
            last_mixflow_stats = saved_mixflow_stats

    consensus_states = manifold_consensus_helper.build_consensus_states()
    if not consensus_states:
        return None, {}, last_mixflow_stats

    inject_handles = manifold_consensus_helper.install_consensus_injection_hooks(
        consensus_states
    )
    consensus_teacher_pred: Optional[torch.Tensor] = None
    try:
        with torch.no_grad():
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
                    model_timesteps_override=model_timesteps_override,
                    apply_stable_velocity_target=False,
                )
        if (
            isinstance(teacher_result, (tuple, list))
            and len(teacher_result) > 0
            and torch.is_tensor(teacher_result[0])
        ):
            consensus_teacher_pred = teacher_result[0].detach()
    except Exception as exc:
        logger.warning(
            "Manifold consensus continuation pass failed once and will be skipped for this step: %s",
            exc,
        )
        semanticgen_state.kl_loss = saved_semantic_kl
        semanticgen_state.tokens_for_alignment = saved_semantic_tokens
        return None, {}, saved_mixflow_stats
    finally:
        for handle in inject_handles:
            try:
                handle.remove()
            except Exception:
                pass
        semanticgen_state.kl_loss = saved_semantic_kl
        semanticgen_state.tokens_for_alignment = saved_semantic_tokens
        last_mixflow_stats = saved_mixflow_stats

    loss, metrics = manifold_consensus_helper.compute_loss(
        student_pred=model_pred,
        consensus_teacher_pred=consensus_teacher_pred,
        weighting=weighting,
    )
    return loss, metrics, last_mixflow_stats
