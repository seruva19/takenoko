from __future__ import annotations

import argparse
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from common.logger import get_logger
from enhancements.self_resampling.self_resampling_helper import (
    SelfResamplingHelper,
    SelfResamplingState,
)

logger = get_logger(__name__, level=logging.INFO)


def can_use_fast_self_resampling_rollout(
    args: argparse.Namespace,
    controlnet: Optional[Any],
    bfm_conditioning_helper: Optional[Any],
    catlvdm_corruption_helper: Optional[Any],
) -> bool:
    if not bool(getattr(args, "self_resampling_fast_rollout", True)):
        return False
    if bool(getattr(args, "enable_control_lora", False)):
        return False
    if bool(getattr(args, "enable_controlnet", False)) and controlnet is not None:
        return False
    if bfm_conditioning_helper is not None:
        return False
    if catlvdm_corruption_helper is not None:
        return False
    if bool(getattr(args, "enable_mixflow", False)):
        return False
    if bool(getattr(args, "enable_internal_guidance", False)):
        return False
    if bool(getattr(args, "enable_dispersive_loss", False)):
        return False
    if bool(getattr(args, "gradient_checkpointing", False)):
        return False
    return True


def _coerce_rollout_prediction(model_pred: Any) -> torch.Tensor:
    if isinstance(model_pred, tuple):
        model_pred = model_pred[0]
    if isinstance(model_pred, list):
        return torch.stack(model_pred, dim=0).detach()
    if torch.is_tensor(model_pred):
        return model_pred.detach()
    raise TypeError(f"Unexpected rollout prediction type: {type(model_pred).__name__}")


def build_full_self_resampling_rollout_predictor(
    predict_with_full_dit: Callable[[torch.Tensor, torch.Tensor], Any],
) -> Callable[..., torch.Tensor]:
    def _predict_velocity_for_self_resampling_full(
        rollout_input: torch.Tensor,
        rollout_timesteps: torch.Tensor,
        rollout_meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        del rollout_meta
        with torch.no_grad():
            model_pred = predict_with_full_dit(rollout_input, rollout_timesteps)
        return _coerce_rollout_prediction(model_pred)

    return _predict_velocity_for_self_resampling_full


def build_fast_self_resampling_rollout_predictor(
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    rollout_transformer: Any,
    rollout_latents: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    network_dtype: torch.dtype,
    global_step: Optional[int],
    patch_size: Tuple[int, int, int],
    semanticgen_state: Any,
    semantic_conditioning_helper: Optional[Any],
) -> Callable[..., torch.Tensor]:
    patch_t, patch_h, patch_w = patch_size
    lat_f, lat_h, lat_w = rollout_latents.shape[2:5]
    seq_len = (lat_f * lat_h * lat_w) // (patch_t * patch_h * patch_w)
    context = [t.to(device=accelerator.device, dtype=network_dtype) for t in batch["t5"]]
    semanticgen_state.reset()
    context = semanticgen_state.apply_context(
        args=args,
        accelerator=accelerator,
        context=context,
        batch=batch,
        global_step=global_step,
        conditioning_helper=semantic_conditioning_helper,
    )

    rollout_kv_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    use_rollout_kv_cache = bool(getattr(args, "self_resampling_rollout_kv_cache", True))
    rollout_self_attn_kv_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    use_rollout_self_attn_kv_cache = bool(
        getattr(args, "self_resampling_rollout_self_attn_kv_cache", True)
    )
    last_rollout_t_scalar: Optional[float] = None
    last_history_frame_count: int = -1

    def _predict_velocity_for_self_resampling_fast(
        rollout_input: torch.Tensor,
        rollout_timesteps: torch.Tensor,
        rollout_meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        nonlocal last_rollout_t_scalar, last_history_frame_count
        history_frame_count = int((rollout_meta or {}).get("history_frame_count", 0))
        t_scalar = float(rollout_timesteps.detach().float().mean().item())
        if (
            last_rollout_t_scalar is None
            or abs(t_scalar - last_rollout_t_scalar) > 1e-6
            or history_frame_count < last_history_frame_count
        ):
            rollout_self_attn_kv_cache.clear()
        last_rollout_t_scalar = t_scalar
        last_history_frame_count = history_frame_count
        with torch.no_grad():
            with accelerator.autocast():
                model_pred = rollout_transformer(
                    rollout_input.to(device=accelerator.device, dtype=network_dtype),
                    t=rollout_timesteps.to(device=accelerator.device),
                    context=context,
                    clip_fea=None,
                    seq_len=seq_len,
                    y=None,
                    force_keep_mask=None,
                    controlnet_states=None,
                    controlnet_weight=1.0,
                    controlnet_stride=1,
                    dispersive_loss_target_block=None,
                    return_intermediate=False,
                    internal_guidance_target_block=None,
                    return_internal_guidance=False,
                    reg_cls_token=None,
                    segment_idx=None,
                    bfm_semfeat_tokens=None,
                    enable_rollout_kv_cache=use_rollout_kv_cache,
                    rollout_kv_cache=rollout_kv_cache if use_rollout_kv_cache else None,
                    enable_rollout_self_attn_kv_cache=use_rollout_self_attn_kv_cache,
                    rollout_self_attn_kv_cache=(
                        rollout_self_attn_kv_cache
                        if use_rollout_self_attn_kv_cache
                        else None
                    ),
                    rollout_history_frame_count=history_frame_count,
                )
        return _coerce_rollout_prediction(model_pred)

    return _predict_velocity_for_self_resampling_fast


def maybe_apply_self_resampling(
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    self_resampling_helper: Optional[SelfResamplingHelper],
    noisy_model_input: torch.Tensor,
    rollout_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    rollout_transformer: Any,
    batch: Dict[str, torch.Tensor],
    network_dtype: torch.dtype,
    patch_size: Tuple[int, int, int],
    global_step: Optional[int],
    controlnet: Optional[Any],
    eqm_enabled: bool,
    warned_self_resampling_eqm: bool,
    warned_self_resampling_fast_rollout_fallback: bool,
    bfm_conditioning_helper: Optional[Any],
    catlvdm_corruption_helper: Optional[Any],
    semanticgen_state: Any,
    semantic_conditioning_helper: Optional[Any],
    predict_with_full_dit: Callable[[torch.Tensor, torch.Tensor], Any],
) -> Tuple[torch.Tensor, Optional[SelfResamplingState], bool, bool]:
    if self_resampling_helper is None or not self_resampling_helper.enabled:
        return (
            noisy_model_input,
            None,
            warned_self_resampling_eqm,
            warned_self_resampling_fast_rollout_fallback,
        )

    if eqm_enabled:
        if not warned_self_resampling_eqm:
            logger.warning("Self-resampling skipped: EqM mode active.")
            warned_self_resampling_eqm = True
        return (
            noisy_model_input,
            None,
            warned_self_resampling_eqm,
            warned_self_resampling_fast_rollout_fallback,
        )

    predict_velocity_fn: Optional[Callable[..., torch.Tensor]] = None
    if bool(getattr(self_resampling_helper, "model_rollout", False)):
        if can_use_fast_self_resampling_rollout(
            args=args,
            controlnet=controlnet,
            bfm_conditioning_helper=bfm_conditioning_helper,
            catlvdm_corruption_helper=catlvdm_corruption_helper,
        ):
            try:
                predict_velocity_fn = build_fast_self_resampling_rollout_predictor(
                    args=args,
                    accelerator=accelerator,
                    rollout_transformer=rollout_transformer,
                    rollout_latents=rollout_latents,
                    batch=batch,
                    network_dtype=network_dtype,
                    global_step=global_step,
                    patch_size=patch_size,
                    semanticgen_state=semanticgen_state,
                    semantic_conditioning_helper=semantic_conditioning_helper,
                )
            except Exception as exc:
                if not warned_self_resampling_fast_rollout_fallback:
                    logger.warning(
                        "Self-resampling fast rollout unavailable (%s); falling back to full call_dit rollout.",
                        exc,
                    )
                    warned_self_resampling_fast_rollout_fallback = True

        if predict_velocity_fn is None:
            predict_velocity_fn = build_full_self_resampling_rollout_predictor(
                predict_with_full_dit
            )

    noisy_model_input, self_resampling_state = self_resampling_helper.apply_to_inputs(
        noisy_model_input=noisy_model_input,
        latents=rollout_latents,
        noise=noise,
        global_step=global_step,
        predict_velocity_fn=predict_velocity_fn,
        timestep_reference=timesteps,
    )
    return (
        noisy_model_input,
        self_resampling_state,
        warned_self_resampling_eqm,
        warned_self_resampling_fast_rollout_fallback,
    )
