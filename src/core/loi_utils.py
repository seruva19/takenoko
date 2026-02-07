"""Helpers for LOI (locally optimal initialization) backprop passes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common.logger import get_logger
from enhancements.slider.slider_integration import compute_slider_loss_if_enabled
from utils.train_utils import compute_loss_weighting_for_sd3
from enhancements.temporal_consistency.training_integration import (
    enhance_loss_with_temporal_consistency,
)

logger = get_logger(__name__)


def run_loi_extra_backward(
    *,
    adapter_setup_fn,
    optimizer,
    training_core,
    args,
    accelerator,
    active_transformer,
    network,
    latents_for_dit,
    batch,
    noise,
    noisy_model_input,
    timesteps,
    network_dtype,
    control_signal_processor,
    controlnet,
    global_step,
    reg_cls_input,
    latents,
    noise_scheduler,
    vae,
    target_loi_context: Optional[Dict[str, Any]],
    repa_helper,
    sft_alignment_helper,
    moalign_helper,
    semfeat_helper,
    reg_helper,
    reg_cls_target,
    sara_helper,
    layer_sync_helper,
    crepa_helper,
    haste_helper,
    contrastive_attention_helper=None,
    transition_loss_context,
    transition_forward_fn=None,
) -> Optional[tuple]:
    """Run one LOI extra backward pass with temporary adapter overrides."""
    adapter_state = adapter_setup_fn()
    optimizer.zero_grad(set_to_none=True)
    model_result = training_core.call_dit(
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
        reg_cls_token=reg_cls_input,
        apply_stable_velocity_target=False,
    )
    if model_result is None or model_result[0] is None:
        return None
    reg_cls_pred_loi = None
    intermediate_z_loi = None
    internal_guidance_pred_loi = None
    internal_guidance_shift_loi = None
    if len(model_result) == 6:
        (
            model_pred_loi,
            target_loi,
            intermediate_z_loi,
            internal_guidance_pred_loi,
            internal_guidance_shift_loi,
            reg_cls_pred_loi,
        ) = model_result
    else:
        (
            model_pred_loi,
            target_loi,
            intermediate_z_loi,
            internal_guidance_pred_loi,
            internal_guidance_shift_loi,
        ) = model_result

    weighting_loi = compute_loss_weighting_for_sd3(
        args.weighting_scheme,
        noise_scheduler,
        timesteps,
        accelerator.device,
        network_dtype,
    )
    transition_ctx = transition_loss_context
    if transition_forward_fn is not None:
        transition_ctx = dict(transition_loss_context or {})
        transition_ctx["transition_forward_fn"] = transition_forward_fn
    loss_components_loi = compute_slider_loss_if_enabled(
        loss_computer=training_core.loss_computer,
        transformer=active_transformer,
        network=network,
        noisy_latents=noisy_model_input,
        timesteps=timesteps,
        batch=batch,
        noise=noise,
        noise_scheduler=noise_scheduler,
        args=args,
        accelerator=accelerator,
        latents=latents,
        network_dtype=network_dtype,
        model_pred=model_pred_loi,
        target=target_loi,
        weighting=weighting_loi,
        intermediate_z=intermediate_z_loi,
        internal_guidance_pred=internal_guidance_pred_loi,
        internal_guidance_shift=internal_guidance_shift_loi,
        vae=vae,
        control_signal_processor=control_signal_processor,
        repa_helper=(repa_helper if sara_helper is None else None),
        sft_alignment_helper=sft_alignment_helper,
        moalign_helper=moalign_helper,
        semfeat_helper=semfeat_helper,
        bfm_conditioning_helper=training_core.bfm_conditioning_helper,
        reg_helper=reg_helper,
        reg_cls_pred=reg_cls_pred_loi,
        reg_cls_target=reg_cls_target,
        sara_helper=sara_helper,
        layer_sync_helper=layer_sync_helper,
        crepa_helper=crepa_helper,
        internal_guidance_helper=getattr(training_core, "internal_guidance_helper", None),
        haste_helper=haste_helper,
        contrastive_attention_helper=contrastive_attention_helper,
        raft=getattr(training_core, "raft", None),
        warp_fn=getattr(training_core, "warp", None),
        adaptive_manager=training_core.adaptive_manager,
        transition_loss_context=transition_ctx,
        global_step=global_step,
        **(target_loi_context or {}),
    )
    training_core.semanticgen_state.apply_losses(
        args=args,
        loss_components=loss_components_loi,
        alignment_helper=training_core.semantic_alignment_helper,
    )
    loss_for_backward = enhance_loss_with_temporal_consistency(
        training_core.temporal_consistency_integration,
        base_loss=loss_components_loi.total_loss,
        model_pred=model_pred_loi,
        target=target_loi,
        step=global_step,
    )
    accelerator.backward(loss_for_backward)
    return adapter_state
