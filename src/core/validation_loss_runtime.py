"""Shared validation-loss runtime helpers."""

import argparse
import logging
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

from common.logger import get_logger
from core.frame_aware_history_corruption_runtime import (
    maybe_apply_frame_aware_history_corruption,
)
from core.handlers.dual_head_alignment_handler import compute_dual_head_alignment_loss
from core.handlers.manifold_consensus_handler import compute_manifold_consensus_loss
from core.self_resampling_runtime import maybe_apply_self_resampling
from criteria.hfato_loss import (
    build_hfato_noisy_input,
    compute_hfato_x0_reconstruction_loss,
)
from enhancements.motion_preservation.trainer_integration import (
    apply_ewc_penalty,
    attach_motion_preservation_health_metrics,
    attach_motion_preservation_last_metrics,
)
from enhancements.differential_guidance.training_integration import (
    transform_target_with_differential_guidance,
)
from enhancements.memflow_guidance.training_integration import (
    begin_memflow_guidance_step,
    consume_memflow_guidance_loss,
    end_memflow_guidance_step,
    suspend_memflow_guidance,
)
from enhancements.reflexflow.runtime import (
    build_reflexflow_context,
    maybe_apply_reflexflow,
)
from enhancements.self_flow.noising import (
    build_self_flow_alignment_context,
    maybe_apply_self_flow_dual_timestep,
    reduce_model_timesteps_for_runtime,
)
from enhancements.slider.slider_integration import compute_slider_loss_if_enabled
from enhancements.temporal_consistency.training_integration import (
    enhance_loss_with_temporal_consistency,
)
from transition.pipeline import finalize_batch_if_enabled, prepare_transition_batch
from utils.train_utils import compute_loss_weighting_for_sd3, get_sigmas
from enhancements.ic_lora.training_integration import should_skip_ic_lora_batch

logger = get_logger(__name__, level=logging.INFO)


def prepare_validation_runtime_context(
    training_core: Any,
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer: Any,
    network: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    control_signal_processor: Optional[Any],
    vae: Optional[Any],
    global_step: Optional[int],
    network_dtype: torch.dtype,
) -> Dict[str, Any]:
    """Build the pre-forward runtime context validation should share with training."""
    active_transformer = (
        training_core.dual_model_manager.active_model
        if training_core.dual_model_manager is not None
        else transformer
    )
    controlnet = getattr(training_core, "controlnet", None)
    runtime_batch = batch
    if isinstance(batch, dict):
        runtime_batch = dict(batch)
        image_emb = runtime_batch.get("image_emb")
        if isinstance(image_emb, dict):
            runtime_batch["image_emb"] = dict(image_emb)
    latents_for_dit = latents
    weighting = None
    sigmas = get_sigmas(
        training_core.noise_scheduler,
        timesteps,
        accelerator.device,
        latents.dim(),
        latents.dtype,
        source="validation_shared",
    )
    hfato_clean_target: Optional[torch.Tensor] = None
    hfato_sigmas_snapshot: Optional[torch.Tensor] = None

    active_transition_manager = None
    need_intermediate = False
    if (
        training_core.transition_manager is not None
        and training_core.dual_model_manager is None
    ):
        active_transition_manager = training_core.transition_manager
        prepared_transition = prepare_transition_batch(
            active_transition_manager,
            accelerator,
            latents,
            noise,
        )
        if prepared_transition is not None:
            noisy_model_input = prepared_transition.noisy_latents.to(
                device=accelerator.device,
                dtype=noisy_model_input.dtype,
            )
            timesteps = prepared_transition.timesteps.to(
                device=accelerator.device,
                dtype=timesteps.dtype,
            )
            weighting = prepared_transition.weights.to(
                device=accelerator.device,
                dtype=network_dtype,
            )
            sigmas = None
            need_intermediate = prepared_transition.need_directional

    if getattr(args, "enable_hfato", False) and sigmas is not None:
        hfato_clean_target = latents
        hfato_sigmas_snapshot = sigmas
        noisy_model_input = build_hfato_noisy_input(
            clean_latents=latents,
            noise=noise,
            sigmas=sigmas,
            ratio=float(getattr(args, "hfato_downsample_ratio", 0.5)),
            mode=str(getattr(args, "hfato_interpolation", "bilinear")),
        ).to(dtype=noisy_model_input.dtype)

    if (
        training_core.error_recycling_helper is not None
        and training_core.error_recycling_helper.enabled
    ):
        training_core.error_recycling_helper.maybe_build_svi_y(
            batch=runtime_batch,
            latents=latents,
            vae=vae,
        )
        previous_error_recycling_iteration = (
            training_core.error_recycling_helper.iteration_count
        )
        try:
            (
                noise,
                latents_for_noisy,
                rebuilt_noisy,
                _error_recycling_state,
            ) = training_core.error_recycling_helper.apply_to_inputs(
                noise=noise,
                latents=latents,
                timesteps=timesteps,
                sigmas=sigmas,
                batch=runtime_batch,
                noise_scheduler=training_core.noise_scheduler,
            )
        finally:
            training_core.error_recycling_helper.iteration_count = (
                previous_error_recycling_iteration
            )
        if rebuilt_noisy is not None:
            noisy_model_input = rebuilt_noisy
            latents_for_dit = latents_for_noisy

    rollout_transformer = active_transformer

    def _predict_self_resampling_with_full_dit(
        rollout_input: torch.Tensor,
        rollout_timesteps: torch.Tensor,
    ) -> Any:
        with torch.no_grad():
            return training_core.call_dit(
                args,
                accelerator,
                rollout_transformer,
                latents_for_dit,
                runtime_batch,
                noise,
                rollout_input,
                rollout_timesteps,
                network_dtype,
                control_signal_processor,
                controlnet,
                global_step=global_step,
                reg_cls_token=None,
                apply_stable_velocity_target=False,
            )

    (
        noisy_model_input,
        _self_resampling_state,
        training_core._warned_self_resampling_eqm,
        training_core._warned_self_resampling_fast_rollout_fallback,
    ) = (noisy_model_input, None, training_core._warned_self_resampling_eqm, training_core._warned_self_resampling_fast_rollout_fallback)
    if training_core.self_resampling_helper is not None:
        previous_self_resampling_iteration = (
            training_core.self_resampling_helper.iteration_count
        )
        try:
            (
                noisy_model_input,
                _self_resampling_state,
                training_core._warned_self_resampling_eqm,
                training_core._warned_self_resampling_fast_rollout_fallback,
            ) = maybe_apply_self_resampling(
                args=args,
                accelerator=accelerator,
                self_resampling_helper=training_core.self_resampling_helper,
                noisy_model_input=noisy_model_input,
                rollout_latents=latents_for_dit,
                noise=noise,
                timesteps=timesteps,
                rollout_transformer=rollout_transformer,
                batch=runtime_batch,
                network_dtype=network_dtype,
                patch_size=training_core.config.patch_size,
                global_step=global_step,
                controlnet=controlnet,
                eqm_enabled=False,
                warned_self_resampling_eqm=training_core._warned_self_resampling_eqm,
                warned_self_resampling_fast_rollout_fallback=(
                    training_core._warned_self_resampling_fast_rollout_fallback
                ),
                bfm_conditioning_helper=training_core.bfm_conditioning_helper,
                catlvdm_corruption_helper=training_core.catlvdm_corruption_helper,
                semanticgen_state=training_core.semanticgen_state,
                semantic_conditioning_helper=getattr(
                    training_core,
                    "semantic_conditioning_helper",
                    None,
                ),
                predict_with_full_dit=_predict_self_resampling_with_full_dit,
            )
        finally:
            training_core.self_resampling_helper.iteration_count = (
                previous_self_resampling_iteration
            )
    (
        noisy_model_input,
        _frame_aware_history_corruption_state,
        training_core._warned_frame_aware_history_corruption_eqm,
    ) = maybe_apply_frame_aware_history_corruption(
        frame_aware_history_corruption_helper=(
            training_core.frame_aware_history_corruption_helper
        ),
        noisy_model_input=noisy_model_input,
        global_step=global_step,
        eqm_enabled=False,
        warned_frame_aware_history_corruption_eqm=(
            training_core._warned_frame_aware_history_corruption_eqm
        ),
    )

    (
        noisy_model_input,
        reflexflow_state,
        reflexflow_clean_noisy_input,
        training_core._warned_reflexflow_eqm,
    ) = maybe_apply_reflexflow(
        args=args,
        reflexflow_helper=training_core.reflexflow_helper,
        noisy_model_input=noisy_model_input,
        latents=latents_for_dit,
        noise=noise,
        global_step=global_step,
        eqm_enabled=False,
        warned_reflexflow_eqm=training_core._warned_reflexflow_eqm,
    )

    self_flow_context = None
    self_flow_model_timesteps = timesteps
    self_flow_runtime_timesteps = timesteps
    if (
        training_core.self_flow_helper is not None
        and bool(getattr(training_core.self_flow_helper, "dual_timestep_enabled", True))
    ):
        try:
            self_flow_context = maybe_apply_self_flow_dual_timestep(
                args=args,
                latents=latents_for_dit,
                noise=noise,
                base_timesteps=timesteps,
                patch_size=training_core.config.patch_size,
                batch=runtime_batch,
            )
            if self_flow_context is not None:
                noisy_model_input = self_flow_context.student_noisy_model_input.to(
                    device=accelerator.device,
                    dtype=noisy_model_input.dtype,
                )
                self_flow_model_timesteps = (
                    self_flow_context.student_model_timesteps.to(
                        device=accelerator.device,
                        dtype=network_dtype,
                    )
                )
                self_flow_runtime_timesteps = reduce_model_timesteps_for_runtime(
                    self_flow_model_timesteps
                ).to(device=accelerator.device, dtype=timesteps.dtype)
        except Exception as exc:
            if bool(getattr(args, "self_flow_strict_mode", True)):
                raise
            logger.warning(
                "Self-Flow dual-timestep path failed during validation; using base noising. (%s)",
                exc,
            )
    if (
        training_core.self_flow_helper is not None
        and self_flow_context is None
        and bool(
            getattr(training_core.self_flow_helper, "feature_alignment_enabled", False)
        )
    ):
        self_flow_context = build_self_flow_alignment_context(
            args=args,
            latents=latents_for_dit,
            noisy_model_input=noisy_model_input,
            model_timesteps=self_flow_model_timesteps,
            patch_size=training_core.config.patch_size,
        )

    unwrapped_net = accelerator.unwrap_model(network)
    try:
        if hasattr(unwrapped_net, "update_rank_mask_from_timesteps"):
            unwrapped_net.update_rank_mask_from_timesteps(
                self_flow_runtime_timesteps,
                max_timestep=1000,
                device=accelerator.device,
            )
        if hasattr(unwrapped_net, "set_tc_lora_runtime_condition"):
            unwrapped_net.set_tc_lora_runtime_condition(
                latents_for_dit,
                self_flow_runtime_timesteps,
                max_timestep=getattr(args, "tc_lora_timestep_max", 1000),
            )
        elif hasattr(unwrapped_net, "set_tc_lora_timestep"):
            unwrapped_net.set_tc_lora_timestep(
                self_flow_runtime_timesteps,
                max_timestep=getattr(args, "tc_lora_timestep_max", 1000),
            )
    except Exception:
        pass
    if getattr(args, "network_module", "") == "networks.mhc_lora":
        try:
            if hasattr(unwrapped_net, "set_mhc_timestep"):
                unwrapped_net.set_mhc_timestep(self_flow_runtime_timesteps)
        except Exception:
            pass

    control_signal = None
    pixels = None
    analogy_boxes = None
    if isinstance(runtime_batch, dict):
        maybe_control = runtime_batch.get("control_signal")
        if isinstance(maybe_control, torch.Tensor):
            control_signal = maybe_control.to(device=latents_for_dit.device)
        maybe_pixels = runtime_batch.get("pixels")
        if isinstance(maybe_pixels, list) and len(maybe_pixels) > 0:
            tensor_pixels = [p for p in maybe_pixels if isinstance(p, torch.Tensor)]
            if len(tensor_pixels) == len(maybe_pixels):
                pixels = torch.stack(tensor_pixels, dim=0).to(
                    device=latents_for_dit.device
                )
        elif isinstance(maybe_pixels, torch.Tensor):
            pixels = maybe_pixels.to(device=latents_for_dit.device)
        maybe_boxes = runtime_batch.get("lorweb_analogy_boxes")
        if isinstance(maybe_boxes, torch.Tensor):
            analogy_boxes = maybe_boxes.to(device=latents_for_dit.device)

    if hasattr(unwrapped_net, "set_video2lora_runtime_condition"):
        video2lora_reference = None
        if isinstance(control_signal, torch.Tensor) and control_signal.dim() == 5:
            expected_channels = int(
                getattr(
                    unwrapped_net,
                    "video2lora_reference_feature_dim",
                    control_signal.shape[1],
                )
            )
            if int(control_signal.shape[1]) == expected_channels:
                video2lora_reference = control_signal.to(dtype=latents_for_dit.dtype)
            elif vae is not None:
                reference_pixels = control_signal.to(
                    device=latents_for_dit.device,
                    dtype=getattr(vae, "dtype", control_signal.dtype),
                )
                max_val = float(reference_pixels.max().detach().cpu())
                min_val = float(reference_pixels.min().detach().cpu())
                if max_val > 1.5:
                    reference_pixels = reference_pixels / 127.5 - 1.0
                elif min_val >= 0.0 and max_val <= 1.0:
                    reference_pixels = reference_pixels * 2.0 - 1.0
                vae_device = getattr(vae, "device", latents_for_dit.device)
                restore_vae_device = None
                if str(vae_device) != str(latents_for_dit.device):
                    restore_vae_device = vae_device
                    vae.to(latents_for_dit.device)
                try:
                    encoded_reference = vae.encode(
                        [reference_pixels[idx] for idx in range(reference_pixels.shape[0])]
                    )
                finally:
                    if restore_vae_device is not None:
                        vae.to(restore_vae_device)
                if isinstance(encoded_reference, list):
                    video2lora_reference = torch.stack(encoded_reference, dim=0)
                elif isinstance(encoded_reference, torch.Tensor):
                    video2lora_reference = encoded_reference
                else:
                    raise TypeError(
                        "Video2LoRA reference encoding returned an unsupported type."
                    )
                video2lora_reference = video2lora_reference.to(
                    device=latents_for_dit.device,
                    dtype=latents_for_dit.dtype,
                )
        video2lora_ready = unwrapped_net.set_video2lora_runtime_condition(
            latents=latents_for_dit,
            control_signal=video2lora_reference,
            pixels=None,
            timesteps=self_flow_runtime_timesteps,
        )
        if not video2lora_ready:
            return {"skip_batch": True}

    try:
        if hasattr(unwrapped_net, "set_lorweb_runtime_condition"):
            unwrapped_net.set_lorweb_runtime_condition(
                latents=latents_for_dit,
                control_signal=control_signal,
                pixels=pixels,
                analogy_boxes=analogy_boxes,
                timesteps=self_flow_runtime_timesteps,
            )
    except Exception:
        pass

    if weighting is None:
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme,
            training_core.noise_scheduler,
            self_flow_runtime_timesteps,
            accelerator.device,
            network_dtype,
        )

    skip_ic_lora_batch, _ = should_skip_ic_lora_batch(
        args=args,
        batch=runtime_batch,
    )
    if skip_ic_lora_batch:
        return {"skip_batch": True}

    reg_cls_input = None
    reg_cls_target = None
    if getattr(args, "enable_reg", False) and training_core.reg_helper is not None:
        if "pixels" in runtime_batch:
            clean_pixels = torch.stack(runtime_batch["pixels"], dim=0)
            reg_cls_input, reg_cls_target = (
                training_core.reg_helper.prepare_class_token_inputs(
                    clean_pixels,
                    timesteps,
                    training_core.noise_scheduler,
                    accelerator.device,
                    network_dtype,
                )
            )

    context_override = None
    if "t5" not in runtime_batch:
        t5_keys = [k for k in runtime_batch.keys() if k.startswith("varlen_t5_")]
        if t5_keys:
            context_override = runtime_batch[t5_keys[0]]

    return {
        "active_transformer": active_transformer,
        "batch": runtime_batch,
        "controlnet": controlnet,
        "latents_for_dit": latents_for_dit,
        "noisy_model_input": noisy_model_input,
        "timesteps": timesteps,
        "weighting": weighting,
        "hfato_clean_target": hfato_clean_target,
        "hfato_sigmas_snapshot": hfato_sigmas_snapshot,
        "reflexflow_state": reflexflow_state,
        "reflexflow_clean_noisy_input": reflexflow_clean_noisy_input,
        "self_flow_context": self_flow_context,
        "self_flow_model_timesteps": self_flow_model_timesteps,
        "self_flow_runtime_timesteps": self_flow_runtime_timesteps,
        "active_transition_manager": active_transition_manager,
        "need_intermediate": need_intermediate,
        "reg_cls_input": reg_cls_input,
        "reg_cls_target": reg_cls_target,
        "unwrapped_net": unwrapped_net,
        "context_override": context_override,
    }


def run_validation_model_forward(
    training_core: Any,
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    network: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    control_signal_processor: Optional[Any],
    global_step: Optional[int],
    network_dtype: torch.dtype,
    runtime: Dict[str, Any],
) -> Optional[Any]:
    """Run the validation forward pass after runtime context is prepared."""
    from enhancements.contrastive_attention.training_integration import (
        apply_concept_multiplier,
        apply_latent_update,
        begin_contrastive_step,
        restore_concept_multiplier,
        run_extra_prompt_passes,
    )

    active_transformer = runtime["active_transformer"]
    runtime_batch = runtime["batch"]
    latents_for_dit = runtime["latents_for_dit"]
    noisy_model_input = runtime["noisy_model_input"]
    timesteps = runtime["timesteps"]
    unwrapped_net = runtime["unwrapped_net"]
    controlnet = runtime["controlnet"]
    self_flow_model_timesteps = runtime["self_flow_model_timesteps"]
    context_override = runtime["context_override"]
    reg_cls_input = runtime["reg_cls_input"]
    prev_multiplier, multiplier_applied = apply_concept_multiplier(
        network=unwrapped_net,
        args=args,
        batch=runtime_batch,
    )

    if (
        training_core.memflow_guidance_config is not None
        and training_core.memflow_guidance_config.enable_memflow_guidance
    ):
        begin_memflow_guidance_step()
        training_core._validation_memflow_step_started = True

    if (
        training_core.contrastive_attention_helper is not None
        and global_step is not None
    ):

        def _call_dit_fn(
            model_input: torch.Tensor,
            context_override_local: Optional[Any] = None,
        ) -> Any:
            return training_core.call_dit(
                args,
                accelerator,
                active_transformer,
                latents_for_dit,
                runtime_batch,
                noise,
                model_input,
                timesteps,
                network_dtype,
                control_signal_processor=control_signal_processor,
                controlnet=controlnet,
                global_step=global_step,
                reg_cls_token=None,
                model_timesteps_override=self_flow_model_timesteps,
                context_override=context_override_local,
                apply_stable_velocity_target=False,
            )

        noisy_model_input, did_latent_update = apply_latent_update(
            helper=training_core.contrastive_attention_helper,
            args=args,
            batch=runtime_batch,
            call_dit_fn=_call_dit_fn,
            noisy_model_input=noisy_model_input,
            accelerator=accelerator,
            global_step=global_step,
        )
        if did_latent_update:
            begin_contrastive_step(
                training_core.contrastive_attention_helper,
                runtime_batch,
                global_step,
            )
        run_extra_prompt_passes(
            helper=training_core.contrastive_attention_helper,
            args=args,
            batch=runtime_batch,
            call_dit_fn=_call_dit_fn,
            noisy_model_input=noisy_model_input,
            accelerator=accelerator,
        )
        runtime["noisy_model_input"] = noisy_model_input

    manifold_consensus_helper = training_core.manifold_consensus_helper
    if manifold_consensus_helper is not None:
        try:
            if manifold_consensus_helper.prepare_step(global_step, timesteps):
                manifold_consensus_helper.begin_student_capture()
        except Exception as exc:
            logger.warning(
                "Manifold consensus prepare failed during validation; skipping its auxiliary loss. (%s)",
                exc,
            )
            manifold_consensus_helper.reset_step_state()
            manifold_consensus_helper = None
    runtime["manifold_consensus_helper"] = manifold_consensus_helper

    try:
        if training_core.self_flow_helper is not None:
            mark_student_forward = getattr(
                training_core.self_flow_helper,
                "mark_student_forward",
                None,
            )
            if callable(mark_student_forward):
                mark_student_forward()
        with accelerator.autocast():
            return training_core.call_dit(
                args,
                accelerator,
                active_transformer,
                latents_for_dit,
                runtime_batch,
                noise,
                noisy_model_input,
                timesteps,
                network_dtype,
                control_signal_processor,
                controlnet,
                global_step=global_step,
                reg_cls_token=reg_cls_input,
                model_timesteps_override=self_flow_model_timesteps,
                context_override=context_override,
                apply_pafm_target=False,
                update_stable_velocity_target_bank=False,
            )
    finally:
        if manifold_consensus_helper is not None:
            try:
                manifold_consensus_helper.finish_student_capture()
            except Exception:
                manifold_consensus_helper.reset_step_state()
        restore_concept_multiplier(
            network=unwrapped_net,
            previous_multiplier=prev_multiplier,
            applied=multiplier_applied,
        )


def finalize_validation_loss(
    training_core: Any,
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer: Any,
    network: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    control_signal_processor: Optional[Any],
    vae: Optional[Any],
    global_step: Optional[int],
    current_epoch: Optional[int],
    batch_step: Optional[int],
    network_dtype: torch.dtype,
    runtime: Dict[str, Any],
    model_result: Any,
) -> Dict[str, Any]:
    """Reuse the training loss assembly path for validation reporting."""
    active_transformer = runtime["active_transformer"]
    runtime_batch = runtime["batch"]
    controlnet = runtime["controlnet"]
    latents_for_dit = runtime["latents_for_dit"]
    noisy_model_input = runtime["noisy_model_input"]
    timesteps = runtime["timesteps"]
    weighting = runtime["weighting"]
    self_flow_context = runtime["self_flow_context"]
    self_flow_model_timesteps = runtime["self_flow_model_timesteps"]
    active_transition_manager = runtime["active_transition_manager"]
    need_intermediate = runtime["need_intermediate"]
    reg_cls_target = runtime["reg_cls_target"]
    reflexflow_state = runtime["reflexflow_state"]
    reflexflow_clean_noisy_input = runtime["reflexflow_clean_noisy_input"]
    manifold_consensus_helper = runtime.get("manifold_consensus_helper")

    reg_cls_pred = None
    if len(model_result) == 6:
        (
            model_pred,
            target,
            intermediate_z,
            internal_guidance_pred,
            internal_guidance_shift,
            reg_cls_pred,
        ) = model_result
    else:
        (
            model_pred,
            target,
            intermediate_z,
            internal_guidance_pred,
            internal_guidance_shift,
        ) = model_result

    def _get_reflexflow_clean_prediction() -> Optional[torch.Tensor]:
        saved_semantic_kl = training_core.semanticgen_state.kl_loss
        saved_semantic_tokens = training_core.semanticgen_state.tokens_for_alignment
        saved_mixflow_stats = training_core._last_mixflow_stats
        try:
            with torch.no_grad():
                clean_result = training_core.call_dit(
                    args,
                    accelerator,
                    active_transformer,
                    latents_for_dit,
                    runtime_batch,
                    noise,
                    reflexflow_clean_noisy_input,
                    timesteps,
                    network_dtype,
                    control_signal_processor,
                    controlnet,
                    global_step=global_step,
                    reg_cls_token=None,
                    model_timesteps_override=self_flow_model_timesteps,
                    apply_stable_velocity_target=False,
                )
            if (
                isinstance(clean_result, (tuple, list))
                and len(clean_result) > 0
                and torch.is_tensor(clean_result[0])
            ):
                return clean_result[0].detach()
            return None
        finally:
            training_core.semanticgen_state.kl_loss = saved_semantic_kl
            training_core.semanticgen_state.tokens_for_alignment = (
                saved_semantic_tokens
            )
            training_core._last_mixflow_stats = saved_mixflow_stats

    (
        reflexflow_context,
        training_core._warned_reflexflow_clean_pred,
    ) = build_reflexflow_context(
        args=args,
        reflexflow_state=reflexflow_state,
        reflexflow_clean_noisy_input=reflexflow_clean_noisy_input,
        get_clean_prediction=_get_reflexflow_clean_prediction,
        warned_reflexflow_clean_pred=training_core._warned_reflexflow_clean_pred,
    )

    transition_loss_context = {"transition_training_enabled": False}
    if active_transition_manager is not None:

        def _transition_forward(
            model_input: torch.Tensor,
            step_tensor: torch.Tensor,
        ) -> torch.Tensor:
            local_ctx = active_transition_manager.lora_modulation_context(
                accelerator.unwrap_model(network),
                step_tensor,
                step_tensor,
            )
            with local_ctx:
                with suspend_memflow_guidance():
                    call_result = training_core.call_dit(
                        args,
                        accelerator,
                        active_transformer,
                        latents_for_dit,
                        runtime_batch,
                        noise,
                        model_input,
                        step_tensor,
                        network_dtype,
                        control_signal_processor,
                        controlnet,
                        global_step=global_step,
                        apply_stable_velocity_target=False,
                        return_intermediate=False,
                        override_target=True,
                    )
            return call_result[0]

        (
            transition_target,
            transition_weighting,
            _transition_metrics,
            transition_directional_loss,
            per_sample_transition_loss,
        ) = finalize_batch_if_enabled(
            active_transition_manager,
            accelerator,
            model_pred,
            _transition_forward,
            accelerator.unwrap_model(active_transformer),
            need_intermediate,
            intermediate_z,
        )
        if transition_target is not None:
            target = transition_target
            if transition_weighting is not None:
                weighting = transition_weighting
            transition_loss_context = {
                "transition_training_enabled": True,
                "per_sample_loss": per_sample_transition_loss,
                "directional_loss": transition_directional_loss,
                "directional_weight": getattr(
                    training_core.transition_manager,
                    "directional_weight",
                    0.0,
                ),
            }

    context_memory_loss, _ = training_core.context_memory_manager.process_training_step(
        latents=latents,
        global_step=int(global_step or 0),
        step=int(batch_step or 0),
        args=args,
        accelerator=accelerator,
        temporal_consistency_loss_fn=training_core.context_memory_manager.create_temporal_consistency_loss_fn(),
        batch=runtime_batch,
        update_memory=False,
    )

    target = transform_target_with_differential_guidance(
        training_core.differential_guidance_integration,
        target=target,
        model_pred=model_pred,
        step=global_step,
    )

    loss_components = compute_slider_loss_if_enabled(
        loss_computer=training_core.loss_computer,
        transformer=active_transformer,
        network=network,
        noisy_latents=noisy_model_input,
        timesteps=timesteps,
        batch=runtime_batch,
        noise=noise,
        noise_scheduler=training_core.noise_scheduler,
        args=args,
        accelerator=accelerator,
        latents=latents,
        network_dtype=network_dtype,
        model_pred=model_pred,
        target=target,
        weighting=weighting,
        intermediate_z=intermediate_z,
        internal_guidance_pred=internal_guidance_pred,
        internal_guidance_shift=internal_guidance_shift,
        vae=vae,
        control_signal_processor=control_signal_processor,
        repa_helper=(
            training_core.repa_helper if training_core.sara_helper is None else None
        ),
        sft_alignment_helper=training_core.sft_alignment_helper,
        det_motion_helper=training_core.det_motion_helper,
        moalign_helper=training_core.moalign_helper,
        semfeat_helper=training_core.semfeat_helper,
        bfm_conditioning_helper=training_core.bfm_conditioning_helper,
        reg_helper=training_core.reg_helper,
        reg_cls_pred=reg_cls_pred,
        reg_cls_target=reg_cls_target,
        sara_helper=training_core.sara_helper,
        layer_sync_helper=training_core.layer_sync_helper,
        crepa_helper=training_core.crepa_helper,
        flowc2s_transport_helper=training_core.flowc2s_transport_helper,
        internal_guidance_helper=training_core.internal_guidance_helper,
        self_transcendence_helper=training_core.self_transcendence_helper,
        self_flow_helper=training_core.self_flow_helper,
        self_flow_context=self_flow_context,
        motion_preservation_helper=training_core.motion_preservation_helper,
        drifting_helper=getattr(training_core, "drifting_helper", None),
        haste_helper=training_core.haste_helper,
        contrastive_attention_helper=training_core.contrastive_attention_helper,
        raft=getattr(training_core, "raft", None),
        warp_fn=getattr(training_core, "warp", None),
        adaptive_manager=training_core.adaptive_manager,
        transition_loss_context=transition_loss_context,
        reflexflow_context=reflexflow_context,
        global_step=global_step,
        current_epoch=current_epoch,
        validation_mode=True,
    )
    training_core.semanticgen_state.apply_losses(
        args=args,
        loss_components=loss_components,
        alignment_helper=training_core.semantic_alignment_helper,
    )
    training_core._attach_stable_velocity_target_metrics(loss_components)
    training_core._attach_flexam_metrics(loss_components)

    hfato_clean_target = runtime["hfato_clean_target"]
    hfato_sigmas_snapshot = runtime["hfato_sigmas_snapshot"]
    if (
        hfato_clean_target is not None
        and hfato_sigmas_snapshot is not None
        and model_pred is not None
    ):
        try:
            hfato_base = compute_hfato_x0_reconstruction_loss(
                noisy_model_input=noisy_model_input,
                sigmas=hfato_sigmas_snapshot,
                model_pred=model_pred,
                target_clean=hfato_clean_target,
            )
            hfato_w = max(0.0, min(1.0, float(getattr(args, "hfato_weight", 1.0))))
            prev_base = loss_components.base_loss
            if hfato_w >= 1.0 or prev_base is None:
                new_base = hfato_base
            else:
                new_base = (1.0 - hfato_w) * prev_base + hfato_w * hfato_base
            if prev_base is not None:
                loss_components.total_loss = (
                    loss_components.total_loss - prev_base + new_base
                )
            else:
                loss_components.total_loss = loss_components.total_loss + new_base
            loss_components.base_loss = new_base
            loss_components.hfato_loss = hfato_base.detach()
        except Exception as exc:
            logger.warning("HFATO validation loss override failed: %s", exc)

    if manifold_consensus_helper is not None:
        try:
            (
                manifold_loss,
                manifold_metrics,
                training_core._last_mixflow_stats,
            ) = compute_manifold_consensus_loss(
                args=args,
                accelerator=accelerator,
                manifold_consensus_helper=manifold_consensus_helper,
                global_step=int(global_step or 0),
                active_transformer=active_transformer,
                latents_for_dit=latents_for_dit,
                batch=runtime_batch,
                noise=noise,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                network_dtype=network_dtype,
                control_signal_processor=control_signal_processor,
                controlnet=controlnet,
                noise_scheduler=training_core.noise_scheduler,
                model_pred=model_pred,
                weighting=weighting,
                call_dit_fn=training_core.call_dit,
                semanticgen_state=training_core.semanticgen_state,
                last_mixflow_stats=training_core._last_mixflow_stats,
                model_timesteps_override=self_flow_model_timesteps,
            )
            if manifold_loss is not None:
                loss_components.total_loss = loss_components.total_loss + manifold_loss
                loss_components.manifold_consensus_loss = manifold_loss.detach()
                loss_components.manifold_consensus_metrics = manifold_metrics
        finally:
            manifold_consensus_helper.reset_step_state()

    if training_core.dual_head_alignment_helper is not None:
        (
            dual_loss,
            dual_metrics,
            training_core._warned_dual_head_no_multiplier,
            training_core._warned_dual_head_teacher_failed,
            training_core._last_mixflow_stats,
        ) = compute_dual_head_alignment_loss(
            args=args,
            accelerator=accelerator,
            dual_head_alignment_helper=training_core.dual_head_alignment_helper,
            global_step=int(global_step or 0),
            unwrapped_net=runtime["unwrapped_net"],
            active_transformer=active_transformer,
            latents_for_dit=latents_for_dit,
            batch=runtime_batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=network_dtype,
            control_signal_processor=control_signal_processor,
            controlnet=controlnet,
            model_pred=model_pred,
            target=target,
            weighting=weighting,
            call_dit_fn=training_core.call_dit,
            semanticgen_state=training_core.semanticgen_state,
            warned_no_multiplier=training_core._warned_dual_head_no_multiplier,
            warned_teacher_failed=training_core._warned_dual_head_teacher_failed,
            last_mixflow_stats=training_core._last_mixflow_stats,
        )
        if dual_loss is not None:
            loss_components.total_loss = loss_components.total_loss + dual_loss
            loss_components.dual_head_alignment_loss = dual_loss.detach()
            loss_components.dual_head_metrics = dual_metrics

    if training_core.isofm_helper is not None and model_pred is not None:
        saved_semantic_kl = training_core.semanticgen_state.kl_loss
        saved_semantic_tokens = training_core.semanticgen_state.tokens_for_alignment
        saved_mixflow_stats = training_core._last_mixflow_stats

        def _predict_isofm_lookahead(
            lookahead_input: torch.Tensor,
            lookahead_timesteps: torch.Tensor,
        ) -> torch.Tensor:
            try:
                with suspend_memflow_guidance():
                    lookahead_result = training_core.call_dit(
                        args,
                        accelerator,
                        active_transformer,
                        latents_for_dit,
                        runtime_batch,
                        noise,
                        lookahead_input,
                        lookahead_timesteps,
                        network_dtype,
                        control_signal_processor,
                        controlnet,
                        global_step=global_step,
                        reg_cls_token=None,
                        model_timesteps_override=lookahead_timesteps,
                        apply_stable_velocity_target=False,
                    )
                if (
                    isinstance(lookahead_result, (tuple, list))
                    and len(lookahead_result) > 0
                    and torch.is_tensor(lookahead_result[0])
                ):
                    return lookahead_result[0].detach()
                raise RuntimeError(
                    "IsoFM lookahead forward returned no prediction tensor."
                )
            finally:
                training_core.semanticgen_state.kl_loss = saved_semantic_kl
                training_core.semanticgen_state.tokens_for_alignment = (
                    saved_semantic_tokens
                )
                training_core._last_mixflow_stats = saved_mixflow_stats

        try:
            isofm_result = training_core.isofm_helper.compute_loss(
                model_pred=model_pred,
                noisy_model_input=noisy_model_input,
                model_timesteps=self_flow_model_timesteps,
                predict_lookahead=_predict_isofm_lookahead,
                global_step=global_step,
            )
        except Exception as exc:
            logger.warning("IsoFM validation loss skipped: %s", exc)
            isofm_result = None
        if isofm_result is not None:
            weighted_isofm = training_core.isofm_helper.weight * isofm_result.loss
            loss_components.total_loss = loss_components.total_loss + weighted_isofm
            loss_components.isofm_loss = isofm_result.loss.detach()
            loss_components.isofm_weight_mean = (
                isofm_result.metrics["isofm/weight_mean"].detach()
            )
            loss_components.isofm_eps_mean = (
                isofm_result.metrics["isofm/eps_mean"].detach()
            )
            loss_components.isofm_speed_mean = (
                isofm_result.metrics["isofm/speed_mean"].detach()
            )
            loss_components.isofm_active_ratio = (
                isofm_result.metrics["isofm/active_ratio"].detach()
            )
            loss_components.isofm_timestep_mean = (
                isofm_result.metrics["isofm/timestep_mean"].detach()
            )

    if (
        training_core.memflow_guidance_config is not None
        and training_core.memflow_guidance_config.enable_memflow_guidance
    ):
        memflow_loss = consume_memflow_guidance_loss()
        if (
            memflow_loss is not None
            and training_core.memflow_guidance_config.memflow_guidance_weight > 0.0
        ):
            weighted_memflow = (
                memflow_loss
                * training_core.memflow_guidance_config.memflow_guidance_weight
            )
            loss_components.total_loss = loss_components.total_loss + weighted_memflow
            loss_components.memflow_guidance_loss = weighted_memflow.detach()
        end_memflow_guidance_step()
        training_core._validation_memflow_step_started = False

    training_core.context_memory_manager.integrate_context_loss(
        loss_components=loss_components,
        context_memory_loss=context_memory_loss,
        config=training_core.config.__dict__,
        accelerator=accelerator,
        global_step=int(global_step or 0),
    )

    if training_core.fvdm_manager.enabled:
        try:
            (
                fvdm_additional_loss,
                fvdm_loss_details,
            ) = training_core.fvdm_manager.get_additional_loss(
                frames=latents,
                timesteps=timesteps,
                prediction=model_pred,
            )
            if fvdm_additional_loss.item() > 0:
                loss_components.total_loss = (
                    loss_components.total_loss + fvdm_additional_loss
                )
                for key, value in fvdm_loss_details.items():
                    setattr(loss_components, f"fvdm_{key}", value)
        except Exception as exc:
            logger.warning("FVDM validation additional loss failed: %s", exc)

    if getattr(training_core, "full_finetune_ewc_helper", None) is not None:
        loss_components.total_loss = apply_ewc_penalty(
            args=args,
            ewc_helper=training_core.full_finetune_ewc_helper,
            loss=loss_components.total_loss,
            loss_components=loss_components,
            model=active_transformer,
            accelerator=accelerator,
        )

    full_finetune_motion_helper = getattr(
        training_core,
        "full_finetune_motion_preservation_helper",
        None,
    )
    if full_finetune_motion_helper is not None:
        training_model_dtype = next(active_transformer.parameters()).dtype
        motion_preservation_loss = full_finetune_motion_helper.compute_loss(
            accelerator,
            active_transformer,
            training_model_dtype,
            global_step=int(global_step or 0),
            base_task_loss=loss_components.total_loss,
        )
        if motion_preservation_loss is not None:
            loss_components.total_loss = (
                loss_components.total_loss + motion_preservation_loss
            )
            attach_motion_preservation_last_metrics(
                loss_components,
                full_finetune_motion_helper,
            )
        attach_motion_preservation_health_metrics(
            loss_components,
            full_finetune_motion_helper,
            loss_components.total_loss,
        )

    total_loss = enhance_loss_with_temporal_consistency(
        training_core.temporal_consistency_integration,
        base_loss=loss_components.total_loss,
        model_pred=model_pred,
        target=target,
        step=global_step or 0,
    )
    return {
        "model_pred": model_pred,
        "target": target,
        "loss_components": loss_components,
        "total_loss": total_loss,
    }


def compute_validation_forward_and_loss(
    training_core: Any,
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer: Any,
    network: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    control_signal_processor: Optional[Any] = None,
    vae: Optional[Any] = None,
    global_step: Optional[int] = None,
    current_epoch: Optional[int] = None,
    batch_step: Optional[int] = None,
    network_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Run the validation forward/loss path using training-equivalent semantics."""
    previous_control_vae = None
    restore_control_vae = False
    if (
        control_signal_processor is not None
        and vae is not None
        and hasattr(control_signal_processor, "vae")
    ):
        previous_control_vae = getattr(control_signal_processor, "vae", None)
        if previous_control_vae is None:
            control_signal_processor.vae = vae
            restore_control_vae = True

    try:
        runtime = prepare_validation_runtime_context(
            training_core,
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            network=network,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            control_signal_processor=control_signal_processor,
            vae=vae,
            global_step=global_step,
            network_dtype=network_dtype,
        )
        if runtime.get("skip_batch", False):
            return {"skip_batch": True}
        model_result = run_validation_model_forward(
            training_core,
            args=args,
            accelerator=accelerator,
            network=network,
            latents=latents,
            batch=batch,
            noise=noise,
            control_signal_processor=control_signal_processor,
            global_step=global_step,
            network_dtype=network_dtype,
            runtime=runtime,
        )
        if model_result is None or model_result[0] is None:
            return {"skip_batch": True}
        return finalize_validation_loss(
            training_core,
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            network=network,
            latents=latents,
            batch=batch,
            noise=noise,
            control_signal_processor=control_signal_processor,
            vae=vae,
            global_step=global_step,
            current_epoch=current_epoch,
            batch_step=batch_step,
            network_dtype=network_dtype,
            runtime=runtime,
            model_result=model_result,
        )
    finally:
        if (
            hasattr(training_core, "_validation_memflow_step_started")
            and training_core._validation_memflow_step_started
        ):
            end_memflow_guidance_step()
            training_core._validation_memflow_step_started = False
        if restore_control_vae:
            control_signal_processor.vae = previous_control_vae
