from __future__ import annotations

from typing import Any, Optional

import torch

from .ewc_helper import EwcHelper
from .full_ft_controls import (
    build_full_ft_optimizer_groups,
    summarize_full_ft_motion_controls,
)
from .motion_preservation_helper import MotionPreservationHelper


def register_motion_preservation_state_hooks(
    accelerator: Any,
    motion_preservation_helper: Any,
) -> None:
    def save_hook(_models, _weights, output_dir: str) -> None:
        if not accelerator.is_main_process:
            return
        motion_preservation_helper.save_runtime_state(output_dir)

    def load_hook(_models, input_dir: str) -> None:
        motion_preservation_helper.load_runtime_state(input_dir)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def create_motion_preservation_helper(
    *,
    args: Any,
    training_core: Any,
    config: Any,
    mode: str,
    logger: Any,
) -> Optional[MotionPreservationHelper]:
    if not getattr(args, "motion_preservation", False):
        return None

    mode_name = str(mode).lower()
    if mode_name == "lora":
        logger.info(
            "Motion preservation is enabled. Initializing adapter-aware replay helper."
        )
        ignored_motion_options = []
        if int(getattr(args, "freeze_early_blocks", 0) or 0) > 0:
            ignored_motion_options.append("freeze_early_blocks")
        if getattr(args, "freeze_block_indices", None):
            ignored_motion_options.append("freeze_block_indices")
        if getattr(args, "block_lr_scales", None):
            ignored_motion_options.append("block_lr_scales")
        if float(getattr(args, "non_block_lr_scale", 1.0) or 1.0) != 1.0:
            ignored_motion_options.append("non_block_lr_scale")
        if float(getattr(args, "attn_geometry_lr_scale", 1.0) or 1.0) != 1.0:
            ignored_motion_options.append("attn_geometry_lr_scale")
        if bool(getattr(args, "freeze_attn_geometry", False)):
            ignored_motion_options.append("freeze_attn_geometry")
        if float(getattr(args, "ewc_lambda", 0.0) or 0.0) > 0.0:
            ignored_motion_options.append("ewc_lambda")
        if bool(getattr(args, "motion_preservation_separate_backward", False)):
            ignored_motion_options.append("motion_preservation_separate_backward")
        if bool(getattr(args, "motion_preservation_fused_defer_step", False)):
            ignored_motion_options.append("motion_preservation_fused_defer_step")
        if ignored_motion_options:
            logger.warning(
                "LoRA motion preservation ignores full-finetune-only options: %s",
                ", ".join(ignored_motion_options),
            )
    else:
        logger.info(
            "Motion preservation is enabled. Initializing full-finetune replay helper."
        )

    return MotionPreservationHelper(training_core, args, config)


def prepare_motion_preservation_helper(
    *,
    helper: Optional[MotionPreservationHelper],
    args: Any,
    transformer: torch.nn.Module,
    accelerator: Any,
    train_dataloader: Any,
    network_dtype: torch.dtype,
    timestep_distribution: Any,
    logger: Any,
    blocks_to_swap: Optional[int] = None,
    network: Optional[Any] = None,
    use_base_model_teacher: bool = False,
    fail_on_missing_temporal: bool = False,
) -> Optional[MotionPreservationHelper]:
    if helper is None:
        setattr(args, "_motion_attention_preservation_active", False)
        setattr(args, "_motion_attention_preservation_module_count", 0)
        return None

    helper.configure_attention_modules(
        transformer,
        accelerator,
        blocks_to_swap=blocks_to_swap,
    )
    setattr(
        args,
        "_motion_attention_preservation_active",
        bool(getattr(args, "motion_attention_preservation", False)),
    )
    setattr(
        args,
        "_motion_attention_preservation_module_count",
        len(getattr(helper, "attention_modules", [])),
    )

    resolved_anchor_cache_size = helper.resolve_anchor_cache_size(len(train_dataloader))
    if resolved_anchor_cache_size > 0:
        helper.build_anchor_cache(
            accelerator,
            transformer,
            train_dataloader,
            network_dtype,
            timestep_distribution=timestep_distribution,
            network=network,
            use_base_model_teacher=use_base_model_teacher,
        )

    if not helper.has_anchors():
        logger.warning(
            "Motion preservation requested, but no anchors were built. Disabling replay loss."
        )
        setattr(args, "_motion_attention_preservation_active", False)
        setattr(args, "_motion_attention_preservation_module_count", 0)
        return None

    if bool(getattr(args, "motion_prior_require_temporal", False)):
        temporal_anchor_count = helper.count_temporal_anchors()
        if temporal_anchor_count <= 0:
            message = (
                "motion_prior_require_temporal is enabled, but no multi-frame anchors were built. "
                "Set motion_preservation_anchor_source to synthetic or hybrid."
            )
            if fail_on_missing_temporal:
                raise ValueError(message)
            logger.warning("%s Disabling motion_preservation.", message)
            setattr(args, "_motion_attention_preservation_active", False)
            setattr(args, "_motion_attention_preservation_module_count", 0)
            return None

    logger.info(
        "Motion preservation ready with %d anchors.",
        len(helper.anchor_cache),
    )
    return helper


def attach_motion_preservation_last_metrics(
    loss_components: Any,
    motion_preservation_helper: Any,
) -> None:
    loss_components.motion_preservation_loss = motion_preservation_helper.last_loss
    loss_components.motion_preservation_weight = motion_preservation_helper.last_weight
    loss_components.motion_preservation_sigma = motion_preservation_helper.last_sigma
    loss_components.motion_preservation_anchor_source = (
        motion_preservation_helper.last_anchor_source
    )
    loss_components.motion_preservation_temporal_fallback = (
        motion_preservation_helper.last_temporal_fallback
    )
    loss_components.motion_attention_preservation_loss = (
        motion_preservation_helper.last_attention_loss
    )
    loss_components.motion_preservation_anchor_frames = (
        motion_preservation_helper.last_anchor_frames
    )
    loss_components.motion_preservation_total_to_task_ratio = (
        motion_preservation_helper.last_total_to_task_ratio
    )


def attach_motion_preservation_health_metrics(
    loss_components: Any,
    motion_preservation_helper: Any,
    reference_tensor: torch.Tensor,
) -> None:
    snapshot = motion_preservation_helper.health.as_dict()
    to_tensor = lambda value: reference_tensor.detach().new_tensor(float(value))
    loss_components.motion_preservation_apply_rate = to_tensor(
        snapshot.get("apply_rate", 0.0)
    )
    loss_components.motion_preservation_schedule_skip_rate = to_tensor(
        snapshot.get("schedule_skip_rate", 0.0)
    )
    loss_components.motion_preservation_zero_weight_skip_rate = to_tensor(
        snapshot.get("zero_weight_skip_rate", 0.0)
    )
    loss_components.motion_preservation_no_anchor_skip_rate = to_tensor(
        snapshot.get("no_anchor_skip_rate", 0.0)
    )
    loss_components.motion_preservation_invalid_anchor_skip_rate = to_tensor(
        snapshot.get("invalid_anchor_skip_rate", 0.0)
    )
    loss_components.motion_preservation_temporal_fallback_rate = to_tensor(
        snapshot.get("temporal_fallback_rate", 0.0)
    )
    loss_components.motion_preservation_attention_apply_rate = to_tensor(
        snapshot.get("attention_apply_rate", 0.0)
    )
    loss_components.motion_preservation_anchor_cache_size = to_tensor(
        snapshot.get("anchor_cache_size", 0.0)
    )
    loss_components.motion_preservation_temporal_anchor_ratio = to_tensor(
        snapshot.get("temporal_anchor_ratio", 0.0)
    )
    anchor_cache_size = max(1.0, float(snapshot.get("anchor_cache_size", 0.0)))
    loss_components.motion_preservation_synthetic_anchor_ratio = to_tensor(
        float(snapshot.get("synthetic_anchor_count", 0.0)) / anchor_cache_size
    )


def attach_motion_preservation_logs(
    logs: dict[str, float],
    loss_components: Any,
) -> None:
    scalar_fields = {
        "loss/motion_preservation": "motion_preservation_loss",
        "motion_preservation/weight": "motion_preservation_weight",
        "motion_preservation/sigma": "motion_preservation_sigma",
        "motion_preservation/anchor_source_synthetic": "motion_preservation_anchor_source",
        "motion_preservation/temporal_fallback": "motion_preservation_temporal_fallback",
        "loss/motion_attention_preservation": "motion_attention_preservation_loss",
        "motion_preservation/anchor_frames": "motion_preservation_anchor_frames",
        "motion_preservation/total_to_task": "motion_preservation_total_to_task_ratio",
        "motion_preservation/apply_rate": "motion_preservation_apply_rate",
        "motion_preservation/schedule_skip_rate": "motion_preservation_schedule_skip_rate",
        "motion_preservation/zero_weight_skip_rate": "motion_preservation_zero_weight_skip_rate",
        "motion_preservation/no_anchor_skip_rate": "motion_preservation_no_anchor_skip_rate",
        "motion_preservation/invalid_anchor_skip_rate": "motion_preservation_invalid_anchor_skip_rate",
        "motion_preservation/temporal_fallback_rate": "motion_preservation_temporal_fallback_rate",
        "motion_preservation/attention_apply_rate": "motion_preservation_attention_apply_rate",
        "motion_preservation/anchor_cache_size": "motion_preservation_anchor_cache_size",
        "motion_preservation/temporal_anchor_ratio": "motion_preservation_temporal_anchor_ratio",
        "motion_preservation/synthetic_anchor_ratio": "motion_preservation_synthetic_anchor_ratio",
    }
    for log_key, attr_name in scalar_fields.items():
        value = getattr(loss_components, attr_name, None)
        if value is not None:
            logs[log_key] = float(value.item())


def create_ewc_helper(
    *,
    args: Any,
    training_core: Any,
    accelerator: Any,
    transformer: torch.nn.Module,
    train_dataloader: Any,
    optimizer: Any,
    network_dtype: torch.dtype,
    logger: Any,
    fused_step_state: Optional[dict[str, bool]] = None,
    timestep_distribution: Any = None,
) -> Optional[EwcHelper]:
    if float(getattr(args, "ewc_lambda", 0.0) or 0.0) <= 0.0:
        return None

    helper = EwcHelper(training_core, args)
    ewc_state = helper.build_or_load(
        accelerator,
        transformer,
        train_dataloader,
        optimizer,
        network_dtype,
        fused_step_state=fused_step_state,
        timestep_distribution=timestep_distribution,
    )
    if ewc_state is None:
        logger.warning(
            "EWC requested, but fisher state was not built. Disabling EWC penalty."
        )
        return None
    return helper


def apply_ewc_penalty(
    *,
    args: Any,
    ewc_helper: Optional[EwcHelper],
    loss: torch.Tensor,
    loss_components: Any,
    model: torch.nn.Module,
    accelerator: Any,
) -> torch.Tensor:
    if ewc_helper is None:
        return loss

    ewc_penalty_raw, _, _ = ewc_helper.compute_penalty(
        dtype=next(model.parameters()).dtype,
        target_device=accelerator.device,
    )
    if ewc_penalty_raw is None:
        return loss

    ewc_loss = ewc_penalty_raw * float(getattr(args, "ewc_lambda", 0.0) or 0.0)
    loss = loss + ewc_loss
    loss_components.ewc_loss = ewc_loss.detach()
    loss_components.ewc_penalty_raw = ewc_helper.last_penalty_raw
    loss_components.ewc_used_tensors = ewc_helper.last_used_tensors
    loss_components.ewc_skipped_tensors = ewc_helper.last_skipped_tensors
    return loss


def run_full_ft_motion_replay(
    *,
    args: Any,
    accelerator: Any,
    training_model: torch.nn.Module,
    motion_preservation_helper: Optional[MotionPreservationHelper],
    global_step: int,
    loss: torch.Tensor,
    loss_components: Any,
    fused_backward_pass: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], bool, bool]:
    motion_preservation_loss = None
    should_apply_motion_replay = (
        motion_preservation_helper is not None
        and motion_preservation_helper.should_apply_replay(global_step)
    )
    if motion_preservation_helper is not None and not should_apply_motion_replay:
        motion_preservation_helper.record_skip("schedule")

    separate_motion_backward = bool(
        getattr(args, "motion_preservation_separate_backward", False)
    )
    fused_defer_motion_step = bool(
        fused_backward_pass
        and bool(getattr(args, "motion_preservation_fused_defer_step", False))
        and separate_motion_backward
        and should_apply_motion_replay
    )

    if motion_preservation_helper is not None and should_apply_motion_replay:
        training_model_dtype = next(training_model.parameters()).dtype
        motion_preservation_loss = motion_preservation_helper.compute_loss(
            accelerator,
            training_model,
            training_model_dtype,
            global_step=global_step,
            base_task_loss=loss,
            skip_schedule_check=True,
        )
        if motion_preservation_loss is not None:
            attach_motion_preservation_last_metrics(
                loss_components,
                motion_preservation_helper,
            )
    if motion_preservation_helper is not None:
        attach_motion_preservation_health_metrics(
            loss_components,
            motion_preservation_helper,
            loss,
        )
    return (
        loss,
        motion_preservation_loss,
        separate_motion_backward,
        fused_defer_motion_step,
    )
