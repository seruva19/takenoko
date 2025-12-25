"""High-level training loop scaffolding for RCM distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

from common.logger import get_logger

from .buffers import RCMReplayBuffer
from .config_loader import RCMConfig
from .models import RCMModelBundle, prepare_model_bundle
from .ema import RCMEMAHelper
from core.training_inputs import prepare_standard_training_inputs
from core.training_core import TrainingCore
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from scheduling.timestep_utils import initialize_timestep_distribution
from utils.model_utils import str_to_dtype
import utils.fluxflow_augmentation as fluxflow_augmentation
from wan.configs.config import WAN_CONFIGS
from distillation.rcm_core.tokenizer import RCMTokenizer
from .timestep import (
    build_simple_edm_sde,
    draw_trigflow_time,
    draw_trigflow_time_discriminator,
)
from .conditioning import build_condition_pair
from .losses import (
    BehaviorCloningLossConfig,
    DistillationLossOutput,
    LogitsKLLossConfig,
    ReconstructionLossConfig,
    DivergenceLossConfig,
)
from .loop import execute_student_step, execute_fake_score_step, update_ema_if_needed
from .validation import validate_replay_buffer

logger = get_logger(__name__)


def _collate_payloads(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack per-sample payload tensors into batched tensors."""

    if not payloads:
        return {}

    collated: Dict[str, Any] = {}
    keys = set()
    for payload in payloads:
        keys.update(payload.keys())

    for key in keys:
        values = [payload[key] for payload in payloads if key in payload and payload[key] is not None]
        if not values:
            continue

        first = values[0]
        if isinstance(first, torch.Tensor):
            if first.ndim <= 1:
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = torch.cat(values, dim=0)
        elif isinstance(first, list):
            if not first:
                collated[key] = []
                continue
            stacked_list = []
            list_length = len(first)
            for idx in range(list_length):
                items = [
                    value[idx]
                    for value in values
                    if isinstance(value, list) and len(value) > idx and value[idx] is not None
                ]
                if not items:
                    stacked_list.append(None)
                    continue
                elem = items[0]
                if isinstance(elem, torch.Tensor):
                    if elem.ndim <= 1:
                        stacked_list.append(torch.stack(items, dim=0))
                    else:
                        stacked_list.append(torch.cat(items, dim=0))
                else:
                    stacked_list.append(items)
            collated[key] = stacked_list
        elif isinstance(first, tuple):
            tuple_length = len(first)
            stacked_tuple = []
            for idx in range(tuple_length):
                items = [
                    value[idx]
                    for value in values
                    if isinstance(value, tuple) and len(value) > idx and value[idx] is not None
                ]
                if not items:
                    stacked_tuple.append(None)
                    continue
                elem = items[0]
                if isinstance(elem, torch.Tensor):
                    if elem.ndim <= 1:
                        stacked_tuple.append(torch.stack(items, dim=0))
                    else:
                        stacked_tuple.append(torch.cat(items, dim=0))
                else:
                    stacked_tuple.append(items)
            collated[key] = tuple(stacked_tuple)
        else:
            collated[key] = values
    return collated


@dataclass(slots=True)
class RCMRunnerContext:
    """Aggregates resources passed around the training loop."""

    config: RCMConfig
    accelerator: Accelerator
    bundle: RCMModelBundle
    optimizer: Optimizer
    scheduler: Optional[_LRScheduler] = None
    replay_buffer: Optional[RCMReplayBuffer] = None
    optimizer_train_fn: Optional[Callable[[], None]] = None
    optimizer_eval_fn: Optional[Callable[[], None]] = None
    fake_score_optimizer: Optional[Optimizer] = None
    fake_score_scheduler: Optional[_LRScheduler] = None
    fake_score_optimizer_train_fn: Optional[Callable[[], None]] = None
    fake_score_optimizer_eval_fn: Optional[Callable[[], None]] = None
    ema_helper: Optional[RCMEMAHelper] = None


def run_distillation(
    *,
    accelerator: Accelerator,
    dataset: RCMReplayBuffer,
    teacher: nn.Module,
    student: nn.Module,
    teacher_ema: Optional[nn.Module] = None,
    fake_score: Optional[nn.Module] = None,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    config: RCMConfig,
    overrides: Optional[Mapping[str, object]] = None,
    checkpoint_callback: Optional[Callable[[Mapping[str, object]], None]] = None,
    optimizer_train_fn: Optional[Callable[[], None]] = None,
    optimizer_eval_fn: Optional[Callable[[], None]] = None,
    args: Optional[Any] = None,
    tokenizer: Optional[RCMTokenizer] = None,
    ema_helper: Optional[RCMEMAHelper] = None,
    fake_score_optimizer: Optional[Optimizer] = None,
    fake_score_scheduler: Optional[_LRScheduler] = None,
) -> None:
    """Execute the RCM distillation loop.

    This scaffolding method establishes the structure and resource wiring,
    while the concrete optimisation steps will be implemented in Stage 3/4.
    """

    if args is None:
        raise RuntimeError("RCM distillation requires access to the main training args namespace.")

    if bool(getattr(args, 'rcm_enabled', False)):
        validation_errors = validate_replay_buffer(dataset)
        for msg in validation_errors:
            logger.warning("RCM replay buffer validation warning: %s", msg)

    bundle = prepare_model_bundle(
        teacher=teacher,
        student=student,
        teacher_ema=teacher_ema,
        fake_score=fake_score,
        freeze_teacher=True,
    )

    ctx = RCMRunnerContext(
        config=config,
        accelerator=accelerator,
        bundle=bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        replay_buffer=dataset,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        fake_score_optimizer=fake_score_optimizer,
        fake_score_scheduler=fake_score_scheduler,
        fake_score_optimizer_train_fn=fake_score_optimizer_train_fn,
        fake_score_optimizer_eval_fn=fake_score_optimizer_eval_fn,
        ema_helper=ema_helper,
    )

    accelerator.print(
        ">> RCM distillation initialised "
        f"(variant={ctx.config.trainer_variant}, max_steps={ctx.config.max_steps}, "
        f"mixed_precision={ctx.config.mixed_precision})"
    )

    bundle.teacher.to(accelerator.device)
    bundle.freeze_teacher()
    bundle.teacher.eval()

    if bundle.teacher_ema is not None:
        bundle.teacher_ema.to(accelerator.device)
        bundle.teacher_ema.eval()

    bundle.student.to(accelerator.device)
    bundle.student.train()

    if bundle.fake_score is not None:
        bundle.fake_score.to(accelerator.device)

    # Prepare optimiser with accelerator
    student_model, optimizer = accelerator.prepare(bundle.student, optimizer)
    ctx.bundle.student = student_model
    ctx.optimizer = optimizer

    if bundle.fake_score is not None and ctx.fake_score_optimizer is not None:
        fake_model, fake_opt = accelerator.prepare(bundle.fake_score, ctx.fake_score_optimizer)
        ctx.bundle.fake_score = fake_model
        ctx.fake_score_optimizer = fake_opt

    if ctx.optimizer_train_fn:
        ctx.optimizer_train_fn()
    if ctx.fake_score_optimizer_train_fn:
        ctx.fake_score_optimizer_train_fn()

    batch_size = max(int(ctx.config.extra_args.get("batch_size", 4)), 1)
    max_steps = ctx.config.max_steps or math.ceil(len(dataset) / batch_size)

    task = getattr(args, "task", None)
    if task is None:
        raise RuntimeError("WAN task is not specified on args; cannot initialise training core.")

    if task not in WAN_CONFIGS:
        raise KeyError(f"WAN configuration '{task}' not found in WAN_CONFIGS.")

    fluxflow_config = fluxflow_augmentation.get_fluxflow_config_from_args(args)
    training_core = TrainingCore(WAN_CONFIGS[task], fluxflow_config)
    training_core.noise_scheduler = None
    initialize_timestep_distribution(args, training_core.timestep_distribution)

    discrete_shift = float(getattr(args, "discrete_flow_shift", 1.0))
    noise_scheduler = FlowMatchDiscreteScheduler(shift=discrete_shift, reverse=True, solver="euler")
    training_core.noise_scheduler = noise_scheduler

    dit_dtype = str_to_dtype(getattr(args, "dit_dtype", "float16"), default_dtype=torch.float16)

    recon_cfg = ReconstructionLossConfig(
        weight=float(ctx.config.extra_args.get("reconstruction_weight", 1.0)),
        reduction=str(ctx.config.extra_args.get("reconstruction_reduction", "mean")),
    )
    bc_cfg = BehaviorCloningLossConfig(
        weight=float(ctx.config.extra_args.get("behavior_weight", 1.0)),
        label_smoothing=float(ctx.config.extra_args.get("behavior_label_smoothing", 0.0)),
        reduction=str(ctx.config.extra_args.get("behavior_reduction", "mean")),
    )
    kl_cfg_value = float(ctx.config.extra_args.get("kl_weight", 0.0))
    kl_cfg = (
        LogitsKLLossConfig(
            weight=kl_cfg_value,
            temperature=float(ctx.config.extra_args.get("kl_temperature", 1.0)),
            reduction=str(ctx.config.extra_args.get("kl_reduction", "batchmean")),
        )
        if kl_cfg_value > 0
        else None
    )
    div_cfg = DivergenceLossConfig(
        forward_weight=float(ctx.config.extra_args.get("divergence_forward_weight", 0.0)),
        reverse_weight=float(ctx.config.extra_args.get("divergence_reverse_weight", 0.0)),
        temperature=float(ctx.config.extra_args.get("divergence_temperature", 1.0)),
        reduction=str(ctx.config.extra_args.get("divergence_reduction", "batchmean")),
    )

    log_interval = max(int(ctx.config.extra_args.get("log_interval", 10)), 1)
    checkpoint_interval = int(ctx.config.extra_args.get("checkpoint_interval", 0))

    global_step = 0
    running_loss = 0.0
    rcm_active = bool(getattr(args, "rcm_enabled", False))
    rcm_t_scaling_factor = ctx.config.extra_args.get("rcm_t_scaling_factor")
    rcm_call_kwargs = {}
    if rcm_active:
        rcm_sigma_min = float(ctx.config.extra_args.get("rcm_sigma_min", 0.0002))
        rcm_sigma_max = float(ctx.config.extra_args.get("rcm_sigma_max", 80.0))
        rcm_p_mean = float(ctx.config.extra_args.get("rcm_p_mean", -0.8))
        rcm_p_std = float(ctx.config.extra_args.get("rcm_p_std", 1.6))
        rcm_video_noise_multiplier = float(
            ctx.config.extra_args.get("rcm_video_noise_multiplier", 1.0)
        )
        rcm_timestep_shift = float(ctx.config.extra_args.get("rcm_timestep_shift", 0.0))
        rcm_sde = build_simple_edm_sde(
            sigma_min=rcm_sigma_min,
            sigma_max=rcm_sigma_max,
            p_mean=rcm_p_mean,
            p_std=rcm_p_std,
        )
        rcm_sde_d = build_simple_edm_sde(
            sigma_min=float(ctx.config.extra_args.get("rcm_sigma_min_d", rcm_sigma_min)),
            sigma_max=float(ctx.config.extra_args.get("rcm_sigma_max_d", rcm_sigma_max)),
            p_mean=float(ctx.config.extra_args.get("rcm_p_mean_d", 0.0)),
            p_std=float(ctx.config.extra_args.get("rcm_p_std_d", rcm_p_std)),
        )
        rcm_call_kwargs = {
            "rcm_mode": True,
            "rcm_t_scaling_factor": (
                float(rcm_t_scaling_factor) if rcm_t_scaling_factor is not None else None
            ),
        }
    else:
        rcm_sde = None
        rcm_sde_d = None
        rcm_video_noise_multiplier = 1.0
        rcm_timestep_shift = 0.0

    for batch_idx, (observations, actions, teacher_logits, metadata, payloads) in enumerate(
        dataset.iter_batches(batch_size=batch_size, drop_last=True)
    ):
        if batch_idx >= max_steps:
            break

        batch_payload = _collate_payloads(payloads)
        if metadata:
            modality_hint = metadata[0].get('modality', 'video')
        else:
            modality_hint = 'video'
        condition, uncondition = build_condition_pair(batch_payload, modality=modality_hint)
        if condition is not None:
            batch_payload['rcm_condition'] = condition
        if uncondition is not None:
            batch_payload['rcm_uncondition'] = uncondition
        if not batch_payload:
            logger.warning("Skipping batch %s: missing payload tensors for WAN conditioning.", batch_idx)
            continue

        latents = (
            batch_payload.get("latents")
            or batch_payload.get("vae_latents")
            or batch_payload.get("latent")
        )

        if latents is None:
            # Fallback path using simple observation-based encoder/decoder.
            obs = observations.to(accelerator.device).float()
            _, teacher_logits_out = bundle.teacher(obs)
            _, student_logits_out = ctx.bundle.student(obs)

            loss_output = compute_distillation_losses(
                student_pred=student_logits_out,
                teacher_pred=teacher_logits_out.detach(),
                flow_target=teacher_logits_out.detach(),
                weighting=None,
                recon_cfg=recon_cfg,
                bc_cfg=bc_cfg,
                kl_cfg=kl_cfg,
                div_cfg=div_cfg,
            )
        else:
            latents = latents.to(device=accelerator.device, dtype=dit_dtype)
            noise = torch.randn_like(latents, device=accelerator.device, dtype=dit_dtype)

            # Prepare WAN-standard noisy inputs and timesteps.
            noisy_model_input, timesteps, sigmas, weighting = prepare_standard_training_inputs(
                args=args,
                accelerator=accelerator,
                latents=latents,
                noise=noise,
                noise_scheduler=noise_scheduler,
                dit_dtype=dit_dtype,
                timestep_distribution=training_core.timestep_distribution,
                dual_model_manager=None,
                batch=batch_payload,
            )

            # Ensure auxiliary tensors are on the accelerator device.
            timesteps = timesteps.to(device=accelerator.device)
            if sigmas is not None:
                sigmas = sigmas.to(device=accelerator.device)
            if weighting is not None:
                weighting = weighting.to(device=accelerator.device, dtype=latents.dtype)

        rcm_call_kwargs_local = dict(rcm_call_kwargs)
        if rcm_active:
            modality = "video" if latents.dim() >= 3 and latents.shape[2] > 1 else "image"
            trigflow_times = draw_trigflow_time(
                batch_size=latents.shape[0],
                condition_modality=modality,
                sde=rcm_sde,
                device=accelerator.device,
                video_noise_multiplier=rcm_video_noise_multiplier,
            )
            trigflow_times = trigflow_times.view(latents.shape[0], 1, 1, 1, 1).to(
                device=accelerator.device, dtype=latents.dtype
            )
            rcm_call_kwargs_local["rcm_trigflow"] = trigflow_times
            # Pre-compute critic times for later phases (stored on metadata for now).
            critic_time = draw_trigflow_time_discriminator(
                batch_size=latents.shape[0],
                condition_modality=modality,
                sde_d=rcm_sde_d,
                device=accelerator.device,
                video_noise_multiplier=rcm_video_noise_multiplier,
                timestep_shift=rcm_timestep_shift,
            )
            for meta_entry, crit_val in zip(metadata, critic_time.view(-1).tolist()):
                meta_entry.setdefault("rcm_critic_time", crit_val)

        teacher_rcm = None
        with torch.no_grad():
            teacher_call = training_core.call_dit(
                args=args,
                accelerator=accelerator,
                transformer=ctx.bundle.teacher,
                latents=latents,
                batch=batch_payload,
                noise=noise,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                network_dtype=dit_dtype,
                control_signal_processor=None,
                controlnet=None,
                **rcm_call_kwargs_local,
            )
            if isinstance(teacher_call, tuple) and len(teacher_call) == 5:
                teacher_outputs, teacher_target, _, teacher_rcm, _ = teacher_call
            elif isinstance(teacher_call, tuple) and len(teacher_call) == 4:
                teacher_outputs, teacher_target, _, teacher_rcm = teacher_call
            else:
                teacher_outputs, teacher_target, _ = teacher_call  # type: ignore[misc]
                teacher_rcm = None

        student_rcm = None
        student_call = training_core.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=ctx.bundle.student,
            latents=latents,
            batch=batch_payload,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=dit_dtype,
            control_signal_processor=None,
            controlnet=None,
            **rcm_call_kwargs_local,
        )
        if isinstance(student_call, tuple) and len(student_call) == 5:
            student_outputs, student_target, _, student_rcm, _ = student_call
        elif isinstance(student_call, tuple) and len(student_call) == 4:
            student_outputs, student_target, _, student_rcm = student_call
        else:
            student_outputs, student_target, _ = student_call  # type: ignore[misc]
            student_rcm = None

        step_metrics = execute_student_step(
            ctx=ctx,
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            teacher_target=teacher_target,
            weighting=weighting,
            recon_cfg=recon_cfg,
            bc_cfg=bc_cfg,
            kl_cfg=kl_cfg,
            div_cfg=div_cfg,
            student_rcm=student_rcm,
            teacher_rcm=teacher_rcm,
        )

        loss_output = step_metrics.loss_output
        loss_output.metrics.update(step_metrics.metrics)

        ctx.optimizer.zero_grad(set_to_none=True)
        if step_metrics.fake_loss_active and ctx.fake_score_optimizer is not None:
            ctx.fake_score_optimizer.zero_grad(set_to_none=True)

        accelerator.backward(loss_output.total)
        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

        execute_fake_score_step(
            ctx=ctx,
            loss_output=loss_output,
            fake_loss_active=step_metrics.fake_loss_active,
        )

        update_ema_if_needed(ctx)

        loss_value = loss_output.total.detach()
        running_loss += accelerator.gather(loss_value).mean().item()
        global_step += 1

        if accelerator.is_main_process and global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            logger.info(
                "RCM step %s/%s | loss=%.6f | recon=%.6f | bc=%.6f | kl=%.6f | div_f=%.6f | div_r=%.6f",
                global_step,
                max_steps,
                avg_loss,
                loss_output.reconstruction.item(),
                loss_output.behavior_clone.item(),
                loss_output.kl_divergence.item(),
                loss_output.forward_divergence.item(),
                loss_output.reverse_divergence.item(),
            )
            running_loss = 0.0

        if checkpoint_callback and checkpoint_interval > 0:
            if global_step % checkpoint_interval == 0 or global_step == max_steps:
                unwrapped_model = accelerator.unwrap_model(ctx.bundle.student)
                student_state = unwrapped_model.state_dict()
                checkpoint_callback(
                    {
                        "step": global_step,
                        "student_state_dict": student_state,
                        "model": unwrapped_model,
                        "metadata": {"batch_idx": batch_idx, "max_steps": max_steps},
                        "epoch": 0,
                    }
                )

    accelerator.wait_for_everyone()
    if ctx.optimizer_eval_fn:
        ctx.optimizer_eval_fn()
    if ctx.fake_score_optimizer_eval_fn:
        ctx.fake_score_optimizer_eval_fn()
    logger.info("RCM distillation finished after %s steps.", global_step)
