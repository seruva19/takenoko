"""Parallel TwinFlow runner for full WAN transformer distillation."""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import contextmanager
from ctypes import c_int
from multiprocessing import Value
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from accelerate import Accelerator

from common.logger import get_logger
from core.model_manager import ModelManager
from core.optimizer_manager import OptimizerManager
from core.scheduler_manager import SchedulerManager
from core.training_core import TrainingCore
from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from distillation.twinflow_core.config_loader import TwinFlowConfig
from distillation.twinflow_core.ema import TwinFlowEMAHelper
from distillation.twinflow_core.objective import (
    build_enhancement_mask,
    compute_rcgm_target,
    match_time_shape,
    sample_primary_time,
    sample_target_time,
)
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import collator_class, prepare_accelerator
from wan.configs.config import WAN_CONFIGS

logger = get_logger(__name__)


def run_twinflow(
    *,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    raw_config_content: str,
    config_path: str,
    config: TwinFlowConfig,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    _validate_twinflow_args(args, config)

    args.mixed_precision = config.mixed_precision
    if config.cpu_debug or config.accelerator_mode == "cpu":
        accelerator = Accelerator(cpu=True)
    else:
        accelerator = prepare_accelerator(args)
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(
                "twinflow_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=raw_config,
            )
        except Exception as exc:
            logger.warning("TwinFlow tracker initialization failed: %s", exc)

    model_manager = ModelManager()
    model_manager.handle_model_specific_args(args)
    model_config = WAN_CONFIGS[args.task]

    train_dataset_group, train_dataloader, current_epoch, current_step = _build_train_dataloader(args)
    max_steps = _resolve_max_steps(args, config, train_dataloader)
    train_dataset_group.set_max_train_steps(max_steps)
    args.max_train_steps = max_steps

    attn_mode = model_manager.get_attention_mode(args)
    transformer, dual_model_manager = model_manager.load_transformer(
        accelerator,
        args,
        args.dit,
        attn_mode,
        args.split_attn,
        accelerator.device,
        model_manager.dit_dtype,
        model_config,
    )
    if dual_model_manager is not None:
        raise NotImplementedError("TwinFlow runner does not support dual-model WAN training.")

    transformer.train()
    transformer.requires_grad_(True)

    trainable_params = [
        {
            "params": [param for param in transformer.parameters() if param.requires_grad],
            "lr": float(getattr(args, "learning_rate", 1e-4)),
        }
    ]

    optimizer_manager = OptimizerManager()
    _, _, optimizer, optimizer_train_fn, _optimizer_eval_fn = optimizer_manager.get_optimizer(
        args,
        transformer,
        trainable_params,
        accelerator=accelerator,
    )
    lr_scheduler = SchedulerManager.get_lr_scheduler(
        args,
        optimizer,
        accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    optimizer_train_fn()

    unwrapped_transformer = accelerator.unwrap_model(transformer)

    training_core = TrainingCore(model_config, model_manager.fluxflow_config)
    training_core.noise_scheduler = FlowMatchDiscreteScheduler(
        shift=float(getattr(args, "discrete_flow_shift", 1.0)),
        reverse=True,
        solver="euler",
    )

    network_dtype = unwrapped_transformer.dtype
    max_scheduler_timestep = float(
        getattr(getattr(training_core.noise_scheduler, "config", object()), "num_train_timesteps", 1000)
    )
    ema_helper = _create_ema_helper(unwrapped_transformer, config)

    log_interval = max(1, int(config.log_interval))
    checkpoint_interval = max(0, int(config.checkpoint_interval))
    global_step = 0
    num_update_steps_per_epoch = max(
        1,
        math.ceil(len(train_dataloader) / max(int(args.gradient_accumulation_steps), 1)),
    )
    num_train_epochs = max(1, math.ceil(max_steps / num_update_steps_per_epoch))

    for epoch in range(num_train_epochs):
        if global_step >= max_steps:
            break
        current_epoch.value = epoch
        for batch in train_dataloader:
            if global_step >= max_steps:
                break

            latents = batch["latents"].to(device=accelerator.device, dtype=network_dtype)
            if current_step is not None:
                current_step.value = global_step

            with accelerator.accumulate(transformer):
                args.current_step = global_step

                (
                    noise,
                    noisy_model_input,
                    sigma_values,
                ) = _prepare_twinflow_inputs(
                    latents=latents,
                    config=config,
                    network_dtype=network_dtype,
                )
                tt = sample_target_time(sigma_values, config.consistency_ratio)
                target = (noise - latents).to(dtype=network_dtype)
                rng_state = _snapshot_rng_state()

                base_pred = _predict_flow(
                    args=args,
                    accelerator=accelerator,
                    training_core=training_core,
                    transformer=transformer,
                    latents=latents,
                    batch=batch,
                    noise=noise,
                    noisy_latents=noisy_model_input,
                    sigma_values=sigma_values,
                    network_dtype=network_dtype,
                    max_scheduler_timestep=max_scheduler_timestep,
                )

                with _teacher_forward_context(
                    args=args,
                    accelerator=accelerator,
                    training_core=training_core,
                    transformer=transformer,
                    student=transformer,
                    ema_helper=ema_helper,
                    rng_state=rng_state,
                    require_ema=config.require_ema,
                    allow_student_teacher=config.allow_student_teacher,
                ) as teacher_forward:
                    target = _enhance_target_if_needed(
                        args=args,
                        accelerator=accelerator,
                        training_core=training_core,
                        transformer=transformer,
                        latents=latents,
                        batch=batch,
                        noise=noise,
                        noisy_latents=noisy_model_input,
                        sigma_values=sigma_values,
                        target=target,
                        tt=tt,
                        network_dtype=network_dtype,
                        max_scheduler_timestep=max_scheduler_timestep,
                        enhanced_ratio=config.enhanced_ratio,
                        enhanced_range=config.enhanced_range,
                        teacher_forward=teacher_forward,
                    )
                    rcgm_target = compute_rcgm_target(
                        base_pred=base_pred,
                        target=target,
                        noisy_latents=noisy_model_input,
                        sigma=sigma_values,
                        tt=tt,
                        estimate_order=config.estimate_order,
                        delta_t=config.delta_t,
                        clamp_target=config.clamp_target,
                        teacher_forward=lambda x_t, sigma_t: teacher_forward(
                            noisy_latents=x_t,
                            sigma_values=(
                                sigma_t.view(sigma_t.shape[0], -1)[:, 0]
                                if sigma_t.ndim > 1
                                else sigma_t
                            ),
                            batch=batch,
                            noise=noise,
                            latents=latents,
                            network_dtype=network_dtype,
                            max_scheduler_timestep=max_scheduler_timestep,
                        ),
                    )

                loss_base = torch.nn.functional.mse_loss(
                    base_pred.float(),
                    rcgm_target.float(),
                    reduction="mean",
                )
                loss_real = torch.nn.functional.mse_loss(
                    base_pred.float(),
                    target.float(),
                    reduction="mean",
                )
                total_loss = loss_base + float(config.real_velocity_weight) * loss_real

                loss_adv = None
                loss_rectify = None
                if config.adversarial_enabled:
                    x_fake, z = _generate_fake_samples(
                        args=args,
                        accelerator=accelerator,
                        training_core=training_core,
                        transformer=transformer,
                        batch=batch,
                        latents=latents,
                        noise=noise,
                        network_dtype=network_dtype,
                        max_scheduler_timestep=max_scheduler_timestep,
                    )
                    loss_adv = _compute_adversarial_loss(
                        args=args,
                        accelerator=accelerator,
                        training_core=training_core,
                        transformer=transformer,
                        batch=batch,
                        x_fake=x_fake,
                        z=z,
                        latents=latents,
                        noise=noise,
                        network_dtype=network_dtype,
                        max_scheduler_timestep=max_scheduler_timestep,
                    )
                    loss_rectify = _compute_rectify_loss(
                        args=args,
                        accelerator=accelerator,
                        training_core=training_core,
                        transformer=transformer,
                        batch=batch,
                        base_pred=base_pred,
                        noisy_latents=noisy_model_input,
                        sigma_values=sigma_values,
                        latents=latents,
                        noise=noise,
                        network_dtype=network_dtype,
                        max_scheduler_timestep=max_scheduler_timestep,
                    )
                    total_loss = (
                        total_loss
                        + float(config.adversarial_weight) * loss_adv
                        + float(config.rectify_weight) * loss_rectify
                    )

                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if ema_helper is not None:
                    ema_helper.update(accelerator.unwrap_model(transformer))

                logs = {
                    "twinflow/loss_total": float(total_loss.detach().item()),
                    "twinflow/loss_base": float(loss_base.detach().item()),
                    "twinflow/loss_real": float(loss_real.detach().item()),
                    "twinflow/sigma_mean": float(sigma_values.detach().float().mean().item()),
                    "twinflow/tt_mean": float(tt.detach().float().mean().item()),
                    "twinflow/enhanced_active_fraction": float(
                        build_enhancement_mask(sigma_values, config.enhanced_range)
                        .float()
                        .mean()
                        .item()
                    ),
                }
                if loss_adv is not None:
                    logs["twinflow/loss_adv"] = float(loss_adv.detach().item())
                if loss_rectify is not None:
                    logs["twinflow/loss_rectify"] = float(loss_rectify.detach().item())

                if accelerator.is_main_process and accelerator.trackers:
                    accelerator.log(logs, step=global_step)
                if accelerator.is_main_process and (global_step % log_interval == 0 or global_step == 1):
                    logger.info(
                        "TwinFlow step %s/%s | total=%.4f base=%.4f real=%.4f%s%s sigma=%.4f tt=%.4f",
                        global_step,
                        max_steps,
                        logs["twinflow/loss_total"],
                        logs["twinflow/loss_base"],
                        logs["twinflow/loss_real"],
                        "" if "twinflow/loss_adv" not in logs else f" adv={logs['twinflow/loss_adv']:.4f}",
                        "" if "twinflow/loss_rectify" not in logs else f" rectify={logs['twinflow/loss_rectify']:.4f}",
                        logs["twinflow/sigma_mean"],
                        logs["twinflow/tt_mean"],
                    )
                if (
                    checkpoint_callback is not None
                    and checkpoint_interval > 0
                    and global_step % checkpoint_interval == 0
                    and accelerator.is_main_process
                ):
                    checkpoint_callback(
                        {
                            "model": accelerator.unwrap_model(transformer),
                            "step": global_step,
                            "epoch": epoch + 1,
                        }
                    )

    if checkpoint_callback is not None and accelerator.is_main_process:
        checkpoint_callback(
            {
                "model": accelerator.unwrap_model(transformer),
                "step": global_step,
                "epoch": num_train_epochs,
            }
        )

    accelerator.wait_for_everyone()
    try:
        accelerator.end_training()
    except Exception:
        pass


def _validate_twinflow_args(args: argparse.Namespace, config: TwinFlowConfig) -> None:
    if config.trainer_variant != "full":
        raise NotImplementedError("TwinFlow currently supports only twinflow_trainer_variant='full'.")
    network_module = str(getattr(args, "network_module", "") or "")
    if network_module and network_module != "networks.wan_finetune":
        raise NotImplementedError(
            "TwinFlow full distillation expects network_module='networks.wan_finetune' or an unset network_module."
        )
    if bool(getattr(args, "enable_control_lora", False)):
        raise NotImplementedError("TwinFlow does not currently support Control LoRA.")
    if bool(getattr(args, "enable_polylora", False)):
        raise NotImplementedError("TwinFlow does not currently support PolyLoRA live training.")
    if bool(getattr(args, "enable_dual_model_training", False)):
        raise NotImplementedError("TwinFlow does not currently support dual-model WAN training.")


def _build_train_dataloader(args: argparse.Namespace):
    current_epoch = Value(c_int, 0)
    current_step = Value(c_int, 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.train_dataset_group,
        training=True,
        prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
        num_timestep_buckets=(
            None
            if getattr(args, "use_precomputed_timesteps", False)
            else getattr(args, "num_timestep_buckets", None)
        ),
        shared_epoch=current_epoch,
    )

    ds_for_collator = (
        train_dataset_group.datasets[0]
        if hasattr(train_dataset_group, "datasets") and train_dataset_group.datasets
        else train_dataset_group
    )
    collator = collator_class(current_epoch, current_step, ds_for_collator)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)
    loader_kwargs = {
        "pin_memory": bool(getattr(args, "data_loader_pin_memory", False)),
    }
    prefetch_factor = int(getattr(args, "data_loader_prefetch_factor", 0) or 0)
    if n_workers > 0 and prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=(not getattr(args, "bucket_shuffle_across_datasets", False)),
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=(bool(args.persistent_data_loader_workers) and n_workers > 0),
        **loader_kwargs,
    )
    return train_dataset_group, train_dataloader, current_epoch, current_step


def _resolve_max_steps(
    args: argparse.Namespace,
    config: TwinFlowConfig,
    train_dataloader: Iterable[Any],
) -> int:
    if config.max_steps is not None and int(config.max_steps) > 0:
        return int(config.max_steps)
    return int(getattr(args, "max_train_steps", len(train_dataloader)))


def _create_ema_helper(network: Any, config: TwinFlowConfig) -> Optional[TwinFlowEMAHelper]:
    if 0.0 < float(config.ema_decay) < 1.0:
        return TwinFlowEMAHelper.from_model(network, decay=float(config.ema_decay))
    return None


def _sigma_to_model_timesteps(
    sigma_values: torch.Tensor,
    max_scheduler_timestep: float,
) -> torch.Tensor:
    return sigma_values.abs().float().clamp(min=0.0) * max(max_scheduler_timestep, 1.0)


def _prepare_twinflow_inputs(
    *,
    latents: torch.Tensor,
    config: TwinFlowConfig,
    network_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    sigma_values = sample_primary_time(
        latents.shape[0],
        device=latents.device,
        dtype=torch.float32,
        time_dist_ctrl=config.time_dist_ctrl,
    )
    sigma_values = sigma_values.clamp_(0.0, 1.0)
    sigma_b = match_time_shape(sigma_values.to(device=latents.device, dtype=network_dtype), latents)
    noisy_model_input = sigma_b * noise + (1.0 - sigma_b) * latents
    return noise, noisy_model_input, sigma_values.detach()


def _snapshot_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            logger.debug("TwinFlow could not snapshot CUDA RNG state.", exc_info=True)
    return state


def _restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    if rng_state is None:
        return
    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    torch_cpu_state = rng_state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.random.set_rng_state(torch_cpu_state)
    torch_cuda_state = rng_state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(torch_cuda_state)
        except Exception:
            logger.debug("TwinFlow could not restore CUDA RNG state.", exc_info=True)


def _predict_flow(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigma_values: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
    context_override: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    model_timesteps = _sigma_to_model_timesteps(sigma_values, max_scheduler_timestep)
    sigma_sign = torch.sign(sigma_values)
    sigma_sign = torch.where(sigma_sign == 0, torch.ones_like(sigma_sign), sigma_sign)
    result = training_core.call_dit(
        args=args,
        accelerator=accelerator,
        transformer=transformer,
        latents=latents,
        batch=batch,
        noise=noise,
        noisy_model_input=noisy_latents,
        timesteps=model_timesteps,
        network_dtype=network_dtype,
        model_timesteps_override=model_timesteps,
        timestep_sign_override=sigma_sign,
        context_override=context_override,
        apply_stable_velocity_target=False,
    )
    return result[0]


@contextmanager
def _teacher_forward_context(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    student: Any,
    ema_helper: Optional[TwinFlowEMAHelper],
    rng_state: Optional[Dict[str, Any]],
    require_ema: bool,
    allow_student_teacher: bool,
):
    unwrapped_student = accelerator.unwrap_model(student)
    backup = None
    if ema_helper is not None:
        backup = ema_helper.apply_to(unwrapped_student)
    elif require_ema and not allow_student_teacher:
        raise ValueError(
            "TwinFlow requires EMA teacher weights unless twinflow_allow_student_teacher=true."
        )

    try:
        def _forward_fn(
            *,
            noisy_latents: torch.Tensor,
            sigma_values: torch.Tensor,
            batch: Dict[str, Any],
            noise: torch.Tensor,
            latents: torch.Tensor,
            network_dtype: torch.dtype,
            max_scheduler_timestep: float,
            context_override: Optional[list[torch.Tensor]] = None,
        ) -> torch.Tensor:
            _restore_rng_state(rng_state)
            return _teacher_forward(
                args=args,
                accelerator=accelerator,
                training_core=training_core,
                transformer=transformer,
                latents=latents,
                batch=batch,
                noise=noise,
                noisy_latents=noisy_latents,
                sigma_values=sigma_values,
                network_dtype=network_dtype,
                max_scheduler_timestep=max_scheduler_timestep,
                context_override=context_override,
            )

        yield _forward_fn
    finally:
        if ema_helper is not None and backup is not None:
            ema_helper.restore(unwrapped_student, backup)


def _enhance_target_if_needed(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigma_values: torch.Tensor,
    target: torch.Tensor,
    tt: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
    enhanced_ratio: float,
    enhanced_range: list[float],
    teacher_forward: Callable[..., torch.Tensor],
) -> torch.Tensor:
    if float(enhanced_ratio) <= 0.0:
        return target

    enhancement_mask = build_enhancement_mask(sigma_values, enhanced_range)
    if not enhancement_mask.any():
        return target

    context = batch.get("t5")
    if not isinstance(context, list):
        return target
    zero_context = [torch.zeros_like(item) for item in context]
    try:
        teacher_cond = teacher_forward(
            noisy_latents=noisy_latents,
            sigma_values=sigma_values,
            batch=batch,
            noise=noise,
            latents=latents,
            network_dtype=network_dtype,
            max_scheduler_timestep=max_scheduler_timestep,
        )
        teacher_uncond = teacher_forward(
            noisy_latents=noisy_latents,
            sigma_values=sigma_values,
            batch=batch,
            noise=noise,
            latents=latents,
            network_dtype=network_dtype,
            max_scheduler_timestep=max_scheduler_timestep,
            context_override=zero_context,
        )
    except Exception as exc:
        logger.warning("TwinFlow target enhancement failed, falling back to base target: %s", exc)
        return target
    enhanced = target + float(enhanced_ratio) * (teacher_cond - teacher_uncond)
    mask = match_time_shape(enhancement_mask.to(device=target.device), target)
    return torch.where(mask, enhanced, target)


def _teacher_forward(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigma_values: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
    context_override: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return _predict_flow(
            args=args,
            accelerator=accelerator,
            training_core=training_core,
            transformer=transformer,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_latents=noisy_latents,
            sigma_values=sigma_values,
            network_dtype=network_dtype,
            max_scheduler_timestep=max_scheduler_timestep,
            context_override=context_override,
        )


def _generate_fake_samples(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    batch: Dict[str, Any],
    latents: torch.Tensor,
    noise: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn_like(latents)
    ones = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    fake_flow = _teacher_forward(
        args=args,
        accelerator=accelerator,
        training_core=training_core,
        transformer=transformer,
        latents=z,
        batch=batch,
        noise=noise,
        noisy_latents=z,
        sigma_values=ones,
        network_dtype=network_dtype,
        max_scheduler_timestep=max_scheduler_timestep,
    )
    x_fake = z - fake_flow
    return x_fake.detach(), z


def _compute_adversarial_loss(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    batch: Dict[str, Any],
    x_fake: torch.Tensor,
    z: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
) -> torch.Tensor:
    t = torch.rand(x_fake.shape[0], device=x_fake.device, dtype=x_fake.dtype).clamp(min=0.01, max=0.99)
    t_b = t.view(t.shape[0], *([1] * (x_fake.dim() - 1)))
    x_t_fake = t_b * z + (1.0 - t_b) * x_fake
    target_fake = z - x_fake
    pred_fake = _predict_flow(
        args=args,
        accelerator=accelerator,
        training_core=training_core,
        transformer=transformer,
        latents=latents,
        batch=batch,
        noise=noise,
        noisy_latents=x_t_fake,
        sigma_values=-t,
        network_dtype=network_dtype,
        max_scheduler_timestep=max_scheduler_timestep,
    )
    return torch.nn.functional.mse_loss(
        pred_fake.float(),
        target_fake.float(),
        reduction="mean",
    )


def _compute_rectify_loss(
    *,
    args: argparse.Namespace,
    accelerator: Any,
    training_core: TrainingCore,
    transformer: Any,
    batch: Dict[str, Any],
    base_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigma_values: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    network_dtype: torch.dtype,
    max_scheduler_timestep: float,
) -> torch.Tensor:
    with torch.no_grad():
        pred_negative = _predict_flow(
            args=args,
            accelerator=accelerator,
            training_core=training_core,
            transformer=transformer,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_latents=noisy_latents,
            sigma_values=-sigma_values,
            network_dtype=network_dtype,
            max_scheduler_timestep=max_scheduler_timestep,
        )
    rectify_target = (base_pred.detach() - (pred_negative - base_pred.detach())).detach()
    return torch.nn.functional.mse_loss(
        base_pred.float(),
        rectify_target.float(),
        reduction="mean",
    )
