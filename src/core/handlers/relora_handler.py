"""ReLoRA training hooks."""

from __future__ import annotations

import argparse
from typing import Any, Optional

from accelerate import Accelerator

from common.logger import get_logger
import torch.distributed as dist

from utils.relora_utils import optimizer_reset


logger = get_logger(__name__)


def _get_reset_params(network: Any) -> Optional[list]:
    if hasattr(network, "get_relora_params"):
        params = network.get_relora_params()
        if params:
            return list(params)
    if hasattr(network, "get_trainable_params"):
        try:
            return list(network.get_trainable_params())
        except Exception:
            return None
    return None


def _resolve_warmup_steps(
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> int:
    warmup_steps = getattr(args, "lr_warmup_steps", 0) or 0
    if isinstance(warmup_steps, float):
        max_steps = int(getattr(args, "max_train_steps", 0) or 0)
        num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
        if max_steps > 0:
            warmup_steps = int(warmup_steps * max_steps * num_processes)
        else:
            warmup_steps = int(warmup_steps)
    return int(warmup_steps)


def handle_relora_reset(
    args: argparse.Namespace,
    accelerator: Accelerator,
    network: Any,
    optimizer: Any,
    global_step: int,
) -> None:
    relora_interval = int(getattr(args, "relora_interval", 0) or 0)
    relora_cycle_length = int(getattr(args, "relora_cycle_length", 0) or 0)
    relora_start_step = int(getattr(args, "relora_start_step", 0) or 0)
    relora_adjust_step = int(getattr(args, "relora_adjust_step", 0) or 0)
    if relora_interval <= 0 and relora_cycle_length <= 0:
        return

    if not hasattr(network, "merge_and_reinit"):
        return

    update_step = global_step + 1
    effective_step = update_step + relora_adjust_step
    if effective_step < relora_start_step:
        return

    do_merge = relora_interval > 0 and effective_step % relora_interval == 0
    do_reset = relora_cycle_length > 0 and effective_step % relora_cycle_length == 0
    if not do_merge and not do_reset:
        return

    if getattr(args, "relora_skip_reset_until_warmup", False):
        warmup_steps = _resolve_warmup_steps(args, accelerator)
        if warmup_steps > 0 and update_step <= warmup_steps:
            if accelerator.is_main_process:
                logger.info(
                    "ReLoRA: skipping reset until warmup ends (step=%s <= %s)",
                    update_step,
                    warmup_steps,
                )
            return

    if accelerator.is_main_process:
        logger.info(
            "ReLoRA: reset at update step %s (effective=%s, merge=%s, opt_reset=%s)",
            update_step,
            effective_step,
            do_merge,
            do_reset,
        )

    unwrapped = accelerator.unwrap_model(network)
    reset_params = None
    if do_merge:
        unwrapped.merge_and_reinit()
        reset_params = _get_reset_params(unwrapped)
        if reset_params and dist.is_available() and dist.is_initialized():
            for param in reset_params:
                if hasattr(param, "data"):
                    dist.broadcast(param.data, src=0)

    if do_reset:
        if reset_params is None:
            reset_params = _get_reset_params(unwrapped)
        if not reset_params:
            logger.warning("ReLoRA: no parameters available for optimizer reset")
            return
        named_reset_params = None
        if getattr(args, "relora_enable_deepspeed_zero_reset", False):
            reset_param_set = {param for param in reset_params}
            named_reset_params = [
                (name, param)
                for name, param in unwrapped.named_parameters()
                if param in reset_param_set
            ]
        zeroed_ratio = optimizer_reset(
            optimizer,
            reset_params=reset_params,
            optimizer_state_keys=list(
                getattr(
                    args,
                    "relora_optimizer_state_keys",
                    ["exp_avg", "exp_avg_sq"],
                )
            ),
            reset_optimizer_on_relora=bool(
                getattr(args, "relora_reset_optimizer_on_relora", True)
            ),
            optimizer_random_pruning=float(
                getattr(args, "relora_optimizer_random_pruning", 0.0) or 0.0
            ),
            optimizer_magnitude_pruning=float(
                getattr(args, "relora_optimizer_magnitude_pruning", 0.0) or 0.0
            ),
            reset_prune_ratio=float(
                getattr(args, "relora_reset_optimizer_prune_ratio", 0.0) or 0.0
            ),
            enable_deepspeed_zero_reset=bool(
                getattr(args, "relora_enable_deepspeed_zero_reset", False)
            ),
            named_reset_params=named_reset_params,
        )
        if getattr(args, "relora_log_reset_metrics", False):
            reset_count = int(getattr(args, "_relora_reset_count", 0) or 0) + 1
            setattr(args, "_relora_reset_count", reset_count)
            if accelerator.is_main_process:
                if zeroed_ratio is None:
                    logger.info("ReLoRA: optimizer reset count=%s", reset_count)
                else:
                    logger.info(
                        "ReLoRA: optimizer reset count=%s zeroed=%.2f%%",
                        reset_count,
                        zeroed_ratio * 100.0,
                    )
            try:
                logs = {"relora/reset_count": reset_count}
                if zeroed_ratio is not None:
                    logs["relora/optimizer_zeroed"] = float(zeroed_ratio)
                accelerator.log(logs, step=global_step)
            except Exception:
                pass
