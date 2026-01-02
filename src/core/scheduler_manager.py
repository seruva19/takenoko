"""Learning rate scheduler management for WAN network trainer."""
from typing import Any, Optional
import ast
import importlib
import argparse
import logging

import torch
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SchedulerManager:
    """Handles learning rate scheduler creation and management."""

    @staticmethod
    def is_schedulefree_optimizer(
        optimizer: torch.optim.Optimizer, args: argparse.Namespace
    ) -> bool:
        """Check if the optimizer is a schedulefree optimizer."""
        return args.optimizer_type.lower().endswith("schedulefree")

    @staticmethod
    def get_dummy_scheduler(optimizer: torch.optim.Optimizer) -> Any:
        """Get a dummy scheduler for schedulefree optimizer.

        This scheduler supports only empty step(), get_last_lr() and optimizers.
        This scheduler is used for logging only.
        This isn't wrapped by accelerator because this class is not a subclass of
        torch.optim.lr_scheduler._LRScheduler.
        """

        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self) -> None:
                return None

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    @staticmethod
    def get_lr_scheduler(
        args: argparse.Namespace, optimizer: torch.optim.Optimizer, num_processes: int
    ) -> Any:
        """Unified API to get any scheduler from its name."""
        # if schedulefree optimizer, return dummy scheduler
        if SchedulerManager.is_schedulefree_optimizer(optimizer, args):
            return SchedulerManager.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = args.max_train_steps * num_processes
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps)
            if isinstance(args.lr_warmup_steps, float)
            else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps)
            if isinstance(args.lr_decay_steps, float)
            else args.lr_decay_steps
        )

        # Fix potential None issues
        if num_warmup_steps is None:
            num_warmup_steps = 0
        if num_decay_steps is None:
            num_decay_steps = 0

        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(
                    f"{name} does not require `num_warmup_steps`. Set None or 0."
                )
            return return_vals

        # using any lr_scheduler from other library
        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type

            # Built-in aliases for custom schedulers
            alias_map = {
                # Short alias -> fully-qualified class path
                "per_cycle_cosine": "optimizers.custom_schedulers.per_cycle_cosine.PerCycleWarmupCosineWithFloor",
                "relora_jagged_cosine": "optimizers.custom_schedulers.relora_jagged_cosine.ReLoRAJaggedCosineScheduler",
                "ema_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.EMAAdaptiveScheduler",
                "noise_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.NoiseAdaptiveScheduler",
                "hybrid_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.HybridAdaptiveScheduler",
                "adaptive_per_cycle_cosine": "optimizers.custom_schedulers.adaptive_schedulers.AdaptivePerCycleWarmupCosineScheduler",
                "cycle_adaptive_per_cycle": "optimizers.custom_schedulers.adaptive_schedulers.CycleAdaptivePerCycleScheduler",
                "rex": "optimizers.custom_schedulers.rex_scheduler.RexLR",
            }

            if lr_scheduler_type in alias_map:
                fqcn = alias_map[lr_scheduler_type]
                module_path, class_name = fqcn.rsplit(".", 1)
                logger.info(
                    f"using alias '{lr_scheduler_type}' -> {fqcn} | {lr_scheduler_kwargs} as lr_scheduler"
                )
                lr_scheduler_module = importlib.import_module(module_path)
                lr_scheduler_class = getattr(lr_scheduler_module, class_name)

                # Special handling for REX scheduler to auto-populate parameters
                if lr_scheduler_type == "rex":
                    # Set default parameters if not provided
                    if "max_lr" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["max_lr"] = args.learning_rate
                    if "num_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["num_steps"] = num_training_steps
                    if "num_warmup_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["num_warmup_steps"] = num_warmup_steps
                    if (
                        "min_lr_ratio" not in lr_scheduler_kwargs
                        and "min_lr" not in lr_scheduler_kwargs
                    ):
                        lr_scheduler_kwargs["min_lr_ratio"] = (
                            min_lr_ratio if min_lr_ratio is not None else 0.01
                        )
                elif lr_scheduler_type == "relora_jagged_cosine":
                    if "first_warmup_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["first_warmup_steps"] = num_warmup_steps
                    if "max_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["max_steps"] = num_training_steps
                    if (
                        "restart_frequency" not in lr_scheduler_kwargs
                        and hasattr(args, "relora_cycle_length")
                    ):
                        lr_scheduler_kwargs["restart_frequency"] = int(
                            args.relora_cycle_length
                        )
                    if (
                        "restart_warmup_steps" not in lr_scheduler_kwargs
                        and hasattr(args, "relora_restart_warmup_steps")
                    ):
                        lr_scheduler_kwargs["restart_warmup_steps"] = int(
                            args.relora_restart_warmup_steps
                        )
                    if (
                        "min_lr_ratio" not in lr_scheduler_kwargs
                        and min_lr_ratio is not None
                        and min_lr_ratio > 0.0
                    ):
                        lr_scheduler_kwargs["min_lr_ratio"] = min_lr_ratio

                return lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

            logger.info(
                f"using {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler"
            )
            if "." not in lr_scheduler_type:  # default to use torch.optim
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert (
                type(optimizer) == transformers.optimization.Adafactor
            ), "adafactor scheduler must be used with Adafactor optimizer"
            initial_lr = float(name.split(":")[1])
            return wrap_check_needless_num_warmup_steps(
                transformers.optimization.AdafactorSchedule(optimizer, initial_lr)
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **lr_scheduler_kwargs)

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(
                schedule_func(optimizer, **lr_scheduler_kwargs)
            )

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(
                f"{name} requires `num_warmup_steps`, please provide that argument."
            )

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs
            )

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                timescale=timescale,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(
                f"{name} requires `num_training_steps`, please provide that argument."
            )

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        # these schedulers do not require `num_decay_steps`
        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_decay_steps`
        if num_decay_steps is None:
            raise ValueError(
                f"{name} requires `num_decay_steps`, please provide that argument."
            )
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        )
