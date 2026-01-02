"""ReLoRA config parsing and validation."""

from __future__ import annotations

import ast
from typing import Any, Dict


def apply_relora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.relora_interval = int(config.get("relora_interval", 0) or 0)
    args.relora_cycle_length = int(config.get("relora_cycle_length", 0) or 0)
    args.relora_start_step = int(config.get("relora_start_step", 0) or 0)
    args.relora_adjust_step = int(config.get("relora_adjust_step", 0) or 0)
    args.relora_restart_warmup_steps = int(
        config.get("relora_restart_warmup_steps", 0) or 0
    )
    args.relora_reset_optimizer_on_relora = bool(
        config.get("relora_reset_optimizer_on_relora", True)
    )
    args.relora_optimizer_random_pruning = float(
        config.get("relora_optimizer_random_pruning", 0.0) or 0.0
    )
    args.relora_optimizer_magnitude_pruning = float(
        config.get("relora_optimizer_magnitude_pruning", 0.0) or 0.0
    )
    args.relora_reset_optimizer_prune_ratio = float(
        config.get("relora_reset_optimizer_prune_ratio", 0.0) or 0.0
    )
    args.relora_enable_deepspeed_zero_reset = bool(
        config.get("relora_enable_deepspeed_zero_reset", False)
    )
    args.relora_log_reset_metrics = bool(
        config.get("relora_log_reset_metrics", False)
    )
    args.relora_skip_reset_until_warmup = bool(
        config.get("relora_skip_reset_until_warmup", False)
    )
    relora_state_keys = config.get(
        "relora_optimizer_state_keys", ["exp_avg", "exp_avg_sq"]
    )
    if isinstance(relora_state_keys, str):
        try:
            relora_state_keys = ast.literal_eval(relora_state_keys)
        except Exception:
            relora_state_keys = [relora_state_keys]
    args.relora_optimizer_state_keys = relora_state_keys

    if (
        args.relora_interval < 0
        or args.relora_cycle_length < 0
        or args.relora_start_step < 0
        or args.relora_adjust_step < 0
        or args.relora_restart_warmup_steps < 0
    ):
        raise ValueError("ReLoRA step values must be >= 0")

    if not 0.0 <= args.relora_optimizer_random_pruning <= 1.0:
        raise ValueError("relora_optimizer_random_pruning must be in [0, 1]")
    if not 0.0 <= args.relora_optimizer_magnitude_pruning <= 1.0:
        raise ValueError("relora_optimizer_magnitude_pruning must be in [0, 1]")
    if not 0.0 <= args.relora_reset_optimizer_prune_ratio <= 1.0:
        raise ValueError("relora_reset_optimizer_prune_ratio must be in [0, 1]")
    if (
        args.relora_reset_optimizer_prune_ratio > 0.0
        and not args.relora_reset_optimizer_on_relora
    ):
        logger.warning(
            "relora_reset_optimizer_prune_ratio ignored unless relora_reset_optimizer_on_relora is enabled"
        )

    if args.relora_cycle_length > 0:
        n_reset_types = (
            int(bool(args.relora_reset_optimizer_on_relora))
            + int(bool(args.relora_optimizer_random_pruning))
            + int(bool(args.relora_optimizer_magnitude_pruning))
        )
        if n_reset_types != 1:
            raise ValueError(
                "Exactly one of relora_reset_optimizer_on_relora, "
                "relora_optimizer_random_pruning, "
                "relora_optimizer_magnitude_pruning must be set"
            )
        if not isinstance(args.relora_optimizer_state_keys, list) or not all(
            isinstance(k, str) for k in args.relora_optimizer_state_keys
        ):
            raise ValueError("relora_optimizer_state_keys must be a list of strings")
        if args.network_module == "networks.relora_wan":
            logger.info(
                "ReLoRA enabled: interval=%s, cycle_length=%s, reset=%s",
                args.relora_interval,
                args.relora_cycle_length,
                args.relora_reset_optimizer_on_relora,
            )
        if args.relora_interval <= 0:
            logger.warning(
                "ReLoRA cycle_length set but relora_interval <= 0; no merges will occur"
            )
    elif args.network_module == "networks.relora_wan" and args.relora_interval <= 0:
        logger.warning("ReLoRA network selected but relora_interval <= 0 (disabled)")

    if args.network_module == "networks.relora_wan":
        if args.relora_cycle_length <= 0:
            raise ValueError(
                "ReLoRA requires relora_cycle_length > 0 to align scheduler restarts"
            )
        if args.relora_restart_warmup_steps > args.relora_cycle_length:
            logger.warning(
                "relora_restart_warmup_steps (%s) exceeds relora_cycle_length (%s)",
                args.relora_restart_warmup_steps,
                args.relora_cycle_length,
            )

        allowed_sched = {"per_cycle_cosine", "relora_jagged_cosine"}
        if args.lr_scheduler_type and args.lr_scheduler_type not in allowed_sched:
            raise ValueError(
                "ReLoRA requires lr_scheduler_type = 'per_cycle_cosine' or 'relora_jagged_cosine'"
            )
        if not args.lr_scheduler_type:
            args.lr_scheduler_type = "per_cycle_cosine"
            logger.info("ReLoRA: defaulting lr_scheduler_type to per_cycle_cosine")

        relora_lr_args: list[str] = []
        if isinstance(args.lr_scheduler_args, str) and args.lr_scheduler_args:
            relora_lr_args = [args.lr_scheduler_args]
        elif isinstance(args.lr_scheduler_args, list):
            relora_lr_args = [str(v) for v in args.lr_scheduler_args]

        lr_arg_map: Dict[str, Any] = {}
        for arg in relora_lr_args:
            if "=" not in arg:
                continue
            key, value = arg.split("=", 1)
            key = key.strip()
            try:
                lr_arg_map[key] = ast.literal_eval(value)
            except Exception:
                lr_arg_map[key] = value

        if args.lr_scheduler_type == "per_cycle_cosine":
            if "cycle_steps" not in lr_arg_map:
                lr_arg_map["cycle_steps"] = int(args.relora_cycle_length)
            if "warmup_steps" not in lr_arg_map:
                lr_arg_map["warmup_steps"] = int(args.relora_restart_warmup_steps)
            if "decay_steps" not in lr_arg_map:
                lr_arg_map["decay_steps"] = max(
                    0, int(args.relora_cycle_length - args.relora_restart_warmup_steps)
                )
            if (
                "min_lr_ratio" not in lr_arg_map
                and args.lr_scheduler_min_lr_ratio is not None
            ):
                lr_arg_map["min_lr_ratio"] = float(args.lr_scheduler_min_lr_ratio)

            if lr_arg_map["warmup_steps"] > lr_arg_map["cycle_steps"]:
                raise ValueError(
                    "per_cycle_cosine warmup_steps must be <= cycle_steps for ReLoRA"
                )
        else:
            if args.relora_restart_warmup_steps <= 0:
                raise ValueError(
                    "relora_jagged_cosine requires relora_restart_warmup_steps > 0"
                )
            if args.lr_scheduler_min_lr_ratio is None or float(
                args.lr_scheduler_min_lr_ratio
            ) <= 0.0:
                raise ValueError(
                    "relora_jagged_cosine requires lr_scheduler_min_lr_ratio > 0"
                )

            first_warmup_steps = args.lr_warmup_steps
            if isinstance(first_warmup_steps, float):
                max_steps = int(getattr(args, "max_train_steps", 0) or 0)
                if max_steps > 0:
                    first_warmup_steps = int(first_warmup_steps * max_steps)
                else:
                    first_warmup_steps = int(first_warmup_steps)
            if int(first_warmup_steps) <= 0:
                raise ValueError(
                    "relora_jagged_cosine requires lr_warmup_steps > 0"
                )
            lr_arg_map.setdefault("first_warmup_steps", int(first_warmup_steps))
            lr_arg_map.setdefault(
                "restart_warmup_steps", int(args.relora_restart_warmup_steps)
            )
            lr_arg_map.setdefault("restart_frequency", int(args.relora_cycle_length))
            lr_arg_map.setdefault(
                "min_lr_ratio", float(args.lr_scheduler_min_lr_ratio)
            )
            if "max_steps" not in lr_arg_map:
                max_steps = int(getattr(args, "max_train_steps", 0) or 0)
                if max_steps <= 0:
                    logger.warning(
                        "relora_jagged_cosine max_steps not set; defaulting to 0"
                    )
                lr_arg_map["max_steps"] = max_steps

        args.lr_scheduler_args = [
            f"{key}={repr(value)}" for key, value in lr_arg_map.items()
        ]
