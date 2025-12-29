"""Q-GaLore config parsing helpers for Takenoko."""

from __future__ import annotations

from typing import Any, Dict

import argparse


Q_GALORE_OPTIMIZER_ARGS_EXAMPLE = [
    "q_galore_rank=256",
    "q_galore_update_proj_gap=200",
    "q_galore_scale=0.25",
    "q_galore_proj_type='std'",
    "q_galore_quant=True",
    "q_galore_quant_n_bit=4",
    "q_galore_quant_group_size=256",
    "q_galore_cos_threshold=0.4",
    "q_galore_gamma_proj=2",
    "q_galore_queue_size=5",
]


def apply_q_galore_config(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger,
) -> argparse.Namespace:
    args.q_galore_target_modules = config.get("q_galore_target_modules", None)
    args.q_galore_weight_quant = bool(config.get("q_galore_weight_quant", False))
    args.q_galore_weight_bits = config.get("q_galore_weight_bits", 8)
    args.q_galore_weight_group_size = config.get("q_galore_weight_group_size", 256)
    args.q_galore_stochastic_round = bool(
        config.get("q_galore_stochastic_round", False)
    )

    if args.q_galore_target_modules is not None:
        if isinstance(args.q_galore_target_modules, str):
            if not args.q_galore_target_modules.strip():
                args.q_galore_target_modules = None
        elif isinstance(args.q_galore_target_modules, list):
            if not args.q_galore_target_modules:
                raise ValueError("q_galore_target_modules must not be an empty list")
            if not all(isinstance(item, str) for item in args.q_galore_target_modules):
                raise ValueError("q_galore_target_modules must be a list of strings")
        else:
            raise ValueError(
                "q_galore_target_modules must be a string or list of strings"
            )

    try:
        args.q_galore_weight_bits = int(args.q_galore_weight_bits)
    except (TypeError, ValueError):
        raise ValueError("q_galore_weight_bits must be an int") from None
    if args.q_galore_weight_bits != 8:
        raise ValueError("q_galore_weight_bits must be 8 for Q-GaLore weights")

    try:
        args.q_galore_weight_group_size = int(args.q_galore_weight_group_size)
    except (TypeError, ValueError):
        raise ValueError("q_galore_weight_group_size must be an int") from None
    if args.q_galore_weight_group_size <= 0:
        raise ValueError("q_galore_weight_group_size must be > 0")

    if args.q_galore_weight_quant:
        logger.info(
            "Q-GaLore weight quantization enabled (bits=%d group_size=%d stochastic_round=%s)",
            args.q_galore_weight_bits,
            args.q_galore_weight_group_size,
            args.q_galore_stochastic_round,
        )

    return args
