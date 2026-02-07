"""Temporal-Conditional LoRA configuration parsing helpers."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _parse_network_args(args: argparse.Namespace) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    raw_args = getattr(args, "network_args", None)
    if not isinstance(raw_args, list):
        return parsed
    for net_arg in raw_args:
        if not isinstance(net_arg, str) or "=" not in net_arg:
            continue
        key, value = net_arg.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_tc_lora_config(
    config: Dict[str, Any], args: argparse.Namespace, logger: Any
) -> None:
    """Parse and validate Temporal-Conditional LoRA settings from network_args only."""

    # Defaults (used when keys are missing from network_args).
    args.tc_lora_enabled = False
    args.tc_lora_use_timestep = True
    args.tc_lora_condition_dim = 64
    args.tc_lora_hidden_dim = 128
    args.tc_lora_modulation_scale = 1.0
    args.tc_lora_allow_sequence_condition = True
    args.tc_lora_use_motion_condition = False
    args.tc_lora_motion_include_timestep = True
    args.tc_lora_motion_frame_dim = 2
    args.tc_lora_timestep_max = 1000

    legacy_keys = [
        "tc_lora_enabled",
        "tc_lora_use_timestep",
        "tc_lora_condition_dim",
        "tc_lora_hidden_dim",
        "tc_lora_modulation_scale",
        "tc_lora_allow_sequence_condition",
        "tc_lora_use_motion_condition",
        "tc_lora_motion_include_timestep",
        "tc_lora_motion_frame_dim",
        "tc_lora_timestep_max",
    ]
    present_legacy = [k for k in legacy_keys if k in config]
    if present_legacy and getattr(args, "network_module", "") == "networks.tc_lora_wan":
        logger.warning(
            "TC-LoRA standalone TOML keys are deprecated and ignored. "
            "Use network_args entries instead. Ignored keys: %s",
            ", ".join(present_legacy),
        )

    net_args_map = _parse_network_args(args)
    args.tc_lora_enabled = _parse_bool(net_args_map.get("tc_lora_enabled", False), False)
    args.tc_lora_use_timestep = _parse_bool(
        net_args_map.get("tc_lora_use_timestep", True), True
    )
    args.tc_lora_condition_dim = _parse_int(
        net_args_map.get("tc_lora_condition_dim", 64), 64
    )
    args.tc_lora_hidden_dim = _parse_int(
        net_args_map.get("tc_lora_hidden_dim", 128), 128
    )
    args.tc_lora_modulation_scale = _parse_float(
        net_args_map.get("tc_lora_modulation_scale", 1.0), 1.0
    )
    args.tc_lora_allow_sequence_condition = _parse_bool(
        net_args_map.get("tc_lora_allow_sequence_condition", True), True
    )
    args.tc_lora_use_motion_condition = _parse_bool(
        net_args_map.get("tc_lora_use_motion_condition", False), False
    )
    args.tc_lora_motion_include_timestep = _parse_bool(
        net_args_map.get("tc_lora_motion_include_timestep", True), True
    )
    args.tc_lora_motion_frame_dim = _parse_int(
        net_args_map.get("tc_lora_motion_frame_dim", 2), 2
    )
    args.tc_lora_timestep_max = _parse_int(
        net_args_map.get("tc_lora_timestep_max", 1000), 1000
    )

    if args.tc_lora_condition_dim <= 0:
        raise ValueError("tc_lora_condition_dim must be > 0")
    if args.tc_lora_hidden_dim <= 0:
        raise ValueError("tc_lora_hidden_dim must be > 0")
    if args.tc_lora_modulation_scale < 0.0:
        raise ValueError("tc_lora_modulation_scale must be >= 0")
    if args.tc_lora_motion_frame_dim < 0:
        raise ValueError("tc_lora_motion_frame_dim must be >= 0")
    if args.tc_lora_timestep_max <= 0:
        raise ValueError("tc_lora_timestep_max must be > 0")

    if getattr(args, "network_module", "") != "networks.tc_lora_wan":
        return

    logger.info(
        "TC-LoRA module selected (enabled=%s, use_timestep=%s, use_motion=%s, cond_dim=%s, hidden_dim=%s).",
        args.tc_lora_enabled,
        args.tc_lora_use_timestep,
        args.tc_lora_use_motion_condition,
        args.tc_lora_condition_dim,
        args.tc_lora_hidden_dim,
    )
