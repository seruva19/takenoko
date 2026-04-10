"""LoRA-drop network_args parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


def _parse_network_args(raw_args: Any) -> List[str]:
    if isinstance(raw_args, list):
        return [str(v) for v in raw_args]
    if isinstance(raw_args, str):
        value = raw_args.strip()
        if not value:
            return []
        return [value]
    return []


def _upsert_network_arg(network_args: List[str], key: str, value: Any) -> None:
    prefix = f"{key}="
    rendered = f"{key}={value}"
    for idx, entry in enumerate(network_args):
        if not isinstance(entry, str):
            continue
        if entry.strip().lower().startswith(prefix.lower()):
            network_args[idx] = rendered
            return
    network_args.append(rendered)


def _parse_network_args_map(raw_args: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in raw_args:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def apply_lora_drop_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse output-based LoRA-drop sharing settings from network_args only."""
    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    legacy_top_level_keys = (
        "enable_lora_drop",
        "lora_drop_importance_ema_decay",
        "lora_drop_apply_after_steps",
        "lora_drop_apply_interval",
        "lora_drop_share_fraction",
        "lora_drop_min_group_size",
    )
    if any(key in config for key in legacy_top_level_keys):
        raise ValueError(
            "LoRA-drop now reads only from network_args. "
            "Move enable_lora_drop and lora_drop_* settings into network_args."
        )

    raw_enabled = net_args_map.get("lora_drop_enabled", None)
    lora_drop_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    args.enable_lora_drop = bool(lora_drop_enabled)
    args.lora_drop_importance_ema_decay = float(
        net_args_map.get("lora_drop_importance_ema_decay", 0.95)
    )
    args.lora_drop_apply_after_steps = int(
        net_args_map.get("lora_drop_apply_after_steps", 500)
    )
    args.lora_drop_apply_interval = int(
        net_args_map.get("lora_drop_apply_interval", 500)
    )
    args.lora_drop_share_fraction = float(
        net_args_map.get("lora_drop_share_fraction", 0.25)
    )
    args.lora_drop_min_group_size = int(
        net_args_map.get("lora_drop_min_group_size", 2)
    )

    if not 0.0 <= args.lora_drop_importance_ema_decay < 1.0:
        raise ValueError("lora_drop_importance_ema_decay must be in [0, 1)")
    if args.lora_drop_apply_after_steps < 0:
        raise ValueError("lora_drop_apply_after_steps must be >= 0")
    if args.lora_drop_apply_interval < 1:
        raise ValueError("lora_drop_apply_interval must be >= 1")
    if not 0.0 <= args.lora_drop_share_fraction < 1.0:
        raise ValueError("lora_drop_share_fraction must be in [0, 1)")
    if args.lora_drop_min_group_size < 2:
        raise ValueError("lora_drop_min_group_size must be >= 2")

    if not args.enable_lora_drop:
        args.network_args = network_args
        return

    network_module = str(getattr(args, "network_module", "") or "")
    if network_module != "networks.lora_wan":
        raise ValueError(
            "LoRA-drop requires network_module='networks.lora_wan'."
        )

    _upsert_network_arg(network_args, "lora_drop_enabled", "true")
    _upsert_network_arg(
        network_args,
        "lora_drop_importance_ema_decay",
        float(args.lora_drop_importance_ema_decay),
    )
    _upsert_network_arg(
        network_args,
        "lora_drop_apply_after_steps",
        int(args.lora_drop_apply_after_steps),
    )
    _upsert_network_arg(
        network_args,
        "lora_drop_apply_interval",
        int(args.lora_drop_apply_interval),
    )
    _upsert_network_arg(
        network_args,
        "lora_drop_share_fraction",
        float(args.lora_drop_share_fraction),
    )
    _upsert_network_arg(
        network_args,
        "lora_drop_min_group_size",
        int(args.lora_drop_min_group_size),
    )
    args.network_args = network_args

    logger.info(
        "LoRA-drop enabled: ema_decay=%.3f, apply_after=%s, interval=%s, share_fraction=%.3f, min_group_size=%s.",
        args.lora_drop_importance_ema_decay,
        args.lora_drop_apply_after_steps,
        args.lora_drop_apply_interval,
        args.lora_drop_share_fraction,
        args.lora_drop_min_group_size,
    )
