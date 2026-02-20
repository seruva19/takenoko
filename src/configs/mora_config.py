"""MoRA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List, Set


_MORA_MODULE = "networks.mora_wan"
_SUPPORTED_MORA_TYPES: Set[int] = {1, 2, 3, 4, 6}


def _parse_network_args(raw_args: Any) -> List[str]:
    if isinstance(raw_args, list):
        return [str(v) for v in raw_args]
    if isinstance(raw_args, str):
        value = raw_args.strip()
        return [value] if value else []
    return []


def _parse_network_args_map(raw_args: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in raw_args:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


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


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_positive_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _parse_mora_type(raw: Any) -> int:
    parsed = _parse_positive_int(raw, "mora_type")
    if parsed not in _SUPPORTED_MORA_TYPES:
        raise ValueError(
            f"mora_type must be one of {sorted(_SUPPORTED_MORA_TYPES)}, got {parsed}"
        )
    return parsed


def apply_mora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # MoRA settings are parsed from network_args only.

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)
    network_module = str(getattr(args, "network_module", "") or "")

    raw_enabled = net_args_map.get("mora_enabled", None)
    mora_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if mora_enabled and network_module == "networks.lora_wan":
        logger.info(
            "mora_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.mora_wan."
        )
        args.network_module = _MORA_MODULE
        network_module = _MORA_MODULE

    if network_module == _MORA_MODULE and raw_enabled is None:
        mora_enabled = True
        _upsert_network_arg(network_args, "mora_enabled", "true")
        net_args_map["mora_enabled"] = "true"
        logger.info(
            "network_module=networks.mora_wan without mora_enabled; defaulting mora_enabled=true."
        )

    if not mora_enabled:
        args.enable_mora = False
        args.network_args = network_args
        return

    if network_module != _MORA_MODULE:
        raise ValueError(
            "MoRA requires network_module='networks.mora_wan' "
            "(or set network_args mora_enabled=true with network_module='networks.lora_wan')."
        )

    args.enable_mora = True
    args.mora_type = _parse_mora_type(net_args_map.get("mora_type", 1))
    args.mora_merge_chunk_size = _parse_positive_int(
        net_args_map.get("mora_merge_chunk_size", 64),
        "mora_merge_chunk_size",
    )

    _upsert_network_arg(network_args, "mora_enabled", "true")
    _upsert_network_arg(network_args, "mora_type", int(args.mora_type))
    _upsert_network_arg(
        network_args, "mora_merge_chunk_size", int(args.mora_merge_chunk_size)
    )
    args.network_args = network_args

    logger.info(
        "MoRA config enabled (mora_type=%s, merge_chunk_size=%s).",
        args.mora_type,
        args.mora_merge_chunk_size,
    )
