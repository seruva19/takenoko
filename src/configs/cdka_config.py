"""CDKA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_CDKA_MODULE = "networks.cdka_wan"


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


def apply_cdka_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    network_module = str(getattr(args, "network_module", "") or "")
    raw_enabled = net_args_map.get("cdka_enabled", None)
    cdka_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if cdka_enabled and network_module == "networks.lora_wan":
        logger.info(
            "cdka_enabled=true in network_args with network_module=networks.lora_wan; "
            "switching to networks.cdka_wan."
        )
        args.network_module = _CDKA_MODULE
        network_module = _CDKA_MODULE

    if network_module == _CDKA_MODULE and raw_enabled is None:
        cdka_enabled = True
        _upsert_network_arg(network_args, "cdka_enabled", "true")
        net_args_map["cdka_enabled"] = "true"
        logger.info(
            "network_module=networks.cdka_wan without cdka_enabled; defaulting cdka_enabled=true."
        )

    if not cdka_enabled:
        args.enable_cdka = False
        args.network_args = network_args
        return

    if network_module != _CDKA_MODULE:
        raise ValueError(
            "CDKA requires network_module='networks.cdka_wan' "
            "(or set network_args cdka_enabled=true with network_module='networks.lora_wan')."
        )

    default_rank = _parse_positive_int(getattr(args, "network_dim", 32), "network_dim")

    args.enable_cdka = True
    args.cdka_rank = _parse_positive_int(
        net_args_map.get("cdka_rank", default_rank),
        "cdka_rank",
    )
    args.cdka_r1 = _parse_positive_int(net_args_map.get("cdka_r1", 1), "cdka_r1")
    args.cdka_r2 = _parse_positive_int(net_args_map.get("cdka_r2", 1), "cdka_r2")
    args.cdka_merge_chunk_size = _parse_positive_int(
        net_args_map.get("cdka_merge_chunk_size", 64),
        "cdka_merge_chunk_size",
    )
    args.cdka_allow_padding = _parse_bool(
        net_args_map.get("cdka_allow_padding", True)
    )

    # Keep network_dim aligned with the explicit CDKA rank.
    args.network_dim = int(args.cdka_rank)

    _upsert_network_arg(network_args, "cdka_enabled", "true")
    _upsert_network_arg(network_args, "cdka_rank", int(args.cdka_rank))
    _upsert_network_arg(network_args, "cdka_r1", int(args.cdka_r1))
    _upsert_network_arg(network_args, "cdka_r2", int(args.cdka_r2))
    _upsert_network_arg(
        network_args,
        "cdka_merge_chunk_size",
        int(args.cdka_merge_chunk_size),
    )
    _upsert_network_arg(
        network_args,
        "cdka_allow_padding",
        "true" if args.cdka_allow_padding else "false",
    )
    args.network_args = network_args

    logger.info(
        "CDKA config enabled (rank=%s, r1=%s, r2=%s, allow_padding=%s, merge_chunk_size=%s).",
        args.cdka_rank,
        args.cdka_r1,
        args.cdka_r2,
        args.cdka_allow_padding,
        args.cdka_merge_chunk_size,
    )
