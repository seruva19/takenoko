"""OFTv2 config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_OFTV2_MODULE = "networks.oftv2_wan"


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


def _parse_non_negative_dropout(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float in [0, 1), got {raw!r}") from exc
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{key} must be in [0, 1), got {value}")
    return value


def _parse_positive_float(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float > 0, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def apply_oftv2_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # OFTv2 settings are parsed from network_args only.

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)
    network_module = str(getattr(args, "network_module", "") or "")

    raw_enabled = net_args_map.get("oftv2_enabled", None)
    oftv2_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if oftv2_enabled and network_module == "networks.lora_wan":
        logger.info(
            "oftv2_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.oftv2_wan."
        )
        args.network_module = _OFTV2_MODULE
        network_module = _OFTV2_MODULE

    if network_module == _OFTV2_MODULE and raw_enabled is None:
        oftv2_enabled = True
        _upsert_network_arg(network_args, "oftv2_enabled", "true")
        net_args_map["oftv2_enabled"] = "true"
        logger.info(
            "network_module=networks.oftv2_wan without oftv2_enabled; defaulting oftv2_enabled=true."
        )

    if not oftv2_enabled:
        args.enable_oftv2 = False
        args.network_args = network_args
        return

    if network_module != _OFTV2_MODULE:
        raise ValueError(
            "OFTv2 requires network_module='networks.oftv2_wan' "
            "(or set network_args oftv2_enabled=true with network_module='networks.lora_wan')."
        )

    fallback_block_size = int(getattr(args, "network_dim", 32) or 32)

    args.enable_oftv2 = True
    args.train_architecture = "lora"
    args.oftv2_block_size = _parse_positive_int(
        net_args_map.get("oftv2_block_size", fallback_block_size),
        "oftv2_block_size",
    )
    args.oftv2_coft = _parse_bool(net_args_map.get("oftv2_coft", False))
    args.oftv2_eps = _parse_positive_float(
        net_args_map.get("oftv2_eps", 1e-4),
        "oftv2_eps",
    )
    args.oftv2_block_share = _parse_bool(
        net_args_map.get("oftv2_block_share", False)
    )
    args.oftv2_use_cayley_neumann = _parse_bool(
        net_args_map.get("oftv2_use_cayley_neumann", True)
    )
    args.oftv2_num_cayley_neumann_terms = _parse_positive_int(
        net_args_map.get("oftv2_num_cayley_neumann_terms", 5),
        "oftv2_num_cayley_neumann_terms",
    )
    args.oftv2_module_dropout = _parse_non_negative_dropout(
        net_args_map.get(
            "oftv2_module_dropout",
            net_args_map.get("module_dropout", 0.0),
        ),
        "oftv2_module_dropout",
    )

    _upsert_network_arg(network_args, "oftv2_enabled", "true")
    _upsert_network_arg(
        network_args, "oftv2_block_size", int(args.oftv2_block_size)
    )
    _upsert_network_arg(
        network_args, "oftv2_coft", "true" if args.oftv2_coft else "false"
    )
    _upsert_network_arg(network_args, "oftv2_eps", float(args.oftv2_eps))
    _upsert_network_arg(
        network_args,
        "oftv2_block_share",
        "true" if args.oftv2_block_share else "false",
    )
    _upsert_network_arg(
        network_args,
        "oftv2_use_cayley_neumann",
        "true" if args.oftv2_use_cayley_neumann else "false",
    )
    _upsert_network_arg(
        network_args,
        "oftv2_num_cayley_neumann_terms",
        int(args.oftv2_num_cayley_neumann_terms),
    )
    _upsert_network_arg(
        network_args,
        "oftv2_module_dropout",
        float(args.oftv2_module_dropout),
    )
    args.network_args = network_args

    logger.info(
        "OFTv2 config enabled (block_size=%s, coft=%s, eps=%s, block_share=%s, cayley_neumann=%s, neumann_terms=%s, module_dropout=%s).",
        args.oftv2_block_size,
        bool(args.oftv2_coft),
        args.oftv2_eps,
        bool(args.oftv2_block_share),
        bool(args.oftv2_use_cayley_neumann),
        args.oftv2_num_cayley_neumann_terms,
        args.oftv2_module_dropout,
    )
