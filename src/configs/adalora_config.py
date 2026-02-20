"""AdaLoRA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_ADALORA_MODULE = "networks.adalora_wan"


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


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_network_args_map(raw_args: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in raw_args:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _parse_positive_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _parse_non_negative_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _parse_beta(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float in [0.0, 1.0), got {raw!r}") from exc
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{key} must be in [0.0, 1.0), got {value}")
    return value


def _parse_non_negative_float(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float >= 0, got {raw!r}") from exc
    if value < 0.0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def apply_adalora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    network_module = str(getattr(args, "network_module", "") or "")
    raw_enabled = net_args_map.get("adalora_enabled", None)
    adalora_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if adalora_enabled and network_module == "networks.lora_wan":
        logger.info(
            "adalora_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.adalora_wan."
        )
        args.network_module = _ADALORA_MODULE
        network_module = _ADALORA_MODULE

    if network_module == _ADALORA_MODULE and raw_enabled is None:
        adalora_enabled = True
        _upsert_network_arg(network_args, "adalora_enabled", "true")
        net_args_map["adalora_enabled"] = "true"
        logger.info(
            "network_module=networks.adalora_wan without adalora_enabled; defaulting adalora_enabled=true."
        )

    if not adalora_enabled:
        args.enable_adalora = False
        args.network_args = network_args
        return

    if network_module != _ADALORA_MODULE:
        raise ValueError(
            "AdaLoRA requires network_module='networks.adalora_wan' "
            "(or set network_args adalora_enabled=true with network_module='networks.lora_wan')."
        )

    default_dim = int(getattr(args, "network_dim", 32))
    args.enable_adalora = True
    args.adalora_init_rank = _parse_positive_int(
        net_args_map.get("adalora_init_rank", default_dim),
        "adalora_init_rank",
    )
    args.adalora_target_rank = _parse_positive_int(
        net_args_map.get("adalora_target_rank", max(1, args.adalora_init_rank // 2)),
        "adalora_target_rank",
    )
    args.adalora_tinit = _parse_non_negative_int(
        net_args_map.get("adalora_tinit", 0),
        "adalora_tinit",
    )
    args.adalora_tfinal = _parse_non_negative_int(
        net_args_map.get("adalora_tfinal", 0),
        "adalora_tfinal",
    )
    args.adalora_delta_t = _parse_positive_int(
        net_args_map.get("adalora_delta_t", 100),
        "adalora_delta_t",
    )
    args.adalora_beta1 = _parse_beta(
        net_args_map.get("adalora_beta1", 0.85),
        "adalora_beta1",
    )
    args.adalora_beta2 = _parse_beta(
        net_args_map.get("adalora_beta2", 0.85),
        "adalora_beta2",
    )
    args.adalora_orth_reg_weight = _parse_non_negative_float(
        net_args_map.get("adalora_orth_reg_weight", 0.0),
        "adalora_orth_reg_weight",
    )

    if args.adalora_target_rank > args.adalora_init_rank:
        raise ValueError("adalora_target_rank must be <= adalora_init_rank")

    _upsert_network_arg(network_args, "adalora_enabled", "true")
    _upsert_network_arg(network_args, "adalora_init_rank", int(args.adalora_init_rank))
    _upsert_network_arg(
        network_args, "adalora_target_rank", int(args.adalora_target_rank)
    )
    _upsert_network_arg(network_args, "adalora_tinit", int(args.adalora_tinit))
    _upsert_network_arg(network_args, "adalora_tfinal", int(args.adalora_tfinal))
    _upsert_network_arg(network_args, "adalora_delta_t", int(args.adalora_delta_t))
    _upsert_network_arg(network_args, "adalora_beta1", float(args.adalora_beta1))
    _upsert_network_arg(network_args, "adalora_beta2", float(args.adalora_beta2))
    _upsert_network_arg(
        network_args,
        "adalora_orth_reg_weight",
        float(args.adalora_orth_reg_weight),
    )
    args.network_args = network_args

    logger.info(
        "AdaLoRA config enabled (init_rank=%s, target_rank=%s, tinit=%s, tfinal=%s, delta_t=%s).",
        args.adalora_init_rank,
        args.adalora_target_rank,
        args.adalora_tinit,
        args.adalora_tfinal,
        args.adalora_delta_t,
    )
