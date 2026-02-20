"""VeRA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_VERA_MODULE = "networks.vera_wan"
_VALID_VERA_MATRIX_INITS = {"kaiming_uniform", "kaiming_normal"}


def _parse_network_args(raw_args: Any) -> List[str]:
    if isinstance(raw_args, list):
        return [str(v) for v in raw_args]
    if isinstance(raw_args, str):
        value = raw_args.strip()
        if not value:
            return []
        return [value]
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


def _parse_non_negative_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _parse_positive_float(raw: Any, key: str, default: float) -> float:
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def _parse_choice(raw: Any, key: str, allowed: set[str], default: str) -> str:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return value


def apply_vera_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # VeRA settings are read from network_args.

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    network_module = str(getattr(args, "network_module", "") or "")
    raw_enabled = net_args_map.get("vera_enabled", None)
    vera_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if vera_enabled and network_module == "networks.lora_wan":
        logger.info(
            "vera_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.vera_wan."
        )
        args.network_module = _VERA_MODULE
        network_module = _VERA_MODULE

    if network_module == _VERA_MODULE and raw_enabled is None:
        vera_enabled = True
        _upsert_network_arg(network_args, "vera_enabled", "true")
        net_args_map["vera_enabled"] = "true"
        logger.info(
            "network_module=networks.vera_wan without vera_enabled; defaulting vera_enabled=true."
        )

    if not vera_enabled:
        args.enable_vera = False
        args.network_args = network_args
        return

    if network_module != _VERA_MODULE:
        raise ValueError(
            "VeRA requires network_module='networks.vera_wan' "
            "(or set network_args vera_enabled=true with network_module='networks.lora_wan')."
        )

    args.enable_vera = True
    args.vera_projection_prng_key = _parse_non_negative_int(
        net_args_map.get("vera_projection_prng_key", 0),
        "vera_projection_prng_key",
    )
    args.vera_d_initial = _parse_positive_float(
        net_args_map.get("vera_d_initial", 1.0),
        "vera_d_initial",
        default=1.0,
    )
    args.vera_matrix_init = _parse_choice(
        net_args_map.get("vera_matrix_init", "kaiming_uniform"),
        "vera_matrix_init",
        _VALID_VERA_MATRIX_INITS,
        default="kaiming_uniform",
    )

    _upsert_network_arg(network_args, "vera_enabled", "true")
    _upsert_network_arg(
        network_args,
        "vera_projection_prng_key",
        int(args.vera_projection_prng_key),
    )
    _upsert_network_arg(
        network_args,
        "vera_d_initial",
        float(args.vera_d_initial),
    )
    _upsert_network_arg(
        network_args,
        "vera_matrix_init",
        str(args.vera_matrix_init),
    )
    args.network_args = network_args

    logger.info(
        "VeRA config enabled (projection_prng_key=%s, d_initial=%s, matrix_init=%s).",
        args.vera_projection_prng_key,
        args.vera_d_initial,
        args.vera_matrix_init,
    )
