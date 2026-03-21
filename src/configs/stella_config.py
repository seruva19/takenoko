"""StelLA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_STELLA_MODULE = "networks.stella_wan"
_VALID_STELLA_RETRACTIONS = {"polar", "exp_map"}
_VALID_STELLA_INITS = {"orthonormal", "svd_major", "svd_minor", "kaiming"}
_STELLA_INIT_ALIASES = {
    "orthonormal": "orthonormal",
    "orthogonal": "orthonormal",
    "rando": "orthonormal",
    "random_qr": "orthonormal",
    "random": "orthonormal",
    "kaiming": "kaiming",
    "default": "kaiming",
    "svd_major": "svd_major",
    "svd_minor": "svd_minor",
}


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


def _parse_choice(raw: Any, key: str, allowed: set[str], default: str) -> str:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _normalize_stella_init(raw: Any) -> str:
    if raw is None:
        return "orthonormal"
    if isinstance(raw, bool):
        return "orthonormal" if raw else "kaiming"
    lowered = str(raw).strip().lower()
    if lowered == "":
        return "orthonormal"
    if lowered in {"1", "true", "yes", "y", "on"}:
        return "orthonormal"
    if lowered in {"0", "false", "no", "n", "off"}:
        return "kaiming"
    normalized = _STELLA_INIT_ALIASES.get(lowered, lowered)
    if normalized not in _VALID_STELLA_INITS:
        raise ValueError(
            f"stella_init must be one of {sorted(_VALID_STELLA_INITS)} "
            f"(aliases: {sorted(_STELLA_INIT_ALIASES.keys())}), got {raw!r}"
        )
    return normalized


def _parse_non_negative_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _parse_grad_scaling(raw: Any, key: str) -> bool | float:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value <= 0.0:
            raise ValueError(f"{key} must be > 0 when provided as float, got {value}")
        return value
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        try:
            value = float(lowered)
        except Exception as exc:
            raise ValueError(
                f"{key} must be bool or float > 0, got {raw!r}"
            ) from exc
        if value <= 0.0:
            raise ValueError(f"{key} must be > 0 when provided as float, got {value}")
        return value
    raise ValueError(f"{key} must be bool or float > 0, got {raw!r}")


def _render_grad_scaling(value: bool | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(float(value))


def apply_stella_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # StelLA settings are parsed from network_args.

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)
    network_module = str(getattr(args, "network_module", "") or "")

    raw_enabled = net_args_map.get("stella_enabled", None)
    stella_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if stella_enabled and network_module == "networks.lora_wan":
        logger.info(
            "stella_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.stella_wan."
        )
        args.network_module = _STELLA_MODULE
        network_module = _STELLA_MODULE

    if network_module == _STELLA_MODULE and raw_enabled is None:
        stella_enabled = True
        _upsert_network_arg(network_args, "stella_enabled", "true")
        net_args_map["stella_enabled"] = "true"
        logger.info(
            "network_module=networks.stella_wan without stella_enabled; defaulting stella_enabled=true."
        )

    if not stella_enabled:
        args.enable_stella = False
        args.network_args = network_args
        return

    if network_module != _STELLA_MODULE:
        raise ValueError(
            "StelLA requires network_module='networks.stella_wan' "
            "(or set network_args stella_enabled=true with network_module='networks.lora_wan')."
        )

    args.enable_stella = True
    args.stella_retraction = _parse_choice(
        net_args_map.get("stella_retraction", "polar"),
        "stella_retraction",
        _VALID_STELLA_RETRACTIONS,
        default="polar",
    )
    args.stella_diag_s = _parse_bool(net_args_map.get("stella_diag_s", False))
    args.stella_grad_scaling = _parse_grad_scaling(
        net_args_map.get("stella_grad_scaling", True),
        "stella_grad_scaling",
    )
    args.stella_init = _parse_choice(
        _normalize_stella_init(net_args_map.get("stella_init", "orthonormal")),
        "stella_init",
        _VALID_STELLA_INITS,
        default="orthonormal",
    )

    raw_ref_dim = net_args_map.get("stella_grad_reference_dim", None)
    if raw_ref_dim is None or str(raw_ref_dim).strip() == "":
        args.stella_grad_reference_dim = 0
    else:
        args.stella_grad_reference_dim = _parse_non_negative_int(
            raw_ref_dim,
            "stella_grad_reference_dim",
        )

    _upsert_network_arg(network_args, "stella_enabled", "true")
    _upsert_network_arg(network_args, "stella_retraction", args.stella_retraction)
    _upsert_network_arg(
        network_args,
        "stella_diag_s",
        "true" if bool(args.stella_diag_s) else "false",
    )
    _upsert_network_arg(
        network_args,
        "stella_grad_scaling",
        _render_grad_scaling(args.stella_grad_scaling),
    )
    _upsert_network_arg(network_args, "stella_init", args.stella_init)
    if int(args.stella_grad_reference_dim) > 0:
        _upsert_network_arg(
            network_args,
            "stella_grad_reference_dim",
            int(args.stella_grad_reference_dim),
        )
    args.network_args = network_args

    logger.info(
        "StelLA config enabled (retraction=%s, diag_s=%s, grad_scaling=%s, init=%s, grad_reference_dim=%s).",
        args.stella_retraction,
        bool(args.stella_diag_s),
        args.stella_grad_scaling,
        args.stella_init,
        int(args.stella_grad_reference_dim),
    )
