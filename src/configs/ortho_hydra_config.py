"""Ortho-Hydra config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_ORTHO_HYDRA_MODULE = "networks.ortho_hydra_wan"


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


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw)


def _parse_int_at_least(raw: Any, key: str, minimum: int) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _parse_float_at_least(raw: Any, key: str, minimum: float) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float >= {minimum}, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def apply_ortho_hydra_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Ortho-Hydra settings into args and network_args."""

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)
    network_module = str(getattr(args, "network_module", "") or "")

    args.enable_ortho_hydra = False
    args.ortho_hydra_num_experts = 4
    args.ortho_hydra_balance_loss_weight = 0.001
    args.ortho_hydra_balance_warmup_steps = 0
    args.ortho_hydra_svd_niter = 2
    args.ortho_hydra_router_init_std = 0.01

    raw_enabled = net_args_map.get("ortho_hydra_enabled", None)
    enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if network_module == _ORTHO_HYDRA_MODULE and not enabled:
        raise ValueError(
            "network_module='networks.ortho_hydra_wan' requires "
            "network_args=['ortho_hydra_enabled=true', ...]."
        )

    if not enabled:
        args.network_args = network_args
        return

    if network_module != _ORTHO_HYDRA_MODULE:
        raise ValueError(
            "Ortho-Hydra requires network_module='networks.ortho_hydra_wan' "
            "when ortho_hydra_enabled=true is set in network_args."
        )

    args.enable_ortho_hydra = True
    args.train_architecture = "lora"
    args.ortho_hydra_num_experts = _parse_int_at_least(
        net_args_map.get("ortho_hydra_num_experts", 4),
        "ortho_hydra_num_experts",
        2,
    )
    args.ortho_hydra_balance_loss_weight = _parse_float_at_least(
        net_args_map.get("ortho_hydra_balance_loss_weight", 0.001),
        "ortho_hydra_balance_loss_weight",
        0.0,
    )
    args.ortho_hydra_balance_warmup_steps = _parse_int_at_least(
        net_args_map.get("ortho_hydra_balance_warmup_steps", 0),
        "ortho_hydra_balance_warmup_steps",
        0,
    )
    args.ortho_hydra_svd_niter = _parse_int_at_least(
        net_args_map.get("ortho_hydra_svd_niter", 2),
        "ortho_hydra_svd_niter",
        0,
    )
    args.ortho_hydra_router_init_std = _parse_float_at_least(
        net_args_map.get("ortho_hydra_router_init_std", 0.01),
        "ortho_hydra_router_init_std",
        0.0,
    )

    _upsert_network_arg(network_args, "ortho_hydra_enabled", "true")
    _upsert_network_arg(
        network_args, "ortho_hydra_num_experts", int(args.ortho_hydra_num_experts)
    )
    _upsert_network_arg(
        network_args,
        "ortho_hydra_balance_loss_weight",
        float(args.ortho_hydra_balance_loss_weight),
    )
    _upsert_network_arg(
        network_args,
        "ortho_hydra_balance_warmup_steps",
        int(args.ortho_hydra_balance_warmup_steps),
    )
    _upsert_network_arg(
        network_args, "ortho_hydra_svd_niter", int(args.ortho_hydra_svd_niter)
    )
    _upsert_network_arg(
        network_args,
        "ortho_hydra_router_init_std",
        float(args.ortho_hydra_router_init_std),
    )
    args.network_args = network_args

    logger.info(
        "Ortho-Hydra enabled (experts=%s, balance_weight=%s, balance_warmup_steps=%s, svd_niter=%s, router_init_std=%s).",
        args.ortho_hydra_num_experts,
        args.ortho_hydra_balance_loss_weight,
        args.ortho_hydra_balance_warmup_steps,
        args.ortho_hydra_svd_niter,
        args.ortho_hydra_router_init_std,
    )
