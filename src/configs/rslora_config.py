"""RS-LoRA network_args parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_RSLORA_SUPPORTED_MODULES = {
    "networks.lora_wan",
    "networks.ic_lora_wan",
    "networks.relora_wan",
    "networks.riemann_lora_wan",
    "networks.tc_lora_wan",
    "networks.moc_lora",
    "networks.qlora_wan",
    "networks.singlora_wan",
}


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
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"use_rslora must be boolean-like, got {raw!r}")
    return bool(raw)


def apply_rslora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    # Prefer network_args key for compatibility with LoRA module kwargs.
    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    raw_use_rslora = net_args_map.get("use_rslora", None)
    if raw_use_rslora is None:
        raw_use_rslora = config.get("use_rslora", False)

    args.use_rslora = _parse_bool(raw_use_rslora)
    network_module = str(getattr(args, "network_module", "") or "")
    if args.use_rslora and network_module not in _RSLORA_SUPPORTED_MODULES:
        raise ValueError(
            "use_rslora=true is not supported for "
            f"network_module={network_module!r}. Supported modules: "
            f"{sorted(_RSLORA_SUPPORTED_MODULES)}."
        )

    _upsert_network_arg(
        network_args,
        "use_rslora",
        "true" if args.use_rslora else "false",
    )
    args.network_args = network_args

    if args.use_rslora:
        logger.info(
            "RS-LoRA enabled: adapter scaling uses alpha/sqrt(rank) instead of alpha/rank."
        )
