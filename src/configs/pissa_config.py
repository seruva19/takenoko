"""PiSSA network_args parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, Optional


_PISSA_SUPPORTED_MODULES = {
    "networks.lora_wan",
    "networks.ic_lora_wan",
    "networks.relora_wan",
    "networks.control_lora_wan",
    "networks.reward_lora",
}


def _parse_network_args(args: Any) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    raw_args = getattr(args, "network_args", None)
    if not isinstance(raw_args, list):
        return parsed

    for net_arg in raw_args:
        if not isinstance(net_arg, str) or "=" not in net_arg:
            continue
        key, value = net_arg.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _parse_positive_int(raw_value: str, key_name: str) -> int:
    try:
        parsed = int(raw_value)
    except Exception as exc:
        raise ValueError(f"{key_name} must be an integer, got {raw_value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{key_name} must be > 0, got {parsed}")
    return parsed


def apply_pissa_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # PiSSA settings are read from network_args.
    args.pissa_initialize = False
    args.pissa_niter = None

    net_args_map = _parse_network_args(args)
    raw_initialize = str(net_args_map.get("initialize", "")).strip().lower()
    raw_pissa_niter: Optional[str] = net_args_map.get("pissa_niter")

    if raw_initialize == "" and raw_pissa_niter is None:
        return

    parsed_niter: Optional[int] = None
    if raw_pissa_niter is not None and raw_pissa_niter != "":
        parsed_niter = _parse_positive_int(raw_pissa_niter, "pissa_niter")

    is_pissa_requested = False
    if raw_initialize.startswith("pissa_niter_"):
        parsed_niter = _parse_positive_int(
            raw_initialize.rsplit("_", 1)[-1], "initialize=pissa_niter_<k>"
        )
        is_pissa_requested = True
    elif raw_initialize in {"pissa", "pissa_niter"}:
        is_pissa_requested = True
    elif raw_initialize in {"", "default", "kaiming"}:
        if parsed_niter is not None:
            raise ValueError(
                "pissa_niter requires initialize to be 'pissa' or 'pissa_niter_<k>'."
            )
        return
    else:
        # Unknown initialize modes can be used by other adapters; only validate PiSSA-specific fields.
        if parsed_niter is not None:
            raise ValueError(
                "pissa_niter is only valid when initialize='pissa' or initialize='pissa_niter_<k>'."
            )
        return

    if not is_pissa_requested:
        return

    network_module = str(getattr(args, "network_module", "") or "")
    if network_module not in _PISSA_SUPPORTED_MODULES:
        raise ValueError(
            "PiSSA initialization is only supported for "
            f"{sorted(_PISSA_SUPPORTED_MODULES)}; got {network_module!r}."
        )

    args.pissa_initialize = True
    args.pissa_niter = parsed_niter

    if parsed_niter is None:
        logger.info("PiSSA initialization enabled (exact SVD).")
    else:
        logger.info("PiSSA initialization enabled (fast SVD niter=%s).", parsed_niter)

    try:
        dropout = float(getattr(args, "network_dropout", 0.0) or 0.0)
        if dropout > 0.0:
            logger.warning(
                "PiSSA is configured with network_dropout=%s; PiSSA papers/code typically use dropout=0.",
                dropout,
            )
    except Exception:
        pass
