
"""VB-LoRA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict, List


_VBLORA_MODULE = "networks.vblora_wan"


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
        return False
    return bool(raw)


def _parse_positive_int(raw: Any, key: str) -> int:
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _parse_float(raw: Any, key: str) -> float:
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float, got {raw!r}") from exc


def _parse_positive_float(raw: Any, key: str) -> float:
    value = _parse_float(raw, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def _parse_dropout(raw: Any, key: str) -> float:
    value = _parse_float(raw, key)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{key} must be in [0, 1), got {value}")
    return value


def apply_vblora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    del config  # VB-LoRA settings are parsed from network_args only.

    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)
    network_module = str(getattr(args, "network_module", "") or "")

    raw_enabled = net_args_map.get("vblora_enabled", None)
    vblora_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if vblora_enabled and network_module == "networks.lora_wan":
        logger.info(
            "vblora_enabled=true in network_args with network_module=networks.lora_wan; switching to networks.vblora_wan."
        )
        args.network_module = _VBLORA_MODULE
        network_module = _VBLORA_MODULE

    if network_module == _VBLORA_MODULE and raw_enabled is None:
        vblora_enabled = True
        _upsert_network_arg(network_args, "vblora_enabled", "true")
        net_args_map["vblora_enabled"] = "true"
        logger.info(
            "network_module=networks.vblora_wan without vblora_enabled; defaulting vblora_enabled=true."
        )

    if not vblora_enabled:
        args.enable_vblora = False
        args.network_args = network_args
        return

    if network_module != _VBLORA_MODULE:
        raise ValueError(
            "VB-LoRA requires network_module='networks.vblora_wan' "
            "(or set network_args vblora_enabled=true with network_module='networks.lora_wan')."
        )

    args.enable_vblora = True

    vector_length_raw = net_args_map.get(
        "vblora_vector_length",
        net_args_map.get("vblora_vector_dim", 256),
    )
    init_logits_std_raw = net_args_map.get(
        "vblora_init_logits_std",
        net_args_map.get("vblora_init_std", 0.1),
    )

    args.vblora_num_vectors = _parse_positive_int(
        net_args_map.get("vblora_num_vectors", 256),
        "vblora_num_vectors",
    )
    args.vblora_vector_length = _parse_positive_int(
        vector_length_raw,
        "vblora_vector_length",
    )
    args.vblora_topk = _parse_positive_int(
        net_args_map.get("vblora_topk", 2),
        "vblora_topk",
    )
    args.vblora_dropout = _parse_dropout(
        net_args_map.get("vblora_dropout", 0.0),
        "vblora_dropout",
    )
    args.vblora_init_vector_bank_bound = _parse_positive_float(
        net_args_map.get("vblora_init_vector_bank_bound", 0.02),
        "vblora_init_vector_bank_bound",
    )
    args.vblora_init_logits_std = _parse_positive_float(
        init_logits_std_raw,
        "vblora_init_logits_std",
    )
    args.vblora_bank_lr_ratio = _parse_positive_float(
        net_args_map.get("vblora_bank_lr_ratio", 1.0),
        "vblora_bank_lr_ratio",
    )
    args.vblora_logits_lr_ratio = _parse_positive_float(
        net_args_map.get("vblora_logits_lr_ratio", 1.0),
        "vblora_logits_lr_ratio",
    )
    args.vblora_train_vector_bank = _parse_bool(
        net_args_map.get("vblora_train_vector_bank", True)
    )
    args.vblora_save_only_topk_weights = _parse_bool(
        net_args_map.get("vblora_save_only_topk_weights", False)
    )

    if args.vblora_topk > args.vblora_num_vectors:
        raise ValueError(
            "vblora_topk must be <= vblora_num_vectors "
            f"(got topk={args.vblora_topk}, num_vectors={args.vblora_num_vectors})."
        )

    _upsert_network_arg(network_args, "vblora_enabled", "true")
    _upsert_network_arg(network_args, "vblora_num_vectors", int(args.vblora_num_vectors))
    _upsert_network_arg(network_args, "vblora_vector_length", int(args.vblora_vector_length))
    _upsert_network_arg(network_args, "vblora_topk", int(args.vblora_topk))
    _upsert_network_arg(network_args, "vblora_dropout", float(args.vblora_dropout))
    _upsert_network_arg(
        network_args,
        "vblora_init_vector_bank_bound",
        float(args.vblora_init_vector_bank_bound),
    )
    _upsert_network_arg(
        network_args,
        "vblora_init_logits_std",
        float(args.vblora_init_logits_std),
    )
    _upsert_network_arg(
        network_args,
        "vblora_bank_lr_ratio",
        float(args.vblora_bank_lr_ratio),
    )
    _upsert_network_arg(
        network_args,
        "vblora_logits_lr_ratio",
        float(args.vblora_logits_lr_ratio),
    )
    _upsert_network_arg(
        network_args,
        "vblora_train_vector_bank",
        "true" if args.vblora_train_vector_bank else "false",
    )
    _upsert_network_arg(
        network_args,
        "vblora_save_only_topk_weights",
        "true" if args.vblora_save_only_topk_weights else "false",
    )
    args.network_args = network_args

    logger.info(
        "VB-LoRA config enabled (num_vectors=%s, vector_length=%s, topk=%s, dropout=%s, save_only_topk=%s).",
        args.vblora_num_vectors,
        args.vblora_vector_length,
        args.vblora_topk,
        args.vblora_dropout,
        args.vblora_save_only_topk_weights,
    )
