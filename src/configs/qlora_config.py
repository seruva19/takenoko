"""QLoRA config parsing and validation."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional


_QLORA_MODULE = "networks.qlora_wan"

_VALID_QUANT_TYPES = {"nf4", "fp4"}
_VALID_COMPUTE_DTYPES = {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
_VALID_STORAGE_DTYPES = {
    "uint8",
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
    "float32",
    "fp32",
}
_VALID_MATCH_MODES = {"contains", "regex", "exact"}
_VALID_PAGED_BITS = {8, 32}
_VALID_DTYPE_AUTO = {"auto"}


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


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


def _parse_string_list(raw: Any, key_name: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed_literal = ast.literal_eval(value)
            except Exception:
                parsed_literal = None
            if isinstance(parsed_literal, list):
                return _parse_string_list(parsed_literal, key_name)
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if isinstance(raw, list):
        parsed: List[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, str):
                raise ValueError(
                    f"{key_name}[{idx}] must be a string, got {type(item).__name__}"
                )
            stripped = item.strip()
            if stripped:
                parsed.append(stripped)
        return parsed
    raise ValueError(
        f"{key_name} must be a list of strings or comma-separated string, got {type(raw).__name__}"
    )


def _parse_int(raw: Any, key_name: str) -> int:
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(f"{key_name} must be an integer, got {raw!r}") from exc


def _get_net_arg(
    net_args_map: Dict[str, str],
    key: str,
    default: Any,
    aliases: Optional[List[str]] = None,
) -> Any:
    if key in net_args_map:
        return net_args_map[key]
    if aliases:
        for alias in aliases:
            if alias in net_args_map:
                return net_args_map[alias]
    return default


def _resolve_compute_dtype(raw: Any, args: Any) -> str:
    value = str(raw).strip().lower()
    if value not in _VALID_DTYPE_AUTO:
        return value
    mixed_precision = str(getattr(args, "mixed_precision", "") or "").strip().lower()
    if mixed_precision == "bf16":
        return "bfloat16"
    if mixed_precision == "fp16":
        return "float16"
    return "float32"


def _parse_optional_float(raw: Any, key_name: str) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"{key_name} must be a float or null, got {raw!r}") from exc


def apply_qlora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    network_args = _parse_network_args(getattr(args, "network_args", []))
    net_args_map = _parse_network_args_map(network_args)

    network_module = str(getattr(args, "network_module", "") or "")
    raw_enabled = net_args_map.get("qlora_enabled", None)
    qlora_enabled = _parse_bool(raw_enabled) if raw_enabled is not None else False

    if network_module == _QLORA_MODULE and raw_enabled is None:
        raise ValueError(
            "QLoRA initialization is network_args-only. "
            "Set network_args with qlora_enabled=true when using network_module='networks.qlora_wan'."
        )

    args.enable_qlora = bool(qlora_enabled)

    quant_type_raw = _get_net_arg(
        net_args_map,
        "qlora_quant_type",
        "nf4",
        aliases=["bnb_4bit_quant_type"],
    )
    compute_dtype_raw = _get_net_arg(
        net_args_map,
        "qlora_compute_dtype",
        "bfloat16",
        aliases=["bnb_4bit_compute_dtype"],
    )
    double_quant_raw = _get_net_arg(
        net_args_map,
        "qlora_use_double_quant",
        True,
        aliases=["bnb_4bit_use_double_quant"],
    )
    quant_storage_raw = _get_net_arg(
        net_args_map,
        "qlora_quant_storage",
        "uint8",
        aliases=["bnb_4bit_quant_storage"],
    )
    target_modules_raw = _get_net_arg(net_args_map, "qlora_target_modules", [])
    skip_modules_raw = _get_net_arg(net_args_map, "qlora_skip_modules", [])
    target_match_raw = _get_net_arg(net_args_map, "qlora_target_match", "contains")
    skip_match_raw = _get_net_arg(net_args_map, "qlora_skip_match", "contains")
    upcast_layernorm_raw = _get_net_arg(net_args_map, "qlora_upcast_layernorm", True)
    layernorm_patterns_raw = _get_net_arg(net_args_map, "qlora_layernorm_patterns", [])
    keep_fp32_modules_raw = _get_net_arg(
        net_args_map,
        "qlora_keep_in_fp32_modules",
        [],
    )
    use_paged_optimizer_raw = _get_net_arg(
        net_args_map, "qlora_use_paged_optimizer", True
    )
    paged_bits_raw = _get_net_arg(net_args_map, "qlora_paged_optimizer_bits", 8)
    fail_on_replacement_error_raw = _get_net_arg(
        net_args_map,
        "qlora_fail_on_replacement_error",
        False,
    )
    min_replaced_modules_raw = _get_net_arg(
        net_args_map,
        "qlora_min_replaced_modules",
        1,
    )
    modules_to_save_raw = _get_net_arg(net_args_map, "qlora_modules_to_save", [])
    modules_to_save_match_raw = _get_net_arg(
        net_args_map, "qlora_modules_to_save_match", "exact"
    )
    modules_to_save_lr_raw = _get_net_arg(net_args_map, "qlora_modules_to_save_lr", None)
    qlora_enable_memory_mapping_raw = _get_net_arg(
        net_args_map,
        "qlora_enable_memory_mapping",
        getattr(args, "enable_memory_mapping", False),
    )
    qlora_enable_zero_copy_loading_raw = _get_net_arg(
        net_args_map,
        "qlora_enable_zero_copy_loading",
        getattr(args, "enable_zero_copy_loading", False),
    )
    qlora_enable_non_blocking_transfers_raw = _get_net_arg(
        net_args_map,
        "qlora_enable_non_blocking_transfers",
        getattr(args, "enable_non_blocking_transfers", False),
    )
    qlora_memory_mapping_threshold_mb_raw = _get_net_arg(
        net_args_map,
        "qlora_memory_mapping_threshold_mb",
        None,
    )
    qlora_memory_mapping_threshold_raw = _get_net_arg(
        net_args_map,
        "qlora_memory_mapping_threshold",
        None,
        aliases=["qlora_memory_mapping_threshold_bytes"],
    )

    args.qlora_quant_type = str(quant_type_raw).strip().lower()
    args.qlora_compute_dtype = _resolve_compute_dtype(compute_dtype_raw, args)
    args.qlora_use_double_quant = _parse_bool(double_quant_raw)
    args.qlora_quant_storage = str(quant_storage_raw).strip().lower()
    args.qlora_target_modules = _parse_string_list(
        target_modules_raw, "qlora_target_modules"
    )
    args.qlora_skip_modules = _parse_string_list(skip_modules_raw, "qlora_skip_modules")
    args.qlora_target_match = str(target_match_raw).strip().lower()
    args.qlora_skip_match = str(skip_match_raw).strip().lower()
    args.qlora_upcast_layernorm = _parse_bool(upcast_layernorm_raw)
    args.qlora_layernorm_patterns = _parse_string_list(
        layernorm_patterns_raw, "qlora_layernorm_patterns"
    )
    args.qlora_keep_in_fp32_modules = _parse_string_list(
        keep_fp32_modules_raw, "qlora_keep_in_fp32_modules"
    )
    args.qlora_use_paged_optimizer = _parse_bool(use_paged_optimizer_raw)
    args.qlora_paged_optimizer_bits = _parse_int(
        paged_bits_raw,
        "qlora_paged_optimizer_bits",
    )
    args.qlora_fail_on_replacement_error = _parse_bool(fail_on_replacement_error_raw)
    args.qlora_min_replaced_modules = _parse_int(
        min_replaced_modules_raw,
        "qlora_min_replaced_modules",
    )
    args.qlora_modules_to_save = _parse_string_list(
        modules_to_save_raw,
        "qlora_modules_to_save",
    )
    args.qlora_modules_to_save_match = str(modules_to_save_match_raw).strip().lower()
    args.qlora_modules_to_save_lr = _parse_optional_float(
        modules_to_save_lr_raw,
        "qlora_modules_to_save_lr",
    )
    args.qlora_enable_memory_mapping = _parse_bool(qlora_enable_memory_mapping_raw)
    args.qlora_enable_zero_copy_loading = _parse_bool(
        qlora_enable_zero_copy_loading_raw
    )
    args.qlora_enable_non_blocking_transfers = _parse_bool(
        qlora_enable_non_blocking_transfers_raw
    )
    qlora_memory_mapping_threshold_mb = _parse_optional_float(
        qlora_memory_mapping_threshold_mb_raw,
        "qlora_memory_mapping_threshold_mb",
    )
    qlora_memory_mapping_threshold_bytes = (
        _parse_int(
            qlora_memory_mapping_threshold_raw,
            "qlora_memory_mapping_threshold",
        )
        if qlora_memory_mapping_threshold_raw is not None
        else None
    )

    if args.qlora_quant_type not in _VALID_QUANT_TYPES:
        raise ValueError(
            f"qlora_quant_type must be one of {sorted(_VALID_QUANT_TYPES)}, got {args.qlora_quant_type!r}"
        )
    if args.qlora_compute_dtype not in _VALID_COMPUTE_DTYPES:
        raise ValueError(
            f"qlora_compute_dtype must be one of {sorted(_VALID_COMPUTE_DTYPES)}, got {args.qlora_compute_dtype!r}"
        )
    if args.qlora_quant_storage not in _VALID_STORAGE_DTYPES:
        raise ValueError(
            f"qlora_quant_storage must be one of {sorted(_VALID_STORAGE_DTYPES)}, got {args.qlora_quant_storage!r}"
        )
    if args.qlora_target_match not in _VALID_MATCH_MODES:
        raise ValueError(
            f"qlora_target_match must be one of {sorted(_VALID_MATCH_MODES)}, got {args.qlora_target_match!r}"
        )
    if args.qlora_skip_match not in _VALID_MATCH_MODES:
        raise ValueError(
            f"qlora_skip_match must be one of {sorted(_VALID_MATCH_MODES)}, got {args.qlora_skip_match!r}"
        )
    if args.qlora_modules_to_save_match not in _VALID_MATCH_MODES:
        raise ValueError(
            "qlora_modules_to_save_match must be one of "
            f"{sorted(_VALID_MATCH_MODES)}, got {args.qlora_modules_to_save_match!r}"
        )
    if args.qlora_paged_optimizer_bits not in _VALID_PAGED_BITS:
        raise ValueError(
            "qlora_paged_optimizer_bits must be 8 or 32, "
            f"got {args.qlora_paged_optimizer_bits!r}"
        )
    if args.qlora_min_replaced_modules < 1:
        raise ValueError("qlora_min_replaced_modules must be >= 1")
    if args.qlora_modules_to_save_lr is not None and args.qlora_modules_to_save_lr <= 0.0:
        raise ValueError("qlora_modules_to_save_lr must be > 0 when provided")
    if qlora_memory_mapping_threshold_mb is not None and qlora_memory_mapping_threshold_mb <= 0.0:
        raise ValueError("qlora_memory_mapping_threshold_mb must be > 0 when provided")
    if qlora_memory_mapping_threshold_bytes is not None and qlora_memory_mapping_threshold_bytes <= 0:
        raise ValueError("qlora_memory_mapping_threshold must be > 0 when provided")

    if not args.enable_qlora:
        args.network_args = network_args
        return

    if network_module != _QLORA_MODULE:
        raise ValueError(
            "QLoRA requires network_module='networks.qlora_wan' "
            "and network_args qlora_enabled=true."
        )

    if bool(getattr(args, "fp8_scaled", False)):
        raise ValueError("QLoRA cannot be combined with fp8_scaled=true.")
    if bool(getattr(args, "torchao_fp8_enabled", False)):
        raise ValueError("QLoRA cannot be combined with torchao_fp8_enabled=true.")
    if bool(config.get("enable_dual_model_training", False)):
        raise ValueError("QLoRA currently does not support enable_dual_model_training=true.")
    if int(config.get("blocks_to_swap", 0) or 0) > 0:
        raise ValueError("QLoRA currently does not support blocks_to_swap > 0.")

    optimizer_type = str(getattr(args, "optimizer_type", "") or "")
    if optimizer_type.strip() == "":
        if args.qlora_use_paged_optimizer:
            args.optimizer_type = (
                "PagedAdamW8bit"
                if int(args.qlora_paged_optimizer_bits) == 8
                else "PagedAdamW32bit"
            )
        else:
            args.optimizer_type = "AdamW8bit"
        logger.info(
            "QLoRA enabled with no optimizer_type set; defaulting to %s.",
            args.optimizer_type,
        )
    elif args.qlora_use_paged_optimizer and optimizer_type.lower() == "adamw8bit":
        args.optimizer_type = (
            "PagedAdamW8bit"
            if int(args.qlora_paged_optimizer_bits) == 8
            else "PagedAdamW32bit"
        )
        logger.info(
            "QLoRA paged optimizer requested; switching optimizer_type to %s.",
            args.optimizer_type,
        )
    elif (
        "8bit" not in optimizer_type.lower()
        and "pagedadamw" not in optimizer_type.lower()
    ):
        logger.warning(
            "QLoRA usually pairs best with 8-bit or paged bitsandbytes optimizers; current optimizer_type=%s.",
            optimizer_type,
        )

    _upsert_network_arg(network_args, "qlora_enabled", "true")
    _upsert_network_arg(network_args, "qlora_quant_type", args.qlora_quant_type)
    _upsert_network_arg(network_args, "qlora_compute_dtype", args.qlora_compute_dtype)
    _upsert_network_arg(network_args, "bnb_4bit_quant_type", args.qlora_quant_type)
    _upsert_network_arg(network_args, "bnb_4bit_compute_dtype", args.qlora_compute_dtype)
    _upsert_network_arg(
        network_args,
        "qlora_use_double_quant",
        "true" if args.qlora_use_double_quant else "false",
    )
    _upsert_network_arg(
        network_args,
        "bnb_4bit_use_double_quant",
        "true" if args.qlora_use_double_quant else "false",
    )
    _upsert_network_arg(network_args, "qlora_quant_storage", args.qlora_quant_storage)
    _upsert_network_arg(
        network_args,
        "bnb_4bit_quant_storage",
        args.qlora_quant_storage,
    )
    _upsert_network_arg(network_args, "qlora_target_modules", repr(args.qlora_target_modules))
    _upsert_network_arg(network_args, "qlora_skip_modules", repr(args.qlora_skip_modules))
    _upsert_network_arg(network_args, "qlora_target_match", args.qlora_target_match)
    _upsert_network_arg(network_args, "qlora_skip_match", args.qlora_skip_match)
    _upsert_network_arg(
        network_args,
        "qlora_upcast_layernorm",
        "true" if args.qlora_upcast_layernorm else "false",
    )
    _upsert_network_arg(
        network_args,
        "qlora_layernorm_patterns",
        repr(args.qlora_layernorm_patterns),
    )
    _upsert_network_arg(
        network_args,
        "qlora_keep_in_fp32_modules",
        repr(args.qlora_keep_in_fp32_modules),
    )
    _upsert_network_arg(
        network_args,
        "qlora_use_paged_optimizer",
        "true" if args.qlora_use_paged_optimizer else "false",
    )
    _upsert_network_arg(
        network_args,
        "qlora_paged_optimizer_bits",
        int(args.qlora_paged_optimizer_bits),
    )
    _upsert_network_arg(
        network_args,
        "qlora_fail_on_replacement_error",
        "true" if args.qlora_fail_on_replacement_error else "false",
    )
    _upsert_network_arg(
        network_args,
        "qlora_min_replaced_modules",
        int(args.qlora_min_replaced_modules),
    )
    _upsert_network_arg(
        network_args,
        "qlora_modules_to_save",
        repr(args.qlora_modules_to_save),
    )
    _upsert_network_arg(
        network_args,
        "qlora_modules_to_save_match",
        args.qlora_modules_to_save_match,
    )
    if args.qlora_modules_to_save_lr is not None:
        _upsert_network_arg(
            network_args,
            "qlora_modules_to_save_lr",
            float(args.qlora_modules_to_save_lr),
        )
    _upsert_network_arg(
        network_args,
        "qlora_enable_memory_mapping",
        "true" if args.qlora_enable_memory_mapping else "false",
    )
    _upsert_network_arg(
        network_args,
        "qlora_enable_zero_copy_loading",
        "true" if args.qlora_enable_zero_copy_loading else "false",
    )
    _upsert_network_arg(
        network_args,
        "qlora_enable_non_blocking_transfers",
        "true" if args.qlora_enable_non_blocking_transfers else "false",
    )
    if qlora_memory_mapping_threshold_mb is not None:
        _upsert_network_arg(
            network_args,
            "qlora_memory_mapping_threshold_mb",
            float(qlora_memory_mapping_threshold_mb),
        )
    if qlora_memory_mapping_threshold_bytes is not None:
        _upsert_network_arg(
            network_args,
            "qlora_memory_mapping_threshold",
            int(qlora_memory_mapping_threshold_bytes),
        )
    args.network_args = network_args

    # QLoRA-specific memory-mapping overrides apply at model-load time.
    args.enable_memory_mapping = bool(args.qlora_enable_memory_mapping)
    args.enable_zero_copy_loading = bool(args.qlora_enable_zero_copy_loading)
    args.enable_non_blocking_transfers = bool(args.qlora_enable_non_blocking_transfers)
    if qlora_memory_mapping_threshold_bytes is not None:
        args.memory_mapping_threshold = int(qlora_memory_mapping_threshold_bytes)
    elif qlora_memory_mapping_threshold_mb is not None:
        args.memory_mapping_threshold = int(qlora_memory_mapping_threshold_mb * 1024 * 1024)

    logger.info(
        "QLoRA enabled via network_args on %s (quant_type=%s, compute_dtype=%s, double_quant=%s, quant_storage=%s, "
        "target_modules=%s, skip_modules=%s, target_match=%s, skip_match=%s, upcast_layernorm=%s, "
        "min_replaced=%s, fail_on_replace_error=%s, modules_to_save=%s).",
        args.network_module,
        args.qlora_quant_type,
        args.qlora_compute_dtype,
        args.qlora_use_double_quant,
        args.qlora_quant_storage,
        args.qlora_target_modules,
        args.qlora_skip_modules,
        args.qlora_target_match,
        args.qlora_skip_match,
        args.qlora_upcast_layernorm,
        args.qlora_min_replaced_modules,
        args.qlora_fail_on_replacement_error,
        args.qlora_modules_to_save,
    )
