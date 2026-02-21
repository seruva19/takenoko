"""LoRWeB-LoRA configuration parsing helpers."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def _parse_network_args(args: argparse.Namespace) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    raw_args = getattr(args, "network_args", None)
    if not isinstance(raw_args, list):
        return parsed
    for net_arg in raw_args:
        if not isinstance(net_arg, str) or "=" not in net_arg:
            continue
        key, value = net_arg.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
        return default
    return bool(value)


def _parse_int(value: Any, name: str, min_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {parsed}")
    return parsed


def _parse_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _parse_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return _parse_float(value, name)


def _upsert_network_arg(args: argparse.Namespace, key: str, value: Any) -> None:
    if not hasattr(args, "network_args") or args.network_args is None:
        args.network_args = []

    key_eq = f"{key}="
    idx = None
    for i, item in enumerate(args.network_args):
        if isinstance(item, str) and item.startswith(key_eq):
            idx = i
            break

    if isinstance(value, bool):
        value_str = "true" if value else "false"
    else:
        value_str = str(value)

    token = f"{key}={value_str}"
    if idx is None:
        args.network_args.append(token)
    else:
        args.network_args[idx] = token


def apply_lorweb_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse and validate LoRWeB-LoRA settings from network_args."""

    network_module = str(getattr(args, "network_module", "") or "")
    module_selected = network_module == "networks.lorweb_lora_wan"
    net_args_map = _parse_network_args(args)

    # Defaults.
    args.lorweb_basis_size = _parse_int(net_args_map.get("lorweb_basis_size", 32), "lorweb_basis_size", 1)
    args.lorweb_keys_dim = _parse_int(net_args_map.get("lorweb_keys_dim", 128), "lorweb_keys_dim", 1)
    args.lorweb_heads = _parse_int(net_args_map.get("lorweb_heads", 1), "lorweb_heads", 1)
    args.lorweb_softmax = _parse_bool(net_args_map.get("lorweb_softmax", True), True)
    args.lorweb_mixing_coeffs_type = str(
        net_args_map.get("lorweb_mixing_coeffs_type", "mean")
    ).strip().lower()
    args.lorweb_query_projection_type = str(
        net_args_map.get("lorweb_query_projection_type", "linear")
    ).strip().lower()
    args.lorweb_query_pooling = str(
        net_args_map.get("lorweb_query_pooling", "avg")
    ).strip().lower()
    args.lorweb_query_mode = str(
        net_args_map.get("lorweb_query_mode", "all_tokens")
    ).strip().lower()
    args.lorweb_query_l2_normalize = _parse_bool(
        net_args_map.get("lorweb_query_l2_normalize", False),
        False,
    )
    args.lorweb_external_query = _parse_bool(
        net_args_map.get("lorweb_external_query", False),
        False,
    )
    args.lorweb_external_query_dim = _parse_int(
        net_args_map.get("lorweb_external_query_dim", 128),
        "lorweb_external_query_dim",
        1,
    )
    args.lorweb_external_query_mode = str(
        net_args_map.get("lorweb_external_query_mode", "global")
    ).strip().lower()
    args.lorweb_runtime_source = str(
        net_args_map.get("lorweb_runtime_source", "auto")
    ).strip().lower()
    args.lorweb_external_query_encoder = str(
        net_args_map.get("lorweb_external_query_encoder", "none")
    ).strip().lower()
    args.lorweb_external_query_model_name = str(
        net_args_map.get("lorweb_external_query_model_name", "")
    ).strip()
    args.lorweb_external_query_image_size = _parse_int(
        net_args_map.get("lorweb_external_query_image_size", 224),
        "lorweb_external_query_image_size",
        8,
    )
    args.lorweb_external_query_cache = _parse_bool(
        net_args_map.get("lorweb_external_query_cache", False),
        False,
    )
    args.lorweb_external_query_cache_size = _parse_int(
        net_args_map.get("lorweb_external_query_cache_size", 512),
        "lorweb_external_query_cache_size",
        1,
    )
    args.lorweb_external_query_encode_batch_size = _parse_int(
        net_args_map.get("lorweb_external_query_encode_batch_size", 0),
        "lorweb_external_query_encode_batch_size",
        0,
    )
    args.lorweb_external_query_autocast_dtype = str(
        net_args_map.get("lorweb_external_query_autocast_dtype", "none")
    ).strip().lower()
    args.lorweb_external_query_encoder_offload_interval = _parse_int(
        net_args_map.get("lorweb_external_query_encoder_offload_interval", 0),
        "lorweb_external_query_encoder_offload_interval",
        0,
    )
    args.lorweb_video_query_frame_strategy = str(
        net_args_map.get("lorweb_video_query_frame_strategy", "mean")
    ).strip().lower()
    args.lorweb_entropy_reg_lambda = _parse_float(
        net_args_map.get("lorweb_entropy_reg_lambda", 0.0),
        "lorweb_entropy_reg_lambda",
    )
    args.lorweb_entropy_reg_target = _parse_optional_float(
        net_args_map.get("lorweb_entropy_reg_target", None),
        "lorweb_entropy_reg_target",
    )
    args.lorweb_diversity_reg_lambda = _parse_float(
        net_args_map.get("lorweb_diversity_reg_lambda", 0.0),
        "lorweb_diversity_reg_lambda",
    )
    args.lorweb_layer_stats_limit = _parse_int(
        net_args_map.get("lorweb_layer_stats_limit", 0),
        "lorweb_layer_stats_limit",
        0,
    )
    args.lorweb_topk_basis = _parse_int(
        net_args_map.get("lorweb_topk_basis", 0),
        "lorweb_topk_basis",
        0,
    )
    args.lorweb_mix_log_interval = _parse_int(
        net_args_map.get("lorweb_mix_log_interval", 100),
        "lorweb_mix_log_interval",
        1,
    )
    args.lorweb_mix_histogram_interval = _parse_int(
        net_args_map.get("lorweb_mix_histogram_interval", 500),
        "lorweb_mix_histogram_interval",
        1,
    )
    args.lorweb_mix_warn_entropy_min = _parse_float(
        net_args_map.get("lorweb_mix_warn_entropy_min", 0.05),
        "lorweb_mix_warn_entropy_min",
    )
    args.lorweb_mix_warn_max_coeff_max = _parse_float(
        net_args_map.get("lorweb_mix_warn_max_coeff_max", 0.95),
        "lorweb_mix_warn_max_coeff_max",
    )

    valid_mixing = {"mean", "sum"}
    if args.lorweb_mixing_coeffs_type not in valid_mixing:
        raise ValueError(
            f"lorweb_mixing_coeffs_type must be one of {sorted(valid_mixing)}, "
            f"got {args.lorweb_mixing_coeffs_type!r}"
        )
    valid_proj = {"linear", "none"}
    if args.lorweb_query_projection_type not in valid_proj:
        raise ValueError(
            f"lorweb_query_projection_type must be one of {sorted(valid_proj)}, "
            f"got {args.lorweb_query_projection_type!r}"
        )
    valid_pooling = {"avg", "max"}
    if args.lorweb_query_pooling not in valid_pooling:
        raise ValueError(
            f"lorweb_query_pooling must be one of {sorted(valid_pooling)}, "
            f"got {args.lorweb_query_pooling!r}"
        )
    valid_query_modes = {
        "all_tokens",
        "first_half",
        "last_half",
        "first_three_quarters",
        "aa'bb'",
        "caa'bb'",
        "caa'b",
        "caa'",
    }
    if args.lorweb_query_mode not in valid_query_modes:
        raise ValueError(
            f"lorweb_query_mode must be one of {sorted(valid_query_modes)}, "
            f"got {args.lorweb_query_mode!r}"
        )
    valid_external_modes = {
        "global",
        "triplet_concat",
        "aa'bb'",
        "caa'bb'",
        "caa'b",
        "caa'",
        "paa'b",
        "paa'b3",
        "ca'-ca",
        "ca'-ca+cb",
        "cat-aa'b",
        "cat-paa'b",
        "cat-paa'b3",
    }
    if args.lorweb_external_query_mode not in valid_external_modes:
        raise ValueError(
            f"lorweb_external_query_mode must be one of {sorted(valid_external_modes)}, "
            f"got {args.lorweb_external_query_mode!r}"
        )
    valid_runtime_sources = {"auto", "latents", "control_signal", "pixels"}
    if args.lorweb_runtime_source not in valid_runtime_sources:
        raise ValueError(
            f"lorweb_runtime_source must be one of {sorted(valid_runtime_sources)}, "
            f"got {args.lorweb_runtime_source!r}"
        )
    valid_external_encoders = {"none", "clip", "siglip2"}
    if args.lorweb_external_query_encoder not in valid_external_encoders:
        raise ValueError(
            "lorweb_external_query_encoder must be one of "
            f"{sorted(valid_external_encoders)}, "
            f"got {args.lorweb_external_query_encoder!r}"
        )
    valid_autocast_dtypes = {"none", "fp16", "float16", "bf16", "bfloat16"}
    if args.lorweb_external_query_autocast_dtype not in valid_autocast_dtypes:
        raise ValueError(
            "lorweb_external_query_autocast_dtype must be one of "
            f"{sorted(valid_autocast_dtypes)}, "
            f"got {args.lorweb_external_query_autocast_dtype!r}"
        )
    valid_frame_strategies = {"mean", "first", "middle", "last", "motion_weighted"}
    if args.lorweb_video_query_frame_strategy not in valid_frame_strategies:
        raise ValueError(
            "lorweb_video_query_frame_strategy must be one of "
            f"{sorted(valid_frame_strategies)}, "
            f"got {args.lorweb_video_query_frame_strategy!r}"
        )
    if args.lorweb_keys_dim % args.lorweb_heads != 0:
        raise ValueError(
            "lorweb_keys_dim must be divisible by lorweb_heads "
            f"(got {args.lorweb_keys_dim} and {args.lorweb_heads})."
        )
    if args.lorweb_entropy_reg_lambda < 0.0:
        raise ValueError("lorweb_entropy_reg_lambda must be >= 0.0")
    if args.lorweb_diversity_reg_lambda < 0.0:
        raise ValueError("lorweb_diversity_reg_lambda must be >= 0.0")
    if args.lorweb_mix_warn_entropy_min < 0.0:
        raise ValueError("lorweb_mix_warn_entropy_min must be >= 0.0")
    if not (0.0 <= args.lorweb_mix_warn_max_coeff_max <= 1.0):
        raise ValueError("lorweb_mix_warn_max_coeff_max must be in [0.0, 1.0]")

    if not module_selected:
        return

    # Make defaults explicit in network_args so network module receives resolved values.
    _upsert_network_arg(args, "lorweb_basis_size", args.lorweb_basis_size)
    _upsert_network_arg(args, "lorweb_keys_dim", args.lorweb_keys_dim)
    _upsert_network_arg(args, "lorweb_heads", args.lorweb_heads)
    _upsert_network_arg(args, "lorweb_softmax", args.lorweb_softmax)
    _upsert_network_arg(
        args, "lorweb_mixing_coeffs_type", args.lorweb_mixing_coeffs_type
    )
    _upsert_network_arg(
        args,
        "lorweb_query_projection_type",
        args.lorweb_query_projection_type,
    )
    _upsert_network_arg(args, "lorweb_query_pooling", args.lorweb_query_pooling)
    _upsert_network_arg(args, "lorweb_query_mode", args.lorweb_query_mode)
    _upsert_network_arg(
        args,
        "lorweb_query_l2_normalize",
        args.lorweb_query_l2_normalize,
    )
    _upsert_network_arg(args, "lorweb_external_query", args.lorweb_external_query)
    _upsert_network_arg(
        args, "lorweb_external_query_dim", args.lorweb_external_query_dim
    )
    _upsert_network_arg(
        args, "lorweb_external_query_mode", args.lorweb_external_query_mode
    )
    _upsert_network_arg(args, "lorweb_runtime_source", args.lorweb_runtime_source)
    _upsert_network_arg(
        args, "lorweb_external_query_encoder", args.lorweb_external_query_encoder
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_model_name",
        args.lorweb_external_query_model_name,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_image_size",
        args.lorweb_external_query_image_size,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_cache",
        args.lorweb_external_query_cache,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_cache_size",
        args.lorweb_external_query_cache_size,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_encode_batch_size",
        args.lorweb_external_query_encode_batch_size,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_autocast_dtype",
        args.lorweb_external_query_autocast_dtype,
    )
    _upsert_network_arg(
        args,
        "lorweb_external_query_encoder_offload_interval",
        args.lorweb_external_query_encoder_offload_interval,
    )
    _upsert_network_arg(
        args,
        "lorweb_video_query_frame_strategy",
        args.lorweb_video_query_frame_strategy,
    )
    _upsert_network_arg(
        args,
        "lorweb_entropy_reg_lambda",
        args.lorweb_entropy_reg_lambda,
    )
    _upsert_network_arg(
        args,
        "lorweb_entropy_reg_target",
        args.lorweb_entropy_reg_target,
    )
    _upsert_network_arg(
        args,
        "lorweb_diversity_reg_lambda",
        args.lorweb_diversity_reg_lambda,
    )
    _upsert_network_arg(
        args,
        "lorweb_layer_stats_limit",
        args.lorweb_layer_stats_limit,
    )
    _upsert_network_arg(
        args,
        "lorweb_topk_basis",
        args.lorweb_topk_basis,
    )
    _upsert_network_arg(args, "lorweb_mix_log_interval", args.lorweb_mix_log_interval)
    _upsert_network_arg(
        args, "lorweb_mix_histogram_interval", args.lorweb_mix_histogram_interval
    )
    _upsert_network_arg(
        args, "lorweb_mix_warn_entropy_min", args.lorweb_mix_warn_entropy_min
    )
    _upsert_network_arg(
        args, "lorweb_mix_warn_max_coeff_max", args.lorweb_mix_warn_max_coeff_max
    )

    logger.info(
        "LoRWeB-LoRA enabled (basis=%s, keys_dim=%s, heads=%s, external_query=%s, source=%s, encoder=%s, ext_mode=%s, cache=%s, frame_strategy=%s).",
        args.lorweb_basis_size,
        args.lorweb_keys_dim,
        args.lorweb_heads,
        args.lorweb_external_query,
        args.lorweb_runtime_source,
        args.lorweb_external_query_encoder,
        args.lorweb_external_query_mode,
        args.lorweb_external_query_cache,
        args.lorweb_video_query_frame_strategy,
    )
