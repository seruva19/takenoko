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
        parsed[key.strip().lower()] = value.strip()
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


def _parse_int(value: Any, name: str, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _parse_float(value: Any, name: str, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _upsert_network_arg(args: argparse.Namespace, key: str, value: Any) -> None:
    if not hasattr(args, "network_args") or args.network_args is None:
        args.network_args = []

    value_str = str(value).lower() if isinstance(value, bool) else str(value)
    token = f"{key}={value_str}"
    key_eq = f"{key}="
    for idx, current in enumerate(args.network_args):
        if isinstance(current, str) and current.startswith(key_eq):
            args.network_args[idx] = token
            return
    args.network_args.append(token)


def apply_video2lora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse and validate Video2LoRA settings from network_args."""

    network_module = str(getattr(args, "network_module", "") or "")
    module_selected = network_module == "networks.video2lora_wan"
    net_args_map = _parse_network_args(args)

    legacy_keys = [
        "video2lora_enabled",
        "video2lora_aux_down_dim",
        "video2lora_aux_up_dim",
        "video2lora_reference_feature_dim",
        "video2lora_feature_dim",
        "video2lora_decoder_blocks",
        "video2lora_attention_heads",
        "video2lora_sample_iters",
        "video2lora_max_reference_frames",
        "video2lora_spatial_pool_size",
        "video2lora_reference_suffix",
        "video2lora_runtime_source",
        "video2lora_require_reference",
        "video2lora_reference_dropout_p",
        "video2lora_hypernet_lr_ratio",
    ]
    present_legacy = [key for key in legacy_keys if key in config]
    if present_legacy and module_selected:
        logger.warning(
            "Video2LoRA standalone TOML keys are deprecated and ignored. "
            "Use network_args entries instead. Ignored keys: %s",
            ", ".join(present_legacy),
        )

    args.enable_video2lora = _parse_bool(
        net_args_map.get("video2lora_enabled", False),
        False,
    )
    args.video2lora_aux_down_dim = _parse_int(
        net_args_map.get("video2lora_aux_down_dim", 128),
        "video2lora_aux_down_dim",
    )
    args.video2lora_aux_up_dim = _parse_int(
        net_args_map.get("video2lora_aux_up_dim", 64),
        "video2lora_aux_up_dim",
    )
    args.video2lora_reference_feature_dim = _parse_int(
        net_args_map.get("video2lora_reference_feature_dim", 16),
        "video2lora_reference_feature_dim",
    )
    args.video2lora_feature_dim = _parse_int(
        net_args_map.get("video2lora_feature_dim", 256),
        "video2lora_feature_dim",
    )
    args.video2lora_decoder_blocks = _parse_int(
        net_args_map.get("video2lora_decoder_blocks", 4),
        "video2lora_decoder_blocks",
    )
    args.video2lora_attention_heads = _parse_int(
        net_args_map.get("video2lora_attention_heads", 4),
        "video2lora_attention_heads",
    )
    args.video2lora_sample_iters = _parse_int(
        net_args_map.get("video2lora_sample_iters", 4),
        "video2lora_sample_iters",
    )
    args.video2lora_max_reference_frames = _parse_int(
        net_args_map.get("video2lora_max_reference_frames", 0),
        "video2lora_max_reference_frames",
        minimum=0,
    )
    args.video2lora_spatial_pool_size = _parse_int(
        net_args_map.get("video2lora_spatial_pool_size", 0),
        "video2lora_spatial_pool_size",
        minimum=0,
    )
    args.video2lora_reference_suffix = str(
        net_args_map.get("video2lora_reference_suffix", "_reference")
    )
    args.video2lora_runtime_source = str(
        net_args_map.get("video2lora_runtime_source", "auto")
    ).strip().lower()
    args.video2lora_require_reference = _parse_bool(
        net_args_map.get("video2lora_require_reference", True),
        True,
    )
    args.video2lora_reference_dropout_p = _parse_float(
        net_args_map.get("video2lora_reference_dropout_p", 0.0),
        "video2lora_reference_dropout_p",
        0.0,
    )
    args.video2lora_hypernet_lr_ratio = _parse_float(
        net_args_map.get("video2lora_hypernet_lr_ratio", 1.0),
        "video2lora_hypernet_lr_ratio",
        0.0,
    )

    if not args.video2lora_reference_suffix:
        raise ValueError("video2lora_reference_suffix cannot be empty")
    if not args.video2lora_reference_suffix.startswith("_"):
        args.video2lora_reference_suffix = f"_{args.video2lora_reference_suffix}"
        logger.info(
            "Video2LoRA normalized video2lora_reference_suffix to '%s'.",
            args.video2lora_reference_suffix,
        )

    valid_runtime_sources = {"auto", "control_signal"}
    if args.video2lora_runtime_source not in valid_runtime_sources:
        raise ValueError(
            "video2lora_runtime_source must be one of "
            f"{sorted(valid_runtime_sources)}, got {args.video2lora_runtime_source!r}"
        )
    if args.video2lora_runtime_source == "auto":
        logger.info(
            "Video2LoRA runtime_source='auto' is normalized to paired control/reference latents only."
        )
    if not 0.0 <= args.video2lora_reference_dropout_p <= 1.0:
        raise ValueError("video2lora_reference_dropout_p must be in [0, 1]")
    if args.video2lora_hypernet_lr_ratio <= 0.0:
        raise ValueError("video2lora_hypernet_lr_ratio must be > 0")

    if not args.enable_video2lora:
        if module_selected:
            raise ValueError(
                "network_module='networks.video2lora_wan' requires "
                "network_args entry 'video2lora_enabled=true' for explicit opt-in."
            )
        return

    if not module_selected:
        raise ValueError(
            "video2lora_enabled=true requires network_module='networks.video2lora_wan'."
        )
    if int(getattr(args, "network_dim", 0) or 0) <= 0:
        raise ValueError("Video2LoRA requires network_dim >= 1.")
    if bool(getattr(args, "enable_control_lora", False)):
        raise ValueError("Video2LoRA is incompatible with enable_control_lora=true.")

    if int(getattr(args, "network_dim", 1) or 1) > 8:
        logger.warning(
            "Video2LoRA network_dim=%s is unusually high for LightLoRA; "
            "the paper-style path is typically low-rank (for example 1-4).",
            getattr(args, "network_dim", 1),
        )

    args.load_control = True
    args.control_suffix = args.video2lora_reference_suffix

    _upsert_network_arg(args, "video2lora_enabled", args.enable_video2lora)
    _upsert_network_arg(args, "video2lora_aux_down_dim", args.video2lora_aux_down_dim)
    _upsert_network_arg(args, "video2lora_aux_up_dim", args.video2lora_aux_up_dim)
    _upsert_network_arg(
        args,
        "video2lora_reference_feature_dim",
        args.video2lora_reference_feature_dim,
    )
    _upsert_network_arg(args, "video2lora_feature_dim", args.video2lora_feature_dim)
    _upsert_network_arg(
        args, "video2lora_decoder_blocks", args.video2lora_decoder_blocks
    )
    _upsert_network_arg(
        args, "video2lora_attention_heads", args.video2lora_attention_heads
    )
    _upsert_network_arg(args, "video2lora_sample_iters", args.video2lora_sample_iters)
    _upsert_network_arg(
        args,
        "video2lora_max_reference_frames",
        args.video2lora_max_reference_frames,
    )
    _upsert_network_arg(
        args, "video2lora_spatial_pool_size", args.video2lora_spatial_pool_size
    )
    _upsert_network_arg(
        args, "video2lora_reference_suffix", args.video2lora_reference_suffix
    )
    _upsert_network_arg(
        args, "video2lora_runtime_source", args.video2lora_runtime_source
    )
    _upsert_network_arg(
        args, "video2lora_require_reference", args.video2lora_require_reference
    )
    _upsert_network_arg(
        args,
        "video2lora_reference_dropout_p",
        args.video2lora_reference_dropout_p,
    )
    _upsert_network_arg(
        args, "video2lora_hypernet_lr_ratio", args.video2lora_hypernet_lr_ratio
    )

    logger.info(
        "Video2LoRA enabled (rank=%s, aux=%s/%s, ref_feat_dim=%s, feature_dim=%s, blocks=%s, heads=%s, sample_iters=%s, source=%s, suffix=%s, hyper_lr_ratio=%.3f).",
        getattr(args, "network_dim", None),
        args.video2lora_aux_down_dim,
        args.video2lora_aux_up_dim,
        args.video2lora_reference_feature_dim,
        args.video2lora_feature_dim,
        args.video2lora_decoder_blocks,
        args.video2lora_attention_heads,
        args.video2lora_sample_iters,
        args.video2lora_runtime_source,
        args.video2lora_reference_suffix,
        args.video2lora_hypernet_lr_ratio,
    )
