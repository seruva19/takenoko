from __future__ import annotations

import argparse
import logging
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
        key = key.strip().lower()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key] = value
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


def parse_ic_lora_config(config: dict[str, Any], args: Any, logger: logging.Logger) -> None:
    """Parse and validate IC-LoRA settings from network_args only."""

    network_module = str(getattr(args, "network_module", "") or "")
    module_selected = network_module == "networks.ic_lora_wan"

    legacy_keys = [
        "enable_ic_lora",
        "ic_lora_enabled",
        "ic_lora_concat_mode",
        "ic_lora_reference_suffix",
        "ic_lora_reference_downscale_factor",
        "ic_lora_reference_dropout_p",
        "ic_lora_first_frame_conditioning_p",
        "ic_lora_require_reference",
        "ic_lora_target_only_loss",
        "ic_lora_use_masked_conditioning",
        "ic_lora_masked_loss_only",
        "ic_lora_enforce_panel_tags",
        "ic_lora_sync_score_key",
        "ic_lora_min_sync_score",
    ]
    present_legacy = [k for k in legacy_keys if k in config]
    if present_legacy and module_selected:
        logger.warning(
            "IC-LoRA standalone TOML keys are deprecated and ignored. "
            "Use network_args entries instead. Ignored keys: %s",
            ", ".join(present_legacy),
        )

    net_args_map = _parse_network_args(args)

    args.enable_ic_lora = _parse_bool(
        net_args_map.get("ic_lora_enabled", net_args_map.get("enable_ic_lora", module_selected)),
        module_selected,
    )
    args.ic_lora_concat_mode = str(
        net_args_map.get("ic_lora_concat_mode", "reference_target_frames")
    ).lower()
    if args.ic_lora_concat_mode not in {"reference_target_frames"}:
        raise ValueError(
            "ic_lora_concat_mode must be 'reference_target_frames', got "
            f"{args.ic_lora_concat_mode!r}"
        )

    args.ic_lora_reference_suffix = str(
        net_args_map.get("ic_lora_reference_suffix", "_reference")
    )
    if not args.ic_lora_reference_suffix:
        raise ValueError("ic_lora_reference_suffix cannot be empty")
    if not args.ic_lora_reference_suffix.startswith("_"):
        args.ic_lora_reference_suffix = f"_{args.ic_lora_reference_suffix}"
        logger.info(
            "IC-LoRA normalized ic_lora_reference_suffix to '%s'.",
            args.ic_lora_reference_suffix,
        )

    try:
        args.ic_lora_reference_downscale_factor = int(
            net_args_map.get("ic_lora_reference_downscale_factor", 1)
        )
    except Exception as exc:
        raise ValueError(
            "ic_lora_reference_downscale_factor must be an integer"
        ) from exc
    if args.ic_lora_reference_downscale_factor < 1:
        raise ValueError("ic_lora_reference_downscale_factor must be >= 1")

    try:
        args.ic_lora_reference_dropout_p = float(
            net_args_map.get("ic_lora_reference_dropout_p", 0.0)
        )
    except Exception as exc:
        raise ValueError("ic_lora_reference_dropout_p must be numeric") from exc
    if not 0.0 <= args.ic_lora_reference_dropout_p <= 1.0:
        raise ValueError("ic_lora_reference_dropout_p must be in [0, 1]")

    try:
        args.ic_lora_first_frame_conditioning_p = float(
            net_args_map.get("ic_lora_first_frame_conditioning_p", 0.0)
        )
    except Exception as exc:
        raise ValueError(
            "ic_lora_first_frame_conditioning_p must be numeric"
        ) from exc
    if not 0.0 <= args.ic_lora_first_frame_conditioning_p <= 1.0:
        raise ValueError("ic_lora_first_frame_conditioning_p must be in [0, 1]")

    args.ic_lora_require_reference = _parse_bool(
        net_args_map.get("ic_lora_require_reference", True),
        True,
    )
    args.ic_lora_target_only_loss = _parse_bool(
        net_args_map.get("ic_lora_target_only_loss", True),
        True,
    )
    args.ic_lora_use_masked_conditioning = _parse_bool(
        net_args_map.get("ic_lora_use_masked_conditioning", False),
        False,
    )
    args.ic_lora_masked_loss_only = _parse_bool(
        net_args_map.get("ic_lora_masked_loss_only", False),
        False,
    )
    args.ic_lora_enforce_panel_tags = _parse_bool(
        net_args_map.get("ic_lora_enforce_panel_tags", False),
        False,
    )
    args.ic_lora_sync_score_key = str(
        net_args_map.get("ic_lora_sync_score_key", "sync_score")
    )
    if not args.ic_lora_sync_score_key:
        raise ValueError("ic_lora_sync_score_key cannot be empty")

    min_sync_score = net_args_map.get("ic_lora_min_sync_score", None)
    if min_sync_score is None or str(min_sync_score).strip() == "":
        args.ic_lora_min_sync_score = None
    else:
        try:
            args.ic_lora_min_sync_score = float(min_sync_score)
        except Exception as exc:
            raise ValueError("ic_lora_min_sync_score must be numeric") from exc
    if args.ic_lora_min_sync_score is not None and not (0.0 <= args.ic_lora_min_sync_score <= 1.0):
        raise ValueError("ic_lora_min_sync_score must be in [0, 1] when provided")

    if not args.enable_ic_lora:
        if module_selected:
            logger.info(
                "IC-LoRA module selected but ic_lora_enabled=false; IC-LoRA conditioning path is disabled."
            )
        return

    if not module_selected:
        raise ValueError(
            "enable_ic_lora=true requires network_module='networks.ic_lora_wan'."
        )

    if bool(getattr(args, "enable_control_lora", False)):
        raise ValueError("IC-LoRA is incompatible with enable_control_lora=true.")

    # IC-LoRA relies on paired reference media; reuse the control loading path.
    args.load_control = True
    args.control_suffix = args.ic_lora_reference_suffix

    if not args.ic_lora_target_only_loss:
        logger.warning(
            "ic_lora_target_only_loss=false is not supported in this implementation; forcing true."
        )
        args.ic_lora_target_only_loss = True

    if args.ic_lora_masked_loss_only and not args.ic_lora_use_masked_conditioning:
        logger.warning(
            "ic_lora_masked_loss_only=true requires ic_lora_use_masked_conditioning=true; forcing masked_loss_only=false."
        )
        args.ic_lora_masked_loss_only = False

    logger.info(
        "IC-LoRA enabled (module=%s, suffix=%s, first_frame_p=%.3f, ref_dropout_p=%.3f, downscale=%d, require_reference=%s, masked_cond=%s, masked_loss_only=%s, enforce_panel_tags=%s, min_sync_score=%s).",
        network_module,
        args.ic_lora_reference_suffix,
        args.ic_lora_first_frame_conditioning_p,
        args.ic_lora_reference_dropout_p,
        args.ic_lora_reference_downscale_factor,
        args.ic_lora_require_reference,
        args.ic_lora_use_masked_conditioning,
        args.ic_lora_masked_loss_only,
        args.ic_lora_enforce_panel_tags,
        (
            f"{args.ic_lora_min_sync_score:.3f}"
            if args.ic_lora_min_sync_score is not None
            else "None"
        ),
    )
