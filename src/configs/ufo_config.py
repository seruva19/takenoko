"""Configuration parser for UFO-style training-only temporal stabilization."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _parse_pattern_list(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return list(default)
        return [stripped]
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return list(default)


def apply_ufo_config(
    args: argparse.Namespace, config: Dict[str, Any], logger: Any
) -> None:
    """Parse and validate UFO-style train-time-only controls."""

    args.enable_ufo_static_image_training = _parse_bool(
        config.get("enable_ufo_static_image_training", False),
        False,
    )
    args.ufo_static_target_frames = int(config.get("ufo_static_target_frames", 17))
    args.enable_ufo_static_video_training = _parse_bool(
        config.get("enable_ufo_static_video_training", False),
        False,
    )
    args.ufo_static_video_source_frame = str(
        config.get("ufo_static_video_source_frame", "middle")
    ).lower()

    args.enable_ufo_noise_share_in_frames = _parse_bool(
        config.get("enable_ufo_noise_share_in_frames", False),
        False,
    )
    args.ufo_noise_share_in_frames_ratio = float(
        config.get("ufo_noise_share_in_frames_ratio", 0.5)
    )
    args.ufo_noise_share_mode = str(
        config.get("ufo_noise_share_mode", "autoregressive")
    ).lower()

    args.enable_ufo_motion_sub_loss = _parse_bool(
        config.get("enable_ufo_motion_sub_loss", False),
        False,
    )
    args.ufo_motion_sub_loss_ratio = float(config.get("ufo_motion_sub_loss_ratio", 0.25))
    args.enable_ufo_temporal_attn_lora_targeting = _parse_bool(
        config.get("enable_ufo_temporal_attn_lora_targeting", False),
        False,
    )
    args.ufo_temporal_attn_lora_targeting_mode = str(
        config.get("ufo_temporal_attn_lora_targeting_mode", "augment")
    ).lower()
    default_patterns = [
        ".*temporal.*(attn|attention).*",
        ".*attn_temporal.*",
        ".*time_mix.*",
    ]
    args.ufo_temporal_attn_include_patterns = _parse_pattern_list(
        config.get("ufo_temporal_attn_include_patterns"),
        default_patterns,
    )
    args.enable_ufo_runtime_inference_profile = _parse_bool(
        config.get("enable_ufo_runtime_inference_profile", False),
        False,
    )
    args.ufo_inference_lora_multiplier = float(
        config.get("ufo_inference_lora_multiplier", 0.15)
    )

    if args.ufo_static_target_frames < 2:
        raise ValueError("ufo_static_target_frames must be >= 2")
    if args.ufo_static_video_source_frame not in {"middle", "first", "last"}:
        raise ValueError(
            "ufo_static_video_source_frame must be one of: middle, first, last"
        )
    if not (0.0 <= args.ufo_noise_share_in_frames_ratio <= 1.0):
        raise ValueError("ufo_noise_share_in_frames_ratio must be in [0.0, 1.0]")
    if args.ufo_noise_share_mode not in {"autoregressive", "shared_first"}:
        raise ValueError(
            "ufo_noise_share_mode must be one of: autoregressive, shared_first"
        )
    if not (0.0 <= args.ufo_motion_sub_loss_ratio <= 1.0):
        raise ValueError("ufo_motion_sub_loss_ratio must be in [0.0, 1.0]")
    if args.ufo_temporal_attn_lora_targeting_mode not in {"augment", "strict"}:
        raise ValueError(
            "ufo_temporal_attn_lora_targeting_mode must be one of: augment, strict"
        )
    if args.ufo_inference_lora_multiplier <= 0.0:
        raise ValueError("ufo_inference_lora_multiplier must be > 0.0")

    if args.enable_ufo_static_image_training:
        logger.info(
            "UFO static-image training enabled (target_frames=%s).",
            args.ufo_static_target_frames,
        )
    if args.enable_ufo_static_video_training:
        logger.info(
            "UFO static-video training enabled (target_frames=%s, source_frame=%s).",
            args.ufo_static_target_frames,
            args.ufo_static_video_source_frame,
        )
    if args.enable_ufo_noise_share_in_frames:
        logger.info(
            "UFO frame noise sharing enabled (ratio=%.4f, mode=%s).",
            args.ufo_noise_share_in_frames_ratio,
            args.ufo_noise_share_mode,
        )
    if args.enable_ufo_motion_sub_loss:
        logger.info(
            "UFO motion-sub loss enabled (ratio=%.4f).",
            args.ufo_motion_sub_loss_ratio,
        )
    if args.enable_ufo_temporal_attn_lora_targeting:
        if not hasattr(args, "network_args") or args.network_args is None:
            args.network_args = []
        existing_keys = {
            net_arg.split("=", 1)[0].strip()
            for net_arg in args.network_args
            if isinstance(net_arg, str) and "=" in net_arg
        }
        if getattr(args, "network_module", "") != "networks.lora_wan":
            logger.warning(
                "UFO temporal-attention LoRA targeting is only supported for networks.lora_wan; current module=%s",
                getattr(args, "network_module", ""),
            )
        elif args.ufo_temporal_attn_lora_targeting_mode == "strict":
            if "include_patterns" not in existing_keys:
                args.network_args.append(
                    f"include_patterns={repr(args.ufo_temporal_attn_include_patterns)}"
                )
        else:
            if "extra_include_patterns" not in existing_keys:
                args.network_args.append(
                    f"extra_include_patterns={repr(args.ufo_temporal_attn_include_patterns)}"
                )
        logger.info(
            "UFO temporal-attention LoRA targeting enabled (mode=%s, patterns=%s).",
            args.ufo_temporal_attn_lora_targeting_mode,
            args.ufo_temporal_attn_include_patterns,
        )
    if args.enable_ufo_runtime_inference_profile:
        logger.info(
            "UFO runtime inference profile export enabled (recommended_lora_multiplier=%.4f).",
            args.ufo_inference_lora_multiplier,
        )
