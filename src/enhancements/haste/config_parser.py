from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def parse_haste_config(config: Dict[str, Any], args: Any) -> None:
    args.enable_haste = bool(config.get("enable_haste", False))
    args.haste_encoder_name = str(
        config.get("haste_encoder_name", "dinov2-vit-b")
    )
    args.haste_use_teacher_attention = bool(
        config.get("haste_use_teacher_attention", True)
    )
    args.haste_alignment_depth = int(config.get("haste_alignment_depth", 8))
    if args.haste_alignment_depth < 0:
        raise ValueError(
            f"haste_alignment_depth must be >= 0, got {args.haste_alignment_depth}"
        )
    args.haste_attn_layer_start = int(config.get("haste_attn_layer_start", 4))
    args.haste_attn_layer_end = int(config.get("haste_attn_layer_end", 8))
    if args.haste_attn_layer_start < 0:
        raise ValueError(
            "haste_attn_layer_start must be >= 0, got "
            f"{args.haste_attn_layer_start}"
        )
    if args.haste_attn_layer_end <= args.haste_attn_layer_start:
        raise ValueError(
            "haste_attn_layer_end must be greater than haste_attn_layer_start "
            f"(start={args.haste_attn_layer_start}, end={args.haste_attn_layer_end})"
        )
    args.haste_teacher_attn_layer_offset = int(
        config.get("haste_teacher_attn_layer_offset", 4)
    )
    if args.haste_teacher_attn_layer_offset < 0:
        raise ValueError(
            "haste_teacher_attn_layer_offset must be >= 0, got "
            f"{args.haste_teacher_attn_layer_offset}"
        )
    args.haste_attn_head_limit = int(config.get("haste_attn_head_limit", 12))
    if args.haste_attn_head_limit < 0:
        raise ValueError(
            f"haste_attn_head_limit must be >= 0, got {args.haste_attn_head_limit}"
        )
    args.haste_proj_coeff = float(config.get("haste_proj_coeff", 0.5))
    if args.haste_proj_coeff < 0:
        raise ValueError(
            f"haste_proj_coeff must be >= 0, got {args.haste_proj_coeff}"
        )
    args.haste_attn_coeff = float(config.get("haste_attn_coeff", 0.5))
    if args.haste_attn_coeff < 0:
        raise ValueError(
            f"haste_attn_coeff must be >= 0, got {args.haste_attn_coeff}"
        )
    args.haste_early_stop_step = int(config.get("haste_early_stop_step", 250000))
    if args.haste_early_stop_step <= 0:
        raise ValueError(
            f"haste_early_stop_step must be > 0, got {args.haste_early_stop_step}"
        )
    args.haste_input_resolution = int(config.get("haste_input_resolution", 256))
    if args.haste_input_resolution not in (256, 512):
        raise ValueError(
            "haste_input_resolution must be 256 or 512, got "
            f"{args.haste_input_resolution}"
        )
    if args.haste_input_resolution == 512 and "dinov2" not in args.haste_encoder_name:
        raise ValueError(
            "haste_input_resolution=512 requires a DINOv2 encoder name, got "
            f"{args.haste_encoder_name}"
        )

    if not args.enable_haste:
        return

    if args.haste_use_teacher_attention and "dinov2" not in args.haste_encoder_name:
        logger.warning(
            "HASTE: haste_use_teacher_attention requires a DINOv2 encoder; disabling."
        )
        args.haste_use_teacher_attention = False

    logger.info(
        "HASTE enabled (encoder=%s, align_depth=%d, attn_layers=%d-%d, "
        "proj_coeff=%.3f, attn_coeff=%.3f, early_stop=%d, "
        "teacher_offset=%d, head_limit=%d, "
        "teacher_attention=%s)",
        args.haste_encoder_name,
        args.haste_alignment_depth,
        args.haste_attn_layer_start,
        args.haste_attn_layer_end,
        args.haste_proj_coeff,
        args.haste_attn_coeff,
        args.haste_early_stop_step,
        args.haste_teacher_attn_layer_offset,
        args.haste_attn_head_limit,
        args.haste_use_teacher_attention,
    )
