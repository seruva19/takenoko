from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def parse_memflow_guidance_config(config: Dict[str, Any], args: Any) -> None:
    args.enable_memflow_guidance = bool(config.get("enable_memflow_guidance", False))
    args.memflow_guidance_weight = float(config.get("memflow_guidance_weight", 0.05))
    args.memflow_guidance_bank_size = int(
        config.get("memflow_guidance_bank_size", 3)
    )
    args.memflow_guidance_record_interval = int(
        config.get("memflow_guidance_record_interval", 3)
    )
    args.memflow_guidance_local_attn_size = int(
        config.get("memflow_guidance_local_attn_size", 8)
    )
    args.memflow_guidance_top_k = int(config.get("memflow_guidance_top_k", 3))
    args.memflow_guidance_min_frames = int(
        config.get("memflow_guidance_min_frames", 2)
    )

    if not args.enable_memflow_guidance:
        return

    if args.memflow_guidance_weight < 0.0:
        raise ValueError("memflow_guidance_weight must be >= 0.0")
    if args.memflow_guidance_bank_size < 1:
        raise ValueError("memflow_guidance_bank_size must be >= 1")
    if args.memflow_guidance_record_interval < 1:
        raise ValueError("memflow_guidance_record_interval must be >= 1")
    if args.memflow_guidance_local_attn_size == 0 or args.memflow_guidance_local_attn_size < -1:
        raise ValueError("memflow_guidance_local_attn_size must be -1 or >= 1")
    if args.memflow_guidance_top_k < 0:
        raise ValueError("memflow_guidance_top_k must be >= 0")
    if args.memflow_guidance_min_frames < 1:
        raise ValueError("memflow_guidance_min_frames must be >= 1")

    logger.info(
        "MemFlow guidance enabled (weight=%.3f, bank=%d, record_interval=%d, "
        "local_attn=%d, top_k=%d, min_frames=%d)",
        args.memflow_guidance_weight,
        args.memflow_guidance_bank_size,
        args.memflow_guidance_record_interval,
        args.memflow_guidance_local_attn_size,
        args.memflow_guidance_top_k,
        args.memflow_guidance_min_frames,
    )
