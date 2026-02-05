"""Configuration parsing for Immiscible Diffusion noise assignment."""

from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def parse_immiscible_noise_config(config: Dict[str, Any], args: Any) -> None:
    """Parse training-only Immiscible Diffusion options."""
    args.enable_immiscible_diffusion = bool(
        config.get("enable_immiscible_diffusion", False)
    )
    args.immiscible_mode = str(config.get("immiscible_mode", "knn")).lower()
    if args.immiscible_mode not in {
        "knn",
        "linear_assignment",
        "linear_assignment_candidates",
    }:
        raise ValueError(
            "immiscible_mode must be one of "
            "'knn', 'linear_assignment', 'linear_assignment_candidates', "
            f"got {args.immiscible_mode}"
        )

    args.immiscible_candidate_count = int(config.get("immiscible_candidate_count", 4))
    if args.immiscible_candidate_count < 1:
        raise ValueError(
            "immiscible_candidate_count must be >= 1, "
            f"got {args.immiscible_candidate_count}"
        )
    args.immiscible_assignment_pool_factor = int(
        config.get("immiscible_assignment_pool_factor", 2)
    )
    if args.immiscible_assignment_pool_factor < 1:
        raise ValueError(
            "immiscible_assignment_pool_factor must be >= 1, "
            f"got {args.immiscible_assignment_pool_factor}"
        )

    args.immiscible_distance_dtype = str(
        config.get("immiscible_distance_dtype", "float32")
    ).lower()
    if args.immiscible_distance_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(
            "immiscible_distance_dtype must be one of "
            "'float32', 'float16', 'bfloat16', got "
            f"{args.immiscible_distance_dtype}"
        )
    args.immiscible_use_scipy = bool(config.get("immiscible_use_scipy", True))
    args.immiscible_fallback_mode = str(
        config.get("immiscible_fallback_mode", "knn")
    ).lower()
    if args.immiscible_fallback_mode not in {"knn", "random"}:
        raise ValueError(
            "immiscible_fallback_mode must be one of 'knn', 'random', "
            f"got {args.immiscible_fallback_mode}"
        )

    if args.enable_immiscible_diffusion:
        logger.info(
            "Immiscible Diffusion enabled (mode=%s, k=%d, pool_factor=%d, distance_dtype=%s, use_scipy=%s, fallback=%s)",
            args.immiscible_mode,
            args.immiscible_candidate_count,
            args.immiscible_assignment_pool_factor,
            args.immiscible_distance_dtype,
            args.immiscible_use_scipy,
            args.immiscible_fallback_mode,
        )
