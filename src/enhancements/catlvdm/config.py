"""Configuration parsing for CAT-LVDM corruption-aware training."""
from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def parse_catlvdm_config(config: Dict[str, Any], args: Any) -> None:
    args.enable_catlvdm_corruption = bool(
        config.get("enable_catlvdm_corruption", False)
    )
    args.catlvdm_noise_type = str(config.get("catlvdm_noise_type", "bcni")).lower()
    allowed_noise_types = {"bcni", "sacn"}
    if args.catlvdm_noise_type not in allowed_noise_types:
        raise ValueError(
            "catlvdm_noise_type must be one of: "
            + ", ".join(sorted(allowed_noise_types))
        )

    args.catlvdm_noise_ratio = float(config.get("catlvdm_noise_ratio", 0.075))
    if not (0.0 <= args.catlvdm_noise_ratio <= 0.2):
        raise ValueError(
            "catlvdm_noise_ratio must be in [0.0, 0.2], "
            f"got {args.catlvdm_noise_ratio}"
        )

    args.catlvdm_apply_to = str(config.get("catlvdm_apply_to", "t5")).lower()
    allowed_apply_targets = {"t5"}
    if args.catlvdm_apply_to not in allowed_apply_targets:
        raise ValueError(
            "catlvdm_apply_to must be one of: "
            + ", ".join(sorted(allowed_apply_targets))
        )

    if args.enable_catlvdm_corruption:
        logger.info(
            "CAT-LVDM corruption enabled: type=%s ratio=%.3f apply_to=%s",
            args.catlvdm_noise_type,
            args.catlvdm_noise_ratio,
            args.catlvdm_apply_to,
        )
