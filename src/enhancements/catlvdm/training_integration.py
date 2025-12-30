"""Training integration helpers for CAT-LVDM corruption-aware conditioning."""
from __future__ import annotations

from typing import Optional

from common.logger import get_logger

from enhancements.catlvdm.corruption_helper import CatLVDMCorruptionHelper

logger = get_logger(__name__)


def create_catlvdm_corruption_helper(args) -> Optional[CatLVDMCorruptionHelper]:
    if not getattr(args, "enable_catlvdm_corruption", False):
        return None
    noise_ratio = float(getattr(args, "catlvdm_noise_ratio", 0.0))
    if noise_ratio <= 0:
        logger.warning("CAT-LVDM corruption enabled but noise ratio <= 0; skipping.")
        return None
    noise_type = str(getattr(args, "catlvdm_noise_type", "bcni")).lower()
    return CatLVDMCorruptionHelper(noise_type=noise_type, noise_ratio=noise_ratio)
