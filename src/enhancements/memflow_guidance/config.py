from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class MemFlowGuidanceConfig:
    enable_memflow_guidance: bool = False
    memflow_guidance_weight: float = 0.05
    memflow_guidance_bank_size: int = 3
    memflow_guidance_record_interval: int = 3
    memflow_guidance_local_attn_size: int = 8
    memflow_guidance_top_k: int = 3
    memflow_guidance_min_frames: int = 2

    @classmethod
    def from_args(cls, args) -> "MemFlowGuidanceConfig":
        cfg = cls(
            enable_memflow_guidance=bool(
                getattr(args, "enable_memflow_guidance", False)
            ),
            memflow_guidance_weight=float(
                getattr(args, "memflow_guidance_weight", 0.05)
            ),
            memflow_guidance_bank_size=int(
                getattr(args, "memflow_guidance_bank_size", 3)
            ),
            memflow_guidance_record_interval=int(
                getattr(args, "memflow_guidance_record_interval", 3)
            ),
            memflow_guidance_local_attn_size=int(
                getattr(args, "memflow_guidance_local_attn_size", 8)
            ),
            memflow_guidance_top_k=int(
                getattr(args, "memflow_guidance_top_k", 3)
            ),
            memflow_guidance_min_frames=int(
                getattr(args, "memflow_guidance_min_frames", 2)
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not isinstance(self.enable_memflow_guidance, bool):
            raise ValueError("enable_memflow_guidance must be a boolean")
        if self.memflow_guidance_weight < 0.0:
            raise ValueError("memflow_guidance_weight must be >= 0.0")
        if self.memflow_guidance_bank_size < 1:
            raise ValueError("memflow_guidance_bank_size must be >= 1")
        if self.memflow_guidance_record_interval < 1:
            raise ValueError("memflow_guidance_record_interval must be >= 1")
        if self.memflow_guidance_local_attn_size == 0 or self.memflow_guidance_local_attn_size < -1:
            raise ValueError(
                "memflow_guidance_local_attn_size must be -1 or >= 1"
            )
        if self.memflow_guidance_top_k < 0:
            raise ValueError("memflow_guidance_top_k must be >= 0")
        if self.memflow_guidance_min_frames < 1:
            raise ValueError("memflow_guidance_min_frames must be >= 1")

    def log_summary(self) -> None:
        if not self.enable_memflow_guidance:
            return
        logger.info(
            "MemFlow guidance enabled (weight=%.3f, bank=%d, record_interval=%d, "
            "local_attn=%d, top_k=%d, min_frames=%d)",
            self.memflow_guidance_weight,
            self.memflow_guidance_bank_size,
            self.memflow_guidance_record_interval,
            self.memflow_guidance_local_attn_size,
            self.memflow_guidance_top_k,
            self.memflow_guidance_min_frames,
        )
