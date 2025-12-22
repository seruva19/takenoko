from __future__ import annotations

import logging
from typing import Optional

import torch

from common.logger import get_logger

from .collector import MemFlowGuidanceCollector
from .config import MemFlowGuidanceConfig

logger = get_logger(__name__, level=logging.INFO)

_collector: Optional[MemFlowGuidanceCollector] = None
_config: Optional[MemFlowGuidanceConfig] = None


def get_memflow_guidance_collector() -> MemFlowGuidanceCollector:
    global _collector
    if _collector is None:
        _collector = MemFlowGuidanceCollector()
    return _collector


def initialize_memflow_guidance_from_args(args) -> Optional[MemFlowGuidanceConfig]:
    global _config
    cfg = MemFlowGuidanceConfig.from_args(args)
    _config = cfg
    if cfg.enable_memflow_guidance:
        cfg.log_summary()
    return cfg


def get_memflow_guidance_config() -> Optional[MemFlowGuidanceConfig]:
    return _config


def begin_memflow_guidance_step() -> None:
    collector = get_memflow_guidance_collector()
    collector.reset()
    collector.enabled = True


def end_memflow_guidance_step() -> None:
    collector = get_memflow_guidance_collector()
    collector.enabled = False


def consume_memflow_guidance_loss() -> Optional[torch.Tensor]:
    collector = get_memflow_guidance_collector()
    return collector.consume()


def suspend_memflow_guidance():
    collector = get_memflow_guidance_collector()
    return collector.suspend()
