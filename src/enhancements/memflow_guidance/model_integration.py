from __future__ import annotations

import logging
from typing import Optional

import torch

from common.logger import get_logger

from .attention_guidance import (
    install_memflow_guidance_on_attention,
    remove_memflow_guidance_from_attention,
)
from .collector import MemFlowGuidanceCollector
from .config import MemFlowGuidanceConfig

logger = get_logger(__name__, level=logging.INFO)


def enable_memflow_guidance(
    model: torch.nn.Module,
    config: MemFlowGuidanceConfig,
    collector: MemFlowGuidanceCollector,
) -> int:
    if not config.enable_memflow_guidance:
        return 0
    patched = install_memflow_guidance_on_attention(model, config, collector)
    if patched == 0:
        logger.warning("MemFlow guidance requested but no WanSelfAttention modules found")
    else:
        logger.info("MemFlow guidance patched %d WanSelfAttention modules", patched)
    return patched


def disable_memflow_guidance(model: torch.nn.Module) -> int:
    restored = remove_memflow_guidance_from_attention(model)
    if restored > 0:
        logger.info("MemFlow guidance restored %d WanSelfAttention modules", restored)
    return restored
