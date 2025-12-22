from __future__ import annotations

from typing import Any

from .config import MemFlowGuidanceConfig
from .model_integration import disable_memflow_guidance, enable_memflow_guidance
from .training_integration import get_memflow_guidance_collector


def setup_hooks(model: Any, args: Any) -> bool:
    cfg = MemFlowGuidanceConfig.from_args(args)
    if not cfg.enable_memflow_guidance:
        return False
    enable_memflow_guidance(model, cfg, get_memflow_guidance_collector())
    return True


def remove_hooks(model: Any) -> None:
    disable_memflow_guidance(model)
