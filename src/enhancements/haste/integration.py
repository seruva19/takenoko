from __future__ import annotations

import logging
from typing import Any, Optional

from common.logger import get_logger
from enhancements.haste.haste_helper import HasteHelper

logger = get_logger(__name__, level=logging.INFO)


def setup_haste_helper(
    args: Any, transformer: Any, accelerator: Any
) -> Optional[HasteHelper]:
    if not getattr(args, "enable_haste", False):
        return None
    try:
        helper = HasteHelper(transformer, args, device=accelerator.device)
        helper.setup_hooks()
        helper = accelerator.prepare(helper)
        logger.info("HASTE helper initialized.")
        return helper
    except Exception as exc:
        logger.warning(f"HASTE setup failed: {exc}")
        return None


def add_haste_params(optimizer: Any, helper: Optional[HasteHelper], args: Any) -> None:
    if helper is None or optimizer is None:
        return
    params = getattr(helper, "get_trainable_params", None)
    if not callable(params):
        return
    trainable = list(params())
    if not trainable:
        return
    existing = {
        id(p) for group in optimizer.param_groups for p in group.get("params", [])
    }
    new_params = [p for p in trainable if id(p) not in existing]
    if not new_params:
        return
    lr = float(getattr(args, "learning_rate", 1e-4)) * float(
        getattr(args, "input_lr_scale", 1.0)
    )
    optimizer.add_param_group({"params": new_params, "lr": lr})
    logger.info("HASTE: added %d projector params to optimizer (lr=%.6f).", len(new_params), lr)


def remove_haste_helper(helper: Optional[HasteHelper]) -> None:
    if helper is None:
        return
    helper.remove_hooks()
