from __future__ import annotations

import logging
from typing import Any, Optional

from common.logger import get_logger
from enhancements.contrastive_attention.contrastive_attention_helper import (
    ContrastiveAttentionHelper,
)

logger = get_logger(__name__, level=logging.INFO)


def setup_contrastive_attention_helper(
    args: Any, transformer: Any, accelerator: Any
) -> Optional[ContrastiveAttentionHelper]:
    if not getattr(args, "enable_contrastive_attention", False):
        return None
    try:
        helper = ContrastiveAttentionHelper(
            transformer, args, device=accelerator.device
        )
        helper.setup_hooks()
        helper = accelerator.prepare(helper)
        logger.info("Contrastive attention helper initialized.")
        return helper
    except Exception as exc:
        logger.warning(f"Contrastive attention setup failed: {exc}")
        return None


def remove_contrastive_attention_helper(
    helper: Optional[ContrastiveAttentionHelper],
) -> None:
    if helper is None:
        return
    helper.remove_hooks()
