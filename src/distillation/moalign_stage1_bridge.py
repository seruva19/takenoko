"""Bridge wrapper that dispatches MOALIGN Stage-1 teacher training."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from common.logger import get_logger
from enhancements.moalign.stage1_trainer import train_moalign_stage1

logger = get_logger(__name__)


def dispatch_moalign_stage1_pipeline(
    *,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    raw_config_content: str,
    config_path: str,
) -> bool:
    """Main integration point called by ``UnifiedTrainer`` when Stage-1 is enabled."""

    del raw_config
    del raw_config_content

    logger.info(
        "Initialising MOALIGN Stage-1 pipeline dispatch from config '%s'",
        config_path,
    )
    try:
        checkpoint_path = train_moalign_stage1(args)
    except Exception:
        logger.exception("MOALIGN Stage-1 pipeline execution failed.")
        return False

    logger.info("MOALIGN Stage-1 pipeline completed successfully.")
    logger.info("MOALIGN Stage-1 checkpoint saved at: %s", checkpoint_path)
    return True

