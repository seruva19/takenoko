"""Entry-point wrapper that wires Takenoko config and resources into the TwinFlow core."""

from __future__ import annotations

import argparse
import copy
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import torch

from common.logger import get_logger
from core.checkpoint_manager import CheckpointManager
from distillation.twinflow_core.config_loader import load_twinflow_config
from distillation.twinflow_core.runner import run_twinflow

logger = get_logger(__name__)


def dispatch_twinflow_pipeline(
    *,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    raw_config_content: str,
    config_path: str,
) -> bool:
    """Main integration point called by ``UnifiedTrainer`` when TwinFlow is enabled."""

    logger.info("Initialising TwinFlow pipeline dispatch from config '%s'", config_path)

    twinflow_config = load_twinflow_config(args)

    try:
        run_twinflow(
            args=args,
            raw_config=raw_config,
            raw_config_content=raw_config_content,
            config_path=config_path,
            config=twinflow_config,
            checkpoint_callback=_build_checkpoint_callback(args, raw_config, twinflow_config),
        )
    except NotImplementedError as exc:
        logger.error("TwinFlow pipeline is not fully implemented: %s", exc)
        raise
    except Exception:
        logger.exception("TwinFlow pipeline execution failed.")
        return False

    logger.info("TwinFlow pipeline completed successfully.")
    return True


def _build_checkpoint_callback(
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    twinflow_config: Any,
) -> Optional[Any]:
    base_output = Path(getattr(args, "output_dir", "") or raw_config.get("output_dir", "output"))
    output_root = base_output / "twinflow"
    output_root.mkdir(parents=True, exist_ok=True)

    args_for_checkpoint = copy.copy(args)
    args_for_checkpoint.output_dir = str(output_root)

    checkpoint_manager = CheckpointManager()
    metadata = {
        "twinflow_teacher": getattr(args, "dit", ""),
        "twinflow_variant": getattr(twinflow_config, "trainer_variant", "full"),
        "twinflow_target": getattr(args, "target_model", "wan21"),
    }
    minimum_metadata: Dict[str, str] = {}

    dit_dtype = getattr(args, "dit_dtype", torch.float16)
    save_model_fn = checkpoint_manager.create_save_model_function(
        args=args_for_checkpoint,
        metadata=metadata,
        minimum_metadata=minimum_metadata,
        dit_dtype=dit_dtype,
    )
    remove_model_fn = checkpoint_manager.create_remove_model_function(args_for_checkpoint)

    keep_last = 3
    saved_history: Deque[str] = deque()

    def _callback(state: Dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            return

        step = int(state.get("step", 0))
        ckpt_name = f"twinflow_step_{step:06d}.safetensors"
        save_model_fn(
            ckpt_name,
            model,
            step,
            state.get("epoch", 0),
            force_sync_upload=False,
        )
        logger.info("TwinFlow checkpoint written to %s", output_root / ckpt_name)

        saved_history.append(ckpt_name)
        while len(saved_history) > keep_last:
            remove_model_fn(saved_history.popleft())

    return _callback
