"""Optimizer and scheduler helpers for the RCM pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from accelerate import Accelerator
import torch.nn as nn

from common.logger import get_logger

from core.optimizer_manager import OptimizerManager
from distillation.rcm_core.config_loader import RCMConfig

logger = get_logger(__name__)


def build_rcm_optimizer(
    args: Any,
    config: RCMConfig,
    student: nn.Module,
    *,
    accelerator: Accelerator,
    raw_config: Dict[str, Any] | None = None,
    config_path: str | None = None,
) -> SimpleNamespace:
    """Create optimizer + scheduler bundles for the student network."""

    optimizer_manager = OptimizerManager()

    learning_rate = getattr(args, "learning_rate", 1e-4)
    trainable_params = [
        {
            "params": [p for p in student.parameters() if p.requires_grad],
            "lr": learning_rate,
        }
    ]

    (
        optimizer_name,
        optimizer_args,
        optimizer,
        optimizer_train_fn,
        optimizer_eval_fn,
    ) = optimizer_manager.get_optimizer(args, student, trainable_params)

    scheduler = optimizer_manager.get_lr_scheduler(
        args, optimizer, accelerator.num_processes
    )

    return SimpleNamespace(
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
    )
