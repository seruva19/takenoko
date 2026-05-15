from __future__ import annotations

import argparse
from typing import Any

import torch

from common.logger import get_logger

logger = get_logger(__name__)


def _collect_new_params(existing_groups: list[Any], params: list[Any]) -> list[Any]:
    existing = set()
    for group in existing_groups:
        if isinstance(group, dict) and "params" in group:
            existing.update(id(p) for p in group["params"])
        elif isinstance(group, torch.nn.Parameter):
            existing.add(id(group))
    return [param for param in params if id(param) not in existing]


def _projector_lr(args: argparse.Namespace) -> float:
    projector_lr = getattr(args, "self_flow_projector_lr", None)
    if projector_lr is not None:
        return float(projector_lr)
    return float(getattr(args, "learning_rate", 1e-4)) * float(
        getattr(args, "input_lr_scale", 1.0)
    )


def maybe_add_m2_repa_params_for_lora(
    trainable_params: list[Any],
    lr_descriptions: list[str],
    m2_repa_helper: Any,
    args: argparse.Namespace,
) -> None:
    """Ensure M2-REPA projector parameters are included in LoRA optimizer groups."""
    params = getattr(m2_repa_helper, "get_trainable_params", None)
    if not callable(params):
        return
    new_params = _collect_new_params(trainable_params, list(params()))
    if not new_params:
        return
    lr = _projector_lr(args)
    trainable_params.append({"params": new_params, "lr": lr})
    lr_descriptions.append("m2_repa_projectors")
    logger.info(
        "M2-REPA: added %d projector params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )


def maybe_add_m2_repa_params_for_finetune(
    params_to_optimize: list[dict[str, Any]],
    param_names: list[Any],
    m2_repa_helper: Any,
    args: argparse.Namespace,
) -> None:
    """Ensure M2-REPA projector parameters are included in finetune optimizer groups."""
    params = getattr(m2_repa_helper, "get_trainable_params", None)
    if not callable(params):
        return
    new_params = _collect_new_params(params_to_optimize, list(params()))
    if not new_params:
        return
    lr = _projector_lr(args)
    params_to_optimize.append({"params": new_params, "lr": lr})
    param_names.append([f"m2_repa_projector.{i}" for i in range(len(new_params))])
    logger.info(
        "M2-REPA: added %d projector params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )
