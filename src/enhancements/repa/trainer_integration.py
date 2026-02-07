from __future__ import annotations

import argparse
from typing import Any

import torch

from common.logger import get_logger

logger = get_logger(__name__)


def maybe_add_vae_repa_params_for_lora(
    trainable_params: list[Any],
    lr_descriptions: list[str],
    vae_repa_helper: Any,
    args: argparse.Namespace,
) -> None:
    """Ensure VAE-REPA projector parameters are included in LoRA optimizer groups."""
    params = getattr(vae_repa_helper, "get_trainable_params", None)
    if not callable(params):
        return
    trainable = list(params())
    if not trainable:
        return
    existing = set()
    for group in trainable_params:
        if isinstance(group, dict) and "params" in group:
            existing.update(id(p) for p in group["params"])
        elif isinstance(group, torch.nn.Parameter):
            existing.add(id(group))
    new_params = [p for p in trainable if id(p) not in existing]
    if not new_params:
        return
    lr = float(getattr(args, "learning_rate", 1e-4)) * float(
        getattr(args, "input_lr_scale", 1.0)
    )
    trainable_params.append({"params": new_params, "lr": lr})
    lr_descriptions.append("vae_repa_projector")
    logger.info(
        "VAE-REPA: added %d projector params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )


def maybe_add_vae_repa_params_for_finetune(
    params_to_optimize: list[dict[str, Any]],
    param_names: list[list[str]],
    vae_repa_helper: Any,
    args: argparse.Namespace,
) -> None:
    """Ensure VAE-REPA projector parameters are included in finetune optimizer groups."""
    params = getattr(vae_repa_helper, "get_trainable_params", None)
    if not callable(params):
        return
    trainable = list(params())
    if not trainable:
        return
    existing = set()
    for group in params_to_optimize:
        if "params" in group:
            existing.update(id(p) for p in group["params"])
    new_params = [p for p in trainable if id(p) not in existing]
    if not new_params:
        return
    lr = float(getattr(args, "learning_rate", 1e-4)) * float(
        getattr(args, "input_lr_scale", 1.0)
    )
    params_to_optimize.append({"params": new_params, "lr": lr})
    param_names.append([f"vae_repa_projector.{i}" for i in range(len(new_params))])
    logger.info(
        "VAE-REPA: added %d projector params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )
