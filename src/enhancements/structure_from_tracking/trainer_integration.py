"""Trainer-side integration helpers for Structure-From-Tracking."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from common.logger import get_logger

logger = get_logger(__name__)


def maybe_precompute_sft_teacher_cache_before_training(
    args: argparse.Namespace,
) -> None:
    """Run optional SFT teacher cache precompute before training starts."""
    if not bool(getattr(args, "enable_structure_from_tracking", False)):
        return
    if not bool(getattr(args, "sft_teacher_cache_before_training", False)):
        return

    overwrite_existing = bool(
        getattr(args, "sft_teacher_cache_overwrite_existing", False)
    )
    logger.info(
        "Structure-From-Tracking: pre-train teacher cache precompute enabled "
        "(overwrite_existing=%s).",
        overwrite_existing,
    )
    from caching.cache_sft_teacher_action import (
        build_sft_cache_precompute_args,
        run_cache_sft_teacher_action,
    )

    cache_args = build_sft_cache_precompute_args(
        args,
        overwrite_existing=overwrite_existing,
    )
    success = run_cache_sft_teacher_action(cache_args)
    if not success:
        raise RuntimeError("Structure-From-Tracking teacher cache precompute failed.")

    if str(getattr(args, "sft_teacher_cache_mode", "off")).lower() == "off":
        args.sft_teacher_cache_mode = "read_write"
        logger.info(
            "Structure-From-Tracking: sft_teacher_cache_mode was 'off'; switched to "
            "'read_write' for training to reuse precomputed teacher features."
        )


def maybe_add_structure_from_tracking_params(
    trainable_params: list[Any],
    lr_descriptions: list[str],
    sft_helper: Any,
    args: argparse.Namespace,
) -> None:
    """Ensure SFT projector parameters are included in optimizer groups."""
    params = getattr(sft_helper, "get_trainable_params", None)
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
    lr_descriptions.append("structure_from_tracking_projector")
    logger.info(
        "Structure-From-Tracking: added %d projector params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )
