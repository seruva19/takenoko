"""Motion disentanglement diagnostics config parsing."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def apply_motion_disentanglement_config(
    args: argparse.Namespace, config: Dict[str, Any], logger: Any
) -> argparse.Namespace:
    args.enable_motion_disentanglement_eval = _parse_bool(
        config.get("enable_motion_disentanglement_eval", False), False
    )
    args.motion_disentanglement_eval_max_items = int(
        config.get("motion_disentanglement_eval_max_items", 2)
    )
    args.motion_disentanglement_eval_frame_stride = int(
        config.get("motion_disentanglement_eval_frame_stride", 2)
    )
    args.motion_disentanglement_metrics_namespace = str(
        config.get("motion_disentanglement_metrics_namespace", "motion_eval")
    ).strip()

    if args.motion_disentanglement_eval_max_items < 1:
        raise ValueError("motion_disentanglement_eval_max_items must be >= 1")
    if args.motion_disentanglement_eval_frame_stride < 1:
        raise ValueError("motion_disentanglement_eval_frame_stride must be >= 1")
    if not args.motion_disentanglement_metrics_namespace:
        raise ValueError("motion_disentanglement_metrics_namespace must be non-empty")

    if args.enable_motion_disentanglement_eval:
        logger.info(
            "Motion disentanglement diagnostics enabled (max_items=%s, frame_stride=%s, namespace=%s).",
            args.motion_disentanglement_eval_max_items,
            args.motion_disentanglement_eval_frame_stride,
            args.motion_disentanglement_metrics_namespace,
        )
    return args

