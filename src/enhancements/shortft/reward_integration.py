"""ShortFT integration helpers for reward LoRA training.

This module implements a train-time-only progressive shortcut mask that
shortens the effective backpropagation chain while preserving the standard
inference denoising path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class ShortFTBackpropPlan:
    """Per-step backprop plan for one reward-sampling rollout."""

    mask: List[bool]
    stage_index: int
    stage_count: int
    dense_segments: int
    total_segments: int
    anchor_count: int
    total_backprop_steps: int


def _split_step_indices(num_steps: int, num_segments: int) -> List[List[int]]:
    segments: List[List[int]] = []
    for seg_idx in range(num_segments):
        start = math.floor(seg_idx * num_steps / num_segments)
        end = math.floor((seg_idx + 1) * num_steps / num_segments)
        segments.append(list(range(start, end)))
    return segments


def _resolve_stage_index(
    *,
    global_step: int,
    max_train_steps: int,
    start_step: int,
    stage_transition_steps: int,
    stage_count: int,
) -> int:
    if stage_count <= 1:
        return 0
    if global_step <= start_step:
        return 0

    transition = int(stage_transition_steps)
    if transition <= 0:
        remaining = max(1, int(max_train_steps) - int(start_step))
        transition = max(1, math.ceil(remaining / stage_count))

    stage = (int(global_step) - int(start_step)) // transition
    if stage < 0:
        return 0
    if stage >= stage_count:
        return stage_count - 1
    return stage


def _collect_shortcut_anchor_indices(
    *,
    segments: Sequence[Sequence[int]],
    dense_segment_max: int,
    anchor_count: int,
) -> Set[int]:
    anchors: Set[int] = set()
    for seg_idx, indices in enumerate(segments):
        if seg_idx <= dense_segment_max or not indices:
            continue
        keep = min(len(indices), max(1, int(anchor_count)))
        anchors.update(indices[-keep:])
    return anchors


def build_shortft_backprop_plan(
    *,
    args: Any,
    base_backprop_mask: Sequence[bool],
    num_inference_steps: int,
    global_step: int,
    max_train_steps: int,
) -> Optional[ShortFTBackpropPlan]:
    """Build a progressive ShortFT backprop mask.

    Returns None when ShortFT is disabled.
    """
    if not bool(getattr(args, "enable_shortft", False)):
        return None

    if num_inference_steps <= 0:
        return None

    if len(base_backprop_mask) != num_inference_steps:
        raise ValueError(
            "base_backprop_mask length must match num_inference_steps "
            f"({len(base_backprop_mask)} != {num_inference_steps})"
        )

    total_segments = max(2, int(getattr(args, "shortft_num_segments", 4)))
    total_segments = min(total_segments, num_inference_steps)
    segments = _split_step_indices(num_inference_steps, total_segments)

    stage_index = _resolve_stage_index(
        global_step=global_step,
        max_train_steps=max_train_steps,
        start_step=int(getattr(args, "shortft_start_step", 0)),
        stage_transition_steps=int(getattr(args, "shortft_stage_transition_steps", 0)),
        stage_count=total_segments,
    )

    dense_mode = str(getattr(args, "shortft_dense_backprop_mode", "all")).lower()
    anchor_count = max(1, int(getattr(args, "shortft_shortcut_anchor_count", 1)))
    anchors = _collect_shortcut_anchor_indices(
        segments=segments,
        dense_segment_max=stage_index,
        anchor_count=anchor_count,
    )

    mask: List[bool] = [False] * num_inference_steps
    for seg_idx, indices in enumerate(segments):
        for i in indices:
            if seg_idx <= stage_index:
                mask[i] = True if dense_mode == "all" else bool(base_backprop_mask[i])
            else:
                mask[i] = i in anchors

    return ShortFTBackpropPlan(
        mask=mask,
        stage_index=stage_index,
        stage_count=total_segments,
        dense_segments=stage_index + 1,
        total_segments=total_segments,
        anchor_count=anchor_count,
        total_backprop_steps=sum(1 for x in mask if x),
    )


def shortft_plan_to_metrics(plan: Optional[ShortFTBackpropPlan]) -> Dict[str, float]:
    """Convert a ShortFT plan to scalar metrics for tracker logging."""
    if plan is None:
        return {}
    denom = max(1, len(plan.mask))
    return {
        "shortft/stage_index": float(plan.stage_index),
        "shortft/stage_count": float(plan.stage_count),
        "shortft/dense_segments": float(plan.dense_segments),
        "shortft/total_segments": float(plan.total_segments),
        "shortft/anchor_count": float(plan.anchor_count),
        "shortft/backprop_steps": float(plan.total_backprop_steps),
        "shortft/backprop_ratio": float(plan.total_backprop_steps / denom),
    }
