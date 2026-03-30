from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

TRANSFER_TASK_TYPES = (
    "appearance",
    "temporal",
    "id",
    "style",
    "motion",
    "camera",
    "effect",
)
TRANSFER_TASK_TYPES_WITH_AUTO = ("auto",) + TRANSFER_TASK_TYPES

_TASK_ALIASES = {
    "appearance": "appearance",
    "temporal": "temporal",
    "id": "id",
    "identity": "id",
    "style": "style",
    "motion": "motion",
    "motion_transfer": "motion",
    "camera": "camera",
    "camera_movement": "camera",
    "effect": "effect",
    "effect_transfer": "effect",
    "auto": "auto",
}
_APPEARANCE_TASKS = {"appearance", "id", "style"}
_TEMPORAL_TASKS = {"temporal", "motion", "camera", "effect"}
_BATCH_TASK_KEYS = (
    "reference_conditioning_task_type",
    "task_type",
    "transfer_task",
    "semantic_task_type",
)


def normalize_transfer_task_type(value: Any) -> str:
    if value is None:
        return "auto"
    lowered = str(value).strip().lower()
    if lowered not in _TASK_ALIASES:
        raise ValueError(
            "Unsupported transfer task type "
            f"{value!r}; expected one of {sorted(TRANSFER_TASK_TYPES_WITH_AUTO)}."
        )
    return _TASK_ALIASES[lowered]


def resolve_transfer_task_type(
    args: Any,
    batch: Optional[dict[str, Any]] = None,
) -> str:
    configured = normalize_transfer_task_type(
        getattr(args, "reference_conditioning_task_type", "auto")
    )
    if configured != "auto":
        return configured

    batch_task = _resolve_batch_task_type(batch)
    if batch_task is not None:
        return batch_task

    if bool(getattr(args, "enable_ic_lora", False)):
        return "appearance"
    return "appearance"


def classify_transfer_task(task_type: str) -> str:
    canonical = normalize_transfer_task_type(task_type)
    if canonical in _APPEARANCE_TASKS:
        return "appearance"
    if canonical in _TEMPORAL_TASKS:
        return "temporal"
    return "appearance"


def transfer_task_index(task_type: str) -> int:
    canonical = normalize_transfer_task_type(task_type)
    if canonical == "auto":
        canonical = "appearance"
    return TRANSFER_TASK_TYPES.index(canonical)


def build_ic_lora_reference_positional_bias_metadata(
    args: Any,
    batch: Optional[dict[str, Any]],
    model_input: torch.Tensor,
    reference_frame_count: int,
    patch_size: Sequence[int],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (
        not bool(getattr(args, "enable_ic_lora_reference_positional_bias", False))
        or int(reference_frame_count) <= 0
        or str(getattr(args, "ic_lora_reference_positional_bias_scope", "ic_lora")).lower()
        != "ic_lora"
    ):
        return None, None

    if model_input.dim() < 5:
        return None, None

    patch_t = max(1, int(patch_size[0]))
    patch_w = max(1, int(patch_size[2]))
    total_frame_tokens = int(model_input.shape[2]) // patch_t
    total_width_tokens = int(model_input.shape[4]) // patch_w
    reference_frame_tokens = min(
        total_frame_tokens,
        max(0, int(reference_frame_count) // patch_t),
    )
    if total_frame_tokens <= 0 or total_width_tokens <= 0 or reference_frame_tokens <= 0:
        return None, None

    target_frame_tokens = max(0, total_frame_tokens - reference_frame_tokens)
    task_type = resolve_transfer_task_type(args, batch=batch)
    category = classify_transfer_task(task_type)
    offsets = torch.zeros(
        (int(model_input.shape[0]), 3),
        device=model_input.device,
        dtype=torch.long,
    )
    if category == "temporal":
        width_offset = int(
            round(
                total_width_tokens
                * float(getattr(args, "ic_lora_reference_width_position_bias_scale", 1.0))
            )
        )
        if width_offset <= 0:
            return None, None
        offsets[:, 1] = width_offset
    else:
        temporal_offset = int(
            round(
                target_frame_tokens
                * float(getattr(args, "ic_lora_reference_temporal_position_bias_scale", 1.0))
            )
        )
        if temporal_offset <= 0:
            return None, None
        offsets[:, 0] = temporal_offset

    reference_counts = torch.full(
        (int(model_input.shape[0]),),
        int(reference_frame_tokens),
        device=model_input.device,
        dtype=torch.long,
    )
    return offsets, reference_counts


def _resolve_batch_task_type(batch: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(batch, dict):
        return None
    for key in _BATCH_TASK_KEYS:
        if key not in batch:
            continue
        value = batch.get(key)
        resolved = _coerce_task_value(value)
        if resolved is not None:
            return resolved
    return None


def _coerce_task_value(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        try:
            return normalize_transfer_task_type(value)
        except ValueError:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            resolved = _coerce_task_value(item)
            if resolved is not None:
                return resolved
    return None
