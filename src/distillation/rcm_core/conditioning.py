"""Lightweight conditioning wrappers for rCM distillation.

The upstream rCM implementation wraps text embeddings inside ``TextCondition``
objects that carry modality hints and helper methods such as ``to_dict`` or
``broadcast``.  This module provides a trimmed version suitable for the
Takenoko integration while keeping the public API familiar so we can port the
trainer logic with minimal friction in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch


ConditionPayload = Dict[str, torch.Tensor]


@dataclass(slots=True)
class TextCondition:
    """Container for conditional inputs consumed by the WAN DiT."""

    encoder_hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None
    data_type: str = "video"  # "video" or "image"

    def to_dict(self) -> ConditionPayload:
        payload: ConditionPayload = {"encoder_hidden_states": self.encoder_hidden_states}
        if self.attention_mask is not None:
            payload["attention_mask"] = self.attention_mask
        if self.pooled_output is not None:
            payload["pooled_output"] = self.pooled_output
        return payload

    def edit_data_type(self, data_type: str) -> "TextCondition":
        return replace(self, data_type=data_type)

    def broadcast(self, _process_group: Optional[object]) -> "TextCondition":
        # Context parallelism is not yet supported in the Takenoko integration;
        # return self unchanged to mimic the upstream API.
        return self

    @property
    def is_video(self) -> bool:
        return self.data_type == "video"


def _stack_tensor_candidate(value: Optional[object]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value:
        if all(isinstance(elem, torch.Tensor) for elem in value):
            return torch.stack(list(value), dim=0)
    return None


def build_text_condition_from_payload(
    payload: Dict[str, object],
    *,
    positive_key: str,
    mask_key: Optional[str] = None,
    pooled_key: Optional[str] = None,
    data_type: str = "video",
) -> Optional[TextCondition]:
    """Construct a :class:`TextCondition` from replay payload entries."""

    embeddings = _stack_tensor_candidate(payload.get(positive_key))
    if embeddings is None:
        return None
    mask = _stack_tensor_candidate(payload.get(mask_key)) if mask_key else None
    pooled = _stack_tensor_candidate(payload.get(pooled_key)) if pooled_key else None
    return TextCondition(
        encoder_hidden_states=embeddings,
        attention_mask=mask,
        pooled_output=pooled,
        data_type=data_type,
    )


def build_condition_pair(
    payload: Dict[str, object],
    *,
    modality: str,
) -> Tuple[Optional[TextCondition], Optional[TextCondition]]:
    """Return (condition, uncondition) text wrappers when available."""

    condition = build_text_condition_from_payload(
        payload,
        positive_key="t5",
        mask_key="t5_attention_mask",
        pooled_key="t5_pooled_output",
        data_type=modality,
    )
    uncondition = build_text_condition_from_payload(
        payload,
        positive_key="t5_negative",
        mask_key="t5_negative_attention_mask",
        pooled_key="t5_negative_pooled_output",
        data_type=modality,
    )
    return condition, uncondition
