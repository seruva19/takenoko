from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class MotionReplayHealthTracker:
    anchor_cache_size: int = 0
    temporal_anchor_count: int = 0
    synthetic_anchor_count: int = 0
    dataset_anchor_count: int = 0
    cache_loaded: bool = False
    cache_built: bool = False
    invocations: int = 0
    applied: int = 0
    skipped_schedule: int = 0
    skipped_zero_weight: int = 0
    skipped_no_anchor: int = 0
    skipped_invalid_anchor: int = 0
    attention_applied: int = 0
    temporal_fallbacks: int = 0
    last_error: str = ""
    extra_skips: Dict[str, int] = field(default_factory=dict)

    def note_cache(self, anchors: Iterable[Any], *, loaded: bool) -> None:
        anchor_list = list(anchors)
        self.anchor_cache_size = len(anchor_list)
        self.temporal_anchor_count = 0
        self.synthetic_anchor_count = 0
        self.dataset_anchor_count = 0
        for anchor in anchor_list:
            source = str(getattr(anchor, "source", "dataset") or "dataset").lower()
            if source == "synthetic":
                self.synthetic_anchor_count += 1
            else:
                self.dataset_anchor_count += 1
            latents = getattr(anchor, "latents", None)
            if getattr(latents, "dim", lambda: 0)() == 5 and int(latents.shape[2]) > 1:
                self.temporal_anchor_count += 1
        self.cache_loaded = bool(loaded)
        self.cache_built = not loaded and self.anchor_cache_size > 0

    def note_skip(self, reason: str) -> None:
        self.invocations += 1
        if reason == "schedule":
            self.skipped_schedule += 1
        elif reason == "zero_weight":
            self.skipped_zero_weight += 1
        elif reason == "no_anchor":
            self.skipped_no_anchor += 1
        elif reason == "invalid_anchor":
            self.skipped_invalid_anchor += 1
        else:
            self.extra_skips[reason] = self.extra_skips.get(reason, 0) + 1

    def note_applied(
        self,
        *,
        attention_applied: bool,
        temporal_fallback: bool,
    ) -> None:
        self.invocations += 1
        self.applied += 1
        if attention_applied:
            self.attention_applied += 1
        if temporal_fallback:
            self.temporal_fallbacks += 1

    def note_error(self, message: str) -> None:
        self.last_error = str(message or "")

    def as_dict(self) -> Dict[str, Any]:
        invocations = max(1, int(self.invocations))
        applied = int(self.applied)
        temporal_anchor_count = int(self.temporal_anchor_count)
        anchor_cache_size = max(1, int(self.anchor_cache_size))
        attention_apply_rate = float(self.attention_applied) / float(max(1, applied))
        zero_weight_skip_rate = float(self.skipped_zero_weight) / float(invocations)
        no_anchor_skip_rate = float(self.skipped_no_anchor) / float(invocations)
        invalid_anchor_skip_rate = float(self.skipped_invalid_anchor) / float(invocations)
        return {
            "anchor_cache_size": int(self.anchor_cache_size),
            "temporal_anchor_count": int(self.temporal_anchor_count),
            "synthetic_anchor_count": int(self.synthetic_anchor_count),
            "dataset_anchor_count": int(self.dataset_anchor_count),
            "cache_loaded": bool(self.cache_loaded),
            "cache_built": bool(self.cache_built),
            "invocations": int(self.invocations),
            "applied": applied,
            "skipped_schedule": int(self.skipped_schedule),
            "skipped_zero_weight": int(self.skipped_zero_weight),
            "skipped_no_anchor": int(self.skipped_no_anchor),
            "skipped_invalid_anchor": int(self.skipped_invalid_anchor),
            "attention_applied": int(self.attention_applied),
            "temporal_fallbacks": int(self.temporal_fallbacks),
            "apply_rate": float(applied) / float(invocations),
            "attention_apply_rate": attention_apply_rate,
            "schedule_skip_rate": float(self.skipped_schedule) / float(invocations),
            "zero_weight_skip_rate": zero_weight_skip_rate,
            "no_anchor_skip_rate": no_anchor_skip_rate,
            "invalid_anchor_skip_rate": invalid_anchor_skip_rate,
            "temporal_fallback_rate": float(self.temporal_fallbacks) / float(max(1, applied)),
            "temporal_anchor_ratio": float(temporal_anchor_count) / float(anchor_cache_size),
            "last_error": self.last_error,
            "extra_skips": dict(self.extra_skips),
        }

    def publish_to_args(self, args: Any) -> None:
        prefix = "_motion_preservation_runtime_"
        snapshot = self.as_dict()
        for key, value in snapshot.items():
            setattr(args, prefix + key, value)

    def load_dict(self, payload: Dict[str, Any]) -> None:
        for key in (
            "anchor_cache_size",
            "temporal_anchor_count",
            "synthetic_anchor_count",
            "dataset_anchor_count",
            "cache_loaded",
            "cache_built",
            "invocations",
            "applied",
            "skipped_schedule",
            "skipped_zero_weight",
            "skipped_no_anchor",
            "skipped_invalid_anchor",
            "attention_applied",
            "temporal_fallbacks",
            "last_error",
        ):
            if key in payload:
                setattr(self, key, payload[key])
        extra = payload.get("extra_skips")
        if isinstance(extra, dict):
            self.extra_skips = {str(k): int(v) for k, v in extra.items()}
