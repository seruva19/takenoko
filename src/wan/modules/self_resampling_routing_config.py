from __future__ import annotations

from typing import Any, Dict


ALLOWED_HISTORY_ROUTING_BACKENDS = {
    "exact",
    "kernel_frame_topk",
}


def history_routing_in_range(
    config: Dict[str, Any],
    block_index: int,
) -> bool:
    start = int(config.get("start_layer_idx", 0))
    end = int(config.get("end_layer_idx", start))
    return start <= int(block_index) <= end


def normalize_history_routing_config(
    config: Dict[str, Any],
    total_layers: int,
) -> Dict[str, Any]:
    normalized = dict(config)
    start = int(normalized.get("start_layer_idx", 0))
    end = int(normalized.get("end_layer_idx", total_layers - 1))
    if start < 0:
        start = total_layers + start
    if end < 0:
        end = total_layers + end
    start = max(0, min(total_layers - 1, start))
    end = max(0, min(total_layers - 1, end))
    if end < start:
        end = start

    backend = str(normalized.get("backend", "exact")).lower()
    if backend not in ALLOWED_HISTORY_ROUTING_BACKENDS:
        backend = "exact"

    normalized["start_layer_idx"] = start
    normalized["end_layer_idx"] = end
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["backend"] = backend
    return normalized
