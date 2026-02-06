from __future__ import annotations

from typing import Any, Dict

from wan.modules.self_resampling_routing_config import normalize_history_routing_config


def build_self_resampling_history_routing_config(
    config: Dict[str, Any],
    *,
    total_layers: int,
) -> Dict[str, Any]:
    return normalize_history_routing_config(config, total_layers=total_layers)
