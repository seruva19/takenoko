from __future__ import annotations

# Backward-compatible shim. New code should import from:
# - wan.modules.self_resampling_routing_config
# - wan.modules.self_resampling_routing_attention

from wan.modules.self_resampling_routing_config import (
    history_routing_in_range,
    normalize_history_routing_config,
)
from wan.modules.self_resampling_routing_attention import (
    history_branch_attention,
)

__all__ = [
    "history_routing_in_range",
    "normalize_history_routing_config",
    "history_branch_attention",
]
