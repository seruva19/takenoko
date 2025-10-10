from __future__ import annotations

from typing import Any, Optional


def resolve_dispersive_target_block(
    num_blocks: int, target: Any
) -> Optional[int]:
    """Convert a user-specified target block into a valid index.

    Accepts integers, negative indices, or string aliases such as "last".
    Returns ``None`` if the request is invalid or if the model has no blocks.
    """
    if num_blocks <= 0:
        return None
    if target is None:
        return None

    candidate = target
    try:
        if isinstance(candidate, str):
            lowered = candidate.strip().lower()
            if lowered in {"last", "final", "end"}:
                return num_blocks - 1
            candidate = int(candidate, 10)
        else:
            candidate = int(candidate)
    except (TypeError, ValueError):
        return None

    if candidate < 0:
        candidate = num_blocks + candidate

    if 0 <= candidate < num_blocks:
        return candidate
    return None
