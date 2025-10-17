from __future__ import annotations

import math
from typing import Optional


def compute_weight_scale(
    schedule: Optional[str],
    total_steps: Optional[int],
    current_step: int,
) -> Optional[float]:
    """
    Compute a scalar multiplier for EqM transport loss weighting.

    Args:
        schedule: Name of schedule ('linear', 'cosine', 'sigmoid').
        total_steps: Number of steps over which to ramp the weight.
        current_step: Current global step (0-indexed).

    Returns:
        A float in [0, 1] when a schedule is active; otherwise None.
    """
    if schedule is None or total_steps is None:
        return None

    try:
        total_steps = int(total_steps)
    except (TypeError, ValueError):
        return None

    if total_steps <= 0:
        return None

    progress = max(0.0, min(float(current_step) / float(total_steps), 1.0))
    schedule = schedule.lower()

    if schedule == "linear":
        weight = progress
    elif schedule == "cosine":
        weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif schedule == "sigmoid":
        weight = 1.0 / (1.0 + math.exp(-12.0 * (progress - 0.5)))
    else:
        return None

    return max(0.0, min(weight, 1.0))
