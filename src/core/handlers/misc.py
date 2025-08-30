"""Training utility functions."""

import torch
from typing import Any


def scale_shift_latents(latents: torch.Tensor) -> torch.Tensor:
    """Scale and shift latents if needed."""
    return latents


def record_training_step(batch_size: int) -> None:
    """Record a training step for throughput tracking."""
    from core.metrics import record_training_step as _rts
    _rts(batch_size)


def initialize_throughput_tracker(args: Any) -> None:
    """Initialize throughput tracker with configuration."""
    from core.metrics import initialize_throughput_tracker as _init_tracker

    window_size = getattr(args, "throughput_window_size", 100)
    _init_tracker(window_size)