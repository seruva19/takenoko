"""Placeholder module for custom CUDA/CPU operators used by RCM.

For the initial integration we rely on PyTorch reference implementations.
Later stages can fill in optimised kernels or bindings where necessary.
"""

from __future__ import annotations

__all__ = ["has_custom_ops"]


def has_custom_ops() -> bool:
    """Return ``True`` when bespoke RCM operators are available."""
    return False
