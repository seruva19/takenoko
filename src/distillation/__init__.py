"""Distillation subsystem package.

This namespace hosts alternate training pipelines such as the RCM
distillation flow. Modules are intentionally lightweight and decoupled
from WAN training so that optional features can be imported lazily.
"""

__all__ = ["rcm_core"]
