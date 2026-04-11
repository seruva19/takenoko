"""EMA helper utilities for TwinFlow distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@dataclass(slots=True)
class TwinFlowEMAHelper:
    """Minimal EMA tracker for the trainable TwinFlow student model."""

    decay: float
    shadow: dict[str, torch.Tensor]

    @classmethod
    def from_model(cls, model: nn.Module, decay: float) -> "TwinFlowEMAHelper":
        return cls(decay=decay, shadow=_clone_state(model))

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            current = model.state_dict()
            for name, value in current.items():
                if name not in self.shadow:
                    self.shadow[name] = value.detach().clone()
                    continue
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = _clone_state(model)
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=False)
