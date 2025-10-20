"""EMA helper utilities for rCM distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _clone_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.state_dict().items()}


@dataclass(slots=True)
class RCMEMAHelper:
    """Minimal exponential moving average tracker for the student network."""

    decay: float
    shadow: dict[str, torch.Tensor]

    @classmethod
    def from_model(cls, model: nn.Module, decay: float) -> "RCMEMAHelper":
        return cls(decay=decay, shadow=_clone_params(model))

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            current_state = model.state_dict()
            for name, param in current_state.items():
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                    continue
                shadow_param = self.shadow[name]
                shadow_param.lerp_(param.detach(), 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = _clone_params(model)
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=False)
