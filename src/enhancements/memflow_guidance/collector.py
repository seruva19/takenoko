from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


class MemFlowGuidanceCollector:
    def __init__(self) -> None:
        self.enabled: bool = False
        self._loss_accum: Optional[torch.Tensor] = None
        self._loss_count: int = 0
        self._warned: bool = False

    def reset(self) -> None:
        self._loss_accum = None
        self._loss_count = 0

    def record(self, loss: torch.Tensor) -> None:
        if not self.enabled:
            return
        if self._loss_accum is None:
            self._loss_accum = loss
        else:
            self._loss_accum = self._loss_accum + loss
        self._loss_count += 1

    def consume(self) -> Optional[torch.Tensor]:
        if self._loss_accum is None or self._loss_count <= 0:
            self.reset()
            return None
        loss = self._loss_accum / float(self._loss_count)
        self.reset()
        return loss

    def warn_once(self) -> bool:
        if self._warned:
            return False
        self._warned = True
        return True

    @contextmanager
    def suspend(self):
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev
