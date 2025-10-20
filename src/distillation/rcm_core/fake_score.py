"""Auxiliary fake-score head for rCM distillation."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRCMFakeScoreHead(nn.Module):
    """Lightweight critic that mirrors the student embedding dimensionality."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.view(inputs.size(0), -1))
