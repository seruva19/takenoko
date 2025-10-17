from __future__ import annotations

import torch
from torch import nn


class EqMHead(nn.Module):
    """Placeholder EqM head that mirrors the diffusion output shape.

    The real EqM implementation will replace this with a head that produces the
    equilibrium gradient. For now it simply wraps an identity projection so the
    scaffolding compiles.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Identity()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
