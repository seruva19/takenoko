from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class SemanticAlignResult:
    loss: Optional[torch.Tensor]
    similarity: Optional[torch.Tensor]


class SemanticAlignmentHelper(nn.Module):
    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.block_index = int(getattr(args, "semantic_align_block_index", 8))
        self.align_lambda = float(getattr(args, "semantic_align_lambda", 0.1))
        self._hook_handle: Optional[Any] = None
        self._cached_features: Optional[torch.Tensor] = None
        self.semantic_embed_dim = int(getattr(args, "semantic_embed_dim", 1024))
        self._proj: Optional[nn.Linear] = _init_projection(
            diffusion_model, self.semantic_embed_dim
        )

    def setup_hooks(self) -> None:
        # SPEC:semanticgen_lora:alignment - capture diffusion block features to align model semantics with encoder embeddings.
        target_module = _resolve_block(self.diffusion_model, self.block_index)
        self._hook_handle = target_module.register_forward_hook(self._get_hook())
        logger.info(
            "Semantic alignment hook attached to block %d.", self.block_index
        )

    def remove_hooks(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info("Semantic alignment hook removed.")

    def _get_hook(self):
        def hook(module, inputs, outputs):
            features = outputs[0] if isinstance(outputs, tuple) else outputs
            self._cached_features = features

        return hook

    def compute_loss(
        self, semantic_tokens: Optional[torch.Tensor]
    ) -> SemanticAlignResult:
        if (
            semantic_tokens is None
            or self._cached_features is None
            or self.align_lambda <= 0
        ):
            return SemanticAlignResult(None, None)

        features = self._cached_features
        if features.dim() == 4:
            features = features.flatten(1, 2)
        if features.dim() != 3:
            return SemanticAlignResult(None, None)

        sem_tokens = semantic_tokens
        if sem_tokens.dim() == 2:
            sem_tokens = sem_tokens.unsqueeze(1)

        pooled_features = features.mean(dim=1)
        pooled_semantics = sem_tokens.mean(dim=1)

        if self._proj is None or self._proj.out_features != pooled_semantics.shape[-1]:
            self._proj = nn.Linear(
                pooled_features.shape[-1], pooled_semantics.shape[-1]
            ).to(pooled_features.device)
        elif self._proj.weight.device != pooled_features.device:
            self._proj = self._proj.to(pooled_features.device)

        projected = self._proj(pooled_features)
        projected = F.normalize(projected, dim=-1)
        pooled_semantics = F.normalize(pooled_semantics, dim=-1)
        similarity = (projected * pooled_semantics).sum(dim=-1).mean()
        loss = (1.0 - similarity) * self.align_lambda
        return SemanticAlignResult(loss, similarity)

    def get_trainable_params(self) -> list[nn.Parameter]:
        if self._proj is None:
            return []
        return list(self._proj.parameters())


def _resolve_block(model: Any, block_index: int) -> Any:
    if hasattr(model, "blocks"):
        return model.blocks[block_index]
    if hasattr(model, "transformer_blocks"):
        return model.transformer_blocks[block_index]
    if hasattr(model, "layers"):
        return model.layers[block_index]
    raise ValueError("Semantic alignment: no transformer blocks found.")


def _init_projection(model: Any, target_dim: int) -> Optional[nn.Linear]:
    if not hasattr(model, "dim"):
        return None
    in_dim = int(getattr(model, "dim"))
    return nn.Linear(in_dim, target_dim)
