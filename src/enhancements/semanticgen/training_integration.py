from __future__ import annotations

from typing import Any, Optional

import torch

from criteria.training_loss import LossComponents


class SemanticGenTrainingState:
    """Tracks SemanticGen conditioning state during a training step."""

    def __init__(self) -> None:
        self.kl_loss: Optional[torch.Tensor] = None
        self.tokens_for_alignment: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.kl_loss = None
        self.tokens_for_alignment = None

    def apply_context(
        self,
        *,
        args: Any,
        accelerator: Any,
        context: Any,
        batch: Any,
        global_step: Optional[int],
        conditioning_helper: Any,
    ) -> Any:
        semanticgen_enabled = getattr(args, "enable_semanticgen_lora", False)
        semantic_align_enabled = getattr(args, "semantic_align_enabled", False)
        if conditioning_helper is None or not (semanticgen_enabled or semantic_align_enabled):
            return context

        if semanticgen_enabled:
            # SPEC:semanticgen_lora:conditioning - inject compressed semantic tokens.
            sem_out = conditioning_helper.build_context(
                context=context,
                batch=batch,
                global_step=global_step,
            )
            self.kl_loss = sem_out.kl_loss
            self.tokens_for_alignment = sem_out.semantic_tokens
            return sem_out.context

        tokens = conditioning_helper.get_semantic_tokens(batch)
        if tokens is not None:
            tokens = tokens.to(device=accelerator.device)
        self.tokens_for_alignment = tokens
        return context

    def apply_losses(
        self,
        *,
        args: Any,
        loss_components: LossComponents,
        alignment_helper: Any,
    ) -> None:
        if self.kl_loss is not None and getattr(args, "enable_semanticgen_lora", False):
            loss_components.total_loss = loss_components.total_loss + self.kl_loss
            loss_components.semantic_kl_loss = self.kl_loss.detach()

        if (
            alignment_helper is not None
            and getattr(args, "semantic_align_enabled", False)
        ):
            # SPEC:semanticgen_lora:alignment - add semantic alignment loss.
            align_result = alignment_helper.compute_loss(self.tokens_for_alignment)
            if align_result.loss is not None:
                loss_components.total_loss = loss_components.total_loss + align_result.loss
                loss_components.semantic_align_loss = align_result.loss.detach()
                if align_result.similarity is not None:
                    loss_components.semantic_align_similarity = (
                        align_result.similarity.detach()
                    )
