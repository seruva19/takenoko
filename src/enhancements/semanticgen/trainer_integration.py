from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__)


def get_semantic_cache_requirements(args: Any) -> dict[str, Any]:
    return {
        "require_semantic_encoder_cache": bool(
            getattr(args, "semantic_cache_enabled", False)
            and getattr(args, "semantic_cache_require", False)
        ),
        "semantic_cache_directory": getattr(args, "semantic_cache_directory", None),
    }


def build_semantic_prepare_items(
    *,
    network: Any,
    controlnet: Optional[Any],
    optimizer: Any,
    train_dataloader: Any,
    lr_scheduler: Any,
    semantic_conditioning_helper: Optional[Any],
    semantic_alignment_helper: Optional[Any],
    bfm_conditioning_helper: Optional[Any],
) -> tuple[list[Any], list[str]]:
    items: list[Any] = [network]
    slots: list[str] = ["network"]
    if controlnet is not None:
        items.append(controlnet)
        slots.append("controlnet")
    if semantic_conditioning_helper is not None:
        items.append(semantic_conditioning_helper)
        slots.append("semantic_conditioning_helper")
    if semantic_alignment_helper is not None:
        items.append(semantic_alignment_helper)
        slots.append("semantic_alignment_helper")
    if bfm_conditioning_helper is not None:
        items.append(bfm_conditioning_helper)
        slots.append("bfm_conditioning_helper")
    items.extend([optimizer, train_dataloader, lr_scheduler])
    slots.extend(["optimizer", "train_dataloader", "lr_scheduler"])
    return items, slots


def create_semantic_helpers(
    args: Any,
    transformer: Any,
    accelerator: Any,
    trainable_params: list[Any],
    lr_descriptions: list[str],
) -> Tuple[Optional[Any], Optional[Any]]:
    semantic_conditioning_helper = None
    semantic_alignment_helper = None
    if getattr(args, "enable_semanticgen_lora", False) or getattr(
        args, "semantic_align_enabled", False
    ):
        try:
            from enhancements.semanticgen.semantic_conditioning import (
                SemanticConditioningHelper,
            )

            text_dim = _infer_text_context_dim(transformer)
            semantic_conditioning_helper = SemanticConditioningHelper(
                args, text_dim, accelerator.device
            )
            _maybe_add_semanticgen_params(
                trainable_params,
                lr_descriptions,
                semantic_conditioning_helper,
                args,
            )
            logger.info("SemanticGen LoRA conditioning helper initialized.")
        except Exception as exc:
            logger.warning(f"SemanticGen conditioning setup failed: {exc}")
            semantic_conditioning_helper = None

    if getattr(args, "semantic_align_enabled", False):
        try:
            from enhancements.semanticgen.semantic_alignment import (
                SemanticAlignmentHelper,
            )

            semantic_alignment_helper = SemanticAlignmentHelper(transformer, args)
            _maybe_add_semanticgen_params(
                trainable_params,
                lr_descriptions,
                semantic_alignment_helper,
                args,
            )
            logger.info("SemanticGen alignment helper initialized.")
        except Exception as exc:
            logger.warning(f"Semantic alignment setup failed: {exc}")
            semantic_alignment_helper = None

    return semantic_conditioning_helper, semantic_alignment_helper


def setup_semantic_training_integration(
    *,
    args: Any,
    transformer: Any,
    accelerator: Any,
    training_core: Any,
    trainable_params: Optional[list[Any]] = None,
    lr_descriptions: Optional[list[str]] = None,
    semantic_conditioning_helper: Optional[Any] = None,
    semantic_alignment_helper: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    if semantic_conditioning_helper is None and semantic_alignment_helper is None:
        if trainable_params is not None and lr_descriptions is not None:
            semantic_conditioning_helper, semantic_alignment_helper = (
                create_semantic_helpers(
                    args=args,
                    transformer=transformer,
                    accelerator=accelerator,
                    trainable_params=trainable_params,
                    lr_descriptions=lr_descriptions,
                )
            )
        else:
            semantic_conditioning_helper = None
            semantic_alignment_helper = None
    attach_semantic_helpers_to_training_core(
        training_core=training_core,
        transformer=transformer,
        semantic_conditioning_helper=semantic_conditioning_helper,
        semantic_alignment_helper=semantic_alignment_helper,
    )
    return semantic_conditioning_helper, semantic_alignment_helper


def teardown_semantic_training_integration(
    semantic_alignment_helper: Optional[Any],
) -> None:
    cleanup_semantic_helpers(semantic_alignment_helper)


def attach_semantic_helpers_to_training_core(
    *,
    training_core: Any,
    transformer: Any,
    semantic_conditioning_helper: Optional[Any],
    semantic_alignment_helper: Optional[Any],
) -> None:
    if semantic_alignment_helper is not None:
        semantic_alignment_helper.diffusion_model = transformer
        semantic_alignment_helper.setup_hooks()
    # SPEC:semanticgen_lora:conditioning - attach semantic helpers to the core.
    training_core.semantic_conditioning_helper = semantic_conditioning_helper
    training_core.semantic_alignment_helper = semantic_alignment_helper


def cleanup_semantic_helpers(semantic_alignment_helper: Optional[Any]) -> None:
    if semantic_alignment_helper is not None:
        semantic_alignment_helper.remove_hooks()


def _maybe_add_semanticgen_params(
    trainable_params: list[Any],
    lr_descriptions: list[str],
    helper: Any,
    args: Any,
) -> None:
    params = getattr(helper, "get_trainable_params", None)
    if not callable(params):
        return
    trainable = list(params())
    if not trainable:
        return
    existing = set()
    for group in trainable_params:
        if isinstance(group, dict) and "params" in group:
            existing.update(id(p) for p in group["params"])
        elif isinstance(group, torch.nn.Parameter):
            existing.add(id(group))
    new_params = [p for p in trainable if id(p) not in existing]
    if not new_params:
        return
    lr = float(getattr(args, "semanticgen_lr", args.learning_rate))
    trainable_params.append({"params": new_params, "lr": lr})
    lr_descriptions.append("semanticgen_helper")
    logger.info(
        "SemanticGen: added %d helper params to optimizer groups (lr=%.6f).",
        len(new_params),
        lr,
    )


def _infer_text_context_dim(transformer: Any) -> int:
    if hasattr(transformer, "text_embedding") and transformer.text_embedding:
        layer = transformer.text_embedding[0]
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
    return int(getattr(transformer, "dim", 1024))
