"""DION2 optimizer creation helper for WAN network trainer."""

from __future__ import annotations

import fnmatch
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def _coerce_pair(name: str, value: Any) -> Tuple[float, float]:
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence, got {value!r}")
    return float(value[0]), float(value[1])


def create_dion2_optimizer(
    transformer: Optional[torch.nn.Module],
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    distributed_context: Optional[Dict[str, Any]],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [
            str,
            str,
            List[Any],
            List[torch.nn.Parameter],
            List[torch.nn.Parameter],
            List[torch.nn.Parameter],
        ],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using DION2 optimizer | {optimizer_kwargs}")

    from optimizers.dion2 import (
        SingleDeviceDion2WithAuxAdam,
        apply_dion2_config_overrides,
    )

    apply_dion2_config_overrides(optimizer_kwargs)

    all_params = extract_params(trainable_params)
    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    exclude_embeddings = bool(optimizer_kwargs.get("dion2_exclude_embeddings", True))
    exclude_lm_head = bool(optimizer_kwargs.get("dion2_exclude_lm_head", True))
    patterns_value = optimizer_kwargs.get("dion2_exclude_name_patterns", [])
    if isinstance(patterns_value, str):
        exclude_name_patterns = [patterns_value] if patterns_value else []
    elif isinstance(patterns_value, (list, tuple)):
        exclude_name_patterns = [str(p).strip() for p in patterns_value if str(p).strip()]
    else:
        raise ValueError(
            "dion2_exclude_name_patterns must be a string or sequence of strings."
        )

    embedding_ids = set()
    lm_head_ids = set()
    named_exclusion_ids = set()

    if exclude_embeddings or exclude_lm_head or exclude_name_patterns:
        if transformer is None:
            logger.warning(
                "DION2 exclusion controls were requested, but transformer is unavailable. "
                "Exclusions will be skipped."
            )
        else:
            if exclude_embeddings:
                for _, module in transformer.named_modules():
                    if isinstance(module, torch.nn.Embedding):
                        for param in module.parameters(recurse=False):
                            embedding_ids.add(id(param))

            if exclude_lm_head and hasattr(transformer, "lm_head"):
                lm_head_obj = getattr(transformer, "lm_head")
                if isinstance(lm_head_obj, torch.nn.Parameter):
                    lm_head_ids.add(id(lm_head_obj))
                elif isinstance(lm_head_obj, torch.nn.Module):
                    for param in lm_head_obj.parameters():
                        lm_head_ids.add(id(param))

            if exclude_name_patterns:
                for name, param in transformer.named_parameters():
                    if any(
                        fnmatch.fnmatch(name, pattern)
                        for pattern in exclude_name_patterns
                    ):
                        named_exclusion_ids.add(id(param))

    special_aux_ids = embedding_ids | lm_head_ids | named_exclusion_ids
    dion_hidden_weights = [p for p in hidden_weights if id(p) not in special_aux_ids]
    special_aux_weights = [p for p in hidden_weights if id(p) in special_aux_ids]

    aux_optimizer = str(optimizer_kwargs.get("dion2_aux_optimizer", "lion")).strip().lower()

    log_param_structure(
        "DION2",
        f"Aux {aux_optimizer.upper()}",
        trainable_params,
        all_params,
        dion_hidden_weights,
        hidden_gains_biases + special_aux_weights,
    )

    if not hidden_weights and not hidden_gains_biases:
        raise ValueError("No trainable parameters found for DION2 optimizer.")
    if not dion_hidden_weights:
        raise ValueError(
            "DION2 requires at least one matrix parameter after exclusions. "
            "Use AdamW/Prodigy/Lion for scalar-only or embedding-only parameter sets."
        )
    if not hidden_gains_biases and not special_aux_weights:
        logger.info(
            "No auxiliary scalar parameters (<2D) found for DION2. "
            "This is normal for many LoRA runs."
        )

    rank_fraction = float(
        optimizer_kwargs.get(
            "dion2_rank_fraction",
            0.25,
        )
    )
    momentum = float(
        optimizer_kwargs.get(
            "dion2_momentum",
            0.95,
        )
    )
    ns_steps = int(
        optimizer_kwargs.get(
            "dion2_ns_steps",
            5,
        )
    )
    matrix_lr_scale = float(
        optimizer_kwargs.get(
            "dion2_lr_scale",
            1.0,
        )
    )
    aux_lr_scale = float(
        optimizer_kwargs.get(
            "dion2_aux_lr_scale",
            1.0,
        )
    )
    weight_decay = float(
        optimizer_kwargs.get(
            "dion2_weight_decay",
            optimizer_kwargs.get(
                "weight_decay",
                0.001,
            ),
        )
    )
    aux_betas = _coerce_pair(
        "dion2_aux_betas",
        optimizer_kwargs.get(
            "dion2_aux_betas",
            optimizer_kwargs.get(
                "betas",
                (0.9, 0.999),
            ),
        ),
    )
    aux_eps = float(
        optimizer_kwargs.get(
            "dion2_aux_eps",
            optimizer_kwargs.get(
                "eps",
                1e-8,
            ),
        )
    )
    selection = str(
        optimizer_kwargs.get(
            "dion2_selection",
            "norm",
        )
    ).strip().lower()
    subspace_cap = int(
        optimizer_kwargs.get(
            "dion2_subspace_cap",
            128,
        )
    )
    weight_decay_type = str(
        optimizer_kwargs.get("weight_decay_type", "default")
    ).strip()
    log_dion2_metrics = bool(
        optimizer_kwargs.get(
            "dion2_log_metrics",
            False,
        )
    )
    process_group = None if distributed_context is None else distributed_context.get("process_group")
    world_size = 1 if distributed_context is None else int(distributed_context.get("world_size", 1) or 1)
    rank = 0 if distributed_context is None else int(distributed_context.get("rank", 0) or 0)
    lm_head_lr_scale = optimizer_kwargs.get("dion2_lm_head_lr_scale", None)
    if lm_head_lr_scale is not None:
        lm_head_lr_scale = float(lm_head_lr_scale)
    embedding_weight_decay = optimizer_kwargs.get("dion2_embedding_weight_decay", None)
    if embedding_weight_decay is not None:
        embedding_weight_decay = float(embedding_weight_decay)
    lm_head_weight_decay = optimizer_kwargs.get("dion2_lm_head_weight_decay", None)
    if lm_head_weight_decay is not None:
        lm_head_weight_decay = float(lm_head_weight_decay)
    error_feedback = bool(optimizer_kwargs.get("dion2_error_feedback", True))
    error_decay = float(optimizer_kwargs.get("dion2_error_decay", 1.0))
    distribute_work = bool(optimizer_kwargs.get("dion2_distribute_work", False))

    param_groups: List[Dict[str, Any]] = []
    for item in trainable_params:
        if isinstance(item, dict):
            params = list(item.get("params", []))
            if not params:
                continue
            base_lr = float(item.get("lr", lr))
            base_weight_decay = float(item.get("weight_decay", weight_decay))
            base_wd_type = str(item.get("weight_decay_type", weight_decay_type))
        elif isinstance(item, torch.nn.Parameter):
            params = [item]
            base_lr = float(lr)
            base_weight_decay = float(weight_decay)
            base_wd_type = weight_decay_type
        else:
            continue

        matrix_params = [
            param for param in params if param.ndim >= 2 and id(param) not in special_aux_ids
        ]
        aux_params = [
            param for param in params if param.ndim < 2 and id(param) not in special_aux_ids
        ]
        embedding_params = [param for param in params if id(param) in embedding_ids]
        lm_head_params = [param for param in params if id(param) in lm_head_ids]
        named_aux_params = [
            param
            for param in params
            if id(param) in named_exclusion_ids
            and id(param) not in embedding_ids
            and id(param) not in lm_head_ids
        ]

        if matrix_params:
            matrix_lr = base_lr * matrix_lr_scale
            param_groups.append(
                {
                    "params": matrix_params,
                    "use_dion2": True,
                    "lr": matrix_lr,
                    "momentum": momentum,
                    "weight_decay": base_weight_decay,
                    "rank_fraction": rank_fraction,
                    "ns_steps": ns_steps,
                    "selection": selection,
                    "subspace_cap": subspace_cap,
                    "error_feedback": error_feedback,
                    "error_decay": error_decay,
                    "distribute_work": distribute_work,
                    "initial_lr": matrix_lr,
                    "weight_decay_type": base_wd_type,
                    "log_dion2_metrics": log_dion2_metrics,
                }
            )

        if aux_params:
            aux_lr = base_lr * aux_lr_scale
            param_groups.append(
                {
                    "params": aux_params,
                    "use_dion2": False,
                    "algorithm": aux_optimizer,
                    "lr": aux_lr,
                    "betas": aux_betas,
                    "eps": aux_eps,
                    "weight_decay": base_weight_decay,
                    "initial_lr": aux_lr,
                    "weight_decay_type": base_wd_type,
                }
            )

        if embedding_params:
            embed_lr = base_lr * aux_lr_scale
            embed_wd = (
                base_weight_decay
                if embedding_weight_decay is None
                else embedding_weight_decay
            )
            param_groups.append(
                {
                    "params": embedding_params,
                    "use_dion2": False,
                    "algorithm": aux_optimizer,
                    "lr": embed_lr,
                    "betas": aux_betas,
                    "eps": aux_eps,
                    "weight_decay": embed_wd,
                    "initial_lr": embed_lr,
                    "weight_decay_type": base_wd_type,
                }
            )

        if lm_head_params:
            inferred_scale = lm_head_lr_scale
            if inferred_scale is None:
                lm_head_dim = next(
                    (param.shape[1] for param in lm_head_params if param.ndim >= 2),
                    None,
                )
                inferred_scale = 1.0 if lm_head_dim is None else 1.0 / math.sqrt(
                    max(1, int(lm_head_dim))
                )
            lm_lr = base_lr * aux_lr_scale * inferred_scale
            lm_wd = (
                base_weight_decay if lm_head_weight_decay is None else lm_head_weight_decay
            )
            param_groups.append(
                {
                    "params": lm_head_params,
                    "use_dion2": False,
                    "algorithm": aux_optimizer,
                    "lr": lm_lr,
                    "betas": aux_betas,
                    "eps": aux_eps,
                    "weight_decay": lm_wd,
                    "initial_lr": lm_lr,
                    "weight_decay_type": base_wd_type,
                }
            )

        if named_aux_params:
            aux_lr = base_lr * aux_lr_scale
            param_groups.append(
                {
                    "params": named_aux_params,
                    "use_dion2": False,
                    "algorithm": aux_optimizer,
                    "lr": aux_lr,
                    "betas": aux_betas,
                    "eps": aux_eps,
                    "weight_decay": base_weight_decay,
                    "initial_lr": aux_lr,
                    "weight_decay_type": base_wd_type,
                }
            )

    if not param_groups:
        raise ValueError("No parameter groups created for DION2 optimizer.")

    optimizer_class = SingleDeviceDion2WithAuxAdam
    optimizer = optimizer_class(
        param_groups,
        process_group=process_group,
        world_size=world_size,
        rank=rank,
    )

    logger.info("DION2 configuration:")
    logger.info("  - rank_fraction: %s", rank_fraction)
    logger.info("  - momentum: %s", momentum)
    logger.info("  - ns_steps: %s", ns_steps)
    logger.info("  - matrix_lr_scale: %s", matrix_lr_scale)
    logger.info("  - aux_lr_scale: %s", aux_lr_scale)
    logger.info("  - aux_optimizer: %s", aux_optimizer)
    logger.info("  - selection: %s", selection)
    logger.info("  - subspace_cap: %s", subspace_cap)
    logger.info("  - exclude_embeddings: %s", exclude_embeddings)
    logger.info("  - exclude_lm_head: %s", exclude_lm_head)
    logger.info("  - exclude_name_patterns: %s", exclude_name_patterns)
    logger.info("  - error_feedback: %s", error_feedback)
    logger.info("  - distribute_work: %s", distribute_work)
    logger.info("  - distributed_world_size: %s", world_size)
    logger.info("  - embedding params routed to aux: %s", len(embedding_ids))
    logger.info("  - lm_head params routed to aux: %s", len(lm_head_ids))
    logger.info("  - named exclusions routed to aux: %s", len(named_exclusion_ids))

    return optimizer_class, optimizer
