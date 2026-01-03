from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Callable

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def begin_contrastive_step(
    helper: Optional[Any],
    batch: Dict[str, Any],
    global_step: Optional[int],
) -> None:
    if helper is None or global_step is None:
        return
    helper.begin_step(global_step)
    helper.set_concept_ids(batch.get("concept_id"))


def _build_context_override(
    context_list: list[torch.Tensor],
    *,
    strategy: str,
    pass_index: int,
) -> list[torch.Tensor]:
    if not context_list:
        return context_list
    batch_size = context_list[0].shape[0]
    if strategy == "roll":
        shift = max(1, pass_index + 1)
        return [torch.roll(t, shifts=shift, dims=0) for t in context_list]
    perm = torch.randperm(batch_size, device=context_list[0].device)
    return [t[perm] for t in context_list]


def run_extra_prompt_passes(
    *,
    helper: Optional[Any],
    args: Any,
    batch: Dict[str, Any],
    call_dit_fn: Callable[[torch.Tensor, Optional[list[torch.Tensor]]], Any],
    noisy_model_input: torch.Tensor,
    accelerator: Any,
) -> None:
    if helper is None:
        return
    passes = int(getattr(args, "contrastive_attention_extra_prompt_passes", 0))
    if passes <= 0:
        return
    context_list = batch.get("t5")
    if not isinstance(context_list, list) or not context_list:
        return
    strategy = str(
        getattr(args, "contrastive_attention_extra_prompt_strategy", "shuffle")
    ).lower()
    with torch.no_grad():
        for pass_index in range(passes):
            override = _build_context_override(
                context_list, strategy=strategy, pass_index=pass_index
            )
            _ = call_dit_fn(noisy_model_input, override)
    _ = accelerator  # keep signature symmetric for future usage


def apply_concept_multiplier(
    *,
    network: Any,
    args: Any,
    batch: Dict[str, Any],
) -> tuple[Optional[float], bool]:
    if network is None:
        return None, False
    if not hasattr(network, "set_multiplier") and not hasattr(network, "multiplier"):
        return None, False
    mapping = getattr(args, "contrastive_attention_concept_multipliers", None)
    if not isinstance(mapping, dict) or not mapping:
        return None, False
    concept_ids = batch.get("concept_id")
    if concept_ids is None:
        return None, False
    if torch.is_tensor(concept_ids):
        values = concept_ids.detach().cpu().tolist()
    else:
        values = list(concept_ids)
    values = [int(v) for v in values if v is not None]
    if not values:
        return None, False
    if len(set(values)) != 1:
        return None, False
    concept_id = values[0]
    if concept_id not in mapping:
        return None, False
    try:
        multiplier = float(mapping[concept_id])
    except Exception:
        return None, False
    prev = getattr(network, "multiplier", None)
    if hasattr(network, "set_multiplier"):
        network.set_multiplier(multiplier)
    else:
        setattr(network, "multiplier", multiplier)
    return prev, True


def restore_concept_multiplier(
    *,
    network: Any,
    previous_multiplier: Optional[float],
    applied: bool,
) -> None:
    if not applied or network is None:
        return
    if previous_multiplier is None:
        return
    if hasattr(network, "set_multiplier"):
        network.set_multiplier(previous_multiplier)
    else:
        setattr(network, "multiplier", previous_multiplier)


def apply_latent_update(
    *,
    helper: Optional[Any],
    args: Any,
    batch: Dict[str, Any],
    call_dit_fn: Callable[[torch.Tensor, Optional[list[torch.Tensor]]], Any],
    noisy_model_input: torch.Tensor,
    accelerator: Any,
    global_step: Optional[int],
) -> tuple[torch.Tensor, bool]:
    if helper is None:
        return noisy_model_input, False
    if not bool(getattr(args, "contrastive_attention_latent_update", False)):
        return noisy_model_input, False
    interval = int(
        getattr(args, "contrastive_attention_latent_update_interval", 1)
    )
    if interval <= 0 or global_step is None or global_step % interval != 0:
        return noisy_model_input, False
    steps = int(getattr(args, "contrastive_attention_latent_update_steps", 0))
    if steps <= 0:
        return noisy_model_input, False
    step_size = float(
        getattr(args, "contrastive_attention_latent_update_step_size", 0.1)
    )
    if step_size <= 0:
        return noisy_model_input, False

    updated = noisy_model_input
    for _ in range(steps):
        updated = updated.detach().requires_grad_(True)
        with accelerator.autocast():
            _ = call_dit_fn(updated, None)
        contrastive_loss = helper.compute_loss(batch.get("concept_id"))
        if contrastive_loss is None or contrastive_loss.item() == 0.0:
            break
        grads = torch.autograd.grad(
            contrastive_loss, [updated], retain_graph=False, allow_unused=True
        )[0]
        if grads is None:
            break
        updated = updated - step_size * grads
    return updated.detach(), True
