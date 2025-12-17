from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from .model import PolyLoRANetwork, predict_lora_state_dict
from .spec import LoRATargetSpec, load_spec_file


def load_polylora_checkpoint(path: Path, model: PolyLoRANetwork) -> PolyLoRANetwork:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def run_prediction(
    model: PolyLoRANetwork,
    embeddings: Iterable[torch.Tensor],
    residuals: Optional[Iterable[Optional[torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    emb_list = list(embeddings)
    res_list = list(residuals) if residuals is not None else [None] * len(emb_list)
    predictions: Dict[str, torch.Tensor] = {}
    for embedding, residual in zip(emb_list, res_list):
        pred = predict_lora_state_dict(model, embedding, residual=residual, detach=True)
        predictions = merge_state_dicts(predictions, pred)
    return predictions


def merge_state_dicts(base: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not base:
        return {k: v.clone() for k, v in new.items()}
    merged: Dict[str, torch.Tensor] = {}
    for key, value in new.items():
        if key not in base:
            merged[key] = value.clone()
            continue
        if base[key].shape != value.shape:
            raise ValueError(f"Shape mismatch for key {key}: {base[key].shape} vs {value.shape}")
        merged[key] = (base[key] + value) / 2.0
    for key, value in base.items():
        if key not in merged:
            merged[key] = value.clone()
    return merged
