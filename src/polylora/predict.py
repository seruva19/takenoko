from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from .model import PolyLoRANetwork, predict_lora_state_dict
from .spec import LoRATargetSpec, load_spec_file


def load_polylora_checkpoint(path: Path, model: PolyLoRANetwork) -> PolyLoRANetwork:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def load_polylora_metadata(path: Path) -> Dict[str, Any]:
    metadata_path = Path(path)
    if metadata_path.suffix != ".json":
        metadata_path = Path(f"{metadata_path}.meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"PolyLoRA metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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


def weighted_merge_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        return {}
    if weights is None:
        weights = [1.0] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError("weights length must match state_dicts length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    merged: Dict[str, torch.Tensor] = {}
    base_keys = set(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if set(state_dict.keys()) != base_keys:
            raise ValueError("All state dicts must have identical keys for weighted merge")
    for key in base_keys:
        ref_shape = state_dicts[0][key].shape
        acc = None
        for state_dict, weight in zip(state_dicts, weights):
            value = state_dict[key]
            if value.shape != ref_shape:
                raise ValueError(f"Shape mismatch for key {key}: {value.shape} != {ref_shape}")
            term = value.to(dtype=torch.float32) * (float(weight) / total)
            acc = term if acc is None else acc + term
        assert acc is not None
        merged[key] = acc.to(dtype=state_dicts[0][key].dtype)
    return merged
