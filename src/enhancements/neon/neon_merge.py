from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

import torch

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NeonMergeStats:
    merged_tensors: int
    copied_tensors: int
    skipped_tensors: int
    current_tensors: int
    match_ratio: float


def load_lora_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a LoRA-style state dict from safetensors or torch checkpoint."""

    if not path:
        raise ValueError("Neon reference checkpoint path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neon reference checkpoint not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from memory.safetensors_loader import load_file

        state_dict = load_file(path, device="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu")

    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Neon reference checkpoint must contain a state dict, got {type(state_dict)!r}"
        )

    return {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def build_neon_state_dict(
    current_state: Mapping[str, torch.Tensor],
    reference_state: Optional[Mapping[str, torch.Tensor]],
    weight: float,
    *,
    reference_mode: str = "checkpoint",
    merge_alpha: bool = False,
    allow_shape_mismatch: bool = False,
    min_match_ratio: float = 0.8,
) -> tuple[Dict[str, torch.Tensor], NeonMergeStats]:
    """Build theta_neon = (1 + w) * theta_ref - w * theta_current."""

    if weight <= 0.0:
        raise ValueError("Neon extrapolation weight must be > 0")
    if reference_mode not in {"checkpoint", "zeros"}:
        raise ValueError("reference_mode must be 'checkpoint' or 'zeros'")
    if reference_mode == "checkpoint" and reference_state is None:
        raise ValueError("reference_state is required for checkpoint reference mode")
    if not (0.0 <= min_match_ratio <= 1.0):
        raise ValueError("min_match_ratio must be in [0.0, 1.0]")

    merged: Dict[str, torch.Tensor] = {}
    merged_tensors = 0
    copied_tensors = 0
    skipped_tensors = 0
    eligible_tensors = 0

    for key, current in current_state.items():
        if not isinstance(current, torch.Tensor):
            continue
        current_cpu = current.detach().cpu()
        can_merge = torch.is_floating_point(current_cpu)
        can_merge = can_merge and (merge_alpha or not key.endswith(".alpha"))

        if not can_merge:
            merged[key] = current_cpu.clone()
            copied_tensors += 1
            continue

        eligible_tensors += 1
        if reference_mode == "zeros":
            reference_cpu = torch.zeros_like(current_cpu)
        else:
            assert reference_state is not None
            reference = reference_state.get(key)
            if reference is None:
                merged[key] = current_cpu.clone()
                skipped_tensors += 1
                continue
            reference_cpu = reference.detach().cpu()
            if reference_cpu.shape != current_cpu.shape:
                if not allow_shape_mismatch:
                    raise ValueError(
                        "Neon reference tensor shape mismatch for "
                        f"{key}: current={tuple(current_cpu.shape)} "
                        f"reference={tuple(reference_cpu.shape)}"
                    )
                merged[key] = current_cpu.clone()
                skipped_tensors += 1
                continue
            reference_cpu = reference_cpu.to(dtype=current_cpu.dtype)

        finite_mask = torch.isfinite(reference_cpu) & torch.isfinite(current_cpu)
        neon_tensor = current_cpu.clone()
        if finite_mask.any():
            neon_tensor[finite_mask] = (
                (1.0 + weight) * reference_cpu[finite_mask]
                - weight * current_cpu[finite_mask]
            )
        merged[key] = neon_tensor
        merged_tensors += 1

    match_ratio = merged_tensors / eligible_tensors if eligible_tensors else 1.0
    if match_ratio < min_match_ratio:
        raise ValueError(
            "Neon reference matched too few mergeable tensors: "
            f"{match_ratio:.2%} < required {min_match_ratio:.2%}"
        )

    stats = NeonMergeStats(
        merged_tensors=merged_tensors,
        copied_tensors=copied_tensors,
        skipped_tensors=skipped_tensors,
        current_tensors=len(current_state),
        match_ratio=match_ratio,
    )
    return merged, stats


def save_neon_merged_weights(
    network: Any,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[MutableMapping[str, str]],
    args: Any,
) -> NeonMergeStats:
    """Save a Neon-merged LoRA checkpoint without changing live network weights."""

    current_state = network.state_dict()
    reference_mode = str(getattr(args, "neon_reference_mode", "checkpoint"))
    reference_state = None
    if reference_mode == "checkpoint":
        reference_state = load_lora_state_dict(
            str(getattr(args, "neon_reference_lora_path", ""))
        )

    merged_state, stats = build_neon_state_dict(
        current_state,
        reference_state,
        float(getattr(args, "neon_extrapolation_weight", 1.0)),
        reference_mode=reference_mode,
        merge_alpha=bool(getattr(args, "neon_merge_alpha", False)),
        allow_shape_mismatch=bool(getattr(args, "neon_allow_shape_mismatch", False)),
        min_match_ratio=float(getattr(args, "neon_min_match_ratio", 0.8)),
    )

    if dtype is not None:
        for key, value in list(merged_state.items()):
            merged_state[key] = value.to(dtype=dtype)

    metadata_to_save = metadata if metadata is not None else {}
    metadata_to_save["takenoko_neon_negative_extrapolation"] = "true"
    metadata_to_save["takenoko_neon_reference_mode"] = reference_mode
    metadata_to_save["takenoko_neon_extrapolation_weight"] = str(
        float(getattr(args, "neon_extrapolation_weight", 1.0))
    )
    metadata_to_save["takenoko_neon_merged_tensors"] = str(stats.merged_tensors)
    metadata_to_save["takenoko_neon_match_ratio"] = f"{stats.match_ratio:.6f}"
    if reference_mode == "checkpoint":
        metadata_to_save["takenoko_neon_reference_lora_path"] = str(
            getattr(args, "neon_reference_lora_path", "")
        )

    if os.path.splitext(file)[1].lower() == ".safetensors":
        from safetensors.torch import save_file
        from utils import model_utils

        model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(
            merged_state, metadata_to_save
        )
        metadata_to_save["sshs_model_hash"] = model_hash
        metadata_to_save["sshs_legacy_hash"] = legacy_hash
        save_file(merged_state, file, metadata_to_save)
    else:
        torch.save(merged_state, file)

    logger.info(
        "Saved Neon negative-extrapolated checkpoint: %s "
        "(merged=%d, copied=%d, skipped=%d, match_ratio=%.2f%%)",
        file,
        stats.merged_tensors,
        stats.copied_tensors,
        stats.skipped_tensors,
        stats.match_ratio * 100.0,
    )
    return stats
