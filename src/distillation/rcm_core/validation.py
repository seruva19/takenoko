"""Replay buffer validation helpers for rCM distillation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import torch

from .buffers import RCMReplayBuffer, RCMReplaySample


_REQUIRED_PAYLOAD_KEYS = {
    "latents",
    "t5",
    "t5_attention_mask",
    "t5_pooled_output",
    "t5_negative",
    "t5_negative_attention_mask",
    "t5_negative_pooled_output",
}


def _seq_len(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 1
    if tensor.ndim == 1:
        return int(tensor.shape[0])
    return int(tensor.shape[-2])


def _hidden_dim(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 1
    return int(tensor.shape[-1])


def _validate_condition_block(
    sample_idx: int,
    role: str,
    embeddings: object,
    masks: object,
    pooled: object,
    issues: List[str],
) -> None:
    if not isinstance(embeddings, list) or len(embeddings) == 0:
        issues.append(f"sample[{sample_idx}] missing {role} embeddings")
        return

    if not isinstance(masks, list) or len(masks) != len(embeddings):
        issues.append(
            f"sample[{sample_idx}] {role} attention mask count mismatch "
            f"(expected {len(embeddings)}, got {len(masks) if isinstance(masks, list) else 'N/A'})"
        )
        masks_list = [None] * len(embeddings)
    else:
        masks_list = masks

    if not isinstance(pooled, list) or len(pooled) != len(embeddings):
        issues.append(
            f"sample[{sample_idx}] {role} pooled output count mismatch "
            f"(expected {len(embeddings)}, got {len(pooled) if isinstance(pooled, list) else 'N/A'})"
        )
        pooled_list = [None] * len(embeddings)
    else:
        pooled_list = pooled

    for item_idx, embedding in enumerate(embeddings):
        if not isinstance(embedding, torch.Tensor):
            issues.append(f"sample[{sample_idx}] {role}[{item_idx}] is not a tensor")
            continue

        seq_len = _seq_len(embedding)
        hidden = _hidden_dim(embedding)

        mask_tensor = masks_list[item_idx] if item_idx < len(masks_list) else None
        if isinstance(mask_tensor, torch.Tensor):
            if mask_tensor.numel() == 0:
                issues.append(f"sample[{sample_idx}] {role} mask[{item_idx}] is empty")
            else:
                mask_len = mask_tensor.shape[-1] if mask_tensor.ndim > 1 else mask_tensor.shape[0]
                if mask_len != seq_len:
                    issues.append(
                        f"sample[{sample_idx}] {role} mask[{item_idx}] length {mask_len} "
                        f"does not match embedding sequence length {seq_len}"
                    )
        else:
            issues.append(f"sample[{sample_idx}] missing {role} mask[{item_idx}]")

        pooled_tensor = pooled_list[item_idx] if item_idx < len(pooled_list) else None
        if isinstance(pooled_tensor, torch.Tensor):
            if _hidden_dim(pooled_tensor) != hidden:
                issues.append(
                    f"sample[{sample_idx}] {role} pooled[{item_idx}] hidden dimension "
                    f"{_hidden_dim(pooled_tensor)} != {hidden}"
                )
        else:
            issues.append(f"sample[{sample_idx}] missing {role} pooled[{item_idx}]")


def _validate_latents(sample_idx: int, sample: RCMReplaySample, issues: List[str]) -> None:
    latents = sample.payload.get("latents")
    if not isinstance(latents, torch.Tensor):
        issues.append(f"sample[{sample_idx}] missing latents tensor")
        return

    if latents.ndim < 4:
        issues.append(
            f"sample[{sample_idx}] latents expected >=4 dims, got shape {tuple(latents.shape)}"
        )
    elif latents.shape[-1] == 0:
        issues.append(f"sample[{sample_idx}] latents have zero width: {tuple(latents.shape)}")


def _validate_timestep_alignment(sample_idx: int, sample: RCMReplaySample, issues: List[str]) -> None:
    timesteps = sample.payload.get("timesteps")
    if timesteps is None:
        return

    if isinstance(timesteps, torch.Tensor):
        if timesteps.numel() == 0:
            issues.append(f"sample[{sample_idx}] timesteps tensor is empty")
    elif isinstance(timesteps, (list, tuple)):
        if len(timesteps) == 0:
            issues.append(f"sample[{sample_idx}] timesteps list is empty")
    else:
        issues.append(f"sample[{sample_idx}] timesteps expected tensor/list, got {type(timesteps).__name__}")


def _validate_batch_alignment(sample_idx: int, sample: RCMReplaySample, issues: List[str]) -> None:
    latents = sample.payload.get("latents")
    positive = sample.payload.get("t5")
    if isinstance(latents, torch.Tensor) and isinstance(positive, list) and latents.ndim >= 1:
        batch_dim = latents.shape[0]
        if batch_dim != len(positive):
            issues.append(
                f"sample[{sample_idx}] latent batch {batch_dim} != number of prompts {len(positive)}"
            )


def validate_replay_buffer(buffer: RCMReplayBuffer) -> List[str]:
    """Return a list of human-readable issues detected in the replay buffer."""

    issues: List[str] = []

    if len(buffer) == 0:
        issues.append("Replay buffer contains no samples.")
        return issues

    for sample_idx, sample in enumerate(buffer.iter_samples()):
        payload_keys = set(sample.payload.keys())
        missing_keys = _REQUIRED_PAYLOAD_KEYS - payload_keys
        for key in sorted(missing_keys):
            issues.append(f"sample[{sample_idx}] missing payload key '{key}'")

        _validate_latents(sample_idx, sample, issues)
        _validate_batch_alignment(sample_idx, sample, issues)
        _validate_timestep_alignment(sample_idx, sample, issues)

        _validate_condition_block(
            sample_idx,
            "positive",
            sample.payload.get("t5"),
            sample.payload.get("t5_attention_mask"),
            sample.payload.get("t5_pooled_output"),
            issues,
        )

        _validate_condition_block(
            sample_idx,
            "negative",
            sample.payload.get("t5_negative"),
            sample.payload.get("t5_negative_attention_mask"),
            sample.payload.get("t5_negative_pooled_output"),
            issues,
        )

    return issues


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _update_tensor_summary(summary: Dict[str, Any], tensor: torch.Tensor) -> None:
    dtype_set: set[str] = summary.setdefault("dtypes", set())
    dtype_set.add(_dtype_name(tensor))
    shapes: set[tuple[int, ...]] = summary.setdefault("shapes", set())
    shapes.add(tuple(int(dim) for dim in tensor.shape))


def _update_condition_stats(stats: Dict[str, Any], tensors: Iterable[torch.Tensor]) -> None:
    seq_lengths: List[int] = stats.setdefault("seq_lengths", [])
    hidden_dims: set[int] = stats.setdefault("hidden_dims", set())
    dtypes: set[str] = stats.setdefault("dtypes", set())
    for tensor in tensors:
        seq_lengths.append(_seq_len(tensor))
        hidden_dims.add(_hidden_dim(tensor))
        dtypes.add(_dtype_name(tensor))


def summarize_replay_buffer(buffer: RCMReplayBuffer) -> Dict[str, Any]:
    """Return aggregate statistics for key replay payload fields."""

    summary: Dict[str, Any] = {
        "sample_count": len(buffer),
        "latents": {},
        "positive": defaultdict(set),
        "negative": defaultdict(set),
        "timesteps": {"count": 0},
    }

    timesteps_min: float | None = None
    timesteps_max: float | None = None

    for sample in buffer.iter_samples():
        latents = sample.payload.get("latents")
        if isinstance(latents, torch.Tensor):
            _update_tensor_summary(summary["latents"], latents)

        positive_embeddings = sample.payload.get("t5")
        if isinstance(positive_embeddings, list):
            tensors = [tensor for tensor in positive_embeddings if isinstance(tensor, torch.Tensor)]
            if tensors:
                _update_condition_stats(summary["positive"], tensors)

        negative_embeddings = sample.payload.get("t5_negative")
        if isinstance(negative_embeddings, list):
            tensors = [tensor for tensor in negative_embeddings if isinstance(tensor, torch.Tensor)]
            if tensors:
                _update_condition_stats(summary["negative"], tensors)

        timesteps = sample.payload.get("timesteps")
        values: List[float] = []
        if isinstance(timesteps, torch.Tensor):
            values = [float(v) for v in timesteps.view(-1).cpu().tolist()]
        elif isinstance(timesteps, (list, tuple)):
            try:
                values = [float(v) for v in timesteps]
            except (TypeError, ValueError):
                values = []

        if values:
            summary["timesteps"]["count"] += len(values)
            current_min = min(values)
            current_max = max(values)
            timesteps_min = current_min if timesteps_min is None else min(timesteps_min, current_min)
            timesteps_max = current_max if timesteps_max is None else max(timesteps_max, current_max)

    latents_summary = summary["latents"]
    if latents_summary:
        latents_summary["dtypes"] = sorted(latents_summary["dtypes"])
        latents_summary["shapes"] = [list(shape) for shape in sorted(latents_summary["shapes"])]

    for key in ("positive", "negative"):
        condition_stats: Dict[str, Any] = dict(summary[key])
        if "seq_lengths" in condition_stats and condition_stats["seq_lengths"]:
            seq_lengths = condition_stats.pop("seq_lengths")
            condition_stats["seq_len_min"] = min(seq_lengths)
            condition_stats["seq_len_max"] = max(seq_lengths)
            condition_stats["samples"] = len(seq_lengths)
        if "hidden_dims" in condition_stats:
            hidden_dims = condition_stats["hidden_dims"]
            condition_stats["hidden_dims"] = sorted(hidden_dims)
        if "dtypes" in condition_stats:
            condition_stats["dtypes"] = sorted(condition_stats["dtypes"])
        summary[key] = condition_stats

    if summary["timesteps"]["count"]:
        summary["timesteps"]["min"] = timesteps_min
        summary["timesteps"]["max"] = timesteps_max

    return summary


def compare_replay_statistics(
    buffer: RCMReplayBuffer,
    reference_summary: Dict[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> List[str]:
    """Compare replay statistics against a reference summary."""

    current = summarize_replay_buffer(buffer)
    issues: List[str] = []

    if "sample_count" in reference_summary:
        ref_count = int(reference_summary["sample_count"])
        if current.get("sample_count") != ref_count:
            issues.append(
                f"sample_count mismatch: expected {ref_count}, observed {current.get('sample_count')}"
            )

    def _compare_list(path: str, current_values: Iterable[Any], reference_values: Iterable[Any]) -> None:
        if sorted(current_values) != sorted(reference_values):
            issues.append(
                f"{path} mismatch: expected {sorted(reference_values)}, observed {sorted(current_values)}"
            )

    def _compare_scalar(path: str, current_value: float | None, reference_value: float | None) -> None:
        if reference_value is None:
            return
        if current_value is None:
            issues.append(f"{path} missing (expected {reference_value})")
        elif not math.isclose(current_value, reference_value, rel_tol=rtol, abs_tol=atol):
            issues.append(
                f"{path} mismatch: expected {reference_value:.6f}, observed {current_value:.6f}"
            )

    ref_latents = reference_summary.get("latents") or {}
    cur_latents = current.get("latents") or {}
    if ref_latents:
        _compare_list("latents.dtypes", cur_latents.get("dtypes", []), ref_latents.get("dtypes", []))
        _compare_list("latents.shapes", cur_latents.get("shapes", []), ref_latents.get("shapes", []))

    for key in ("positive", "negative"):
        ref_stats = reference_summary.get(key) or {}
        cur_stats = current.get(key) or {}
        if ref_stats:
            _compare_list(f"{key}.dtypes", cur_stats.get("dtypes", []), ref_stats.get("dtypes", []))
            _compare_list(
                f"{key}.hidden_dims", cur_stats.get("hidden_dims", []), ref_stats.get("hidden_dims", [])
            )
            _compare_scalar(
                f"{key}.seq_len_min", cur_stats.get("seq_len_min"), ref_stats.get("seq_len_min"))
            _compare_scalar(
                f"{key}.seq_len_max", cur_stats.get("seq_len_max"), ref_stats.get("seq_len_max"))

    ref_timesteps = reference_summary.get("timesteps") or {}
    cur_timesteps = current.get("timesteps") or {}
    if ref_timesteps:
        if int(cur_timesteps.get("count", 0)) != int(ref_timesteps.get("count", 0)):
            issues.append(
                f"timesteps.count mismatch: expected {ref_timesteps.get('count')}, observed {cur_timesteps.get('count')}"
            )
        _compare_scalar("timesteps.min", cur_timesteps.get("min"), ref_timesteps.get("min"))
        _compare_scalar("timesteps.max", cur_timesteps.get("max"), ref_timesteps.get("max"))

    return issues
