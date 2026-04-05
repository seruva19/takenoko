from __future__ import annotations

import re
from typing import Any, Optional

import torch


def parse_range_entries(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        entries = raw_value.split(",")
    elif isinstance(raw_value, (list, tuple)):
        entries = []
        for value in raw_value:
            if not isinstance(value, str):
                raise ValueError(
                    "Expected string entries for block range configuration, "
                    f"got {type(value)}"
                )
            entries.extend(value.split(","))
    else:
        raise ValueError(
            "Expected a string or list of strings for block range configuration, "
            f"got {type(raw_value)}"
        )
    return [entry.strip() for entry in entries if entry and entry.strip()]


def parse_block_index_spec(
    raw_value: Any,
    depth: int,
    option_name: str,
) -> set[int]:
    indices: set[int] = set()
    for entry in parse_range_entries(raw_value):
        if "-" in entry:
            start_text, end_text = entry.split("-", 1)
            if not start_text.strip():
                raise ValueError(
                    f"{option_name} entry {entry!r} must specify a start index"
                )
            start = int(start_text)
            end = depth - 1 if not end_text.strip() else int(end_text)
            if start < 0 or end < 0:
                raise ValueError(
                    f"{option_name} entries must be >= 0, got {entry!r}"
                )
            if end < start:
                raise ValueError(
                    f"{option_name} range end must be >= start, got {entry!r}"
                )
            indices.update(range(start, end + 1))
        else:
            idx = int(entry)
            if idx < 0:
                raise ValueError(
                    f"{option_name} entries must be >= 0, got {entry!r}"
                )
            indices.add(idx)

    if depth > 0:
        invalid = sorted(idx for idx in indices if idx >= depth)
        if invalid:
            raise ValueError(
                f"{option_name} contains block indices outside [0, {depth - 1}]: "
                f"{invalid}"
            )
    return indices


def parse_block_lr_rules(
    raw_value: Any,
    depth: int,
) -> list[tuple[int, Optional[int], float]]:
    rules: list[tuple[int, Optional[int], float]] = []
    for entry in parse_range_entries(raw_value):
        if ":" not in entry:
            raise ValueError(
                f"Invalid block_lr_scales entry {entry!r}. Expected format like "
                "'0-11:0.1' or '12-:1.0'."
            )
        range_part, scale_part = entry.split(":", 1)
        range_part = range_part.strip()
        scale = float(scale_part.strip())
        if scale < 0.0:
            raise ValueError(
                f"block_lr_scales must use non-negative scales, got {entry!r}"
            )

        if "-" in range_part:
            start_text, end_text = range_part.split("-", 1)
            if not start_text.strip():
                raise ValueError(
                    f"Invalid block_lr_scales range {range_part!r} in {entry!r}"
                )
            start = int(start_text)
            end = None if not end_text.strip() else int(end_text)
        else:
            start = int(range_part)
            end = start

        if start < 0 or (end is not None and end < 0):
            raise ValueError(
                f"block_lr_scales ranges must use non-negative indices, got {entry!r}"
            )
        if end is not None and end < start:
            raise ValueError(
                f"block_lr_scales end must be >= start, got {entry!r}"
            )
        if depth > 0 and start >= depth:
            raise ValueError(
                f"block_lr_scales start index {start} is outside [0, {depth - 1}]"
            )
        if depth > 0 and end is not None and end >= depth:
            raise ValueError(
                f"block_lr_scales end index {end} is outside [0, {depth - 1}]"
            )
        rules.append((start, end, scale))
    return rules


def extract_block_index(param_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)blocks\.(\d+)\.", param_name)
    return None if match is None else int(match.group(1))


def is_attention_geometry_param(param_name: str) -> bool:
    geometry_markers = (
        ".self_attn.q.",
        ".self_attn.k.",
        ".self_attn.norm_q.",
        ".self_attn.norm_k.",
        ".cross_attn.q.",
        ".cross_attn.k.",
        ".cross_attn.norm_q.",
        ".cross_attn.norm_k.",
    )
    return any(marker in param_name for marker in geometry_markers)


def resolve_block_lr_scale(
    block_index: int,
    rules: list[tuple[int, Optional[int], float]],
) -> Optional[float]:
    matched_scale: Optional[float] = None
    for start, end, scale in rules:
        upper = block_index if end is None else end
        if start <= block_index <= upper:
            matched_scale = scale
    return matched_scale


def summarize_full_ft_motion_controls(
    *,
    transformer: torch.nn.Module,
    args: Any,
    logger: Any,
) -> dict[str, Any]:
    blocks = list(getattr(transformer, "blocks", []))
    depth = len(blocks)
    frozen_blocks = set(range(min(max(0, int(args.freeze_early_blocks)), depth)))
    frozen_blocks.update(
        parse_block_index_spec(
            getattr(args, "freeze_block_indices", None),
            depth,
            "freeze_block_indices",
        )
    )
    block_lr_rules = parse_block_lr_rules(
        getattr(args, "block_lr_scales", None),
        depth,
    )

    frozen_param_count = 0
    geometry_param_count = 0
    geometry_frozen_count = 0
    geometry_param_tensors = 0
    geometry_frozen_tensors = 0
    for name, param in transformer.named_parameters():
        block_index = extract_block_index(name)
        is_geometry = is_attention_geometry_param(name)
        if is_geometry:
            geometry_param_count += param.numel()
            geometry_param_tensors += 1

        if block_index is not None and block_index in frozen_blocks:
            if param.requires_grad:
                param.requires_grad_(False)
                frozen_param_count += param.numel()
            if is_geometry:
                geometry_frozen_count += param.numel()
                geometry_frozen_tensors += 1
            continue

        if is_geometry and bool(getattr(args, "freeze_attn_geometry", False)):
            if param.requires_grad:
                param.requires_grad_(False)
                frozen_param_count += param.numel()
                geometry_frozen_count += param.numel()
                geometry_frozen_tensors += 1

    summary = {
        "depth": depth,
        "frozen_blocks": sorted(frozen_blocks),
        "block_lr_rules": block_lr_rules,
        "non_block_lr_scale": float(getattr(args, "non_block_lr_scale", 1.0)),
        "attn_geometry_lr_scale": float(
            getattr(args, "attn_geometry_lr_scale", 1.0)
        ),
        "freeze_attn_geometry": bool(getattr(args, "freeze_attn_geometry", False)),
        "frozen_param_count": frozen_param_count,
        "geometry_param_count": geometry_param_count,
        "geometry_frozen_count": geometry_frozen_count,
        "geometry_param_tensors": geometry_param_tensors,
        "geometry_frozen_tensors": geometry_frozen_tensors,
    }
    logger.info(
        "Full-FT motion preservation controls: depth=%d frozen_blocks=%s "
        "block_lr_rules=%s non_block_lr_scale=%.3f attn_geometry_lr_scale=%.3f "
        "freeze_attn_geometry=%s frozen_params=%d geometry_frozen=%d/%d",
        depth,
        summary["frozen_blocks"],
        block_lr_rules,
        summary["non_block_lr_scale"],
        summary["attn_geometry_lr_scale"],
        str(summary["freeze_attn_geometry"]).lower(),
        frozen_param_count,
        geometry_frozen_count,
        geometry_param_count,
    )
    return summary


def build_full_ft_optimizer_groups(
    *,
    transformer: torch.nn.Module,
    args: Any,
    motion_preservation_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[list[str]], dict[str, int], list[float], int]:
    block_lr_rules = motion_preservation_summary["block_lr_rules"]
    non_block_lr_scale = motion_preservation_summary["non_block_lr_scale"]
    attn_geometry_lr_scale = motion_preservation_summary["attn_geometry_lr_scale"]
    freeze_attn_geometry = motion_preservation_summary["freeze_attn_geometry"]
    base_lr = float(args.learning_rate)

    params_to_optimize = []
    param_names = []
    lr_groups: dict[str, dict[str, Any]] = {}
    lr_scales_applied = set()
    trainable_by_block: dict[str, int] = {}
    trainable_attn_geometry_tensors = 0

    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue

        block_index = extract_block_index(name)
        if block_index is None:
            lr_scale = non_block_lr_scale
        else:
            lr_scale = resolve_block_lr_scale(block_index, block_lr_rules)
            if lr_scale is None:
                lr_scale = 1.0

        if is_attention_geometry_param(name):
            if freeze_attn_geometry:
                param.requires_grad_(False)
                continue
            lr_scale *= attn_geometry_lr_scale
            trainable_attn_geometry_tensors += 1

        if lr_scale <= 0.0:
            param.requires_grad_(False)
            continue

        lr_scales_applied.add(float(lr_scale))
        if block_index is not None:
            block_key = str(block_index)
            trainable_by_block[block_key] = trainable_by_block.get(block_key, 0) + 1
        effective_lr = base_lr * lr_scale
        lr_key = f"{effective_lr:.16g}"
        if lr_key not in lr_groups:
            lr_groups[lr_key] = {
                "lr": effective_lr,
                "params": [],
                "names": [],
            }
        lr_groups[lr_key]["params"].append(param)
        lr_groups[lr_key]["names"].append(name)

    for group in sorted(lr_groups.values(), key=lambda item: item["lr"]):
        params_to_optimize.append({"params": group["params"], "lr": group["lr"]})
        param_names.append(group["names"])

    return (
        params_to_optimize,
        param_names,
        trainable_by_block,
        sorted(lr_scales_applied),
        int(trainable_attn_geometry_tensors),
    )
