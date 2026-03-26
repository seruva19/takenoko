"""OrthoLORA config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


_VALID_GROUP_BY = {"dataset_index", "concept_id", "media_type"}


def apply_ortholora_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.enable_ortholora = bool(config.get("enable_ortholora", False))
    args.ortholora_group_by = str(
        config.get("ortholora_group_by", "dataset_index")
    ).strip()
    args.ortholora_min_active_groups = int(
        config.get("ortholora_min_active_groups", 2)
    )
    args.ortholora_min_group_samples = int(
        config.get("ortholora_min_group_samples", 1)
    )
    args.ortholora_log_metrics = bool(config.get("ortholora_log_metrics", True))

    if args.ortholora_group_by not in _VALID_GROUP_BY:
        raise ValueError(
            f"ortholora_group_by must be one of {sorted(_VALID_GROUP_BY)}, "
            f"got {args.ortholora_group_by!r}"
        )
    if args.ortholora_min_active_groups < 2:
        raise ValueError(
            "ortholora_min_active_groups must be >= 2 so projection has at least "
            f"two groups to compare, got {args.ortholora_min_active_groups}"
        )
    if args.ortholora_min_group_samples < 1:
        raise ValueError(
            f"ortholora_min_group_samples must be >= 1, got {args.ortholora_min_group_samples}"
        )

    if args.enable_ortholora:
        logger.info(
            "OrthoLORA enabled (group_by=%s, min_groups=%d, min_group_samples=%d, log_metrics=%s)",
            args.ortholora_group_by,
            args.ortholora_min_active_groups,
            args.ortholora_min_group_samples,
            args.ortholora_log_metrics,
        )
