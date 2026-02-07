from __future__ import annotations

from typing import Any, Dict


_ALLOWED_WEIGHT_SCHEDULES = {"hard", "hard_high", "sigmoid", "cosine", "snr"}
_ALLOWED_PATH_TYPES = {"linear", "cosine"}
_ALLOWED_STABLEVM_LABEL_SOURCES = {"auto", "concept_id", "dataset_index"}
_ALLOWED_STABLE_VELOCITY_KEYS = {
    "enable_stable_velocity",
    "stable_velocity_repa_enabled",
    "stable_velocity_repa_weight_schedule",
    "stable_velocity_repa_tau",
    "stable_velocity_repa_k",
    "stable_velocity_repa_path_type",
    "stable_velocity_repa_min_weight",
    "stable_velocity_repa_log_interval",
    "stable_velocity_stablevm_enable_target",
    "stable_velocity_stablevm_path_type",
    "stable_velocity_stablevm_label_source",
    "stable_velocity_stablevm_t_min",
    "stable_velocity_stablevm_blend",
    "stable_velocity_stablevm_bank_capacity_per_label",
    "stable_velocity_stablevm_refs_per_sample",
    "stable_velocity_stablevm_min_refs",
    "stable_velocity_stablevm_ref_chunk_size",
    "stable_velocity_stablevm_use_global_fallback",
    "stable_velocity_stablevm_numerical_eps",
    "stable_velocity_stablevm_log_interval",
}


def apply_stable_velocity_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse StableVelocity settings. Train-time only; default disabled."""
    unknown_keys = sorted(
        key
        for key in config
        if key.startswith("stable_velocity_")
        and key not in _ALLOWED_STABLE_VELOCITY_KEYS
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported StableVelocity config keys: "
            + ", ".join(unknown_keys)
            + ". Use only stable_velocity_repa_* / stable_velocity_stablevm_* keys."
        )

    args.enable_stable_velocity = bool(config.get("enable_stable_velocity", False))
    args.stable_velocity_repa_enabled = bool(
        config.get("stable_velocity_repa_enabled", True)
    )
    args.stable_velocity_repa_weight_schedule = str(
        config.get("stable_velocity_repa_weight_schedule", "sigmoid")
    ).lower()
    args.stable_velocity_repa_tau = float(config.get("stable_velocity_repa_tau", 0.7))
    args.stable_velocity_repa_k = float(config.get("stable_velocity_repa_k", 20.0))
    args.stable_velocity_repa_path_type = str(
        config.get("stable_velocity_repa_path_type", "linear")
    ).lower()
    args.stable_velocity_repa_min_weight = float(
        config.get("stable_velocity_repa_min_weight", 0.0)
    )
    args.stable_velocity_repa_log_interval = int(
        config.get("stable_velocity_repa_log_interval", 100)
    )

    args.stable_velocity_stablevm_enable_target = bool(
        config.get("stable_velocity_stablevm_enable_target", False)
    )
    args.stable_velocity_stablevm_path_type = str(
        config.get("stable_velocity_stablevm_path_type", "linear")
    ).lower()
    args.stable_velocity_stablevm_label_source = str(
        config.get("stable_velocity_stablevm_label_source", "auto")
    ).lower()
    args.stable_velocity_stablevm_t_min = float(
        config.get("stable_velocity_stablevm_t_min", 0.5)
    )
    args.stable_velocity_stablevm_blend = float(
        config.get("stable_velocity_stablevm_blend", 1.0)
    )
    args.stable_velocity_stablevm_bank_capacity_per_label = int(
        config.get("stable_velocity_stablevm_bank_capacity_per_label", 256)
    )
    args.stable_velocity_stablevm_refs_per_sample = int(
        config.get("stable_velocity_stablevm_refs_per_sample", 64)
    )
    args.stable_velocity_stablevm_min_refs = int(
        config.get("stable_velocity_stablevm_min_refs", 8)
    )
    args.stable_velocity_stablevm_ref_chunk_size = int(
        config.get("stable_velocity_stablevm_ref_chunk_size", 16)
    )
    args.stable_velocity_stablevm_use_global_fallback = bool(
        config.get("stable_velocity_stablevm_use_global_fallback", True)
    )
    args.stable_velocity_stablevm_numerical_eps = float(
        config.get("stable_velocity_stablevm_numerical_eps", 1e-8)
    )
    args.stable_velocity_stablevm_log_interval = int(
        config.get("stable_velocity_stablevm_log_interval", 100)
    )

    if args.stable_velocity_repa_weight_schedule not in _ALLOWED_WEIGHT_SCHEDULES:
        raise ValueError(
            "stable_velocity_repa_weight_schedule must be one of "
            f"{sorted(_ALLOWED_WEIGHT_SCHEDULES)}, got "
            f"{args.stable_velocity_repa_weight_schedule!r}"
        )
    if args.stable_velocity_repa_path_type not in _ALLOWED_PATH_TYPES:
        raise ValueError(
            "stable_velocity_repa_path_type must be one of "
            f"{sorted(_ALLOWED_PATH_TYPES)}, got "
            f"{args.stable_velocity_repa_path_type!r}"
        )
    if args.stable_velocity_stablevm_path_type not in _ALLOWED_PATH_TYPES:
        raise ValueError(
            "stable_velocity_stablevm_path_type must be one of "
            f"{sorted(_ALLOWED_PATH_TYPES)}, got "
            f"{args.stable_velocity_stablevm_path_type!r}"
        )
    if not (0.0 < args.stable_velocity_repa_tau <= 1.0):
        raise ValueError(
            "stable_velocity_repa_tau must be in (0, 1], got "
            f"{args.stable_velocity_repa_tau}"
        )
    if args.stable_velocity_repa_k <= 0.0:
        raise ValueError(
            f"stable_velocity_repa_k must be > 0, got {args.stable_velocity_repa_k}"
        )
    if not (0.0 <= args.stable_velocity_repa_min_weight <= 1.0):
        raise ValueError(
            "stable_velocity_repa_min_weight must be in [0, 1], got "
            f"{args.stable_velocity_repa_min_weight}"
        )
    if args.stable_velocity_repa_log_interval <= 0:
        raise ValueError(
            "stable_velocity_repa_log_interval must be > 0, got "
            f"{args.stable_velocity_repa_log_interval}"
        )
    if args.stable_velocity_stablevm_label_source not in _ALLOWED_STABLEVM_LABEL_SOURCES:
        raise ValueError(
            "stable_velocity_stablevm_label_source must be one of "
            f"{sorted(_ALLOWED_STABLEVM_LABEL_SOURCES)}, got "
            f"{args.stable_velocity_stablevm_label_source!r}"
        )
    if not (0.0 <= args.stable_velocity_stablevm_t_min < 1.0):
        raise ValueError(
            "stable_velocity_stablevm_t_min must be in [0, 1), got "
            f"{args.stable_velocity_stablevm_t_min}"
        )
    if not (0.0 <= args.stable_velocity_stablevm_blend <= 1.0):
        raise ValueError(
            "stable_velocity_stablevm_blend must be in [0, 1], got "
            f"{args.stable_velocity_stablevm_blend}"
        )
    if args.stable_velocity_stablevm_bank_capacity_per_label <= 0:
        raise ValueError(
            "stable_velocity_stablevm_bank_capacity_per_label must be > 0, got "
            f"{args.stable_velocity_stablevm_bank_capacity_per_label}"
        )
    if args.stable_velocity_stablevm_refs_per_sample <= 0:
        raise ValueError(
            "stable_velocity_stablevm_refs_per_sample must be > 0, got "
            f"{args.stable_velocity_stablevm_refs_per_sample}"
        )
    if args.stable_velocity_stablevm_min_refs <= 0:
        raise ValueError(
            "stable_velocity_stablevm_min_refs must be > 0, got "
            f"{args.stable_velocity_stablevm_min_refs}"
        )
    if (
        args.stable_velocity_stablevm_min_refs
        > args.stable_velocity_stablevm_refs_per_sample
    ):
        raise ValueError(
            "stable_velocity_stablevm_min_refs must be <= "
            "stable_velocity_stablevm_refs_per_sample"
        )
    if args.stable_velocity_stablevm_ref_chunk_size <= 0:
        raise ValueError(
            "stable_velocity_stablevm_ref_chunk_size must be > 0, got "
            f"{args.stable_velocity_stablevm_ref_chunk_size}"
        )
    if args.stable_velocity_stablevm_numerical_eps <= 0.0:
        raise ValueError(
            "stable_velocity_stablevm_numerical_eps must be > 0, got "
            f"{args.stable_velocity_stablevm_numerical_eps}"
        )
    if args.stable_velocity_stablevm_log_interval <= 0:
        raise ValueError(
            "stable_velocity_stablevm_log_interval must be > 0, got "
            f"{args.stable_velocity_stablevm_log_interval}"
        )

    if not args.enable_stable_velocity:
        if args.stable_velocity_stablevm_enable_target:
            logger.warning(
                "stable_velocity_stablevm_enable_target=true has no effect because "
                "enable_stable_velocity=false."
            )
        return

    if (
        not args.stable_velocity_repa_enabled
        and not args.stable_velocity_stablevm_enable_target
    ):
        logger.warning(
            "StableVelocity is enabled but both stable_velocity_repa_enabled and "
            "stable_velocity_stablevm_enable_target are false; no current module "
            "will consume these settings."
        )
        return

    repa_active = bool(config.get("enable_repa", False)) or bool(
        config.get("enable_irepa", False)
    )
    if args.stable_velocity_repa_enabled and not repa_active:
        logger.warning(
            "StableVelocity is enabled but both enable_repa and enable_irepa are false. "
            "No REPA loss path is active."
        )

    if args.stable_velocity_repa_enabled:
        logger.info(
            "StableVelocity REPA weighting enabled (schedule=%s, tau=%.3f, k=%.3f, "
            "path=%s, min_weight=%.3f, log_interval=%d).",
            args.stable_velocity_repa_weight_schedule,
            args.stable_velocity_repa_tau,
            args.stable_velocity_repa_k,
            args.stable_velocity_repa_path_type,
            args.stable_velocity_repa_min_weight,
            args.stable_velocity_repa_log_interval,
        )
    if args.stable_velocity_stablevm_enable_target:
        logger.info(
            "StableVelocity StableVM target enabled (label_source=%s, t_min=%.3f, "
            "blend=%.3f, path=%s, bank=%d, refs=%d, min_refs=%d, chunk=%d, "
            "fallback=%s, eps=%.1e, log_interval=%d).",
            args.stable_velocity_stablevm_label_source,
            args.stable_velocity_stablevm_t_min,
            args.stable_velocity_stablevm_blend,
            args.stable_velocity_stablevm_path_type,
            args.stable_velocity_stablevm_bank_capacity_per_label,
            args.stable_velocity_stablevm_refs_per_sample,
            args.stable_velocity_stablevm_min_refs,
            args.stable_velocity_stablevm_ref_chunk_size,
            args.stable_velocity_stablevm_use_global_fallback,
            args.stable_velocity_stablevm_numerical_eps,
            args.stable_velocity_stablevm_log_interval,
        )
