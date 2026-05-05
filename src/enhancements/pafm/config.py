"""Configuration parsing for Posterior-Augmented Flow Matching."""

from __future__ import annotations

from typing import Any, Dict


_ALLOWED_CANDIDATE_SOURCES = {"in_batch_nearest", "in_batch_random"}
_ALLOWED_CONDITION_SOURCES = {"auto", "concept_id", "dataset_index", "none"}
_ALLOWED_LIKELIHOOD_REDUCTIONS = {"sum", "mean"}
_ALLOWED_PAFM_KEYS = {
    "enable_pafm",
    "pafm_candidate_source",
    "pafm_num_candidates",
    "pafm_min_candidates",
    "pafm_condition_source",
    "pafm_allow_cross_condition_fallback",
    "pafm_likelihood_reduction",
    "pafm_weight_temperature",
    "pafm_t_min",
    "pafm_blend",
    "pafm_log_interval",
}


def parse_pafm_config(config: Dict[str, Any], args: Any, logger: Any) -> None:
    """Parse PAFM target-mixture settings into args.

    PAFM is train-time only and default-off. The implementation uses in-batch
    target proposals so regular LoRA inference and dataset cache formats stay
    unchanged.
    """

    unknown_keys = sorted(
        key
        for key in config
        if (key == "enable_pafm" or key.startswith("pafm_"))
        and key not in _ALLOWED_PAFM_KEYS
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported PAFM config keys: "
            + ", ".join(unknown_keys)
            + ". Use only documented pafm_* keys."
        )

    args.enable_pafm = bool(config.get("enable_pafm", False))
    args.pafm_candidate_source = str(
        config.get("pafm_candidate_source", "in_batch_nearest")
    ).lower()
    args.pafm_num_candidates = int(config.get("pafm_num_candidates", 4))
    args.pafm_min_candidates = int(config.get("pafm_min_candidates", 2))
    args.pafm_condition_source = str(
        config.get("pafm_condition_source", "auto")
    ).lower()
    args.pafm_allow_cross_condition_fallback = bool(
        config.get("pafm_allow_cross_condition_fallback", False)
    )
    args.pafm_likelihood_reduction = str(
        config.get("pafm_likelihood_reduction", "sum")
    ).lower()
    args.pafm_weight_temperature = float(config.get("pafm_weight_temperature", 1.0))
    args.pafm_t_min = float(config.get("pafm_t_min", 1e-4))
    args.pafm_blend = float(config.get("pafm_blend", 1.0))
    args.pafm_log_interval = int(config.get("pafm_log_interval", 100))

    if args.pafm_candidate_source not in _ALLOWED_CANDIDATE_SOURCES:
        raise ValueError(
            "pafm_candidate_source must be one of "
            f"{sorted(_ALLOWED_CANDIDATE_SOURCES)}, got "
            f"{args.pafm_candidate_source!r}"
        )
    if args.pafm_condition_source not in _ALLOWED_CONDITION_SOURCES:
        raise ValueError(
            "pafm_condition_source must be one of "
            f"{sorted(_ALLOWED_CONDITION_SOURCES)}, got "
            f"{args.pafm_condition_source!r}"
        )
    if args.pafm_likelihood_reduction not in _ALLOWED_LIKELIHOOD_REDUCTIONS:
        raise ValueError(
            "pafm_likelihood_reduction must be one of "
            f"{sorted(_ALLOWED_LIKELIHOOD_REDUCTIONS)}, got "
            f"{args.pafm_likelihood_reduction!r}"
        )
    if args.pafm_num_candidates < 1:
        raise ValueError(
            f"pafm_num_candidates must be >= 1, got {args.pafm_num_candidates}"
        )
    if args.pafm_min_candidates < 1:
        raise ValueError(
            f"pafm_min_candidates must be >= 1, got {args.pafm_min_candidates}"
        )
    if args.pafm_min_candidates > args.pafm_num_candidates:
        raise ValueError(
            "pafm_min_candidates must be <= pafm_num_candidates "
            f"({args.pafm_min_candidates} > {args.pafm_num_candidates})"
        )
    if args.pafm_weight_temperature <= 0.0:
        raise ValueError(
            "pafm_weight_temperature must be > 0, got "
            f"{args.pafm_weight_temperature}"
        )
    if not (0.0 <= args.pafm_t_min < 1.0):
        raise ValueError(f"pafm_t_min must be in [0, 1), got {args.pafm_t_min}")
    if not (0.0 <= args.pafm_blend <= 1.0):
        raise ValueError(f"pafm_blend must be in [0, 1], got {args.pafm_blend}")
    if args.pafm_log_interval < 0:
        raise ValueError(
            f"pafm_log_interval must be >= 0, got {args.pafm_log_interval}"
        )

    if not args.enable_pafm:
        return

    if args.pafm_num_candidates == 1:
        logger.warning(
            "PAFM is enabled with pafm_num_candidates=1; this reduces to the "
            "standard flow-matching target."
        )
    if bool(getattr(args, "enable_custom_loss_target", False)):
        logger.warning(
            "PAFM is enabled with enable_custom_loss_target=true. PAFM will only "
            "apply on steps where Takenoko falls back to the standard FM target."
        )
    if bool(getattr(args, "enable_temporal_pyramid_stagewise_target", False)):
        logger.warning(
            "PAFM is enabled with temporal pyramid stagewise targets. PAFM will "
            "skip those non-standard target steps."
        )
    if bool(config.get("enable_cdc_fm", False)):
        logger.warning(
            "PAFM is enabled with CDC-FM. PAFM will skip CDC-FM steps because "
            "CDC-FM changes the Gaussian noise path used by PAFM weights."
        )

    logger.info(
        "PAFM enabled (source=%s, K=%d, min_candidates=%d, condition=%s, "
        "cross_fallback=%s, likelihood=%s, temperature=%.4f, t_min=%.6f, "
        "blend=%.3f, log_interval=%d).",
        args.pafm_candidate_source,
        args.pafm_num_candidates,
        args.pafm_min_candidates,
        args.pafm_condition_source,
        args.pafm_allow_cross_condition_fallback,
        args.pafm_likelihood_reduction,
        args.pafm_weight_temperature,
        args.pafm_t_min,
        args.pafm_blend,
        args.pafm_log_interval,
    )
