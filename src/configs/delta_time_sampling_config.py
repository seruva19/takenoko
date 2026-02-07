"""Stochastic delta-time sampling configuration helpers."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def apply_delta_time_sampling_config(
    args: argparse.Namespace, config: Dict[str, Any], logger: Any
) -> argparse.Namespace:
    """Parse and validate top-level defaults for dataset delta-time sampling."""

    args.enable_stochastic_delta_time_sampling = _parse_bool(
        config.get("enable_stochastic_delta_time_sampling", False),
        False,
    )
    args.delta_time_sampling_distribution = str(
        config.get("delta_time_sampling_distribution", "gamma")
    ).lower()
    args.delta_time_sampling_gamma_concentration = float(
        config.get("delta_time_sampling_gamma_concentration", 3.0)
    )
    args.delta_time_sampling_gamma_rate = float(
        config.get("delta_time_sampling_gamma_rate", 12.0)
    )
    args.delta_time_sampling_max_offset_frames = int(
        config.get("delta_time_sampling_max_offset_frames", 8)
    )
    args.delta_time_sampling_min_step_frames = int(
        config.get("delta_time_sampling_min_step_frames", 1)
    )
    args.delta_time_sampling_seed_offset = int(
        config.get("delta_time_sampling_seed_offset", 0)
    )

    allowed_distributions = {"gamma", "uniform"}
    if args.delta_time_sampling_distribution not in allowed_distributions:
        raise ValueError(
            "delta_time_sampling_distribution must be one of "
            f"{sorted(allowed_distributions)}, got "
            f"'{args.delta_time_sampling_distribution}'"
        )
    if args.delta_time_sampling_gamma_concentration <= 0.0:
        raise ValueError("delta_time_sampling_gamma_concentration must be > 0")
    if args.delta_time_sampling_gamma_rate <= 0.0:
        raise ValueError("delta_time_sampling_gamma_rate must be > 0")
    if args.delta_time_sampling_max_offset_frames < 0:
        raise ValueError("delta_time_sampling_max_offset_frames must be >= 0")
    if args.delta_time_sampling_min_step_frames < 1:
        raise ValueError("delta_time_sampling_min_step_frames must be >= 1")

    if args.enable_stochastic_delta_time_sampling:
        logger.info(
            "Stochastic delta-time sampling enabled (distribution=%s, max_offset=%s, min_step=%s).",
            args.delta_time_sampling_distribution,
            args.delta_time_sampling_max_offset_frames,
            args.delta_time_sampling_min_step_frames,
        )

    return args

