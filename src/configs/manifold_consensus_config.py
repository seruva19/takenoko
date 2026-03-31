"""Config parsing and validation for manifold consensus training."""

from __future__ import annotations

from typing import Any, Dict


_ALLOWED_LOSS_TYPES = {"one_minus_cosine", "mse"}


def apply_manifold_consensus_config(
    args: Any, config: Dict[str, Any], logger: Any
) -> None:
    args.enable_manifold_consensus = bool(config.get("enable_manifold_consensus", False))
    args.manifold_consensus_weight = float(config.get("manifold_consensus_weight", 0.1))
    args.manifold_consensus_num_views = int(config.get("manifold_consensus_num_views", 3))
    args.manifold_consensus_layer_start = int(
        config.get("manifold_consensus_layer_start", 20)
    )
    args.manifold_consensus_layer_end = int(
        config.get("manifold_consensus_layer_end", 29)
    )
    args.manifold_consensus_min_timestep = float(
        config.get("manifold_consensus_min_timestep", 970.0)
    )
    args.manifold_consensus_max_timestep = float(
        config.get("manifold_consensus_max_timestep", 1000.0)
    )
    args.manifold_consensus_start_step = int(
        config.get("manifold_consensus_start_step", 0)
    )
    args.manifold_consensus_interval_steps = int(
        config.get("manifold_consensus_interval_steps", 1)
    )
    args.manifold_consensus_apply_prob = float(
        config.get("manifold_consensus_apply_prob", 1.0)
    )
    args.manifold_consensus_loss_type = str(
        config.get("manifold_consensus_loss_type", "mse")
    ).lower()
    args.manifold_consensus_include_student_in_target = bool(
        config.get("manifold_consensus_include_student_in_target", False)
    )
    args.manifold_consensus_normalize_features = bool(
        config.get("manifold_consensus_normalize_features", True)
    )

    if not args.enable_manifold_consensus:
        return

    if args.manifold_consensus_weight < 0.0:
        raise ValueError("manifold_consensus_weight must be >= 0")
    if args.manifold_consensus_num_views < 2:
        raise ValueError("manifold_consensus_num_views must be >= 2")
    if args.manifold_consensus_layer_start < 0:
        raise ValueError("manifold_consensus_layer_start must be >= 0")
    if args.manifold_consensus_layer_end < args.manifold_consensus_layer_start:
        raise ValueError(
            "manifold_consensus_layer_end must be >= manifold_consensus_layer_start"
        )
    if (
        args.manifold_consensus_min_timestep
        > args.manifold_consensus_max_timestep
    ):
        raise ValueError(
            "manifold_consensus_min_timestep must be <= manifold_consensus_max_timestep"
        )
    if args.manifold_consensus_start_step < 0:
        raise ValueError("manifold_consensus_start_step must be >= 0")
    if args.manifold_consensus_interval_steps < 1:
        raise ValueError("manifold_consensus_interval_steps must be >= 1")
    if not (0.0 <= args.manifold_consensus_apply_prob <= 1.0):
        raise ValueError("manifold_consensus_apply_prob must be in [0, 1]")
    if args.manifold_consensus_loss_type not in _ALLOWED_LOSS_TYPES:
        raise ValueError(
            "manifold_consensus_loss_type must be one of "
            f"{sorted(_ALLOWED_LOSS_TYPES)}"
        )

    logger.info(
        (
            "Manifold consensus enabled (weight=%.3f, views=%d, layers=%d-%d, "
            "timesteps=%.1f-%.1f, start_step=%d, interval=%d, loss=%s)"
        ),
        args.manifold_consensus_weight,
        args.manifold_consensus_num_views,
        args.manifold_consensus_layer_start,
        args.manifold_consensus_layer_end,
        args.manifold_consensus_min_timestep,
        args.manifold_consensus_max_timestep,
        args.manifold_consensus_start_step,
        args.manifold_consensus_interval_steps,
        args.manifold_consensus_loss_type,
    )
