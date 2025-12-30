from __future__ import annotations

from typing import Any, Dict


def apply_error_recycling_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Error-Recycling settings onto args (train-only, LoRA-gated)."""
    args.enable_error_recycling = bool(config.get("enable_error_recycling", False))
    args.error_buffer_k = int(config.get("error_buffer_k", 500))
    args.timestep_grid_size = int(config.get("timestep_grid_size", 25))
    args.num_grids = int(config.get("num_grids", 50))
    args.buffer_replacement_strategy = str(
        config.get("buffer_replacement_strategy", "random")
    )
    args.buffer_warmup_iter = int(config.get("buffer_warmup_iter", 50))
    args.error_modulate_factor = float(config.get("error_modulate_factor", 0.0))
    args.y_error_num = int(config.get("y_error_num", 1))
    args.y_error_sample_from_all_grids = bool(
        config.get("y_error_sample_from_all_grids", False)
    )
    args.y_error_sample_range = config.get("y_error_sample_range", None)
    args.error_setting = int(config.get("error_setting", 1))
    args.noise_prob = float(config.get("noise_prob", 0.9))
    args.y_prob = float(config.get("y_prob", 0.9))
    args.latent_prob = float(config.get("latent_prob", 0.9))
    args.clean_prob = float(config.get("clean_prob", 0.1))
    args.clean_buffer_update_prob = float(config.get("clean_buffer_update_prob", 0.1))
    args.use_last_y_error = bool(config.get("use_last_y_error", False))
    args.error_recycling_require_scheduler_errors = bool(
        config.get("error_recycling_require_scheduler_errors", False)
    )
    args.enable_svi_y_builder = bool(config.get("enable_svi_y_builder", False))
    args.svi_y_motion_source = str(config.get("svi_y_motion_source", "latent_last"))
    args.svi_y_num_motion_latent = int(config.get("svi_y_num_motion_latent", 1))
    args.svi_y_first_clip_mode = str(config.get("svi_y_first_clip_mode", "zeros"))
    args.svi_y_replay_buffer_size = int(
        config.get("svi_y_replay_buffer_size", 256)
    )
    args.svi_y_replay_key_mode = str(config.get("svi_y_replay_key_mode", "item_key"))
    args.svi_y_replay_use_sequence_index = bool(
        config.get("svi_y_replay_use_sequence_index", False)
    )
    args.svi_y_replay_sequence_pattern = str(
        config.get("svi_y_replay_sequence_pattern", r"_(\\d+)-(\\d+)$")
    )

    if args.error_buffer_k < 0:
        raise ValueError("error_buffer_k must be >= 0")
    if args.timestep_grid_size <= 0:
        raise ValueError("timestep_grid_size must be > 0")
    if args.num_grids <= 0:
        raise ValueError("num_grids must be > 0")
    if args.y_error_num <= 0:
        raise ValueError("y_error_num must be > 0")
    if args.error_modulate_factor < 0.0 or args.error_modulate_factor > 1.0:
        raise ValueError("error_modulate_factor must be between 0 and 1")
    if args.error_setting not in (0, 1, 2, 3, 4):
        raise ValueError("error_setting must be one of: 0, 1, 2, 3, 4")
    if args.svi_y_num_motion_latent < 0:
        raise ValueError("svi_y_num_motion_latent must be >= 0")
    if args.svi_y_replay_buffer_size < 0:
        raise ValueError("svi_y_replay_buffer_size must be >= 0")

    allowed_motion_sources = {"latent_last", "zeros", "replay_buffer"}
    if args.svi_y_motion_source not in allowed_motion_sources:
        raise ValueError(
            "svi_y_motion_source must be one of: "
            + ", ".join(sorted(allowed_motion_sources))
        )
    allowed_first_clip_modes = {"zeros", "current"}
    if args.svi_y_first_clip_mode not in allowed_first_clip_modes:
        raise ValueError(
            "svi_y_first_clip_mode must be one of: "
            + ", ".join(sorted(allowed_first_clip_modes))
        )
    allowed_replay_key_modes = {"item_key", "base_key"}
    if args.svi_y_replay_key_mode not in allowed_replay_key_modes:
        raise ValueError(
            "svi_y_replay_key_mode must be one of: "
            + ", ".join(sorted(allowed_replay_key_modes))
        )
    if not isinstance(args.svi_y_replay_sequence_pattern, str) or not args.svi_y_replay_sequence_pattern:
        raise ValueError("svi_y_replay_sequence_pattern must be a non-empty string")

    allowed_strategies = {"random", "l2_similarity", "l2_batch", "fifo"}
    if args.buffer_replacement_strategy not in allowed_strategies:
        raise ValueError(
            "buffer_replacement_strategy must be one of: "
            + ", ".join(sorted(allowed_strategies))
        )

    for name in (
        "noise_prob",
        "y_prob",
        "latent_prob",
        "clean_prob",
        "clean_buffer_update_prob",
    ):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if args.y_error_sample_range:
        if not isinstance(args.y_error_sample_range, str):
            raise ValueError("y_error_sample_range must be a string like 'start,end'")
        parts = [p.strip() for p in args.y_error_sample_range.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("y_error_sample_range must be in 'start,end' format")
        try:
            start = int(parts[0])
            end = int(parts[1])
        except Exception as exc:
            raise ValueError(
                "y_error_sample_range must contain integer values"
            ) from exc
        if start < 0 or end < start:
            raise ValueError("y_error_sample_range must satisfy 0 <= start <= end")

    if args.enable_error_recycling:
        logger.info(
            "Error recycling enabled: buffer_k=%d grid_size=%d num_grids=%d strategy=%s",
            args.error_buffer_k,
            args.timestep_grid_size,
            args.num_grids,
            args.buffer_replacement_strategy,
        )
        if args.error_recycling_require_scheduler_errors:
            logger.info("Error recycling requires scheduler-based errors.")
        if args.enable_svi_y_builder:
            logger.info(
                "SVI y builder enabled: motion_source=%s num_motion_latent=%d",
                args.svi_y_motion_source,
                args.svi_y_num_motion_latent,
            )
            if args.svi_y_motion_source == "replay_buffer":
                logger.info(
                    "SVI y replay buffer: size=%d key_mode=%s first_clip_mode=%s",
                    args.svi_y_replay_buffer_size,
                    args.svi_y_replay_key_mode,
                    args.svi_y_first_clip_mode,
                )
                if args.svi_y_replay_use_sequence_index:
                    logger.info(
                        "SVI y replay sequence indexing enabled: pattern=%s",
                        args.svi_y_replay_sequence_pattern,
                    )
    elif args.enable_svi_y_builder:
        logger.info(
            "SVI y builder requested but enable_error_recycling is false; it will remain inactive."
        )
