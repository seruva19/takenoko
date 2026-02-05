"""Configuration parser for MOALIGN (motion-centric representation alignment)."""

from __future__ import annotations

from typing import Any, Dict
import logging
import os


def parse_moalign_config(config: Dict[str, Any], args: Any, logger: logging.Logger) -> None:
    """Populate MOALIGN args with validation and safe defaults."""

    args.enable_moalign = bool(config.get("enable_moalign", False))
    args.moalign_encoder_name = str(config.get("moalign_encoder_name", "dinov2-vit-b"))
    args.moalign_alignment_depth = int(config.get("moalign_alignment_depth", 18))
    raw_alignment_depths = config.get("moalign_alignment_depths", None)
    if raw_alignment_depths is None:
        args.moalign_alignment_depths = [args.moalign_alignment_depth]
    else:
        if not isinstance(raw_alignment_depths, (list, tuple)):
            raise ValueError(
                "moalign_alignment_depths must be a list of integers "
                f"or omitted, got {type(raw_alignment_depths).__name__}"
            )
        if len(raw_alignment_depths) == 0:
            raise ValueError("moalign_alignment_depths must not be empty when provided")
        parsed_depths = []
        for depth in raw_alignment_depths:
            parsed_depths.append(int(depth))
        # Stable de-duplication while preserving config order.
        args.moalign_alignment_depths = list(dict.fromkeys(parsed_depths))

    args.moalign_loss_lambda = float(config.get("moalign_loss_lambda", 0.5))
    args.moalign_temporal_tau = float(config.get("moalign_temporal_tau", 10.0))
    args.moalign_projection_hidden_dim = int(
        config.get("moalign_projection_hidden_dim", 256)
    )
    args.moalign_projection_out_dim = int(config.get("moalign_projection_out_dim", 64))
    args.moalign_max_tokens = int(config.get("moalign_max_tokens", 256))
    args.moalign_motion_target_mode = str(
        config.get("moalign_motion_target_mode", "delta")
    ).lower()
    args.moalign_spatial_weight = float(config.get("moalign_spatial_weight", 1.0))
    args.moalign_temporal_weight = float(config.get("moalign_temporal_weight", 1.0))
    args.moalign_spatial_align = bool(config.get("moalign_spatial_align", True))
    args.moalign_input_resolution = int(config.get("moalign_input_resolution", 256))
    args.moalign_detach_targets = bool(config.get("moalign_detach_targets", False))
    args.enable_moalign_stage1_training = bool(
        config.get("enable_moalign_stage1_training", False)
    )
    args.moalign_use_stage1_teacher = bool(config.get("moalign_use_stage1_teacher", False))
    args.moalign_stage1_checkpoint = str(config.get("moalign_stage1_checkpoint", "") or "")
    args.moalign_stage1_num_epochs = int(config.get("moalign_stage1_num_epochs", 1))
    args.moalign_stage1_max_steps = int(config.get("moalign_stage1_max_steps", 2000))
    args.moalign_stage1_learning_rate = float(
        config.get("moalign_stage1_learning_rate", 1e-4)
    )
    args.moalign_stage1_weight_decay = float(
        config.get("moalign_stage1_weight_decay", 0.0)
    )
    args.moalign_stage1_batch_size = int(config.get("moalign_stage1_batch_size", 1))
    args.moalign_stage1_num_workers = int(config.get("moalign_stage1_num_workers", 0))
    args.moalign_stage1_log_interval = int(config.get("moalign_stage1_log_interval", 20))
    args.moalign_stage1_save_interval = int(config.get("moalign_stage1_save_interval", 500))
    args.moalign_stage1_grad_clip_norm = float(
        config.get("moalign_stage1_grad_clip_norm", 1.0)
    )
    args.moalign_stage1_token_reg_weight = float(
        config.get("moalign_stage1_token_reg_weight", 0.0)
    )
    args.moalign_stage1_flow_source = str(
        config.get("moalign_stage1_flow_source", "auto")
    ).lower()
    args.moalign_stage1_allow_raft_fallback = bool(
        config.get("moalign_stage1_allow_raft_fallback", True)
    )
    args.moalign_stage1_raft_model = str(
        config.get("moalign_stage1_raft_model", "raft_small")
    ).lower()

    allowed_modes = {"delta", "encoder"}
    if args.moalign_motion_target_mode not in allowed_modes:
        raise ValueError(
            f"moalign_motion_target_mode must be one of {sorted(allowed_modes)}, "
            f"got {args.moalign_motion_target_mode!r}"
        )

    if args.moalign_alignment_depth < 0:
        raise ValueError(
            f"moalign_alignment_depth must be >= 0, got {args.moalign_alignment_depth}"
        )
    for depth in args.moalign_alignment_depths:
        if depth < 0:
            raise ValueError(
                f"moalign_alignment_depths entries must be >= 0, got {depth}"
            )
    if args.moalign_loss_lambda < 0:
        raise ValueError(
            f"moalign_loss_lambda must be >= 0, got {args.moalign_loss_lambda}"
        )
    if args.enable_moalign and args.moalign_loss_lambda <= 0:
        raise ValueError("moalign_loss_lambda must be > 0 when enable_moalign is true")
    if args.moalign_temporal_tau <= 0:
        raise ValueError(
            f"moalign_temporal_tau must be > 0, got {args.moalign_temporal_tau}"
        )
    if args.moalign_projection_hidden_dim <= 0:
        raise ValueError(
            "moalign_projection_hidden_dim must be > 0, "
            f"got {args.moalign_projection_hidden_dim}"
        )
    if args.moalign_projection_out_dim <= 0:
        raise ValueError(
            "moalign_projection_out_dim must be > 0, "
            f"got {args.moalign_projection_out_dim}"
        )
    if args.moalign_max_tokens <= 0:
        raise ValueError(f"moalign_max_tokens must be > 0, got {args.moalign_max_tokens}")
    if args.moalign_spatial_weight < 0:
        raise ValueError(
            f"moalign_spatial_weight must be >= 0, got {args.moalign_spatial_weight}"
        )
    if args.moalign_temporal_weight < 0:
        raise ValueError(
            f"moalign_temporal_weight must be >= 0, got {args.moalign_temporal_weight}"
        )
    if args.moalign_input_resolution <= 0:
        raise ValueError(
            f"moalign_input_resolution must be > 0, got {args.moalign_input_resolution}"
        )
    if args.moalign_stage1_num_epochs <= 0:
        raise ValueError(
            f"moalign_stage1_num_epochs must be > 0, got {args.moalign_stage1_num_epochs}"
        )
    if args.moalign_stage1_max_steps <= 0:
        raise ValueError(
            f"moalign_stage1_max_steps must be > 0, got {args.moalign_stage1_max_steps}"
        )
    if args.moalign_stage1_learning_rate <= 0:
        raise ValueError(
            "moalign_stage1_learning_rate must be > 0, "
            f"got {args.moalign_stage1_learning_rate}"
        )
    if args.moalign_stage1_weight_decay < 0:
        raise ValueError(
            "moalign_stage1_weight_decay must be >= 0, "
            f"got {args.moalign_stage1_weight_decay}"
        )
    if args.moalign_stage1_batch_size <= 0:
        raise ValueError(
            f"moalign_stage1_batch_size must be > 0, got {args.moalign_stage1_batch_size}"
        )
    if args.moalign_stage1_num_workers < 0:
        raise ValueError(
            f"moalign_stage1_num_workers must be >= 0, got {args.moalign_stage1_num_workers}"
        )
    if args.moalign_stage1_log_interval <= 0:
        raise ValueError(
            "moalign_stage1_log_interval must be > 0, "
            f"got {args.moalign_stage1_log_interval}"
        )
    if args.moalign_stage1_save_interval < 0:
        raise ValueError(
            "moalign_stage1_save_interval must be >= 0, "
            f"got {args.moalign_stage1_save_interval}"
        )
    if args.moalign_stage1_grad_clip_norm < 0:
        raise ValueError(
            "moalign_stage1_grad_clip_norm must be >= 0, "
            f"got {args.moalign_stage1_grad_clip_norm}"
        )
    if args.moalign_stage1_token_reg_weight < 0:
        raise ValueError(
            "moalign_stage1_token_reg_weight must be >= 0, "
            f"got {args.moalign_stage1_token_reg_weight}"
        )

    allowed_stage1_sources = {"auto", "cache", "raft"}
    if args.moalign_stage1_flow_source not in allowed_stage1_sources:
        raise ValueError(
            "moalign_stage1_flow_source must be one of "
            f"{sorted(allowed_stage1_sources)}, got {args.moalign_stage1_flow_source!r}"
        )

    allowed_raft_models = {"raft_small", "raft_large"}
    if args.moalign_stage1_raft_model not in allowed_raft_models:
        raise ValueError(
            "moalign_stage1_raft_model must be one of "
            f"{sorted(allowed_raft_models)}, got {args.moalign_stage1_raft_model!r}"
        )

    if args.enable_moalign and bool(getattr(args, "sara_enabled", False)):
        raise ValueError("enable_moalign and sara_enabled are mutually exclusive")
    if args.enable_moalign and bool(getattr(args, "enable_repa", False)):
        raise ValueError("enable_moalign and enable_repa are mutually exclusive")
    if args.enable_moalign and bool(getattr(args, "enable_irepa", False)):
        raise ValueError("enable_moalign and enable_irepa are mutually exclusive")
    if args.moalign_use_stage1_teacher and args.moalign_stage1_checkpoint == "":
        raise ValueError(
            "moalign_stage1_checkpoint must be set when moalign_use_stage1_teacher is true"
        )
    if (
        args.moalign_use_stage1_teacher
        and args.moalign_stage1_checkpoint
        and not os.path.exists(args.moalign_stage1_checkpoint)
    ):
        logger.warning(
            "MOALIGN Stage-1 checkpoint does not exist yet: %s. "
            "This is allowed for Stage-1 runs; Stage-2 training will fail if the file is still missing at helper initialization.",
            args.moalign_stage1_checkpoint,
        )

    if args.enable_moalign:
        logger.info(
            "MOALIGN enabled (encoder=%s, depths=%s, lambda=%.4f, tau=%.3f, "
            "mode=%s, spatial=%.3f, temporal=%.3f).",
            args.moalign_encoder_name,
            args.moalign_alignment_depths,
            args.moalign_loss_lambda,
            args.moalign_temporal_tau,
            args.moalign_motion_target_mode,
            args.moalign_spatial_weight,
            args.moalign_temporal_weight,
        )
    if args.moalign_use_stage1_teacher:
        logger.info(
            "MOALIGN Stage-1 teacher enabled (checkpoint=%s).",
            args.moalign_stage1_checkpoint,
        )
    if args.enable_moalign_stage1_training:
        logger.info(
            "MOALIGN Stage-1 training requested (steps=%d, epochs=%d, lr=%.6f, flow_source=%s).",
            args.moalign_stage1_max_steps,
            args.moalign_stage1_num_epochs,
            args.moalign_stage1_learning_rate,
            args.moalign_stage1_flow_source,
        )
