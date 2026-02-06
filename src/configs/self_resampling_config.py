from __future__ import annotations

from typing import Any, Dict


def apply_self_resampling_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse Self-Resampling settings onto args (train-only, LoRA-gated)."""
    args.enable_self_resampling = bool(config.get("enable_self_resampling", False))
    args.self_resampling_warmup_steps = int(
        config.get("self_resampling_warmup_steps", 0)
    )
    args.self_resampling_apply_prob = float(
        config.get("self_resampling_apply_prob", 1.0)
    )
    args.self_resampling_shift = float(config.get("self_resampling_shift", 0.6))
    args.self_resampling_min_t = float(config.get("self_resampling_min_t", 0.05))
    args.self_resampling_max_t = float(config.get("self_resampling_max_t", 0.95))
    args.self_resampling_blend = float(config.get("self_resampling_blend", 1.0))
    args.self_resampling_history_start_frame = int(
        config.get("self_resampling_history_start_frame", 0)
    )
    args.self_resampling_history_exclude_tail_frames = int(
        config.get("self_resampling_history_exclude_tail_frames", 1)
    )
    args.self_resampling_autoregressive_history = bool(
        config.get("self_resampling_autoregressive_history", True)
    )
    args.self_resampling_autoregressive_decay = float(
        config.get("self_resampling_autoregressive_decay", 0.5)
    )
    args.self_resampling_autoregressive_clip_multiplier = float(
        config.get("self_resampling_autoregressive_clip_multiplier", 3.0)
    )
    args.self_resampling_model_rollout = bool(
        config.get("self_resampling_model_rollout", False)
    )
    args.self_resampling_fast_rollout = bool(
        config.get("self_resampling_fast_rollout", True)
    )
    args.self_resampling_rollout_kv_cache = bool(
        config.get("self_resampling_rollout_kv_cache", True)
    )
    args.self_resampling_rollout_self_attn_kv_cache = bool(
        config.get("self_resampling_rollout_self_attn_kv_cache", True)
    )
    args.self_resampling_model_rollout_steps = int(
        config.get("self_resampling_model_rollout_steps", 1)
    )
    args.enable_self_resampling_history_routing = bool(
        config.get("enable_self_resampling_history_routing", False)
    )
    args.self_resampling_history_routing_mode = str(
        config.get("self_resampling_history_routing_mode", "frame_topk")
    )
    args.self_resampling_history_routing_top_k = int(
        config.get("self_resampling_history_routing_top_k", 5)
    )
    args.self_resampling_history_routing_expected_frames = int(
        config.get("self_resampling_history_routing_expected_frames", 20)
    )
    args.self_resampling_history_routing_start_layer_idx = int(
        config.get("self_resampling_history_routing_start_layer_idx", 2)
    )
    args.self_resampling_history_routing_end_layer_idx = int(
        config.get("self_resampling_history_routing_end_layer_idx", -2)
    )
    args.self_resampling_history_routing_always_keep_first_frame = bool(
        config.get("self_resampling_history_routing_always_keep_first_frame", True)
    )
    args.self_resampling_history_routing_always_keep_last_frame = bool(
        config.get("self_resampling_history_routing_always_keep_last_frame", False)
    )
    args.enable_self_resampling_attention_routing = bool(
        config.get("enable_self_resampling_attention_routing", False)
    )
    args.self_resampling_attention_routing_top_k = int(
        config.get("self_resampling_attention_routing_top_k", 5)
    )
    args.self_resampling_attention_routing_start_layer_idx = int(
        config.get("self_resampling_attention_routing_start_layer_idx", 2)
    )
    args.self_resampling_attention_routing_end_layer_idx = int(
        config.get("self_resampling_attention_routing_end_layer_idx", -2)
    )
    args.self_resampling_attention_routing_always_keep_first_frame = bool(
        config.get("self_resampling_attention_routing_always_keep_first_frame", True)
    )
    args.self_resampling_attention_routing_always_keep_last_frame = bool(
        config.get("self_resampling_attention_routing_always_keep_last_frame", False)
    )
    args.self_resampling_attention_routing_backend = str(
        config.get("self_resampling_attention_routing_backend", "exact")
    )
    args.self_resampling_log_interval = int(
        config.get("self_resampling_log_interval", 50)
    )

    if args.self_resampling_warmup_steps < 0:
        raise ValueError("self_resampling_warmup_steps must be >= 0")
    if not 0.0 <= args.self_resampling_apply_prob <= 1.0:
        raise ValueError("self_resampling_apply_prob must be between 0 and 1")
    if args.self_resampling_shift <= 0.0:
        raise ValueError("self_resampling_shift must be > 0")
    if not 0.0 <= args.self_resampling_min_t < 1.0:
        raise ValueError("self_resampling_min_t must be in [0, 1)")
    if not 0.0 < args.self_resampling_max_t <= 1.0:
        raise ValueError("self_resampling_max_t must be in (0, 1]")
    if args.self_resampling_min_t >= args.self_resampling_max_t:
        raise ValueError("self_resampling_min_t must be < self_resampling_max_t")
    if not 0.0 <= args.self_resampling_blend <= 1.0:
        raise ValueError("self_resampling_blend must be between 0 and 1")
    if args.self_resampling_history_start_frame < 0:
        raise ValueError("self_resampling_history_start_frame must be >= 0")
    if args.self_resampling_history_exclude_tail_frames < 0:
        raise ValueError("self_resampling_history_exclude_tail_frames must be >= 0")
    if not 0.0 <= args.self_resampling_autoregressive_decay <= 1.0:
        raise ValueError("self_resampling_autoregressive_decay must be between 0 and 1")
    if args.self_resampling_autoregressive_clip_multiplier <= 0.0:
        raise ValueError("self_resampling_autoregressive_clip_multiplier must be > 0")
    if args.self_resampling_model_rollout_steps <= 0:
        raise ValueError("self_resampling_model_rollout_steps must be > 0")
    if args.self_resampling_history_routing_top_k <= 0:
        raise ValueError("self_resampling_history_routing_top_k must be > 0")
    if args.self_resampling_history_routing_expected_frames <= 0:
        raise ValueError("self_resampling_history_routing_expected_frames must be > 0")
    if args.self_resampling_attention_routing_top_k <= 0:
        raise ValueError("self_resampling_attention_routing_top_k must be > 0")
    if args.self_resampling_attention_routing_backend not in (
        "exact",
        "kernel_frame_topk",
    ):
        raise ValueError(
            "self_resampling_attention_routing_backend must be one of: "
            "exact, kernel_frame_topk"
        )
    if args.self_resampling_history_routing_mode not in (
        "frame_topk",
        "frame_stride",
        "frame_contiguous",
    ):
        raise ValueError(
            "self_resampling_history_routing_mode must be one of: "
            "frame_topk, frame_stride, frame_contiguous"
        )
    if args.self_resampling_log_interval <= 0:
        raise ValueError("self_resampling_log_interval must be > 0")

    if args.enable_self_resampling_history_routing:
        # Approximate paper top-k history routing through existing TREAD routes.
        keep_ratio = min(
            1.0,
            max(
                1.0 / float(args.self_resampling_history_routing_expected_frames),
                float(args.self_resampling_history_routing_top_k)
                / float(args.self_resampling_history_routing_expected_frames),
            ),
        )
        selection_ratio = max(0.0, min(1.0, 1.0 - keep_ratio))
        route = {
            "selection_ratio": selection_ratio,
            "start_layer_idx": args.self_resampling_history_routing_start_layer_idx,
            "end_layer_idx": args.self_resampling_history_routing_end_layer_idx,
            "top_k_frames": int(args.self_resampling_history_routing_top_k),
            "always_keep_first_frame": bool(
                args.self_resampling_history_routing_always_keep_first_frame
            ),
            "always_keep_last_frame": bool(
                args.self_resampling_history_routing_always_keep_last_frame
            ),
        }
        existing_cfg = getattr(args, "tread_config", None)
        existing_routes = (
            existing_cfg.get("routes")
            if isinstance(existing_cfg, dict)
            and isinstance(existing_cfg.get("routes"), list)
            else None
        )
        if existing_routes:
            logger.info(
                "Self-resampling history routing requested; appending a route to existing tread_config."
            )
            current_mode = str(getattr(args, "tread_mode", "full"))
            if current_mode != args.self_resampling_history_routing_mode:
                logger.info(
                    "Switching tread_mode from '%s' to '%s' for self-resampling history routing.",
                    current_mode,
                    args.self_resampling_history_routing_mode,
                )
                args.tread_mode = args.self_resampling_history_routing_mode
            existing_routes.append(route)
            args.enable_tread = True
        else:
            args.enable_tread = True
            args.tread_mode = args.self_resampling_history_routing_mode
            args.tread_config = {"routes": [route]}
        logger.info(
            "Self-resampling history routing enabled via TREAD: mode=%s top_k=%d expected_frames=%d",
            args.tread_mode,
            args.self_resampling_history_routing_top_k,
            args.self_resampling_history_routing_expected_frames,
        )

    args.self_resampling_attention_routing_config = None
    if args.enable_self_resampling_attention_routing:
        args.self_resampling_attention_routing_config = {
            "enabled": True,
            "top_k_frames": args.self_resampling_attention_routing_top_k,
            "start_layer_idx": args.self_resampling_attention_routing_start_layer_idx,
            "end_layer_idx": args.self_resampling_attention_routing_end_layer_idx,
            "always_keep_first_frame": (
                args.self_resampling_attention_routing_always_keep_first_frame
            ),
            "always_keep_last_frame": (
                args.self_resampling_attention_routing_always_keep_last_frame
            ),
            "backend": args.self_resampling_attention_routing_backend,
        }
        logger.info(
            "Self-resampling token-wise attention routing enabled: backend=%s top_k=%d layers=[%d,%d]",
            args.self_resampling_attention_routing_backend,
            args.self_resampling_attention_routing_top_k,
            args.self_resampling_attention_routing_start_layer_idx,
            args.self_resampling_attention_routing_end_layer_idx,
        )

    if args.enable_self_resampling:
        logger.info(
            "Self-resampling enabled: warmup_steps=%d shift=%.3f t_range=[%.3f, %.3f] blend=%.3f ar_history=%s model_rollout=%s fast_rollout=%s rollout_kv_cache=%s rollout_self_attn_kv_cache=%s rollout_steps=%d",
            args.self_resampling_warmup_steps,
            args.self_resampling_shift,
            args.self_resampling_min_t,
            args.self_resampling_max_t,
            args.self_resampling_blend,
            str(args.self_resampling_autoregressive_history).lower(),
            str(args.self_resampling_model_rollout).lower(),
            str(args.self_resampling_fast_rollout).lower(),
            str(args.self_resampling_rollout_kv_cache).lower(),
            str(args.self_resampling_rollout_self_attn_kv_cache).lower(),
            args.self_resampling_model_rollout_steps,
        )
