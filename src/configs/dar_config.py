from __future__ import annotations

from typing import Any, Dict


_QUERY_MODES = {"static", "static_timestep", "dynamic"}


def apply_dar_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Diffusion-Adaptive Routing config and validate safe defaults."""
    args.enable_dar = bool(config.get("enable_dar", False))
    args.dar_query_mode = str(
        config.get("dar_query_mode", "static_timestep")
    ).strip().lower()
    args.dar_chunk_size = int(config.get("dar_chunk_size", 4))
    args.dar_residual_blend = float(config.get("dar_residual_blend", 1.0))
    args.dar_dynamic_rank = int(config.get("dar_dynamic_rank", 16))
    args.dar_query_temperature = float(config.get("dar_query_temperature", 1.0))
    args.dar_norm_eps = float(config.get("dar_norm_eps", 1e-6))
    args.dar_final_aggregate = bool(config.get("dar_final_aggregate", True))
    args.dar_apply_in_eval = bool(config.get("dar_apply_in_eval", False))
    args.dar_lr_scale = float(config.get("dar_lr_scale", 1.0))
    args.dar_disable_block_checkpointing = bool(
        config.get("dar_disable_block_checkpointing", True)
    )

    if args.dar_query_mode not in _QUERY_MODES:
        raise ValueError(
            "dar_query_mode must be one of "
            f"{sorted(_QUERY_MODES)}, got {args.dar_query_mode!r}"
        )
    if args.dar_chunk_size < 1:
        raise ValueError("dar_chunk_size must be >= 1")
    if not 0.0 <= args.dar_residual_blend <= 1.0:
        raise ValueError("dar_residual_blend must be in [0, 1]")
    if args.dar_dynamic_rank < 1:
        raise ValueError("dar_dynamic_rank must be >= 1")
    if args.dar_query_temperature <= 0.0:
        raise ValueError("dar_query_temperature must be > 0")
    if args.dar_norm_eps <= 0.0:
        raise ValueError("dar_norm_eps must be > 0")
    if args.dar_lr_scale <= 0.0:
        raise ValueError("dar_lr_scale must be > 0")

    if args.enable_dar:
        if bool(config.get("enable_sprint", False)):
            raise ValueError(
                "enable_dar=true is incompatible with enable_sprint=true; "
                "both replace the Wan block stream."
            )
        if bool(config.get("enable_self_resampling_attention_routing", False)):
            raise ValueError(
                "enable_dar=true is incompatible with "
                "enable_self_resampling_attention_routing=true because both "
                "rewrite cross-layer attention history."
            )
        if bool(config.get("enable_tread", False)):
            raise ValueError(
                "enable_dar=true is incompatible with enable_tread=true because "
                "token routing changes sequence shapes while DAR keeps a "
                "cross-layer source history."
            )
        if bool(config.get("lean_attn_math", False)):
            logger.warning(
                "DAR uses the standard Wan block path; lean_attn_math will be "
                "ignored while enable_dar=true."
            )
        logger.info(
            "DAR enabled (query_mode=%s, chunk_size=%d, blend=%.3f, "
            "final_aggregate=%s, lr_scale=%.3f).",
            args.dar_query_mode,
            args.dar_chunk_size,
            args.dar_residual_blend,
            args.dar_final_aggregate,
            args.dar_lr_scale,
        )
