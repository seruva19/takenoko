from __future__ import annotations

from typing import Any, Dict

from utils.model_utils import str_to_dtype


def parse_semanticgen_config(
    args: Any, config: Dict[str, Any], logger: Any
) -> None:
    """Parse SemanticGen LoRA settings onto args."""
    # SPEC:semanticgen_lora:config - parse training-only semantic conditioning.
    args.enable_semanticgen_lora = bool(config.get("enable_semanticgen_lora", False))
    args.semantic_encoder_name = config.get(
        "semantic_encoder_name", "dinov2-vit-b14"
    )
    args.semantic_encoder_type = str(config.get("semantic_encoder_type", "repa"))
    allowed_semantic_encoder_types = {"repa", "hf", "qwen_vl"}
    if args.semantic_encoder_type not in allowed_semantic_encoder_types:
        raise ValueError(
            f"Invalid semantic_encoder_type '{args.semantic_encoder_type}'. "
            f"Expected one of {sorted(allowed_semantic_encoder_types)}."
        )
    args.semantic_encoder_dtype = str(
        config.get("semantic_encoder_dtype", "float16")
    )
    try:
        str_to_dtype(args.semantic_encoder_dtype)
    except Exception as exc:
        raise ValueError(
            f"Invalid semantic_encoder_dtype '{args.semantic_encoder_dtype}'."
        ) from exc
    args.semantic_encoder_resolution = int(
        config.get("semantic_encoder_resolution", 256)
    )
    if args.semantic_encoder_resolution <= 0:
        raise ValueError(
            "semantic_encoder_resolution must be > 0, got "
            f"{args.semantic_encoder_resolution}"
        )
    args.semantic_encoder_fps = float(config.get("semantic_encoder_fps", 2.0))
    if args.semantic_encoder_fps <= 0:
        raise ValueError(
            f"semantic_encoder_fps must be > 0, got {args.semantic_encoder_fps}"
        )
    args.semantic_encoder_target_fps = float(
        config.get("semantic_encoder_target_fps", 16.0)
    )
    if args.semantic_encoder_target_fps <= 0:
        raise ValueError(
            "semantic_encoder_target_fps must be > 0, got "
            f"{args.semantic_encoder_target_fps}"
        )
    args.semantic_encoder_stride = int(config.get("semantic_encoder_stride", 1))
    if args.semantic_encoder_stride < 1:
        raise ValueError(
            f"semantic_encoder_stride must be >= 1, got {args.semantic_encoder_stride}"
        )
    args.semantic_encoder_frame_limit = config.get(
        "semantic_encoder_frame_limit", None
    )
    if args.semantic_encoder_frame_limit is not None:
        args.semantic_encoder_frame_limit = int(args.semantic_encoder_frame_limit)
        if args.semantic_encoder_frame_limit < 1:
            raise ValueError(
                "semantic_encoder_frame_limit must be >= 1 when set, got "
                f"{args.semantic_encoder_frame_limit}"
            )
    args.semantic_embed_dim = int(config.get("semantic_embed_dim", 1024))
    if args.semantic_embed_dim <= 0:
        raise ValueError(
            f"semantic_embed_dim must be > 0, got {args.semantic_embed_dim}"
        )
    args.semantic_compress_dim = int(config.get("semantic_compress_dim", 256))
    if args.semantic_compress_dim <= 0 or args.semantic_compress_dim > args.semantic_embed_dim:
        raise ValueError(
            "semantic_compress_dim must be > 0 and <= semantic_embed_dim, got "
            f"{args.semantic_compress_dim}"
        )
    args.semantic_kl_weight = float(config.get("semantic_kl_weight", 0.1))
    if args.semantic_kl_weight < 0:
        raise ValueError(
            f"semantic_kl_weight must be >= 0, got {args.semantic_kl_weight}"
        )
    args.semantic_noise_std = float(config.get("semantic_noise_std", 0.0))
    if args.semantic_noise_std < 0:
        raise ValueError(
            f"semantic_noise_std must be >= 0, got {args.semantic_noise_std}"
        )
    args.semantic_context_mode = str(
        config.get("semantic_context_mode", "concat_text")
    )
    allowed_semantic_context_modes = {"concat_text", "concat_tokens"}
    if args.semantic_context_mode not in allowed_semantic_context_modes:
        raise ValueError(
            f"Invalid semantic_context_mode '{args.semantic_context_mode}'. "
            f"Expected one of {sorted(allowed_semantic_context_modes)}."
        )
    args.semantic_context_scale = float(
        config.get("semantic_context_scale", 1.0)
    )
    if args.semantic_context_scale < 0:
        raise ValueError(
            f"semantic_context_scale must be >= 0, got {args.semantic_context_scale}"
        )
    args.semantic_condition_dropout = float(
        config.get("semantic_condition_dropout", 0.0)
    )
    if not 0.0 <= args.semantic_condition_dropout <= 1.0:
        raise ValueError(
            "semantic_condition_dropout must be in [0, 1], got "
            f"{args.semantic_condition_dropout}"
        )
    args.semantic_condition_anneal_steps = int(
        config.get("semantic_condition_anneal_steps", 0)
    )
    if args.semantic_condition_anneal_steps < 0:
        raise ValueError(
            "semantic_condition_anneal_steps must be >= 0, got "
            f"{args.semantic_condition_anneal_steps}"
        )
    args.semantic_condition_min_scale = float(
        config.get("semantic_condition_min_scale", 0.0)
    )
    if not 0.0 <= args.semantic_condition_min_scale <= 1.0:
        raise ValueError(
            "semantic_condition_min_scale must be in [0, 1], got "
            f"{args.semantic_condition_min_scale}"
        )
    args.semantic_cache_enabled = bool(config.get("semantic_cache_enabled", False))
    args.semantic_cache_directory = config.get("semantic_cache_directory", None)
    args.semantic_cache_require = bool(config.get("semantic_cache_require", False))
    args.semantic_align_enabled = bool(config.get("semantic_align_enabled", False))
    args.semantic_align_lambda = float(config.get("semantic_align_lambda", 0.1))
    if args.semantic_align_lambda < 0:
        raise ValueError(
            f"semantic_align_lambda must be >= 0, got {args.semantic_align_lambda}"
        )
    args.semantic_align_block_index = config.get("semantic_align_block_index", 8)
    if args.semantic_align_block_index is not None:
        args.semantic_align_block_index = int(args.semantic_align_block_index)
        if args.semantic_align_block_index < 0:
            raise ValueError(
                "semantic_align_block_index must be >= 0, got "
                f"{args.semantic_align_block_index}"
            )
    args.semanticgen_lr = float(
        config.get("semanticgen_lr", getattr(args, "learning_rate", 1e-4))
    )
    if args.enable_semanticgen_lora:
        logger.info(
            "SemanticGen LoRA enabled (encoder=%s, type=%s, fps=%.2f, compress_dim=%d)",
            args.semantic_encoder_name,
            args.semantic_encoder_type,
            args.semantic_encoder_fps,
            args.semantic_compress_dim,
        )
    if args.semantic_align_enabled:
        logger.info(
            "Semantic alignment enabled (block=%d, lambda=%.3f).",
            args.semantic_align_block_index,
            args.semantic_align_lambda,
        )
