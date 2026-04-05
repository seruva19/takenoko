from __future__ import annotations

import os
from typing import Any, Dict


def parse_motion_preservation_config(
    config: Dict[str, Any],
    args: Any,
    logger: Any,
) -> None:
    args.freeze_early_blocks = int(config.get("freeze_early_blocks", 0))
    args.freeze_block_indices = config.get("freeze_block_indices", None)
    args.block_lr_scales = config.get("block_lr_scales", None)
    args.non_block_lr_scale = float(config.get("non_block_lr_scale", 1.0))
    args.attn_geometry_lr_scale = float(config.get("attn_geometry_lr_scale", 1.0))
    args.freeze_attn_geometry = bool(config.get("freeze_attn_geometry", False))

    args.motion_preservation = bool(config.get("motion_preservation", False))
    args.motion_preservation_strict_mode = bool(
        config.get("motion_preservation_strict_mode", True)
    )
    args.motion_preservation_multiplier = float(
        config.get("motion_preservation_multiplier", 0.0)
    )
    args.motion_preservation_mode = str(
        config.get("motion_preservation_mode", "temporal") or "temporal"
    ).lower()
    args.motion_preservation_anchor_cache_size = int(
        config.get("motion_preservation_anchor_cache_size", 0)
    )
    args.motion_preservation_anchor_cache_auto = bool(
        config.get("motion_preservation_anchor_cache_auto", False)
    )
    args.motion_preservation_anchor_cache_auto_ratio = float(
        config.get("motion_preservation_anchor_cache_auto_ratio", 0.2)
    )
    args.motion_preservation_anchor_cache_auto_min = int(
        config.get("motion_preservation_anchor_cache_auto_min", 8)
    )
    args.motion_preservation_anchor_cache_auto_max = int(
        config.get("motion_preservation_anchor_cache_auto_max", 256)
    )
    args.motion_preservation_anchor_source = str(
        config.get("motion_preservation_anchor_source", "synthetic") or "synthetic"
    ).lower()
    args.motion_preservation_synthetic_frames = int(
        config.get("motion_preservation_synthetic_frames", 8)
    )
    args.motion_preservation_synthetic_temporal_corr = float(
        config.get("motion_preservation_synthetic_temporal_corr", 0.92)
    )
    args.motion_preservation_synthetic_dataset_mix = float(
        config.get("motion_preservation_synthetic_dataset_mix", 0.25)
    )
    args.motion_preservation_synthetic_content_seeded = bool(
        config.get("motion_preservation_synthetic_content_seeded", True)
    )
    args.motion_preservation_warmup_steps = int(
        config.get("motion_preservation_warmup_steps", 0)
    )
    args.motion_preservation_interval = int(
        config.get("motion_preservation_interval", 1)
    )
    args.motion_preservation_health_log_interval = int(
        config.get("motion_preservation_health_log_interval", 100)
    )
    args.motion_preservation_probability = config.get(
        "motion_preservation_probability", None
    )
    args.motion_preservation_num_sigmas = int(
        config.get("motion_preservation_num_sigmas", 1)
    )
    args.motion_preservation_sigma_values = config.get(
        "motion_preservation_sigma_values", None
    )
    args.motion_preservation_sigma_min = float(
        config.get("motion_preservation_sigma_min", 0.2)
    )
    args.motion_preservation_sigma_max = float(
        config.get("motion_preservation_sigma_max", 0.8)
    )
    args.motion_preservation_sigma_sampling = str(
        config.get("motion_preservation_sigma_sampling", "uniform") or "uniform"
    ).lower()
    args.motion_preservation_sigma_sampling_power = float(
        config.get("motion_preservation_sigma_sampling_power", 1.0)
    )
    args.motion_preservation_second_order_weight = float(
        config.get("motion_preservation_second_order_weight", 0.0)
    )
    args.motion_preservation_anchor_cache_path = config.get(
        "motion_preservation_anchor_cache_path", None
    )
    args.motion_preservation_anchor_cache_rebuild = bool(
        config.get("motion_preservation_anchor_cache_rebuild", False)
    )
    args.motion_preservation_teacher_chunk_frames = int(
        config.get("motion_preservation_teacher_chunk_frames", 0)
    )
    args.motion_preservation_separate_backward = bool(
        config.get("motion_preservation_separate_backward", False)
    )
    args.motion_preservation_fused_defer_step = bool(
        config.get("motion_preservation_fused_defer_step", False)
    )
    args.motion_prior_cache_only = bool(config.get("motion_prior_cache_only", False))
    args.motion_prior_require_temporal = bool(
        config.get("motion_prior_require_temporal", False)
    )

    args.motion_attention_preservation = bool(
        config.get("motion_attention_preservation", False)
    )
    args.motion_attention_preservation_weight = float(
        config.get("motion_attention_preservation_weight", 0.0)
    )
    args.motion_attention_preservation_loss = str(
        config.get("motion_attention_preservation_loss", "kl") or "kl"
    ).lower()
    args.motion_attention_preservation_queries = int(
        config.get("motion_attention_preservation_queries", 32)
    )
    args.motion_attention_preservation_keys = int(
        config.get("motion_attention_preservation_keys", 64)
    )
    args.motion_attention_preservation_per_head = bool(
        config.get("motion_attention_preservation_per_head", False)
    )
    args.motion_attention_preservation_temperature = float(
        config.get("motion_attention_preservation_temperature", 1.0)
    )
    args.motion_attention_preservation_symmetric_kl = bool(
        config.get("motion_attention_preservation_symmetric_kl", False)
    )
    args.motion_attention_preservation_blocks = config.get(
        "motion_attention_preservation_blocks", None
    )

    args.ewc_lambda = float(config.get("ewc_lambda", 0.0))
    args.ewc_num_batches = int(config.get("ewc_num_batches", 8))
    args.ewc_target = str(
        config.get("ewc_target", "attn_norm_bias") or "attn_norm_bias"
    ).lower()
    args.ewc_max_param_tensors = int(config.get("ewc_max_param_tensors", 256))
    args.ewc_cache_path = config.get("ewc_cache_path", None)
    args.ewc_cache_rebuild = bool(config.get("ewc_cache_rebuild", False))

    if args.freeze_early_blocks < 0:
        raise ValueError("freeze_early_blocks must be >= 0")
    if args.non_block_lr_scale < 0.0:
        raise ValueError("non_block_lr_scale must be >= 0")
    if args.attn_geometry_lr_scale < 0.0:
        raise ValueError("attn_geometry_lr_scale must be >= 0")
    if args.motion_preservation_multiplier < 0.0:
        raise ValueError("motion_preservation_multiplier must be >= 0")
    if args.motion_preservation_mode not in {"temporal", "full"}:
        raise ValueError("motion_preservation_mode must be one of: temporal, full")
    if args.motion_preservation_anchor_cache_size < 0:
        raise ValueError("motion_preservation_anchor_cache_size must be >= 0")
    if args.motion_preservation_anchor_cache_auto_ratio <= 0.0:
        raise ValueError("motion_preservation_anchor_cache_auto_ratio must be > 0")
    if args.motion_preservation_anchor_cache_auto_min < 1:
        raise ValueError("motion_preservation_anchor_cache_auto_min must be >= 1")
    if args.motion_preservation_anchor_cache_auto_max < (
        args.motion_preservation_anchor_cache_auto_min
    ):
        raise ValueError(
            "motion_preservation_anchor_cache_auto_max must be >= motion_preservation_anchor_cache_auto_min"
        )
    if args.motion_preservation_anchor_source not in {"dataset", "synthetic", "hybrid"}:
        raise ValueError(
            "motion_preservation_anchor_source must be one of: dataset, synthetic, hybrid"
        )
    if args.motion_preservation_synthetic_frames < 2:
        raise ValueError("motion_preservation_synthetic_frames must be >= 2")
    if not (0.0 <= args.motion_preservation_synthetic_temporal_corr <= 0.999):
        raise ValueError(
            "motion_preservation_synthetic_temporal_corr must be in [0, 0.999]"
        )
    if not (0.0 <= args.motion_preservation_synthetic_dataset_mix <= 1.0):
        raise ValueError(
            "motion_preservation_synthetic_dataset_mix must be in [0, 1]"
        )
    if args.motion_preservation_warmup_steps < 0:
        raise ValueError("motion_preservation_warmup_steps must be >= 0")
    if args.motion_preservation_interval < 1:
        raise ValueError("motion_preservation_interval must be >= 1")
    if args.motion_preservation_health_log_interval < 0:
        raise ValueError("motion_preservation_health_log_interval must be >= 0")
    if args.motion_preservation_probability is not None:
        args.motion_preservation_probability = float(
            args.motion_preservation_probability
        )
        if not (0.0 <= args.motion_preservation_probability <= 1.0):
            raise ValueError("motion_preservation_probability must be in [0, 1]")
    if args.motion_preservation_num_sigmas < 1:
        raise ValueError("motion_preservation_num_sigmas must be >= 1")
    if not (0.0 <= args.motion_preservation_sigma_min <= 1.0):
        raise ValueError("motion_preservation_sigma_min must be in [0, 1]")
    if not (0.0 <= args.motion_preservation_sigma_max <= 1.0):
        raise ValueError("motion_preservation_sigma_max must be in [0, 1]")
    if args.motion_preservation_sigma_max < args.motion_preservation_sigma_min:
        raise ValueError(
            "motion_preservation_sigma_max must be >= motion_preservation_sigma_min"
        )
    if args.motion_preservation_sigma_sampling not in {"uniform", "logsnr"}:
        raise ValueError(
            "motion_preservation_sigma_sampling must be one of: uniform, logsnr"
        )
    if args.motion_preservation_sigma_sampling_power <= 0.0:
        raise ValueError("motion_preservation_sigma_sampling_power must be > 0")
    if args.motion_preservation_second_order_weight < 0.0:
        raise ValueError("motion_preservation_second_order_weight must be >= 0")
    if args.motion_preservation_teacher_chunk_frames < 0:
        raise ValueError("motion_preservation_teacher_chunk_frames must be >= 0")
    if args.motion_attention_preservation_weight < 0.0:
        raise ValueError("motion_attention_preservation_weight must be >= 0")
    if args.motion_attention_preservation_loss not in {"kl", "mse"}:
        raise ValueError(
            "motion_attention_preservation_loss must be one of: kl, mse"
        )
    if args.motion_attention_preservation_queries < 1:
        raise ValueError("motion_attention_preservation_queries must be >= 1")
    if args.motion_attention_preservation_keys < 1:
        raise ValueError("motion_attention_preservation_keys must be >= 1")
    if args.motion_attention_preservation_temperature <= 0.0:
        raise ValueError("motion_attention_preservation_temperature must be > 0")
    if args.ewc_lambda < 0.0:
        raise ValueError("ewc_lambda must be >= 0")
    if args.ewc_num_batches < 0:
        raise ValueError("ewc_num_batches must be >= 0")
    if args.ewc_target not in {"attn_norm_bias", "attn_geometry", "all_trainable"}:
        raise ValueError(
            "ewc_target must be one of: attn_norm_bias, attn_geometry, all_trainable"
        )
    if args.ewc_max_param_tensors < 0:
        raise ValueError("ewc_max_param_tensors must be >= 0")

    if args.freeze_block_indices is not None and not isinstance(
        args.freeze_block_indices, (str, list, tuple)
    ):
        raise ValueError(
            "freeze_block_indices must be a string or list of strings when set"
        )
    if isinstance(args.freeze_block_indices, (list, tuple)):
        normalized_freeze_specs = []
        for entry in args.freeze_block_indices:
            if not isinstance(entry, str):
                raise ValueError(
                    "freeze_block_indices list entries must be strings, "
                    f"got {type(entry)}"
                )
            entry = entry.strip()
            if entry:
                normalized_freeze_specs.append(entry)
        args.freeze_block_indices = normalized_freeze_specs
    elif isinstance(args.freeze_block_indices, str):
        args.freeze_block_indices = args.freeze_block_indices.strip() or None

    if args.block_lr_scales is not None and not isinstance(
        args.block_lr_scales, (str, list, tuple)
    ):
        raise ValueError("block_lr_scales must be a string or list of strings when set")
    if isinstance(args.block_lr_scales, (list, tuple)):
        normalized_block_lr_scales = []
        for entry in args.block_lr_scales:
            if not isinstance(entry, str):
                raise ValueError(
                    "block_lr_scales list entries must be strings, "
                    f"got {type(entry)}"
                )
            entry = entry.strip()
            if entry:
                normalized_block_lr_scales.append(entry)
        args.block_lr_scales = normalized_block_lr_scales
    elif isinstance(args.block_lr_scales, str):
        args.block_lr_scales = args.block_lr_scales.strip() or None

    if args.motion_preservation_sigma_values is not None and not isinstance(
        args.motion_preservation_sigma_values, (str, list, tuple)
    ):
        raise ValueError(
            "motion_preservation_sigma_values must be a string or list when set"
        )
    if isinstance(args.motion_preservation_sigma_values, (list, tuple)):
        normalized_sigma_values = []
        for entry in args.motion_preservation_sigma_values:
            if isinstance(entry, (int, float)):
                normalized_sigma_values.append(float(entry))
                continue
            if not isinstance(entry, str):
                raise ValueError(
                    "motion_preservation_sigma_values list entries must be numbers or strings, "
                    f"got {type(entry)}"
                )
            entry = entry.strip()
            if entry:
                normalized_sigma_values.append(float(entry))
        args.motion_preservation_sigma_values = normalized_sigma_values or None
    elif isinstance(args.motion_preservation_sigma_values, str):
        sigma_tokens = [
            token.strip()
            for token in args.motion_preservation_sigma_values.replace(";", ",").split(",")
            if token.strip()
        ]
        args.motion_preservation_sigma_values = (
            [float(token) for token in sigma_tokens] if sigma_tokens else None
        )
    if args.motion_preservation_sigma_values is not None:
        for sigma_value in args.motion_preservation_sigma_values:
            if sigma_value < 0.0 or sigma_value > 1.0:
                raise ValueError(
                    "motion_preservation_sigma_values entries must be in [0, 1]"
                )

    if args.motion_attention_preservation_blocks is not None and not isinstance(
        args.motion_attention_preservation_blocks, (str, list, tuple)
    ):
        raise ValueError(
            "motion_attention_preservation_blocks must be a string or list of strings when set"
        )
    if isinstance(args.motion_attention_preservation_blocks, (list, tuple)):
        normalized_attention_blocks = []
        for entry in args.motion_attention_preservation_blocks:
            if not isinstance(entry, str):
                raise ValueError(
                    "motion_attention_preservation_blocks list entries must be strings, "
                    f"got {type(entry)}"
                )
            entry = entry.strip()
            if entry:
                normalized_attention_blocks.append(entry)
        args.motion_attention_preservation_blocks = (
            ",".join(normalized_attention_blocks) if normalized_attention_blocks else None
        )
    elif isinstance(args.motion_attention_preservation_blocks, str):
        args.motion_attention_preservation_blocks = (
            args.motion_attention_preservation_blocks.strip() or None
        )

    if args.motion_preservation_anchor_cache_path:
        args.motion_preservation_anchor_cache_path = os.path.abspath(
            str(args.motion_preservation_anchor_cache_path)
        )
    if args.ewc_cache_path:
        args.ewc_cache_path = os.path.abspath(str(args.ewc_cache_path))

    if (
        args.motion_preservation_anchor_cache_rebuild
        and not args.motion_preservation_anchor_cache_path
    ):
        logger.warning(
            "motion_preservation_anchor_cache_rebuild is set but motion_preservation_anchor_cache_path is empty; ignoring rebuild flag."
        )
        args.motion_preservation_anchor_cache_rebuild = False
    if args.ewc_cache_rebuild and not args.ewc_cache_path:
        logger.warning(
            "ewc_cache_rebuild is set but ewc_cache_path is empty; ignoring rebuild flag."
        )
        args.ewc_cache_rebuild = False
