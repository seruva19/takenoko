from __future__ import annotations

import os
from typing import Any, Dict, List


_ALLOWED_TEACHER_TYPES = {"sam2", "sam3"}
_ALLOWED_TEACHER_DTYPES = {"float16", "bfloat16", "float32"}
_ALLOWED_LOSS_MODES = {"kl", "l2"}
_ALLOWED_FUSION_MODES = {"lgf", "feature"}
_ALLOWED_CACHE_MODES = {"off", "read", "write", "read_write"}
_ALLOWED_PROMPT_SOURCES = {"caption", "item_key", "caption_or_item_key"}
_ALLOWED_TEACHER_BACKENDS = {"vision_encoder", "tracker_memory"}


def _parse_alignment_depths(config: Dict[str, Any]) -> List[int]:
    default_depth = int(config.get("sft_alignment_depth", 25))
    raw_depths = config.get("sft_alignment_depths", None)
    if raw_depths is None:
        return [default_depth]
    if not isinstance(raw_depths, (list, tuple)):
        raise ValueError(
            "sft_alignment_depths must be a list of ints or omitted, got "
            f"{type(raw_depths).__name__}"
        )
    if len(raw_depths) == 0:
        raise ValueError("sft_alignment_depths must not be empty when provided")
    parsed = [int(v) for v in raw_depths]
    return list(dict.fromkeys(parsed))


def _parse_projector_dims(config: Dict[str, Any]) -> List[int]:
    raw_dims = config.get("sft_projector_dims", [512, 256, 256])
    if not isinstance(raw_dims, (list, tuple)):
        raise ValueError(
            "sft_projector_dims must be a list of positive integers, got "
            f"{type(raw_dims).__name__}"
        )
    if len(raw_dims) != 3:
        raise ValueError(
            "sft_projector_dims must contain exactly 3 integers (e.g. [512, 256, 256])."
        )
    dims = [int(v) for v in raw_dims]
    for dim in dims:
        if dim <= 0:
            raise ValueError(f"sft_projector_dims entries must be > 0, got {dim}")
    return dims


def apply_structure_from_tracking_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse Structure-From-Tracking config and validate compatibility."""
    args.enable_structure_from_tracking = bool(
        config.get("enable_structure_from_tracking", False)
    )
    args.sft_teacher_type = str(config.get("sft_teacher_type", "sam2")).lower()
    args.sft_teacher_model_id = str(
        config.get("sft_teacher_model_id", "facebook/sam2-hiera-large")
    )
    args.sft_teacher_checkpoint = str(config.get("sft_teacher_checkpoint", "") or "")
    args.sft_teacher_image_size = int(config.get("sft_teacher_image_size", 512))
    args.sft_teacher_chunk_size = int(config.get("sft_teacher_chunk_size", 8))
    args.sft_teacher_max_frames = int(config.get("sft_teacher_max_frames", 0))
    args.sft_teacher_dtype = str(config.get("sft_teacher_dtype", "float16")).lower()
    args.sft_paper_strict_mode = bool(config.get("sft_paper_strict_mode", False))
    args.sft_teacher_backend = str(
        config.get("sft_teacher_backend", "vision_encoder")
    ).lower()
    args.sft_tracker_memory_sam2_config = str(
        config.get("sft_tracker_memory_sam2_config", "sam2_hiera_l.yaml")
    )
    args.sft_tracker_memory_sam2_checkpoint = str(
        config.get(
            "sft_tracker_memory_sam2_checkpoint",
            "checkpoints/sam2_hiera_large.pt",
        )
    )
    args.sft_tracker_memory_download_checkpoint = bool(
        config.get("sft_tracker_memory_download_checkpoint", False)
    )
    args.sft_tracker_memory_strict = bool(
        config.get("sft_tracker_memory_strict", False)
    )
    args.sft_use_causal_memory = bool(config.get("sft_use_causal_memory", True))
    args.sft_teacher_memory_decay = float(config.get("sft_teacher_memory_decay", 0.9))
    args.sft_use_mask_prompting = bool(config.get("sft_use_mask_prompting", False))
    args.sft_mask_prompt_strength = float(config.get("sft_mask_prompt_strength", 0.85))
    args.sft_mask_prompt_blur_kernel = int(config.get("sft_mask_prompt_blur_kernel", 5))
    args.sft_use_groundingdino_prompts = bool(
        config.get("sft_use_groundingdino_prompts", False)
    )
    args.sft_groundingdino_model_id = str(
        config.get("sft_groundingdino_model_id", "IDEA-Research/grounding-dino-base")
    )
    args.sft_groundingdino_box_threshold = float(
        config.get("sft_groundingdino_box_threshold", 0.35)
    )
    args.sft_groundingdino_text_threshold = float(
        config.get("sft_groundingdino_text_threshold", 0.25)
    )
    args.sft_groundingdino_prompt_source = str(
        config.get("sft_groundingdino_prompt_source", "caption_or_item_key")
    ).lower()
    args.sft_groundingdino_apply_to_last_frame = bool(
        config.get("sft_groundingdino_apply_to_last_frame", True)
    )
    args.sft_groundingdino_use_sam2_refine = bool(
        config.get("sft_groundingdino_use_sam2_refine", False)
    )
    args.sft_groundingdino_sam2_config = str(
        config.get("sft_groundingdino_sam2_config", "sam2_hiera_l.yaml")
    )
    args.sft_groundingdino_sam2_checkpoint = str(
        config.get("sft_groundingdino_sam2_checkpoint", "checkpoints/sam2_hiera_large.pt")
    )
    args.sft_groundingdino_download_sam2_checkpoint = bool(
        config.get("sft_groundingdino_download_sam2_checkpoint", False)
    )
    args.sft_teacher_cache_mode = str(config.get("sft_teacher_cache_mode", "off")).lower()
    args.sft_teacher_cache_dir = str(config.get("sft_teacher_cache_dir", "") or "")
    args.sft_teacher_cache_num_workers = int(
        config.get("sft_teacher_cache_num_workers", 4)
    )
    args.sft_teacher_cache_batch_size = int(
        config.get("sft_teacher_cache_batch_size", 1)
    )
    args.sft_teacher_cache_skip_existing = bool(
        config.get("sft_teacher_cache_skip_existing", True)
    )
    args.sft_teacher_cache_purge = bool(config.get("sft_teacher_cache_purge", False))
    args.sft_teacher_cache_frame_start_stride = int(
        config.get("sft_teacher_cache_frame_start_stride", 1)
    )
    args.sft_teacher_cache_before_training = bool(
        config.get("sft_teacher_cache_before_training", False)
    )
    args.sft_teacher_cache_overwrite_existing = bool(
        config.get("sft_teacher_cache_overwrite_existing", False)
    )
    args.sft_alignment_depths = _parse_alignment_depths(config)
    args.sft_alignment_depth = int(args.sft_alignment_depths[0])
    args.sft_loss_lambda = float(config.get("sft_loss_lambda", 0.5))
    args.sft_loss_mode = str(config.get("sft_loss_mode", "kl")).lower()
    args.sft_fusion_mode = str(config.get("sft_fusion_mode", "lgf")).lower()
    args.sft_lgf_kernel_size = int(config.get("sft_lgf_kernel_size", 7))
    args.sft_lgf_temperature = float(config.get("sft_lgf_temperature", 0.1))
    args.sft_teacher_fusion_weight = float(config.get("sft_teacher_fusion_weight", 0.5))
    args.sft_temporal_interp_factor = int(config.get("sft_temporal_interp_factor", 4))
    args.sft_temporal_kernel_size = int(config.get("sft_temporal_kernel_size", 3))
    args.sft_enable_backward_teacher = bool(config.get("sft_enable_backward_teacher", True))
    args.sft_detach_teacher = bool(config.get("sft_detach_teacher", True))
    args.sft_projector_hidden_dim = int(config.get("sft_projector_hidden_dim", 2048))
    args.sft_projector_group_norm = bool(config.get("sft_projector_group_norm", True))
    args.sft_projector_gn_groups = int(config.get("sft_projector_gn_groups", 32))
    args.sft_projector_dims = _parse_projector_dims(config)
    args.sft_max_spatial_tokens = int(config.get("sft_max_spatial_tokens", -1))
    args.sft_spatial_align = bool(config.get("sft_spatial_align", True))
    args.sft_temporal_align = bool(config.get("sft_temporal_align", True))

    if args.sft_teacher_type not in _ALLOWED_TEACHER_TYPES:
        raise ValueError(
            "sft_teacher_type must be one of "
            f"{sorted(_ALLOWED_TEACHER_TYPES)}, got {args.sft_teacher_type!r}"
        )
    if args.sft_teacher_dtype not in _ALLOWED_TEACHER_DTYPES:
        raise ValueError(
            "sft_teacher_dtype must be one of "
            f"{sorted(_ALLOWED_TEACHER_DTYPES)}, got {args.sft_teacher_dtype!r}"
        )
    if args.sft_teacher_backend not in _ALLOWED_TEACHER_BACKENDS:
        raise ValueError(
            "sft_teacher_backend must be one of "
            f"{sorted(_ALLOWED_TEACHER_BACKENDS)}, got {args.sft_teacher_backend!r}"
        )
    if args.sft_loss_mode not in _ALLOWED_LOSS_MODES:
        raise ValueError(
            "sft_loss_mode must be one of "
            f"{sorted(_ALLOWED_LOSS_MODES)}, got {args.sft_loss_mode!r}"
        )
    if args.sft_fusion_mode not in _ALLOWED_FUSION_MODES:
        raise ValueError(
            "sft_fusion_mode must be one of "
            f"{sorted(_ALLOWED_FUSION_MODES)}, got {args.sft_fusion_mode!r}"
        )
    if args.sft_teacher_cache_mode not in _ALLOWED_CACHE_MODES:
        raise ValueError(
            "sft_teacher_cache_mode must be one of "
            f"{sorted(_ALLOWED_CACHE_MODES)}, got {args.sft_teacher_cache_mode!r}"
        )
    if args.sft_groundingdino_prompt_source not in _ALLOWED_PROMPT_SOURCES:
        raise ValueError(
            "sft_groundingdino_prompt_source must be one of "
            f"{sorted(_ALLOWED_PROMPT_SOURCES)}, got {args.sft_groundingdino_prompt_source!r}"
        )
    if args.sft_teacher_checkpoint and not os.path.exists(args.sft_teacher_checkpoint):
        raise ValueError(
            "sft_teacher_checkpoint does not exist: "
            f"{args.sft_teacher_checkpoint}"
        )
    if not args.sft_teacher_checkpoint and args.sft_teacher_model_id.strip() == "":
        raise ValueError(
            "Either sft_teacher_checkpoint or sft_teacher_model_id must be set."
        )
    if args.sft_teacher_image_size <= 0:
        raise ValueError(
            f"sft_teacher_image_size must be > 0, got {args.sft_teacher_image_size}"
        )
    if args.sft_teacher_chunk_size <= 0:
        raise ValueError(
            f"sft_teacher_chunk_size must be > 0, got {args.sft_teacher_chunk_size}"
        )
    if args.sft_teacher_max_frames < 0:
        raise ValueError(
            f"sft_teacher_max_frames must be >= 0, got {args.sft_teacher_max_frames}"
        )
    if not 0.0 <= args.sft_teacher_memory_decay < 1.0:
        raise ValueError(
            "sft_teacher_memory_decay must be in [0, 1), got "
            f"{args.sft_teacher_memory_decay}"
        )
    if not 0.0 <= args.sft_mask_prompt_strength <= 1.0:
        raise ValueError(
            "sft_mask_prompt_strength must be in [0, 1], got "
            f"{args.sft_mask_prompt_strength}"
        )
    if args.sft_mask_prompt_blur_kernel <= 0 or args.sft_mask_prompt_blur_kernel % 2 == 0:
        raise ValueError(
            "sft_mask_prompt_blur_kernel must be an odd integer > 0, got "
            f"{args.sft_mask_prompt_blur_kernel}"
        )
    if args.sft_teacher_cache_mode != "off" and args.sft_teacher_cache_dir.strip() == "":
        raise ValueError(
            "sft_teacher_cache_dir must be non-empty when sft_teacher_cache_mode is not 'off'"
        )
    if args.sft_teacher_cache_mode == "read" and not os.path.isdir(args.sft_teacher_cache_dir):
        raise ValueError(
            "sft_teacher_cache_mode='read' requires an existing directory: "
            f"{args.sft_teacher_cache_dir}"
        )
    if args.sft_teacher_cache_before_training and not args.enable_structure_from_tracking:
        raise ValueError(
            "sft_teacher_cache_before_training requires enable_structure_from_tracking=true"
        )
    if (
        args.sft_teacher_cache_before_training
        and args.sft_teacher_cache_dir.strip() == ""
    ):
        raise ValueError(
            "sft_teacher_cache_dir must be non-empty when sft_teacher_cache_before_training is true"
        )
    if args.sft_teacher_cache_num_workers <= 0:
        raise ValueError(
            "sft_teacher_cache_num_workers must be > 0, got "
            f"{args.sft_teacher_cache_num_workers}"
        )
    if args.sft_teacher_cache_batch_size <= 0:
        raise ValueError(
            "sft_teacher_cache_batch_size must be > 0, got "
            f"{args.sft_teacher_cache_batch_size}"
        )
    if args.sft_teacher_cache_frame_start_stride <= 0:
        raise ValueError(
            "sft_teacher_cache_frame_start_stride must be > 0, got "
            f"{args.sft_teacher_cache_frame_start_stride}"
        )
    if not 0.0 <= args.sft_groundingdino_box_threshold <= 1.0:
        raise ValueError(
            "sft_groundingdino_box_threshold must be in [0, 1], got "
            f"{args.sft_groundingdino_box_threshold}"
        )
    if not 0.0 <= args.sft_groundingdino_text_threshold <= 1.0:
        raise ValueError(
            "sft_groundingdino_text_threshold must be in [0, 1], got "
            f"{args.sft_groundingdino_text_threshold}"
        )
    if args.sft_use_groundingdino_prompts and args.sft_groundingdino_model_id.strip() == "":
        raise ValueError(
            "sft_groundingdino_model_id must be non-empty when sft_use_groundingdino_prompts is true"
        )
    if args.sft_groundingdino_use_sam2_refine and args.sft_groundingdino_sam2_config.strip() == "":
        raise ValueError(
            "sft_groundingdino_sam2_config must be non-empty when sft_groundingdino_use_sam2_refine is true"
        )
    if args.sft_groundingdino_use_sam2_refine and args.sft_groundingdino_sam2_checkpoint.strip() == "":
        raise ValueError(
            "sft_groundingdino_sam2_checkpoint must be non-empty when sft_groundingdino_use_sam2_refine is true"
        )
    if args.sft_teacher_backend == "tracker_memory":
        if args.sft_teacher_type != "sam2":
            raise ValueError(
                "sft_teacher_backend='tracker_memory' is currently supported only when sft_teacher_type='sam2'"
            )
        if args.sft_tracker_memory_sam2_config.strip() == "":
            raise ValueError(
                "sft_tracker_memory_sam2_config must be non-empty when sft_teacher_backend='tracker_memory'"
            )
        if args.sft_tracker_memory_sam2_checkpoint.strip() == "":
            raise ValueError(
                "sft_tracker_memory_sam2_checkpoint must be non-empty when sft_teacher_backend='tracker_memory'"
            )
        if (
            not args.sft_tracker_memory_download_checkpoint
            and not os.path.exists(args.sft_tracker_memory_sam2_checkpoint)
        ):
            raise ValueError(
                "sft_tracker_memory_sam2_checkpoint does not exist and "
                "sft_tracker_memory_download_checkpoint is false: "
                f"{args.sft_tracker_memory_sam2_checkpoint}"
            )
    if args.sft_paper_strict_mode:
        if not args.enable_structure_from_tracking:
            raise ValueError(
                "sft_paper_strict_mode requires enable_structure_from_tracking=true"
            )
        if args.sft_teacher_backend != "tracker_memory":
            raise ValueError(
                "sft_paper_strict_mode requires sft_teacher_backend='tracker_memory'"
            )
        if args.sft_teacher_type != "sam2":
            raise ValueError(
                "sft_paper_strict_mode currently requires sft_teacher_type='sam2'"
            )
        if not args.sft_use_mask_prompting:
            raise ValueError(
                "sft_paper_strict_mode requires sft_use_mask_prompting=true"
            )
        if not args.sft_enable_backward_teacher:
            raise ValueError(
                "sft_paper_strict_mode requires sft_enable_backward_teacher=true"
            )
        if args.sft_loss_mode != "kl":
            raise ValueError(
                "sft_paper_strict_mode requires sft_loss_mode='kl'"
            )
        if args.sft_fusion_mode != "lgf":
            raise ValueError(
                "sft_paper_strict_mode requires sft_fusion_mode='lgf'"
            )
        if not args.sft_tracker_memory_strict:
            args.sft_tracker_memory_strict = True
            logger.info(
                "Structure-From-Tracking: forcing sft_tracker_memory_strict=true because sft_paper_strict_mode is enabled."
            )
    for depth in args.sft_alignment_depths:
        if depth < 0:
            raise ValueError(f"sft_alignment_depths entries must be >= 0, got {depth}")
    if args.sft_loss_lambda < 0:
        raise ValueError(f"sft_loss_lambda must be >= 0, got {args.sft_loss_lambda}")
    if args.enable_structure_from_tracking and args.sft_loss_lambda <= 0:
        raise ValueError(
            "sft_loss_lambda must be > 0 when enable_structure_from_tracking is true"
        )
    if args.sft_lgf_kernel_size < 3 or args.sft_lgf_kernel_size % 2 == 0:
        raise ValueError(
            "sft_lgf_kernel_size must be an odd integer >= 3, got "
            f"{args.sft_lgf_kernel_size}"
        )
    if args.sft_lgf_temperature <= 0:
        raise ValueError(
            f"sft_lgf_temperature must be > 0, got {args.sft_lgf_temperature}"
        )
    if not 0.0 <= args.sft_teacher_fusion_weight <= 1.0:
        raise ValueError(
            "sft_teacher_fusion_weight must be in [0, 1], got "
            f"{args.sft_teacher_fusion_weight}"
        )
    if args.sft_temporal_interp_factor < 1:
        raise ValueError(
            "sft_temporal_interp_factor must be >= 1, got "
            f"{args.sft_temporal_interp_factor}"
        )
    if args.sft_temporal_kernel_size <= 0 or args.sft_temporal_kernel_size % 2 == 0:
        raise ValueError(
            "sft_temporal_kernel_size must be an odd integer > 0, got "
            f"{args.sft_temporal_kernel_size}"
        )
    if args.sft_projector_hidden_dim <= 0:
        raise ValueError(
            "sft_projector_hidden_dim must be > 0, got "
            f"{args.sft_projector_hidden_dim}"
        )
    if args.sft_projector_gn_groups <= 0:
        raise ValueError(
            "sft_projector_gn_groups must be > 0, got "
            f"{args.sft_projector_gn_groups}"
        )
    if args.sft_max_spatial_tokens == 0 or args.sft_max_spatial_tokens < -1:
        raise ValueError(
            "sft_max_spatial_tokens must be -1 (disabled) or > 0, got "
            f"{args.sft_max_spatial_tokens}"
        )

    if args.enable_structure_from_tracking:
        concurrent_alignment_flags = [
            flag
            for flag in (
                "enable_repa",
                "enable_irepa",
                "enable_videorepa",
                "sara_enabled",
                "enable_moalign",
                "crepa_enabled",
            )
            if bool(config.get(flag, False))
        ]
        if concurrent_alignment_flags:
            logger.info(
                "Structure-From-Tracking enabled with additional alignment helpers: %s",
                ", ".join(concurrent_alignment_flags),
            )

        teacher_source = (
            args.sft_teacher_checkpoint
            if args.sft_teacher_checkpoint
            else args.sft_teacher_model_id
        )
        logger.info(
            "Structure-From-Tracking enabled (teacher=%s, source=%s, depths=%s, "
            "lambda=%.4f, loss=%s, fusion=%s, kernel=%d, temp=%.3f, backward=%s, fusion_w=%.3f, "
            "causal_memory=%s, paper_strict=%s, backend=%s, tracker_strict=%s, mask_prompt=%s, dino_prompt=%s, dino_sam2_refine=%s, "
            "cache_mode=%s, cache_stride=%d, cache_before_train=%s, cache_overwrite=%s).",
            args.sft_teacher_type,
            teacher_source,
            args.sft_alignment_depths,
            args.sft_loss_lambda,
            args.sft_loss_mode,
            args.sft_fusion_mode,
            args.sft_lgf_kernel_size,
            args.sft_lgf_temperature,
            args.sft_enable_backward_teacher,
            args.sft_teacher_fusion_weight,
            args.sft_use_causal_memory,
            args.sft_paper_strict_mode,
            args.sft_teacher_backend,
            args.sft_tracker_memory_strict,
            args.sft_use_mask_prompting,
            args.sft_use_groundingdino_prompts,
            args.sft_groundingdino_use_sam2_refine,
            args.sft_teacher_cache_mode,
            args.sft_teacher_cache_frame_start_stride,
            args.sft_teacher_cache_before_training,
            args.sft_teacher_cache_overwrite_existing,
        )
