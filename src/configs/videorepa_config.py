from __future__ import annotations

from typing import Any, Dict, List


_VIDEOREPA_ALLOWED_MODES = {
    "cosine_similarity",
    "token_relation_distillation",
    "token_relation_distillation_only_spatial",
    "token_relation_distillation_only_temporal",
}


def _parse_alignment_depths(config: Dict[str, Any]) -> List[int]:
    default_depth = int(config.get("videorepa_alignment_depth", 18))
    raw_depths = config.get("videorepa_alignment_depths", None)
    if raw_depths is None:
        return [default_depth]
    if not isinstance(raw_depths, (list, tuple)):
        raise ValueError(
            "videorepa_alignment_depths must be a list of ints or omitted, got "
            f"{type(raw_depths).__name__}"
        )
    if len(raw_depths) == 0:
        raise ValueError("videorepa_alignment_depths must not be empty when provided")
    parsed = [int(v) for v in raw_depths]
    return list(dict.fromkeys(parsed))


def apply_videorepa_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse VideoREPA configuration and validate compatibility."""
    args.enable_videorepa = bool(config.get("enable_videorepa", False))
    args.videorepa_encoder_name = str(
        config.get("videorepa_encoder_name", "dinov2-vit-b")
    )
    args.videorepa_input_resolution = int(config.get("videorepa_input_resolution", 256))
    args.videorepa_alignment_depths = _parse_alignment_depths(config)
    args.videorepa_alignment_depth = int(args.videorepa_alignment_depths[0])
    args.videorepa_loss_lambda = float(config.get("videorepa_loss_lambda", 0.5))
    args.videorepa_relation_mode = str(
        config.get("videorepa_relation_mode", "token_relation_distillation")
    ).lower()
    args.videorepa_margin = float(config.get("videorepa_margin", 0.1))
    raw_margin_matrix = config.get("videorepa_margin_matrix", None)
    if raw_margin_matrix is None:
        args.videorepa_margin_matrix = float(args.videorepa_margin)
    else:
        args.videorepa_margin_matrix = float(raw_margin_matrix)
    args.videorepa_projector_hidden_dim = int(
        config.get("videorepa_projector_hidden_dim", 2048)
    )
    args.videorepa_align_dim = int(config.get("videorepa_align_dim", 768))
    args.videorepa_max_spatial_tokens = int(
        config.get("videorepa_max_spatial_tokens", -1)
    )
    args.videorepa_spatial_align = bool(config.get("videorepa_spatial_align", True))
    args.videorepa_temporal_align = bool(config.get("videorepa_temporal_align", True))
    args.videorepa_temporal_exclude_same_frame = bool(
        config.get("videorepa_temporal_exclude_same_frame", True)
    )
    args.videorepa_detach_teacher = bool(config.get("videorepa_detach_teacher", True))
    args.videorepa_encoder_chunk_size = int(
        config.get("videorepa_encoder_chunk_size", 0)
    )
    args.videorepa_teacher_checkpoint = str(
        config.get("videorepa_teacher_checkpoint", "") or ""
    )
    args.videorepa_video_teacher_frames = int(
        config.get("videorepa_video_teacher_frames", 16)
    )
    args.videorepa_video_teacher_tubelet_size = int(
        config.get("videorepa_video_teacher_tubelet_size", 2)
    )
    args.videorepa_video_teacher_patch_size = int(
        config.get("videorepa_video_teacher_patch_size", 16)
    )
    args.videorepa_video_teacher_image_size = int(
        config.get("videorepa_video_teacher_image_size", 224)
    )
    args.videorepa_video_teacher_align_resolution = config.get(
        "videorepa_video_teacher_align_resolution",
        [480, 720],
    )
    args.videorepa_video_teacher_drop_first_frame = bool(
        config.get("videorepa_video_teacher_drop_first_frame", True)
    )
    args.videorepa_vjepa_checkpoint_key = str(
        config.get("videorepa_vjepa_checkpoint_key", "target_encoder")
    )
    args.videorepa_vjepa_uniform_power = bool(
        config.get("videorepa_vjepa_uniform_power", True)
    )
    args.videorepa_vjepa_use_sdpa = bool(config.get("videorepa_vjepa_use_sdpa", True))
    args.videorepa_vjepa_use_silu = bool(config.get("videorepa_vjepa_use_silu", False))
    args.videorepa_vjepa_tight_silu = bool(
        config.get("videorepa_vjepa_tight_silu", False)
    )

    if args.videorepa_relation_mode not in _VIDEOREPA_ALLOWED_MODES:
        raise ValueError(
            "videorepa_relation_mode must be one of "
            f"{sorted(_VIDEOREPA_ALLOWED_MODES)}, got "
            f"{args.videorepa_relation_mode!r}"
        )
    if args.videorepa_input_resolution not in (256, 512):
        raise ValueError(
            "videorepa_input_resolution must be 256 or 512, got "
            f"{args.videorepa_input_resolution}"
        )
    if args.videorepa_loss_lambda < 0:
        raise ValueError(
            f"videorepa_loss_lambda must be >= 0, got {args.videorepa_loss_lambda}"
        )
    if args.enable_videorepa and args.videorepa_loss_lambda <= 0:
        raise ValueError(
            "videorepa_loss_lambda must be > 0 when enable_videorepa is true"
        )
    if args.videorepa_margin < 0:
        raise ValueError(f"videorepa_margin must be >= 0, got {args.videorepa_margin}")
    if args.videorepa_margin_matrix < 0:
        raise ValueError(
            "videorepa_margin_matrix must be >= 0, got "
            f"{args.videorepa_margin_matrix}"
        )
    if args.videorepa_projector_hidden_dim <= 0:
        raise ValueError(
            "videorepa_projector_hidden_dim must be > 0, got "
            f"{args.videorepa_projector_hidden_dim}"
        )
    if args.videorepa_align_dim <= 0:
        raise ValueError(
            f"videorepa_align_dim must be > 0, got {args.videorepa_align_dim}"
        )
    if args.videorepa_max_spatial_tokens == 0 or args.videorepa_max_spatial_tokens < -1:
        raise ValueError(
            "videorepa_max_spatial_tokens must be -1 (disabled) or > 0, got "
            f"{args.videorepa_max_spatial_tokens}"
        )
    if args.videorepa_encoder_chunk_size < 0:
        raise ValueError(
            "videorepa_encoder_chunk_size must be >= 0, got "
            f"{args.videorepa_encoder_chunk_size}"
        )
    if args.videorepa_video_teacher_frames <= 0:
        raise ValueError(
            "videorepa_video_teacher_frames must be > 0, got "
            f"{args.videorepa_video_teacher_frames}"
        )
    if args.videorepa_video_teacher_tubelet_size <= 0:
        raise ValueError(
            "videorepa_video_teacher_tubelet_size must be > 0, got "
            f"{args.videorepa_video_teacher_tubelet_size}"
        )
    if args.videorepa_video_teacher_patch_size <= 0:
        raise ValueError(
            "videorepa_video_teacher_patch_size must be > 0, got "
            f"{args.videorepa_video_teacher_patch_size}"
        )
    if args.videorepa_video_teacher_image_size <= 0:
        raise ValueError(
            "videorepa_video_teacher_image_size must be > 0, got "
            f"{args.videorepa_video_teacher_image_size}"
        )
    if (
        not isinstance(args.videorepa_video_teacher_align_resolution, (list, tuple))
        or len(args.videorepa_video_teacher_align_resolution) != 2
    ):
        raise ValueError(
            "videorepa_video_teacher_align_resolution must be [height, width]"
        )
    align_h = int(args.videorepa_video_teacher_align_resolution[0])
    align_w = int(args.videorepa_video_teacher_align_resolution[1])
    if align_h <= 0 or align_w <= 0:
        raise ValueError(
            "videorepa_video_teacher_align_resolution entries must be > 0, got "
            f"{args.videorepa_video_teacher_align_resolution}"
        )
    args.videorepa_video_teacher_align_resolution = [align_h, align_w]
    for depth in args.videorepa_alignment_depths:
        if depth < 0:
            raise ValueError(
                f"videorepa_alignment_depths entries must be >= 0, got {depth}"
            )

    if args.enable_videorepa and bool(config.get("enable_repa", False)):
        raise ValueError("enable_videorepa and enable_repa are mutually exclusive")
    if args.enable_videorepa and bool(config.get("enable_irepa", False)):
        raise ValueError("enable_videorepa and enable_irepa are mutually exclusive")
    if args.enable_videorepa and bool(config.get("sara_enabled", False)):
        raise ValueError("enable_videorepa and sara_enabled are mutually exclusive")
    if args.enable_videorepa and bool(config.get("enable_moalign", False)):
        raise ValueError("enable_videorepa and enable_moalign are mutually exclusive")
    if args.enable_videorepa and bool(config.get("crepa_enabled", False)):
        raise ValueError("enable_videorepa and crepa_enabled are mutually exclusive")

    encoder_name = str(args.videorepa_encoder_name).lower().strip()
    uses_native_teacher = (
        encoder_name.startswith("videomaev2-")
        or encoder_name.startswith("vjepa-")
        or encoder_name.startswith("vjepa2-")
    )
    if args.enable_videorepa and uses_native_teacher:
        if (
            not encoder_name.startswith("vjepa2-")
            and args.videorepa_teacher_checkpoint == ""
        ):
            raise ValueError(
                "videorepa_teacher_checkpoint must be set when using native "
                "VideoREPA teachers (videomaev2-* or vjepa-*)."
            )
    if (
        args.enable_videorepa
        and encoder_name.startswith("vjepa2-")
        and args.videorepa_teacher_checkpoint
    ):
        logger.info(
            "VideoREPA teacher checkpoint is set but encoder '%s' uses VJEPA2 torch.hub loading; checkpoint will be ignored.",
            args.videorepa_encoder_name,
        )
    if args.enable_videorepa and not uses_native_teacher and args.videorepa_teacher_checkpoint:
        logger.info(
            "VideoREPA teacher checkpoint is set but encoder '%s' uses REPA encoder manager; checkpoint will be ignored.",
            args.videorepa_encoder_name,
        )

    if args.enable_videorepa:
        logger.info(
            "VideoREPA enabled (encoder=%s, depths=%s, mode=%s, lambda=%.4f, margin=%.4f, margin_matrix=%.4f, align_dim=%d, native_teacher=%s)",
            args.videorepa_encoder_name,
            args.videorepa_alignment_depths,
            args.videorepa_relation_mode,
            args.videorepa_loss_lambda,
            args.videorepa_margin,
            args.videorepa_margin_matrix,
            args.videorepa_align_dim,
            uses_native_teacher,
        )
