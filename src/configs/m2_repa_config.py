from __future__ import annotations

from typing import Any, Dict, List


def _parse_alignment_depths(config: Dict[str, Any]) -> List[int]:
    default_depth = int(config.get("m2_repa_alignment_depth", 8))
    raw_depths = config.get("m2_repa_alignment_depths", None)
    if raw_depths is None:
        return [default_depth]
    if not isinstance(raw_depths, (list, tuple)):
        raise ValueError(
            "m2_repa_alignment_depths must be a list of ints or omitted, got "
            f"{type(raw_depths).__name__}"
        )
    if len(raw_depths) == 0:
        raise ValueError("m2_repa_alignment_depths must not be empty when provided")
    parsed = [int(v) for v in raw_depths]
    return list(dict.fromkeys(parsed))


def _parse_encoder_names(raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        names = [part.strip() for part in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        names = [str(part).strip() for part in raw_value]
    else:
        raise ValueError(
            "m2_repa_encoder_names must be a comma-separated string or list of strings"
        )
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError("m2_repa_encoder_names must not contain duplicate entries")
    return names


def apply_m2_repa_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse M2-REPA configuration and validate train-time compatibility."""
    args.enable_m2_repa = bool(config.get("enable_m2_repa", False))
    raw_encoder_names = config.get(
        "m2_repa_encoder_names",
        "dinov2-vit-b,dinov2-vit-l",
    )
    args.m2_repa_encoder_name_list = _parse_encoder_names(raw_encoder_names)
    args.m2_repa_encoder_names = ",".join(args.m2_repa_encoder_name_list)
    args.m2_repa_input_resolution = int(config.get("m2_repa_input_resolution", 256))
    args.m2_repa_alignment_depths = _parse_alignment_depths(config)
    args.m2_repa_alignment_depth = int(args.m2_repa_alignment_depths[0])
    args.m2_repa_align_lambda = float(config.get("m2_repa_align_lambda", 0.5))
    args.m2_repa_decouple_lambda = float(config.get("m2_repa_decouple_lambda", 0.05))
    args.m2_repa_projector_hidden_dim = int(
        config.get("m2_repa_projector_hidden_dim", 2048)
    )
    args.m2_repa_projector_layers = int(config.get("m2_repa_projector_layers", 3))
    args.m2_repa_max_spatial_tokens = int(
        config.get("m2_repa_max_spatial_tokens", -1)
    )
    args.m2_repa_decouple_max_samples = int(
        config.get("m2_repa_decouple_max_samples", 4096)
    )
    args.m2_repa_spatial_align = bool(config.get("m2_repa_spatial_align", True))
    args.m2_repa_temporal_align = bool(config.get("m2_repa_temporal_align", True))
    args.m2_repa_detach_teacher = bool(config.get("m2_repa_detach_teacher", True))
    args.m2_repa_encoder_chunk_size = int(
        config.get("m2_repa_encoder_chunk_size", 0)
    )
    args.m2_repa_sam_model_id = str(
        config.get("m2_repa_sam_model_id", "facebook/sam2-hiera-large")
    )
    args.m2_repa_sam_checkpoint = str(
        config.get("m2_repa_sam_checkpoint", "") or ""
    )
    args.m2_repa_sam_image_size = int(config.get("m2_repa_sam_image_size", 512))
    args.m2_repa_sam_teacher_dtype = str(
        config.get("m2_repa_sam_teacher_dtype", "float16")
    ).lower()
    args.m2_repa_sam_use_causal_memory = bool(
        config.get("m2_repa_sam_use_causal_memory", True)
    )
    args.m2_repa_depth_model_id = str(
        config.get(
            "m2_repa_depth_model_id",
            "depth-anything/Depth-Anything-V2-Base-hf",
        )
    )
    args.m2_repa_depth_checkpoint = str(
        config.get("m2_repa_depth_checkpoint", "") or ""
    )
    args.m2_repa_depth_image_size = int(config.get("m2_repa_depth_image_size", 518))
    args.m2_repa_depth_teacher_dtype = str(
        config.get("m2_repa_depth_teacher_dtype", "float16")
    ).lower()
    args.m2_repa_depth_feature_source = str(
        config.get("m2_repa_depth_feature_source", "auto")
    ).lower()

    if args.m2_repa_input_resolution not in (256, 512):
        raise ValueError(
            "m2_repa_input_resolution must be 256 or 512, got "
            f"{args.m2_repa_input_resolution}"
        )
    for depth in args.m2_repa_alignment_depths:
        if depth < 0:
            raise ValueError(
                f"m2_repa_alignment_depths entries must be >= 0, got {depth}"
            )
    if args.m2_repa_align_lambda < 0:
        raise ValueError(
            f"m2_repa_align_lambda must be >= 0, got {args.m2_repa_align_lambda}"
        )
    if args.enable_m2_repa and args.m2_repa_align_lambda <= 0:
        raise ValueError("m2_repa_align_lambda must be > 0 when enable_m2_repa is true")
    if args.m2_repa_decouple_lambda < 0:
        raise ValueError(
            "m2_repa_decouple_lambda must be >= 0, got "
            f"{args.m2_repa_decouple_lambda}"
        )
    if args.m2_repa_projector_hidden_dim <= 0:
        raise ValueError(
            "m2_repa_projector_hidden_dim must be > 0, got "
            f"{args.m2_repa_projector_hidden_dim}"
        )
    if args.m2_repa_projector_layers < 2:
        raise ValueError(
            "m2_repa_projector_layers must be >= 2, got "
            f"{args.m2_repa_projector_layers}"
        )
    if args.m2_repa_max_spatial_tokens == 0 or args.m2_repa_max_spatial_tokens < -1:
        raise ValueError(
            "m2_repa_max_spatial_tokens must be -1 (disabled) or > 0, got "
            f"{args.m2_repa_max_spatial_tokens}"
        )
    if args.m2_repa_decouple_max_samples < 2:
        raise ValueError(
            "m2_repa_decouple_max_samples must be >= 2, got "
            f"{args.m2_repa_decouple_max_samples}"
        )
    if args.m2_repa_encoder_chunk_size < 0:
        raise ValueError(
            "m2_repa_encoder_chunk_size must be >= 0, got "
            f"{args.m2_repa_encoder_chunk_size}"
        )
    if args.m2_repa_sam_image_size <= 0:
        raise ValueError(
            "m2_repa_sam_image_size must be > 0, got "
            f"{args.m2_repa_sam_image_size}"
        )
    if args.m2_repa_sam_teacher_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            "m2_repa_sam_teacher_dtype must be one of float16, bfloat16, float32; "
            f"got {args.m2_repa_sam_teacher_dtype!r}"
        )
    if args.m2_repa_depth_image_size <= 0:
        raise ValueError(
            "m2_repa_depth_image_size must be > 0, got "
            f"{args.m2_repa_depth_image_size}"
        )
    if args.m2_repa_depth_teacher_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            "m2_repa_depth_teacher_dtype must be one of float16, bfloat16, float32; "
            f"got {args.m2_repa_depth_teacher_dtype!r}"
        )
    if args.m2_repa_depth_feature_source not in {
        "auto",
        "hidden_states",
        "predicted_depth",
    }:
        raise ValueError(
            "m2_repa_depth_feature_source must be one of auto, hidden_states, "
            f"predicted_depth; got {args.m2_repa_depth_feature_source!r}"
        )
    if args.enable_m2_repa and len(args.m2_repa_encoder_name_list) < 2:
        raise ValueError(
            "enable_m2_repa requires at least two comma-separated encoders in "
            "m2_repa_encoder_names"
        )
    if args.enable_m2_repa:
        for name in args.m2_repa_encoder_name_list:
            lowered = name.lower()
            if lowered == "sam3":
                raise ValueError(
                    "M2-REPA SAM3 experts must be specified as "
                    "sam3:<huggingface-model-id>"
                )
            if (
                lowered.startswith("depth-anything-v2:")
                or lowered.startswith("depthanythingv2:")
                or lowered.startswith("depth_anything_v2:")
            ) and not name.split(":", 1)[1].strip():
                raise ValueError(
                    "M2-REPA Depth Anything V2 inline specs must include a "
                    "model ID after ':'"
                )

    mutually_exclusive_flags = [
        "enable_repa",
        "enable_irepa",
        "enable_videorepa",
        "enable_moalign",
        "crepa_enabled",
        "sara_enabled",
        "enable_structure_from_tracking",
    ]
    if args.enable_m2_repa:
        conflicts = [flag for flag in mutually_exclusive_flags if bool(config.get(flag, False))]
        if conflicts:
            raise ValueError(
                "enable_m2_repa is mutually exclusive with "
                + ", ".join(sorted(conflicts))
            )
        logger.info(
            "M2-REPA enabled (encoders=%s, depths=%s, align_lambda=%.4f, decouple_lambda=%.4f)",
            args.m2_repa_encoder_name_list,
            args.m2_repa_alignment_depths,
            args.m2_repa_align_lambda,
            args.m2_repa_decouple_lambda,
        )
