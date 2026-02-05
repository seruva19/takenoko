from __future__ import annotations

from typing import Any, Iterable, List


_VCD_LAYER_NAME_TO_INDEX = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu5_1": 29,
}


def _normalize_feature_layers(value: Any) -> List[int]:
    if value is None:
        return [1, 6, 11, 20, 29]

    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raise ValueError(
            "vcd_feature_layers must be a list/tuple or comma-separated string."
        )

    indices: list[int] = []
    for item in raw_items:
        if isinstance(item, int):
            indices.append(item)
            continue

        text = str(item).strip().lower()
        if not text:
            continue
        if text in _VCD_LAYER_NAME_TO_INDEX:
            indices.append(_VCD_LAYER_NAME_TO_INDEX[text])
            continue
        try:
            indices.append(int(text))
        except Exception as exc:
            raise ValueError(
                "Invalid vcd_feature_layers entry "
                f"'{item}'. Use layer names ({sorted(_VCD_LAYER_NAME_TO_INDEX)}) "
                "or integer feature indices."
            ) from exc

    if not indices:
        raise ValueError("vcd_feature_layers must contain at least one layer index.")

    indices = sorted(set(indices))
    if indices[0] < 0:
        raise ValueError(
            f"vcd_feature_layers indices must be >= 0, got {indices[0]}."
        )
    return indices


def apply_video_consistency_distance_config(
    args: Any, config: dict[str, Any], logger: Any
) -> Any:
    args.enable_video_consistency_distance = bool(
        config.get("enable_video_consistency_distance", False)
    )
    args.vcd_loss_weight = float(config.get("vcd_loss_weight", 0.0))
    args.vcd_use_amplitude = bool(config.get("vcd_use_amplitude", True))
    args.vcd_use_phase = bool(config.get("vcd_use_phase", True))
    args.vcd_amplitude_weight = float(config.get("vcd_amplitude_weight", 1.0))
    args.vcd_phase_weight = float(config.get("vcd_phase_weight", 1.0))
    args.vcd_num_sampled_frames = int(config.get("vcd_num_sampled_frames", 1))
    args.vcd_random_frame_sampling = bool(config.get("vcd_random_frame_sampling", True))
    args.vcd_use_temporal_weight = bool(config.get("vcd_use_temporal_weight", True))
    args.vcd_start_step = int(config.get("vcd_start_step", 0))
    raw_vcd_end_step = config.get("vcd_end_step", None)
    args.vcd_end_step = (
        None if raw_vcd_end_step is None else int(raw_vcd_end_step)
    )
    args.vcd_warmup_steps = int(config.get("vcd_warmup_steps", 0))
    args.vcd_apply_every_n_steps = int(config.get("vcd_apply_every_n_steps", 1))
    args.vcd_feature_layers = _normalize_feature_layers(
        config.get(
            "vcd_feature_layers",
            ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        )
    )
    args.vcd_feature_resolution = int(config.get("vcd_feature_resolution", 224))
    args.vcd_max_coeffs = int(config.get("vcd_max_coeffs", 16384))
    args.vcd_random_coeff_sampling = bool(config.get("vcd_random_coeff_sampling", True))
    args.vcd_use_pretrained_vgg = bool(config.get("vcd_use_pretrained_vgg", True))
    args.vcd_conditioning_source = str(
        config.get("vcd_conditioning_source", "batch_first_frame")
    ).lower()
    args.vcd_detach_conditioning_frame = bool(
        config.get("vcd_detach_conditioning_frame", True)
    )
    args.vcd_assume_neg_one_to_one = bool(
        config.get("vcd_assume_neg_one_to_one", True)
    )

    if args.vcd_loss_weight < 0.0:
        raise ValueError(f"vcd_loss_weight must be >= 0, got {args.vcd_loss_weight}")
    if not args.vcd_use_amplitude and not args.vcd_use_phase:
        raise ValueError("At least one of vcd_use_amplitude or vcd_use_phase must be true.")
    if args.vcd_amplitude_weight < 0.0:
        raise ValueError(
            "vcd_amplitude_weight must be >= 0, got "
            f"{args.vcd_amplitude_weight}"
        )
    if args.vcd_phase_weight < 0.0:
        raise ValueError(
            f"vcd_phase_weight must be >= 0, got {args.vcd_phase_weight}"
        )
    if args.vcd_num_sampled_frames < 1:
        raise ValueError(
            "vcd_num_sampled_frames must be >= 1, got "
            f"{args.vcd_num_sampled_frames}"
        )
    if args.vcd_start_step < 0:
        raise ValueError(f"vcd_start_step must be >= 0, got {args.vcd_start_step}")
    if args.vcd_end_step is not None and args.vcd_end_step < args.vcd_start_step:
        raise ValueError(
            f"vcd_end_step ({args.vcd_end_step}) must be >= vcd_start_step ({args.vcd_start_step})."
        )
    if args.vcd_warmup_steps < 0:
        raise ValueError(
            f"vcd_warmup_steps must be >= 0, got {args.vcd_warmup_steps}"
        )
    if args.vcd_apply_every_n_steps < 1:
        raise ValueError(
            "vcd_apply_every_n_steps must be >= 1, got "
            f"{args.vcd_apply_every_n_steps}"
        )
    if args.vcd_feature_resolution < 16:
        raise ValueError(
            "vcd_feature_resolution must be >= 16, got "
            f"{args.vcd_feature_resolution}"
        )
    if args.vcd_max_coeffs < 0:
        raise ValueError(f"vcd_max_coeffs must be >= 0, got {args.vcd_max_coeffs}")
    if args.vcd_conditioning_source not in {"batch_first_frame", "pred_first_frame"}:
        raise ValueError(
            "vcd_conditioning_source must be 'batch_first_frame' or "
            f"'pred_first_frame', got '{args.vcd_conditioning_source}'."
        )

    if args.enable_video_consistency_distance:
        logger.info(
            "VCD enabled (weight=%.4f, amp=%s, phase=%s, sampled_frames=%d, "
            "temporal_weight=%s, layers=%s, resolution=%d, conditioning=%s).",
            args.vcd_loss_weight,
            args.vcd_use_amplitude,
            args.vcd_use_phase,
            args.vcd_num_sampled_frames,
            args.vcd_use_temporal_weight,
            args.vcd_feature_layers,
            args.vcd_feature_resolution,
            args.vcd_conditioning_source,
        )
        if args.vcd_loss_weight == 0.0:
            logger.warning(
                "VCD is enabled but vcd_loss_weight is 0.0; it will have no training effect."
            )

    return args
