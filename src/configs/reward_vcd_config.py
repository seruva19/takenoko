from __future__ import annotations

from typing import Any, Iterable, List


_REWARD_VCD_LAYER_NAME_TO_INDEX = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu5_1": 29,
}


def _normalize_reward_vcd_layers(value: Any) -> List[int]:
    if value is None:
        return [1, 6, 11, 20, 29]

    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raise ValueError(
            "reward_vcd_feature_layers must be a list/tuple or comma-separated string."
        )

    indices: list[int] = []
    for item in raw_items:
        if isinstance(item, int):
            indices.append(item)
            continue

        text = str(item).strip().lower()
        if not text:
            continue
        if text in _REWARD_VCD_LAYER_NAME_TO_INDEX:
            indices.append(_REWARD_VCD_LAYER_NAME_TO_INDEX[text])
            continue
        try:
            indices.append(int(text))
        except Exception as exc:
            raise ValueError(
                "Invalid reward_vcd_feature_layers entry "
                f"'{item}'. Use layer names "
                f"({sorted(_REWARD_VCD_LAYER_NAME_TO_INDEX)}) or integer indices."
            ) from exc

    if not indices:
        raise ValueError(
            "reward_vcd_feature_layers must contain at least one layer index."
        )

    indices = sorted(set(indices))
    if indices[0] < 0:
        raise ValueError(
            f"reward_vcd_feature_layers indices must be >= 0, got {indices[0]}."
        )
    return indices


def apply_reward_vcd_config(args: Any, config: dict[str, Any], logger: Any) -> Any:
    args.reward_vcd_loss_scale = float(config.get("reward_vcd_loss_scale", 1.0))
    args.reward_vcd_use_amplitude = bool(config.get("reward_vcd_use_amplitude", True))
    args.reward_vcd_use_phase = bool(config.get("reward_vcd_use_phase", True))
    args.reward_vcd_amplitude_weight = float(
        config.get("reward_vcd_amplitude_weight", 1.0)
    )
    args.reward_vcd_phase_weight = float(config.get("reward_vcd_phase_weight", 1.0))
    args.reward_vcd_num_sampled_frames = int(
        config.get("reward_vcd_num_sampled_frames", 4)
    )
    args.reward_vcd_random_frame_sampling = bool(
        config.get("reward_vcd_random_frame_sampling", True)
    )
    args.reward_vcd_use_temporal_weight = bool(
        config.get("reward_vcd_use_temporal_weight", True)
    )
    args.reward_vcd_feature_layers = _normalize_reward_vcd_layers(
        config.get(
            "reward_vcd_feature_layers",
            ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        )
    )
    args.reward_vcd_feature_resolution = int(
        config.get("reward_vcd_feature_resolution", 224)
    )
    args.reward_vcd_max_coeffs = int(config.get("reward_vcd_max_coeffs", 16384))
    args.reward_vcd_random_coeff_sampling = bool(
        config.get("reward_vcd_random_coeff_sampling", True)
    )
    args.reward_vcd_use_pretrained_vgg = bool(
        config.get("reward_vcd_use_pretrained_vgg", True)
    )
    args.reward_vcd_conditioning_source = str(
        config.get("reward_vcd_conditioning_source", "first_generated_frame")
    ).lower()
    args.reward_vcd_detach_conditioning_frame = bool(
        config.get("reward_vcd_detach_conditioning_frame", True)
    )
    args.reward_vcd_assume_neg_one_to_one = bool(
        config.get("reward_vcd_assume_neg_one_to_one", False)
    )
    args.reward_vcd_force_min_decoded_frames = bool(
        config.get("reward_vcd_force_min_decoded_frames", True)
    )
    args.reward_vcd_min_decoded_frames = int(
        config.get("reward_vcd_min_decoded_frames", 2)
    )

    if args.reward_vcd_loss_scale <= 0.0:
        raise ValueError(
            f"reward_vcd_loss_scale must be > 0, got {args.reward_vcd_loss_scale}"
        )
    if not args.reward_vcd_use_amplitude and not args.reward_vcd_use_phase:
        raise ValueError(
            "At least one of reward_vcd_use_amplitude or reward_vcd_use_phase must be true."
        )
    if args.reward_vcd_amplitude_weight < 0.0:
        raise ValueError(
            "reward_vcd_amplitude_weight must be >= 0, got "
            f"{args.reward_vcd_amplitude_weight}"
        )
    if args.reward_vcd_phase_weight < 0.0:
        raise ValueError(
            f"reward_vcd_phase_weight must be >= 0, got {args.reward_vcd_phase_weight}"
        )
    if args.reward_vcd_num_sampled_frames < 1:
        raise ValueError(
            "reward_vcd_num_sampled_frames must be >= 1, got "
            f"{args.reward_vcd_num_sampled_frames}"
        )
    if args.reward_vcd_feature_resolution < 16:
        raise ValueError(
            "reward_vcd_feature_resolution must be >= 16, got "
            f"{args.reward_vcd_feature_resolution}"
        )
    if args.reward_vcd_max_coeffs < 0:
        raise ValueError(
            f"reward_vcd_max_coeffs must be >= 0, got {args.reward_vcd_max_coeffs}"
        )
    if args.reward_vcd_conditioning_source not in {
        "first_generated_frame",
        "provided_first_frame",
    }:
        raise ValueError(
            "reward_vcd_conditioning_source must be 'first_generated_frame' or "
            f"'provided_first_frame', got '{args.reward_vcd_conditioning_source}'."
        )
    if args.reward_vcd_min_decoded_frames < 2:
        raise ValueError(
            "reward_vcd_min_decoded_frames must be >= 2, got "
            f"{args.reward_vcd_min_decoded_frames}"
        )

    if (
        bool(getattr(args, "enable_reward_lora", False))
        and str(getattr(args, "reward_fn", "")).lower() == "vcdreward"
    ):
        logger.info(
            "Reward VCD configured (scale=%.4f, amp=%s, phase=%s, sampled_frames=%d, "
            "temporal_weight=%s, layers=%s, resolution=%d).",
            args.reward_vcd_loss_scale,
            args.reward_vcd_use_amplitude,
            args.reward_vcd_use_phase,
            args.reward_vcd_num_sampled_frames,
            args.reward_vcd_use_temporal_weight,
            args.reward_vcd_feature_layers,
            args.reward_vcd_feature_resolution,
        )

    return args
