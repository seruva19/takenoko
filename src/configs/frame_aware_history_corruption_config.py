from __future__ import annotations

from typing import Any, Dict


def apply_frame_aware_history_corruption_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse frame-aware history corruption settings onto args."""
    deprecated_frame_aware_keys = (
        "self_resampling_frame_aware_corrupt_enabled",
        "self_resampling_frame_aware_corrupt_keep_first_frame",
        "self_resampling_frame_aware_corrupt_exposure_prob",
        "self_resampling_frame_aware_corrupt_noise_prob",
        "self_resampling_frame_aware_corrupt_blur_prob",
        "self_resampling_frame_aware_corrupt_clean_prob",
        "self_resampling_frame_aware_corrupt_exposure_min",
        "self_resampling_frame_aware_corrupt_exposure_max",
        "self_resampling_frame_aware_corrupt_noise_min",
        "self_resampling_frame_aware_corrupt_noise_max",
        "self_resampling_frame_aware_corrupt_downsample_min",
        "self_resampling_frame_aware_corrupt_downsample_max",
    )
    found_deprecated_frame_aware_keys = [
        key for key in deprecated_frame_aware_keys if key in config
    ]
    if found_deprecated_frame_aware_keys:
        raise ValueError(
            "Deprecated frame-aware history corruption keys are no longer supported: "
            + ", ".join(found_deprecated_frame_aware_keys)
        )

    args.enable_frame_aware_history_corruption = bool(
        config.get("enable_frame_aware_history_corruption", False)
    )
    args.frame_aware_history_corruption_keep_first_frame = bool(
        config.get("frame_aware_history_corruption_keep_first_frame", True)
    )
    args.frame_aware_history_corruption_exposure_prob = float(
        config.get("frame_aware_history_corruption_exposure_prob", 0.0)
    )
    args.frame_aware_history_corruption_noise_prob = float(
        config.get("frame_aware_history_corruption_noise_prob", 0.0)
    )
    args.frame_aware_history_corruption_blur_prob = float(
        config.get("frame_aware_history_corruption_blur_prob", 0.0)
    )
    args.frame_aware_history_corruption_clean_prob = float(
        config.get("frame_aware_history_corruption_clean_prob", 1.0)
    )
    args.frame_aware_history_corruption_exposure_min = float(
        config.get("frame_aware_history_corruption_exposure_min", 0.85)
    )
    args.frame_aware_history_corruption_exposure_max = float(
        config.get("frame_aware_history_corruption_exposure_max", 1.15)
    )
    args.frame_aware_history_corruption_noise_min = float(
        config.get("frame_aware_history_corruption_noise_min", 0.01)
    )
    args.frame_aware_history_corruption_noise_max = float(
        config.get("frame_aware_history_corruption_noise_max", 0.08)
    )
    args.frame_aware_history_corruption_downsample_min = float(
        config.get("frame_aware_history_corruption_downsample_min", 1.25)
    )
    args.frame_aware_history_corruption_downsample_max = float(
        config.get("frame_aware_history_corruption_downsample_max", 2.5)
    )
    args.frame_aware_history_corruption_history_start_frame = int(
        config.get("frame_aware_history_corruption_history_start_frame", 0)
    )
    args.frame_aware_history_corruption_history_exclude_tail_frames = int(
        config.get("frame_aware_history_corruption_history_exclude_tail_frames", 1)
    )
    args.frame_aware_history_corruption_blend = float(
        config.get("frame_aware_history_corruption_blend", 1.0)
    )
    args.frame_aware_history_corruption_log_interval = int(
        config.get("frame_aware_history_corruption_log_interval", 50)
    )

    if not 0.0 <= args.frame_aware_history_corruption_exposure_prob <= 1.0:
        raise ValueError(
            "frame_aware_history_corruption_exposure_prob must be between 0 and 1"
        )
    if not 0.0 <= args.frame_aware_history_corruption_noise_prob <= 1.0:
        raise ValueError(
            "frame_aware_history_corruption_noise_prob must be between 0 and 1"
        )
    if not 0.0 <= args.frame_aware_history_corruption_blur_prob <= 1.0:
        raise ValueError(
            "frame_aware_history_corruption_blur_prob must be between 0 and 1"
        )
    if not 0.0 <= args.frame_aware_history_corruption_clean_prob <= 1.0:
        raise ValueError(
            "frame_aware_history_corruption_clean_prob must be between 0 and 1"
        )
    total_prob = (
        args.frame_aware_history_corruption_exposure_prob
        + args.frame_aware_history_corruption_noise_prob
        + args.frame_aware_history_corruption_blur_prob
        + args.frame_aware_history_corruption_clean_prob
    )
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(
            "frame_aware_history_corruption_* probabilities must sum to 1.0"
        )
    if args.frame_aware_history_corruption_exposure_min <= 0.0:
        raise ValueError("frame_aware_history_corruption_exposure_min must be > 0")
    if (
        args.frame_aware_history_corruption_exposure_max
        < args.frame_aware_history_corruption_exposure_min
    ):
        raise ValueError(
            "frame_aware_history_corruption_exposure_max must be >= "
            "frame_aware_history_corruption_exposure_min"
        )
    if args.frame_aware_history_corruption_noise_min < 0.0:
        raise ValueError("frame_aware_history_corruption_noise_min must be >= 0")
    if (
        args.frame_aware_history_corruption_noise_max
        < args.frame_aware_history_corruption_noise_min
    ):
        raise ValueError(
            "frame_aware_history_corruption_noise_max must be >= "
            "frame_aware_history_corruption_noise_min"
        )
    if args.frame_aware_history_corruption_downsample_min < 1.0:
        raise ValueError(
            "frame_aware_history_corruption_downsample_min must be >= 1"
        )
    if (
        args.frame_aware_history_corruption_downsample_max
        < args.frame_aware_history_corruption_downsample_min
    ):
        raise ValueError(
            "frame_aware_history_corruption_downsample_max must be >= "
            "frame_aware_history_corruption_downsample_min"
        )
    if args.frame_aware_history_corruption_history_start_frame < 0:
        raise ValueError(
            "frame_aware_history_corruption_history_start_frame must be >= 0"
        )
    if args.frame_aware_history_corruption_history_exclude_tail_frames < 0:
        raise ValueError(
            "frame_aware_history_corruption_history_exclude_tail_frames must be >= 0"
        )
    if not 0.0 <= args.frame_aware_history_corruption_blend <= 1.0:
        raise ValueError("frame_aware_history_corruption_blend must be between 0 and 1")
    if args.frame_aware_history_corruption_log_interval <= 0:
        raise ValueError("frame_aware_history_corruption_log_interval must be > 0")

    if args.enable_frame_aware_history_corruption:
        logger.info(
            "Frame-aware history corruption enabled: keep_first_frame=%s probs(exposure=%.3f noise=%.3f blur=%.3f clean=%.3f) exposure=[%.3f, %.3f] noise=[%.3f, %.3f] downsample=[%.3f, %.3f] history_start=%d exclude_tail=%d blend=%.3f",
            str(args.frame_aware_history_corruption_keep_first_frame).lower(),
            args.frame_aware_history_corruption_exposure_prob,
            args.frame_aware_history_corruption_noise_prob,
            args.frame_aware_history_corruption_blur_prob,
            args.frame_aware_history_corruption_clean_prob,
            args.frame_aware_history_corruption_exposure_min,
            args.frame_aware_history_corruption_exposure_max,
            args.frame_aware_history_corruption_noise_min,
            args.frame_aware_history_corruption_noise_max,
            args.frame_aware_history_corruption_downsample_min,
            args.frame_aware_history_corruption_downsample_max,
            args.frame_aware_history_corruption_history_start_frame,
            args.frame_aware_history_corruption_history_exclude_tail_frames,
            args.frame_aware_history_corruption_blend,
        )
