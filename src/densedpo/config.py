"""DenseDPO config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


def parse_densedpo_config(config: Dict[str, Any], args: Any, logger) -> None:
    """Parse DenseDPO settings into args with validation and logging."""
    args.enable_densedpo_training = bool(
        config.get("enable_densedpo_training", False)
    )
    args.densedpo_partial_noise_eta = float(
        config.get("densedpo_partial_noise_eta", 0.5)
    )
    args.densedpo_num_inference_steps = int(
        config.get("densedpo_num_inference_steps", 50)
    )
    args.densedpo_segment_frames = int(
        config.get("densedpo_segment_frames", 16)
    )
    args.densedpo_beta = float(config.get("densedpo_beta", 0.1))
    args.densedpo_label_source = config.get(
        "densedpo_label_source", "reward"
    )
    args.densedpo_vlm_model_path = config.get(
        "densedpo_vlm_model_path", None
    )
    if isinstance(args.densedpo_vlm_model_path, str) and not args.densedpo_vlm_model_path.strip():
        args.densedpo_vlm_model_path = None
    args.densedpo_vlm_dtype = config.get(
        "densedpo_vlm_dtype", "bfloat16"
    )
    args.densedpo_vlm_prompt = config.get(
        "densedpo_vlm_prompt",
        "Rate the visual quality and motion consistency of this short video clip "
        "on a scale of 1 to 10. Respond with a single number.",
    )
    args.densedpo_vlm_max_new_tokens = int(
        config.get("densedpo_vlm_max_new_tokens", 8)
    )
    args.densedpo_vlm_temperature = float(
        config.get("densedpo_vlm_temperature", 0.0)
    )
    args.densedpo_vlm_cache_dir = config.get(
        "densedpo_vlm_cache_dir", None
    )
    if isinstance(args.densedpo_vlm_cache_dir, str) and not args.densedpo_vlm_cache_dir.strip():
        args.densedpo_vlm_cache_dir = None
    args.densedpo_vlm_max_frames = int(
        config.get("densedpo_vlm_max_frames", 8)
    )
    args.densedpo_segment_preference_key = config.get(
        "densedpo_segment_preference_key", "densedpo_segment_preferences"
    )
    args.densedpo_reward_model_name = config.get(
        "densedpo_reward_model_name", "hps"
    )
    args.densedpo_reward_model_dtype = config.get(
        "densedpo_reward_model_dtype", "float32"
    )
    args.densedpo_reward_frame_strategy = config.get(
        "densedpo_reward_frame_strategy", "first"
    )
    args.densedpo_reward_num_frames = int(
        config.get("densedpo_reward_num_frames", 1)
    )
    args.densedpo_reward_aggregation = config.get(
        "densedpo_reward_aggregation", "mean"
    )

    if not (0.0 <= args.densedpo_partial_noise_eta <= 1.0):
        raise ValueError("densedpo_partial_noise_eta must be in [0, 1].")
    if args.densedpo_num_inference_steps < 2:
        raise ValueError("densedpo_num_inference_steps must be >= 2.")
    if args.densedpo_segment_frames < 1:
        raise ValueError("densedpo_segment_frames must be >= 1.")
    if args.densedpo_beta <= 0.0:
        raise ValueError("densedpo_beta must be > 0.")
    if args.densedpo_label_source not in ("reward", "provided", "vlm"):
        raise ValueError(
            "densedpo_label_source must be 'reward', 'provided', or 'vlm'."
        )
    if args.densedpo_label_source == "vlm" and not args.densedpo_vlm_model_path:
        raise ValueError(
            "densedpo_vlm_model_path is required when densedpo_label_source='vlm'."
        )
    if args.densedpo_vlm_max_new_tokens < 1:
        raise ValueError("densedpo_vlm_max_new_tokens must be >= 1.")
    if args.densedpo_vlm_temperature < 0.0:
        raise ValueError("densedpo_vlm_temperature must be >= 0.")
    if args.densedpo_vlm_max_frames < 1:
        raise ValueError("densedpo_vlm_max_frames must be >= 1.")
    if args.densedpo_reward_frame_strategy not in (
        "first",
        "uniform",
        "all",
        "boundary",
    ):
        raise ValueError(
            "densedpo_reward_frame_strategy must be "
            "'first', 'uniform', 'all', or 'boundary'."
        )
    if args.densedpo_reward_num_frames < 1:
        raise ValueError("densedpo_reward_num_frames must be >= 1.")
    if args.densedpo_reward_aggregation not in (
        "mean",
        "min",
        "max",
        "weighted",
    ):
        raise ValueError(
            "densedpo_reward_aggregation must be "
            "'mean', 'min', 'max', or 'weighted'."
        )
    if not args.densedpo_segment_preference_key:
        raise ValueError("densedpo_segment_preference_key must be non-empty.")

    if args.enable_densedpo_training:
        logger.info(
            "DenseDPO enabled (eta=%s, segments=%s, beta=%s, label_source=%s).",
            args.densedpo_partial_noise_eta,
            args.densedpo_segment_frames,
            args.densedpo_beta,
            args.densedpo_label_source,
        )
