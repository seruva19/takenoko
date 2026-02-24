"""Config parsing for Self-Paced SRPO reward mixing."""

from __future__ import annotations

from typing import Any, Dict


def apply_srpo_self_paced_config(args: Any, config: Dict[str, Any], logger):
    """Parse/validate self-paced SRPO reward options into args."""
    args.srpo_enable_self_paced_reward = bool(
        config.get("srpo_enable_self_paced_reward", False)
    )
    args.srpo_self_paced_visual_threshold = float(
        config.get("srpo_self_paced_visual_threshold", 0.75)
    )
    args.srpo_self_paced_temporal_threshold = float(
        config.get("srpo_self_paced_temporal_threshold", 0.75)
    )
    args.srpo_self_paced_semantic_threshold = float(
        config.get("srpo_self_paced_semantic_threshold", 0.75)
    )
    args.srpo_self_paced_softmax_beta = float(
        config.get("srpo_self_paced_softmax_beta", 5.0)
    )
    args.srpo_self_paced_sparsity_lambda = float(
        config.get("srpo_self_paced_sparsity_lambda", 0.5)
    )
    args.srpo_self_paced_sigmoid_scale = float(
        config.get("srpo_self_paced_sigmoid_scale", 8.0)
    )
    args.srpo_self_paced_visual_scale = float(
        config.get("srpo_self_paced_visual_scale", 1.0)
    )
    args.srpo_self_paced_temporal_scale = float(
        config.get("srpo_self_paced_temporal_scale", 1.0)
    )
    args.srpo_self_paced_semantic_scale = float(
        config.get("srpo_self_paced_semantic_scale", 1.0)
    )
    args.srpo_self_paced_enable_advantage_norm = bool(
        config.get("srpo_self_paced_enable_advantage_norm", True)
    )
    args.srpo_self_paced_advantage_norm_eps = float(
        config.get("srpo_self_paced_advantage_norm_eps", 1e-6)
    )
    args.srpo_self_paced_use_grpo_clip_objective = bool(
        config.get("srpo_self_paced_use_grpo_clip_objective", False)
    )
    args.srpo_self_paced_grpo_clip_eps = float(
        config.get("srpo_self_paced_grpo_clip_eps", 1e-4)
    )
    args.srpo_self_paced_policy_std = float(
        config.get("srpo_self_paced_policy_std", 1.0)
    )
    args.srpo_self_paced_apply_discount_to_grpo = bool(
        config.get("srpo_self_paced_apply_discount_to_grpo", True)
    )
    args.srpo_self_paced_use_lagged_old_policy = bool(
        config.get("srpo_self_paced_use_lagged_old_policy", True)
    )
    args.srpo_self_paced_old_policy_update_interval = int(
        config.get("srpo_self_paced_old_policy_update_interval", 1)
    )
    args.srpo_self_paced_old_policy_offload_cpu = bool(
        config.get("srpo_self_paced_old_policy_offload_cpu", True)
    )
    args.srpo_self_paced_logprob_mode = str(
        config.get("srpo_self_paced_logprob_mode", "action_gaussian")
    )
    args.srpo_self_paced_enable_offline_threshold_calibration = bool(
        config.get("srpo_self_paced_enable_offline_threshold_calibration", False)
    )
    args.srpo_self_paced_offline_calibration_steps = int(
        config.get("srpo_self_paced_offline_calibration_steps", 12)
    )
    args.srpo_self_paced_offline_calibration_batch_size = int(
        config.get("srpo_self_paced_offline_calibration_batch_size", 0)
    )
    args.srpo_self_paced_offline_calibration_update_network = bool(
        config.get("srpo_self_paced_offline_calibration_update_network", False)
    )
    args.srpo_self_paced_enable_prompt_curriculum = bool(
        config.get("srpo_self_paced_enable_prompt_curriculum", False)
    )
    args.srpo_self_paced_stage1_visual_prompt_suffix = str(
        config.get("srpo_self_paced_stage1_visual_prompt_suffix", "") or ""
    )
    args.srpo_self_paced_stage3_semantic_prompt_suffix = str(
        config.get("srpo_self_paced_stage3_semantic_prompt_suffix", "") or ""
    )
    args.srpo_self_paced_auto_calibrate_thresholds = bool(
        config.get("srpo_self_paced_auto_calibrate_thresholds", False)
    )
    args.srpo_self_paced_threshold_calibration_factor = float(
        config.get("srpo_self_paced_threshold_calibration_factor", 0.7)
    )
    args.srpo_self_paced_threshold_calibration_warmup_steps = int(
        config.get("srpo_self_paced_threshold_calibration_warmup_steps", 50)
    )
    args.srpo_self_paced_threshold_calibration_momentum = float(
        config.get("srpo_self_paced_threshold_calibration_momentum", 0.9)
    )

    if args.srpo_enable_self_paced_reward and not bool(
        getattr(args, "enable_srpo_training", False)
    ):
        raise ValueError("srpo_enable_self_paced_reward requires enable_srpo_training=true.")

    if args.srpo_enable_self_paced_reward:
        if args.srpo_self_paced_softmax_beta <= 0.0:
            raise ValueError("srpo_self_paced_softmax_beta must be > 0.")
        if args.srpo_self_paced_sigmoid_scale <= 0.0:
            raise ValueError("srpo_self_paced_sigmoid_scale must be > 0.")
        if args.srpo_self_paced_advantage_norm_eps <= 0.0:
            raise ValueError("srpo_self_paced_advantage_norm_eps must be > 0.")
        if not (0.0 < args.srpo_self_paced_grpo_clip_eps <= 1.0):
            raise ValueError("srpo_self_paced_grpo_clip_eps must be in (0, 1].")
        if args.srpo_self_paced_policy_std <= 0.0:
            raise ValueError("srpo_self_paced_policy_std must be > 0.")
        if args.srpo_self_paced_old_policy_update_interval < 1:
            raise ValueError("srpo_self_paced_old_policy_update_interval must be >= 1.")
        if args.srpo_self_paced_logprob_mode not in (
            "action_gaussian",
            "latent_error_proxy",
        ):
            raise ValueError(
                "srpo_self_paced_logprob_mode must be 'action_gaussian' or 'latent_error_proxy'."
            )
        if args.srpo_self_paced_offline_calibration_steps < 1:
            raise ValueError("srpo_self_paced_offline_calibration_steps must be >= 1.")
        if args.srpo_self_paced_offline_calibration_batch_size < 0:
            raise ValueError("srpo_self_paced_offline_calibration_batch_size must be >= 0.")
        if args.srpo_self_paced_visual_scale < 0.0:
            raise ValueError("srpo_self_paced_visual_scale must be >= 0.")
        if args.srpo_self_paced_temporal_scale < 0.0:
            raise ValueError("srpo_self_paced_temporal_scale must be >= 0.")
        if args.srpo_self_paced_semantic_scale < 0.0:
            raise ValueError("srpo_self_paced_semantic_scale must be >= 0.")
        if args.srpo_self_paced_threshold_calibration_warmup_steps < 1:
            raise ValueError(
                "srpo_self_paced_threshold_calibration_warmup_steps must be >= 1."
            )
        if not (0.0 <= args.srpo_self_paced_threshold_calibration_momentum < 1.0):
            raise ValueError(
                "srpo_self_paced_threshold_calibration_momentum must be in [0, 1)."
            )
        if (
            args.srpo_self_paced_visual_scale
            + args.srpo_self_paced_temporal_scale
            + args.srpo_self_paced_semantic_scale
            <= 0.0
        ):
            raise ValueError(
                "At least one of srpo_self_paced_visual_scale, "
                "srpo_self_paced_temporal_scale, srpo_self_paced_semantic_scale must be > 0."
            )

        logger.info(
            "Self-Paced SRPO reward enabled (beta=%.3f, lambda=%.3f, thresholds=(%.3f, %.3f, %.3f), "
            "scales=(%.3f, %.3f, %.3f), advantage_norm=%s, grpo_clip=%s, lagged_old=%s, old_update=%d, "
            "logprob=%s, prompt_curriculum=%s, auto_calibrate=%s, offline_calibration=%s/%d).",
            args.srpo_self_paced_softmax_beta,
            args.srpo_self_paced_sparsity_lambda,
            args.srpo_self_paced_visual_threshold,
            args.srpo_self_paced_temporal_threshold,
            args.srpo_self_paced_semantic_threshold,
            args.srpo_self_paced_visual_scale,
            args.srpo_self_paced_temporal_scale,
            args.srpo_self_paced_semantic_scale,
            args.srpo_self_paced_enable_advantage_norm,
            args.srpo_self_paced_use_grpo_clip_objective,
            args.srpo_self_paced_use_lagged_old_policy,
            args.srpo_self_paced_old_policy_update_interval,
            args.srpo_self_paced_logprob_mode,
            args.srpo_self_paced_enable_prompt_curriculum,
            args.srpo_self_paced_auto_calibrate_thresholds,
            args.srpo_self_paced_enable_offline_threshold_calibration,
            args.srpo_self_paced_offline_calibration_steps,
        )

    return args
