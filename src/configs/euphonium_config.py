from __future__ import annotations

from typing import Any, Dict


_ALLOWED_DUAL_REWARD_MODES = {"none", "only", "both"}
_ALLOWED_PROCESS_MODEL_TYPES = {"none", "torchscript", "python_callable"}
_ALLOWED_PROCESS_MODEL_DTYPES = {"float32", "bfloat16", "float16"}
_ALLOWED_PROCESS_GRADIENT_MODES = {"autograd", "spsa"}


def apply_euphonium_config(args: Any, config: Dict[str, Any], logger: Any) -> Any:
    """Parse Euphonium-inspired SRPO settings (training-only, default disabled)."""
    args.srpo_enable_euphonium = bool(config.get("srpo_enable_euphonium", False))
    args.srpo_euphonium_process_reward_guidance_enabled = bool(
        config.get("srpo_euphonium_process_reward_guidance_enabled", False)
    )
    args.srpo_euphonium_process_reward_model_type = str(
        config.get("srpo_euphonium_process_reward_model_type", "none")
    ).lower()
    args.srpo_euphonium_process_reward_model_path = str(
        config.get("srpo_euphonium_process_reward_model_path", "") or ""
    )
    args.srpo_euphonium_process_reward_model_entry = str(
        config.get("srpo_euphonium_process_reward_model_entry", "") or ""
    )
    args.srpo_euphonium_process_reward_model_dtype = str(
        config.get("srpo_euphonium_process_reward_model_dtype", "float32")
    ).lower()
    args.srpo_euphonium_process_reward_allow_proxy_fallback = bool(
        config.get("srpo_euphonium_process_reward_allow_proxy_fallback", True)
    )
    args.srpo_euphonium_process_reward_gradient_mode = str(
        config.get("srpo_euphonium_process_reward_gradient_mode", "autograd")
    ).lower()
    args.srpo_euphonium_process_reward_spsa_sigma = float(
        config.get("srpo_euphonium_process_reward_spsa_sigma", 0.01)
    )
    args.srpo_euphonium_process_reward_spsa_num_samples = int(
        config.get("srpo_euphonium_process_reward_spsa_num_samples", 1)
    )
    args.srpo_euphonium_process_reward_guidance_scale = float(
        config.get("srpo_euphonium_process_reward_guidance_scale", 0.1)
    )
    args.srpo_euphonium_process_reward_guidance_kl_beta = float(
        config.get("srpo_euphonium_process_reward_guidance_kl_beta", 0.1)
    )
    args.srpo_euphonium_process_reward_guidance_eta = float(
        config.get("srpo_euphonium_process_reward_guidance_eta", 1.0)
    )
    args.srpo_euphonium_process_reward_start_step = int(
        config.get("srpo_euphonium_process_reward_start_step", 0)
    )
    args.srpo_euphonium_process_reward_end_step = int(
        config.get("srpo_euphonium_process_reward_end_step", -1)
    )
    args.srpo_euphonium_process_reward_interval = int(
        config.get("srpo_euphonium_process_reward_interval", 1)
    )
    args.srpo_euphonium_process_reward_normalize_gradient = bool(
        config.get("srpo_euphonium_process_reward_normalize_gradient", True)
    )
    args.srpo_euphonium_use_delta_t_for_guidance = bool(
        config.get("srpo_euphonium_use_delta_t_for_guidance", False)
    )
    args.srpo_euphonium_process_reward_apply_in_recovery = bool(
        config.get("srpo_euphonium_process_reward_apply_in_recovery", False)
    )
    args.srpo_euphonium_process_reward_detach_target = bool(
        config.get("srpo_euphonium_process_reward_detach_target", True)
    )
    args.srpo_euphonium_dual_reward_advantage_mode = str(
        config.get("srpo_euphonium_dual_reward_advantage_mode", "none")
    ).lower()
    args.srpo_euphonium_process_reward_advantage_coef = float(
        config.get("srpo_euphonium_process_reward_advantage_coef", 1.0)
    )
    args.srpo_euphonium_outcome_reward_advantage_coef = float(
        config.get("srpo_euphonium_outcome_reward_advantage_coef", 1.0)
    )
    args.srpo_euphonium_log_interval = int(
        config.get("srpo_euphonium_log_interval", 50)
    )

    if args.srpo_euphonium_process_reward_guidance_kl_beta <= 0.0:
        raise ValueError(
            "srpo_euphonium_process_reward_guidance_kl_beta must be > 0, got "
            f"{args.srpo_euphonium_process_reward_guidance_kl_beta}"
        )
    if args.srpo_euphonium_process_reward_guidance_eta < 0.0:
        raise ValueError(
            "srpo_euphonium_process_reward_guidance_eta must be >= 0, got "
            f"{args.srpo_euphonium_process_reward_guidance_eta}"
        )
    if args.srpo_euphonium_process_reward_start_step < 0:
        raise ValueError(
            "srpo_euphonium_process_reward_start_step must be >= 0, got "
            f"{args.srpo_euphonium_process_reward_start_step}"
        )
    if (
        args.srpo_euphonium_process_reward_end_step != -1
        and args.srpo_euphonium_process_reward_end_step
        < args.srpo_euphonium_process_reward_start_step
    ):
        raise ValueError(
            "srpo_euphonium_process_reward_end_step must be -1 or >= "
            f"srpo_euphonium_process_reward_start_step ({args.srpo_euphonium_process_reward_start_step})"
        )
    if args.srpo_euphonium_process_reward_interval <= 0:
        raise ValueError(
            "srpo_euphonium_process_reward_interval must be > 0, got "
            f"{args.srpo_euphonium_process_reward_interval}"
        )
    if (
        args.srpo_euphonium_process_reward_gradient_mode
        not in _ALLOWED_PROCESS_GRADIENT_MODES
    ):
        raise ValueError(
            "srpo_euphonium_process_reward_gradient_mode must be one of "
            f"{sorted(_ALLOWED_PROCESS_GRADIENT_MODES)}, got "
            f"{args.srpo_euphonium_process_reward_gradient_mode!r}"
        )
    if args.srpo_euphonium_process_reward_spsa_sigma <= 0.0:
        raise ValueError(
            "srpo_euphonium_process_reward_spsa_sigma must be > 0, got "
            f"{args.srpo_euphonium_process_reward_spsa_sigma}"
        )
    if args.srpo_euphonium_process_reward_spsa_num_samples <= 0:
        raise ValueError(
            "srpo_euphonium_process_reward_spsa_num_samples must be > 0, got "
            f"{args.srpo_euphonium_process_reward_spsa_num_samples}"
        )
    if (
        args.srpo_euphonium_dual_reward_advantage_mode
        not in _ALLOWED_DUAL_REWARD_MODES
    ):
        raise ValueError(
            "srpo_euphonium_dual_reward_advantage_mode must be one of "
            f"{sorted(_ALLOWED_DUAL_REWARD_MODES)}, got "
            f"{args.srpo_euphonium_dual_reward_advantage_mode!r}"
        )
    if args.srpo_euphonium_process_reward_model_type not in _ALLOWED_PROCESS_MODEL_TYPES:
        raise ValueError(
            "srpo_euphonium_process_reward_model_type must be one of "
            f"{sorted(_ALLOWED_PROCESS_MODEL_TYPES)}, got "
            f"{args.srpo_euphonium_process_reward_model_type!r}"
        )
    if (
        args.srpo_euphonium_process_reward_model_dtype
        not in _ALLOWED_PROCESS_MODEL_DTYPES
    ):
        raise ValueError(
            "srpo_euphonium_process_reward_model_dtype must be one of "
            f"{sorted(_ALLOWED_PROCESS_MODEL_DTYPES)}, got "
            f"{args.srpo_euphonium_process_reward_model_dtype!r}"
        )
    if (
        args.srpo_euphonium_process_reward_model_type == "torchscript"
        and args.srpo_euphonium_process_reward_model_path.strip() == ""
    ):
        raise ValueError(
            "srpo_euphonium_process_reward_model_path must be set when "
            "srpo_euphonium_process_reward_model_type='torchscript'."
        )
    if (
        args.srpo_euphonium_process_reward_model_type == "python_callable"
        and args.srpo_euphonium_process_reward_model_entry.strip() == ""
    ):
        raise ValueError(
            "srpo_euphonium_process_reward_model_entry must be set when "
            "srpo_euphonium_process_reward_model_type='python_callable'."
        )
    if args.srpo_euphonium_process_reward_advantage_coef < 0.0:
        raise ValueError(
            "srpo_euphonium_process_reward_advantage_coef must be >= 0, got "
            f"{args.srpo_euphonium_process_reward_advantage_coef}"
        )
    if args.srpo_euphonium_outcome_reward_advantage_coef < 0.0:
        raise ValueError(
            "srpo_euphonium_outcome_reward_advantage_coef must be >= 0, got "
            f"{args.srpo_euphonium_outcome_reward_advantage_coef}"
        )
    if args.srpo_euphonium_log_interval <= 0:
        raise ValueError(
            f"srpo_euphonium_log_interval must be > 0, got {args.srpo_euphonium_log_interval}"
        )
    needs_process_signal = (
        args.srpo_euphonium_process_reward_guidance_enabled
        or args.srpo_euphonium_dual_reward_advantage_mode in {"only", "both"}
    )
    if (
        needs_process_signal
        and args.srpo_euphonium_process_reward_model_type == "none"
        and not args.srpo_euphonium_process_reward_allow_proxy_fallback
    ):
        raise ValueError(
            "Euphonium process reward guidance/dual mode requires either "
            "a process reward model backend or "
            "srpo_euphonium_process_reward_allow_proxy_fallback=true."
        )
    if (
        needs_process_signal
        and args.srpo_euphonium_process_reward_gradient_mode == "spsa"
        and args.srpo_euphonium_process_reward_model_type == "none"
        and args.srpo_euphonium_process_reward_allow_proxy_fallback
    ):
        logger.warning(
            "srpo_euphonium_process_reward_gradient_mode='spsa' is enabled without "
            "an external process reward model; proxy guidance path will ignore SPSA and use legacy proxy gradient."
        )

    if not args.srpo_enable_euphonium:
        return args

    if not bool(getattr(args, "enable_srpo_training", False)):
        raise ValueError(
            "srpo_enable_euphonium requires enable_srpo_training=true."
        )

    if (
        args.srpo_euphonium_process_reward_guidance_enabled
        and args.srpo_euphonium_process_reward_guidance_scale == 0.0
    ):
        logger.warning(
            "srpo_euphonium_process_reward_guidance_enabled=true but "
            "srpo_euphonium_process_reward_guidance_scale=0.0; guidance has no effect."
        )

    srpo_batch_size = int(getattr(args, "srpo_batch_size", 1))
    if (
        args.srpo_euphonium_dual_reward_advantage_mode in {"only", "both"}
        and srpo_batch_size < 2
    ):
        logger.warning(
            "Euphonium dual reward mode '%s' works best with srpo_batch_size >= 2; "
            "current batch size is %d.",
            args.srpo_euphonium_dual_reward_advantage_mode,
            srpo_batch_size,
        )

    logger.info(
        "Euphonium SRPO integration enabled (guidance=%s process_model=%s dtype=%s "
        "allow_proxy_fallback=%s grad_mode=%s spsa_sigma=%.6f spsa_samples=%d "
        "scale=%.4f kl_beta=%.4f eta=%.4f "
        "window=[%d,%d] interval=%d normalize_grad=%s delta_t_scaling=%s "
        "recovery_guidance=%s detach_target=%s dual_mode=%s process_coef=%.3f outcome_coef=%.3f "
        "log_interval=%d).",
        args.srpo_euphonium_process_reward_guidance_enabled,
        args.srpo_euphonium_process_reward_model_type,
        args.srpo_euphonium_process_reward_model_dtype,
        args.srpo_euphonium_process_reward_allow_proxy_fallback,
        args.srpo_euphonium_process_reward_gradient_mode,
        args.srpo_euphonium_process_reward_spsa_sigma,
        args.srpo_euphonium_process_reward_spsa_num_samples,
        args.srpo_euphonium_process_reward_guidance_scale,
        args.srpo_euphonium_process_reward_guidance_kl_beta,
        args.srpo_euphonium_process_reward_guidance_eta,
        args.srpo_euphonium_process_reward_start_step,
        args.srpo_euphonium_process_reward_end_step,
        args.srpo_euphonium_process_reward_interval,
        args.srpo_euphonium_process_reward_normalize_gradient,
        args.srpo_euphonium_use_delta_t_for_guidance,
        args.srpo_euphonium_process_reward_apply_in_recovery,
        args.srpo_euphonium_process_reward_detach_target,
        args.srpo_euphonium_dual_reward_advantage_mode,
        args.srpo_euphonium_process_reward_advantage_coef,
        args.srpo_euphonium_outcome_reward_advantage_coef,
        args.srpo_euphonium_log_interval,
    )
    return args
