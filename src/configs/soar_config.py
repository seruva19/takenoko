from __future__ import annotations

from typing import Any, Dict


_ALLOWED_SDE_ROLLOUT_TYPES = {"simple", "sde", "flow_sde", "cps"}


def apply_soar_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse HY-SOAR settings for the standard LoRA SFT path."""

    args.enable_soar = bool(config.get("enable_soar", False))
    args.soar_lambda_aux = float(config.get("soar_lambda_aux", 1.0))
    args.soar_num_rollout_paths = int(config.get("soar_num_rollout_paths", 1))
    args.soar_aux_points_per_path = int(config.get("soar_aux_points_per_path", 6))
    args.soar_rollout_step_count = int(config.get("soar_rollout_step_count", 40))
    args.soar_rollout_cfg_scale = float(config.get("soar_rollout_cfg_scale", 4.5))
    args.soar_use_same_noise = bool(config.get("soar_use_same_noise", True))
    args.soar_enable_sde_branch = bool(config.get("soar_enable_sde_branch", False))
    args.soar_sde_rollout_type = str(
        config.get("soar_sde_rollout_type", "flow_sde")
    ).lower()
    args.soar_sde_noise_scale = float(config.get("soar_sde_noise_scale", 0.5))
    args.soar_uncond_context_mode = str(
        config.get("soar_uncond_context_mode", "zero")
    ).lower()
    args.soar_log_interval = int(config.get("soar_log_interval", 100))

    if args.soar_lambda_aux < 0.0:
        raise ValueError("soar_lambda_aux must be >= 0")
    if args.soar_num_rollout_paths < 1:
        raise ValueError("soar_num_rollout_paths must be >= 1")
    if args.soar_aux_points_per_path < 0:
        raise ValueError("soar_aux_points_per_path must be >= 0")
    if args.soar_rollout_step_count < 1:
        raise ValueError("soar_rollout_step_count must be >= 1")
    if args.soar_rollout_cfg_scale < 0.0:
        raise ValueError("soar_rollout_cfg_scale must be >= 0")
    if args.soar_sde_noise_scale < 0.0:
        raise ValueError("soar_sde_noise_scale must be >= 0")
    if args.soar_sde_rollout_type not in _ALLOWED_SDE_ROLLOUT_TYPES:
        raise ValueError(
            "soar_sde_rollout_type must be one of "
            f"{sorted(_ALLOWED_SDE_ROLLOUT_TYPES)}"
        )
    if args.soar_uncond_context_mode != "zero":
        raise ValueError(
            "soar_uncond_context_mode must be 'zero' in the first HY-SOAR pass"
        )
    if args.soar_log_interval <= 0:
        raise ValueError("soar_log_interval must be > 0")
    if not args.soar_enable_sde_branch and args.soar_num_rollout_paths != 1:
        raise ValueError(
            "soar_num_rollout_paths must be 1 when soar_enable_sde_branch is false"
        )

    if args.enable_soar:
        logger.info(
            "HY-SOAR enabled: lambda_aux=%.3f paths=%d aux_points=%d rollout_steps=%d "
            "cfg_scale=%.3f same_noise=%s sde_enabled=%s sde_type=%s "
            "sde_noise_scale=%.3f uncond_mode=%s log_interval=%d",
            args.soar_lambda_aux,
            args.soar_num_rollout_paths,
            args.soar_aux_points_per_path,
            args.soar_rollout_step_count,
            args.soar_rollout_cfg_scale,
            str(args.soar_use_same_noise).lower(),
            str(args.soar_enable_sde_branch).lower(),
            args.soar_sde_rollout_type,
            args.soar_sde_noise_scale,
            args.soar_uncond_context_mode,
            args.soar_log_interval,
        )
