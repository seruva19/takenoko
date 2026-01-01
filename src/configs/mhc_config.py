"""mHC-LoRA configuration parsing helpers."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional


def parse_mhc_config(
    config: Dict[str, Any], args: argparse.Namespace, logger: Any
) -> None:
    """Parse and validate mHC-LoRA settings, updating args and network_args."""

    def _parse_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        try:
            return float(value)
        except Exception:
            return None

    args.mhc_num_paths = int(config.get("mhc_num_paths", 2))
    args.mhc_sinkhorn_iters = int(config.get("mhc_sinkhorn_iters", 20))
    args.mhc_mixing_init = str(config.get("mhc_mixing_init", "identity"))
    args.mhc_mixing_strength = float(config.get("mhc_mixing_strength", 1.0))
    args.mhc_mixing_strength_end = _parse_optional_float(
        config.get("mhc_mixing_strength_end")
    )
    args.mhc_mixing_temperature = float(config.get("mhc_mixing_temperature", 1.0))
    args.mhc_mixing_temperature_end = _parse_optional_float(
        config.get("mhc_mixing_temperature_end")
    )
    args.mhc_mixing_schedule_steps = int(
        config.get("mhc_mixing_schedule_steps", 0)
    )
    args.mhc_output_stream = int(config.get("mhc_output_stream", 0))
    args.mhc_output_mode = str(config.get("mhc_output_mode", "stream"))
    args.mhc_nonneg_mixing = bool(config.get("mhc_nonneg_mixing", True))
    args.mhc_dynamic_mixing = bool(config.get("mhc_dynamic_mixing", False))
    args.mhc_dynamic_hidden_dim = int(config.get("mhc_dynamic_hidden_dim", 0))
    args.mhc_dynamic_scale = float(config.get("mhc_dynamic_scale", 1.0))
    args.mhc_dynamic_share = str(config.get("mhc_dynamic_share", "none"))
    args.mhc_timestep_mixing = bool(config.get("mhc_timestep_mixing", False))
    args.mhc_timestep_max = int(config.get("mhc_timestep_max", 1000))
    args.mhc_timestep_gamma = float(config.get("mhc_timestep_gamma", 1.0))
    args.mhc_timestep_strength_min = float(
        config.get("mhc_timestep_strength_min", 0.0)
    )
    args.mhc_path_scale_init = float(config.get("mhc_path_scale_init", 1.0))
    args.mhc_path_scale_trainable = bool(
        config.get("mhc_path_scale_trainable", True)
    )
    args.mhc_path_dropout = _parse_optional_float(config.get("mhc_path_dropout"))
    args.mhc_freeze_mixing_steps = int(config.get("mhc_freeze_mixing_steps", 0))
    args.mhc_identity_clamp_steps = int(config.get("mhc_identity_clamp_steps", 0))
    args.mhc_identity_clamp_max_offdiag = float(
        config.get("mhc_identity_clamp_max_offdiag", 0.0)
    )
    args.mhc_identity_reg_lambda = float(config.get("mhc_identity_reg_lambda", 0.0))
    args.mhc_identity_reg_warmup_steps = int(
        config.get("mhc_identity_reg_warmup_steps", 0)
    )
    args.mhc_identity_reg_power = float(config.get("mhc_identity_reg_power", 1.0))
    args.mhc_entropy_reg_lambda = float(config.get("mhc_entropy_reg_lambda", 0.0))
    args.mhc_entropy_reg_target = _parse_optional_float(
        config.get("mhc_entropy_reg_target")
    )
    args.mhc_mix_log_interval = int(config.get("mhc_mix_log_interval", 100))
    args.mhc_mix_histogram_interval = int(
        config.get("mhc_mix_histogram_interval", 500)
    )
    args.mhc_mix_warn_entropy_min = float(
        config.get("mhc_mix_warn_entropy_min", 0.05)
    )
    args.mhc_mix_warn_offdiag_max = float(
        config.get("mhc_mix_warn_offdiag_max", 0.5)
    )

    if args.mhc_num_paths < 1:
        raise ValueError("mhc_num_paths must be >= 1")
    if args.mhc_sinkhorn_iters < 1:
        raise ValueError("mhc_sinkhorn_iters must be >= 1")
    if not 0.0 <= args.mhc_mixing_strength <= 1.0:
        raise ValueError("mhc_mixing_strength must be between 0 and 1")
    if args.mhc_mixing_strength_end is not None and not 0.0 <= args.mhc_mixing_strength_end <= 1.0:
        raise ValueError("mhc_mixing_strength_end must be between 0 and 1")
    if args.mhc_mixing_temperature <= 0.0:
        raise ValueError("mhc_mixing_temperature must be > 0")
    if args.mhc_mixing_temperature_end is not None and args.mhc_mixing_temperature_end <= 0.0:
        raise ValueError("mhc_mixing_temperature_end must be > 0")
    if args.mhc_mixing_schedule_steps < 0:
        raise ValueError("mhc_mixing_schedule_steps must be >= 0")
    valid_mhc_inits = {"identity", "uniform", "random"}
    if args.mhc_mixing_init not in valid_mhc_inits:
        raise ValueError(
            f"mhc_mixing_init must be one of {sorted(valid_mhc_inits)}"
        )
    valid_mhc_output_modes = {"stream", "sum", "mean"}
    if args.mhc_output_mode not in valid_mhc_output_modes:
        raise ValueError(
            f"mhc_output_mode must be one of {sorted(valid_mhc_output_modes)}"
        )
    valid_mhc_dynamic_share = {"none", "layer"}
    if args.mhc_dynamic_share not in valid_mhc_dynamic_share:
        raise ValueError(
            f"mhc_dynamic_share must be one of {sorted(valid_mhc_dynamic_share)}"
        )
    if args.mhc_dynamic_hidden_dim < 0:
        raise ValueError("mhc_dynamic_hidden_dim must be >= 0")
    if args.mhc_dynamic_scale < 0.0:
        raise ValueError("mhc_dynamic_scale must be >= 0")
    if args.mhc_timestep_max <= 0:
        raise ValueError("mhc_timestep_max must be > 0")
    if args.mhc_timestep_gamma < 0.0:
        raise ValueError("mhc_timestep_gamma must be >= 0")
    if not 0.0 <= args.mhc_timestep_strength_min <= 1.0:
        raise ValueError("mhc_timestep_strength_min must be between 0 and 1")
    if args.mhc_path_scale_init <= 0.0:
        raise ValueError("mhc_path_scale_init must be > 0")
    if args.mhc_path_dropout is not None and not 0.0 <= args.mhc_path_dropout <= 1.0:
        raise ValueError("mhc_path_dropout must be between 0 and 1")
    if args.mhc_freeze_mixing_steps < 0:
        raise ValueError("mhc_freeze_mixing_steps must be >= 0")
    if args.mhc_identity_clamp_steps < 0:
        raise ValueError("mhc_identity_clamp_steps must be >= 0")
    if not 0.0 <= args.mhc_identity_clamp_max_offdiag <= 1.0:
        raise ValueError("mhc_identity_clamp_max_offdiag must be between 0 and 1")
    if args.mhc_identity_reg_lambda < 0.0:
        raise ValueError("mhc_identity_reg_lambda must be >= 0")
    if args.mhc_identity_reg_warmup_steps < 0:
        raise ValueError("mhc_identity_reg_warmup_steps must be >= 0")
    if args.mhc_identity_reg_power < 0.0:
        raise ValueError("mhc_identity_reg_power must be >= 0")
    if args.mhc_entropy_reg_lambda < 0.0:
        raise ValueError("mhc_entropy_reg_lambda must be >= 0")
    if args.mhc_mix_log_interval < 1:
        raise ValueError("mhc_mix_log_interval must be >= 1")
    if args.mhc_mix_histogram_interval < 1:
        raise ValueError("mhc_mix_histogram_interval must be >= 1")
    if args.mhc_mix_warn_entropy_min < 0.0:
        raise ValueError("mhc_mix_warn_entropy_min must be >= 0")
    if not 0.0 <= args.mhc_mix_warn_offdiag_max <= 1.0:
        raise ValueError("mhc_mix_warn_offdiag_max must be between 0 and 1")
    if args.mhc_output_stream < 0 or args.mhc_output_stream >= args.mhc_num_paths:
        logger.warning(
            "mhc_output_stream=%s is out of range for mhc_num_paths=%s; using 0",
            args.mhc_output_stream,
            args.mhc_num_paths,
        )
        args.mhc_output_stream = 0

    if getattr(args, "network_module", "") == "networks.mhc_lora":
        logger.info(
            "mHC-LoRA enabled (paths=%s, sinkhorn_iters=%s, init=%s, strength=%.3f)",
            args.mhc_num_paths,
            args.mhc_sinkhorn_iters,
            args.mhc_mixing_init,
            args.mhc_mixing_strength,
        )
        if not hasattr(args, "network_args") or args.network_args is None:
            args.network_args = []
        existing_keys = {
            net_arg.split("=", 1)[0].strip()
            for net_arg in args.network_args
            if isinstance(net_arg, str) and "=" in net_arg
        }

        def _append_network_arg(key: str, value: Any) -> None:
            if key in existing_keys:
                return
            if isinstance(value, bool):
                val_str = "true" if value else "false"
            else:
                val_str = str(value)
            args.network_args.append(f"{key}={val_str}")

        _append_network_arg("mhc_num_paths", args.mhc_num_paths)
        _append_network_arg("mhc_sinkhorn_iters", args.mhc_sinkhorn_iters)
        _append_network_arg("mhc_mixing_init", args.mhc_mixing_init)
        _append_network_arg("mhc_mixing_strength", args.mhc_mixing_strength)
        if args.mhc_mixing_strength_end is not None:
            _append_network_arg("mhc_mixing_strength_end", args.mhc_mixing_strength_end)
        _append_network_arg("mhc_mixing_temperature", args.mhc_mixing_temperature)
        if args.mhc_mixing_temperature_end is not None:
            _append_network_arg(
                "mhc_mixing_temperature_end", args.mhc_mixing_temperature_end
            )
        _append_network_arg("mhc_mixing_schedule_steps", args.mhc_mixing_schedule_steps)
        _append_network_arg("mhc_output_stream", args.mhc_output_stream)
        _append_network_arg("mhc_output_mode", args.mhc_output_mode)
        _append_network_arg("mhc_nonneg_mixing", args.mhc_nonneg_mixing)
        _append_network_arg("mhc_dynamic_mixing", args.mhc_dynamic_mixing)
        _append_network_arg("mhc_dynamic_hidden_dim", args.mhc_dynamic_hidden_dim)
        _append_network_arg("mhc_dynamic_scale", args.mhc_dynamic_scale)
        _append_network_arg("mhc_dynamic_share", args.mhc_dynamic_share)
        _append_network_arg("mhc_timestep_mixing", args.mhc_timestep_mixing)
        _append_network_arg("mhc_timestep_max", args.mhc_timestep_max)
        _append_network_arg("mhc_timestep_gamma", args.mhc_timestep_gamma)
        _append_network_arg("mhc_timestep_strength_min", args.mhc_timestep_strength_min)
        _append_network_arg("mhc_path_scale_init", args.mhc_path_scale_init)
        _append_network_arg(
            "mhc_path_scale_trainable", args.mhc_path_scale_trainable
        )
        if args.mhc_path_dropout is not None:
            _append_network_arg("mhc_path_dropout", args.mhc_path_dropout)
        _append_network_arg("mhc_freeze_mixing_steps", args.mhc_freeze_mixing_steps)
        _append_network_arg("mhc_identity_clamp_steps", args.mhc_identity_clamp_steps)
        _append_network_arg(
            "mhc_identity_clamp_max_offdiag", args.mhc_identity_clamp_max_offdiag
        )
        _append_network_arg("mhc_identity_reg_lambda", args.mhc_identity_reg_lambda)
        _append_network_arg(
            "mhc_identity_reg_warmup_steps", args.mhc_identity_reg_warmup_steps
        )
        _append_network_arg("mhc_identity_reg_power", args.mhc_identity_reg_power)
        _append_network_arg("mhc_entropy_reg_lambda", args.mhc_entropy_reg_lambda)
        if args.mhc_entropy_reg_target is not None:
            _append_network_arg("mhc_entropy_reg_target", args.mhc_entropy_reg_target)
        _append_network_arg("mhc_mix_log_interval", args.mhc_mix_log_interval)
        _append_network_arg(
            "mhc_mix_histogram_interval", args.mhc_mix_histogram_interval
        )
        _append_network_arg("mhc_mix_warn_entropy_min", args.mhc_mix_warn_entropy_min)
        _append_network_arg("mhc_mix_warn_offdiag_max", args.mhc_mix_warn_offdiag_max)
