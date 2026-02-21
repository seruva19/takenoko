"""S2D (Selective Spectral Decay) config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


def apply_s2d_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.enable_s2d = bool(config.get("enable_s2d", False))
    args.s2d_decay_lambda = float(config.get("s2d_decay_lambda", 5e-4))
    args.s2d_power_n = float(config.get("s2d_power_n", 2.0))
    args.s2d_pcdr_tau = float(config.get("s2d_pcdr_tau", 0.95))
    args.s2d_k_max = int(config.get("s2d_k_max", 3))
    args.s2d_update_interval = int(config.get("s2d_update_interval", 100))
    args.s2d_max_modules = int(config.get("s2d_max_modules", 0))
    args.s2d_include_text_encoder = bool(
        config.get("s2d_include_text_encoder", False)
    )
    args.s2d_use_abs_response = bool(config.get("s2d_use_abs_response", True))
    args.s2d_selection_mode = str(
        config.get("s2d_selection_mode", "pcdr_activation")
    )
    args.s2d_pcdr_variant = str(config.get("s2d_pcdr_variant", "outlier_neuron"))
    args.s2d_activation_samples = int(config.get("s2d_activation_samples", 512))
    args.s2d_hook_batch_samples = int(config.get("s2d_hook_batch_samples", 64))
    args.s2d_selection_fallback_to_kmax = bool(
        config.get("s2d_selection_fallback_to_kmax", False)
    )

    if args.s2d_decay_lambda < 0.0:
        raise ValueError("s2d_decay_lambda must be >= 0")
    if args.s2d_power_n <= 1.0:
        raise ValueError("s2d_power_n must be > 1.0")
    if args.s2d_pcdr_tau <= 0.0 or args.s2d_pcdr_tau > 1.0:
        raise ValueError("s2d_pcdr_tau must be in (0, 1]")
    if args.s2d_k_max < 1:
        raise ValueError("s2d_k_max must be >= 1")
    if args.s2d_update_interval < 1:
        raise ValueError("s2d_update_interval must be >= 1")
    if args.s2d_max_modules < 0:
        raise ValueError("s2d_max_modules must be >= 0")
    if args.s2d_activation_samples < 1:
        raise ValueError("s2d_activation_samples must be >= 1")
    if args.s2d_hook_batch_samples < 1:
        raise ValueError("s2d_hook_batch_samples must be >= 1")

    args.s2d_selection_mode = args.s2d_selection_mode.strip().lower()
    if args.s2d_selection_mode not in {
        "pcdr_activation",
        "sigma_mass",
        "pcdr_dense_full_svd",
    }:
        raise ValueError(
            "s2d_selection_mode must be one of: pcdr_activation, sigma_mass, pcdr_dense_full_svd"
        )

    args.s2d_pcdr_variant = args.s2d_pcdr_variant.strip().lower()
    if args.s2d_pcdr_variant not in {"outlier_neuron", "projection_mass"}:
        raise ValueError(
            "s2d_pcdr_variant must be one of: outlier_neuron, projection_mass"
        )

    if args.enable_s2d:
        logger.info(
            "S2D enabled (lambda=%.6f, n=%.3f, tau=%.3f, k_max=%d, interval=%d, max_modules=%d, include_text_encoder=%s, abs_response=%s, mode=%s, pcdr_variant=%s, act_samples=%d, hook_samples=%d, fallback_to_kmax=%s).",
            args.s2d_decay_lambda,
            args.s2d_power_n,
            args.s2d_pcdr_tau,
            args.s2d_k_max,
            args.s2d_update_interval,
            args.s2d_max_modules,
            args.s2d_include_text_encoder,
            args.s2d_use_abs_response,
            args.s2d_selection_mode,
            args.s2d_pcdr_variant,
            args.s2d_activation_samples,
            args.s2d_hook_batch_samples,
            args.s2d_selection_fallback_to_kmax,
        )
