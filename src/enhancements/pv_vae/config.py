"""Configuration parsing for PV-VAE predictive reconstruction."""

from __future__ import annotations

from typing import Any, Dict


def _parse_float(raw: Any, key: str) -> float:
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float, got {raw!r}") from exc


def _parse_int(raw: Any, key: str) -> int:
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def apply_pv_vae_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse Predictive Video VAE objective settings."""

    args.enable_pv_vae = bool(config.get("enable_pv_vae", False))
    args.pv_vae_temporal_compression = _parse_int(
        config.get("pv_vae_temporal_compression", 4),
        "pv_vae_temporal_compression",
    )
    args.pv_vae_max_drop_ratio = _parse_float(
        config.get("pv_vae_max_drop_ratio", 1.0),
        "pv_vae_max_drop_ratio",
    )
    args.pv_vae_min_observed_groups = _parse_int(
        config.get("pv_vae_min_observed_groups", 1),
        "pv_vae_min_observed_groups",
    )
    args.pv_vae_min_drop_groups = _parse_int(
        config.get("pv_vae_min_drop_groups", 1),
        "pv_vae_min_drop_groups",
    )
    args.pv_vae_padding_mode = str(
        config.get("pv_vae_padding_mode", "gaussian")
    ).lower()
    args.pv_vae_temporal_diff_weight = _parse_float(
        config.get("pv_vae_temporal_diff_weight", 0.0),
        "pv_vae_temporal_diff_weight",
    )
    args.pv_vae_temporal_diff_loss = str(
        config.get("pv_vae_temporal_diff_loss", "l1")
    ).lower()
    args.pv_vae_apply_in_validation = bool(
        config.get("pv_vae_apply_in_validation", False)
    )

    if args.pv_vae_temporal_compression < 1:
        raise ValueError("pv_vae_temporal_compression must be >= 1")
    if not (0.0 <= args.pv_vae_max_drop_ratio <= 1.0):
        raise ValueError("pv_vae_max_drop_ratio must be in [0, 1]")
    if args.pv_vae_min_observed_groups < 1:
        raise ValueError("pv_vae_min_observed_groups must be >= 1")
    if args.pv_vae_min_drop_groups < 0:
        raise ValueError("pv_vae_min_drop_groups must be >= 0")
    if args.pv_vae_padding_mode not in {"gaussian", "zeros"}:
        raise ValueError("pv_vae_padding_mode must be one of: gaussian, zeros")
    if args.pv_vae_temporal_diff_weight < 0.0:
        raise ValueError("pv_vae_temporal_diff_weight must be >= 0")
    if args.pv_vae_temporal_diff_loss not in {"l1", "mse", "huber"}:
        raise ValueError("pv_vae_temporal_diff_loss must be one of: l1, mse, huber")

    if args.enable_pv_vae:
        network_module = str(getattr(args, "network_module", ""))
        if network_module != "networks.vae_wan":
            raise ValueError(
                "enable_pv_vae requires network_module='networks.vae_wan'"
            )
        if args.pv_vae_max_drop_ratio <= 0.0:
            raise ValueError(
                "pv_vae_max_drop_ratio must be > 0 when enable_pv_vae is true"
            )
        if str(getattr(args, "vae_training_mode", "full")).lower() == "decoder_only":
            logger.warning(
                "PV-VAE is enabled in decoder_only mode. The PV-VAE paper uses a separate decoder fine-tuning stage with predictive dropping disabled; consider setting enable_pv_vae=false for that stage."
            )
        logger.info(
            "PV-VAE enabled (temporal_compression=%s, max_drop_ratio=%.3f, min_observed_groups=%s, min_drop_groups=%s, padding=%s, temporal_diff_weight=%.4f, validation=%s).",
            args.pv_vae_temporal_compression,
            args.pv_vae_max_drop_ratio,
            args.pv_vae_min_observed_groups,
            args.pv_vae_min_drop_groups,
            args.pv_vae_padding_mode,
            args.pv_vae_temporal_diff_weight,
            args.pv_vae_apply_in_validation,
        )
