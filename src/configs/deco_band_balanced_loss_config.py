from __future__ import annotations

from typing import Any, Dict


def apply_deco_band_balanced_loss_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse DeCo-inspired band-balanced reconstruction loss settings."""
    args.enable_band_balanced_loss = bool(
        config.get("enable_band_balanced_loss", False)
    )
    args.band_balanced_loss_weight = float(
        config.get("band_balanced_loss_weight", 0.05)
    )
    args.band_balanced_low_freq_weight = float(
        config.get("band_balanced_low_freq_weight", 1.0)
    )
    args.band_balanced_high_freq_weight = float(
        config.get("band_balanced_high_freq_weight", 1.0)
    )
    args.band_balanced_block_size = int(config.get("band_balanced_block_size", 8))
    args.band_balanced_low_freq_ratio = float(
        config.get("band_balanced_low_freq_ratio", 0.5)
    )

    if args.band_balanced_loss_weight < 0.0:
        raise ValueError("band_balanced_loss_weight must be >= 0")
    if args.band_balanced_low_freq_weight < 0.0:
        raise ValueError("band_balanced_low_freq_weight must be >= 0")
    if args.band_balanced_high_freq_weight < 0.0:
        raise ValueError("band_balanced_high_freq_weight must be >= 0")
    if args.band_balanced_block_size < 2:
        raise ValueError("band_balanced_block_size must be >= 2")
    if not (0.0 < args.band_balanced_low_freq_ratio <= 1.0):
        raise ValueError("band_balanced_low_freq_ratio must be in (0, 1]")
    if (
        args.band_balanced_low_freq_weight <= 0.0
        and args.band_balanced_high_freq_weight <= 0.0
    ):
        raise ValueError(
            "band_balanced_low_freq_weight and band_balanced_high_freq_weight "
            "cannot both be <= 0"
        )
    if args.enable_band_balanced_loss and args.band_balanced_loss_weight <= 0.0:
        raise ValueError(
            "band_balanced_loss_weight must be > 0 when enable_band_balanced_loss "
            "is true"
        )

    if args.enable_band_balanced_loss:
        logger.info(
            "DeCo band-balanced loss enabled: weight=%.4f, low_weight=%.3f, "
            "high_weight=%.3f, block_size=%d, low_freq_ratio=%.3f",
            args.band_balanced_loss_weight,
            args.band_balanced_low_freq_weight,
            args.band_balanced_high_freq_weight,
            args.band_balanced_block_size,
            args.band_balanced_low_freq_ratio,
        )
