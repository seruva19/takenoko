from __future__ import annotations

from typing import Any, Dict


def apply_deco_frequency_diagnostics_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse DeCo-inspired frequency decoupling diagnostic settings."""
    args.enable_frequency_decoupling_diagnostics = bool(
        config.get("enable_frequency_decoupling_diagnostics", False)
    )
    args.frequency_decoupling_block_size = int(
        config.get("frequency_decoupling_block_size", 8)
    )
    args.frequency_decoupling_low_freq_ratio = float(
        config.get("frequency_decoupling_low_freq_ratio", 0.5)
    )

    if args.frequency_decoupling_block_size < 2:
        raise ValueError("frequency_decoupling_block_size must be >= 2")
    if not (0.0 < args.frequency_decoupling_low_freq_ratio <= 1.0):
        raise ValueError("frequency_decoupling_low_freq_ratio must be in (0, 1]")

    if args.enable_frequency_decoupling_diagnostics:
        logger.info(
            "DeCo frequency diagnostics enabled: block_size=%d, low_freq_ratio=%.3f",
            args.frequency_decoupling_block_size,
            args.frequency_decoupling_low_freq_ratio,
        )
