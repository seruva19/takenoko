"""CDC-FM config parsing helpers."""

from __future__ import annotations

from typing import Any


def parse_cdc_config(config: dict, args: Any, logger) -> None:
    args.enable_cdc_fm = bool(config.get("enable_cdc_fm", False))
    args.cdc_k_neighbors = int(config.get("cdc_k_neighbors", 256))
    args.cdc_k_bandwidth = int(config.get("cdc_k_bandwidth", 8))
    args.cdc_d_cdc = int(config.get("cdc_d_cdc", 8))
    args.cdc_gamma = float(config.get("cdc_gamma", 1.0))
    args.cdc_min_bucket_size = int(config.get("cdc_min_bucket_size", 16))
    args.cdc_force_recache = bool(config.get("cdc_force_recache", False))

    if args.cdc_k_neighbors < 2:
        raise ValueError(
            f"cdc_k_neighbors must be >= 2, got {args.cdc_k_neighbors}"
        )
    if args.cdc_k_bandwidth < 1:
        raise ValueError(
            f"cdc_k_bandwidth must be >= 1, got {args.cdc_k_bandwidth}"
        )
    if args.cdc_d_cdc < 1:
        raise ValueError(f"cdc_d_cdc must be >= 1, got {args.cdc_d_cdc}")
    if args.cdc_gamma <= 0:
        raise ValueError(f"cdc_gamma must be > 0, got {args.cdc_gamma}")
    if args.cdc_min_bucket_size < 2:
        raise ValueError(
            f"cdc_min_bucket_size must be >= 2, got {args.cdc_min_bucket_size}"
        )
    if args.enable_cdc_fm:
        logger.info(
            "CDC-FM enabled (k=%d, bandwidth=%d, d_cdc=%d, gamma=%.3f, min_bucket=%d, force_recache=%s)",
            args.cdc_k_neighbors,
            args.cdc_k_bandwidth,
            args.cdc_d_cdc,
            args.cdc_gamma,
            args.cdc_min_bucket_size,
            args.cdc_force_recache,
        )
