"""Configuration parsing for EquiVDM consistent noise."""

from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def parse_equivdm_noise_config(config: Dict[str, Any], args: Any) -> None:
    args.enable_equivdm_consistent_noise = bool(
        config.get("enable_equivdm_consistent_noise", False)
    )
    args.equivdm_noise_beta = float(config.get("equivdm_noise_beta", 0.9))
    if not (0.0 <= args.equivdm_noise_beta <= 1.0):
        raise ValueError(
            f"equivdm_noise_beta must be in [0, 1], got {args.equivdm_noise_beta}"
        )
    args.equivdm_flow_source = str(config.get("equivdm_flow_source", "batch"))
    if args.equivdm_flow_source not in {"batch"}:
        raise ValueError(
            f"equivdm_flow_source must be 'batch', got {args.equivdm_flow_source}"
        )
    args.equivdm_flow_key = str(config.get("equivdm_flow_key", "optical_flow"))
    args.equivdm_flow_frame_dim_in_batch = int(
        config.get("equivdm_flow_frame_dim_in_batch", 2)
    )
    if args.equivdm_flow_frame_dim_in_batch < 2:
        raise ValueError(
            "equivdm_flow_frame_dim_in_batch must be >= 2 for EquiVDM noise."
        )
    args.equivdm_flow_resize_mode = str(
        config.get("equivdm_flow_resize_mode", "none")
    )
    if args.equivdm_flow_resize_mode not in {"none", "bilinear", "nearest"}:
        raise ValueError(
            "equivdm_flow_resize_mode must be one of 'none', 'bilinear', 'nearest', "
            f"got {args.equivdm_flow_resize_mode}"
        )
    args.equivdm_flow_resize_align_corners = bool(
        config.get("equivdm_flow_resize_align_corners", True)
    )
    args.equivdm_flow_spatial_stride = int(
        config.get("equivdm_flow_spatial_stride", -1)
    )
    if args.equivdm_flow_spatial_stride == 0:
        raise ValueError("equivdm_flow_spatial_stride must be != 0")
    args.equivdm_warp_mode = str(config.get("equivdm_warp_mode", "grid"))
    if args.equivdm_warp_mode not in {"grid", "stochastic", "nte"}:
        raise ValueError(
            "equivdm_warp_mode must be one of 'grid', 'stochastic', 'nte', "
            f"got {args.equivdm_warp_mode}"
        )
    args.equivdm_warp_jitter = float(config.get("equivdm_warp_jitter", 0.5))
    if args.equivdm_warp_jitter < 0:
        raise ValueError(
            f"equivdm_warp_jitter must be >= 0, got {args.equivdm_warp_jitter}"
        )
    args.equivdm_nte_samples = int(config.get("equivdm_nte_samples", 4))
    if args.equivdm_nte_samples < 1:
        raise ValueError(
            f"equivdm_nte_samples must be >= 1, got {args.equivdm_nte_samples}"
        )
    args.equivdm_flow_quality_check = bool(
        config.get("equivdm_flow_quality_check", False)
    )
    args.equivdm_flow_max_magnitude = float(
        config.get("equivdm_flow_max_magnitude", 0.0)
    )
    if args.equivdm_flow_max_magnitude < 0:
        raise ValueError(
            "equivdm_flow_max_magnitude must be >= 0 (0 enables auto threshold)"
        )
    args.equivdm_flow_stride_check = bool(
        config.get("equivdm_flow_stride_check", False)
    )
    args.equivdm_flow_expected_stride = int(
        config.get("equivdm_flow_expected_stride", -1)
    )
    if args.equivdm_flow_expected_stride == 0:
        raise ValueError("equivdm_flow_expected_stride must be != 0")
    if args.enable_equivdm_consistent_noise:
        logger.info(
            "EquiVDM consistent noise enabled: beta=%.3f, flow_source=%s, flow_key=%s, frame_dim=%s, resize=%s, warp=%s",
            args.equivdm_noise_beta,
            args.equivdm_flow_source,
            args.equivdm_flow_key,
            args.equivdm_flow_frame_dim_in_batch,
            args.equivdm_flow_resize_mode,
            args.equivdm_warp_mode,
        )
