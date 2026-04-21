from __future__ import annotations

from typing import Any, Dict


_FLOWC2S_TRANSPORT_LOSSES = {"cosine", "mse"}


def apply_flowc2s_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse FlowC2S-inspired current-to-succeeding transport settings."""
    args.enable_flowc2s_transport = bool(config.get("enable_flowc2s_transport", False))
    args.flowc2s_transport_block_index = config.get("flowc2s_transport_block_index", 8)
    if args.flowc2s_transport_block_index is not None:
        args.flowc2s_transport_block_index = int(args.flowc2s_transport_block_index)
    args.flowc2s_transport_lambda = float(
        config.get("flowc2s_transport_lambda", 0.1)
    )
    args.flowc2s_transport_loss_type = str(
        config.get("flowc2s_transport_loss_type", "cosine")
    ).lower()
    args.flowc2s_transport_chunk_ratio = float(
        config.get("flowc2s_transport_chunk_ratio", 0.5)
    )
    args.flowc2s_transport_min_chunk_frames = int(
        config.get("flowc2s_transport_min_chunk_frames", 2)
    )
    args.flowc2s_transport_normalize_latents = bool(
        config.get("flowc2s_transport_normalize_latents", True)
    )

    if args.flowc2s_transport_loss_type not in _FLOWC2S_TRANSPORT_LOSSES:
        raise ValueError(
            "flowc2s_transport_loss_type must be one of "
            f"{sorted(_FLOWC2S_TRANSPORT_LOSSES)}"
        )
    if not (0.0 < args.flowc2s_transport_chunk_ratio < 1.0):
        raise ValueError("flowc2s_transport_chunk_ratio must be in (0.0, 1.0)")
    if args.flowc2s_transport_lambda < 0.0:
        raise ValueError("flowc2s_transport_lambda must be >= 0")
    if args.flowc2s_transport_min_chunk_frames < 1:
        raise ValueError("flowc2s_transport_min_chunk_frames must be >= 1")

    if args.enable_flowc2s_transport:
        if args.flowc2s_transport_block_index is None:
            raise ValueError(
                "flowc2s_transport_block_index must be set when "
                "enable_flowc2s_transport=true"
            )
        if args.flowc2s_transport_lambda <= 0.0:
            raise ValueError(
                "flowc2s_transport_lambda must be > 0 when "
                "enable_flowc2s_transport=true"
            )
        logger.info(
            "FlowC2S transport enabled: block=%s lambda=%.4f loss=%s "
            "chunk_ratio=%.3f min_chunk_frames=%d normalize_latents=%s",
            args.flowc2s_transport_block_index,
            args.flowc2s_transport_lambda,
            args.flowc2s_transport_loss_type,
            args.flowc2s_transport_chunk_ratio,
            args.flowc2s_transport_min_chunk_frames,
            str(args.flowc2s_transport_normalize_latents).lower(),
        )
