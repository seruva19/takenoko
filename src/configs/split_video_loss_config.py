from __future__ import annotations

from typing import Any, Dict


def apply_split_video_loss_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse opt-in anchor-vs-temporal video loss reduction settings."""
    args.enable_split_video_loss = bool(config.get("enable_split_video_loss", False))
    args.split_video_loss_anchor_weight = float(
        config.get("split_video_loss_anchor_weight", 1.0)
    )
    args.split_video_loss_temporal_weight = float(
        config.get("split_video_loss_temporal_weight", 1.0)
    )

    if args.split_video_loss_anchor_weight < 0.0:
        raise ValueError("split_video_loss_anchor_weight must be >= 0")
    if args.split_video_loss_temporal_weight < 0.0:
        raise ValueError("split_video_loss_temporal_weight must be >= 0")
    if (
        args.split_video_loss_anchor_weight <= 0.0
        and args.split_video_loss_temporal_weight <= 0.0
    ):
        raise ValueError(
            "split_video_loss_anchor_weight and split_video_loss_temporal_weight "
            "cannot both be <= 0"
        )

    if args.enable_split_video_loss:
        logger.info(
            "Split video loss enabled: anchor_weight=%.3f, temporal_weight=%.3f",
            args.split_video_loss_anchor_weight,
            args.split_video_loss_temporal_weight,
        )
