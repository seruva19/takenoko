from __future__ import annotations

from typing import Any, Dict, List


_ALLOWED_PROJECTORS = {"mlp", "dit"}
_ALLOWED_LOSSES = {"negative_cosine", "cosine", "mse", "smooth_l1"}


def _parse_alignment_depths(config: Dict[str, Any]) -> List[int]:
    default_depth = int(config.get("mirai_foresight_alignment_depth", 18))
    raw_depths = config.get("mirai_foresight_alignment_depths", None)
    if raw_depths is None:
        return [default_depth]
    if not isinstance(raw_depths, (list, tuple)):
        raise ValueError(
            "mirai_foresight_alignment_depths must be a list of ints or omitted, "
            f"got {type(raw_depths).__name__}"
        )
    if len(raw_depths) == 0:
        raise ValueError(
            "mirai_foresight_alignment_depths must not be empty when provided"
        )
    return list(dict.fromkeys(int(value) for value in raw_depths))


def apply_mirai_foresight_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse training-only Mirai Foresight configuration."""
    args.enable_mirai_foresight = bool(config.get("enable_mirai_foresight", False))
    args.mirai_foresight_alignment_depths = _parse_alignment_depths(config)
    args.mirai_foresight_alignment_depth = int(
        args.mirai_foresight_alignment_depths[0]
    )
    args.mirai_foresight_loss_lambda = float(
        config.get("mirai_foresight_loss_lambda", 0.1)
    )
    args.mirai_foresight_delta = int(config.get("mirai_foresight_delta", 1))
    args.mirai_foresight_num_frame_per_block = int(
        config.get("mirai_foresight_num_frame_per_block", 1)
    )
    args.mirai_foresight_include_current = bool(
        config.get("mirai_foresight_include_current", False)
    )
    args.mirai_foresight_delta_mean_pool = bool(
        config.get("mirai_foresight_delta_mean_pool", False)
    )
    args.mirai_foresight_projector_type = str(
        config.get("mirai_foresight_projector_type", "mlp")
    ).lower()
    args.mirai_foresight_projector_layers = int(
        config.get("mirai_foresight_projector_layers", 2)
    )
    args.mirai_foresight_projector_hidden_dim = int(
        config.get("mirai_foresight_projector_hidden_dim", 0)
    )
    args.mirai_foresight_ffn_ratio = float(
        config.get("mirai_foresight_ffn_ratio", 4.0)
    )
    args.mirai_foresight_num_heads = int(
        config.get("mirai_foresight_num_heads", 0)
    )
    args.mirai_foresight_loss_type = str(
        config.get("mirai_foresight_loss_type", "negative_cosine")
    ).lower()
    args.mirai_foresight_max_spatial_tokens = int(
        config.get("mirai_foresight_max_spatial_tokens", 256)
    )
    args.mirai_foresight_detach_teacher = bool(
        config.get("mirai_foresight_detach_teacher", True)
    )
    args.mirai_foresight_projector_lr_ratio = float(
        config.get("mirai_foresight_projector_lr_ratio", 1.0)
    )

    if args.mirai_foresight_projector_type not in _ALLOWED_PROJECTORS:
        raise ValueError(
            "mirai_foresight_projector_type must be one of "
            f"{sorted(_ALLOWED_PROJECTORS)}, got "
            f"{args.mirai_foresight_projector_type!r}"
        )
    if args.mirai_foresight_loss_type not in _ALLOWED_LOSSES:
        raise ValueError(
            "mirai_foresight_loss_type must be one of "
            f"{sorted(_ALLOWED_LOSSES)}, got "
            f"{args.mirai_foresight_loss_type!r}"
        )
    if args.mirai_foresight_loss_lambda < 0:
        raise ValueError(
            "mirai_foresight_loss_lambda must be >= 0, got "
            f"{args.mirai_foresight_loss_lambda}"
        )
    if args.enable_mirai_foresight and args.mirai_foresight_loss_lambda <= 0:
        raise ValueError(
            "mirai_foresight_loss_lambda must be > 0 when "
            "enable_mirai_foresight is true"
        )
    if args.mirai_foresight_delta < 1:
        raise ValueError(
            f"mirai_foresight_delta must be >= 1, got {args.mirai_foresight_delta}"
        )
    if args.mirai_foresight_num_frame_per_block < 1:
        raise ValueError(
            "mirai_foresight_num_frame_per_block must be >= 1, got "
            f"{args.mirai_foresight_num_frame_per_block}"
        )
    if args.mirai_foresight_projector_layers < 1:
        raise ValueError(
            "mirai_foresight_projector_layers must be >= 1, got "
            f"{args.mirai_foresight_projector_layers}"
        )
    if args.mirai_foresight_projector_hidden_dim < 0:
        raise ValueError(
            "mirai_foresight_projector_hidden_dim must be 0 (auto) or > 0, got "
            f"{args.mirai_foresight_projector_hidden_dim}"
        )
    if args.mirai_foresight_ffn_ratio <= 0:
        raise ValueError(
            f"mirai_foresight_ffn_ratio must be > 0, got {args.mirai_foresight_ffn_ratio}"
        )
    if args.mirai_foresight_num_heads < 0:
        raise ValueError(
            "mirai_foresight_num_heads must be 0 (auto) or > 0, got "
            f"{args.mirai_foresight_num_heads}"
        )
    if (
        args.mirai_foresight_max_spatial_tokens == 0
        or args.mirai_foresight_max_spatial_tokens < -1
    ):
        raise ValueError(
            "mirai_foresight_max_spatial_tokens must be -1 (disabled) or > 0, got "
            f"{args.mirai_foresight_max_spatial_tokens}"
        )
    if args.mirai_foresight_projector_lr_ratio <= 0:
        raise ValueError(
            "mirai_foresight_projector_lr_ratio must be > 0, got "
            f"{args.mirai_foresight_projector_lr_ratio}"
        )
    for depth in args.mirai_foresight_alignment_depths:
        if depth < 0:
            raise ValueError(
                f"mirai_foresight_alignment_depths entries must be >= 0, got {depth}"
            )

    if args.enable_mirai_foresight:
        mutually_exclusive_flags = {
            "enable_repa": "enable_repa",
            "enable_irepa": "enable_irepa",
            "enable_vae_repa": "enable_vae_repa",
            "enable_videorepa": "enable_videorepa",
            "enable_m2_repa": "enable_m2_repa",
            "sara_enabled": "sara_enabled",
            "enable_moalign": "enable_moalign",
            "crepa_enabled": "crepa_enabled",
            "enable_structure_from_tracking": "enable_structure_from_tracking",
        }
        for config_key, display_name in mutually_exclusive_flags.items():
            if bool(config.get(config_key, False)):
                raise ValueError(
                    "enable_mirai_foresight is mutually exclusive with "
                    f"{display_name}"
                )

        logger.info(
            "Mirai Foresight enabled (depths=%s, delta=%d block(s), "
            "include_current=%s, mean_pool=%s, projector=%s, loss=%s, "
            "lambda=%.4f, max_spatial_tokens=%d).",
            args.mirai_foresight_alignment_depths,
            args.mirai_foresight_delta,
            args.mirai_foresight_include_current,
            args.mirai_foresight_delta_mean_pool,
            args.mirai_foresight_projector_type,
            args.mirai_foresight_loss_type,
            args.mirai_foresight_loss_lambda,
            args.mirai_foresight_max_spatial_tokens,
        )
