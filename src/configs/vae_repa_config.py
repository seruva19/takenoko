from __future__ import annotations

from typing import Any, Dict, List


def _parse_alignment_depths(config: Dict[str, Any]) -> List[int]:
    """Parse VAE-REPA alignment depth(s) into a unique list preserving order."""
    default_depth = int(config.get("vae_repa_alignment_depth", 2))
    raw_depths = config.get("vae_repa_alignment_depths", None)
    if raw_depths is None:
        return [default_depth]
    if not isinstance(raw_depths, (list, tuple)):
        raise ValueError(
            "vae_repa_alignment_depths must be a list of ints or omitted, got "
            f"{type(raw_depths).__name__}"
        )
    if len(raw_depths) == 0:
        raise ValueError("vae_repa_alignment_depths must not be empty when provided")
    parsed = [int(v) for v in raw_depths]
    return list(dict.fromkeys(parsed))


def apply_vae_repa_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse VAE-REPA configuration and validate compatibility."""
    args.enable_vae_repa = bool(config.get("enable_vae_repa", False))
    args.vae_repa_alignment_depths = _parse_alignment_depths(config)
    args.vae_repa_alignment_depth = int(args.vae_repa_alignment_depths[0])
    args.vae_repa_auto_depth = bool(config.get("vae_repa_auto_depth", False))
    args.vae_repa_loss_lambda = float(config.get("vae_repa_loss_lambda", 1.0))
    args.vae_repa_loss_beta = float(config.get("vae_repa_loss_beta", 0.05))
    args.vae_repa_alignment_loss = str(
        config.get("vae_repa_alignment_loss", "smooth_l1")
    ).lower()
    args.vae_repa_projector_hidden_mult = int(
        config.get("vae_repa_projector_hidden_mult", 4)
    )
    args.vae_repa_projector_layers = int(config.get("vae_repa_projector_layers", 5))
    args.vae_repa_target_dim = int(config.get("vae_repa_target_dim", 0))
    args.vae_repa_timestep_min = float(config.get("vae_repa_timestep_min", 0.0))
    args.vae_repa_timestep_max = float(config.get("vae_repa_timestep_max", 1.0))
    args.vae_repa_spatial_align = bool(config.get("vae_repa_spatial_align", True))
    args.vae_repa_use_full_video = bool(config.get("vae_repa_use_full_video", False))

    for depth in args.vae_repa_alignment_depths:
        if depth < 0:
            raise ValueError(
                f"vae_repa_alignment_depths entries must be >= 0, got {depth}"
            )
    if args.vae_repa_loss_lambda < 0:
        raise ValueError(
            f"vae_repa_loss_lambda must be >= 0, got {args.vae_repa_loss_lambda}"
        )
    if args.enable_vae_repa and args.vae_repa_loss_lambda <= 0:
        raise ValueError(
            "vae_repa_loss_lambda must be > 0 when enable_vae_repa is true"
        )
    if args.vae_repa_loss_beta <= 0:
        raise ValueError(f"vae_repa_loss_beta must be > 0, got {args.vae_repa_loss_beta}")
    allowed_loss = {"smooth_l1", "cosine", "l1", "l2"}
    if args.vae_repa_alignment_loss not in allowed_loss:
        raise ValueError(
            "vae_repa_alignment_loss must be one of "
            f"{sorted(allowed_loss)}, got {args.vae_repa_alignment_loss!r}"
        )
    if args.vae_repa_projector_hidden_mult < 1:
        raise ValueError(
            "vae_repa_projector_hidden_mult must be >= 1, got "
            f"{args.vae_repa_projector_hidden_mult}"
        )
    if args.vae_repa_projector_layers < 2:
        raise ValueError(
            "vae_repa_projector_layers must be >= 2, got "
            f"{args.vae_repa_projector_layers}"
        )
    if args.vae_repa_target_dim < 0:
        raise ValueError(
            f"vae_repa_target_dim must be >= 0, got {args.vae_repa_target_dim}"
        )
    if not (0.0 <= args.vae_repa_timestep_min <= 1.0):
        raise ValueError(
            "vae_repa_timestep_min must be in [0,1], got "
            f"{args.vae_repa_timestep_min}"
        )
    if not (0.0 <= args.vae_repa_timestep_max <= 1.0):
        raise ValueError(
            "vae_repa_timestep_max must be in [0,1], got "
            f"{args.vae_repa_timestep_max}"
        )
    if args.vae_repa_timestep_min > args.vae_repa_timestep_max:
        raise ValueError(
            "vae_repa_timestep_min must be <= vae_repa_timestep_max, got "
            f"{args.vae_repa_timestep_min} > {args.vae_repa_timestep_max}"
        )

    if args.enable_vae_repa and bool(config.get("enable_repa", False)):
        raise ValueError("enable_vae_repa and enable_repa are mutually exclusive")
    if args.enable_vae_repa and bool(config.get("enable_irepa", False)):
        raise ValueError("enable_vae_repa and enable_irepa are mutually exclusive")
    if args.enable_vae_repa and bool(config.get("enable_videorepa", False)):
        raise ValueError("enable_vae_repa and enable_videorepa are mutually exclusive")
    if args.enable_vae_repa and bool(config.get("sara_enabled", False)):
        raise ValueError("enable_vae_repa and sara_enabled are mutually exclusive")
    if args.enable_vae_repa and bool(config.get("enable_moalign", False)):
        raise ValueError("enable_vae_repa and enable_moalign are mutually exclusive")

    if args.enable_vae_repa:
        logger.info(
            "VAE-REPA enabled (depths=%s, auto_depth=%s, lambda=%.4f, beta=%.4f, loss=%s, hidden_mult=%d, layers=%d, t_range=[%.3f, %.3f], spatial_align=%s, full_video=%s)",
            args.vae_repa_alignment_depths,
            args.vae_repa_auto_depth,
            args.vae_repa_loss_lambda,
            args.vae_repa_loss_beta,
            args.vae_repa_alignment_loss,
            args.vae_repa_projector_hidden_mult,
            args.vae_repa_projector_layers,
            args.vae_repa_timestep_min,
            args.vae_repa_timestep_max,
            args.vae_repa_spatial_align,
            args.vae_repa_use_full_video,
        )
