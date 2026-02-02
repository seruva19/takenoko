"""Self-Transcendence config parsing and validation."""

from __future__ import annotations

from typing import Any, Dict


_ALLOWED_UNCOND_MODES = {"zero", "batch_null"}


def apply_self_transcendence_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.enable_self_transcendence = bool(
        config.get("enable_self_transcendence", False)
    )
    args.self_transcendence_vae_loss_weight = float(
        config.get("self_transcendence_vae_loss_weight", 1.0)
    )
    args.self_transcendence_guidance_loss_weight = float(
        config.get("self_transcendence_guidance_loss_weight", 0.5)
    )
    args.self_transcendence_guidance_scale = float(
        config.get("self_transcendence_guidance_scale", 30.0)
    )
    args.self_transcendence_guided_layer = int(
        config.get("self_transcendence_guided_layer", -1)
    )
    args.self_transcendence_guiding_layer = int(
        config.get("self_transcendence_guiding_layer", -1)
    )
    args.self_transcendence_warmup_epochs = int(
        config.get("self_transcendence_warmup_epochs", 40)
    )
    args.self_transcendence_guidance_epochs = int(
        config.get("self_transcendence_guidance_epochs", 20)
    )
    args.self_transcendence_warmup_steps = int(
        config.get("self_transcendence_warmup_steps", 0)
    )
    args.self_transcendence_guidance_steps = int(
        config.get("self_transcendence_guidance_steps", 0)
    )
    args.self_transcendence_mlp_multiplier = int(
        config.get("self_transcendence_mlp_multiplier", 2)
    )
    args.self_transcendence_uncond_mode = str(
        config.get("self_transcendence_uncond_mode", "zero")
    ).lower()

    if not args.enable_self_transcendence:
        return

    if args.self_transcendence_vae_loss_weight < 0.0:
        raise ValueError("self_transcendence_vae_loss_weight must be >= 0")
    if args.self_transcendence_guidance_loss_weight < 0.0:
        raise ValueError("self_transcendence_guidance_loss_weight must be >= 0")
    if args.self_transcendence_guidance_scale < 0.0:
        raise ValueError("self_transcendence_guidance_scale must be >= 0")
    if args.self_transcendence_warmup_epochs < 0:
        raise ValueError("self_transcendence_warmup_epochs must be >= 0")
    if args.self_transcendence_guidance_epochs < 0:
        raise ValueError("self_transcendence_guidance_epochs must be >= 0")
    if args.self_transcendence_warmup_steps < 0:
        raise ValueError("self_transcendence_warmup_steps must be >= 0")
    if args.self_transcendence_guidance_steps < 0:
        raise ValueError("self_transcendence_guidance_steps must be >= 0")
    if args.self_transcendence_mlp_multiplier < 1:
        raise ValueError("self_transcendence_mlp_multiplier must be >= 1")
    if args.self_transcendence_uncond_mode not in _ALLOWED_UNCOND_MODES:
        raise ValueError(
            f"self_transcendence_uncond_mode must be one of {_ALLOWED_UNCOND_MODES}"
        )

    if args.self_transcendence_warmup_steps > 0 or args.self_transcendence_guidance_steps > 0:
        logger.info(
            "Self-Transcendence enabled (warmup_steps=%d, guidance_steps=%d, guided_layer=%s, guiding_layer=%s)",
            args.self_transcendence_warmup_steps,
            args.self_transcendence_guidance_steps,
            args.self_transcendence_guided_layer,
            args.self_transcendence_guiding_layer,
        )
    else:
        logger.info(
            "Self-Transcendence enabled (warmup_epochs=%d, guidance_epochs=%d, guided_layer=%s, guiding_layer=%s)",
            args.self_transcendence_warmup_epochs,
            args.self_transcendence_guidance_epochs,
            args.self_transcendence_guided_layer,
            args.self_transcendence_guiding_layer,
        )
