from __future__ import annotations

from typing import Any, Iterable

from common.logger import get_logger

logger = get_logger(__name__)


def _normalize_components(components: Any) -> list[str]:
    if isinstance(components, str):
        parts = [c.strip() for c in components.split(",") if c.strip()]
    elif isinstance(components, Iterable):
        parts = [str(c).strip() for c in components if str(c).strip()]
    else:
        parts = []
    allowed = {"translation", "rotation", "scaling"}
    filtered = [c.lower() for c in parts if c.lower() in allowed]
    if not filtered:
        filtered = ["translation", "rotation", "scaling"]
    return filtered


def apply_physics_guided_motion_config(args: Any, config: dict[str, Any]) -> Any:
    args.enable_physics_guided_motion_loss = config.get(
        "enable_physics_guided_motion_loss", False
    )
    args.physics_motion_loss_weight = config.get("physics_motion_loss_weight", 0.05)
    args.physics_motion_lowpass_ratio = config.get(
        "physics_motion_lowpass_ratio", 0.3
    )
    args.physics_motion_tau = config.get("physics_motion_tau", 0.2)
    args.physics_motion_ridge_lambda = config.get(
        "physics_motion_ridge_lambda", 1e-4
    )
    args.physics_motion_components = config.get(
        "physics_motion_components", ["translation", "rotation", "scaling"]
    )
    args.physics_motion_apply_to_latent = config.get(
        "physics_motion_apply_to_latent", True
    )
    args.physics_motion_x0_source = config.get(
        "physics_motion_x0_source", "flow"
    )
    args.physics_motion_radial_bins = config.get("physics_motion_radial_bins", 32)
    args.physics_motion_max_frames = config.get("physics_motion_max_frames", 0)

    if args.enable_physics_guided_motion_loss:
        logger.info(
            "Physics-guided motion loss enabled (weight=%.4f, lowpass=%.2f, tau=%.3f).",
            float(args.physics_motion_loss_weight),
            float(args.physics_motion_lowpass_ratio),
            float(args.physics_motion_tau),
        )

    try:
        ratio = float(args.physics_motion_lowpass_ratio)
        if ratio <= 0.0 or ratio > 0.5:
            raise ValueError
    except Exception:
        logger.warning(
            "Invalid physics_motion_lowpass_ratio %.3f; must be in (0, 0.5]. Using 0.3.",
            float(args.physics_motion_lowpass_ratio),
        )
        args.physics_motion_lowpass_ratio = 0.3

    try:
        tau = float(args.physics_motion_tau)
        if tau <= 0.0:
            raise ValueError
    except Exception:
        logger.warning(
            "Invalid physics_motion_tau %.3f; must be > 0. Using 0.2.",
            float(args.physics_motion_tau),
        )
        args.physics_motion_tau = 0.2

    try:
        ridge = float(args.physics_motion_ridge_lambda)
        if ridge < 0.0:
            raise ValueError
    except Exception:
        logger.warning(
            "Invalid physics_motion_ridge_lambda %.6f; must be >= 0. Using 1e-4.",
            float(args.physics_motion_ridge_lambda),
        )
        args.physics_motion_ridge_lambda = 1e-4

    args.physics_motion_components = _normalize_components(
        args.physics_motion_components
    )

    x0_source = str(args.physics_motion_x0_source).lower()
    if x0_source not in ("flow", "sigma"):
        logger.warning(
            "Invalid physics_motion_x0_source '%s'; using 'flow'.",
            args.physics_motion_x0_source,
        )
        args.physics_motion_x0_source = "flow"

    try:
        radial_bins = int(args.physics_motion_radial_bins)
        if radial_bins < 8:
            raise ValueError
    except Exception:
        logger.warning(
            "Invalid physics_motion_radial_bins %s; using 32.",
            args.physics_motion_radial_bins,
        )
        args.physics_motion_radial_bins = 32

    try:
        max_frames = int(args.physics_motion_max_frames)
        if max_frames < 0:
            raise ValueError
    except Exception:
        logger.warning(
            "Invalid physics_motion_max_frames %s; using 0.",
            args.physics_motion_max_frames,
        )
        args.physics_motion_max_frames = 0

    return args
