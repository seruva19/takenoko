"""Glance distillation config parsing and validation."""
from __future__ import annotations

from typing import Any, Dict, Optional


def _parse_glance_timesteps(raw: Any) -> Optional[list[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(v) for v in raw]
    if isinstance(raw, tuple):
        return [float(v) for v in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return None
        return [float(v.strip()) for v in s.split(",") if v.strip()]
    return None


def apply_glance_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    args.glance_enabled = bool(config.get("glance_enabled", False))
    args.glance_mode = str(config.get("glance_mode", "slow")).lower()
    raw_glance_timesteps = config.get("glance_timesteps", None)

    parsed_timesteps = _parse_glance_timesteps(raw_glance_timesteps)
    if parsed_timesteps is None and raw_glance_timesteps is not None:
        logger.warning(
            "⚠️  Invalid glance_timesteps type '%s'. Expected list or string.",
            type(raw_glance_timesteps),
        )
    args.glance_timesteps = parsed_timesteps

    if args.glance_mode not in {"slow", "fast", "custom"}:
        raise ValueError(
            f"glance_mode must be one of slow|fast|custom, got '{args.glance_mode}'"
        )
    if args.glance_mode == "custom" and not args.glance_timesteps:
        raise ValueError("glance_mode=custom requires glance_timesteps to be set.")
    if args.glance_mode != "custom" and args.glance_timesteps:
        logger.info(
            "glance_timesteps provided but glance_mode is '%s'; ignoring custom values.",
            args.glance_mode,
        )
        args.glance_timesteps = None
