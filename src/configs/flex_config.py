from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _parse_pattern_list(value: Any, field_name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or None
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return parts or None
    raise ValueError(
        f"{field_name} must be a list[str], a comma-separated string, or null"
    )


def apply_flex_config(args: Any, config: Dict[str, Any], logger: Any) -> None:
    """Parse FleX Fourier regularization settings onto args."""
    args.enable_flex = bool(config.get("enable_flex", False))
    args.flex_lambda = float(config.get("flex_lambda", 0.02))
    args.flex_frequency_threshold = float(config.get("flex_frequency_threshold", 0.5))
    args.flex_phi_low = float(config.get("flex_phi_low", 1.0))
    args.flex_phi_high = float(config.get("flex_phi_high", 0.1))
    args.flex_include_text_encoder = bool(
        config.get("flex_include_text_encoder", False)
    )
    args.flex_target_tensors = str(config.get("flex_target_tensors", "both")).strip().lower()
    args.flex_include_patterns = _parse_pattern_list(
        config.get("flex_include_patterns", None),
        "flex_include_patterns",
    )
    args.flex_exclude_patterns = _parse_pattern_list(
        config.get("flex_exclude_patterns", None),
        "flex_exclude_patterns",
    )

    if args.flex_lambda < 0.0:
        raise ValueError("flex_lambda must be >= 0")
    if not 0.0 < args.flex_frequency_threshold <= 1.0:
        raise ValueError("flex_frequency_threshold must be in (0, 1]")
    if not 0.0 <= args.flex_phi_low <= 1.0:
        raise ValueError("flex_phi_low must be in [0, 1]")
    if not 0.0 <= args.flex_phi_high <= 1.0:
        raise ValueError("flex_phi_high must be in [0, 1]")
    if args.flex_phi_high > args.flex_phi_low:
        raise ValueError(
            "flex_phi_high must be <= flex_phi_low so high frequencies are not preserved more than low frequencies"
        )
    if args.flex_target_tensors not in {"both", "up", "down"}:
        raise ValueError("flex_target_tensors must be one of: both, up, down")
    for field_name, patterns in (
        ("flex_include_patterns", args.flex_include_patterns),
        ("flex_exclude_patterns", args.flex_exclude_patterns),
    ):
        if not patterns:
            continue
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in {field_name}: {pattern!r}"
                ) from exc

    if args.enable_flex:
        logger.info(
            "FleX enabled (lambda=%.6f, threshold=%.3f, phi_low=%.3f, phi_high=%.3f, include_text_encoder=%s, target_tensors=%s, include_patterns=%s, exclude_patterns=%s).",
            args.flex_lambda,
            args.flex_frequency_threshold,
            args.flex_phi_low,
            args.flex_phi_high,
            args.flex_include_text_encoder,
            args.flex_target_tensors,
            args.flex_include_patterns,
            args.flex_exclude_patterns,
        )
