"""Utilities for parsing transition training configuration."""

from __future__ import annotations

from typing import Any, Dict

from easydict import EasyDict


def parse_transition_config(config: Dict[str, Any]) -> EasyDict:
    """
    Parse transition training options from root-level `transition_*` keys
    (with backwards-compatible support for `[transition_training]` tables).
    """

    section = config.get("transition_training", {}) or {}
    if not isinstance(section, dict):
        section = {}

    def _get(name: str, default: Any) -> Any:
        prefixed_key = f"transition_{name}"
        if prefixed_key in config:
            return config[prefixed_key]
        return section.get(name, default)

    def _safe_float(name: str, default: float) -> float:
        try:
            return float(_get(name, default))
        except Exception:
            return default

    def _safe_int(name: str, default: int) -> int:
        try:
            return int(_get(name, default))
        except Exception:
            return default

    enabled_raw = _get("training_enabled", None)
    if enabled_raw is None:
        enabled_raw = _get("enabled", False)
    enabled = bool(enabled_raw)

    diffusion_ratio = max(0.0, min(1.0, _safe_float("diffusion_ratio", 0.5)))
    consistency_ratio = max(0.0, min(1.0, _safe_float("consistency_ratio", 0.1)))
    if diffusion_ratio + consistency_ratio > 1.0:
        consistency_ratio = max(0.0, 1.0 - diffusion_ratio)

    derivative_mode = str(_get("derivative_mode", "dde")).lower()
    if derivative_mode not in {"dde", "jvp", "none", "auto"}:
        derivative_mode = "dde"

    derivative_failover = str(_get("derivative_failover", "jvp")).lower()
    if derivative_failover not in {"dde", "jvp", "none"}:
        derivative_failover = "jvp"

    finite_difference_eps = max(0.0, _safe_float("finite_difference_eps", 5e-3))

    delta_time_domain = str(_get("delta_time_domain", "discrete")).lower()
    if delta_time_domain not in {"discrete", "normalized"}:
        delta_time_domain = "discrete"

    weight_schedule = str(_get("weight_schedule", "sqrt")).lower()

    tangent_weighting = bool(_get("tangent_weighting", True))
    adaptive_weighting = bool(_get("adaptive_weighting", True))
    lora_interval_modulation = bool(_get("lora_interval_modulation", False))

    directional_loss_weight = max(0.0, _safe_float("directional_loss_weight", 0.0))

    use_ema_teacher = bool(_get("use_ema_teacher", False))
    teacher_mix = max(0.0, min(1.0, _safe_float("teacher_mix", 0.2)))
    teacher_decay = _safe_float("teacher_decay", 0.999)
    if not 0.0 < teacher_decay < 1.0:
        teacher_decay = 0.999

    transport_name = str(_get("transport", "linear")).lower()
    if transport_name not in {"linear", "trigflow", "vp"}:
        transport_name = "linear"

    delta_attention = bool(_get("delta_attention", False))
    delta_mlp_hidden = max(1, _safe_int("delta_mlp_hidden", 64))

    return EasyDict(
        enabled=enabled,
        diffusion_ratio=diffusion_ratio,
        consistency_ratio=consistency_ratio,
        derivative_mode=derivative_mode,
        derivative_failover=derivative_failover,
        finite_difference_eps=finite_difference_eps,
        delta_time_domain=delta_time_domain,
        weight_schedule=weight_schedule,
        tangent_weighting=tangent_weighting,
        adaptive_weighting=adaptive_weighting,
        lora_interval_modulation=lora_interval_modulation,
        directional_loss_weight=directional_loss_weight,
        transport=transport_name,
        delta_attention=delta_attention,
        delta_mlp_hidden=delta_mlp_hidden,
        use_ema_teacher=use_ema_teacher,
        teacher_mix=teacher_mix,
        teacher_decay=teacher_decay,
    )
