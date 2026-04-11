"""Helpers for normalizing Takenoko TwinFlow configuration surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TwinFlowConfig:
    """Canonical configuration passed into the TwinFlow distillation runner."""

    trainer_variant: str = "full"
    max_steps: Optional[int] = None
    mixed_precision: str = "bf16"
    accelerator_mode: str = "auto"
    override_wan: bool = True
    cpu_debug: bool = False
    ema_decay: float = 0.999
    consistency_ratio: float = 1.0
    require_ema: bool = True
    allow_student_teacher: bool = False
    enhanced_ratio: float = 0.0
    enhanced_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    estimate_order: int = 1
    delta_t: float = 0.01
    clamp_target: float = 1.0
    adversarial_enabled: bool = True
    adversarial_weight: float = 1.0
    rectify_weight: float = 1.0
    real_velocity_weight: float = 1.0
    time_dist_ctrl: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    log_interval: int = 10
    checkpoint_interval: int = 0
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def merged_overrides(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.extra_args)
        if overrides:
            merged.update(overrides)
        return merged


def load_twinflow_config(
    args: Any,
    overrides: Optional[Mapping[str, Any]] = None,
) -> TwinFlowConfig:
    """Construct a :class:`TwinFlowConfig` from the parsed Takenoko args namespace."""

    twinflow_args = getattr(args, "twinflow", None)
    if twinflow_args is None:
        logger.debug("TwinFlow configuration namespace missing; using defaults.")
        twinflow_args = type("TwinFlowArgs", (), {})()  # type: ignore[misc]

    extra_args: MutableMapping[str, Any] = {}
    candidate_extra = getattr(twinflow_args, "extra_args", {}) or {}
    if isinstance(candidate_extra, Mapping):
        extra_args.update(candidate_extra)
    else:
        logger.warning(
            "TwinFlow extra_args should be a mapping, got %s. Ignoring value.",
            type(candidate_extra),
        )

    config = TwinFlowConfig(
        trainer_variant=str(getattr(twinflow_args, "trainer_variant", "full")),
        max_steps=getattr(twinflow_args, "max_steps", None),
        mixed_precision=str(getattr(twinflow_args, "mixed_precision", "bf16")),
        accelerator_mode=str(getattr(twinflow_args, "accelerator_mode", "auto")),
        override_wan=bool(getattr(twinflow_args, "override_wan", True)),
        cpu_debug=bool(getattr(twinflow_args, "cpu_debug", False)),
        ema_decay=float(getattr(twinflow_args, "ema_decay", 0.999)),
        consistency_ratio=float(getattr(twinflow_args, "consistency_ratio", 1.0)),
        require_ema=bool(getattr(twinflow_args, "require_ema", True)),
        allow_student_teacher=bool(
            getattr(twinflow_args, "allow_student_teacher", False)
        ),
        enhanced_ratio=float(getattr(twinflow_args, "enhanced_ratio", 0.0)),
        enhanced_range=list(getattr(twinflow_args, "enhanced_range", [0.0, 1.0])),
        estimate_order=max(1, int(getattr(twinflow_args, "estimate_order", 1))),
        delta_t=float(getattr(twinflow_args, "delta_t", 0.01)),
        clamp_target=float(getattr(twinflow_args, "clamp_target", 1.0)),
        adversarial_enabled=bool(
            getattr(twinflow_args, "adversarial_enabled", True)
        ),
        adversarial_weight=float(getattr(twinflow_args, "adversarial_weight", 1.0)),
        rectify_weight=float(getattr(twinflow_args, "rectify_weight", 1.0)),
        real_velocity_weight=float(
            getattr(twinflow_args, "real_velocity_weight", 1.0)
        ),
        time_dist_ctrl=list(getattr(twinflow_args, "time_dist_ctrl", [1.0, 1.0, 1.0])),
        log_interval=max(1, int(getattr(twinflow_args, "log_interval", 10))),
        checkpoint_interval=max(
            0,
            int(getattr(twinflow_args, "checkpoint_interval", 0)),
        ),
        extra_args=dict(extra_args),
    )

    if overrides:
        config.extra_args.update(overrides)

    return config
