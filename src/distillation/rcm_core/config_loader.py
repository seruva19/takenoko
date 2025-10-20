"""Helpers for normalising Takenoko RCM configuration surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RCMConfig:
    """Canonical configuration passed into the RCM distillation runner."""

    trainer_variant: str = "distill"
    max_steps: Optional[int] = None
    mixed_precision: str = "bf16"
    accelerator_mode: str = "auto"
    override_wan: bool = True
    cpu_debug: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def merged_overrides(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return a dictionary of overrides suitable for downstream runners."""
        merged = dict(self.extra_args)
        if overrides:
            merged.update(overrides)
        return merged


def load_rcm_config(
    args: Any,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RCMConfig:
    """Construct an :class:`RCMConfig` from the parsed Takenoko args namespace."""

    rcm_args = getattr(args, "rcm", None)
    if rcm_args is None:
        logger.debug("RCM configuration namespace missing; using defaults.")
        rcm_args = type("RCMArgs", (), {})()  # type: ignore[misc]

    extra_args: MutableMapping[str, Any]
    extra_args = {}
    candidate_extra = getattr(rcm_args, "extra_args", {}) or {}
    if isinstance(candidate_extra, Mapping):
        extra_args.update(candidate_extra)
    else:
        logger.warning(
            "RCM extra_args should be a mapping, got %s. Ignoring value.",
            type(candidate_extra),
        )

    config = RCMConfig(
        trainer_variant=str(getattr(rcm_args, "trainer_variant", "distill")),
        max_steps=getattr(rcm_args, "max_steps", None),
        mixed_precision=str(getattr(rcm_args, "mixed_precision", "bf16")),
        accelerator_mode=str(getattr(rcm_args, "accelerator_mode", "auto")),
        override_wan=bool(getattr(rcm_args, "override_wan", True)),
        cpu_debug=bool(getattr(rcm_args, "cpu_debug", False)),
        extra_args=dict(extra_args),
    )

    if overrides:
        config.extra_args.update(overrides)

    return config
