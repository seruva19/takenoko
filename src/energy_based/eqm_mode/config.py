from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import argparse


@dataclass(frozen=True)
class EqMModeConfig:
    """Configuration payload describing the EqM training mode settings."""

    prediction: str = "velocity"
    path_type: str = "Linear"
    loss_weight: float = 1.0
    transport_weighting: Optional[str] = None
    train_eps: Optional[float] = None
    sample_eps: Optional[float] = None
    energy_head: bool = False
    energy_mode: str = "dot"
    weighting_schedule: Optional[str] = None
    weighting_steps: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EqMModeConfig":
        """Build an EqMModeConfig from parsed args."""
        return cls(
            prediction=str(getattr(args, "eqm_prediction", "velocity")).lower(),
            path_type=str(getattr(args, "eqm_path_type", "Linear")),
            loss_weight=float(getattr(args, "eqm_loss_weight", 1.0)),
            transport_weighting=getattr(args, "eqm_transport_weighting", None),
            train_eps=getattr(args, "eqm_train_eps", None),
            sample_eps=getattr(args, "eqm_sample_eps", None),
            energy_head=bool(getattr(args, "eqm_energy_head", False)),
            energy_mode=str(getattr(args, "eqm_energy_mode", "dot")).lower(),
            weighting_schedule=getattr(args, "eqm_weighting_schedule", None),
            weighting_steps=getattr(args, "eqm_weighting_steps", None),
        )
