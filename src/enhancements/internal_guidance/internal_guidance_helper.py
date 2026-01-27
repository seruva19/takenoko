"""Internal Guidance helper for auxiliary supervision loss and target shift."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F


class InternalGuidanceHelper:
    def __init__(self, args) -> None:
        self.loss_type = str(
            getattr(args, "internal_guidance_loss_type", "sml1")
        ).lower()
        self._warned_no_ema: bool = False

    def setup_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        return

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Optional[str] = None,
    ) -> torch.Tensor:
        loss_name = (loss_type or self.loss_type).lower()
        if loss_name == "sml1":
            return F.smooth_l1_loss(pred, target, beta=0.05)
        if loss_name == "l2":
            return F.mse_loss(pred, target)
        if loss_name == "l1":
            return F.l1_loss(pred, target)
        raise ValueError(f"Unsupported internal guidance loss: {loss_name}")

    def compute_shift(
        self,
        *,
        args: Any,
        model: Any,
        model_input: torch.Tensor,
        model_kwargs: Dict[str, Any],
        network_dtype: torch.dtype,
        ema_context: Callable[[], Any],
        weight_ema: Optional[Any],
        logger: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        if not getattr(args, "enable_internal_guidance", False):
            return None
        if str(getattr(args, "internal_guidance_mode", "aux")).lower() != "shift":
            return None
        ig_weight = float(getattr(args, "internal_guidance_weight", 0.0))
        if ig_weight <= 0.0:
            return None

        if weight_ema is None and not self._warned_no_ema:
            if logger is not None:
                logger.warning(
                    "Internal Guidance target shift requested but weight EMA is disabled; using current weights."
                )
            self._warned_no_ema = True

        try:
            with torch.no_grad():
                with ema_context():
                    ema_out = model(model_input, **model_kwargs)
            ema_outputs = ema_out
            ema_internal = None
            if isinstance(ema_out, tuple):
                parts = list(ema_out)
                ema_outputs = parts.pop(0)
                if parts:
                    ema_internal = parts.pop(0)
            if ema_internal is None:
                if logger is not None:
                    logger.warning(
                        "Internal Guidance target shift skipped; intermediate output missing."
                    )
                return None
            ema_outputs = torch.stack(ema_outputs, dim=0)
            ema_internal = torch.stack(ema_internal, dim=0)
            shift = ema_outputs.to(network_dtype) - ema_internal.to(network_dtype)
            return shift.detach() * ig_weight
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Internal Guidance target shift computation failed: %s",
                    exc,
                )
            return None
