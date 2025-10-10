"""Combine individual SARA loss components."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from common.logger import get_logger

from .config import SaraConfig


logger = get_logger(__name__)


@dataclass
class SaraLossComponents:
    """Holds the loss pieces for logging."""

    patch_loss: Optional[torch.Tensor] = None
    autocorr_loss: Optional[torch.Tensor] = None
    adversarial_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if self.patch_loss is not None:
            result["patch_loss"] = float(self.patch_loss.detach().item())
        if self.autocorr_loss is not None:
            result["autocorr_loss"] = float(self.autocorr_loss.detach().item())
        if self.adversarial_loss is not None:
            result["adversarial_loss"] = float(
                self.adversarial_loss.detach().item()
            )
        if self.total_loss is not None:
            result["total_loss"] = float(self.total_loss.detach().item())
        return result


class SaraLossAggregator:
    """Weight and sum the individual losses."""

    def __init__(self, config: SaraConfig) -> None:
        self.config = config
        logger.info(
            "SARA loss weights: patch=%.2f autocorr=%.2f adversarial=%.2f",
            config.patch_loss_weight,
            config.autocorr_loss_weight,
            config.adversarial_loss_weight,
        )

    def aggregate(
        self,
        patch_loss: Optional[torch.Tensor] = None,
        autocorr_loss: Optional[torch.Tensor] = None,
        adversarial_loss: Optional[torch.Tensor] = None,
    ) -> SaraLossComponents:
        device = None
        for component in (patch_loss, autocorr_loss, adversarial_loss):
            if component is not None:
                device = component.device
                break
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        total = torch.tensor(0.0, device=device)

        if patch_loss is not None and self.config.patch_loss_weight > 0:
            total = total + patch_loss * self.config.patch_loss_weight
        if autocorr_loss is not None and self.config.autocorr_loss_weight > 0:
            total = total + autocorr_loss * self.config.autocorr_loss_weight
        if (
            adversarial_loss is not None
            and self.config.adversarial_enabled
            and self.config.adversarial_loss_weight > 0
        ):
            total = total + adversarial_loss * self.config.adversarial_loss_weight

        return SaraLossComponents(
            patch_loss=patch_loss,
            autocorr_loss=autocorr_loss,
            adversarial_loss=adversarial_loss,
            total_loss=total,
        )

    def get_active_components(self) -> Dict[str, bool]:
        return {
            "patch": self.config.patch_loss_weight > 0,
            "autocorr": self.config.autocorr_loss_weight > 0,
            "adversarial": self.config.adversarial_enabled
            and self.config.adversarial_loss_weight > 0,
        }

