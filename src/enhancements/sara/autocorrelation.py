"""Autocorrelation alignment module."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from common.logger import get_logger

from .config import SaraConfig
from .utils import compute_autocorrelation_matrix, autocorrelation_loss


logger = get_logger(__name__)


class AutocorrelationAligner(nn.Module):
    """Structural alignment via autocorrelation matrices."""

    def __init__(self, config: SaraConfig) -> None:
        super().__init__()
        self.config = config
        self.eps = 1e-8
        self._cached_target: Optional[torch.Tensor] = None
        self._cached_source_id: Optional[int] = None
        self._cache_valid = False

    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        return_matrices: bool = False,
    ) -> Dict[str, torch.Tensor]:
        target_autocorr = self.compute_target_autocorrelation(target_features)
        pred_autocorr = self.compute_prediction_autocorrelation(pred_features)

        if pred_autocorr.shape != target_autocorr.shape:
            new_size = target_autocorr.shape[-1]
            pred_autocorr = torch.nn.functional.adaptive_avg_pool2d(
                pred_autocorr.unsqueeze(1), (new_size, new_size)
            ).squeeze(1)

        loss = autocorrelation_loss(
            pred_autocorr,
            target_autocorr,
            use_frobenius=self.config.autocorr_use_frobenius,
        )

        result: Dict[str, torch.Tensor] = {"loss": loss}
        if return_matrices:
            result["pred_autocorr"] = pred_autocorr
            result["target_autocorr"] = target_autocorr
        return result

    def compute_target_autocorrelation(
        self,
        target_features: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if (
            use_cache
            and self.config.cache_encoder_outputs
            and self._cache_valid
            and self._cached_target is not None
            and self._cached_source_id == id(target_features)
        ):
            # Cache hits are scoped to a single forward pass. The helper clears the
            # cache at the start of each loss computation, so comparing Python ids is
            # intentional: it guarantees we only reuse the exact tensor object that
            # triggered the cache without accidentally leaking values across batches.
            return self._cached_target

        autocorr = compute_autocorrelation_matrix(
            target_features,
            normalize=self.config.autocorr_normalize,
            eps=self.eps,
        )

        if use_cache and self.config.cache_encoder_outputs:
            self._cached_target = autocorr.detach()
            self._cached_source_id = id(target_features)
            self._cache_valid = True

        return autocorr

    def compute_prediction_autocorrelation(
        self,
        pred_features: torch.Tensor,
    ) -> torch.Tensor:
        return compute_autocorrelation_matrix(
            pred_features,
            normalize=self.config.autocorr_normalize,
            eps=self.eps,
        )

    def clear_cache(self) -> None:
        self._cached_target = None
        self._cached_source_id = None
        self._cache_valid = False

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        state["_cache_valid"] = self._cache_valid
        state["_cached_source_id"] = self._cached_source_id
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]  # noqa: ANN001
        cache_valid = state_dict.pop("_cache_valid", False)
        cached_source_id = state_dict.pop("_cached_source_id", None)
        super().load_state_dict(state_dict, strict=strict)
        self._cache_valid = cache_valid
        self._cached_source_id = cached_source_id
        self._cached_target = None

    @torch.no_grad()
    def get_metrics(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Dict[str, float]:
        matrices = self.forward(
            pred_features,
            target_features,
            return_matrices=True,
        )
        pred_autocorr = matrices["pred_autocorr"]
        target_autocorr = matrices["target_autocorr"]
        diff = pred_autocorr - target_autocorr

        metrics = {
            "autocorr_loss": matrices["loss"].item(),
            "pred_autocorr_mean": pred_autocorr.mean().item(),
            "pred_autocorr_std": pred_autocorr.std().item(),
            "target_autocorr_mean": target_autocorr.mean().item(),
            "target_autocorr_std": target_autocorr.std().item(),
            "autocorr_diff_max": diff.abs().max().item(),
            "autocorr_diff_mean": diff.abs().mean().item(),
        }

        pred_diag = torch.diagonal(pred_autocorr, dim1=-2, dim2=-1)
        target_diag = torch.diagonal(target_autocorr, dim1=-2, dim2=-1)
        metrics["pred_diag_mean"] = pred_diag.mean().item()
        metrics["target_diag_mean"] = target_diag.mean().item()
        return metrics
