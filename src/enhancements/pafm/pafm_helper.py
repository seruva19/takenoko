"""Posterior-Augmented Flow Matching target helper.

The helper implements the PAFM rectified-flow target mixture from
Stoica et al., "Posterior Augmented Flow Matching" (arXiv:2605.00825v1).
It is training-only, default-off, and uses in-batch candidate targets so no
inference path or cache format changes are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PAFMTargetResult:
    target: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class PosteriorAugmentedFlowMatchingHelper:
    """Build posterior-weighted flow target mixtures for a training batch."""

    def __init__(self, args: Any):
        self.enabled = bool(getattr(args, "enable_pafm", False))
        self.candidate_source = str(
            getattr(args, "pafm_candidate_source", "in_batch_nearest")
        ).lower()
        self.num_candidates = max(int(getattr(args, "pafm_num_candidates", 4)), 1)
        self.min_candidates = max(int(getattr(args, "pafm_min_candidates", 2)), 1)
        self.condition_source = str(
            getattr(args, "pafm_condition_source", "auto")
        ).lower()
        self.allow_cross_condition_fallback = bool(
            getattr(args, "pafm_allow_cross_condition_fallback", False)
        )
        self.likelihood_reduction = str(
            getattr(args, "pafm_likelihood_reduction", "sum")
        ).lower()
        self.weight_temperature = float(
            getattr(args, "pafm_weight_temperature", 1.0)
        )
        self.t_min = float(getattr(args, "pafm_t_min", 1e-4))
        self.blend = float(getattr(args, "pafm_blend", 1.0))
        self.log_interval = int(getattr(args, "pafm_log_interval", 100))
        self.skip_for_cdc_fm = bool(getattr(args, "enable_cdc_fm", False))
        self.last_metrics: Dict[str, torch.Tensor] = {}
        self._hooks_setup = False

    def setup_hooks(self) -> None:
        self._hooks_setup = True

    def remove_hooks(self) -> None:
        self._hooks_setup = False

    def apply_to_target(
        self,
        *,
        base_target: torch.Tensor,
        clean_latents: torch.Tensor,
        noisy_latents: torch.Tensor,
        batch: Dict[str, Any],
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return base_target

        result = self.compute_target(
            base_target=base_target,
            clean_latents=clean_latents,
            noisy_latents=noisy_latents,
            batch=batch,
        )
        self.last_metrics = result.metrics

        if (
            self.log_interval > 0
            and global_step is not None
            and global_step % self.log_interval == 0
        ):
            metrics = {
                key: float(value.detach().cpu().item())
                for key, value in result.metrics.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            logger.info(
                "PAFM step=%d active=%.3f candidates=%.2f ess=%.2f self_w=%.3f",
                global_step,
                metrics.get("pafm/active_ratio", 0.0),
                metrics.get("pafm/candidate_count_mean", 0.0),
                metrics.get("pafm/ess_mean", 0.0),
                metrics.get("pafm/self_weight_mean", 0.0),
            )

        return result.target

    def compute_target(
        self,
        *,
        base_target: torch.Tensor,
        clean_latents: torch.Tensor,
        noisy_latents: torch.Tensor,
        batch: Dict[str, Any],
    ) -> PAFMTargetResult:
        if base_target.shape != clean_latents.shape:
            raise ValueError(
                "PAFM expected base_target and clean_latents to share shape, got "
                f"{tuple(base_target.shape)} and {tuple(clean_latents.shape)}"
            )
        if noisy_latents.shape != clean_latents.shape:
            raise ValueError(
                "PAFM expected noisy_latents and clean_latents to share shape, got "
                f"{tuple(noisy_latents.shape)} and {tuple(clean_latents.shape)}"
            )

        batch_size = int(clean_latents.shape[0])
        if batch_size == 0:
            return PAFMTargetResult(base_target, {})

        with torch.no_grad():
            indices, valid_mask, valid_counts = self._select_candidates(
                clean_latents.detach(),
                batch,
            )
            candidate_latents = clean_latents.detach().index_select(
                0, indices.reshape(-1)
            )
            candidate_latents = candidate_latents.reshape(
                batch_size,
                self.num_candidates,
                *clean_latents.shape[1:],
            )

            sigma = self._estimate_sigma(
                base_target=base_target.detach(),
                clean_latents=clean_latents.detach(),
                noisy_latents=noisy_latents.detach(),
            )
            finite_sigma = torch.isfinite(sigma)
            active = (
                (valid_counts >= self.min_candidates)
                & finite_sigma
                & (sigma > self.t_min)
            )

            sigma_view = sigma.clamp_min(max(self.t_min, 1e-8)).view(
                batch_size,
                *([1] * (clean_latents.dim() - 1)),
            )
            sigma_candidates = sigma_view.unsqueeze(1)
            noisy = noisy_latents.detach().to(dtype=torch.float32).unsqueeze(1)
            candidates = candidate_latents.to(dtype=torch.float32)

            likelihood_terms = (
                noisy - (1.0 - sigma_candidates) * candidates
            ).square() / (2.0 * sigma_candidates.square().clamp_min(1e-12))
            flat_terms = likelihood_terms.flatten(start_dim=2)
            if self.likelihood_reduction == "mean":
                negative_distance = -flat_terms.mean(dim=2)
            else:
                negative_distance = -flat_terms.sum(dim=2)
            log_weights = negative_distance / self.weight_temperature
            log_weights = log_weights.masked_fill(~valid_mask, float("-inf"))

            inactive = ~active
            if bool(inactive.any().item()):
                log_weights[inactive] = float("-inf")
                log_weights[inactive, 0] = 0.0

            weights = torch.softmax(log_weights, dim=1)
            if not bool(torch.isfinite(weights).all().item()):
                weights = torch.zeros_like(weights)
                weights[:, 0] = 1.0

            velocities = (noisy - candidates) / sigma_candidates.clamp_min(1e-8)
            weight_view = weights.to(dtype=torch.float32).view(
                batch_size,
                self.num_candidates,
                *([1] * (clean_latents.dim() - 1)),
            )
            mixed_target = (weight_view * velocities).sum(dim=1).to(
                device=base_target.device,
                dtype=base_target.dtype,
            )
            if self.blend < 1.0:
                mixed_target = base_target + self.blend * (mixed_target - base_target)

            metrics = self._build_metrics(
                active=active,
                valid_counts=valid_counts,
                weights=weights,
                sigma=sigma,
                target_delta=mixed_target - base_target,
            )

        return PAFMTargetResult(mixed_target.detach(), metrics)

    def _select_candidates(
        self,
        clean_latents: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(clean_latents.shape[0])
        device = clean_latents.device
        indices = torch.zeros(
            (batch_size, self.num_candidates),
            dtype=torch.long,
            device=device,
        )
        valid_mask = torch.zeros(
            (batch_size, self.num_candidates),
            dtype=torch.bool,
            device=device,
        )
        valid_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        labels = self._condition_labels(batch, batch_size, device)
        if labels is None:
            allow_ungrouped = self.condition_source == "none"
            allowed = torch.eye(batch_size, dtype=torch.bool, device=device)
            if allow_ungrouped:
                allowed = torch.ones(
                    (batch_size, batch_size),
                    dtype=torch.bool,
                    device=device,
                )
        else:
            labels = labels.view(batch_size)
            valid_labels = labels >= 0
            allowed = (
                (labels[:, None] == labels[None, :])
                & valid_labels[:, None]
                & valid_labels[None, :]
            )
            allowed.fill_diagonal_(True)

        flat = clean_latents.detach().to(dtype=torch.float32).flatten(start_dim=1)
        norms = flat.square().sum(dim=1, keepdim=True)
        distances = (norms + norms.transpose(0, 1) - 2.0 * (flat @ flat.transpose(0, 1))).clamp_min_(0.0)
        arange = torch.arange(batch_size, device=device)

        for row in range(batch_size):
            selected = [row]
            allowed_row = allowed[row] & (arange != row)
            candidates = arange[allowed_row]
            selected.extend(self._order_candidates(candidates, distances[row]).tolist())

            if (
                self.allow_cross_condition_fallback
                and labels is not None
                and len(selected) < self.num_candidates
            ):
                selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
                cross_mask = arange != row
                if selected_tensor.numel() > 0:
                    cross_mask &= ~torch.isin(arange, selected_tensor)
                cross_candidates = arange[cross_mask]
                selected.extend(
                    self._order_candidates(cross_candidates, distances[row]).tolist()
                )

            selected = selected[: self.num_candidates]
            if not selected:
                selected = [row]
            count = len(selected)
            padded = selected + [row] * (self.num_candidates - count)
            indices[row] = torch.tensor(padded, dtype=torch.long, device=device)
            valid_mask[row, :count] = True
            valid_counts[row] = count

        return indices, valid_mask, valid_counts

    def _order_candidates(
        self,
        candidates: torch.Tensor,
        row_distances: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.numel() == 0:
            return candidates
        if self.candidate_source == "in_batch_random":
            order = torch.randperm(candidates.numel(), device=candidates.device)
            return candidates[order][: max(self.num_candidates - 1, 0)]
        scores = row_distances.index_select(0, candidates)
        order = torch.argsort(scores, dim=0)
        return candidates[order][: max(self.num_candidates - 1, 0)]

    def _condition_labels(
        self,
        batch: Dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.condition_source == "none":
            return None

        keys = (
            ("concept_id", "dataset_index")
            if self.condition_source == "auto"
            else (self.condition_source,)
        )
        for key in keys:
            value = batch.get(key)
            labels = self._as_label_tensor(value, batch_size, device)
            if labels is not None and bool((labels >= 0).any().item()):
                return labels
        return None

    @staticmethod
    def _as_label_tensor(
        value: Any,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            labels = value.detach().to(device=device, dtype=torch.long).view(-1)
        else:
            try:
                labels = torch.as_tensor(value, dtype=torch.long, device=device).view(-1)
            except Exception:
                return None
        if labels.numel() != batch_size:
            return None
        return labels

    @staticmethod
    def _estimate_sigma(
        *,
        base_target: torch.Tensor,
        clean_latents: torch.Tensor,
        noisy_latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(clean_latents.shape[0])
        target = base_target.to(dtype=torch.float32)
        delta = noisy_latents.to(dtype=torch.float32) - clean_latents.to(dtype=torch.float32)
        flat_target = target.reshape(batch_size, -1)
        flat_delta = delta.reshape(batch_size, -1)
        numerator = (flat_delta * flat_target).mean(dim=1)
        denominator = flat_target.square().mean(dim=1).clamp_min(1e-12)
        return (numerator / denominator).clamp(0.0, 1.0)

    @staticmethod
    def _safe_mean(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return fallback
        return values.mean()

    def _build_metrics(
        self,
        *,
        active: torch.Tensor,
        valid_counts: torch.Tensor,
        weights: torch.Tensor,
        sigma: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fallback = weights.new_tensor(0.0)
        active_weights = active.to(dtype=torch.float32)
        ess = 1.0 / weights.square().sum(dim=1).clamp_min(1e-12)
        delta_norm = target_delta.detach().to(dtype=torch.float32).flatten(start_dim=1).norm(dim=1)

        return {
            "pafm/active_ratio": active_weights.mean().detach(),
            "pafm/candidate_count_mean": valid_counts.to(dtype=torch.float32).mean().detach(),
            "pafm/ess_mean": self._safe_mean(ess[active], fallback).detach(),
            "pafm/self_weight_mean": weights[:, 0].mean().detach(),
            "pafm/sigma_mean": sigma.to(dtype=torch.float32).mean().detach(),
            "pafm/target_delta_norm": self._safe_mean(delta_norm[active], fallback).detach(),
            "pafm/blend": weights.new_tensor(float(self.blend)).detach(),
        }
