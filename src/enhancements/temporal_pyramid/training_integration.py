"""Temporal pyramid training helpers (TPDiff-inspired, training-only)."""

from __future__ import annotations

from typing import Optional

import torch

from common.logger import get_logger

logger = get_logger(__name__)

try:
    from diffusers.utils.import_utils import is_scipy_available
except Exception:  # pragma: no cover - safety for minimal installs
    def is_scipy_available() -> bool:
        return False


class TemporalPyramidHelper:
    """Optional data-noise alignment for temporal pyramid training."""

    def __init__(self, args) -> None:
        self._enabled = bool(
            getattr(args, "enable_temporal_pyramid_data_noise_alignment", False)
        )
        self._warned_no_scipy = False
        self._warned_small_batch = False

    def setup_hooks(self) -> None:
        """No-op hook to align with enhancement interface."""

    def remove_hooks(self) -> None:
        """No-op hook to align with enhancement interface."""

    def align_noise(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Align noise assignments to latents via summary-stat matching.

        Uses a per-sample mean summary across spatial/temporal dims to keep the
        cost low. When SciPy is unavailable, falls back to greedy assignment.
        """
        if not self._enabled:
            return noise

        if latents.ndim < 4 or noise.ndim != latents.ndim:
            return noise

        batch_size = latents.shape[0]
        if batch_size < 2:
            if not self._warned_small_batch:
                logger.debug(
                    "Temporal pyramid noise alignment skipped: batch size < 2."
                )
                self._warned_small_batch = True
            return noise

        # Summarize per-sample content to reduce assignment cost.
        reduce_dims = tuple(range(2, latents.ndim))
        latents_summary = latents.float().mean(dim=reduce_dims)
        noise_summary = noise.float().mean(dim=reduce_dims)

        distances = torch.cdist(latents_summary, noise_summary, p=2).cpu()

        assignment = None
        if is_scipy_available():
            try:
                import numpy as np
                from scipy.optimize import linear_sum_assignment

                row_ind, col_ind = linear_sum_assignment(
                    distances.numpy().astype("float64")
                )
                assignment = torch.as_tensor(
                    col_ind, device=latents.device, dtype=torch.long
                )
            except Exception as exc:
                logger.warning("Noise alignment (SciPy) failed: %s", exc)

        if assignment is None:
            if not self._warned_no_scipy and not is_scipy_available():
                logger.info(
                    "SciPy not available; using greedy noise alignment fallback."
                )
                self._warned_no_scipy = True
            assignment = _greedy_assignment(distances).to(
                device=latents.device, dtype=torch.long
            )

        return noise.index_select(0, assignment)


def _greedy_assignment(distances: torch.Tensor) -> torch.Tensor:
    """Greedy assignment for noise alignment when SciPy is unavailable."""
    used = set()
    assignment = []
    for row in distances:
        sorted_indices = torch.argsort(row)
        chosen = None
        for idx in sorted_indices.tolist():
            if idx not in used:
                chosen = idx
                used.add(idx)
                break
        if chosen is None:
            chosen = sorted_indices[0].item()
        assignment.append(chosen)
    return torch.tensor(assignment, dtype=torch.long)


def create_temporal_pyramid_helper(args) -> Optional[TemporalPyramidHelper]:
    """Create the temporal pyramid helper only when alignment is enabled."""
    if getattr(args, "enable_temporal_pyramid_data_noise_alignment", False):
        helper = TemporalPyramidHelper(args)
        helper.setup_hooks()
        return helper
    return None
