"""Immiscible Diffusion noise assignment variants for training."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from diffusers.utils.import_utils import is_scipy_available

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ImmiscibleNoiseHelper:
    """Training-only noise sampler with configurable immiscible assignment modes."""

    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        candidate_count: int,
        assignment_pool_factor: int,
        distance_dtype: str,
        use_scipy: bool,
        fallback_mode: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()
        self.candidate_count = int(candidate_count)
        self.assignment_pool_factor = int(assignment_pool_factor)
        self.distance_dtype = str(distance_dtype).lower()
        self.use_scipy = bool(use_scipy)
        self.fallback_mode = str(fallback_mode).lower()
        self._warned_fallback = False
        self._warned_scipy = False

        if not self.enabled:
            return
        if self.mode not in {"knn", "linear_assignment", "linear_assignment_candidates"}:
            raise ValueError(
                "immiscible_mode must be one of 'knn', 'linear_assignment', "
                f"'linear_assignment_candidates', got {self.mode}"
            )
        if self.candidate_count < 1:
            raise ValueError(
                f"immiscible_candidate_count must be >= 1, got {self.candidate_count}"
            )
        if self.assignment_pool_factor < 1:
            raise ValueError(
                "immiscible_assignment_pool_factor must be >= 1, "
                f"got {self.assignment_pool_factor}"
            )
        if self.distance_dtype not in _DTYPE_MAP:
            raise ValueError(
                "immiscible_distance_dtype must be one of "
                f"{sorted(_DTYPE_MAP.keys())}, got {self.distance_dtype}"
            )
        if self.fallback_mode not in {"knn", "random"}:
            raise ValueError(
                "immiscible_fallback_mode must be one of 'knn', 'random', "
                f"got {self.fallback_mode}"
            )
        logger.info(
            "Immiscible Diffusion helper active (mode=%s, k=%d, pool_factor=%d, distance_dtype=%s, use_scipy=%s, fallback=%s)",
            self.mode,
            self.candidate_count,
            self.assignment_pool_factor,
            self.distance_dtype,
            self.use_scipy,
            self.fallback_mode,
        )

    @classmethod
    def create_from_args(cls, args: Any) -> Optional["ImmiscibleNoiseHelper"]:
        if not bool(getattr(args, "enable_immiscible_diffusion", False)):
            return None
        return cls(
            enabled=True,
            mode=str(getattr(args, "immiscible_mode", "knn")),
            candidate_count=int(getattr(args, "immiscible_candidate_count", 4)),
            assignment_pool_factor=int(
                getattr(args, "immiscible_assignment_pool_factor", 2)
            ),
            distance_dtype=str(getattr(args, "immiscible_distance_dtype", "float32")),
            use_scipy=bool(getattr(args, "immiscible_use_scipy", True)),
            fallback_mode=str(getattr(args, "immiscible_fallback_mode", "knn")),
        )

    def setup_hooks(self) -> None:
        """No-op (kept for consistency with enhancement helper interfaces)."""

    def remove_hooks(self) -> None:
        """No-op (kept for consistency with enhancement helper interfaces)."""

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.randn_like(latents)
        if latents.dim() < 2:
            return torch.randn_like(latents)

        try:
            if self.mode == "knn":
                return self._sample_noise_knn(latents)
            if self.mode == "linear_assignment":
                return self._sample_noise_linear_assignment(latents, pool_factor=1)
            if self.mode == "linear_assignment_candidates":
                return self._sample_noise_linear_assignment(
                    latents, pool_factor=self.assignment_pool_factor
                )
            return self._sample_fallback(latents)
        except Exception as exc:
            if not self._warned_fallback:
                logger.warning(
                    "Immiscible Diffusion noise sampling failed; falling back to iid noise: %s",
                    exc,
                )
                self._warned_fallback = True
            return self._sample_fallback(latents)

    def _sample_noise_knn(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.shape[0]
        candidates = self.candidate_count
        noise = torch.randn(
            (batch_size, candidates, *latents.shape[1:]),
            device=latents.device,
            dtype=latents.dtype,
        )

        latents_flat = self._to_compute_dtype(latents.reshape(batch_size, -1))
        noise_flat = self._to_compute_dtype(noise.reshape(batch_size, candidates, -1))

        latents_norm = (latents_flat * latents_flat).sum(dim=1, keepdim=True)
        noise_norm = (noise_flat * noise_flat).sum(dim=2)
        dot = torch.einsum("bd,bkd->bk", latents_flat, noise_flat)
        dist_sq = torch.clamp(noise_norm + latents_norm - 2.0 * dot, min=0.0)
        min_index = torch.argmin(dist_sq, dim=1)
        row_index = torch.arange(batch_size, device=latents.device)
        return noise[row_index, min_index]

    def _sample_noise_linear_assignment(
        self, latents: torch.Tensor, pool_factor: int
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        pool_size = batch_size * max(1, int(pool_factor))
        noise_pool = torch.randn(
            (pool_size, *latents.shape[1:]),
            device=latents.device,
            dtype=latents.dtype,
        )
        dist_sq = self._pairwise_dist_sq(latents, noise_pool)
        col_index = self._assign_columns(dist_sq)
        row_index = torch.arange(batch_size, device=latents.device)
        return noise_pool[col_index[row_index]]

    def _pairwise_dist_sq(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs_flat = self._to_compute_dtype(lhs.reshape(lhs.shape[0], -1))
        rhs_flat = self._to_compute_dtype(rhs.reshape(rhs.shape[0], -1))
        lhs_norm = (lhs_flat * lhs_flat).sum(dim=1, keepdim=True)
        rhs_norm = (rhs_flat * rhs_flat).sum(dim=1).unsqueeze(0)
        dot = torch.matmul(lhs_flat, rhs_flat.transpose(0, 1))
        return torch.clamp(lhs_norm + rhs_norm - 2.0 * dot, min=0.0)

    def _assign_columns(self, dist_sq: torch.Tensor) -> torch.Tensor:
        if self.use_scipy:
            scipy_col = self._hungarian_with_scipy(dist_sq)
            if scipy_col is not None:
                return scipy_col
        return self._greedy_bipartite_assignment(dist_sq)

    def _hungarian_with_scipy(self, dist_sq: torch.Tensor) -> Optional[torch.Tensor]:
        if not is_scipy_available():
            if not self._warned_scipy:
                logger.warning(
                    "Immiscible linear assignment requested with scipy, but scipy is unavailable. "
                    "Using greedy assignment fallback."
                )
                self._warned_scipy = True
            return None
        try:
            from scipy.optimize import linear_sum_assignment

            _, col_ind = linear_sum_assignment(dist_sq.detach().float().cpu().numpy())
            return torch.as_tensor(col_ind, device=dist_sq.device, dtype=torch.long)
        except Exception as exc:
            if not self._warned_scipy:
                logger.warning(
                    "SciPy Hungarian assignment failed (%s). Using greedy assignment fallback.",
                    exc,
                )
                self._warned_scipy = True
            return None

    def _greedy_bipartite_assignment(self, dist_sq: torch.Tensor) -> torch.Tensor:
        rows, cols = dist_sq.shape
        inf = torch.finfo(dist_sq.dtype).max
        work = dist_sq.clone()
        row_taken = torch.zeros(rows, dtype=torch.bool, device=dist_sq.device)
        col_taken = torch.zeros(cols, dtype=torch.bool, device=dist_sq.device)
        row_to_col = torch.zeros(rows, dtype=torch.long, device=dist_sq.device)

        steps = min(rows, cols)
        for _ in range(steps):
            masked = work.clone()
            masked[row_taken, :] = inf
            masked[:, col_taken] = inf
            flat_idx = torch.argmin(masked)
            best = masked.view(-1)[flat_idx]
            if not torch.isfinite(best):
                break
            row = flat_idx // cols
            col = flat_idx % cols
            row_to_col[row] = col
            row_taken[row] = True
            col_taken[col] = True

        # For any row not covered (possible when cols < rows), allow reuse.
        if not bool(torch.all(row_taken)):
            free_rows = torch.nonzero(~row_taken, as_tuple=False).flatten()
            if free_rows.numel() > 0:
                fallback_cols = torch.argmin(dist_sq[free_rows], dim=1)
                row_to_col[free_rows] = fallback_cols
        return row_to_col

    def _sample_fallback(self, latents: torch.Tensor) -> torch.Tensor:
        if self.fallback_mode == "knn":
            return self._sample_noise_knn(latents)
        return torch.randn_like(latents)

    def _to_compute_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        compute_dtype = _DTYPE_MAP[self.distance_dtype]
        if tensor.device.type == "cpu" and compute_dtype == torch.float16:
            compute_dtype = torch.float32
        return tensor.to(dtype=compute_dtype)
