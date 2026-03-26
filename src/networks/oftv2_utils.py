"""OFTv2 rotation helpers adapted to Takenoko's WAN adapter modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiplicativeDropoutLayer(nn.Module):
    """Replaces sampled OFT blocks with identity during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x

        if x.ndim != 3 or x.shape[-1] != x.shape[-2]:
            raise ValueError(
                "MultiplicativeDropoutLayer expects [blocks, block_size, block_size] input."
            )

        num_blocks, block_size, _ = x.shape
        if num_blocks <= 1:
            return x

        keep_prob = 1.0 - self.p
        stochastic_mask = torch.empty(
            num_blocks,
            1,
            1,
            device=x.device,
            dtype=x.dtype,
        ).bernoulli_(p=keep_prob)
        eye_matrix = torch.eye(block_size, device=x.device, dtype=x.dtype).repeat(
            num_blocks, 1, 1
        )
        return stochastic_mask * x + (1.0 - stochastic_mask) * eye_matrix


class OFTRotationModule(nn.Module):
    """Block-diagonal orthogonal rotation used by OFT/OFTv2."""

    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        coft: bool = False,
        coft_eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.r = int(r)
        self.n_elements = int(n_elements)
        self.block_size = int(block_size)
        self.in_features = int(in_features)
        self.coft = bool(coft)
        self.coft_eps = float(coft_eps)
        self.block_share = bool(block_share)
        self.use_cayley_neumann = bool(use_cayley_neumann)
        self.num_cayley_neumann_terms = int(max(1, num_cayley_neumann_terms))
        self.weight = nn.Parameter(torch.empty(self.r, self.n_elements))
        rows, cols = torch.triu_indices(self.block_size, self.block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)

    def _pytorch_skew_symmetric(
        self, vec: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        batch_size = int(vec.shape[0])
        matrix = torch.zeros(
            batch_size,
            block_size,
            block_size,
            device=vec.device,
            dtype=vec.dtype,
        )
        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, self.rows, self.cols), vec)
        return matrix - matrix.transpose(-2, -1)

    def _pytorch_skew_symmetric_inv(
        self, matrix: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        del block_size
        return matrix[:, self.rows, self.cols]

    def _cayley_batch(
        self,
        q: torch.Tensor,
        block_size: int,
        use_cayley_neumann: bool = True,
        num_neumann_terms: int = 5,
    ) -> torch.Tensor:
        batch_size, _ = q.shape
        previous_dtype = q.dtype
        q_skew = self._pytorch_skew_symmetric(q, block_size)

        if use_cayley_neumann:
            rotation = torch.eye(
                block_size,
                device=q.device,
                dtype=q.dtype,
            ).repeat(batch_size, 1, 1)
            if num_neumann_terms > 1:
                rotation.add_(q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    q_squared = torch.bmm(q_skew, q_skew)
                    rotation.add_(q_squared, alpha=2.0)

                    q_power = q_squared
                    for _ in range(3, num_neumann_terms - 1):
                        q_power = torch.bmm(q_power, q_skew)
                        rotation.add_(q_power, alpha=2.0)
                    q_power = torch.bmm(q_power, q_skew)
                    rotation.add_(q_power)
        else:
            identity = (
                torch.eye(block_size, device=q_skew.device, dtype=q_skew.dtype)
                .unsqueeze(0)
                .expand(batch_size, block_size, block_size)
            )
            rotation = torch.linalg.solve(
                identity + q_skew,
                identity - q_skew,
                left=False,
            )

        return rotation.to(previous_dtype)

    def _project_batch(
        self,
        q: torch.Tensor,
        coft_eps: float = 1e-4,
    ) -> torch.Tensor:
        oft_r = self._pytorch_skew_symmetric(q, self.block_size)
        scaled_eps = float(coft_eps) / math.sqrt(float(max(1, oft_r.shape[0])))
        origin_matrix = torch.zeros(
            (oft_r.size(1), oft_r.size(1)),
            device=oft_r.device,
            dtype=oft_r.dtype,
        ).unsqueeze(0).expand_as(oft_r)
        diff = oft_r - origin_matrix
        norm_diff = torch.norm(diff, dim=(1, 2), keepdim=True).clamp_min(1e-12)
        mask = (norm_diff <= scaled_eps).bool()
        projected = torch.where(
            mask,
            oft_r,
            origin_matrix + scaled_eps * (diff / norm_diff),
        )
        return self._pytorch_skew_symmetric_inv(projected, self.block_size)

    def build_rotation(
        self,
        weight_override: torch.Tensor | None = None,
        block_share_override: bool | None = None,
        use_cayley_neumann_override: bool | None = None,
        num_cayley_neumann_terms_override: int | None = None,
        coft_override: bool | None = None,
        coft_eps_override: float | None = None,
    ) -> torch.Tensor:
        weight = self.weight if weight_override is None else weight_override
        block_share = (
            self.block_share
            if block_share_override is None
            else bool(block_share_override)
        )
        use_cayley_neumann = (
            self.use_cayley_neumann
            if use_cayley_neumann_override is None
            else bool(use_cayley_neumann_override)
        )
        num_terms = (
            self.num_cayley_neumann_terms
            if num_cayley_neumann_terms_override is None
            else int(max(1, num_cayley_neumann_terms_override))
        )
        coft = self.coft if coft_override is None else bool(coft_override)
        coft_eps = self.coft_eps if coft_eps_override is None else float(coft_eps_override)

        effective_weight = weight
        if coft:
            effective_weight = self._project_batch(effective_weight, coft_eps=coft_eps)

        rotation = self._cayley_batch(
            effective_weight,
            self.block_size,
            use_cayley_neumann,
            num_terms,
        )
        rotation = self.dropout(rotation)

        if block_share:
            rank = int(self.in_features // self.block_size)
            rotation = rotation.repeat(rank, 1, 1)

        return rotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        if self.coft:
            with torch.no_grad():
                self.weight.copy_(self._project_batch(self.weight, coft_eps=self.coft_eps))

        rotation = self.build_rotation()
        original_shape = x.shape
        batch_dims = x.shape[:-1]
        rank = self.in_features // self.block_size if self.block_share else self.r
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)
        x_rotated = torch.einsum("...rk,rkc->...rc", x_reshaped, rotation)
        return x_rotated.reshape(*original_shape).to(required_dtype)
