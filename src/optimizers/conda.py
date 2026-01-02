"""Conda Optimizer - PyTorch implementation.

Conda: Memory-efficient optimizer with GaLore-style gradient projection.
https://arxiv.org/abs/2509.24218

Copyright 2025 NoteDance. Ported to PyTorch for Takenoko.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, List
import math


class GaLoreProjector:
    """GaLore projector for gradient compression."""

    def __init__(
        self,
        rank: Optional[int] = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: str = "std",
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
        self.ortho_matrix = None
        self.last_svd_step = -1

    @staticmethod
    def get_orthogonal_matrix(
        weights: torch.Tensor,
        rank: Optional[int],
        projection_type: str,
        from_random_matrix: bool = False,
    ):
        if projection_type not in {"right", "left", "full"}:
            raise ValueError("projection_type should be one of left, right or full")

        original_type = weights.dtype
        is_float = original_type == torch.float32
        matrix = weights if is_float else weights.float()

        if from_random_matrix:
            if not isinstance(rank, int):
                raise TypeError("rank should be int when from_random_matrix is True")
            m, n = matrix.shape[0], matrix.shape[1]
            u = torch.randn((m, rank), device=matrix.device, dtype=matrix.dtype) / math.sqrt(rank)
            vh = torch.randn((rank, n), device=matrix.device, dtype=matrix.dtype) / math.sqrt(rank)
        else:
            u, _, vh = torch.linalg.svd(matrix, full_matrices=False)

        if projection_type == "right":
            b = vh if rank is None else vh[:rank, :]
            return b if is_float else b.to(dtype=original_type)
        if projection_type == "left":
            a = u if rank is None else u[:, :rank]
            return a if is_float else a.to(dtype=original_type)

        a = u if rank is None else u[:, :rank]
        b = vh if rank is None else vh[:rank, :]
        if is_float:
            return (a, b)
        return (a.to(dtype=original_type), b.to(dtype=original_type))

    def get_low_rank_grad_std(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        if grad.shape[0] >= grad.shape[1]:
            return torch.matmul(grad, ortho_matrix.t())
        return torch.matmul(ortho_matrix.t(), grad)

    def get_low_rank_grad_reverse_std(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        if grad.shape[0] >= grad.shape[1]:
            return torch.matmul(ortho_matrix.t(), grad)
        return torch.matmul(grad, ortho_matrix.t())

    def get_low_rank_grad_right(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        return torch.matmul(grad, ortho_matrix.t())

    def get_low_rank_grad_left(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        return torch.matmul(ortho_matrix.t(), grad)

    def get_low_rank_grad_full(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        a, b = ortho_matrix
        return torch.matmul(torch.matmul(a.t(), grad), b.t())

    def get_low_rank_grad_random(self, grad, ortho_matrix):
        ortho_matrix = self.ortho_matrix if ortho_matrix is None else ortho_matrix
        if grad.shape[0] >= grad.shape[1]:
            return torch.matmul(grad, ortho_matrix.t())
        return torch.matmul(ortho_matrix.t(), grad)

    def update_ortho_matrix(self, x, from_random_matrix: bool):
        is_right = x.shape[0] >= x.shape[1]

        if self.projection_type == "std":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="right" if is_right else "left",
                from_random_matrix=from_random_matrix,
            )
        if self.projection_type == "reverse_std":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="left" if is_right else "right",
                from_random_matrix=from_random_matrix,
            )
        if self.projection_type == "right":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="right",
                from_random_matrix=from_random_matrix,
            )
        if self.projection_type == "left":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="left",
                from_random_matrix=from_random_matrix,
            )
        if self.projection_type == "full":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="full",
                from_random_matrix=from_random_matrix,
            )
        if self.projection_type == "random":
            return self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type="right" if is_right else "left",
                from_random_matrix=from_random_matrix,
            )
        raise NotImplementedError(f"unsupported projection_type: {self.projection_type}")

    def project(self, grad, num_steps: int, svd_basis_matrix=None, from_random_matrix: bool = False):
        if num_steps % self.update_proj_gap == 0 and num_steps != self.last_svd_step:
            basis = grad if svd_basis_matrix is None else svd_basis_matrix
            new_ortho = self.update_ortho_matrix(basis, from_random_matrix=from_random_matrix)
            self.last_svd_step = int(num_steps)
        else:
            new_ortho = self.ortho_matrix

        if self.projection_type != "full":
            self.ortho_matrix = new_ortho
        else:
            a, b = new_ortho
            self.ortho_matrix = (a, b)

        if self.projection_type == "std":
            return self.get_low_rank_grad_std(grad, None)
        if self.projection_type == "reverse_std":
            return self.get_low_rank_grad_reverse_std(grad, None)
        if self.projection_type == "right":
            return self.get_low_rank_grad_right(grad, None)
        if self.projection_type == "left":
            return self.get_low_rank_grad_left(grad, None)
        if self.projection_type == "full":
            return self.get_low_rank_grad_full(grad, None)
        if self.projection_type == "random":
            return self.get_low_rank_grad_random(grad, None)
        raise NotImplementedError

    def project_(self, grad, ortho_matrix=None):
        if self.projection_type == "std":
            return self.get_low_rank_grad_std(grad, ortho_matrix)
        if self.projection_type == "reverse_std":
            return self.get_low_rank_grad_reverse_std(grad, ortho_matrix)
        if self.projection_type == "right":
            return self.get_low_rank_grad_right(grad, ortho_matrix)
        if self.projection_type == "left":
            return self.get_low_rank_grad_left(grad, ortho_matrix)
        if self.projection_type == "full":
            return self.get_low_rank_grad_full(grad, ortho_matrix)
        if self.projection_type == "random":
            return self.get_low_rank_grad_random(grad, ortho_matrix)
        raise NotImplementedError

    def project_back(self, low_rank_grad):
        if self.projection_type == "std":
            return (
                torch.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else torch.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale
        if self.projection_type == "reverse_std":
            return (
                torch.matmul(self.ortho_matrix, low_rank_grad)
                if low_rank_grad.shape[0] > low_rank_grad.shape[1]
                else torch.matmul(low_rank_grad, self.ortho_matrix)
            ) * self.scale
        if self.projection_type == "right":
            return torch.matmul(low_rank_grad, self.ortho_matrix) * self.scale
        if self.projection_type == "left":
            return torch.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        if self.projection_type == "full":
            return torch.matmul(
                torch.matmul(self.ortho_matrix[0], low_rank_grad), self.ortho_matrix[1]
            ) * self.scale
        if self.projection_type == "random":
            return (
                torch.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else torch.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale
        raise NotImplementedError


class Conda(Optimizer):
    """Conda optimizer.

    Memory-efficient optimizer with GaLore-style gradient projection.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        beta1 (float, optional): Coefficient for computing running averages of
            gradient (default: 0.9)
        beta2 (float, optional): Coefficient for computing running averages of
            gradient squared (default: 0.999)
        epsilon (float, optional): Term added to denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay coefficient (default: 0.0)
        update_proj_gap (int, optional): How often to update projection matrix.
            If None, no projection is used (default: None)
        scale (float, optional): Scale factor for gradient projection (default: None)
        projection_type (str, optional): Type of projection ('std', 'reverse_std',
            'right', 'left', 'full') (default: None)
        maximize (bool, optional): Whether to maximize instead of minimize
            (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        update_proj_gap: Optional[int] = None,
        scale: Optional[float] = None,
        projection_type: Optional[str] = None,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta1 < 0.0 or beta1 > 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0.0 or beta2 > 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            update_proj_gap=update_proj_gap,
            scale=scale,
            projection_type=projection_type,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Conda does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradients
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradients
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Initialize GaLore projector for 2D tensors
                    if group["update_proj_gap"] is not None and len(p.shape) == 2:
                        projector = GaLoreProjector(
                            rank=None,
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"] if group["scale"] is not None else 1.0,
                            projection_type=group["projection_type"]
                            if group["projection_type"] is not None
                            else "std",
                        )
                        projector.ortho_matrix = projector.update_ortho_matrix(p, from_random_matrix=False)
                        state["projector"] = projector

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - group["beta1"] ** step
                bias_correction2_sq = torch.sqrt(
                    torch.tensor(1 - group["beta2"] ** step, device=p.device)
                )
                step_size = group["lr"] * bias_correction2_sq / bias_correction1

                if group["maximize"]:
                    grad = -grad

                # Update first moment estimate
                exp_avg.mul_(group["beta1"]).add_(grad, alpha=1 - group["beta1"])

                # Project gradient and momentum if using GaLore
                if group["update_proj_gap"] is not None and len(p.shape) == 2 and "projector" in state:
                    projector = state["projector"]
                    grad_projected = projector.project(grad, step, exp_avg)
                    exp_avg_projected = projector.project(exp_avg, step)
                else:
                    grad_projected = grad
                    exp_avg_projected = exp_avg

                # Update second moment estimate
                exp_avg_sq.mul_(group["beta2"]).add_(
                    torch.square(grad_projected), alpha=1 - group["beta2"]
                )

                # Compute normalized gradient
                denom = torch.sqrt(exp_avg_sq) + group["epsilon"]
                norm_grad = exp_avg_projected / denom

                # Project back if using GaLore
                if group["update_proj_gap"] is not None and len(p.shape) == 2 and "projector" in state:
                    norm_grad = state["projector"].project_back(norm_grad)

                # Apply update
                p.add_(norm_grad, alpha=-step_size)

                # Apply weight decay
                p.mul_(1 - group["lr"] * group["weight_decay"])

        return loss
