"""Kron Optimizer - PyTorch implementation.

Kron: PSGD with Kronecker-factored preconditioner.
Implements algorithm from https://github.com/lixilinx/psgd_torch

Copyright 2025 NoteDance. Ported to PyTorch for Takenoko.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, List, Tuple, Callable
import string
import numpy as np


class Kron(Optimizer):
    """Kron optimizer with Kronecker-factored preconditioner.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 3e-4)
        weight_decay (float, optional): Weight decay coefficient (default: 0.0)
        b1 (float, optional): Momentum coefficient (default: 0.9)
        preconditioner_update_probability (float or callable, optional):
            Probability of updating preconditioner. If callable, should take
            step number and return probability. If None, uses default schedule.
            (default: None)
        max_size_triangular (int, optional): Maximum size for triangular matrices
            in preconditioner (default: 8192)
        min_ndim_triangular (int, optional): Minimum tensor dimension to use
            triangular preconditioner (default: 2)
        memory_save_mode (str, optional): Memory saving mode ('smart_one_diag',
            'one_diag', 'all_diag', or None) (default: None)
        momentum_into_precond_update (bool, optional): Whether to use momentum
            in preconditioner updates (default: True)
        precond_lr (float, optional): Learning rate for preconditioner updates
            (default: 0.1)
        precond_init_scale (float, optional): Initial scale for preconditioner
            (default: 1.0)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        b1: float = 0.9,
        preconditioner_update_probability: Optional[float] = None,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[str] = None,
        momentum_into_precond_update: bool = True,
        precond_lr: float = 0.1,
        precond_init_scale: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if b1 < 0.0 or b1 > 1.0:
            raise ValueError(f"Invalid b1: {b1}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            b1=b1,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
        )
        super().__init__(params, defaults)

        self._global_step = 0
        self._prob_step = 0
        self._update_counter = 0

        # Set default preconditioner update schedule if not provided
        if preconditioner_update_probability is None:
            self.precond_update_prob_schedule = self._precond_update_prob_schedule()
        else:
            self.precond_update_prob_schedule = preconditioner_update_probability

    @staticmethod
    def _precond_update_prob_schedule(
        max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.001, flat_start: int = 500
    ) -> Callable[[int], float]:
        """Default preconditioner update probability schedule."""

        def _schedule(n: int) -> float:
            """Exponential anneal with flat start."""
            prob = max_prob * np.exp(-decay * (n - flat_start))
            return max(min_prob, min(max_prob, prob))

        return _schedule

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

        self._global_step += 1
        global_step = self._global_step

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Kron does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Momentum buffer
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # Preconditioner Q and expressions
                    state["Q"], state["exprs"] = self._init_Q_exprs(
                        p,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                    )

                momentum_buffer = state["momentum_buffer"]
                Q = state["Q"]
                exprs = state["exprs"]

                # Update momentum
                momentum_buffer.mul_(group["b1"]).add_(grad, alpha=1 - group["b1"])

                # Compute debiased momentum
                debiased_momentum = momentum_buffer / (1 - group["b1"] ** global_step)

                # Get preconditioner update probability
                if callable(self.precond_update_prob_schedule):
                    update_prob = self.precond_update_prob_schedule(self._prob_step)
                else:
                    update_prob = self.precond_update_prob_schedule

                # Check if we should update preconditioner
                self._update_counter += 1
                do_update = self._update_counter >= 1.0 / update_prob
                if do_update:
                    self._update_counter = 0
                self._prob_step += 1

                # Balance preconditioners occasionally (approximately every 100 updates)
                balance = do_update and (torch.rand(()) < 0.01) and len(grad.shape) > 1
                if balance:
                    self._balance_Q(Q)

                # Update preconditioner
                if do_update:
                    if group["momentum_into_precond_update"]:
                        grad_for_precond = debiased_momentum
                    else:
                        grad_for_precond = grad

                    self._update_precond(
                        Q,
                        exprs,
                        grad_for_precond,
                        group["precond_lr"],
                    )

                # Precondition gradients
                pre_grad = self._precond_grad(Q, exprs, debiased_momentum)

                # Clip update RMS
                pre_grad = self._clip_update_rms(pre_grad)

                # Apply weight decay
                if group["weight_decay"] != 0 and len(p.shape) >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])

                # Apply update
                p.add_(pre_grad, alpha=-group["lr"])

        return loss

    def _init_Q_exprs(
        self,
        t: torch.Tensor,
        scale: float,
        max_size: int,
        min_ndim_triangular: int,
        memory_save_mode: Optional[str],
    ) -> Tuple[List[torch.Tensor], Tuple]:
        """Initialize preconditioner Q and reusable einsum expressions."""
        letters = string.ascii_lowercase + string.ascii_uppercase

        shape = t.shape
        dtype = t.dtype

        if len(shape) == 0:  # scalar
            Q = [torch.ones_like(t, dtype=dtype) * scale]
            exprA = ",->"
            exprGs = (",->",)
            exprP = ",,->"
        else:  # tensor
            if len(shape) > 13:
                raise ValueError(
                    f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
                )

            scale = scale ** (1 / len(shape))

            # Determine which dimensions should use diagonal preconditioners
            if memory_save_mode is None:
                dim_diag = [False for _ in shape]
            elif memory_save_mode == "smart_one_diag":
                rev_sorted_dims = np.argsort(shape)[::-1]
                dim_diag = [False for _ in shape]
                sorted_shape = sorted(shape)
                if len(shape) > 1 and sorted_shape[-1] > sorted_shape[-2]:
                    dim_diag[rev_sorted_dims[0]] = True
            elif memory_save_mode == "one_diag":
                rev_sorted_dims = np.argsort(shape)[::-1]
                dim_diag = [False for _ in shape]
                dim_diag[rev_sorted_dims[0]] = True
            elif memory_save_mode == "all_diag":
                dim_diag = [True for _ in shape]
            else:
                raise ValueError(
                    f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                    "[None, 'smart_one_diag', 'one_diag', 'all_diag']"
                )

            Q = []
            piece1A, piece2A, piece3A = ([], "", "")
            exprGs = []
            piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

            for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
                if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
                    # Use diagonal preconditioner
                    Q.append(torch.ones(size, dtype=dtype, device=t.device) * scale)

                    piece1A.append(letters[i])
                    piece2A = piece2A + letters[i]
                    piece3A = piece3A + letters[i]

                    piece1 = "".join(
                        [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                    )
                    subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                    exprGs.append(subscripts)

                    piece1P.append(letters[i + 13])
                    piece2P.append(letters[i + 13])
                    piece3P = piece3P + letters[i + 13]
                    piece4P = piece4P + letters[i + 13]
                else:
                    # Use triangular preconditioner
                    Q.append(torch.eye(size, dtype=dtype, device=t.device) * scale)

                    piece1A.append(letters[i] + letters[i + 13])
                    piece2A = piece2A + letters[i + 13]
                    piece3A = piece3A + letters[i]

                    piece1 = "".join(
                        [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                    )
                    piece2 = "".join(
                        [(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))]
                    )
                    subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                    exprGs.append(subscripts)

                    a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                    piece1P.append(a + b)
                    piece2P.append(a + c)
                    piece3P = piece3P + c
                    piece4P = piece4P + b

            exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
            exprP = (
                ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
            )

            exprGs = tuple(exprGs)

        return Q, (exprA, exprGs, exprP)

    def _balance_Q(self, Q_in: List[torch.Tensor]):
        """Balance preconditioner matrices."""
        norms = torch.stack([torch.norm(q, float("inf")) for q in Q_in])
        geometric_mean = torch.exp(torch.mean(torch.log(norms)))
        norms = geometric_mean / norms
        for i, q in enumerate(Q_in):
            Q_in[i].mul_(norms[i])

    def _lb(self, A: torch.Tensor, max_abs: float) -> torch.Tensor:
        """Cheap lower bound for the spectral norm of A."""
        A = A / max_abs
        a0 = torch.einsum("ij,ij->j", A, A)
        i = torch.argmax(a0)
        x = torch.reshape(A[:, i], [-1])
        x = torch.einsum("i,ij->j", x, A)
        x = x / torch.norm(x)
        x = torch.einsum("j,kj->k", x, A)
        x = torch.norm(x)
        x = x * max_abs
        return x

    def _solve_triangular_right(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Solve X @ inv(A) where A is upper triangular."""
        orig_dtype = A.dtype
        return torch.reshape(
            torch.linalg.solve_triangular(
                A.T, torch.reshape(X, [-1, X.shape[-1]]).T, upper=False, left=True
            ).T,
            X.shape,
        ).to(orig_dtype)

    def _calc_A_and_conjB(
        self, exprA: str, G: torch.Tensor, Q: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate A and conjB for preconditioner update."""
        order = G.ndim
        V = torch.randn_like(G)
        eps = torch.tensor(torch.finfo(G.dtype).eps, dtype=G.dtype, device=G.device)
        G = G + torch.sqrt(eps) * torch.mean(torch.abs(G)) * V

        if order > 0:
            # Move first dimension to last: (d0, d1, ..., dn) -> (d1, ..., dn, d0)
            conjB = torch.permute(V, list(range(1, order)) + [0])
            for i, q in enumerate(Q):
                if q.ndim < 2:
                    conjB = conjB / q
                else:
                    conjB = self._solve_triangular_right(conjB, q)
                if i < order - 1:
                    perm = list(range(order))
                    perm[i], perm[order - 1] = perm[order - 1], perm[i]
                    conjB = torch.permute(conjB, perm)
        else:
            conjB = V

        A = torch.einsum(exprA, *(Q + [G]))
        return A, conjB

    def _update_precond(
        self, Q: List[torch.Tensor], exprs: Tuple, G: torch.Tensor, step: float
    ):
        """Update Kronecker product preconditioner Q."""
        exprA, exprGs, _ = exprs
        A, conjB = self._calc_A_and_conjB(exprA, G, Q)

        for i, (q, exprG) in enumerate(zip(Q, exprGs)):
            term1 = torch.einsum(exprG, A, A)
            term2 = torch.einsum(exprG, conjB, conjB)
            term1, term2 = term1 - term2, term1 + term2
            term1 = term1 * step

            norm = torch.norm(term2, float("inf"))
            tiny = torch.finfo(q.dtype).tiny

            if q.ndim < 2:
                term1 = term1 * q / torch.maximum(norm, torch.tensor(tiny, device=q.device))
            else:
                term1 = torch.tril(term1)
                norm_bound = torch.where(
                    norm > 0, self._lb(term2, norm), norm
                )
                term1 = term1 / torch.maximum(norm_bound, torch.tensor(tiny, device=q.device))
                term1 = torch.matmul(term1, q)

            Q[i].copy_(q - term1)

    def _precond_grad(self, Q: List[torch.Tensor], exprs: Tuple, G: torch.Tensor) -> torch.Tensor:
        """Precondition gradient G with preconditioner Q."""
        return torch.einsum(exprs[-1], *(Q + Q + [G]))

    def _clip_update_rms(self, g: torch.Tensor) -> torch.Tensor:
        """Clip update by RMS."""
        rms = torch.sqrt(torch.mean(torch.square(g))) + 1e-12
        factor = torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            torch.tensor(1.1, dtype=g.dtype, device=g.device) / rms,
        )
        return g * factor
