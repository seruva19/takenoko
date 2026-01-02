"""VSGD Optimizer - PyTorch implementation.

VSGD: Variance Stochastic Gradient Descent
https://arxiv.org/abs/2404.06549

Copyright 2025 NoteDance. Ported to PyTorch for Takenoko.
"""

import torch
from torch.optim import Optimizer
from typing import Optional


class VSGD(Optimizer):
    """VSGD optimizer.

    Variance Stochastic Gradient Descent with sophisticated variance tracking.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 1e-1)
        epsilon (float, optional): Term added to denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay coefficient (default: 0.0)
        weight_decouple (bool, optional): Whether to use decoupled weight decay
            (default: True)
        ghattg (float, optional): Parameter for variance tracking (default: 30.0)
        ps (float, optional): Parameter for variance tracking (default: 1e-8)
        tau1 (float, optional): Decay rate for bg (default: 0.81)
        tau2 (float, optional): Decay rate for bhg (default: 0.9)
        maximize (bool, optional): Whether to maximize instead of minimize
            (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        ghattg: float = 30.0,
        ps: float = 1e-8,
        tau1: float = 0.81,
        tau2: float = 0.9,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ghattg < 0.0:
            raise ValueError(f"Invalid ghattg: {ghattg}")
        if ps < 0.0:
            raise ValueError(f"Invalid ps: {ps}")
        if tau1 < 0.0 or tau1 > 1.0:
            raise ValueError(f"Invalid tau1: {tau1}")
        if tau2 < 0.0 or tau2 > 1.0:
            raise ValueError(f"Invalid tau2: {tau2}")

        pa2 = 2.0 * ps + 1.0 + 1e-4
        pbg2 = 2.0 * ps
        pbhg2 = 2.0 * ghattg * ps

        defaults = dict(
            lr=lr,
            epsilon=epsilon,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            ghattg=ghattg,
            ps=ps,
            tau1=tau1,
            tau2=tau2,
            maximize=maximize,
            pa2=pa2,
            pbg2=pbg2,
            pbhg2=pbhg2,
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
                    raise RuntimeError("VSGD does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Mean of gradients
                    state["mug"] = torch.zeros_like(p)
                    # Variance estimate of gradients
                    state["bg"] = torch.zeros_like(p)
                    # Variance estimate of gradients
                    state["bhg"] = torch.zeros_like(p)

                step = state["step"]
                state["step"] += 1

                mug = state["mug"]
                bg = state["bg"]
                bhg = state["bhg"]

                step_tensor = torch.tensor(step + 1, dtype=p.dtype, device=p.device)
                rho1 = torch.pow(step_tensor, -group["tau1"])
                rho2 = torch.pow(step_tensor, -group["tau2"])

                if group["maximize"]:
                    grad = -grad

                # Apply weight decay
                if group["weight_decouple"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                elif group["weight_decay"] > 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Initialize variance estimates on first step
                if step == 0:
                    sg = group["pbg2"] / (group["pa2"] - 1.0)
                    shg = group["pbhg2"] / (group["pa2"] - 1.0)
                else:
                    sg = bg / group["pa2"]
                    shg = bhg / group["pa2"]

                # Update mean estimate
                mug_prev = mug.clone()
                mug.copy_(mug * shg + grad * sg)
                mug.div_(sg + shg)

                # Compute variance of mean estimate
                sigg = (sg * shg) / (sg + shg)
                mug_sq = torch.square(mug) + sigg

                # Update variance estimates
                bg2 = group["pbg2"] + mug_sq - 2 * mug * mug_prev + torch.square(mug_prev)
                bhg2 = group["pbhg2"] + mug_sq - 2 * grad * mug + torch.square(grad)

                bg.mul_(1 - rho1).add_(bg2, alpha=rho1)
                bhg.mul_(1 - rho2).add_(bhg2, alpha=rho2)

                # Apply update
                denom = torch.sqrt(mug_sq) + group["epsilon"]
                p.add_(mug * (-group["lr"] / denom))

        return loss

    def reset(self):
        """Reset the optimizer state."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    self.state[p]["step"] = 0
                    self.state[p]["mug"].zero_()
                    self.state[p]["bg"].zero_()
                    self.state[p]["bhg"].zero_()
