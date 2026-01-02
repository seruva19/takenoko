"""NvNovoGrad Optimizer - PyTorch implementation.

Nvidia's NovoGrad implementation.
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
https://arxiv.org/abs/1905.11286

Original TensorFlow implementation by NoteDance.
Ported to PyTorch for Takenoko.
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional


class NvNovoGrad(Optimizer):
    """NvNovoGrad optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        beta1 (float, optional): Coefficient for computing running averages of
            gradient (default: 0.95)
        beta2 (float, optional): Coefficient for computing running averages of
            gradient squared (default: 0.98)
        epsilon (float, optional): Term added to denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay coefficient (default: 0)
        grad_averaging (bool, optional): Whether to apply gradient averaging
            (default: False)
        amsgrad (bool, optional): Whether to use the AMSGrad variant
            (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.95,
        beta2: float = 0.98,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        grad_averaging: bool = False,
        amsgrad: bool = False,
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
            grad_averaging=grad_averaging,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

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
                    raise RuntimeError("NvNovoGrad does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradients
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient norms
                    state["exp_avg_sq"] = torch.zeros(1, dtype=p.dtype, device=p.device)
                    if group["amsgrad"]:
                        # Max of exp_avg_sq
                        state["max_exp_avg_sq"] = torch.zeros(1, dtype=p.dtype, device=p.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                beta1, beta2 = group["beta1"], group["beta2"]

                # Compute gradient norm
                norm = torch.sum(torch.square(grad))

                # Update second moment estimate
                if exp_avg_sq.item() == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)

                if group["amsgrad"]:
                    # Maintains maximum of all 2nd moment running avg till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = torch.sqrt(max_exp_avg_sq) + group["epsilon"]
                else:
                    denom = torch.sqrt(exp_avg_sq) + group["epsilon"]

                # Normalize gradient
                grad = grad / denom

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Apply gradient averaging if specified
                if group["grad_averaging"]:
                    grad = grad.mul(1 - beta1)

                # Update first moment estimate
                exp_avg.mul_(beta1).add_(grad)

                # Apply update
                p.add_(exp_avg, alpha=-group["lr"])

                state["step"] += 1

        return loss
