"""RangerVA Optimizer - PyTorch implementation.

RangerVA - Ranger with various improvements.
Combines RAdam with Lookahead and softplus transformer.

Copyright 2025 NoteDance. Ported to PyTorch for Takenoko.
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional
import math


class Softplus:
    """Softplus activation function."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        """
        Args:
            beta: Controls the smoothness of the Softplus function
            threshold: Threshold value to avoid overflow for large inputs
        """
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softplus activation."""
        if self.beta != 1.0:
            x = x * self.beta
        result = torch.where(
            x > self.threshold,
            x,  # Approximation for large inputs
            torch.log1p(torch.exp(x)),
        )
        if self.beta != 1.0:
            result = result / self.beta
        return result


class RangerVA(Optimizer):
    """RangerVA optimizer.

    Combines RAdam with Lookahead and various transformation options.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        beta1 (float, optional): Coefficient for computing running averages of
            gradient (default: 0.95)
        beta2 (float, optional): Coefficient for computing running averages of
            gradient squared (default: 0.999)
        epsilon (float, optional): Term added to denominator to improve
            numerical stability (default: 1e-5)
        weight_decay (float, optional): Weight decay coefficient (default: 0)
        alpha (float, optional): Lookahead blending coefficient (default: 0.5)
        k (int, optional): Lookahead sync interval (default: 6)
        n_sma_threshhold (int, optional): Threshold for N SMA (default: 5)
        amsgrad (bool, optional): Whether to use the AMSGrad variant
            (default: True)
        transformer (str, optional): Type of transformer to use ('softplus' or None)
            (default: 'softplus')
        smooth (float, optional): Smoothing parameter for softplus (default: 50)
        grad_transformer (str, optional): Gradient transformation ('square' or 'abs')
            (default: 'square')
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.95,
        beta2: float = 0.999,
        epsilon: float = 1e-5,
        weight_decay: float = 0.0,
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshhold: int = 5,
        amsgrad: bool = True,
        transformer: str = "softplus",
        smooth: float = 50,
        grad_transformer: str = "square",
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
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            alpha=alpha,
            k=k,
            n_sma_threshhold=n_sma_threshhold,
            amsgrad=amsgrad,
            transformer=transformer,
            smooth=smooth,
            grad_transformer=grad_transformer,
        )
        super().__init__(params, defaults)

        if transformer == "softplus":
            self.softplus = Softplus(beta=smooth, threshold=20.0)
        else:
            self.softplus = None

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
                    raise RuntimeError("RangerVA does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradients
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    # Exponential moving average of squared gradients
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.float32
                        )
                    # Slow buffer for lookahead
                    state["slow_buffer"] = p.clone().detach()

                # Cast grad to float32
                if grad.dtype != torch.float32:
                    grad = grad.float()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                step = state["step"]
                state["step"] += 1

                # Compute variance moving avg
                exp_avg_sq.mul_(group["beta2"]).add_(torch.square(grad), alpha=1 - group["beta2"])

                if group["grad_transformer"] == "square":
                    grad_tmp = torch.square(grad)
                elif group["grad_transformer"] == "abs":
                    grad_tmp = torch.abs(grad)
                else:
                    grad_tmp = torch.square(grad)

                exp_avg_sq.mul_(group["beta2"]).add_(grad_tmp, alpha=1 - group["beta2"])

                if group["amsgrad"]:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denomc = max_exp_avg_sq
                else:
                    denomc = exp_avg_sq

                if group["grad_transformer"] == "square":
                    denomc = torch.sqrt(denomc)

                # Compute mean moving avg
                exp_avg.mul_(group["beta1"]).add_(grad, alpha=1 - group["beta1"])

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p_fp32 = p.float() if p.dtype != torch.float32 else p
                    p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                    if p.dtype != torch.float32:
                        p.copy_(p_fp32)

                bias_correction1 = 1 - group["beta1"] ** (step + 1)
                bias_correction2 = 1 - group["beta2"] ** (step + 1)
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Apply transformer
                p_fp32 = p.float() if p.dtype != torch.float32 else p
                if group["transformer"] == "softplus" and self.softplus is not None:
                    denomf = self.softplus.forward(denomc)
                    p_fp32.add_(exp_avg * (-step_size / denomf))
                else:
                    denom = torch.sqrt(exp_avg_sq) + group["epsilon"]
                    p_fp32.add_(exp_avg * (-step_size * group["lr"] / denom))

                if p.dtype != torch.float32:
                    p.copy_(p_fp32)

                # Integrated lookahead
                if (step + 1) % group["k"] == 0:
                    slow_p = state["slow_buffer"]
                    slow_p.add_(p - slow_p, alpha=group["alpha"])
                    p.copy_(slow_p)

        return loss
