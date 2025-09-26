import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .gradient_projection import GradientProjector


class FiraAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in
    [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future "
                "version. Use the PyTorch implementation torch.optim.AdamW instead, or set "
                "`no_deprecation_warning=True` to disable this warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # Gradient Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GradientProjector(
                            group["rank"],
                            update_proj_gap=group.get("update_proj_gap", 200),
                            alpha=group.get("alpha", 1.0),
                            proj_type=group.get("proj_type", "std"),
                        )
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                norm_grad = exp_avg / denom

                if "rank" in group:
                    subgrad = state["projector"].project_back(grad)
                    norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                    scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (
                        torch.norm(grad, dim=norm_dim) + 1e-8
                    )
                    if norm_dim == 1:
                        scaling_factor = scaling_factor.unsqueeze(1)
                    scaling_grad = (p.grad - subgrad) * scaling_factor

                    if "scaling_grad" in state:
                        scaling_grad_norm = torch.norm(scaling_grad)
                        limiter = (
                            max(
                                scaling_grad_norm / (state["scaling_grad"] + 1e-8),
                                1.01,
                            )
                            / 1.01
                        )
                        scaling_grad = scaling_grad / limiter
                        state["scaling_grad"] = scaling_grad_norm / limiter
                    else:
                        state["scaling_grad"] = torch.norm(scaling_grad)

                    norm_grad = (
                        state["projector"].project_back(norm_grad) + scaling_grad
                    )

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
