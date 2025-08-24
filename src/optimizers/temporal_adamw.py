"""TemporalAdamW optimizer.

An AdamW variant with temporal smoothing and optional adaptive momentum/noise
adaptation tailored for video/temporal training. Designed to be drop-in with
parameter-group usage already present in Takenoko (e.g., LoRA groups created
by `LoRANetwork.prepare_optimizer_params`).

Key features:
- Temporal gradient smoothing via EMA over recent gradients
- Adaptive momentum based on gradient consistency (cosine similarity)
- Noise adaptation using EMA statistics of gradients (mean and variance)

Notes:
- This optimizer intentionally does not guess LoRA parameters. Use param-group
  learning rates to scale LoRA groups as already done in the codebase.

Example (Takenoko config TOML):
    [train]
    optimizer_type = "TemporalAdamW"
    optimizer_args = [
      "betas=(0.9, 0.999)",
      "eps=1e-8",
      "weight_decay=0.01",
      "temporal_smoothing=0.30",
      "adaptive_momentum=True",
      "consistency_interval=2",
      "noise_adaptation=True",
      "noise_ema_alpha=0.10",
      "warmup_steps=300",
    ]
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple
import math

import torch
from torch.optim.optimizer import Optimizer


class TemporalAdamW(Optimizer):
    """AdamW with temporal smoothing and optional adaptive behaviors.

    Parameters
    ----------
    params : Iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-3
        Base learning rate.
    betas : tuple[float, float], default (0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
    eps : float, default 1e-8
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default 0.01
        Decoupled weight decay coefficient.
    temporal_smoothing : float, default 0.3
        EMA coefficient for temporal gradient smoothing in [0, 1]. Higher means
        smoother (more inertia). Set 0 to disable.
    adaptive_momentum : bool, default True
        If True, adapt beta1 using gradient consistency (cosine similarity).
    consistency_interval : int, default 1
        Compute momentum consistency every k steps for efficiency.
    noise_adaptation : bool, default True
        If True, use EMA of gradient mean and variance to scale updates.
    noise_ema_alpha : float, default 0.1
        EMA factor for gradient mean/variance (0 < alpha <= 1).
    warmup_steps : int, default 100
        Steps to warm up noise estimation before applying noise scaling.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        temporal_smoothing: float = 0.3,
        adaptive_momentum: bool = True,
        consistency_interval: int = 1,
        noise_adaptation: bool = True,
        noise_ema_alpha: float = 0.1,
        warmup_steps: int = 100,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= temporal_smoothing <= 1.0:
            raise ValueError(f"Invalid temporal_smoothing: {temporal_smoothing}")
        if consistency_interval <= 0:
            raise ValueError("consistency_interval must be >= 1")
        if not (0.0 < noise_ema_alpha <= 1.0):
            raise ValueError("noise_ema_alpha must be in (0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            temporal_smoothing=temporal_smoothing,
            adaptive_momentum=adaptive_momentum,
            consistency_interval=consistency_interval,
            noise_adaptation=noise_adaptation,
            noise_ema_alpha=noise_ema_alpha,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:  # pragma: no cover - compat
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Returns
        -------
        Optional[torch.Tensor]
            The loss if a closure was provided, else None.
        """
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            temporal_smoothing: float = group["temporal_smoothing"]
            adaptive_momentum: bool = group["adaptive_momentum"]
            consistency_interval: int = group["consistency_interval"]
            noise_adaptation: bool = group["noise_adaptation"]
            noise_ema_alpha: float = group["noise_ema_alpha"]
            warmup_steps: int = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "TemporalAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization (allocate only core states; feature states are lazy)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # For correct bias correction with time-varying beta1
                    state["beta1_cumprod"] = 1.0  # python float
                    # Cache last momentum used (for stability when consistency cannot be computed)
                    state["beta1_effective"] = beta1

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step: int = int(state["step"])

                # Temporal smoothing (EMA of gradients)
                if temporal_smoothing > 0.0:
                    # Lazily allocate smoothed_grad buffer
                    smoothed_grad_buf: Optional[torch.Tensor] = state.get("smoothed_grad")  # type: ignore[assignment]
                    if smoothed_grad_buf is None:
                        smoothed_grad_buf = torch.zeros_like(p)
                        state["smoothed_grad"] = smoothed_grad_buf
                    smoothed_grad_buf.mul_(temporal_smoothing).add_(
                        grad, alpha=1.0 - temporal_smoothing
                    )
                    g_used = smoothed_grad_buf
                else:
                    g_used = grad

                # Adaptive momentum via gradient consistency (cosine similarity)
                beta1_eff = beta1
                if adaptive_momentum and step % consistency_interval == 0:
                    # Lazily allocate prev_grad buffer
                    prev_grad: Optional[torch.Tensor] = state.get("prev_grad")  # type: ignore[assignment]
                    if prev_grad is None:
                        prev_grad = torch.zeros_like(p)
                        state["prev_grad"] = prev_grad
                    # Compute cosine similarity only when both grads have non-zero norms
                    denom = (grad.norm() * prev_grad.norm()).clamp_min(1e-12)
                    if denom > 0:
                        cos_sim = torch.dot(grad.flatten(), prev_grad.flatten()) / denom
                        # Map from [-1, 1] to [0.5, 1.5]
                        momentum_scale = 1.0 + 0.5 * cos_sim.item()
                        # Clamp effective beta1 to a safe range
                        beta1_eff = float(
                            min(0.999, max(0.5 * beta1, beta1 * momentum_scale))
                        )
                        state["beta1_effective"] = beta1_eff
                    else:
                        beta1_eff = float(state["beta1_effective"])  # fallback
                else:
                    beta1_eff = (
                        float(state["beta1_effective"]) if adaptive_momentum else beta1
                    )

                # Update biased first and second moment estimates with g_used
                exp_avg.mul_(beta1_eff).add_(g_used, alpha=1.0 - beta1_eff)
                exp_avg_sq.mul_(beta2).addcmul_(g_used, g_used, value=1.0 - beta2)

                # Maintain cumulative product for bias correction with time-varying beta1
                state["beta1_cumprod"] *= beta1_eff
                bias_correction1 = 1.0 - float(state["beta1_cumprod"])
                bias_correction2 = 1.0 - beta2**step

                # Noise adaptation using EMA statistics (after warmup)
                if noise_adaptation:
                    # Lazily allocate EMA stats
                    grad_mean_ema: Optional[torch.Tensor] = state.get("grad_mean_ema")  # type: ignore[assignment]
                    grad_sq_mean_ema: Optional[torch.Tensor] = state.get("grad_sq_mean_ema")  # type: ignore[assignment]
                    if grad_mean_ema is None:
                        grad_mean_ema = torch.zeros_like(p)
                        state["grad_mean_ema"] = grad_mean_ema
                    if grad_sq_mean_ema is None:
                        grad_sq_mean_ema = torch.zeros_like(p)
                        state["grad_sq_mean_ema"] = grad_sq_mean_ema
                    # Track mean and mean of squares with EMA
                    grad_mean_ema.mul_(1.0 - noise_ema_alpha).add_(
                        grad, alpha=noise_ema_alpha
                    )
                    grad_sq_mean_ema.mul_(1.0 - noise_ema_alpha).addcmul_(
                        grad, grad, value=noise_ema_alpha
                    )

                if noise_adaptation and step > warmup_steps:
                    # var ≈ E[g^2] - (E[g])^2
                    grad_var = (grad_sq_mean_ema - grad_mean_ema.pow(2)).clamp_min(0.0)
                    # inverse relative noise scale, clipped for stability
                    noise_scale = torch.sqrt(
                        (grad_mean_ema.abs() + eps) / (grad_var + eps)
                    )
                    noise_scale = noise_scale.clamp(0.1, 10.0)
                else:
                    noise_scale = None

                # Compute step size and denominator
                step_size = (
                    group["lr"] / bias_correction1 if bias_correction1 != 0.0 else 0.0
                )
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Decoupled weight decay (not bias-corrected)
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # Apply update; incorporate noise scaling elementwise if present
                if noise_scale is not None:
                    # Use elementwise scaling on the numerator
                    p.addcdiv_(exp_avg.mul(noise_scale), denom, value=-step_size)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                # Store gradient for next iteration consistency (only if used)
                if adaptive_momentum:
                    prev_grad = state.get("prev_grad")
                    if prev_grad is None:
                        prev_grad = torch.zeros_like(p)
                        state["prev_grad"] = prev_grad
                    prev_grad.copy_(grad)

        return loss
