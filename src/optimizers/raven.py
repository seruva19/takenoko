"""RavenAdamW optimizer.

A memory-efficient AdamW variant that stores optimizer states on CPU and uses
reusable GPU buffers during updates. Features optional gradient centralization
and configurable bias correction strength.

Key features:
- CPU-offloaded momentum states (BF16 for exp_avg, FP32 for exp_avg_sq)
- Reusable GPU buffers eliminate per-parameter GPU memory overhead
- Gradient centralization for conv/linear layers improves convergence
- Configurable debiasing strength for controlled bias correction
- Fully compatible with Hugging Face Accelerate checkpointing

Memory profile:
- Optimizer states: ~3 bytes per parameter (CPU: 2 bytes + 4 bytes)
- GPU overhead: 8 bytes per largest parameter (reusable across all params)
- Example: 14B parameters → ~42GB CPU, ~112MB GPU (vs ~56GB GPU for standard AdamW)

Usage example:
    optimizer_type = "RavenAdamW"
    optimizer_args = [
        "weight_decay=0.01",              # Decoupled weight decay (default: 0.01)
        "betas=(0.9,0.999)",              # Momentum coefficients (default: (0.9, 0.999))
        "eps=1e-8",                       # Numerical stability term (default: 1e-8)
        "debias_strength=1.0",            # Bias correction strength 0.0-1.0 (default: 1.0)
                                          # 1.0 = full correction, 0.0 = no correction
        "use_grad_centralization=false",  # Enable gradient centering for conv/linear (default: false)
                                          # Subtracts mean gradient per output filter/neuron
        "gc_alpha=1.0"                    # Gradient centralization mixing 0.0-1.0 (default: 1.0)
    ]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class RavenAdamW(Optimizer):
    """AdamW with CPU-offloaded states and optional gradient centralization.

    Parameters
    ----------
    params : Iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-4
        Learning rate.
    betas : tuple[float, float], default (0.9, 0.999)
        Coefficients for computing running averages of gradient and its square.
    weight_decay : float, default 0.01
        Decoupled weight decay coefficient.
    eps : float, default 1e-8
        Term added to denominator for numerical stability.
    debias_strength : float, default 1.0
        Controls bias correction strength in [0.0, 1.0]. 1.0 = full correction,
        0.0 = no correction. Intermediate values provide partial correction.
    use_grad_centralization : bool, default False
        If True, center gradients for parameters with dim > 1. This subtracts
        the mean gradient per output filter (conv) or output neuron (linear).
    gc_alpha : float, default 1.0
        Gradient centralization mixing factor in [0.0, 1.0]. 1.0 = full centering.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        debias_strength: float = 1.0,
        use_grad_centralization: bool = False,
        gc_alpha: float = 1.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= debias_strength <= 1.0:
            raise ValueError(
                f"debias_strength must be between 0.0 and 1.0, got {debias_strength}"
            )
        if use_grad_centralization and not 0.0 <= gc_alpha <= 1.0:
            raise ValueError(f"gc_alpha must be in [0, 1], got {gc_alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            debias_strength=debias_strength,
            use_grad_centralization=use_grad_centralization,
            gc_alpha=gc_alpha,
        )
        super(RavenAdamW, self).__init__(params, defaults)

        # Determine parameter device and maximum parameter size
        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if self.param_device is None:
                        self.param_device = p.device
                    max_param_size = max(max_param_size, p.numel())

        # Create reusable GPU buffers sized for the largest parameter
        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(
                max_param_size, device=self.param_device, dtype=torch.float32
            )
            self.reusable_exp_avg_sq_gpu = torch.zeros(
                max_param_size, device=self.param_device, dtype=torch.float32
            )
            logger.info(
                f"RavenAdamW: Allocated {max_param_size * 8 / 1024**2:.2f} MB GPU memory for reusable buffers"
            )
        else:
            self.reusable_exp_avg_gpu = None
            self.reusable_exp_avg_sq_gpu = None
            logger.warning("RavenAdamW: No trainable parameters found")

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Parameters
        ----------
        closure : Optional[Callable], default None
            A closure that reevaluates the model and returns the loss.

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
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay: float = group["weight_decay"]
            eps: float = group["eps"]
            debias_strength: float = group["debias_strength"]
            use_gc: bool = group["use_grad_centralization"]
            gc_alpha: float = group["gc_alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Ensure reusable buffers are on the same device as current parameter
                # (Accelerate may move parameters after optimizer creation)
                param_device = p.device
                if self.reusable_exp_avg_gpu.device != param_device:
                    logger.debug(
                        f"Moving reusable buffers from {self.reusable_exp_avg_gpu.device} to {param_device}"
                    )
                    self.reusable_exp_avg_gpu = self.reusable_exp_avg_gpu.to(
                        param_device
                    )
                    self.reusable_exp_avg_sq_gpu = self.reusable_exp_avg_sq_gpu.to(
                        param_device
                    )
                    self.param_device = param_device

                grad = p.grad.float()

                if grad.is_sparse:
                    raise RuntimeError("RavenAdamW does not support sparse gradients.")

                # Gradient centralization for conv/linear layers
                if use_gc and grad.dim() > 1:
                    if grad.dim() >= 3:  # Conv layers
                        # Center each output filter: mean over (in_channels, height, width)
                        grad_mean = grad.mean(
                            dim=tuple(range(1, grad.dim())), keepdim=True
                        )
                    else:  # Linear layers (2D)
                        # Center each output neuron: mean over input dimension
                        grad_mean = grad.mean(dim=1, keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)

                state = self.state[p]
                num_param_elements = p.numel()

                # State initialization (CPU-stored, BF16 for exp_avg, FP32 for exp_avg_sq)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        device="cpu",
                        dtype=torch.bfloat16,
                    )
                    state["exp_avg_sq_cpu"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        device="cpu",
                        dtype=torch.float32,
                    )

                state["step"] += 1
                step: int = int(state["step"])

                exp_avg_cpu: torch.Tensor = state["exp_avg_cpu"]
                exp_avg_sq_cpu: torch.Tensor = state["exp_avg_sq_cpu"]

                # Create views into reusable GPU buffers
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[
                    :num_param_elements
                ].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[
                    :num_param_elements
                ].view_as(p)

                # Transfer states from CPU to GPU (non-blocking for efficiency)
                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Work in FP32 for numerical stability
                p_fp32 = p.to(torch.float32)

                # Decoupled weight decay (applied before momentum update)
                if weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)

                # Update biased first and second moment estimates
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Configurable bias correction
                bias_correction1 = 1.0
                bias_correction2 = 1.0
                if debias_strength > 0:
                    bias_correction1 -= math.pow(beta1, step) * debias_strength
                    bias_correction2 -= math.pow(beta2, step) * debias_strength

                step_size = lr / bias_correction1 if bias_correction1 != 0 else lr

                bias_correction2_sqrt = (
                    math.sqrt(bias_correction2) if bias_correction2 > 0 else 1.0
                )
                denom = (exp_avg_sq_gpu_view.sqrt() / bias_correction2_sqrt).add_(eps)

                # Apply parameter update
                p_fp32.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)

                # Copy back to parameter (preserves original dtype)
                p.copy_(p_fp32)

                # Transfer updated states back to CPU (non-blocking)
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

        # Final sync to ensure all transfers complete before next step
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state dict with reusable buffers.

        Overrides base method to include RavenAdamW-specific reusable GPU buffers
        and device information. Required for proper checkpoint saving with Accelerate.

        Returns
        -------
        dict[str, Any]
            State dict containing optimizer state and RavenAdamW-specific buffers.
        """
        state_dict = super().state_dict()

        # Store reusable buffers on CPU to minimize checkpoint size
        if self.reusable_exp_avg_gpu is not None:
            state_dict["reusable_exp_avg_gpu"] = self.reusable_exp_avg_gpu.clone().cpu()
        else:
            state_dict["reusable_exp_avg_gpu"] = None

        if self.reusable_exp_avg_sq_gpu is not None:
            state_dict["reusable_exp_avg_sq_gpu"] = (
                self.reusable_exp_avg_sq_gpu.clone().cpu()
            )
        else:
            state_dict["reusable_exp_avg_sq_gpu"] = None

        # Store device as string for cross-device compatibility
        state_dict["param_device"] = (
            str(self.param_device) if self.param_device else None
        )

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict and restore reusable buffers.

        Overrides base method to restore RavenAdamW-specific reusable GPU buffers
        and device information. Required for proper checkpoint loading with Accelerate.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State dict to load, as returned by state_dict().
        """
        # Extract and restore reusable buffers
        if "reusable_exp_avg_gpu" in state_dict:
            buffer = state_dict.pop("reusable_exp_avg_gpu")
            if buffer is not None:
                # Move to target device (handles device mismatch on resume)
                target_device = (
                    self.param_device
                    if self.param_device is not None
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.reusable_exp_avg_gpu = buffer.to(target_device)
            else:
                self.reusable_exp_avg_gpu = None

        if "reusable_exp_avg_sq_gpu" in state_dict:
            buffer = state_dict.pop("reusable_exp_avg_sq_gpu")
            if buffer is not None:
                target_device = (
                    self.param_device
                    if self.param_device is not None
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.reusable_exp_avg_sq_gpu = buffer.to(target_device)
            else:
                self.reusable_exp_avg_sq_gpu = None

        # Restore device information
        if "param_device" in state_dict:
            device_str = state_dict.pop("param_device")
            if device_str:
                self.param_device = torch.device(device_str)

        # Load base optimizer state (param groups, per-parameter states)
        super().load_state_dict(state_dict)

        logger.info("RavenAdamW: Successfully restored optimizer state from checkpoint")
