"""Vendored RamTorch Linear module with Takenoko-specific fixes.

This module mirrors ``ramtorch.modules.linear`` but is maintained locally so we can
apply dtype fixes without touching the third-party package installation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dedicated CUDA stream and events used for asynchronously moving weights between
# host and device memory. This reproduces RamTorch's CPU "bouncing" linear layer
# behavior where weights live on the CPU and are copied to the GPU only during
# forward/backward passes.
TRANSFER_STREAM = torch.cuda.Stream()

TRANSFER_FORWARD_FINISHED_EVENT = torch.cuda.Event()
COMPUTE_FORWARD_START_EVENT = torch.cuda.Event()
W_BUFFERS = [None, None]
B_BUFFERS = [None, None]

TRANSFER_BACKWARD_FINISHED_EVENT = torch.cuda.Event()
COMPUTE_BACKWARD_START_EVENT = torch.cuda.Event()
W_GRAD_BUFFERS = [None, None]

FORWARD_BUFFER_CLK = 0
BACKWARD_BUFFER_CLK = 0


class BouncingLinearFn(torch.autograd.Function):
    """Autograd implementation of the CPU-bouncing linear layer."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        global TRANSFER_STREAM, TRANSFER_FORWARD_FINISHED_EVENT, COMPUTE_FORWARD_START_EVENT
        global FORWARD_BUFFER_CLK, W_BUFFERS, B_BUFFERS

        selected_buffer = FORWARD_BUFFER_CLK

        with torch.cuda.stream(TRANSFER_STREAM):
            TRANSFER_STREAM.wait_event(COMPUTE_FORWARD_START_EVENT)
            W_BUFFERS[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            B_BUFFERS[selected_buffer] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            FORWARD_BUFFER_CLK ^= 1
            TRANSFER_FORWARD_FINISHED_EVENT.record()

        torch.cuda.current_stream().wait_event(TRANSFER_FORWARD_FINISHED_EVENT)
        COMPUTE_FORWARD_START_EVENT.record()
        out = F.linear(x, W_BUFFERS[selected_buffer], B_BUFFERS[selected_buffer])

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        global TRANSFER_STREAM, TRANSFER_BACKWARD_FINISHED_EVENT, COMPUTE_BACKWARD_START_EVENT
        global BACKWARD_BUFFER_CLK, W_GRAD_BUFFERS

        selected_buffer = BACKWARD_BUFFER_CLK

        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device

        with torch.cuda.stream(TRANSFER_STREAM):
            TRANSFER_STREAM.wait_event(COMPUTE_BACKWARD_START_EVENT)
            W_GRAD_BUFFERS[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            BACKWARD_BUFFER_CLK ^= 1
            TRANSFER_BACKWARD_FINISHED_EVENT.record()

        torch.cuda.current_stream().wait_event(TRANSFER_BACKWARD_FINISHED_EVENT)
        COMPUTE_BACKWARD_START_EVENT.record()

        weight_buffer = W_GRAD_BUFFERS[selected_buffer]
        weight_dtype = weight_buffer.dtype if weight_buffer is not None else grad_out.dtype

        # Mixed-precision guard: cast operands before matmul, then restore orig dtype.
        grad_out_cast = grad_out.to(weight_dtype)
        x_cast = x.to(weight_dtype)

        grad_input = grad_out_cast @ weight_buffer
        if grad_input.dtype != x.dtype:
            grad_input = grad_input.to(x.dtype)

        grad_weight = (grad_out_cast.mT @ x_cast).to("cpu")
        grad_bias = grad_out_cast.sum(dim=0).to("cpu") if bias_cpu is not None else None
        return grad_input, grad_weight, grad_bias, None


class CPUBouncingLinear(nn.Module):
    """Linear layer that stores parameters on CPU and streams them on demand."""

    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device="cpu").share_memory_().pin_memory()
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device="cpu").share_memory_().pin_memory())
            if bias
            else None
        )

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)


Linear = CPUBouncingLinear


__all__ = ["Linear", "CPUBouncingLinear", "BouncingLinearFn"]
