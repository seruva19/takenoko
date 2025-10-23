# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import math
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from modules.ramtorch_linear_factory import make_linear, is_linear_like
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard  # type: ignore

from utils.lora_utils import load_safetensors_with_lora_and_fp8
from utils.safetensors_utils import MemoryEfficientSafeOpen
from utils.model_utils import create_cpu_offloading_wrapper

import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

from utils.tread.tread_routing_handlers import (
    TREADRoutingState,
    handle_routing_start,
    handle_routing_end,
)
from utils.tread.tread_token import (
    normalize_routes_with_neg_indices,
)
from wan.modules.attention import flash_attention
from utils.device_utils import clean_memory_on_device
from modules.custom_offloading_utils import ModelOffloader
from modules.fp8_optimization_utils import (
    apply_fp8_monkey_patch,
    optimize_state_dict_with_fp8,
)

from .attention import local_patching, local_merge, nablaT, sta
from utils.advanced_rope import apply_rope_comfy
from utils.dispersive_loss_utils import resolve_dispersive_target_block

from utils.tread.tread_router import TREADRouter
from wan.modules.lean_attention import forward_wan22_lean_block
from wan.utils.compile_utils import compile_optimize as wan_compile_optimize
from torch.nn.attention.flex_attention import flex_attention

try:
    flex = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True
    )
except Exception:
    # logger.warning("torch.compile failed to compile flex_attention")
    flex = flex_attention


def sinusoidal_embedding_1d(dim, position, use_float32=False):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32 if use_float32 else torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64/float32
def rope_params(max_seq_len, dim, theta=10000, use_float32=False):
    assert dim % 2 == 0
    dtype = torch.float32 if use_float32 else torch.float64
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(dtype).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
try:
    compiler_disable = torch.compiler.disable  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for older torch

    def compiler_disable(fn):  # type: ignore
        return fn


@compiler_disable
def rope_apply(x, grid_sizes, freqs, fractal=False):
    device_type = x.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):  # type: ignore
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
            )
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )

            if fractal:
                freqs_i = local_patching(freqs_i.unsqueeze(0), h, w, 8)
            freqs_i = freqs_i.reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()


def calculate_freqs_i(fhw, c, freqs):
    f, h, w = fhw
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_f = freqs[0][:f]
    freqs_i = torch.cat(
        [
            freqs_f.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs_i


# inplace version of rope_apply
@compiler_disable
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
    # with torch.amp.autocast(device_type=device_type, enabled=False):
    rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

    n, _ = x.size(2), x.size(3) // 2

    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(rope_dtype).reshape(seq_len, n, -1, 2)
        )
        freqs_i = freqs_list[i]

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # inplace update
        x[i, :seq_len] = x_i.to(x.dtype)

    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return self._norm(x.float()).type_as(x) * self.weight
        # support fp8
        return self._norm(x.float()).type_as(x) * self.weight.to(x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, C]
    #     """
    #     # inplace version, also supports fp8 -> does not have significant performance improvement
    #     original_dtype = x.dtype
    #     x = x.float()
    #     y = x.pow(2).mean(dim=-1, keepdim=True)
    #     y.add_(self.eps)
    #     y.rsqrt_()
    #     x *= y
    #     x = x.to(original_dtype)
    #     x *= self.weight.to(original_dtype)
    #     return x


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        attn_mode="torch",
        split_attn=False,
        sparse_algo=None,
        rope_on_the_fly: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.rope_on_the_fly = rope_on_the_fly

        # layers
        self.q = make_linear(dim, dim, True, tag="q")
        self.k = make_linear(dim, dim, True, tag="k")
        self.v = make_linear(dim, dim, True, tag="v")
        self.o = make_linear(dim, dim, True, tag="o")
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.sparse_algo = sparse_algo
        self.mask_func = self.construct_mask_func(sparse_algo) if sparse_algo else None
        # RoPE variant configurable via parent
        self.rope_func = "default"
        self.use_comfy_rope = False

    def construct_mask_func(self, sparse_algo):
        if "_" in sparse_algo:
            nabla_cfg, sta_cfg = sparse_algo.split("_")
            thr = float(nabla_cfg.split("-")[-1])
            wT, wH, wW = [int(cfg) for cfg in sta_cfg.split("-")[1:]]
            return lambda q, k, seq, T, H, W: nablaT(
                q, k, seq, T, H, W, thr=thr, wT=wT, wH=wH, wW=wW
            )

        if "nabla" in sparse_algo:
            thr = float(sparse_algo.split("-")[-1])
            return lambda q, k, seq, T, H, W: nablaT(
                q, k, seq, T, H, W, thr=thr, sta_att=0
            )
        elif "sta" in sparse_algo:
            wT, wH, wW = [int(cfg) for cfg in sparse_algo.split("-")[1:]]
            return lambda q, k, seq, T, H, W: sta(T, H, W, wT=wT, wH=wH, wW=wW)
        else:
            raise ValueError(f"Invalid sparse algorithm: {sparse_algo}")

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        sparse_attention: bool = False,
        batched_rotary: Optional[torch.Tensor] = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            sparse_attention (bool): Whether to use sparse attention (default: False)
        """
        # # query, key, value function
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v
        # q, k, v = qkv_fn(x)
        # del x
        # query, key, value function

        if sparse_attention:
            b, _, n, d = *x.shape[:2], self.num_heads, self.head_dim

            def qkv_fn(x):
                q = self.norm_q(self.q(x)).view(b, -1, n, d)
                k = self.norm_k(self.k(x)).view(b, -1, n, d)
                v = self.v(x).view(b, -1, n, d)
                return q, k, v

            q, k, v = qkv_fn(x)
            # This block is only executed if sparse attention is configured
            q_rope = (
                rope_apply(q, grid_sizes, freqs, fractal=True)
                .to(dtype=torch.bfloat16)
                .transpose(1, 2)
            )
            k_rope = (
                rope_apply(k, grid_sizes, freqs, fractal=True)
                .to(dtype=torch.bfloat16)
                .transpose(1, 2)
            )

            T, H, W = grid_sizes.tolist()[0]
            H, W = H // 8, W // 8
            seq = torch.tensor([0, T], dtype=torch.int32).to(q_rope.device)
            block_mask = self.mask_func(q_rope, k_rope, seq, T, H, W)  # type: ignore

            x = flex(
                q_rope, k_rope, v.transpose(1, 2), block_mask=block_mask
            ).transpose(  # type: ignore
                1, 2
            )  # type: ignore
        else:
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            del x
            q = self.norm_q(q)
            k = self.norm_k(k)
            q = q.view(b, s, n, d)
            k = k.view(b, s, n, d)
            v = v.view(b, s, n, d)

            # Only apply RoPE when not routing with batched rotary
            if batched_rotary is None:
                if (
                    bool(getattr(self, "use_comfy_rope", False))
                    and getattr(self, "rope_func", "default") == "comfy"
                ):
                    try:
                        q_rope, k_rope = self.comfyrope(q, k, freqs)  # type: ignore[attr-defined]
                        qkv = [q_rope, k_rope, v]
                    except Exception:
                        rope_apply_inplace_cached(q, grid_sizes, freqs)
                        rope_apply_inplace_cached(k, grid_sizes, freqs)
                        qkv = [q, k, v]
                elif self.rope_on_the_fly:
                    # freqs is expected to be base frequencies tensor when rope_on_the_fly=True
                    q_rope = rope_apply(q, grid_sizes, freqs)
                    k_rope = rope_apply(k, grid_sizes, freqs)
                    qkv = [q_rope, k_rope, v]
                else:
                    rope_apply_inplace_cached(q, grid_sizes, freqs)
                    rope_apply_inplace_cached(k, grid_sizes, freqs)
                    qkv = [q, k, v]
            else:
                qkv = [q, k, v]
            del q, k, v
            # Only pass k_lens for attention modes that support variable lengths (flash3/sageattn)
            k_lens_arg = seq_lens if self.attn_mode in ("flash3", "sageattn") else None
            x = flash_attention(
                qkv,
                k_lens=k_lens_arg,
                window_size=self.window_size,
                attn_mode=self.attn_mode,
                split_attn=self.split_attn,
                batched_rotary=batched_rotary,
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # k = self.norm_k(self.k(context)).view(b, -1, n, d)
        # v = self.v(context).view(b, -1, n, d)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        del context
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # compute attention
        # Optionally collect attention metrics with minimal overhead
        try:
            from common.attention_metrics import collect_cross_attention, should_collect

            if should_collect():
                # Use detached copies to avoid graph bloat; compute on head-major views
                q_det = q.detach().contiguous()  # [B, Lq, H, D]
                k_det = k.detach().contiguous()  # [B, Lk, H, D]
                collect_cross_attention(q_det, k_det)
        except Exception:
            logger.error("Failed to collect attention metrics")
            pass

        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(
            qkv,
            k_lens=context_lens,
            attn_mode=self.attn_mode,
            split_attn=self.split_attn,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode="torch",
        split_attn=False,
        sparse_algo=None,
        model_version="2.1",
        rope_on_the_fly: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.sparse_algo = sparse_algo
        self.model_version = model_version

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            attn_mode,
            split_attn,
            sparse_algo=sparse_algo,
            rope_on_the_fly=rope_on_the_fly,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, attn_mode, split_attn
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            make_linear(dim, ffn_dim, True, tag="ffn_in"),
            nn.GELU(approximate="tanh"),
            make_linear(ffn_dim, dim, True, tag="ffn_out"),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # attention compute dtype (fp32 by default; can be toggled via _lower_precision_attention)
        self.attention_dtype = torch.float32

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    @torch.compiler.disable()
    def get_modulation(self, e: torch.Tensor):
        """
        Construct per-gate modulation tensors s0..s5 for both 2.1 (sample-wise) and 2.2 (token-wise).

        e: [B, 6, C] (2.1) or [B, L, 6, C] (2.2)
        Returns: tuple(s0..s5) each shaped [B, C] (2.1) or [B, L, C] (2.2)
        """
        # Refresh dtype each call to honor runtime toggles
        self.attention_dtype = (
            torch.float16
            if bool(getattr(self, "_lower_precision_attention", False))
            else torch.float32
        )
        split_dim = e.ndim - 2  # 1 for 2.1 style modulation, 2 for 2.2 style modulation
        m = self.modulation.to(self.attention_dtype, copy=False)  # [1, 6, C]
        if e.ndim == 4:
            m = m.unsqueeze(0)  # [1, 1, 6, C] for broadcasting over tokens
        e = e.to(self.attention_dtype, copy=False)

        m0, m1, m2, m3, m4, m5 = m.unbind(dim=split_dim)
        e0, e1, e2, e3, e4, e5 = e.unbind(dim=split_dim)
        s0 = e0 + m0
        s1 = e1 + m1
        s2 = e2 + m2
        s3 = e3 + m3
        s4 = e4 + m4
        s5 = e5 + m5
        return s0, s1, s2, s3, s4, s5

    def _forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        sparse_attention: bool = False,
        batched_rotary: torch.Tensor | None = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] (v2.1) or [B, L, 6, C] (per-token, v2.2)
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            sparse_attention (bool): Whether to use sparse attention (default: False)
        """
        x_orig_dtype = x.dtype

        # Optional lean attention math to avoid large fp32 intermediates (2.2 only)
        if self.model_version != "2.1" and bool(
            getattr(self, "_lean_attn_math", False)
        ):
            x = forward_wan22_lean_block(
                x,
                e,
                self.modulation,
                self.norm1,
                self.norm2,
                self.norm3,
                self.self_attn,
                self.cross_attn,
                self.ffn,
                seq_lens,
                grid_sizes,
                None if batched_rotary is not None else freqs,
                context,
                context_lens,
                sparse_attention,
                batched_rotary,
                force_fp16=bool(getattr(self, "_lower_precision_attention", False)),
                fp32_default=bool(getattr(self, "_lean_attn_fp32_default", True)),
            )
        else:
            # Unified numerics for 2.1 and 2.2
            s0, s1, s2, s3, s4, s5 = self.get_modulation(e)
            del e

            # Self-attention
            q_in = self.norm1(x).to(self.attention_dtype, copy=False)
            fi1 = q_in.addcmul(q_in, s1).add(s0).contiguous()
            y = self.self_attn(
                fi1,
                seq_lens,
                grid_sizes,
                freqs if batched_rotary is None else None,
                sparse_attention=sparse_attention,
                batched_rotary=batched_rotary,  # type: ignore[arg-type]
            )
            x = x + (y * s2).to(x_orig_dtype, copy=False)
            del y

            # Cross-attention
            x = x + self.cross_attn(
                self.norm3(x).to(self.attention_dtype, copy=False),
                context,
                context_lens,
            )
            del context

            # FFN
            ff_in = self.norm2(x).to(self.attention_dtype, copy=False)
            y = self.ffn(
                (ff_in * (1 + s4) + s3).contiguous().to(x_orig_dtype, copy=False)
            )
            x = x + (y * s5).to(x_orig_dtype, copy=False)
            del y

        return x.to(x_orig_dtype, copy=False)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        sparse_attention: bool = False,
        batched_rotary: torch.Tensor | None = None,
    ):
        """
        Forward pass for WanAttentionBlock.

        Args:
            x (Tensor): Input tensor of shape [B, L, C]
            e (Tensor): Modulation tensor of shape [B, 6, C]
            seq_lens (Tensor): Sequence lengths [B]
            grid_sizes (Tensor): Grid sizes [B, 3]
            freqs (Tensor): Rotary embedding frequencies
            context (Tensor): Context tensor
            context_lens (Tensor): Context lengths
            sparse_attention (bool): Whether to use sparse attention (default: False)
        """
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(
                    forward_fn, self.modulation.device
                )
            return checkpoint(
                forward_fn,
                x,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                sparse_attention,
                batched_rotary,
                use_reentrant=False,
            )
        return self._forward(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            sparse_attention,
            batched_rotary,
        )


class Head(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        patch_size,
        eps=1e-6,
        model_version="2.1",
        lower_precision_attention: bool = False,
        simple_modulation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.model_version = model_version
        self.simple_modulation = simple_modulation
        self.attention_dtype = (
            torch.float16 if lower_precision_attention else torch.float32
        )

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = make_linear(dim, out_dim, True, tag="head")

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, C] for 2.1, [B, L, C] for 2.2
        """
        if self.model_version == "2.1" or self.simple_modulation:
            e = (
                self.modulation.to(self.attention_dtype, copy=False) + e.unsqueeze(1)
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        else:  # For Wan2.2
            e = (
                self.modulation.unsqueeze(0).to(self.attention_dtype, copy=False)
                + e.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))

        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = [
    "norm",
    "patch_embedding",
    "text_embedding",
    "time_embedding",
    "time_projection",
    "head",
    # "modulation",
    # modulation layers are large and benefit from block-wise scaling; allow FP8 by default
    "img_emb",
]


class WanModel(nn.Module):  # ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    # @register_to_config
    def __init__(
        self,
        model_version="2.1",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        attn_mode=None,
        split_attn=False,
        sparse_algo=None,
        use_fvdm: bool = False,
        rope_on_the_fly: bool = False,
        broadcast_time_embed: bool = False,
        strict_e_slicing_checks: bool = False,
        lower_precision_attention: bool = False,
        lean_attention_fp32_default: bool = True,
        simple_modulation: bool = False,
        optimized_torch_compile: bool = False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_version (`str`, *optional*, defaults to '2.1'):
                Version of the model, e.g., '2.1' or '2.2'. This is used to determine the modulation strategy.
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            sparse_algo (`str`, *optional*, defaults to None):
                Sparse attention algorithm, e.g. "nabla-0.5_sta-5-5-11"

        """

        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attn_mode = attn_mode if attn_mode is not None else "torch"
        self.split_attn = split_attn
        self.model_version = model_version
        # FVDM uses per-token modulation; force effective version to 2.2 when enabled
        self.use_fvdm = use_fvdm
        self.effective_model_version = "2.2" if self.use_fvdm else self.model_version

        # Warn about Wan 2.1 + FVDM compatibility issues
        if self.use_fvdm and self.model_version == "2.1":
            logger.warning(
                "⚠️  FVDM COMPATIBILITY WARNING: Training on Wan 2.1 with FVDM enabled\n"
                "   FVDM requires per-token timestep embeddings, forcing internal use of Wan 2.2 architecture.\n"
                "   This creates a training-inference mismatch:\n"
                "   - Training: Uses 2.2-style per-token embeddings (e.shape=[B,L,C])\n"
                "   - Inference: Standard 2.1 uses batch embeddings (e.shape=[B,C])\n"
                "   ⚡ RECOMMENDATION: Use FVDM only with Wan 2.2 models for full compatibility.\n"
                "   If training LORAs for Wan 2.1, test inference thoroughly before deployment."
            )
        # gated features
        self.rope_on_the_fly = rope_on_the_fly
        self.broadcast_time_embed = broadcast_time_embed
        self.strict_e_slicing_checks = strict_e_slicing_checks
        # VRAM/precision controls
        self._lower_precision_attention = bool(lower_precision_attention)
        self._simple_modulation = bool(simple_modulation)
        # Lean attention compute policy: fp32 by default unless overridden
        self._lean_attn_fp32_default = bool(lean_attention_fp32_default)
        if self._lower_precision_attention:
            logger.info(
                "Attention pre-calcs/e tensor in torch.float16 to save VRAM (lower_precision_attention)"
            )
        if self._simple_modulation and self.effective_model_version == "2.2":
            logger.info("Using simple (Wan 2.1 style) modulation strategy to save VRAM")
        self.e_dtype = (
            torch.float16 if self._lower_precision_attention else torch.float32
        )
        self.optimized_torch_compile = bool(optimized_torch_compile)
        self.compile_args: list | tuple | None = None

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            make_linear(text_dim, dim, True, tag="text_in"),
            nn.GELU(approximate="tanh"),
            make_linear(dim, dim, True, tag="text_out"),
        )

        # Optional RoPE variant flag (comfy), configured via loader
        self.rope_func = "default"

        self.time_embedding = nn.Sequential(
            make_linear(freq_dim, dim, True, tag="time_in"),
            nn.SiLU(),
            make_linear(dim, dim, True, tag="time_out"),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), make_linear(dim, dim * 6, True, tag="time_proj")
        )

        # blocks
        cross_attn_type = "t2v_cross_attn"
        self.sparse_algo = sparse_algo

        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    attn_mode,  # type: ignore
                    split_attn,
                    sparse_algo=sparse_algo,
                    model_version=self.effective_model_version,
                    rope_on_the_fly=self.rope_on_the_fly,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(
            dim,
            out_dim,
            patch_size,
            eps,
            model_version=self.effective_model_version,
            lower_precision_attention=self._lower_precision_attention,
            simple_modulation=self._simple_modulation,
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        # Get rope precision setting from model attributes
        rope_float32 = getattr(self, "rope_use_float32", False)
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6), use_float32=rope_float32),
                rope_params(1024, 2 * (d // 6), use_float32=rope_float32),
                rope_params(1024, 2 * (d // 6), use_float32=rope_float32),
            ],
            dim=1,
        )
        self.freqs_fhw = {}

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        # offloading
        self.blocks_to_swap = None
        self.offloader = None

        # TREAD routing (optional; set via set_router)
        self._tread_router: TREADRouter | None = None
        self._tread_routes: list[dict] | None = None

    def set_router(self, router: TREADRouter, routes: list[dict]) -> None:
        """Enable TREAD routing for this model.

        Args:
            router: TREADRouter instance
            routes: list of dicts with keys selection_ratio, start_layer_idx, end_layer_idx
        """
        self._tread_router = router
        self._tread_routes = routes or []

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.patch_embedding.weight.device

    # unused
    def fp8_optimization(
        self,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
        move_to_device: bool,
        use_scaled_mm: bool = False,
        exclude_ffn_from_scaled_mm: bool = False,
        scale_input_tensor: Optional[str] = None,
        upcast_linear: bool = False,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> int:
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
        """

        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(
            state_dict,
            device,
            FP8_OPTIMIZATION_TARGET_KEYS,
            FP8_OPTIMIZATION_EXCLUDE_KEYS,
            move_to_device=move_to_device,
            quant_dtype=quant_dtype,
            quantization_mode="block",
            block_size=64,
        )

        # apply monkey patching
        apply_fp8_monkey_patch(
            self,
            state_dict,
            use_scaled_mm=use_scaled_mm,
            exclude_ffn_from_scaled_mm=exclude_ffn_from_scaled_mm,
            scale_input_tensor=scale_input_tensor,
            upcast_linear=upcast_linear,
            quant_dtype=quant_dtype,
        )

        return state_dict  # type: ignore

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

        for block in self.blocks:  # type: ignore
            block.enable_gradient_checkpointing(activation_cpu_offloading)  # type: ignore

        logger.info(
            f"WanModel: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}"
        )

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        for block in self.blocks:  # type: ignore
            block.disable_gradient_checkpointing()  # type: ignore

        logger.info("WanModel: Gradient checkpointing disabled.")

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool,
        config_args: Optional[argparse.Namespace] = None,
    ):
        """Enable block swapping with optional enhanced offloading features.

        Args:
            blocks_to_swap: Number of blocks to swap
            device: Target device
            supports_backward: Whether to support backward pass
            config_args: Optional configuration namespace for enhanced offloading features
        """
        from modules.custom_offloading_utils import create_enhanced_model_offloader

        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)  # type: ignore

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        # Use factory function to support enhanced offloading features
        self.offloader = create_enhanced_model_offloader(
            block_type="wan_attn_block",
            blocks=self.blocks,  # type: ignore
            num_blocks=self.num_blocks,
            blocks_to_swap=self.blocks_to_swap,
            supports_backward=supports_backward,
            device=device,
            debug=False,
            config_args=config_args,
        )

        # Log enhanced features status if enabled
        enhanced_info = ""
        if config_args is not None:
            enhanced_enabled = getattr(config_args, "offload_enhanced_enabled", False)
            if enhanced_enabled:
                pinned_memory = getattr(
                    config_args, "offload_pinned_memory_enabled", False
                )
                enhanced_info = f" [Enhanced: pinned={pinned_memory}]"

        logger.info(
            f"WanModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. "
            f"Supports backward: {supports_backward}{enhanced_info}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)  # type: ignore
            self.prepare_block_swap_before_forward()
            logger.info("WanModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)  # type: ignore
            self.prepare_block_swap_before_forward()
            logger.info("WanModel: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,  # not used
        y=None,
        skip_block_indices=None,
        sparse_attention=False,
        force_keep_mask: torch.Tensor | None = None,
        controlnet_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        controlnet_weight: float = 1.0,
        controlnet_stride: int = 1,
        dispersive_loss_target_block: int | None = None,
        return_intermediate: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            skip_block_indices (List[int], *optional*):
                Indices of blocks to skip during forward pass
            sparse_attention (bool, *optional*):
                Whether to use sparse attention
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Optional selective compile for critical paths (safely gated)
        # Trigger optimized compile (one-shot) if enabled
        if bool(getattr(self, "optimized_torch_compile", False)):
            wan_compile_optimize(self)

        # remove assertions to work with Fun-Control T2V
        # if self.model_type == "i2v":
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            y = None

        # embeddings
        x = [
            self.patch_embedding(u.unsqueeze(0)) for u in x
        ]  # x[0].shape = [1, 5120, F, H, W]

        if sparse_attention:
            T, H, W = x[0].shape[2:5]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=t.device) for u in x]
        )  # list of [F, H, W] placed on same device as timesteps

        freqs_list = []
        if not self.rope_on_the_fly:
            for i, fhw in enumerate(grid_sizes):
                fhw = tuple(fhw.tolist())
                if fhw not in self.freqs_fhw:
                    c = self.dim // self.num_heads // 2
                    self.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.freqs)
                freqs_list.append(self.freqs_fhw[fhw])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert (
            seq_lens.max() <= seq_len
        ), f"Sequence length exceeds maximum allowed length {seq_len}. Got {seq_lens.max()}"
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):  # type: ignore
            if self.use_fvdm:
                # Vectorized timesteps path (FVDM/PUSA). Accept t of shape [B] or [B, F].
                # Build per-token embeddings without per-sample Python loops.
                B = seq_lens.numel()
                Fp = grid_sizes[:, 0]  # [B]
                Hp = grid_sizes[:, 1]
                Wp = grid_sizes[:, 2]
                patches_per_frame = Hp * Wp  # [B]
                L = Fp * patches_per_frame  # [B]

                # Construct per-frame timesteps flattened across batch
                if t.dim() == 1:
                    # Repeat each scalar timestep by the number of frames in that sample
                    t_frames_flat = t.repeat_interleave(Fp)
                elif t.dim() == 2 and (Fp == t.size(1)).all():
                    # Already per-frame aligned for all samples
                    t_frames_flat = t.reshape(-1)
                else:
                    # Rare mismatch: fall back to safe per-sample handling
                    # This path is uncommon and preserves correctness for ragged inputs
                    t_list = []
                    for i in range(B):
                        Fi = int(Fp[i].item())
                        if t.dim() == 1:
                            t_i = t[i].view(1).expand(Fi)
                        else:
                            t_i_full = t[i]
                            F_in = t_i_full.numel()
                            if F_in == Fi:
                                t_i = t_i_full
                            elif F_in > Fi:
                                t_i = t_i_full[:Fi]
                            else:
                                rep = (Fi + F_in - 1) // F_in
                                t_i = t_i_full.repeat(rep)[:Fi]
                        t_list.append(t_i)
                    t_frames_flat = torch.cat(t_list, dim=0)

                # Expand per-frame to per-token using per-frame repeats = patches_per_frame
                repeats_frames = patches_per_frame.repeat_interleave(Fp)
                t_tokens_flat = t_frames_flat.repeat_interleave(repeats_frames)

                # Compute time embeddings for all tokens at once
                rope_float32 = getattr(self, "rope_use_float32", False)
                e_tokens_flat = self.time_embedding(
                    sinusoidal_embedding_1d(
                        self.freq_dim, t_tokens_flat, use_float32=rope_float32
                    ).float()
                )  # [sum(L), dim]
                e0_tokens_flat = self.time_projection(e_tokens_flat).unflatten(
                    1, (6, self.dim)
                )  # [sum(L), 6, dim]

                # Prepare padded batch tensors [B, seq_len, ...]
                e = x.new_zeros((B, seq_len, self.dim), dtype=e_tokens_flat.dtype)
                e0 = x.new_zeros((B, seq_len, 6, self.dim), dtype=e_tokens_flat.dtype)

                # Build indices to scatter without Python loops
                # batch indices per token
                batch_idx_flat = torch.arange(B, device=x.device).repeat_interleave(L)
                # token positions within each sample
                token_pos_list = [
                    torch.arange(int(L[i].item()), device=x.device) for i in range(B)
                ]
                token_pos_flat = torch.cat(token_pos_list, dim=0)

                e[batch_idx_flat, token_pos_flat, :] = e_tokens_flat
                e0[batch_idx_flat, token_pos_flat, :, :] = e0_tokens_flat
            elif self.effective_model_version == "2.1" or self._simple_modulation:
                rope_float32 = getattr(self, "rope_use_float32", False)
                e = self.time_embedding(
                    sinusoidal_embedding_1d(
                        self.freq_dim, t, use_float32=rope_float32
                    ).float()
                ).to(self.e_dtype)
                e0 = (
                    self.time_projection(e).unflatten(1, (6, self.dim)).to(self.e_dtype)
                )
            else:  # For Wan2.2 (standard per-token path)
                if self.broadcast_time_embed:
                    # Use a single time embedding per sample (broadcast across tokens later)
                    if t.dim() == 2:
                        t_scalar = t[:, 0]
                    else:
                        t_scalar = t
                    rope_float32 = getattr(self, "rope_use_float32", False)
                    e = self.time_embedding(
                        sinusoidal_embedding_1d(
                            self.freq_dim, t_scalar, use_float32=rope_float32
                        ).float()
                    ).to(
                        self.e_dtype
                    )  # [B, dim]
                    e0 = (
                        self.time_projection(e)
                        .unflatten(1, (6, self.dim))
                        .to(self.e_dtype)
                    )  # [B, 6, dim]
                    e0 = e0.unsqueeze(
                        1
                    )  # [B, 1, 6, dim] will broadcast over tokens in block
                else:
                    if t.dim() == 1:
                        t = t.unsqueeze(1).expand(-1, seq_len)

                    bt = t.size(0)
                    t = t.flatten()
                    rope_float32 = getattr(self, "rope_use_float32", False)
                    e = self.time_embedding(
                        sinusoidal_embedding_1d(
                            self.freq_dim, t, use_float32=rope_float32
                        )
                        .unflatten(0, (bt, seq_len))
                        .float()
                    ).to(self.e_dtype)
                    e0 = (
                        self.time_projection(e)
                        .unflatten(2, (6, self.dim))
                        .to(self.e_dtype)
                    )
        # dtype assertions relaxed: e/e0 may be fp16 when lower_precision_attention

        # context
        context_lens = None
        if type(context) is list:
            context = torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        context = self.text_embedding(context)

        # i2v is not used

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)
        #     clip_fea = None
        #     context_clip = None

        # arguments
        # Choose frequency argument for attention per gate
        attn_freqs = self.freqs if self.rope_on_the_fly else freqs_list
        # Propagate rope_func/comfy to attention modules when requested
        rope_mode = getattr(self, "rope_func", "default")
        if rope_mode == "comfy":
            try:
                for blk in self.blocks:  # type: ignore[attr-defined]
                    try:
                        setattr(blk.self_attn, "rope_func", "comfy")
                        setattr(blk.self_attn, "comfyrope", apply_rope_comfy)
                        setattr(blk.self_attn, "use_comfy_rope", True)
                    except Exception:
                        pass
            except Exception:
                pass

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=attn_freqs,
            context=context,
            context_lens=context_lens,
            sparse_attention=sparse_attention,
        )

        if sparse_attention:
            P = 8
            x = x.reshape(1, T, H, W, -1)
            x = local_patching(x, H, W, P)
            x = x.reshape(1, H * W * T, -1)  # type: ignore

        if self.blocks_to_swap:
            clean_memory_on_device(device)

        # print(f"x: {x.shape}, e: {e0.shape}, context: {context.shape}, seq_lens: {seq_lens}")
        # --- TREAD routing state ---
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and torch.is_grad_enabled() and len(routes) > 0
        route_ptr = 0
        tread_state = TREADRoutingState()

        # Normalize negative indices in routes
        if use_routing and routes:
            total_layers = len(self.blocks)  # type: ignore
            routes = normalize_routes_with_neg_indices(routes, total_layers)

        # Normalize control states container to a list for indexing
        if controlnet_states is not None and isinstance(controlnet_states, tuple):
            controlnet_states = list(controlnet_states)

        # Optional intermediate capture for dispersive loss
        intermediate_z = None
        target_block_idx = resolve_dispersive_target_block(
            len(self.blocks), dispersive_loss_target_block
        )  # type: ignore[arg-type]

        # Track input device for consistency check when CPU offloading is enabled
        input_device = x.device

        for block_idx, block in enumerate(self.blocks):  # type: ignore
            is_block_skipped = (
                skip_block_indices is not None and block_idx in skip_block_indices
            )

            if self.blocks_to_swap and not is_block_skipped:
                self.offloader.wait_for_block(block_idx)  # type: ignore

            # ═══════════════════════════════════════════════════════════════════════════════
            # TREAD ROUTING START: Begin token routing at configured layer index
            # ═══════════════════════════════════════════════════════════════════════════════
            if (
                use_routing
                and route_ptr < len(routes)
                and block_idx == int(routes[route_ptr]["start_layer_idx"])  # type: ignore
            ):
                assert router is not None
                route_config = routes[route_ptr]
                x = handle_routing_start(
                    self,
                    x,
                    kwargs,
                    tread_state,
                    route_config,
                    router,
                    force_keep_mask,
                    freqs_list,
                )

            if not is_block_skipped:
                # Switch attention to use batched_rotary when routing
                tread_mode_mid = str(getattr(self, "_tread_mode", "full"))
                if tread_state.routing_now and tread_mode_mid.startswith("frame_"):
                    # Frame-based path uses standard freqs (already aligned by new grid)
                    x = block(x, **kwargs)
                elif (
                    tread_state.routing_now
                    and "batched_rotary" in kwargs
                    and tread_mode_mid == "full"
                ):
                    # We pass batched_rotary down via the attention wrapper
                    # by temporarily overriding freqs with None and adding batched_rotary
                    x = block(
                        x,
                        e=kwargs["e"],
                        seq_lens=kwargs["seq_lens"],
                        grid_sizes=kwargs["grid_sizes"],
                        freqs=None,
                        context=kwargs["context"],
                        context_lens=kwargs["context_lens"],
                        sparse_attention=kwargs["sparse_attention"],
                        batched_rotary=kwargs["batched_rotary"],
                    )
                else:
                    x = block(x, **kwargs)

                # Inject ControlNet states if provided
                if controlnet_states is not None and controlnet_weight != 0.0:
                    if controlnet_stride <= 0:
                        controlnet_idx = block_idx
                    else:
                        # downsample states along layers by stride
                        if block_idx % max(1, int(controlnet_stride)) == 0:
                            controlnet_idx = block_idx // int(max(1, controlnet_stride))
                        else:
                            controlnet_idx = None
                    if controlnet_idx is not None and controlnet_idx < len(
                        controlnet_states
                    ):
                        cn = controlnet_states[controlnet_idx]
                        # Expect shape: (B, L_tokens, dim)
                        if cn is not None and isinstance(cn, torch.Tensor):
                            try:
                                x = x + cn.to(dtype=x.dtype, device=x.device) * float(
                                    controlnet_weight
                                )
                            except Exception:
                                # Best-effort: ignore shape mismatch to avoid breaking non-control runs
                                pass

                # Capture intermediate representation if requested
                if (
                    return_intermediate
                    and target_block_idx is not None
                    and block_idx == target_block_idx
                ):
                    # x shape is (B, L, C). Keep as-is for downstream flattening.
                    intermediate_z = x

            # ═══════════════════════════════════════════════════════════════════════════════
            # TREAD ROUTING END: Reconstruct full tensors at configured layer index
            # ═══════════════════════════════════════════════════════════════════════════════
            if tread_state.routing_now and route_ptr < len(routes) and block_idx == int(routes[route_ptr]["end_layer_idx"]):  # type: ignore
                x = handle_routing_end(self, x, kwargs, tread_state, freqs_list)
                route_ptr += 1

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)  # type: ignore

        # Ensure device consistency after CPU offloading operations
        if x.device != input_device:
            x = x.to(input_device)

        # head
        x = self.head(x, e)

        if sparse_attention:
            P = 8
            x = x.reshape(1, T, (H * W) // (P * P), P * P, -1)  # type: ignore
            x = local_merge(x, H, W, P)
            x = x.reshape(1, T * H * W, -1)  # type: ignore

        # Final cleanup of all TREAD routing state
        tread_state.cleanup_all()

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        outputs = [u.float() for u in x]
        if return_intermediate:
            return outputs, intermediate_z
        return outputs

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)
        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if is_linear_like(m):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if is_linear_like(m):
                try:
                    nn.init.normal_(m.weight, std=0.02)
                except Exception:
                    pass
        for m in self.time_embedding.modules():
            if is_linear_like(m):
                try:
                    nn.init.normal_(m.weight, std=0.02)
                except Exception:
                    pass

        # init output layer
        nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
        key2 = "blocks.0.cross_attn.k.weight"  # 14B
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype


def load_wan_model(
    config: any,  # type: ignore
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    sparse_algo: Optional[str] = None,
    use_scaled_mm: bool = False,
    use_fvdm: bool = False,
    quant_dtype: Optional[torch.dtype] = None,
    upcast_linear: bool = False,
    exclude_ffn_from_scaled_mm: bool = False,
    scale_input_tensor: Optional[str] = None,
    rope_on_the_fly: bool = False,
    broadcast_time_embed: bool = False,
    strict_e_slicing_checks: bool = False,
    lower_precision_attention: bool = False,
    simple_modulation: bool = False,
    optimized_torch_compile: bool = False,
    lean_attention_fp32_default: bool = True,
    rope_func: str = "default",
    compile_args: Optional[list] = None,
    fp8_format: str = "e4m3",
    rope_use_float32: bool = False,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
    # Enhanced FP8 quantization parameters
    fp8_quantization_mode: str = "tensor",
    fp8_block_size: Optional[int] = None,
    fp8_percentile: Optional[float] = None,
    fp8_exclude_keys: Optional[List[str]] = None,
    fp8_use_enhanced: bool = False,
    # TorchAO integration parameters
    torchao_fp8_enabled: bool = False,
    torchao_fp8_weight_dtype: str = "e4m3fn",
    torchao_fp8_target_modules: Optional[List[str]] = None,
    torchao_fp8_exclude_modules: Optional[List[str]] = None,
) -> WanModel:
    """
    Load a WAN model from the specified checkpoint.

    Args:
        config (any): Configuration object containing model parameters.
        device (Union[str, torch.device]): Device to load the model on.
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights. If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
        sparse_algo (Optional[str]): Sparse attention algorithm to use, if any.
    """

    # If fp8_scaled, dit_weight_dtype must be None (we quantize/patch at load time).
    # Otherwise, allow dit_weight_dtype to be None to support mixed precision transformers
    # where weights keep their original per-tensor dtypes.
    if fp8_scaled and dit_weight_dtype is not None:
        raise ValueError(
            "fp8_scaled=True requires dit_weight_dtype=None (weights will be converted to fp8 at load time)"
        )

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # Performance monitoring - start timing
    start_time = time.perf_counter()
    initial_memory = (
        torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0
    )

    with init_empty_weights():
        logger.info(
            f"Creating WanModel, V2.2: {config.v2_2}, device: {device}, loading_device: {loading_device}, fp8_scaled: {fp8_scaled}"
        )

        model = WanModel(
            model_version="2.1" if not config.v2_2 else "2.2",
            dim=config.dim,
            eps=config.eps,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            in_dim=config.in_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            out_dim=config.out_dim,
            text_len=config.text_len,
            attn_mode=attn_mode,
            split_attn=split_attn,
            sparse_algo=sparse_algo,
            use_fvdm=use_fvdm,
            rope_on_the_fly=rope_on_the_fly,
            broadcast_time_embed=broadcast_time_embed,
            strict_e_slicing_checks=strict_e_slicing_checks,
            lower_precision_attention=lower_precision_attention,
            simple_modulation=simple_modulation,
            optimized_torch_compile=optimized_torch_compile,
            lean_attention_fp32_default=lean_attention_fp32_default,
        )
        # Set advanced RoPE knob on model
        try:
            setattr(model, "rope_func", rope_func)
            setattr(model, "rope_use_float32", rope_use_float32)
            if compile_args is not None:
                setattr(model, "compile_args", compile_args)
        except Exception:
            pass
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    # Use provided exclude keys or default
    exclude_keys_to_use = (
        fp8_exclude_keys
        if fp8_exclude_keys is not None
        else FP8_OPTIMIZATION_EXCLUDE_KEYS
    )

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=exclude_keys_to_use,
        quant_dtype=quant_dtype,
        fp8_format=fp8_format,
        # Enhanced FP8 parameters
        fp8_quantization_mode=fp8_quantization_mode,
        fp8_block_size=fp8_block_size,
        fp8_use_enhanced=fp8_use_enhanced,
        fp8_percentile=fp8_percentile,
        enable_memory_mapping=enable_memory_mapping,
        enable_zero_copy_loading=enable_zero_copy_loading,
        enable_non_blocking_transfers=enable_non_blocking_transfers,
        memory_mapping_threshold=memory_mapping_threshold,
    )

    # remove "model.diffusion_model." prefix: 1.3B model has this prefix
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    if fp8_scaled:
        # Apply monkey patch for FP8 - enhanced or legacy
        if fp8_use_enhanced:
            from modules.fp8_optimization_utils import apply_fp8_monkey_patch

            logger.info("Applying enhanced FP8 monkey patch")
            apply_fp8_monkey_patch(model, sd, use_scaled_mm=use_scaled_mm)
        else:
            # Use legacy monkey patch for backwards compatibility
            apply_fp8_monkey_patch(
                model,
                sd,
                use_scaled_mm=use_scaled_mm,
                upcast_linear=upcast_linear,
                quant_dtype=quant_dtype,
                exclude_ffn_from_scaled_mm=exclude_ffn_from_scaled_mm,
                scale_input_tensor=scale_input_tensor,
            )

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"☄️ INFO: Moving weights to {loading_device}")
            for key in sd.keys():  # type: ignore
                sd[key] = sd[key].to(loading_device)

    # If RamTorch Linear is enabled, and fp8_scaled is off, strip *.scale_weight buffers
    try:
        from modules.ramtorch_linear_factory import ramtorch_enabled  # type: ignore

        if ramtorch_enabled() and not fp8_scaled:
            has_scale_keys = any(k.endswith(".scale_weight") for k in sd.keys())
            if has_scale_keys:
                sd = {k: v for k, v in sd.items() if not k.endswith(".scale_weight")}
                logger.info(
                    "🌲 Stripped *.scale_weight keys from state_dict (RamTorch compatibility)"
                )
    except Exception:
        pass

    info = model.load_state_dict(sd, strict=True, assign=True)
    if dit_weight_dtype is not None:
        # cast model weights to the specified dtype. This makes sure that the model is in the correct dtype
        logger.info(f"Casting model weights to {dit_weight_dtype}")
        model = model.to(dit_weight_dtype)

    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    # Performance monitoring - end timing and memory tracking
    end_time = time.perf_counter()
    loading_time = end_time - start_time
    final_memory = (
        torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0
    )
    memory_used = final_memory - initial_memory

    # Log performance metrics
    logger.info(f"⏱️  Model loading completed in {loading_time:.2f} seconds")
    if device.type == "cuda":
        logger.info(
            f"📊 Memory usage: {memory_used:.2f} GB (Initial: {initial_memory:.2f} GB, Final: {final_memory:.2f} GB)"
        )

        # Log FP8 specific metrics if enabled
        if fp8_scaled:
            estimated_fp16_memory = memory_used * 2  # Rough estimate
            memory_savings = estimated_fp16_memory - memory_used
            savings_percent = (
                (memory_savings / estimated_fp16_memory) * 100
                if estimated_fp16_memory > 0
                else 0
            )
            logger.info(
                f"💾 Estimated FP8 memory savings: {memory_savings:.2f} GB ({savings_percent:.1f}%)"
            )

            if fp8_use_enhanced:
                logger.info(
                    f"🔧 Enhanced FP8 mode: {fp8_quantization_mode} quantization"
                )

    # Apply TorchAO FP8 quantization if enabled (post-loading quantization)
    if torchao_fp8_enabled:
        try:
            from modules.torchao_fp8_utils import (
                TorchAOFP8Config,
                apply_torchao_fp8_quantization,
            )

            # Create TorchAO configuration
            torchao_config = TorchAOFP8Config(
                enabled=True,
                weight_dtype=torchao_fp8_weight_dtype,
                target_modules=torchao_fp8_target_modules,
                exclude_modules=torchao_fp8_exclude_modules,
            )

            logger.info("🧮 TorchAO FP8 quantization starting...")
            torchao_start_time = time.perf_counter()

            # Apply TorchAO quantization
            model = apply_torchao_fp8_quantization(
                model=model,
                config=torchao_config,
                device=device,
            )

            torchao_end_time = time.perf_counter()
            torchao_time = torchao_end_time - torchao_start_time
            logger.info(
                f"🧮 TorchAO FP8 post-quantization completed in {torchao_time:.2f} seconds"
            )

        except ImportError as e:
            logger.error(f"❌ TorchAO FP8 quantization failed: {e}")
            logger.error("💡 Please install torchao or disable torchao_fp8_enabled")
            raise

    return model


def parallelize_seq_T2V(model, tp_mesh):
    if tp_mesh.size() > 1:
        for i, block in enumerate(model.blocks):
            plan = {
                "self_attn.norm_q": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attn.norm_k": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attn.v": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "self_attn.o": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(1)),
                    use_local_output=True,
                ),
                "cross_attn.norm_q": PrepareModuleOutput(
                    output_layouts=(Shard(1)), desired_output_layouts=(Shard(-1))
                ),
                "cross_attn.norm_k": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attn.v": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attn.o": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(1)),
                    use_local_output=True,
                ),
            }
            self_attn = block.self_attn
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()
            cross_attn = block.cross_attn
            cross_attn.num_heads = cross_attn.num_heads // tp_mesh.size()
            parallelize_module(block, tp_mesh, plan)

            if i == 0:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleInput(
                        input_layouts=(Replicate()),
                        desired_input_layouts=(Shard(1)),
                        use_local_output=True,
                    ),
                )

        plan = {
            "head": PrepareModuleOutput(
                output_layouts=(Shard(1)), desired_output_layouts=(Replicate())
            ),
        }
        parallelize_module(model, tp_mesh, plan)  # type: ignore
    return model
