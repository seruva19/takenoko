## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/wan/modules/model.py (Apache)
## Based on: https://github.com/gen-ai-team/Wan2.1-NABLA/blob/main/wan/modules/model.py (Apache)

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard  # type: ignore

from utils.lora_utils import load_safetensors_with_lora_and_fp8
from utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors

import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

from wan.modules.attention import flash_attention
from utils.device_utils import clean_memory_on_device
from modules.custom_offloading_utils import ModelOffloader
from modules.fp8_optimization_utils import (
    apply_fp8_monkey_patch,
    optimize_state_dict_with_fp8,
)

from .attention import local_patching, local_merge, nablaT, sta, sta_nabla
from utils.tread import TREADRouter, MaskInfo  # minimal router integration
from torch.nn.attention.flex_attention import flex_attention

try:
    flex = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True
    )
except:
    logger.warning("torch.compile failed to compile flex_attention")
    logger.warning(
        "Using original Neighborhood Adaptive Block-Level Attention (NABLA) implementation"
    )
    flex = flex_attention


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
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
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
    # with torch.amp.autocast(device_type=device_type, enabled=False):
    rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

    n, c = x.size(2), x.size(3) // 2

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

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.sparse_algo = sparse_algo
        self.mask_func = self.construct_mask_func(sparse_algo) if sparse_algo else None

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

            # Only apply standard RoPE when not routing with batched rotary
            if batched_rotary is None:
                rope_apply_inplace_cached(q, grid_sizes, freqs)
                rope_apply_inplace_cached(k, grid_sizes, freqs)
            qkv = [q, k, v]
            del q, k, v
            x = flash_attention(
                qkv,
                k_lens=seq_lens,
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
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

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
        org_dtype = x.dtype

        if self.model_version == "2.1":
            e = self.modulation.to(torch.float32) + e
            e = e.chunk(6, dim=1)
            assert e[0].dtype == torch.float32

            # self-attention
            y = self.self_attn(
                (self.norm1(x).float() * (1 + e[1]) + e[0]).to(org_dtype),
                seq_lens,
                grid_sizes,
                freqs if batched_rotary is None else None,
                sparse_attention=sparse_attention,
                # pass through when provided during routing
                # type: ignore[arg-type]
                batched_rotary=batched_rotary,
            )
            x = (x + y.to(torch.float32) * e[2]).to(org_dtype)
            del y

            # cross-attention & ffn
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            del context
            y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(org_dtype))
            x = (x + y.to(torch.float32) * e[5]).to(org_dtype)
            del y
        else:  # For Wan2.2
            e = self.modulation.to(torch.float32) + e
            e = e.chunk(6, dim=2)  # e is [B, L, 6, C] for 2.2
            assert e[0].dtype == torch.float32

            # self-attention
            y = self.self_attn(
                (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(
                    org_dtype
                ),
                seq_lens,
                grid_sizes,
                freqs if batched_rotary is None else None,
                sparse_attention=sparse_attention,
                batched_rotary=batched_rotary,  # type: ignore[arg-type]
            )
            x = (x + y.to(torch.float32) * e[2].squeeze(2)).to(org_dtype)
            del y

            # cross-attention & ffn
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            del context
            y = self.ffn(
                (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(
                    org_dtype
                )
            )
            x = (x + y.to(torch.float32) * e[5].squeeze(2)).to(org_dtype)

            del y
        return x

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
            return checkpoint(
                self._forward,
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
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, model_version="2.1"):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.model_version = model_version

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, C] for 2.1, [B, L, 6, C] for 2.2
        """
        assert e.dtype == torch.float32

        if self.model_version == "2.1":
            e = (self.modulation.to(torch.float32) + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        else:  # For Wan2.2
            e = (self.modulation.unsqueeze(0).to(torch.float32) + e.unsqueeze(2)).chunk(
                2, dim=2
            )
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
    "modulation",
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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

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
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(
            dim, out_dim, patch_size, eps, model_version=self.effective_model_version
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.freqs_fhw = {}

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

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

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:  # type: ignore
            block.enable_gradient_checkpointing()  # type: ignore

        print(f"WanModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:  # type: ignore
            block.disable_gradient_checkpointing()  # type: ignore

        print(f"WanModel: Gradient checkpointing disabled.")

    def enable_block_swap(
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool
    ):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)  # type: ignore

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "wan_attn_block",
            self.blocks,  # type: ignore
            self.num_blocks,
            self.blocks_to_swap,
            supports_backward,
            device,  # , debug=True
        )
        print(
            f"WanModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)  # type: ignore
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)  # type: ignore
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward and backward.")

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
        self.offloader.prepare_block_devices_before_forward(self.blocks)  # type: ignore

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
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )  # list of [F, H, W]

        freqs_list = []
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
                e_tokens_flat = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t_tokens_flat).float()
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
            elif self.effective_model_version == "2.1":
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float()
                )
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            else:  # For Wan2.2 (standard per-token path)
                if t.dim() == 1:
                    t = t.unsqueeze(1).expand(-1, seq_len)

                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t)
                    .unflatten(0, (bt, seq_len))
                    .float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

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
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs_list,
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
        routing_now = False
        tread_mask_info: MaskInfo | None = None
        saved_tokens = None

        # Normalize negative indices in routes
        if use_routing and routes:
            total_layers = len(self.blocks)  # type: ignore

            def _to_pos(idx: int) -> int:
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **r,
                    "start_layer_idx": _to_pos(int(r["start_layer_idx"])),
                    "end_layer_idx": _to_pos(int(r["end_layer_idx"])),
                }
                for r in routes
            ]

        # Normalize control states container to a list for indexing
        if controlnet_states is not None and isinstance(controlnet_states, tuple):
            controlnet_states = list(controlnet_states)

        # Optional intermediate capture for dispersive loss
        intermediate_z = None

        for block_idx, block in enumerate(self.blocks):  # type: ignore
            is_block_skipped = (
                skip_block_indices is not None and block_idx in skip_block_indices
            )

            if self.blocks_to_swap and not is_block_skipped:
                self.offloader.wait_for_block(block_idx)  # type: ignore

            # TREAD: START route at configured layer index
            if (
                use_routing
                and route_ptr < len(routes)
                and block_idx == int(routes[route_ptr]["start_layer_idx"])  # type: ignore
            ):
                assert router is not None
                mask_ratio = float(routes[route_ptr]["selection_ratio"])  # type: ignore
                # Only route video tokens (x). Text/context stays full
                tread_mask_info = router.get_mask(
                    x, mask_ratio=mask_ratio, force_keep=force_keep_mask  # type: ignore
                )
                saved_tokens = x.clone()
                x = router.start_route(x, tread_mask_info)
                routing_now = True

                # Build a single batched rotary tensor for the routed tokens
                # Collapse per-sample freqs_list[(L,1,D)] -> (B, S, D) after shuffle, then slice keep_len
                B = x.size(0)
                S_keep = x.size(1)
                # Concatenate per-sample to a tensor (B, S_full, D), then shuffle per batch
                full_rope = []
                for f in freqs_list:
                    full_rope.append(f.squeeze(1))  # (L,D)
                full_rope = torch.stack(full_rope, dim=0)  # (B, S_full, D)
                shuf = torch.take_along_dim(
                    full_rope,
                    tread_mask_info.ids_shuffle.unsqueeze(-1).expand(
                        B, -1, full_rope.size(-1)
                    ),
                    dim=1,
                )
                batched_rotary = shuf[:, :S_keep, :]
                kwargs["batched_rotary"] = batched_rotary

            if not is_block_skipped:
                # Switch attention to use batched_rotary when routing
                if (
                    routing_now
                    and "batched_rotary" in kwargs
                    and getattr(self, "_tread_mode", "full") == "full"
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
                    and dispersive_loss_target_block is not None
                    and block_idx == int(dispersive_loss_target_block)
                ):
                    # x shape is (B, L, C). Keep as-is for downstream flattening.
                    intermediate_z = x

            # TREAD: END route
            if routing_now and route_ptr < len(routes) and block_idx == int(routes[route_ptr]["end_layer_idx"]):  # type: ignore
                assert tread_mask_info is not None and saved_tokens is not None
                x = router.end_route(x, tread_mask_info, original_x=saved_tokens)  # type: ignore
                routing_now = False
                route_ptr += 1
                kwargs.pop("batched_rotary", None)
                kwargs["freqs"] = freqs_list  # restore full rotary embeddings

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)  # type: ignore

        # head
        x = self.head(x, e)

        if sparse_attention:
            P = 8
            x = x.reshape(1, T, (H * W) // (P * P), P * P, -1)  # type: ignore
            x = local_merge(x, H, W, P)
            x = x.reshape(1, T * H * W, -1)  # type: ignore

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
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

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
        )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        quant_dtype=quant_dtype,
    )

    # remove "model.diffusion_model." prefix: 1.3B model has this prefix
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    if fp8_scaled:
        # Apply monkey patch for FP8. Defaults preserve previous behavior
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
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():  # type: ignore
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    if dit_weight_dtype is not None:
        # cast model weights to the specified dtype. This makes sure that the model is in the correct dtype
        logger.info(f"Casting model weights to {dit_weight_dtype}")
        model = model.to(dit_weight_dtype)

    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

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
