# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Optional, List
import torch
import math
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from einops import rearrange

try:
    import flash_attn_interface  # type: ignore

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn  # type: ignore

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import sageattention  # type: ignore

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops  # type: ignore

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


import warnings


def flash_attention(
    qkv,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    attn_mode: Optional[str] = "torch",
    split_attn: bool = False,
    batched_rotary: Optional[torch.Tensor] = None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    q, k, v = qkv
    qkv.clear()

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    # assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Flash attention 3 not tested, so keep the original code.
    # Customized code (except for flash attention 3) is not supported q_lens and k_lens.
    if attn_mode != "flash3" and attn_mode != "sageattn":
        assert q_lens is None, "q_lens is not supported except for flash attention 3."
        assert k_lens is None or (
            min(k_lens) == max(k_lens) and k_lens[0] == lk
        ), "k_lens is not supported except for flash attention 3."

    # SDPA
    if attn_mode == "torch" or attn_mode == "sdpa":
        assert (
            not deterministic
        ), "deterministic is not supported in scaled_dot_product_attention."
        if q_scale is not None:
            q = q * q_scale
        q = half(q.transpose(1, 2))
        k = half(k.transpose(1, 2))
        v = half(v.transpose(1, 2))

        # Apply batched rotary embeddings if provided (for routed attention)
        if batched_rotary is not None:
            # batched_rotary expected shape: (B, S, D) applied to last dim pairs
            def apply_rot(x, rot):
                # x: (B, L, N, D), rot: (B, L, D)
                B, L, N, D = x.shape
                x_c = torch.view_as_complex(x.float().reshape(B, L, N, D // 2, 2))
                rot_c = torch.view_as_complex(rot.float().reshape(B, L, 1, D // 2, 2))
                out = torch.view_as_real(x_c * rot_c).reshape(B, L, N, D).type_as(x)
                return out

            q = apply_rot(q, batched_rotary)
            k = apply_rot(k, batched_rotary)

        if not split_attn:
            q = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale
            )
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = torch.nn.functional.scaled_dot_product_attention(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

        del q, k, v
        x = x.transpose(1, 2).contiguous()
        return x.type(out_dtype)

    # flash attention 2
    if attn_mode == "flash" or attn_mode == "flash2":
        if not FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError(
                "Flash Attention 2 is not available. Please install flash-attn or use a different attention mode."
            )

        if q_scale is not None:
            q = q * q_scale
        q = half(q)
        k = half(k)
        v = half(v)

        if not split_attn:
            q = flash_attn.flash_attn_func(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                deterministic=deterministic,
            )
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = flash_attn.flash_attn_func(  # type: ignore
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    dropout_p,
                    softmax_scale,
                    causal,
                    window_size,
                    deterministic=deterministic,
                )
        del q, k, v
        return x.type(out_dtype)  # type: ignore

    # xformers
    if attn_mode == "xformers":
        if not XFORMERS_AVAILABLE:
            raise RuntimeError(
                "Xformers is not available. Please install xformers or use a different attention mode."
            )

        assert not deterministic, "deterministic is not supported in xformers."
        assert not causal, "causal is not supported in xformers."
        if q_scale is not None:
            q = q * q_scale
        q = half(q)
        k = half(k)
        v = half(v)

        if not split_attn:
            q = xops.memory_efficient_attention(
                q, k, v, p=dropout_p, scale=softmax_scale
            )
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = xops.memory_efficient_attention(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    p=dropout_p,
                    scale=softmax_scale,
                )

        del q, k, v
        return x.type(out_dtype)

    # sage attention with fixed length seems to cause NaN in I2V inference.
    # # sage attention
    # if attn_mode == "sageattn":
    #     print("Using sage attention")
    #     assert not deterministic, "deterministic is not supported in sage attention."
    #     if q_scale is not None:
    #         q = q * q_scale
    #     q, k, v = half(q), half(k), half(v)
    #     x = sageattention.sageattn(q, k, v, "NHD", is_causal=causal, sm_scale=softmax_scale)
    #     del q, k, v
    #     return x.type(out_dtype)

    assert (
        not split_attn
    ), "split_attn is not supported in flash attention 3 or sage attention."

    # preprocess query: in Wan 2.1, q_lens is always None.
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        # Note: in Wan 2.1, all k_lens are same if we have same image size in the batch.
        if min(k_lens) == max(k_lens) and k.shape[1] == k_lens[0]:
            # B, L, N, C -> BN, L, C
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
    #     warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")

    # apply attention
    # if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
    if attn_mode == "flash3":
        if not FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError(
                "Flash Attention 3 is not available. Please install flash-attn>=3.0 or use a different attention mode."
            )

        # Not tested yet
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    # elif (version is None or version == 2) and FLASH_ATTN_2_AVAILABLE:
    #     # assert FLASH_ATTN_2_AVAILABLE
    #     x = flash_attn.flash_attn_varlen_func(
    #         q=q,
    #         k=k,
    #         v=v,
    #         cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    #         cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    #         max_seqlen_q=lq,
    #         max_seqlen_k=lk,
    #         dropout_p=dropout_p,
    #         softmax_scale=softmax_scale,
    #         causal=causal,
    #         window_size=window_size,
    #         deterministic=deterministic,
    #     ).unflatten(0, (b, lq))
    # elif version is None and SAGE_ATTN_AVAILABLE:
    elif attn_mode == "sageattn":
        if not SAGE_ATTN_AVAILABLE:
            raise RuntimeError(
                "SageAttention is not available. Please install sageattention or use a different attention mode."
            )

        # print("Using sage attention")
        assert not causal, "SAGE attention does not support causal attention."
        x = sageattention.sageattn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            sm_scale=softmax_scale,
        ).unflatten(0, (b, lq))
    else:
        raise ValueError(f"Unknown attention mode: {attn_mode}")

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    batched_rotary: Optional[torch.Tensor] = None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,  # type: ignore
            k=k,  # type: ignore
            v=v,  # type: ignore
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
            batched_rotary=batched_rotary,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out


def local_patching(x, height, width, group_size):
    """
    Applies fractal flattening to group tokens from the same spatial patch together.
    This is crucial for block-wise attention to be effective.
    """
    if group_size > 0:
        x = rearrange(
            x,
            "b t (h g1) (w g2) c -> b t (h w) (g1 g2) c",
            h=height // group_size,
            w=width // group_size,
            g1=group_size,
            g2=group_size,
        )
    else:
        x = rearrange(x, "b c t h w -> b c t (h w)", h=height, w=width)
    return x


def local_merge(x, height, width, group_size):
    """
    Reverses the local_patching operation to restore the original token order.
    """
    if group_size > 0:
        x = rearrange(
            x,
            "b t (h w) (g1 g2) c -> b t (h g1) (w g2) c ",
            h=height // group_size,
            w=width // group_size,
            g1=group_size,
            g2=group_size,
        )
    else:
        x = rearrange(x, "b c (h w) -> b c h w", h=height, w=width)
    return x


@torch.no_grad()
def sta(
    T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device: str = "cuda"
) -> BlockMask:
    l = torch.Tensor([T, H, W]).max()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=device)  # type: ignore
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = (
        mat[:T, :T].flatten(),
        mat[:H, :H].flatten(),
        mat[:W, :W].flatten(),
    )
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (
        (sta_h.unsqueeze(1) * sta_w.unsqueeze(0))
        .reshape(H, H, W, W)
        .transpose(1, 2)
        .flatten()
    )
    sta = (
        (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0))
        .reshape(T, T, H * W, H * W)
        .transpose(1, 2)
    )
    sta = sta.reshape(T * H * W, T * H * W).unsqueeze_(0).unsqueeze_(0)

    # BlockMask creation
    kv_nb = sta.sum(-1).to(torch.int32)
    kv_inds = sta.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None
    )


@torch.no_grad()
def sta_nabla(
    T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device: str = "cuda"
) -> Tensor:
    l = torch.Tensor([T, H, W]).max()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=device)  # type: ignore
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = (
        mat[:T, :T].flatten(),
        mat[:H, :H].flatten(),
        mat[:W, :W].flatten(),
    )
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (
        (sta_h.unsqueeze(1) * sta_w.unsqueeze(0))
        .reshape(H, H, W, W)
        .transpose(1, 2)
        .flatten()
    )
    sta = (
        (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0))
        .reshape(T, T, H * W, H * W)
        .transpose(1, 2)
    )
    return sta.reshape(T * H * W, T * H * W)


@torch.no_grad()
def nablaT(
    q: Tensor,
    k: Tensor,
    seq: Tensor,
    T: int,
    H: int,
    W: int,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    thr: float = 0.9,
    sta_att=1,
    device: str = "cuda",
) -> BlockMask:
    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    d = torch.diff(seq)
    doc = (
        torch.eye(d.numel(), dtype=torch.bool, device=device)
        .repeat_interleave(d * H * W, dim=0)
        .repeat_interleave(d * H * W, dim=1)
    )
    map += doc.log()
    map = torch.softmax(map / math.sqrt(D), dim=-1)

    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))
    if sta_att > 0:
        sta = sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None
    )
