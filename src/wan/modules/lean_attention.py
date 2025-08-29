import torch
import torch.nn as nn
from typing import Optional


def forward_wan22_lean_block(
    x: torch.Tensor,
    e: torch.Tensor,
    modulation: torch.Tensor,
    norm1: nn.Module,
    norm2: nn.Module,
    norm3: nn.Module,
    self_attn: nn.Module,
    cross_attn: nn.Module,
    ffn: nn.Module,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Optional[torch.Tensor] | list[torch.Tensor] | None,
    context: torch.Tensor,
    context_lens: Optional[torch.Tensor],
    sparse_attention: bool,
    batched_rotary: Optional[torch.Tensor],
    force_fp16: bool,
    fp32_default: bool,
) -> torch.Tensor:
    """
    Perform the Wan 2.2 attention block forward in a lean compute path to save VRAM.

    This implementation keeps intermediates in the compute dtype (bf16/fp16). When
    force_fp16 is True, it explicitly uses fp16 for lower VRAM usage.

    Args:
        x: Input tensor [B, L, C]
        e: Per-token modulation tensor [B, L, 6, C]
        modulation: Learned modulation parameters [1, 6, C]
        norm1, norm2, norm3: Normalization modules
        self_attn: Self-attention module
        cross_attn: Cross-attention module
        ffn: Feed-forward network module
        seq_lens: Per-sample sequence lengths [B]
        grid_sizes: Per-sample grid sizes [B, 3] (F, H, W)
        freqs: Rotary frequencies (list per-sample or base tensor) or None when using batched_rotary
        context: Text/context tensor [B, L_ctx, C]
        context_lens: Optional per-sample context lengths [B]
        sparse_attention: Whether sparse attention is enabled
        batched_rotary: Optional batched rotary tensor when routing is active
        force_fp16: If True, force compute dtype to torch.float16

    Returns:
        Updated hidden tensor x with the same dtype as input x.
    """
    # Track original and attention compute dtypes
    x_orig_dtype: torch.dtype = x.dtype
    # Attention compute dtype policy: fp16 when forced, else fp32 default can be overridden by flag
    attention_dtype: torch.dtype = (
        torch.float16 if force_fp16 else (torch.float32 if fp32_default else x.dtype)
    )

    # Modulation and conditioning in attention dtype
    m = modulation.to(attention_dtype, copy=False)
    e_comp = e.to(attention_dtype, copy=False)

    # Split along the 6-slot axis
    # m: [1, 6, C] -> six tensors [1, C]; e: [B, L, 6, C] -> six tensors [B, L, C]
    m0, m1, m2, m3, m4, m5 = m.unbind(dim=1)
    e0, e1, e2, e3, e4, e5 = e_comp.unbind(dim=2)

    # Per-slot sums (broadcast modulation over [B, L, C])
    s0 = e0 + m0
    s1 = e1 + m1
    s2 = e2 + m2
    s3 = e3 + m3
    s4 = e4 + m4
    s5 = e5 + m5

    # Self-attention in attention dtype
    q_in = norm1(x).to(attention_dtype, copy=False)
    y = self_attn(
        (q_in * (1 + s1) + s0).contiguous(),
        seq_lens,
        grid_sizes,
        freqs,
        sparse_attention=sparse_attention,
        batched_rotary=batched_rotary,  # type: ignore[arg-type]
    )
    # Gate multiply in attention dtype, then cast back to original dtype for residual add
    x = x + (y.to(attention_dtype, copy=False) * s2).to(x_orig_dtype, copy=False)
    del y

    # Cross-attention; cast attn output back to original dtype for residual add
    x = x + cross_attn(
        norm3(x).to(attention_dtype, copy=False),
        context,
        context_lens,
    ).to(x_orig_dtype, copy=False)

    # FFN in attention dtype
    ff_in = norm2(x).to(attention_dtype, copy=False)
    y = ffn((ff_in * (1 + s4) + s3).contiguous())
    x = x + (y.to(attention_dtype, copy=False) * s5).to(x_orig_dtype, copy=False)
    del y

    return x
