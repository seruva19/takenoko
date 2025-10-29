"""
Sprint Token Sampling Module

Implements video-aware 3D token sampling strategies for sparse-dense residual fusion.

Key Differences from Image DiTs:
- Video tokens have 3D structure (T×H×W) vs 2D (H×W)
- Temporal coherence is critical to prevent flickering
- Structured group-wise sampling maintains spatio-temporal locality
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math


def sample_tokens_3d(
    grid_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
    keep_ratio: float,
    strategy: str = "temporal_coherent",
    batch_idx: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Sample tokens from 3D video sequences maintaining spatio-temporal coherence.

    Args:
        grid_sizes: [B, 3] tensor containing (F, H, W) for each batch element
        seq_lens: [B] tensor of sequence lengths (F*H*W for each batch element)
        keep_ratio: Fraction of tokens to keep (e.g., 0.25 for 75% dropping)
        strategy: Sampling strategy - "uniform", "temporal_coherent", "spatial_coherent"
        batch_idx: Optional batch index for deterministic sampling (uses global step + batch_idx as seed)

    Returns:
        keep_indices: List of [num_kept] tensors with indices of tokens to keep per batch element
        drop_indices: List of [num_dropped] tensors with indices of tokens to drop per batch element
    """
    B = grid_sizes.size(0)
    device = grid_sizes.device

    keep_indices_list = []
    drop_indices_list = []

    for b in range(B):
        F, H, W = grid_sizes[b].tolist()
        seq_len = seq_lens[b].item()

        # Number of tokens to keep
        num_keep = max(1, int(seq_len * keep_ratio))

        # Generate indices based on strategy
        if strategy == "uniform":
            indices = _sample_uniform_3d(F, H, W, num_keep, b if batch_idx is None else batch_idx, device)
        elif strategy == "temporal_coherent":
            indices = _sample_temporal_coherent(F, H, W, num_keep, b if batch_idx is None else batch_idx, device)
        elif strategy == "spatial_coherent":
            indices = _sample_spatial_coherent(F, H, W, num_keep, b if batch_idx is None else batch_idx, device)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Split into keep and drop indices
        all_indices = torch.arange(seq_len, device=device)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[indices] = True

        keep_indices = all_indices[mask]
        drop_indices = all_indices[~mask]

        keep_indices_list.append(keep_indices)
        drop_indices_list.append(drop_indices)

    return keep_indices_list, drop_indices_list


def _sample_uniform_3d(
    F: int, H: int, W: int, num_keep: int, seed: int, device: torch.device
) -> torch.Tensor:
    """
    Uniform random sampling across all dimensions.

    Simple random sampling without spatial or temporal structure.
    """
    seq_len = F * H * W
    indices = torch.randperm(seq_len, device=device)[:num_keep]
    return indices.sort()[0]  # Sort for cache-friendly access


def _sample_temporal_coherent(
    F: int, H: int, W: int, num_keep: int, seed: int, device: torch.device
) -> torch.Tensor:
    """
    Temporal-coherent sampling: sample full frames, then spatial within frames.

    Strategy:
    1. Determine how many frames to keep based on keep_ratio
    2. Sample frames uniformly across temporal dimension
    3. Within selected frames, sample spatial tokens uniformly

    This maintains temporal coherence by keeping complete or near-complete frames.
    """
    seq_len = F * H * W
    tokens_per_frame = H * W

    # Calculate how many frames to keep (at least 1)
    keep_ratio = num_keep / seq_len
    num_frames_keep = max(1, int(F * math.sqrt(keep_ratio)))  # Keep more frames with less spatial detail
    num_frames_keep = min(num_frames_keep, F)  # Can't keep more frames than exist

    # Sample frames uniformly
    frame_indices = torch.randperm(F, device=device)[:num_frames_keep]
    frame_indices = frame_indices.sort()[0]

    # Calculate spatial tokens per selected frame
    tokens_per_selected_frame = num_keep // num_frames_keep
    tokens_per_selected_frame = min(tokens_per_selected_frame, tokens_per_frame)

    # Sample spatial tokens within each selected frame
    all_indices = []
    for i, frame_idx in enumerate(frame_indices):
        frame_start = frame_idx * tokens_per_frame

        # For this frame, sample spatial tokens
        spatial_indices = torch.randperm(tokens_per_frame, device=device)[:tokens_per_selected_frame]

        # Convert to global indices
        global_indices = frame_start + spatial_indices
        all_indices.append(global_indices)

    # Concatenate and sort
    indices = torch.cat(all_indices)
    indices = indices.sort()[0]

    # If we have fewer tokens than needed, pad with random additional tokens
    if indices.size(0) < num_keep:
        remaining = num_keep - indices.size(0)
        all_possible = torch.arange(seq_len, device=device)
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[indices] = False
        available = all_possible[mask]

        extra_indices = available[torch.randperm(available.size(0), device=device)[:remaining]]
        indices = torch.cat([indices, extra_indices])
        indices = indices.sort()[0]

    return indices[:num_keep]


def _sample_spatial_coherent(
    F: int, H: int, W: int, num_keep: int, seed: int, device: torch.device
) -> torch.Tensor:
    """
    Spatial-coherent sampling: sample spatial groups across all frames.

    Strategy:
    1. Sample spatial locations (H×W grid) uniformly
    2. Keep these spatial locations across all temporal frames

    This maintains spatial coherence by keeping consistent spatial regions across time.
    """
    seq_len = F * H * W
    tokens_per_frame = H * W

    # Calculate how many spatial locations to keep
    keep_ratio = num_keep / seq_len
    num_spatial_keep = max(1, int(tokens_per_frame * keep_ratio))
    num_spatial_keep = min(num_spatial_keep, tokens_per_frame)

    # Sample spatial locations uniformly
    spatial_indices = torch.randperm(tokens_per_frame, device=device)[:num_spatial_keep]
    spatial_indices = spatial_indices.sort()[0]

    # Replicate these spatial locations across all frames
    all_indices = []
    for frame_idx in range(F):
        frame_start = frame_idx * tokens_per_frame
        global_indices = frame_start + spatial_indices
        all_indices.append(global_indices)

    indices = torch.cat(all_indices)
    indices = indices.sort()[0]

    # Trim to exact num_keep
    return indices[:num_keep]


def restore_sequence_with_padding(
    sparse_x: torch.Tensor,
    keep_indices_list: List[torch.Tensor],
    original_seq_lens: torch.Tensor,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Restore sparse sequences to original length with padding (MASK tokens).

    Args:
        sparse_x: [B, L_sparse, C] tensor with sparse tokens
        keep_indices_list: List of [num_kept] tensors with indices of kept tokens per batch
        original_seq_lens: [B] tensor of original sequence lengths
        pad_value: Value to use for padding (default 0.0 for MASK tokens)

    Returns:
        restored_x: [B, L_original, C] tensor with restored sequence length
    """
    B, _, C = sparse_x.shape
    device = sparse_x.device
    dtype = sparse_x.dtype

    max_seq_len = original_seq_lens.max().item()

    # Create output tensor filled with pad_value
    restored_x = torch.full(
        (B, max_seq_len, C),
        pad_value,
        dtype=dtype,
        device=device,
    )

    # Fill in kept tokens at their original positions
    for b in range(B):
        num_kept = keep_indices_list[b].size(0)
        if num_kept == 0:
            continue
        kept_tokens = sparse_x[b, :num_kept]
        restored_x[b, keep_indices_list[b]] = kept_tokens

    return restored_x


def apply_token_sampling(
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    keep_ratio: float,
    strategy: str = "temporal_coherent",
    batch_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply token sampling to input tensor, returning sparse and dense versions.

    Args:
        x: [B, L, C] input tensor
        seq_lens: [B] sequence lengths
        grid_sizes: [B, 3] grid sizes (F, H, W)
        keep_ratio: Fraction of tokens to keep
        strategy: Sampling strategy
        batch_idx: Optional batch index for seeding

    Returns:
        sparse_x: [B, L_sparse, C] sparse tokens (kept tokens)
        dense_residual: [B, L, C] dense features (all tokens, for residual connection)
        keep_indices_list: List of kept token indices per batch element
        drop_indices_list: List of dropped token indices per batch element
    """
    B, L, C = x.shape
    device = x.device

    # Sample tokens
    keep_indices_list, drop_indices_list = sample_tokens_3d(
        grid_sizes=grid_sizes,
        seq_lens=seq_lens,
        keep_ratio=keep_ratio,
        strategy=strategy,
        batch_idx=batch_idx,
    )

    # Extract sparse tokens
    sparse_tokens = []
    for b in range(B):
        kept = x[b, keep_indices_list[b]]  # [num_kept, C]
        sparse_tokens.append(kept)

    # Stack sparse tokens (may have different lengths, so pad)
    max_kept = max(t.size(0) for t in sparse_tokens)
    sparse_x = torch.zeros(B, max_kept, C, dtype=x.dtype, device=device)
    for b, tokens in enumerate(sparse_tokens):
        sparse_x[b, : tokens.size(0)] = tokens

    # Dense residual is the original input (for residual connection)
    dense_residual = x.clone()

    return sparse_x, dense_residual, keep_indices_list, drop_indices_list


class TokenSampler(nn.Module):
    """
    Token sampler module for Sprint sparse-dense fusion.

    Wraps token sampling logic as a PyTorch module for integration into WanModel.
    """

    def __init__(
        self,
        keep_ratio: float = 0.25,
        strategy: str = "temporal_coherent",
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.strategy = strategy

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply token sampling during forward pass.

        Args:
            x: [B, L, C] input tensor
            seq_lens: [B] sequence lengths
            grid_sizes: [B, 3] grid sizes (F, H, W)
            batch_idx: Optional batch index for deterministic sampling

        Returns:
            sparse_x, dense_residual, keep_indices_list, drop_indices_list
        """
        return apply_token_sampling(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            keep_ratio=self.keep_ratio,
            strategy=self.strategy,
            batch_idx=batch_idx,
        )

    def restore_sequence(
        self,
        sparse_x: torch.Tensor,
        keep_indices_list: List[torch.Tensor],
        original_seq_lens: torch.Tensor,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """Restore sparse sequence to original length with padding."""
        return restore_sequence_with_padding(
            sparse_x=sparse_x,
            keep_indices_list=keep_indices_list,
            original_seq_lens=original_seq_lens,
            pad_value=pad_value,
        )
