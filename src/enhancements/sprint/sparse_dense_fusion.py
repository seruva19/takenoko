"""
Sprint Sparse-Dense Residual Fusion Module

Implements the core Sprint architecture for efficient DiT training.

Architecture:
- Encoder (fθ): Early blocks process all tokens → dense shallow features
- Middle (gθ): Deep blocks process sparse tokens (75% dropped) → sparse deep features
- Decoder (hθ): Final blocks combine dense + sparse via residual fusion
- Fusion: Concatenate[dense, sparse_restored] → Project → Decoder

Key Benefits:
- Significant training speedup with token dropping
- Dense shallow features capture local details and noise information
- Sparse deep features focus on global semantics
- Residual fusion preserves representation quality
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
import logging

from common.logger import get_logger
from .token_sampling import TokenSampler

logger = get_logger(__name__, level=logging.INFO)


def _gather_time_embeddings(
    e: torch.Tensor,
    keep_indices_list: List[torch.Tensor],
    max_kept: int,
) -> torch.Tensor:
    """
    Gather per-token time embeddings to match sparsified sequences.

    Handles both Wan 2.1 (broadcast time embed) and Wan 2.2 per-token embeddings.
    """
    # Only gather when we have per-token embeddings (shape [B, L, 6, C])
    if not isinstance(e, torch.Tensor) or e.dim() != 4:
        return e
    if e.size(1) <= 1:
        # Broadcasted embeddings already align with any sequence length
        return e

    B, _, gates, dim = e.shape
    device = e.device
    gathered = e.new_zeros((B, max_kept, gates, dim))
    for b, keep_idx in enumerate(keep_indices_list):
        num_kept = keep_idx.numel()
        if num_kept == 0:
            continue
        gathered[b, :num_kept] = torch.index_select(
            e[b], dim=0, index=keep_idx.to(device=device)
        )
    return gathered


def _build_batched_rotary(
    freqs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    keep_indices_list: List[torch.Tensor],
    max_kept: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Construct batched rotary embeddings for sparsified tokens.

    Sprint requires per-sample rotary caches (rope_on_the_fly must be disabled).
    """
    if not isinstance(freqs, (list, tuple)):
        return None
    if len(freqs) != len(keep_indices_list):
        return None

    dim = freqs[0].size(-1)
    rope = torch.zeros(
        (len(keep_indices_list), max_kept, dim),
        dtype=freqs[0].dtype,
        device=device,
    )

    for b, keep_idx in enumerate(keep_indices_list):
        if keep_idx.numel() == 0:
            continue
        gathered = (
            freqs[b].squeeze(1).index_select(0, keep_idx.to(device=freqs[b].device))
        )
        rope[b, : gathered.size(0)] = gathered.to(device=device)

    return rope


class SparseDenseFusion(nn.Module):
    """
    Sparse-Dense Residual Fusion for efficient DiT training (Sprint).

    Partitions transformer blocks into:
    - Encoder: blocks[0:encoder_end_idx] → dense path, all tokens
    - Middle: blocks[encoder_end_idx:middle_end_idx] → sparse path, 25% tokens kept
    - Decoder: blocks[middle_end_idx:] → fusion + final processing

    Usage:
        fusion = SparseDenseFusion(
            dim=2048,
            encoder_end_idx=8,
            middle_end_idx=24,
            token_drop_ratio=0.75,
            sampling_strategy="temporal_coherent"
        )

        # In WanModel.forward():
        if self.training and self.sprint_fusion is not None:
            x = self.sprint_fusion(x, self.blocks, e, context, ...)
        else:
            for block in self.blocks:
                x = block(x, e, ...)
    """

    def __init__(
        self,
        dim: int,
        encoder_end_idx: int = 8,
        middle_end_idx: int = 24,
        token_drop_ratio: float = 0.75,
        sampling_strategy: str = "temporal_coherent",
        path_drop_prob: float = 0.1,
        use_learnable_mask_token: bool = False,
    ):
        """
        Initialize Sprint sparse-dense fusion module.

        Args:
            dim: Hidden dimension of transformer
            encoder_end_idx: Index where encoder ends (exclusive). E.g., 8 means blocks[0:8] are encoder.
            middle_end_idx: Index where middle ends (exclusive). E.g., 24 means blocks[8:24] are middle.
            token_drop_ratio: Ratio of tokens to drop in middle blocks (default 0.75 for 75% dropping)
            sampling_strategy: Token sampling strategy - "uniform", "temporal_coherent", "spatial_coherent"
            path_drop_prob: Probability of replacing sparse path with MASK during training (default 0.1)
                           This helps the model learn to handle missing sparse information.
        """
        super().__init__()

        self.dim = dim
        self.encoder_end_idx = encoder_end_idx
        self.middle_end_idx = middle_end_idx
        self.token_drop_ratio = token_drop_ratio
        self.keep_ratio = 1.0 - token_drop_ratio
        self.path_drop_prob = path_drop_prob
        self.use_learnable_mask_token = use_learnable_mask_token

        self.mask_token = None
        if self.use_learnable_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Token sampler
        self.token_sampler = TokenSampler(
            keep_ratio=self.keep_ratio,
            strategy=sampling_strategy,
        )

        # Fusion projection: combines dense and sparse features
        # Dense features: [B, L, C]
        # Sparse features (restored): [B, L, C]
        # Concatenated: [B, L, 2*C] → Project to [B, L, C]
        self.fusion_proj = nn.Linear(dim * 2, dim, bias=True)

        # Initialize fusion projection to preserve dense features initially
        # This allows gradual learning of sparse path contribution
        with torch.no_grad():
            # Project dense path as identity, sparse path as zero initially
            self.fusion_proj.weight.zero_()
            self.fusion_proj.weight[:dim, :dim] = torch.eye(dim)  # Dense path identity
            # Sparse path weights remain zero initially
            if self.fusion_proj.bias is not None:
                self.fusion_proj.bias.zero_()

        logger.info(
            f"Sprint initialized: encoder=blocks[0:{encoder_end_idx}], "
            f"middle=blocks[{encoder_end_idx}:{middle_end_idx}], "
            f"decoder=blocks[{middle_end_idx}:], "
            f"token_drop_ratio={token_drop_ratio:.2f}, strategy={sampling_strategy}"
        )

    def forward(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        e: torch.Tensor,
        context: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        context_lens: torch.Tensor,
        sparse_attention: bool = False,
        batched_rotary: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sparse-dense residual fusion.

        Args:
            x: [B, L, C] input features (after patch embedding)
            blocks: nn.ModuleList of transformer blocks
            e: [B, 6, C] or [B, L, 6, C] time embeddings
            context: [B, L_text, C] text context for cross-attention
            seq_lens: [B] sequence lengths
            grid_sizes: [B, 3] grid sizes (F, H, W)
            freqs: RoPE frequencies
            context_lens: [B] context sequence lengths
            sparse_attention: Whether to use sparse self-attention (for Nabla, etc.)
            batched_rotary: Optional pre-computed RoPE
            batch_idx: Optional batch index for deterministic token sampling

        Returns:
            x: [B, L, C] output features after all blocks
        """
        B, L, C = x.shape

        # ===== ENCODER: Dense path, all tokens =====
        for i in range(self.encoder_end_idx):
            x = blocks[i](
                x,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                sparse_attention=sparse_attention,
                batched_rotary=batched_rotary,
            )

        # Save dense shallow features for residual connection
        dense_features = x.clone()  # [B, L, C]

        # ===== MIDDLE: Sparse path, dropped tokens =====
        # Sample tokens
        sparse_x, _, keep_indices_list, _ = self.token_sampler(
            x=x,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            batch_idx=batch_idx,
        )

        # Update seq_lens for sparse processing
        sparse_seq_lens = torch.tensor(
            [indices.size(0) for indices in keep_indices_list],
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )

        # Save for diagnostics
        self._last_sparse_seq_lens = sparse_seq_lens

        # Prepare per-token time embeddings and rotary caches for sparse path
        e_sparse = _gather_time_embeddings(e, keep_indices_list, sparse_x.size(1))
        sparse_batched_rotary = _build_batched_rotary(
            freqs=freqs,
            keep_indices_list=keep_indices_list,
            max_kept=sparse_x.size(1),
            device=x.device,
        )

        if sparse_batched_rotary is None:
            raise RuntimeError(
                "Sprint requires cached rotary frequencies (rope_on_the_fly is not supported)."
            )

        # Process sparse tokens through middle blocks
        # NOTE: grid_sizes is kept as original (not sparse) because it's used for RoPE frequency
        # calculation which needs the 3D structure (F, H, W), not the actual sequence length.
        # The blocks use seq_lens for attention masking and grid_sizes for positional encoding,
        # which are independent operations. This is intentional and safe.
        #
        # NOTE: Cross-attention is fully compatible with Sprint. The context (text conditioning)
        # and context_lens remain at full resolution - only the video tokens (x) are sparsified.
        # Self-attention operates on sparse_seq_lens, while cross-attention operates independently
        # on context_lens. This allows Sprint to work with text-conditioned video generation.
        for i in range(self.encoder_end_idx, self.middle_end_idx):
            sparse_x = blocks[i](
                sparse_x,
                e_sparse,
                sparse_seq_lens,
                grid_sizes,  # Keep original grid sizes for RoPE frequency calculation
                freqs,
                context,  # Full context (not sparse) for cross-attention
                context_lens,  # Full context lengths for cross-attention
                sparse_attention=sparse_attention,
                batched_rotary=sparse_batched_rotary,
            )

        # Restore sparse sequence to original length with padding
        sparse_features_restored = self.token_sampler.restore_sequence(
            sparse_x=sparse_x,
            keep_indices_list=keep_indices_list,
            original_seq_lens=seq_lens,
            pad_value=0.0,  # MASK tokens as zeros
            pad_token=self.mask_token,
        )

        # ===== PATH-DROP LEARNING: Randomly replace sparse path with MASK =====
        # During training, features from the sparse path are randomly replaced
        # with [MASK] tokens at the configured probability (default 10%)
        if self.training and self.path_drop_prob > 0.0:
            # Generate per-sample random mask for better regularization
            # mask[i] = 1.0 means use sparse features, 0.0 means use MASK (zeros)
            batch_size = sparse_features_restored.size(0)
            keep_mask = (
                torch.rand(batch_size, device=x.device) > self.path_drop_prob
            ).float()
            keep_mask = keep_mask.view(batch_size, 1, 1)  # [B, 1, 1] for broadcasting

            # Apply mask: keeps sparse features or replaces with mask token/zeros per sample
            if self.mask_token is not None:
                mask_tokens = self.mask_token.to(
                    device=x.device, dtype=sparse_features_restored.dtype
                )
                sparse_features_restored = (
                    sparse_features_restored * keep_mask
                    + mask_tokens * (1.0 - keep_mask)
                )
            else:
                sparse_features_restored = sparse_features_restored * keep_mask

        # ===== FUSION: Combine dense shallow + sparse deep =====
        # Concatenate along channel dimension
        fused = torch.cat(
            [dense_features, sparse_features_restored], dim=-1
        )  # [B, L, 2*C]

        # Project back to original dimension
        x = self.fusion_proj(fused)  # [B, L, C]

        # ===== DECODER: Final blocks =====
        for i in range(self.middle_end_idx, len(blocks)):
            x = blocks[i](
                x,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                sparse_attention=sparse_attention,
                batched_rotary=batched_rotary,
            )

        return x

    def get_num_encoder_blocks(self) -> int:
        """Return number of encoder blocks."""
        return self.encoder_end_idx

    def get_num_middle_blocks(self) -> int:
        """Return number of middle blocks."""
        return self.middle_end_idx - self.encoder_end_idx

    def get_num_decoder_blocks(self, total_blocks: int) -> int:
        """Return number of decoder blocks."""
        return total_blocks - self.middle_end_idx

    def get_effective_token_ratio(self) -> float:
        """
        Return effective token ratio (accounting for encoder + middle + decoder).

        Encoder: 100% tokens
        Middle: (1 - token_drop_ratio) tokens
        Decoder: 100% tokens

        Average effective ratio weighted by number of blocks in each stage.
        """
        # This is an approximation - actual speedup depends on block complexity
        return (
            self.encoder_end_idx * 1.0
            + (self.middle_end_idx - self.encoder_end_idx) * self.keep_ratio
        ) / self.middle_end_idx

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"encoder_blocks=0-{self.encoder_end_idx}, "
            f"middle_blocks={self.encoder_end_idx}-{self.middle_end_idx}, "
            f"token_drop_ratio={self.token_drop_ratio:.2f}, "
            f"keep_ratio={self.keep_ratio:.2f}, "
            f"effective_token_ratio={self.get_effective_token_ratio():.2f}"
        )


def create_sprint_fusion(
    dim: int,
    num_layers: int,
    token_drop_ratio: float = 0.75,
    sampling_strategy: str = "temporal_coherent",
    encoder_ratio: float = 0.25,
    middle_ratio: float = 0.50,
    path_drop_prob: float = 0.1,
    encoder_layers: Optional[int] = None,
    middle_layers: Optional[int] = None,
    partitioning_strategy: str = "percentage",
    use_learnable_mask_token: bool = False,
) -> SparseDenseFusion:
    """
    Factory function to create Sprint fusion module with automatic block partitioning.

    Args:
        dim: Hidden dimension
        num_layers: Total number of transformer blocks
        token_drop_ratio: Ratio of tokens to drop (default 0.75)
        sampling_strategy: Token sampling strategy
        encoder_ratio: Ratio of blocks to use as encoder (default 0.25 = first 25%)
        middle_ratio: Ratio of blocks to use as middle (default 0.50 = middle 50%)
        path_drop_prob: Probability of replacing sparse path with MASK during training (default 0.1)
        encoder_layers: Explicit number of encoder blocks (overrides encoder_ratio if set)
        middle_layers: Explicit number of middle blocks (overrides middle_ratio if set)
        partitioning_strategy: "percentage" (default) or "fixed" (2-N-2 style)

    Returns:
        SparseDenseFusion module

    Examples:
        # Percentage-based (default): For 32-layer model → encoder=8, middle=16, decoder=8
        fusion = create_sprint_fusion(dim=2048, num_layers=32)

        # Fixed 2-N-2: 2-24-2 partitioning
        fusion = create_sprint_fusion(
            dim=2048, num_layers=28,
            partitioning_strategy="fixed",
            encoder_layers=2,
            middle_layers=24
        )

        # Explicit numbers: Custom partitioning
        fusion = create_sprint_fusion(
            dim=2048, num_layers=32,
            encoder_layers=4,
            middle_layers=20
        )
    """
    # Determine block partitioning
    if partitioning_strategy == "fixed":
        # Fixed partitioning: 2-N-2 (or custom if specified)
        if encoder_layers is None:
            encoder_layers = 2  # Default for fixed strategy
        if middle_layers is None:
            # Use most blocks for middle (sparse processing)
            middle_layers = num_layers - encoder_layers - 2  # Reserve 2 for decoder
            middle_layers = max(1, middle_layers)

        encoder_end_idx = encoder_layers
        middle_end_idx = encoder_layers + middle_layers

    elif partitioning_strategy == "percentage":
        # Percentage-based partitioning (flexible for different model sizes)
        if encoder_layers is not None:
            encoder_end_idx = encoder_layers
        else:
            encoder_end_idx = max(1, int(num_layers * encoder_ratio))

        if middle_layers is not None:
            middle_end_idx = encoder_end_idx + middle_layers
        else:
            middle_end_idx = max(
                encoder_end_idx + 1, int(num_layers * (encoder_ratio + middle_ratio))
            )

    else:
        raise ValueError(
            f"Unknown partitioning_strategy: {partitioning_strategy}. Use 'percentage' or 'fixed'"
        )

    # Ensure valid partitioning (with warnings for any adjustments)
    original_encoder_end = encoder_end_idx
    original_middle_end = middle_end_idx

    encoder_end_idx = max(
        1, min(encoder_end_idx, num_layers - 2)
    )  # At least 1 encoder, leave room for middle+decoder
    middle_end_idx = max(
        encoder_end_idx + 1, min(middle_end_idx, num_layers - 1)
    )  # At least 1 middle, 1 decoder

    # Warn if partitioning was adjusted
    if encoder_end_idx != original_encoder_end:
        logger.warning(
            f"⚠️ Sprint encoder blocks adjusted: {original_encoder_end} → {encoder_end_idx} "
            f"(clamped to ensure valid partitioning for {num_layers}-layer model)"
        )

    if middle_end_idx != original_middle_end:
        logger.warning(
            f"⚠️ Sprint middle blocks adjusted: {original_middle_end} → {middle_end_idx} "
            f"(clamped to ensure valid partitioning for {num_layers}-layer model)"
        )

    num_encoder = encoder_end_idx
    num_middle = middle_end_idx - encoder_end_idx
    num_decoder = num_layers - middle_end_idx

    logger.info(
        f"Sprint partitioning ({partitioning_strategy}): {num_layers} blocks total → "
        f"encoder=[0:{encoder_end_idx}] ({num_encoder} blocks), "
        f"middle=[{encoder_end_idx}:{middle_end_idx}] ({num_middle} blocks), "
        f"decoder=[{middle_end_idx}:{num_layers}] ({num_decoder} blocks)"
    )

    return SparseDenseFusion(
        dim=dim,
        encoder_end_idx=encoder_end_idx,
        middle_end_idx=middle_end_idx,
        token_drop_ratio=token_drop_ratio,
        sampling_strategy=sampling_strategy,
        path_drop_prob=path_drop_prob,
        use_learnable_mask_token=use_learnable_mask_token,
    )
