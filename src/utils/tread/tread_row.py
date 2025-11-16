"""Row-based spatial TREAD utilities for efficient image token routing."""

import math
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Import frame routing for fallback functionality
try:
    from .tread_frame import pack_frame_routed_tokens
    _FRAME_ROUTING_AVAILABLE = True
except ImportError:
    _FRAME_ROUTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RowRouteState:
    """State needed to reconstruct full sequences after row routing.

    Attributes
    ----------
    idx_proc_pad: torch.Tensor
        Padded kept-token indices per sample. Shape (B, Lpmax), -1 padded.
    idx_rout_pad: torch.Tensor
        Padded routed-token indices per sample. Shape (B, Lrmax), -1 padded.
    x_rout_pad: torch.Tensor
        Padded original routed token representations. Shape (B, Lrmax, C).
    seq_lens_orig: torch.Tensor
        Original sequence lengths (B,).
    seq_lens_proc: torch.Tensor
        Processed sequence lengths (B,).
    grid_sizes_orig: torch.Tensor
        Original grid sizes (B, 3).
    grid_sizes_proc: torch.Tensor
        Processed grid sizes (B, 3).
    """
    idx_proc_pad: torch.Tensor
    idx_rout_pad: torch.Tensor
    x_rout_pad: torch.Tensor
    seq_lens_orig: torch.Tensor
    seq_lens_proc: torch.Tensor
    grid_sizes_orig: torch.Tensor
    grid_sizes_proc: torch.Tensor


# Global persistent RNG state for reproducible random row selection
_global_tread_rng: Optional[torch.Generator] = None


@dataclass
class SpatialAutoRouteState:
    """Routing state for mixed spatial_auto batches."""

    image_indices: torch.Tensor
    video_indices: torch.Tensor
    row_state: Optional["RowRouteState"]
    frame_state: Optional["FrameRouteState"]
    seq_lens_orig: torch.Tensor
    seq_lens_proc: torch.Tensor
    grid_sizes_orig: torch.Tensor
    grid_sizes_proc: torch.Tensor
    idx_proc_pad: torch.Tensor
    row_lpmax: int
    frame_lpmax: int


def _compute_row_indices(
    num_rows: int,
    keep_ratio: float,
    mode: str = "contiguous",
    device: torch.device = torch.device("cpu"),
    rng_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial row indices for TREAD routing.

    Implements row selection algorithms for spatial token routing.

    Parameters
    ----------
    num_rows: int
        Number of spatial rows H in the sample.
    keep_ratio: float
        Fraction in (0,1] of rows to keep.
    mode: str
        'contiguous', 'stride', or 'random'.
    device: torch.device
        Target device for returned tensors.
    rng_seed: Optional[int]
        Random seed for reproducible random selection.

    Returns
    -------
    kept, routed: Tuple[Tensor, Tensor]
        1D long tensors with kept and routed row indices.
    """
    # Use math.floor for consistent integer conversion
    keep_rows = max(1, int(math.floor(num_rows * keep_ratio)))

    # If keeping all rows, return early
    if keep_rows >= num_rows:
        kept = torch.arange(num_rows, device=device, dtype=torch.long)
        return kept, kept.new_empty((0,), dtype=torch.long)

    # Row selection based on mode
    if mode == "stride":
        # Calculate stride for evenly spaced selection
        stride = max(1, int(round(num_rows / keep_rows)))
        # Select initial rows with stride pattern
        kept = torch.arange(0, num_rows, stride, device=device, dtype=torch.long)[:keep_rows]

        # Fill remaining slots if needed
        if kept.numel() < keep_rows:
            remaining = torch.arange(num_rows, device=device, dtype=torch.long)
            mask = torch.ones(num_rows, dtype=torch.bool, device=device)
            mask[kept] = False
            extras = remaining[mask][:keep_rows - kept.numel()]
            kept = torch.cat([kept, extras], dim=0)
        # Sort kept indices
        kept, _ = torch.sort(kept)

    elif mode == "random":
        # Random row selection with persistent RNG state
        global _global_tread_rng
        if _global_tread_rng is None:
            _global_tread_rng = torch.Generator(device="cpu")
            if rng_seed is not None:
                _global_tread_rng.manual_seed(rng_seed)
            else:
                _global_tread_rng.seed()
        perm = torch.randperm(num_rows, generator=_global_tread_rng)
        kept = perm[:keep_rows].to(device=device, dtype=torch.long)
        kept, _ = torch.sort(kept)

    else:  # contiguous mode
        # Center window selection
        start = max(0, (num_rows - keep_rows) // 2)
        end = min(num_rows, start + keep_rows)
        # Adjust if needed
        if end - start < keep_rows:
            start = max(0, end - keep_rows)
        # Create contiguous range
        kept = torch.arange(start, start + keep_rows, device=device, dtype=torch.long)

    # Compute routed (dropped) indices
    all_indices = torch.arange(num_rows, device=device, dtype=torch.long)
    mask = torch.ones(num_rows, dtype=torch.bool, device=device)
    mask[kept] = False
    routed = all_indices[mask]

    return kept, routed


def pack_row_routed_tokens(
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    keep_ratio: float,
    mode: str = "contiguous",
    rng_seed: Optional[int] = None,
    auto_fallback: bool = True,
) -> Tuple[torch.Tensor, RowRouteState]:
    """Pack tokens corresponding to kept rows using spatial token conversion.

    This function implements efficient token packing for row-based routing.

    Parameters
    ----------
    x: torch.Tensor
        Token sequence (B, Lmax, C), padded to same Lmax across batch.
    seq_lens: torch.Tensor
        Actual sequence lengths (B,).
    grid_sizes: torch.Tensor
        Grid dimensions (B, 3) as [F, H, W].
    keep_ratio: float
        Fraction in (0,1] of rows to keep.
    mode: str
        'contiguous', 'stride', or 'random'.
    rng_seed: Optional[int]
        Random seed for reproducible random selection.
    auto_fallback: bool
        If True, automatically use frame routing for video content (F>1).
        If False, warn and keep all tokens for video content.

    Returns
    -------
    x_proc: torch.Tensor
        Packed kept tokens (B, Lpmax, C).
    state: RowRouteState
        Routing state for reconstruction.
    """
    B, Lmax, C = x.shape
    device = x.device

    idx_proc_list: List[torch.Tensor] = []
    idx_rout_list: List[torch.Tensor] = []
    x_rout_list: List[torch.Tensor] = []
    grid_sizes_proc_list: List[List[int]] = []
    x_proc_list: List[torch.Tensor] = []
    Lpmax = 0
    Lrmax = 0

    for b in range(B):
        Li = int(seq_lens[b].item())
        F, H, W = grid_sizes[b].tolist()
        F, H, W = int(F), int(H), int(W)
        assert Li == F * H * W, f"Unexpected seq_len {Li} != FHW {F*H*W}"

        # Handle video content (F > 1)
        if F > 1:
            if auto_fallback and _FRAME_ROUTING_AVAILABLE:
                # Auto-fallback to frame routing for video content
                logger.info(
                    f"Row-based TREAD with auto-fallback: F={F} frames detected, "
                    f"automatically using frame routing (contiguous mode)."
                )
                # Use frame routing with contiguous mode
                from .tread_frame import pack_frame_routed_tokens
                return pack_frame_routed_tokens(x, seq_lens, grid_sizes, keep_ratio, "contiguous")
            else:
                # Warn and keep all tokens for video content
                logger.warning(
                    f"Row-based TREAD used with video content (F={F} frames). "
                    f"Row routing only works on images (F=1). For video content, "
                    f"use 'frame_contiguous' or 'frame_stride' instead. "
                    f"Falling back to keeping all tokens for this batch."
                )
                # Keep all tokens for video content
                kept_tok = torch.arange(0, Li, device=device, dtype=torch.long)
                routed_tok = torch.empty(0, device=device, dtype=torch.long)
                proc_grid = [F, H, W]  # No change
        else:
            # Single frame: use row-based routing algorithm
            kept_rows, routed_rows = _compute_row_indices(H, keep_ratio, mode, device, rng_seed)

            # Convert row indices to token indices
            # Create width range for vectorized computation
            width_range = torch.arange(W, device=device, dtype=torch.long)

            # Compute kept token indices (vectorized)
            kept_tok = (kept_rows.unsqueeze(1) * W + width_range.unsqueeze(0)).reshape(-1)

            # Compute routed token indices
            if routed_rows.numel() > 0:
                routed_tok = (routed_rows.unsqueeze(1) * W + width_range.unsqueeze(0)).reshape(-1)
            else:
                routed_tok = kept_rows.new_empty((0,), dtype=torch.long)

            # Update processed grid dimensions
            proc_grid = [F, int(kept_rows.numel()), W]

        # Collect per-sample information
        idx_proc_list.append(kept_tok)
        idx_rout_list.append(routed_tok)
        x_rout_list.append(x[b, routed_tok, :] if routed_tok.numel() > 0 else x.new_empty((0, C)))
        grid_sizes_proc_list.append(proc_grid)
        x_proc_list.append(x[b, kept_tok, :])

        # Track maximum lengths
        Lpmax = max(Lpmax, kept_tok.numel())
        Lrmax = max(Lrmax, routed_tok.numel())

    # Pad kept-token indices and tokens to common length
    idx_proc_pad = x.new_full((B, Lpmax), -1, dtype=torch.long)
    x_proc = x.new_zeros((B, Lpmax, C))
    for b in range(B):
        Lpi = x_proc_list[b].size(0)
        if Lpi > 0:
            idx_proc_pad[b, :Lpi] = idx_proc_list[b]
            x_proc[b, :Lpi, :] = x_proc_list[b]

    # Pad routed-token indices and tokens to common length
    idx_rout_pad = x.new_full((B, Lrmax), -1, dtype=torch.long)
    x_rout_pad = x.new_zeros((B, Lrmax, C))
    for b in range(B):
        Lri = x_rout_list[b].size(0)
        if Lri > 0:
            idx_rout_pad[b, :Lri] = idx_rout_list[b]
            x_rout_pad[b, :Lri, :] = x_rout_list[b]

    # Create state
    seq_lens_proc = torch.tensor([x_proc_list[b].size(0) for b in range(B)], device=device)
    grid_sizes_proc = torch.tensor(grid_sizes_proc_list, device=device)

    state = RowRouteState(
        idx_proc_pad=idx_proc_pad,
        idx_rout_pad=idx_rout_pad,
        x_rout_pad=x_rout_pad,
        seq_lens_orig=seq_lens.clone(),
        seq_lens_proc=seq_lens_proc,
        grid_sizes_orig=grid_sizes.clone(),
        grid_sizes_proc=grid_sizes_proc,
    )

    return x_proc, state


def reconstruct_row_routed_tokens(
    x_proc: torch.Tensor, state: RowRouteState
) -> torch.Tensor:
    """Reconstruct full-length x from processed stream and saved routed tokens.

    Parameters
    ----------
    x_proc: torch.Tensor
        Processed kept tokens (B, Lpmax, C) after the final routed block.
    state: RowRouteState
        Routing state returned by pack_row_routed_tokens.

    Returns
    -------
    x_full: torch.Tensor
        Full-length token sequence (B, Lmax, C) matching original padding.
    """
    B, Lpmax, C = x_proc.shape
    device = x_proc.device

    x_full_list: List[torch.Tensor] = []
    Lmax = int(state.seq_lens_orig.max().item())

    for b in range(B):
        Li = int(state.seq_lens_orig[b].item())
        Lpi = int(state.seq_lens_proc[b].item())

        # Initialize full tensor for this sample
        x_b = x_proc.new_zeros((Li, C))

        # Place kept tokens back at their original positions
        if Lpi > 0:
            mask_proc = state.idx_proc_pad[b] >= 0
            idx_proc = state.idx_proc_pad[b, mask_proc]
            x_b[idx_proc, :] = x_proc[b, :Lpi, :]

        # Place routed tokens back at their original positions
        mask_rout = state.idx_rout_pad[b] >= 0
        if mask_rout.any():
            idx_rout = state.idx_rout_pad[b, mask_rout]
            Lri = int(mask_rout.sum().item())
            x_b[idx_rout, :] = state.x_rout_pad[b, :Lri, :]

        # Pad to common length if needed
        if Li < Lmax:
            x_b_padded = x_proc.new_zeros((Lmax, C))
            x_b_padded[:Li, :] = x_b
            x_full_list.append(x_b_padded)
        else:
            x_full_list.append(x_b)

    x_full = torch.stack(x_full_list, dim=0)
    return x_full


def pack_spatial_auto_tokens(
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    keep_ratio: float,
    mode: str = "contiguous",
    rng_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, "RowRouteState | FrameRouteState | SpatialAutoRouteState"]:
    """Auto-detection spatial routing: F=1→rows, F>1→frames.

    This function implements hybrid routing that automatically chooses
    between row and frame routing based on content type.
    """
    device = x.device
    frame_counts = grid_sizes[:, 0]
    image_mask = frame_counts <= 1
    video_mask = ~image_mask

    num_images = int(image_mask.sum().item())
    num_videos = int(video_mask.sum().item())

    # Fast paths: uniform batch
    if num_videos == 0:
        logger.debug("Spatial auto: image-only batch detected, using row routing")
        return pack_row_routed_tokens(
            x,
            seq_lens,
            grid_sizes,
            keep_ratio,
            mode,
            rng_seed,
            auto_fallback=False,
        )
    if num_images == 0:
        if not _FRAME_ROUTING_AVAILABLE:
            raise ImportError("Frame routing not available for spatial_auto mode")
        logger.debug("Spatial auto: video-only batch detected, using frame routing")
        from .tread_frame import pack_frame_routed_tokens

        return pack_frame_routed_tokens(x, seq_lens, grid_sizes, keep_ratio, mode)

    # Mixed batch: process subsets separately, then merge in original order
    if not _FRAME_ROUTING_AVAILABLE:
        raise ImportError("Frame routing not available for mixed spatial_auto mode")

    from .tread_frame import pack_frame_routed_tokens

    image_indices = torch.nonzero(image_mask, as_tuple=False).flatten()
    video_indices = torch.nonzero(video_mask, as_tuple=False).flatten()

    # Process images with row routing
    x_proc_images: torch.Tensor | None = None
    row_state: RowRouteState | None = None
    row_lpmax = 0
    if num_images > 0:
        x_img = x.index_select(0, image_indices)
        seq_img = seq_lens.index_select(0, image_indices)
        grids_img = grid_sizes.index_select(0, image_indices)
        x_proc_images, row_state = pack_row_routed_tokens(
            x_img,
            seq_img,
            grids_img,
            keep_ratio,
            mode,
            rng_seed,
            auto_fallback=False,
        )
        row_lpmax = x_proc_images.size(1)

    # Process videos with frame routing
    x_proc_videos: torch.Tensor | None = None
    frame_state: "FrameRouteState | None" = None
    frame_lpmax = 0
    if num_videos > 0:
        x_vid = x.index_select(0, video_indices)
        seq_vid = seq_lens.index_select(0, video_indices)
        grids_vid = grid_sizes.index_select(0, video_indices)
        x_proc_videos, frame_state = pack_frame_routed_tokens(
            x_vid, seq_vid, grids_vid, keep_ratio, mode
        )
        frame_lpmax = x_proc_videos.size(1)

    Lpmax_total = max(row_lpmax, frame_lpmax)
    x_proc_full = x.new_zeros((x.size(0), Lpmax_total, x.size(2)))
    idx_proc_pad = x.new_full((x.size(0), Lpmax_total), -1, dtype=torch.long)
    seq_lens_proc = torch.zeros_like(seq_lens)
    grid_sizes_proc = torch.zeros_like(grid_sizes)

    if row_state is not None and x_proc_images is not None:
        for local_idx, global_idx in enumerate(image_indices.tolist()):
            kept = int(row_state.seq_lens_proc[local_idx].item())
            if kept > 0:
                x_proc_full[global_idx, :kept, :] = x_proc_images[local_idx, :kept, :]
                idx_vals = row_state.idx_proc_pad[local_idx]
                valid = idx_vals >= 0
                idx_proc_pad[global_idx, : valid.sum()] = idx_vals[valid]
            seq_lens_proc[global_idx] = row_state.seq_lens_proc[local_idx]
            grid_sizes_proc[global_idx] = row_state.grid_sizes_proc[local_idx]

    if frame_state is not None and x_proc_videos is not None:
        for local_idx, global_idx in enumerate(video_indices.tolist()):
            kept = int(frame_state.seq_lens_proc[local_idx].item())
            if kept > 0:
                x_proc_full[global_idx, :kept, :] = x_proc_videos[local_idx, :kept, :]
                idx_vals = frame_state.idx_proc_pad[local_idx]
                valid = idx_vals >= 0
                idx_proc_pad[global_idx, : valid.sum()] = idx_vals[valid]
            seq_lens_proc[global_idx] = frame_state.seq_lens_proc[local_idx]
            grid_sizes_proc[global_idx] = frame_state.grid_sizes_proc[local_idx]

    state = SpatialAutoRouteState(
        image_indices=image_indices.to(device=device),
        video_indices=video_indices.to(device=device),
        row_state=row_state,
        frame_state=frame_state,
        seq_lens_orig=seq_lens.clone(),
        seq_lens_proc=seq_lens_proc,
        grid_sizes_orig=grid_sizes.clone(),
        grid_sizes_proc=grid_sizes_proc,
        idx_proc_pad=idx_proc_pad,
        row_lpmax=row_lpmax,
        frame_lpmax=frame_lpmax,
    )

    return x_proc_full, state


def reconstruct_spatial_auto_tokens(
    x_proc: torch.Tensor,
    state: "RowRouteState | FrameRouteState | SpatialAutoRouteState"
) -> torch.Tensor:
    """Reconstruct tokens for spatial_auto mode.

    Parameters
    ----------
    x_proc: torch.Tensor
        Processed tokens from spatial_auto routing.
    state: RowRouteState | FrameRouteState
        State from pack_spatial_auto_tokens.

    Returns
    -------
    x_full: torch.Tensor
        Reconstructed full token sequence.
    """
    # Determine routing type from state type
    if isinstance(state, RowRouteState):
        return reconstruct_row_routed_tokens(x_proc, state)
    if isinstance(state, SpatialAutoRouteState):
        B = x_proc.size(0)
        C = x_proc.size(2)
        Lmax = int(state.seq_lens_orig.max().item())
        x_full = x_proc.new_zeros((B, Lmax, C))

        if state.row_state is not None and state.image_indices.numel() > 0:
            Lp = state.row_lpmax
            img_subset = x_proc.index_select(
                0, state.image_indices.to(device=x_proc.device)
            )[:, :Lp, :]
            img_full = reconstruct_row_routed_tokens(img_subset, state.row_state)
            for local_idx, global_idx in enumerate(
                state.image_indices.tolist()
            ):
                Li = int(state.row_state.seq_lens_orig[local_idx].item())
                x_full[global_idx, :Li, :] = img_full[local_idx, :Li, :]

        if state.frame_state is not None and state.video_indices.numel() > 0:
            from .tread_frame import reconstruct_frame_routed_tokens

            Lp = state.frame_lpmax
            vid_subset = x_proc.index_select(
                0, state.video_indices.to(device=x_proc.device)
            )[:, :Lp, :]
            vid_full = reconstruct_frame_routed_tokens(
                vid_subset, state.frame_state
            )
            for local_idx, global_idx in enumerate(
                state.video_indices.tolist()
            ):
                Li = int(state.frame_state.seq_lens_orig[local_idx].item())
                x_full[global_idx, :Li, :] = vid_full[local_idx, :Li, :]

        return x_full

    # Must be FrameRouteState
    from .tread_frame import reconstruct_frame_routed_tokens

    return reconstruct_frame_routed_tokens(x_proc, state)
