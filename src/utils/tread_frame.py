import torch
from dataclasses import dataclass
from typing import Optional, Any, cast

# Alternate frame-based routing helpers (contiguous/stride frame selection)


@dataclass
class FrameRouteState:
    """State needed to reconstruct full sequences after frame routing.

    Attributes
    ----------
    idx_proc_pad: torch.Tensor
        Padded kept-token indices per sample. Shape (B, Lpmax), -1 padded.
    idx_rout_pad: torch.Tensor
        Padded routed-token indices per sample. Shape (B, Lrmax), -1 padded.
    x_rout_pad: torch.Tensor
        Padded original routed token representations. Shape (B, Lrmax, C).
    routed_lens: torch.Tensor
        Number of routed tokens per sample. Shape (B,).
    seq_lens_orig: torch.Tensor
        Original token lengths per sample before routing. Shape (B,).
    grid_sizes_orig: torch.Tensor
        Original (F,H,W) per sample before routing. Shape (B,3).
    seq_lens_proc: torch.Tensor
        Kept token lengths per sample for processed stream. Shape (B,).
    grid_sizes_proc: torch.Tensor
        (F_keep,H,W) per sample for processed stream. Shape (B,3).
    """

    idx_proc_pad: torch.Tensor
    idx_rout_pad: torch.Tensor
    x_rout_pad: torch.Tensor
    routed_lens: torch.Tensor
    seq_lens_orig: torch.Tensor
    grid_sizes_orig: torch.Tensor
    seq_lens_proc: torch.Tensor
    grid_sizes_proc: torch.Tensor


def _compute_frame_indices(
    num_frames: int, keep_ratio: float, mode: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute kept and routed frame indices.

    Parameters
    ----------
    num_frames: int
        Number of temporal frames F in the sample.
    keep_ratio: float
        Fraction in (0,1] of frames to keep.
    mode: str
        'contiguous' for a centered contiguous window; 'stride' for even spacing.
    device: torch.device
        Target device for returned tensors.

    Returns
    -------
    kept, routed: tuple[Tensor, Tensor]
        1D long tensors with kept and routed frame indices.
    """
    keep_frames = max(1, int(torch.floor(torch.tensor(num_frames * keep_ratio)).item()))
    if keep_frames >= num_frames:
        kept = torch.arange(0, num_frames, device=device, dtype=torch.long)
        routed = torch.zeros(0, device=device, dtype=torch.long)
        return kept, routed

    if mode == "stride":
        stride = max(1, int(round(num_frames / keep_frames)))
        kept = torch.arange(0, num_frames, stride, device=device, dtype=torch.long)[
            :keep_frames
        ]
        if kept.numel() < keep_frames:
            # Top up with remaining earliest indices not already selected
            chosen = set(kept.tolist())
            additional_idx = [i for i in range(num_frames) if i not in chosen]
            additional = torch.tensor(
                additional_idx[: keep_frames - kept.numel()],
                device=device,
                dtype=torch.long,
            )
            kept = torch.cat([kept, additional], dim=0)
    else:
        # contiguous center window
        start = max(0, (num_frames - keep_frames) // 2)
        kept = torch.arange(start, start + keep_frames, device=device, dtype=torch.long)

    all_idx = torch.arange(0, num_frames, device=device, dtype=torch.long)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask[kept] = False
    routed = all_idx[mask]
    return kept, routed


def pack_frame_routed_tokens(
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    keep_ratio: float,
    mode: str = "contiguous",
) -> tuple[torch.Tensor, FrameRouteState]:
    """Pack tokens corresponding to kept frames; stash routed tokens for later restore.

    Parameters
    ----------
    x: torch.Tensor
        Token sequence (B, Lmax, C), padded to same Lmax across batch.
    seq_lens: torch.Tensor
        Actual token lengths per sample (B,).
    grid_sizes: torch.Tensor
        Per-sample (F,H,W) grid sizes (B,3) for the tokenization.
    keep_ratio: float
        Fraction of frames to keep in (0,1].
    mode: str
        'contiguous' or 'stride'.

    Returns
    -------
    x_proc: torch.Tensor
        Packed kept tokens with uniform length (B, Lpmax, C).
    state: FrameRouteState
        Routing state for reconstruction and metadata for mid-route layers.
    """
    assert x.dim() == 3, "x must be (B, Lmax, C)"
    B, Lmax, C = x.shape
    device = x.device

    idx_proc_list: list[torch.Tensor] = []
    idx_rout_list: list[torch.Tensor] = []
    x_routed_list: list[torch.Tensor] = []
    seq_lens_proc_list: list[int] = []
    grid_sizes_proc_list: list[list[int]] = []
    x_proc_list: list[torch.Tensor] = []
    Lpmax = 0
    Lrmax = 0

    for b in range(B):
        Li = int(seq_lens[b].item())
        F, H, W = grid_sizes[b].tolist()
        assert Li == F * H * W, f"Unexpected seq_len {Li} != FHW {F*H*W}"

        kept_frames, routed_frames = _compute_frame_indices(F, keep_ratio, mode, device)
        hw = H * W
        kept_tok = torch.cat(
            [
                torch.arange(
                    int(f) * hw, int(f) * hw + hw, device=device, dtype=torch.long
                )
                for f in kept_frames
            ],
            dim=0,
        )
        routed_tok = torch.cat(
            [
                torch.arange(
                    int(f) * hw, int(f) * hw + hw, device=device, dtype=torch.long
                )
                for f in routed_frames
            ],
            dim=0,
        )

        idx_proc_list.append(kept_tok)
        idx_rout_list.append(routed_tok)

        x_b = x[b, :Li, :]
        x_proc_b = x_b.index_select(0, kept_tok)
        x_routed_b = x_b.index_select(0, routed_tok)
        x_routed_list.append(x_routed_b)

        Lpi = x_proc_b.shape[0]
        Lri = x_routed_b.shape[0]
        Lpmax = max(Lpmax, Lpi)
        Lrmax = max(Lrmax, Lri)
        seq_lens_proc_list.append(Lpi)
        grid_sizes_proc_list.append([int(kept_frames.numel()), H, W])
        x_proc_list.append(x_proc_b)

    # Pad processed tokens to Lpmax
    x_proc_padded = []
    for b in range(B):
        xpb = x_proc_list[b]
        pad_len = Lpmax - xpb.shape[0]
        if pad_len > 0:
            xpb = torch.cat([xpb, xpb.new_zeros(pad_len, xpb.shape[1])], dim=0)
        x_proc_padded.append(xpb.unsqueeze(0))
    x_proc = torch.cat(x_proc_padded, dim=0)

    # Build padded state tensors
    idx_proc_pad = x.new_full((B, Lpmax), fill_value=-1, dtype=torch.long)
    idx_rout_pad = x.new_full((B, Lrmax), fill_value=-1, dtype=torch.long)
    x_rout_pad = x.new_zeros((B, Lrmax, C), dtype=x.dtype)
    routed_lens = x.new_zeros((B,), dtype=torch.long)

    for b in range(B):
        Lpi = idx_proc_list[b].numel()
        Lri = idx_rout_list[b].numel()
        idx_proc_pad[b, :Lpi] = idx_proc_list[b]
        idx_rout_pad[b, :Lri] = idx_rout_list[b]
        x_rout_pad[b, :Lri, :] = x_routed_list[b]
        routed_lens[b] = Lri

    seq_lens_proc = torch.tensor(
        seq_lens_proc_list, device=device, dtype=seq_lens.dtype
    )
    grid_sizes_proc = torch.tensor(
        grid_sizes_proc_list, device=device, dtype=grid_sizes.dtype
    )

    state = FrameRouteState(
        idx_proc_pad=idx_proc_pad,
        idx_rout_pad=idx_rout_pad,
        x_rout_pad=x_rout_pad,
        routed_lens=routed_lens,
        seq_lens_orig=seq_lens.clone(),
        grid_sizes_orig=grid_sizes.clone(),
        seq_lens_proc=seq_lens_proc,
        grid_sizes_proc=grid_sizes_proc,
    )

    return x_proc, state


def reconstruct_frame_routed_tokens(
    x_proc: torch.Tensor, state: FrameRouteState
) -> torch.Tensor:
    """Reconstruct full-length x from processed stream and saved routed tokens.

    Parameters
    ----------
    x_proc: torch.Tensor
        Processed kept tokens (B, Lpmax, C) after the final routed block.
    state: FrameRouteState
        Routing state returned by pack_frame_routed_tokens.

    Returns
    -------
    x_full: torch.Tensor
        Full-length token sequence (B, Lmax, C) matching original padding.
    """
    B, Lpmax, C = x_proc.shape
    device = x_proc.device

    x_full_list: list[torch.Tensor] = []
    Lmax = int(state.seq_lens_orig.max().item())

    for b in range(B):
        Li = int(state.seq_lens_orig[b].item())
        Lpi = int(state.seq_lens_proc[b].item())
        Lri = int(state.routed_lens[b].item())

        xpb = x_proc[b, :Lpi, :]
        xrb = state.x_rout_pad[b, :Lri, :]
        idx_proc = state.idx_proc_pad[b, :Lpi]
        idx_rout = state.idx_rout_pad[b, :Lri]
        assert (idx_proc.numel() + idx_rout.numel()) == Li

        xb = torch.empty(Li, C, device=device, dtype=x_proc.dtype)
        xb.index_copy_(0, idx_proc, xpb)
        xb.index_copy_(0, idx_rout, xrb)

        if Li < Lmax:
            xb = torch.cat([xb, xb.new_zeros(Lmax - Li, C)], dim=0)
        x_full_list.append(xb.unsqueeze(0))

    x_full = torch.cat(x_full_list, dim=0)
    return x_full
