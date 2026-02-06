from __future__ import annotations

from typing import Optional, Tuple

import torch

from wan.modules.attention import flash_attention


def _select_topk_frames(
    frame_scores: torch.Tensor,
    k_sel: int,
    always_keep_first_frame: bool,
    always_keep_last_frame: bool,
) -> torch.Tensor:
    history_frames = int(frame_scores.shape[0])
    if history_frames <= 0:
        return torch.zeros(0, device=frame_scores.device, dtype=torch.long)

    k_sel = max(1, min(int(k_sel), history_frames))
    scores = frame_scores.clone()

    # Force anchor frames by boosting routing score before top-k selection.
    if always_keep_first_frame:
        scores[0] = float("inf")
    if always_keep_last_frame and history_frames > 1:
        scores[history_frames - 1] = float("inf")

    selected = torch.topk(scores, k=k_sel, largest=True).indices
    selected = torch.unique(selected, sorted=True)
    if selected.numel() < k_sel:
        remaining = torch.arange(
            history_frames,
            device=frame_scores.device,
            dtype=torch.long,
        )
        used = torch.zeros_like(remaining, dtype=torch.bool)
        used[selected] = True
        fill = remaining[~used][: k_sel - selected.numel()]
        selected = torch.cat([selected, fill], dim=0)
        selected = torch.unique(selected, sorted=True)
    return selected


def _flash_attend_single_frame(
    q_frame: torch.Tensor,
    k_cat: torch.Tensor,
    v_cat: torch.Tensor,
    *,
    attn_mode: str,
    split_attn: bool,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    # Input/Output shapes:
    # q_frame: [Tq, H, D], k_cat/v_cat: [Tk, H, D] -> out: [Tq, H, D]
    qkv = [q_frame.unsqueeze(0), k_cat.unsqueeze(0), v_cat.unsqueeze(0)]
    k_lens_arg = None
    if attn_mode in ("flash3", "sageattn"):
        k_lens_arg = torch.tensor(
            [int(k_cat.shape[0])],
            device=q_frame.device,
            dtype=torch.int32,
        )
    out = flash_attention(
        qkv,
        k_lens=k_lens_arg,
        window_size=window_size,
        attn_mode=attn_mode,
        split_attn=split_attn,
    )
    return out[0]


def _history_branch_fused_attention_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    top_k_frames: int,
    always_keep_first_frame: bool,
    always_keep_last_frame: bool,
    extra_tokens: int,
) -> Optional[torch.Tensor]:
    # This mode assumes pure video tokens without prepended aux tokens.
    if extra_tokens != 0:
        return None

    bsz, _, num_heads, head_dim = q.shape
    scale = float(head_dim) ** -0.5
    output = torch.zeros_like(q)

    for b in range(bsz):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue
        f, h, w = [int(x) for x in grid_sizes[b].tolist()]
        tokens_per_frame = int(h * w)
        if f <= 1 or tokens_per_frame <= 0:
            return None
        if seq_len != f * tokens_per_frame:
            return None

        q_b = q[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        k_b = k[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        v_b = v[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        frame_desc = k_b.mean(dim=1)  # [F, H, D]
        out_b = torch.zeros_like(q_b)

        for frame_idx in range(f):
            q_frame = q_b[frame_idx]
            k_intra = k_b[frame_idx]
            v_intra = v_b[frame_idx]

            history_frames = frame_idx
            if history_frames <= 0:
                for head_idx in range(num_heads):
                    q_h = q_frame[:, head_idx, :]
                    k_h = k_intra[:, head_idx, :]
                    v_h = v_intra[:, head_idx, :]
                    logits_intra = torch.matmul(q_h, k_h.transpose(0, 1)) * scale
                    weights_intra = torch.softmax(logits_intra, dim=-1)
                    out_b[frame_idx, :, head_idx, :] = torch.matmul(
                        weights_intra, v_h
                    )
                continue

            hist_desc = frame_desc[:frame_idx]  # [history_frames, H, D]
            hist_k = k_b[:frame_idx]  # [history_frames, T, H, D]
            hist_v = v_b[:frame_idx]  # [history_frames, T, H, D]
            k_sel = min(max(1, int(top_k_frames)), history_frames)

            for token_idx in range(tokens_per_frame):
                q_token = q_frame[token_idx]  # [H, D]
                scores = torch.einsum("hd,fhd->hf", q_token, hist_desc)

                for head_idx in range(num_heads):
                    q_h = q_token[head_idx]
                    intra_k_h = k_intra[:, head_idx, :]
                    intra_v_h = v_intra[:, head_idx, :]

                    idx = torch.topk(scores[head_idx], k=k_sel, largest=True).indices
                    if always_keep_first_frame and 0 not in idx:
                        idx[-1] = 0
                    if always_keep_last_frame and (history_frames - 1) not in idx:
                        idx[-1] = history_frames - 1
                    idx = torch.unique(idx, sorted=True)

                    hist_k_h = hist_k[idx, :, head_idx, :].reshape(-1, head_dim)
                    hist_v_h = hist_v[idx, :, head_idx, :].reshape(-1, head_dim)

                    logits_intra = torch.matmul(intra_k_h, q_h) * scale
                    if hist_k_h.numel() > 0:
                        logits_hist = torch.matmul(hist_k_h, q_h) * scale
                        lse = torch.logaddexp(
                            torch.logsumexp(logits_intra, dim=0),
                            torch.logsumexp(logits_hist, dim=0),
                        )
                        weights_intra = torch.exp(logits_intra - lse)
                        weights_hist = torch.exp(logits_hist - lse)
                        out_h = (
                            torch.matmul(weights_intra, intra_v_h)
                            + torch.matmul(weights_hist, hist_v_h)
                        )
                    else:
                        weights_intra = torch.softmax(logits_intra, dim=0)
                        out_h = torch.matmul(weights_intra, intra_v_h)

                    out_b[frame_idx, token_idx, head_idx, :] = out_h

        output[b, :seq_len] = out_b.view(seq_len, num_heads, head_dim)

    return output


def _history_branch_kernel_frame_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    top_k_frames: int,
    always_keep_first_frame: bool,
    always_keep_last_frame: bool,
    extra_tokens: int,
    *,
    attn_mode: str,
    split_attn: bool,
    window_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    # Kernel-backed approximation:
    # frame-level top-k routing + flash attention; avoids token/head Python loops.
    if extra_tokens != 0:
        return None

    bsz, _, num_heads, head_dim = q.shape
    output = torch.zeros_like(q)

    for b in range(bsz):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue
        f, h, w = [int(x) for x in grid_sizes[b].tolist()]
        tokens_per_frame = int(h * w)
        if f <= 1 or tokens_per_frame <= 0:
            return None
        if seq_len != f * tokens_per_frame:
            return None

        q_b = q[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        k_b = k[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        v_b = v[b, :seq_len].view(f, tokens_per_frame, num_heads, head_dim)
        frame_desc = k_b.mean(dim=1)  # [F, H, D]
        out_b = torch.zeros_like(q_b)

        for frame_idx in range(f):
            q_frame = q_b[frame_idx]  # [T, H, D]
            k_intra = k_b[frame_idx]
            v_intra = v_b[frame_idx]

            history_frames = frame_idx
            if history_frames <= 0:
                out_b[frame_idx] = _flash_attend_single_frame(
                    q_frame,
                    k_intra,
                    v_intra,
                    attn_mode=attn_mode,
                    split_attn=split_attn,
                    window_size=window_size,
                )
                continue

            hist_desc = frame_desc[:frame_idx]  # [Fh, H, D]
            # Frame-level score proxy: mean query descriptor vs frame descriptors.
            q_desc = q_frame.mean(dim=0)  # [H, D]
            frame_scores = torch.einsum("hd,fhd->f", q_desc, hist_desc)
            selected = _select_topk_frames(
                frame_scores,
                k_sel=min(max(1, int(top_k_frames)), history_frames),
                always_keep_first_frame=always_keep_first_frame,
                always_keep_last_frame=always_keep_last_frame,
            )

            hist_k = k_b[selected].reshape(-1, num_heads, head_dim)
            hist_v = v_b[selected].reshape(-1, num_heads, head_dim)
            k_cat = torch.cat([k_intra, hist_k], dim=0)
            v_cat = torch.cat([v_intra, hist_v], dim=0)
            out_b[frame_idx] = _flash_attend_single_frame(
                q_frame,
                k_cat,
                v_cat,
                attn_mode=attn_mode,
                split_attn=split_attn,
                window_size=window_size,
            )

        output[b, :seq_len] = out_b.view(seq_len, num_heads, head_dim)

    return output


def history_branch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    top_k_frames: int,
    always_keep_first_frame: bool,
    always_keep_last_frame: bool,
    extra_tokens: int,
    *,
    backend: str,
    attn_mode: str,
    split_attn: bool,
    window_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    backend = str(backend).lower()
    if backend == "kernel_frame_topk":
        return _history_branch_kernel_frame_topk_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            top_k_frames=top_k_frames,
            always_keep_first_frame=always_keep_first_frame,
            always_keep_last_frame=always_keep_last_frame,
            extra_tokens=extra_tokens,
            attn_mode=attn_mode,
            split_attn=split_attn,
            window_size=window_size,
        )
    return _history_branch_fused_attention_exact(
        q=q,
        k=k,
        v=v,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        top_k_frames=top_k_frames,
        always_keep_first_frame=always_keep_first_frame,
        always_keep_last_frame=always_keep_last_frame,
        extra_tokens=extra_tokens,
    )
