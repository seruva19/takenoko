"""Local Gram Flow utilities for Structure-From-Tracking distillation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as nnf


def _compute_lgf_linear(
    features: torch.Tensor,
    neighborhood_size: int,
) -> torch.Tensor:
    """Fallback LGF for non-square token layouts using 1D token neighborhoods."""
    batch_size, frame_count, token_count, hidden_dim = features.shape
    if frame_count < 2:
        return features.new_zeros((batch_size, 0, token_count, neighborhood_size))

    prev_tokens = features[:, :-1].reshape(batch_size * (frame_count - 1), token_count, hidden_dim)
    next_tokens = features[:, 1:].reshape(batch_size * (frame_count - 1), token_count, hidden_dim)

    next_channels = next_tokens.transpose(1, 2)  # [BF, D, T]
    padded = nnf.pad(
        next_channels,
        (neighborhood_size // 2, neighborhood_size // 2),
        mode="replicate",
    )
    local_windows = padded.unfold(dimension=2, size=neighborhood_size, step=1)
    # [BF, D, T, K] -> [BF, T, K, D]
    local_windows = local_windows.permute(0, 2, 3, 1)

    similarities = (prev_tokens.unsqueeze(2) * local_windows).sum(dim=-1)
    return similarities.view(batch_size, frame_count - 1, token_count, neighborhood_size)


def compute_local_gram_flow(
    features: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Compute Local Gram Flow similarities for adjacent frame pairs.

    Args:
        features: Tensor with shape [B, F, T, D].
        kernel_size: Odd neighborhood size used for local matching.
    """
    if features.dim() != 4:
        raise ValueError(
            f"compute_local_gram_flow expects [B, F, T, D], got {tuple(features.shape)}"
        )
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd integer >= 3, got {kernel_size}"
        )

    batch_size, frame_count, token_count, hidden_dim = features.shape
    if frame_count < 2:
        return features.new_zeros((batch_size, 0, token_count, kernel_size * kernel_size))

    side = int(math.isqrt(token_count))
    if side * side != token_count:
        return _compute_lgf_linear(features, neighborhood_size=kernel_size * kernel_size)

    prev_frame = features[:, :-1].reshape(batch_size * (frame_count - 1), side, side, hidden_dim)
    next_frame = features[:, 1:].reshape(batch_size * (frame_count - 1), side, side, hidden_dim)

    prev_tokens = prev_frame.view(batch_size * (frame_count - 1), side * side, hidden_dim)
    next_channels = next_frame.permute(0, 3, 1, 2)
    local_windows = nnf.unfold(
        next_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )
    # [BF, D*K*K, T] -> [BF, T, K*K, D]
    local_windows = local_windows.transpose(1, 2).reshape(
        batch_size * (frame_count - 1),
        side * side,
        kernel_size * kernel_size,
        hidden_dim,
    )
    similarities = (prev_tokens.unsqueeze(2) * local_windows).sum(dim=-1)
    return similarities.view(
        batch_size,
        frame_count - 1,
        side * side,
        kernel_size * kernel_size,
    )


def lgf_kl_divergence(
    student_similarity: torch.Tensor,
    teacher_similarity: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """KL(P_teacher || Q_student) over LGF local similarity distributions."""
    if student_similarity.shape != teacher_similarity.shape:
        raise ValueError(
            "LGF similarity shapes must match. "
            f"student={tuple(student_similarity.shape)}, "
            f"teacher={tuple(teacher_similarity.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if student_similarity.numel() == 0:
        return student_similarity.new_tensor(0.0)

    teacher_prob = nnf.softmax(teacher_similarity / temperature, dim=-1)
    student_log_prob = nnf.log_softmax(student_similarity / temperature, dim=-1)
    kl = nnf.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return kl.mean()


def compute_lgf_alignment_loss(
    student_similarity: torch.Tensor,
    teacher_similarity: torch.Tensor,
    *,
    mode: str = "kl",
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute LGF alignment loss.

    Modes:
    - "kl": KL(P_teacher || Q_student) over local similarity distributions.
    - "l2": Mean-squared error over raw LGF similarity tensors.
    """
    if mode == "kl":
        return lgf_kl_divergence(
            student_similarity=student_similarity,
            teacher_similarity=teacher_similarity,
            temperature=temperature,
        )
    if mode == "l2":
        if student_similarity.shape != teacher_similarity.shape:
            raise ValueError(
                "LGF similarity shapes must match for l2 mode. "
                f"student={tuple(student_similarity.shape)}, "
                f"teacher={tuple(teacher_similarity.shape)}"
            )
        if student_similarity.numel() == 0:
            return student_similarity.new_tensor(0.0)
        return nnf.mse_loss(student_similarity, teacher_similarity)
    raise ValueError(f"Unsupported LGF alignment mode: {mode}")
