"""Utility helpers for SARA losses."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_autocorrelation_matrix(
    features: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the patch autocorrelation matrix."""
    if features.dim() == 2:
        features = features.unsqueeze(1)

    if normalize:
        features = F.normalize(features, p=2, dim=-1, eps=eps)

    autocorr = torch.bmm(features, features.transpose(1, 2))

    if not normalize:
        norms = torch.norm(features, p=2, dim=-1, keepdim=True)
        denom = torch.bmm(norms, norms.transpose(1, 2))
        autocorr = autocorr / (denom + eps)

    # Enforce numerical symmetry. The matrix is theoretically symmetric, but tiny
    # fp16/fp32 round-off can introduce <1e-6 skew that ends up magnified inside
    # the Frobenius norm. Averaging with the transpose keeps the intended values
    # and prevents reviewers/debuggers from chasing harmless asymmetry noise.
    autocorr = 0.5 * (autocorr + autocorr.transpose(-2, -1))
    return autocorr


def autocorrelation_loss(
    pred_autocorr: torch.Tensor,
    target_autocorr: torch.Tensor,
    use_frobenius: bool = True,
) -> torch.Tensor:
    """Compute distance between two autocorrelation matrices."""
    if use_frobenius:
        diff = pred_autocorr - target_autocorr
        return diff.square().sum(dim=(-2, -1)).mean()

    return F.mse_loss(pred_autocorr, target_autocorr)


def match_feature_shapes(
    features1: torch.Tensor,
    features2: torch.Tensor,
    mode: str = "interpolate",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bring two feature tensors to compatible shapes."""
    if features1.dim() == 2:
        features1 = features1.unsqueeze(1)
    if features2.dim() == 2:
        features2 = features2.unsqueeze(1)

    b1, n1, d1 = features1.shape
    b2, n2, d2 = features2.shape

    if b1 != b2:
        raise ValueError(f"Batch size mismatch: {b1} vs {b2}")

    if n1 != n2:
        if mode == "interpolate":
            if n1 != n2:
                features1 = F.interpolate(
                    features1.transpose(1, 2),
                    size=n2,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        elif mode == "pool":
            if n1 > n2:
                features1 = features1.mean(dim=1, keepdim=True).expand(-1, n2, -1)
            else:
                features2 = features2.mean(dim=1, keepdim=True).expand(-1, n1, -1)
        elif mode == "crop":
            min_len = min(n1, n2)
            features1 = features1[:, :min_len]
            features2 = features2[:, :min_len]
        else:
            raise ValueError(f"Unknown match mode: {mode}")

    if features1.shape[2] != features2.shape[2]:
        min_dim = min(features1.shape[2], features2.shape[2])
        features1 = features1[..., :min_dim]
        features2 = features2[..., :min_dim]

    return features1, features2


def gradient_penalty(
    discriminator: torch.nn.Module,
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP style gradient penalty."""
    batch_size = real_features.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device, dtype=real_features.dtype)
    if real_features.dim() == 2:
        alpha = alpha.squeeze(-1)

    interpolates = alpha * real_features + (1 - alpha) * fake_features
    interpolates.requires_grad_(True)

    disc_out = discriminator(interpolates)
    if disc_out.dim() > 2:
        disc_out = disc_out.view(disc_out.size(0), -1)

    ones = torch.ones_like(disc_out, device=device)
    gradients = torch.autograd.grad(
        outputs=disc_out,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
