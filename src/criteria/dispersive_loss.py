## Based on https://arxiv.org/abs/2506.09027v1

import torch


def _pairwise_l2_sq(z_flat: torch.Tensor) -> torch.Tensor:
    return torch.pdist(z_flat, p=2).pow(2)


def _pairwise_neg_cosine(z_flat: torch.Tensor) -> torch.Tensor:
    z_norm = torch.nn.functional.normalize(z_flat, p=2, dim=1)
    sim = torch.mm(z_norm, z_norm.t())
    idx = torch.triu_indices(sim.size(0), sim.size(1), offset=1)
    return -sim[idx[0], idx[1]]


def dispersive_loss_info_nce(
    z: torch.Tensor, tau: float = 0.5, metric: str = "l2_sq"
) -> torch.Tensor:
    """Compute InfoNCE-style dispersive loss with selectable metric.

    Args:
        z: Tensor (B, ...)
        tau: Temperature (> 0)
        metric: "l2_sq" or "cosine"
    Returns:
        Scalar tensor loss. Zero if batch size <= 1.
    """
    if z is None:
        return torch.tensor(0.0, device="cpu")
    if z.shape[0] <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    z_flat = z.view(z.shape[0], -1)

    if metric == "l2_sq":
        d = _pairwise_l2_sq(z_flat)
    elif metric == "cosine":
        d = _pairwise_neg_cosine(z_flat)
    else:
        raise ValueError(f"Unknown dispersive loss metric: {metric}")

    scaled = -d / max(tau, 1e-6)
    loss = torch.log(torch.mean(torch.exp(scaled)) + 1e-12)
    return loss
