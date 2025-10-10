import torch


def _pairwise_l2_sq_legacy(z_flat: torch.Tensor) -> torch.Tensor:
    """Legacy implementation kept for backwards compatibility."""
    return torch.pdist(z_flat, p=2).pow(2)


def _pairwise_l2_sq_official(z_flat: torch.Tensor) -> torch.Tensor:
    """Official implementation from the public DispLoss PyTorch release."""
    feature_dim = max(z_flat.size(1), 1)
    diff = torch.pdist(z_flat, p=2).pow(2) / feature_dim
    zeros = torch.zeros(
        z_flat.size(0), device=z_flat.device, dtype=z_flat.dtype
    )
    # Duplicate the strictly upper-triangular entries to emulate the full BxB matrix
    diff = torch.cat((diff, diff, zeros), dim=0)
    return diff


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
        metric: "l2_sq", "l2_sq_legacy", or "cosine"
    Returns:
        Scalar tensor loss. Zero if batch size <= 1.
    """
    if z is None:
        return torch.tensor(0.0, device="cpu")
    if z.shape[0] <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    z_flat = z.view(z.shape[0], -1)
    metric_lower = metric.lower()

    if metric_lower == "l2_sq":
        d = _pairwise_l2_sq_official(z_flat)
    elif metric_lower in {"l2_sq_legacy", "legacy"}:
        d = _pairwise_l2_sq_legacy(z_flat)
    elif metric_lower == "cosine":
        d = _pairwise_neg_cosine(z_flat)
    else:
        raise ValueError(f"Unknown dispersive loss metric: {metric}")

    scaled = -d / max(float(tau), 1e-6)
    loss = torch.log(torch.mean(torch.exp(scaled)) + 1e-12)
    return loss
