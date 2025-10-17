from __future__ import annotations

from typing import Tuple

import torch


def compute_energy_from_features(
    *,
    features: torch.Tensor,
    latents: torch.Tensor,
    mode: str = "dot",
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the EqM energy value and its gradient with respect to the latents.

    The default "dot" mode mirrors the EqM-E dot product parameterisation:
        E(x) = sum_i f_theta(x)_i * x_i

    Args:
        features: Network output tensor shaped like the latent input (B, C, F, H, W)
        latents: Input latents (must require_grad=True when create_graph=True)
        mode: Energy formulation ("dot", "l2", or "mean")
        create_graph: Whether to retain the graph for higher-order gradients

    Returns:
        gradient: dE/dx shaped like `latents`
        energy: Per-sample energy values shaped [B]
    """

    supported_modes = {"dot", "l2", "mean"}
    mode_key = mode.lower()
    if mode_key not in supported_modes:
        raise ValueError(
            f"Unsupported EqM energy mode '{mode}'. Supported: {sorted(supported_modes)}."
        )

    if not latents.requires_grad:
        raise RuntimeError("EqM energy head requires latents with requires_grad=True.")

    # Align dtypes for numerical stability
    features = features.to(latents.dtype)

    if mode_key == "dot":
        energy = (features * latents).flatten(1).sum(dim=1)
    elif mode_key == "mean":
        energy = (features * latents).flatten(1).mean(dim=1)
    else:  # mode_key == "l2"
        energy = -0.5 * (features.flatten(1) ** 2).sum(dim=1)

    # Compute gradient of the scalar energy wrt latents
    grad = torch.autograd.grad(
        energy.sum(),
        latents,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )[0]

    return grad, energy.detach()


def register_energy_head_metadata(model: torch.nn.Module, *, mode: str) -> None:
    """Attach lightweight metadata to the transformer for debugging/logging."""
    try:
        setattr(model, "_eqm_energy_mode", mode)
    except Exception:
        # The attribute is purely informational; ignore failures quietly.
        pass
