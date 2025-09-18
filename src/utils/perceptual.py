"""Perceptual loss utilities used during VAE training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

_LPIPS_MODELS: Dict[Tuple[str, str], torch.nn.Module] = {}
_SOBEL_CACHE: Dict[Tuple[torch.device, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _import_lpips() -> "torch.nn.Module":
    try:
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LPIPS is not available. Install it with 'pip install lpips' to enable perceptual loss."
        ) from exc
    return lpips


def get_lpips_model(device: torch.device, net: str = "vgg") -> torch.nn.Module:
    """Return a cached LPIPS model on the requested device.

    Parameters
    ----------
    device: torch.device
        Device to host the LPIPS network.
    net: str
        Backbone variant requested by downstream code (default: ``"vgg"``).

    Raises
    ------
    RuntimeError
        If the optional ``lpips`` dependency is not installed.
    """

    key = (str(device), net)
    if key in _LPIPS_MODELS:
        return _LPIPS_MODELS[key]

    lpips = _import_lpips()
    model = lpips.LPIPS(net=net)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    _LPIPS_MODELS[key] = model
    logger.info("Loaded LPIPS(%s) model onto %s", net, device)
    return model


def _get_sobel_kernels(
    device: torch.device, dtype: torch.dtype, channels: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype, channels)
    if key in _SOBEL_CACHE:
        return _SOBEL_CACHE[key]

    gx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    gy = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    )

    weight_x = gx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    weight_y = gy.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    _SOBEL_CACHE[key] = (weight_x, weight_y)
    return weight_x, weight_y


def sobel_edges(tensor: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for a batch of images.

    The input is expected to be ``[B, C, H, W]``. Gradient information is
    preserved so callers can backpropagate through the operation.
    """

    if tensor.dim() not in (4, 5):
        raise ValueError(
            f"sobel_edges expects a 4D or 5D tensor, got shape {tuple(tensor.shape)}"
        )

    reshape_back = False
    orig_batch = orig_time = None
    if tensor.dim() == 5:
        orig_batch, c, orig_time, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3, 4).reshape(orig_batch * orig_time, c, h, w)
        reshape_back = True

    b, c, _, _ = tensor.shape
    weight_x, weight_y = _get_sobel_kernels(tensor.device, tensor.dtype, c)

    grad_x = F.conv2d(tensor, weight_x, padding=1, groups=c)
    grad_y = F.conv2d(tensor, weight_y, padding=1, groups=c)

    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)

    if reshape_back and orig_batch is not None and orig_time is not None:
        magnitude = magnitude.view(orig_batch, orig_time, c, h, w)
        magnitude = magnitude.permute(0, 2, 1, 3, 4).contiguous()

    return magnitude
