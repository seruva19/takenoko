from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FrequencyDecouplingDiagnosticsResult:
    pred_low_freq_energy: Optional[torch.Tensor]
    pred_high_freq_energy: Optional[torch.Tensor]
    pred_high_low_ratio: Optional[torch.Tensor]
    target_low_freq_energy: Optional[torch.Tensor]
    target_high_freq_energy: Optional[torch.Tensor]
    target_high_low_ratio: Optional[torch.Tensor]
    high_low_ratio_gap: Optional[torch.Tensor]
    block_size: Optional[torch.Tensor]
    low_freq_extent: Optional[torch.Tensor]


@dataclass
class BandBalancedLossResult:
    scaled_loss: Optional[torch.Tensor]
    raw_loss: Optional[torch.Tensor]
    low_freq_loss: Optional[torch.Tensor]
    high_freq_loss: Optional[torch.Tensor]
    low_freq_weight: Optional[torch.Tensor]
    high_freq_weight: Optional[torch.Tensor]
    block_size: Optional[torch.Tensor]
    low_freq_extent: Optional[torch.Tensor]


def _resolve_block_size(x: torch.Tensor, requested_block_size: int) -> int:
    spatial_h = int(x.shape[-2])
    spatial_w = int(x.shape[-1])
    return max(2, min(int(requested_block_size), spatial_h, spatial_w))


def _build_dct_matrix(
    block_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = torch.arange(block_size, device=device, dtype=dtype)
    basis = torch.cos(
        (math.pi / block_size)
        * (indices[:, None] + 0.5)
        * indices[None, :]
    )
    basis[0, :] = basis[0, :] * math.sqrt(1.0 / block_size)
    if block_size > 1:
        basis[1:, :] = basis[1:, :] * math.sqrt(2.0 / block_size)
    return basis


def _extract_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    cropped_h = height - (height % block_size)
    cropped_w = width - (width % block_size)
    if cropped_h <= 0 or cropped_w <= 0:
        raise ValueError("Spatial size is too small for the requested DCT block size")

    x = x[..., :cropped_h, :cropped_w]
    leading = int(torch.tensor(x.shape[:-2]).prod().item()) if x.dim() > 2 else 1
    x = x.reshape(leading, cropped_h, cropped_w)
    x = x.reshape(
        leading,
        cropped_h // block_size,
        block_size,
        cropped_w // block_size,
        block_size,
    )
    x = x.permute(0, 1, 3, 2, 4).reshape(-1, block_size, block_size)
    return x


def _compute_band_energies(
    x: torch.Tensor,
    *,
    block_size: int,
    low_freq_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_block_size = _resolve_block_size(x, block_size)
    dct_basis = _build_dct_matrix(
        resolved_block_size,
        device=x.device,
        dtype=torch.float32,
    )
    blocks = _extract_blocks(x.to(torch.float32), resolved_block_size)
    dct_blocks = torch.einsum("ab,nbc,dc->nad", dct_basis, blocks, dct_basis)
    energy = dct_blocks.square()

    low_extent = max(1, int(math.ceil(resolved_block_size * low_freq_ratio)))
    low_mask = torch.zeros(
        resolved_block_size,
        resolved_block_size,
        device=energy.device,
        dtype=torch.bool,
    )
    low_mask[:low_extent, :low_extent] = True
    high_mask = ~low_mask

    low_energy = energy[:, low_mask].mean()
    if bool(high_mask.any()):
        high_energy = energy[:, high_mask].mean()
    else:
        high_energy = energy.new_zeros(())
    ratio = high_energy / low_energy.clamp_min(1e-8)
    return low_energy, high_energy, ratio, energy.new_tensor(float(low_extent))


def _compute_dct_error_bands(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    block_size: int,
    low_freq_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_block_size = _resolve_block_size(model_pred, block_size)
    dct_basis = _build_dct_matrix(
        resolved_block_size,
        device=model_pred.device,
        dtype=torch.float32,
    )
    pred_blocks = _extract_blocks(model_pred.to(torch.float32), resolved_block_size)
    target_blocks = _extract_blocks(target.to(torch.float32), resolved_block_size)

    pred_dct = torch.einsum("ab,nbc,dc->nad", dct_basis, pred_blocks, dct_basis)
    target_dct = torch.einsum("ab,nbc,dc->nad", dct_basis, target_blocks, dct_basis)
    error = (pred_dct - target_dct).square()

    low_extent = max(1, int(math.ceil(resolved_block_size * low_freq_ratio)))
    low_mask = torch.zeros(
        resolved_block_size,
        resolved_block_size,
        device=error.device,
        dtype=torch.bool,
    )
    low_mask[:low_extent, :low_extent] = True
    high_mask = ~low_mask

    low_loss = error[:, low_mask].mean()
    if bool(high_mask.any()):
        high_loss = error[:, high_mask].mean()
    else:
        high_loss = error.new_zeros(())
    return low_loss, high_loss, error.new_tensor(float(low_extent))


def compute_frequency_decoupling_diagnostics(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    block_size: int,
    low_freq_ratio: float,
) -> FrequencyDecouplingDiagnosticsResult:
    """Compute DeCo-inspired low/high frequency DCT energy diagnostics."""
    if model_pred.dim() not in {4, 5} or target.dim() not in {4, 5}:
        return FrequencyDecouplingDiagnosticsResult(
            pred_low_freq_energy=None,
            pred_high_freq_energy=None,
            pred_high_low_ratio=None,
            target_low_freq_energy=None,
            target_high_freq_energy=None,
            target_high_low_ratio=None,
            high_low_ratio_gap=None,
            block_size=None,
            low_freq_extent=None,
        )

    pred_low, pred_high, pred_ratio, low_extent = _compute_band_energies(
        model_pred.detach(),
        block_size=block_size,
        low_freq_ratio=low_freq_ratio,
    )
    target_low, target_high, target_ratio, _ = _compute_band_energies(
        target.detach(),
        block_size=block_size,
        low_freq_ratio=low_freq_ratio,
    )

    resolved_block_size = _resolve_block_size(model_pred, block_size)
    return FrequencyDecouplingDiagnosticsResult(
        pred_low_freq_energy=pred_low,
        pred_high_freq_energy=pred_high,
        pred_high_low_ratio=pred_ratio,
        target_low_freq_energy=target_low,
        target_high_freq_energy=target_high,
        target_high_low_ratio=target_ratio,
        high_low_ratio_gap=pred_ratio - target_ratio,
        block_size=pred_low.new_tensor(float(resolved_block_size)),
        low_freq_extent=low_extent,
    )


def compute_band_balanced_dct_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_weight: float,
    low_freq_weight: float,
    high_freq_weight: float,
    block_size: int,
    low_freq_ratio: float,
) -> BandBalancedLossResult:
    """Compute a DeCo-inspired weighted DCT reconstruction auxiliary loss."""
    if model_pred.dim() not in {4, 5} or target.dim() not in {4, 5}:
        return BandBalancedLossResult(
            scaled_loss=None,
            raw_loss=None,
            low_freq_loss=None,
            high_freq_loss=None,
            low_freq_weight=None,
            high_freq_weight=None,
            block_size=None,
            low_freq_extent=None,
        )

    low_loss, high_loss, low_extent = _compute_dct_error_bands(
        model_pred,
        target,
        block_size=block_size,
        low_freq_ratio=low_freq_ratio,
    )
    used_low_weight = max(0.0, float(low_freq_weight))
    used_high_weight = max(0.0, float(high_freq_weight))
    if used_low_weight <= 0.0 and used_high_weight <= 0.0:
        used_low_weight = 1.0

    denom = max(used_low_weight + used_high_weight, 1e-8)
    raw_loss = (
        (low_loss * used_low_weight) + (high_loss * used_high_weight)
    ) / denom
    scaled_loss = raw_loss * float(loss_weight)
    resolved_block_size = _resolve_block_size(model_pred, block_size)

    return BandBalancedLossResult(
        scaled_loss=scaled_loss,
        raw_loss=raw_loss.detach(),
        low_freq_loss=low_loss.detach(),
        high_freq_loss=high_loss.detach(),
        low_freq_weight=low_loss.new_tensor(float(used_low_weight)),
        high_freq_weight=low_loss.new_tensor(float(used_high_weight)),
        block_size=low_loss.new_tensor(float(resolved_block_size)),
        low_freq_extent=low_extent,
    )
