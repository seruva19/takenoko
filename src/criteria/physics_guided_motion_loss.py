from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


def _normalize_components(components: Optional[Iterable[str]]) -> List[str]:
    if components is None:
        return []
    normalized = []
    for item in components:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if name:
            normalized.append(name)
    return normalized


def _build_lowpass_mask(
    freq_t: torch.Tensor,
    freq_x: torch.Tensor,
    freq_y: torch.Tensor,
    lowpass_ratio: float,
) -> torch.Tensor:
    return (
        freq_t.abs() <= lowpass_ratio
    ) & (freq_x.abs() <= lowpass_ratio) & (freq_y.abs() <= lowpass_ratio)


def _compute_translation_loss(
    energy: torch.Tensor,
    freq_t: torch.Tensor,
    freq_x: torch.Tensor,
    freq_y: torch.Tensor,
    ridge_lambda: float,
    eps: float,
) -> torch.Tensor:
    losses = []
    for sample_energy in energy:
        weights = sample_energy.flatten()
        if weights.sum() <= eps:
            losses.append(torch.tensor(0.0, device=sample_energy.device))
            continue

        a = torch.stack(
            [freq_x.flatten(), freq_y.flatten(), torch.ones_like(freq_x).flatten()],
            dim=1,
        )
        b = -freq_t.flatten()

        w = weights.unsqueeze(1)
        at_w = (a * w).transpose(0, 1)
        ata = at_w @ a
        ata = ata + ridge_lambda * torch.eye(3, device=ata.device, dtype=ata.dtype)
        atb = at_w @ b

        beta = torch.linalg.solve(ata, atb)
        residual = a @ beta - b
        loss = (weights * residual.pow(2)).sum() / (weights.sum() + eps)
        losses.append(loss)

    return torch.stack(losses).mean()


def _compute_ring_energy(
    energy_2d: torch.Tensor,
    radial_bins: int,
    eps: float,
) -> torch.Tensor:
    h, w = energy_2d.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=energy_2d.device),
        torch.linspace(-1.0, 1.0, w, device=energy_2d.device),
        indexing="ij",
    )
    radius = torch.sqrt(xx**2 + yy**2)
    max_r = radius.max().clamp(min=eps)
    bins = torch.linspace(0.0, max_r, radial_bins + 1, device=energy_2d.device)
    flat_radius = radius.flatten()
    flat_energy = energy_2d.flatten()

    ring_energy = torch.zeros(
        radial_bins, device=energy_2d.device, dtype=energy_2d.dtype
    )
    for idx in range(radial_bins):
        mask = (flat_radius >= bins[idx]) & (flat_radius < bins[idx + 1])
        if mask.any():
            ring_energy[idx] = flat_energy[mask].sum()
    return ring_energy


def _compute_rotation_loss_proxy(
    energy_2d: torch.Tensor,
    radial_bins: int,
    eps: float,
) -> torch.Tensor:
    # Proxy: encourage annular concentration via ring entropy.
    ring_energy = _compute_ring_energy(energy_2d, radial_bins, eps)
    total = ring_energy.sum()
    if total <= eps:
        return torch.tensor(0.0, device=energy_2d.device, dtype=energy_2d.dtype)
    probs = (ring_energy / (total + eps)).clamp_min(eps)
    entropy = -(probs * probs.log()).sum()
    norm_entropy = entropy / torch.log(torch.tensor(float(radial_bins), device=entropy.device))
    return 1.0 - norm_entropy


def _compute_scaling_loss(
    ring_energies: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # ring_energies: [T, R]
    t_frames, radial_bins = ring_energies.shape
    if t_frames < 3:
        return torch.tensor(0.5, device=ring_energies.device, dtype=ring_energies.dtype)

    energy_sum = ring_energies.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = ring_energies / energy_sum
    radii = torch.linspace(
        0.0, 1.0, radial_bins, device=ring_energies.device, dtype=ring_energies.dtype
    )
    rho_c = (probs * radii).sum(dim=1)
    t_idx = torch.linspace(
        0.0, 1.0, t_frames, device=ring_energies.device, dtype=ring_energies.dtype
    )
    cov = torch.mean((rho_c - rho_c.mean()) * (t_idx - t_idx.mean()))
    std = (rho_c.std() * t_idx.std()).clamp_min(eps)
    s_trend = (cov / std).abs().clamp(0.0, 1.0)

    grad_rho = ring_energies[1:, :] - ring_energies[:-1, :]
    grad_t = ring_energies[:, 1:] - ring_energies[:, :-1]
    grad_rho = grad_rho[:, :-1]
    grad_t = grad_t[1:, :]
    denom = (grad_rho.pow(2).sum() * grad_t.pow(2).sum()).clamp_min(eps).sqrt()
    c_flow = (grad_rho * grad_t).sum().abs() / denom
    c_flow = c_flow.clamp(0.0, 1.0)

    return 1.0 - 0.5 * (c_flow + s_trend)


def compute_physics_guided_motion_loss(
    x0_pred: torch.Tensor,
    *,
    lowpass_ratio: float,
    ridge_lambda: float,
    tau: float,
    components: Optional[Iterable[str]] = None,
    radial_bins: int = 32,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if x0_pred.dim() != 5:
        raise ValueError("physics-guided motion loss expects video tensor (B, C, F, H, W)")

    comps = _normalize_components(components)
    use_translation = "translation" in comps
    use_rotation = "rotation" in comps
    use_scaling = "scaling" in comps

    if not (use_translation or use_rotation or use_scaling):
        raise ValueError("physics-guided motion loss requires at least one component")

    x = x0_pred.float()
    batch_size, _, frames, height, width = x.shape

    v = torch.fft.fftn(x, dim=(2, 3, 4))
    energy = v.abs().pow(2).mean(dim=1)

    freq_t = torch.fft.fftfreq(frames, device=x.device).view(frames, 1, 1)
    freq_x = torch.fft.fftfreq(height, device=x.device).view(1, height, 1)
    freq_y = torch.fft.fftfreq(width, device=x.device).view(1, 1, width)
    lowpass_mask = _build_lowpass_mask(freq_t, freq_x, freq_y, lowpass_ratio)
    lowpass_energy = energy * lowpass_mask

    losses: Dict[str, torch.Tensor] = {}
    if use_translation:
        freq_t_full = freq_t.expand(frames, height, width)
        freq_x_full = freq_x.expand(frames, height, width)
        freq_y_full = freq_y.expand(frames, height, width)
        losses["translation"] = _compute_translation_loss(
            lowpass_energy,
            freq_t_full,
            freq_x_full,
            freq_y_full,
            ridge_lambda,
            eps,
        )

    if use_rotation or use_scaling:
        # Use spatial FFT per frame for radial energy statistics.
        v2d = torch.fft.fftn(x, dim=(3, 4))
        energy_2d = v2d.abs().pow(2).mean(dim=1)
        energy_2d = torch.fft.fftshift(energy_2d, dim=(-2, -1))

        rotation_vals = []
        scaling_vals = []
        for b in range(batch_size):
            ring_series = []
            for t in range(frames):
                ring_energy = _compute_ring_energy(energy_2d[b, t], radial_bins, eps)
                ring_series.append(ring_energy)
                if use_rotation:
                    rotation_vals.append(
                        _compute_rotation_loss_proxy(energy_2d[b, t], radial_bins, eps)
                    )
            ring_series_t = torch.stack(ring_series, dim=0)
            if use_scaling:
                scaling_vals.append(_compute_scaling_loss(ring_series_t, eps))

        if use_rotation:
            losses["rotation"] = torch.stack(rotation_vals).mean()
        if use_scaling:
            losses["scaling"] = torch.stack(scaling_vals).mean()

    loss_stack = torch.stack(list(losses.values()))
    weights = torch.softmax(-loss_stack / max(tau, eps), dim=0)
    total = (weights * loss_stack).sum()

    output = {
        "total": total,
        "weights": weights.detach(),
    }
    for idx, key in enumerate(losses.keys()):
        output[key] = losses[key]
        output[f"{key}_weight"] = weights[idx].detach()
    return output
