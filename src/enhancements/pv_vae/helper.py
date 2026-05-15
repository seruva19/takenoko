"""Predictive reconstruction helpers for video VAE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PVVAEContext:
    """Metadata for one predictive VAE reconstruction pass."""

    total_frames: int
    observed_frames: int
    total_latent_groups: int
    observed_latent_groups: int
    dropped_latent_groups: int
    drop_ratio: float
    padding_mode: str
    temporal_diff_weight: float
    temporal_diff_loss: str

    @property
    def active(self) -> bool:
        return self.dropped_latent_groups > 0

    def to_log_dict(self, prefix: str = "vae/pv") -> Dict[str, float]:
        return {
            f"{prefix}/active": 1.0 if self.active else 0.0,
            f"{prefix}/observed_frames": float(self.observed_frames),
            f"{prefix}/total_frames": float(self.total_frames),
            f"{prefix}/observed_latent_groups": float(self.observed_latent_groups),
            f"{prefix}/dropped_latent_groups": float(self.dropped_latent_groups),
            f"{prefix}/drop_ratio": float(self.drop_ratio),
        }


def _latent_group_count(total_frames: int, temporal_compression: int) -> int:
    if total_frames <= 1:
        return 1
    return 1 + ((total_frames - 1) // max(temporal_compression, 1))


def _frames_for_groups(groups: int, total_frames: int, temporal_compression: int) -> int:
    if groups <= 1:
        return min(total_frames, 1)
    return min(total_frames, 1 + (groups - 1) * max(temporal_compression, 1))


def prepare_pv_vae_observed_video(
    args: Any,
    images: torch.Tensor,
    *,
    training: bool,
) -> Tuple[torch.Tensor, Optional[PVVAEContext]]:
    """Return the observed-frame prefix and predictive reconstruction context."""

    if not bool(getattr(args, "enable_pv_vae", False)):
        return images, None
    if not training and not bool(getattr(args, "pv_vae_apply_in_validation", False)):
        return images, None
    if not isinstance(images, torch.Tensor) or images.dim() != 5:
        return images, None

    total_frames = int(images.shape[2])
    if total_frames <= 1:
        return images, None

    temporal_compression = int(getattr(args, "pv_vae_temporal_compression", 4))
    total_groups = _latent_group_count(total_frames, temporal_compression)
    min_observed_groups = max(1, int(getattr(args, "pv_vae_min_observed_groups", 1)))
    if total_groups <= min_observed_groups:
        return images, None

    max_drop_ratio = float(getattr(args, "pv_vae_max_drop_ratio", 1.0))
    max_droppable = total_groups - min_observed_groups
    max_drop_groups = int(max_droppable * max_drop_ratio)
    min_drop_groups = int(getattr(args, "pv_vae_min_drop_groups", 1))
    min_drop_groups = max(0, min_drop_groups)
    if max_drop_groups < min_drop_groups:
        return images, None

    if max_drop_groups == min_drop_groups:
        drop_groups = max_drop_groups
    else:
        drop_groups = int(
            torch.randint(
                min_drop_groups,
                max_drop_groups + 1,
                (),
                device=images.device,
            ).item()
        )

    observed_groups = total_groups - drop_groups
    observed_frames = _frames_for_groups(
        observed_groups,
        total_frames,
        temporal_compression,
    )
    observed = images[:, :, :observed_frames, :, :]

    context = PVVAEContext(
        total_frames=total_frames,
        observed_frames=observed_frames,
        total_latent_groups=total_groups,
        observed_latent_groups=observed_groups,
        dropped_latent_groups=drop_groups,
        drop_ratio=float(drop_groups) / float(max(total_groups, 1)),
        padding_mode=str(getattr(args, "pv_vae_padding_mode", "gaussian")).lower(),
        temporal_diff_weight=float(getattr(args, "pv_vae_temporal_diff_weight", 0.0)),
        temporal_diff_loss=str(getattr(args, "pv_vae_temporal_diff_loss", "l1")).lower(),
    )
    return observed, context


def pad_pv_vae_latents(
    latent: torch.Tensor,
    context: Optional[PVVAEContext],
) -> torch.Tensor:
    """Pad observed latents back to the full temporal latent length."""

    if context is None or not context.active:
        return latent
    if not isinstance(latent, torch.Tensor) or latent.dim() < 3:
        return latent

    current_groups = int(latent.shape[2])
    missing_groups = int(context.total_latent_groups - current_groups)
    if missing_groups <= 0:
        return latent

    pad_shape = list(latent.shape)
    pad_shape[2] = missing_groups
    if context.padding_mode == "zeros":
        padding = latent.new_zeros(pad_shape)
    else:
        padding = torch.randn(pad_shape, device=latent.device, dtype=latent.dtype)
    return torch.cat([latent, padding], dim=2)


def compute_pv_vae_temporal_difference_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    context: Optional[PVVAEContext],
) -> Optional[torch.Tensor]:
    """Compute the motion-aware temporal-difference loss from PV-VAE."""

    if context is None or context.temporal_diff_weight <= 0.0:
        return None
    if reconstructed.dim() != 5 or original.dim() != 5:
        return None
    common_frames = min(int(reconstructed.shape[2]), int(original.shape[2]))
    if common_frames <= 1:
        return None

    rec = reconstructed[:, :, :common_frames, :, :].to(torch.float32)
    tgt = original[:, :, :common_frames, :, :].to(torch.float32)
    rec_delta = rec[:, :, 1:, :, :] - rec[:, :, :-1, :, :]
    tgt_delta = tgt[:, :, 1:, :, :] - tgt[:, :, :-1, :, :]

    if context.temporal_diff_loss == "mse":
        return F.mse_loss(rec_delta, tgt_delta, reduction="mean")
    if context.temporal_diff_loss == "huber":
        return F.smooth_l1_loss(rec_delta, tgt_delta, reduction="mean")
    return F.l1_loss(rec_delta, tgt_delta, reduction="mean")
