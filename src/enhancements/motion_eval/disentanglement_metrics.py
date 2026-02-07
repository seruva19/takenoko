"""Lightweight motion-disentanglement proxy metrics for validation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _to_unit_interval(frames: torch.Tensor) -> torch.Tensor:
    x = frames.to(torch.float32)
    if x.numel() == 0:
        return x
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_max > 1.5:
        x = x / 255.0
    elif x_min < -0.05:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _compute_basic_stats(frames: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    # frames: [B, C, F, H, W], assumed F >= 2
    motion_diffs = (frames[:, :, 1:] - frames[:, :, :-1]).abs()
    motion_intensity = motion_diffs.mean(dim=(1, 2, 3, 4))
    # Per-step motion energy: [B, F-1]
    step_energy = motion_diffs.mean(dim=(1, 3, 4))
    appearance_first = frames[:, :, 0].mean(dim=(2, 3))
    appearance_last = frames[:, :, -1].mean(dim=(2, 3))
    appearance_drift = (appearance_last - appearance_first).abs().mean(dim=1)
    motion_to_appearance_ratio = motion_intensity / (appearance_drift + eps)
    return {
        "motion_intensity": motion_intensity,
        "appearance_drift": appearance_drift,
        "motion_to_appearance_ratio": motion_to_appearance_ratio,
        "motion_temporal_std": step_energy.std(dim=1, unbiased=False),
    }


def compute_motion_disentanglement_proxies(
    pred_frames: torch.Tensor,
    ref_frames: Optional[torch.Tensor] = None,
    *,
    max_items: int = 2,
    frame_stride: int = 2,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Return scalar proxy diagnostics for motion/appearance disentanglement.

    Inputs are expected as ``[B, C, F, H, W]`` and can be in ``[-1,1]``,
    ``[0,1]``, or ``[0,255]``. Outputs are 0-D tensors.
    """

    if pred_frames.dim() != 5:
        return {}
    if frame_stride < 1:
        frame_stride = 1

    pred = _to_unit_interval(pred_frames)
    b = min(int(pred.shape[0]), int(max_items))
    if b <= 0:
        return {}
    pred = pred[:b, :, ::frame_stride]
    if pred.shape[2] < 2:
        return {}

    pred_stats = _compute_basic_stats(pred, eps=eps)
    metrics: Dict[str, torch.Tensor] = {
        "motion_intensity_pred": pred_stats["motion_intensity"].mean(),
        "appearance_drift_pred": pred_stats["appearance_drift"].mean(),
        "motion_to_appearance_ratio_pred": pred_stats[
            "motion_to_appearance_ratio"
        ].mean(),
        "motion_temporal_std_pred": pred_stats["motion_temporal_std"].mean(),
    }

    if ref_frames is None or ref_frames.dim() != 5:
        return metrics

    ref = _to_unit_interval(ref_frames).to(device=pred.device, dtype=pred.dtype)
    b_ref = min(int(ref.shape[0]), b)
    pred = pred[:b_ref]
    ref = ref[:b_ref, :, ::frame_stride]
    if ref.shape[2] < 2:
        return metrics

    f_common = min(int(pred.shape[2]), int(ref.shape[2]))
    pred = pred[:, :, :f_common]
    ref = ref[:, :, :f_common]
    ref_stats = _compute_basic_stats(ref, eps=eps)

    motion_alignment = 1.0 - (
        (pred_stats["motion_intensity"][:b_ref] - ref_stats["motion_intensity"]).abs()
        / (ref_stats["motion_intensity"].abs() + eps)
    )
    appearance_alignment = 1.0 - (
        (pred_stats["appearance_drift"][:b_ref] - ref_stats["appearance_drift"]).abs()
        / (ref_stats["appearance_drift"].abs() + eps)
    )

    pred_first = pred[:, :, 0].flatten(1)
    pred_last = pred[:, :, -1].flatten(1)
    ref_first = ref[:, :, 0].flatten(1)
    identity_cosine_first = F.cosine_similarity(pred_first, ref_first, dim=1)
    identity_cosine_last = F.cosine_similarity(pred_last, ref_first, dim=1)

    ratio_log_diff = (
        (
            torch.log(pred_stats["motion_to_appearance_ratio"][:b_ref] + eps)
            - torch.log(ref_stats["motion_to_appearance_ratio"] + eps)
        )
        .abs()
        .mean()
    )

    metrics.update(
        {
            "motion_intensity_ref": ref_stats["motion_intensity"].mean(),
            "appearance_drift_ref": ref_stats["appearance_drift"].mean(),
            "motion_to_appearance_ratio_ref": ref_stats[
                "motion_to_appearance_ratio"
            ].mean(),
            "motion_alignment_proxy": motion_alignment.clamp(0.0, 1.0).mean(),
            "appearance_alignment_proxy": appearance_alignment.clamp(0.0, 1.0).mean(),
            "identity_cosine_first_proxy": identity_cosine_first.mean(),
            "identity_cosine_last_proxy": identity_cosine_last.mean(),
            "identity_cosine_drop_proxy": (
                identity_cosine_first - identity_cosine_last
            ).mean(),
            "motion_ratio_logdiff_proxy": ratio_log_diff,
        }
    )

    return metrics
