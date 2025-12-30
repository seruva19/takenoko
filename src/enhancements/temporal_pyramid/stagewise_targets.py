"""Stage-wise target helpers for temporal pyramid training (TPDiff-inspired).

Implements stage boundary construction (downsample/upsample) and per-stage target
computation. Supports flow-style gamma/sigma (gamma=1-t, sigma=t) and optional
scheduler-derived gamma/sigma from alpha_cumprod when available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


def _lookup_alpha_cumprod(
    noise_scheduler: object, indices: torch.Tensor
) -> Optional[torch.Tensor]:
    alphas = getattr(noise_scheduler, "alphas_cumprod", None)
    if alphas is None:
        return None
    if not torch.is_tensor(alphas):
        try:
            alphas = torch.tensor(alphas)
        except Exception:
            return None
    if alphas.ndim != 1:
        return None
    indices = indices.clamp(0, alphas.shape[0] - 1)
    return alphas.to(device=indices.device)[indices]


def _downsample_temporal(latents: torch.Tensor, stride: int) -> torch.Tensor:
    if latents.ndim != 5 or stride <= 1:
        return latents
    frame_count = latents.shape[2]
    if frame_count <= stride:
        return latents
    indices = torch.arange(0, frame_count, stride, device=latents.device)
    return latents.index_select(2, indices)


def _upsample_temporal(latents: torch.Tensor, target_frames: int) -> torch.Tensor:
    if latents.ndim != 5:
        return latents
    if latents.shape[2] == target_frames:
        return latents
    return F.interpolate(
        latents,
        size=(target_frames, latents.shape[3], latents.shape[4]),
        mode="trilinear",
        align_corners=False,
    )


def _resample_by_stride(
    latents: torch.Tensor, stride: torch.Tensor
) -> torch.Tensor:
    if latents.ndim != 5:
        return latents
    unique_strides = torch.unique(stride)
    if unique_strides.numel() == 1 and int(unique_strides.item()) <= 1:
        return latents
    output = latents.clone()
    target_frames = latents.shape[2]
    for stride_value in unique_strides.tolist():
        stride_value = int(stride_value)
        if stride_value <= 1:
            continue
        idx = (stride == stride_value).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        subset = latents.index_select(0, idx)
        down = _downsample_temporal(subset, stride_value)
        up = _upsample_temporal(down, target_frames)
        output.index_copy_(0, idx, up)
    return output


class TemporalPyramidStagewiseTargetHelper:
    """Compute stage-wise targets using temporal pyramid resampling."""

    def __init__(self, args) -> None:
        self._enabled = bool(
            getattr(args, "enable_temporal_pyramid_stagewise_target", False)
        )
        self._args = args
        self._warned_missing_scheduler = False
        self._warned_non_1d_timesteps = False
        self._warned_training_disabled = False
        self._warned_non_video_latents = False
        self._warned_scheduler_mapping = False

    def setup_hooks(self) -> None:
        """No-op hook to align with enhancement interface."""

    def remove_hooks(self) -> None:
        """No-op hook to align with enhancement interface."""

    def compute_target(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: Optional[object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return stage-wise training targets for temporal pyramid training."""
        if not self._enabled:
            return noise - latents.to(device=device, dtype=dtype)

        if not getattr(self._args, "enable_temporal_pyramid_training", False):
            if not self._warned_training_disabled:
                logger.warning(
                    "Temporal pyramid stagewise targets require temporal pyramid training; "
                    "falling back to base target."
                )
                self._warned_training_disabled = True
            return noise - latents.to(device=device, dtype=dtype)

        if noise_scheduler is None:
            if not self._warned_missing_scheduler:
                logger.warning(
                    "Temporal pyramid stagewise target requested without noise scheduler; "
                    "falling back to base target."
                )
                self._warned_missing_scheduler = True
            return noise - latents.to(device=device, dtype=dtype)

        if timesteps.ndim != 1:
            if not self._warned_non_1d_timesteps:
                logger.warning(
                    "Temporal pyramid stagewise targets require 1D timesteps; "
                    "falling back to base target."
                )
                self._warned_non_1d_timesteps = True
            return noise - latents.to(device=device, dtype=dtype)

        if latents.ndim != 5:
            if not self._warned_non_video_latents:
                logger.warning(
                    "Temporal pyramid stagewise targets require 5D latents; "
                    "falling back to base target."
                )
                self._warned_non_video_latents = True
            return noise - latents.to(device=device, dtype=dtype)

        num_steps = float(
            getattr(
                getattr(noise_scheduler, "config", object()),
                "num_train_timesteps",
                1000,
            )
        )
        t_norm = (timesteps.float() - 1.0) / max(num_steps, 1.0)
        t_norm = t_norm.clamp(0.0, 1.0 - 1e-6)

        num_stages = int(getattr(self._args, "temporal_pyramid_num_stages", 3))
        num_stages = max(num_stages, 1)
        stride_base = int(getattr(self._args, "temporal_pyramid_stride_base", 2))
        stride_base = max(stride_base, 1)
        max_stride = getattr(self._args, "temporal_pyramid_max_stride", None)
        if max_stride is not None:
            max_stride = max(int(max_stride), 1)

        stage = torch.clamp((t_norm * num_stages).long(), 0, num_stages - 1)
        stride = stride_base ** stage
        stride_next = stride * stride_base
        stride_next = torch.where(
            stage == (num_stages - 1),
            stride,
            stride_next,
        )
        if max_stride is not None:
            stride = torch.clamp(stride, max=max_stride)
            stride_next = torch.clamp(stride_next, max=max_stride)

        x_sk_base = _resample_by_stride(latents, stride_next)
        x_ek_base = _resample_by_stride(latents, stride)

        boundaries = getattr(self._args, "temporal_pyramid_stage_boundaries", None)
        if boundaries is not None and len(boundaries) == num_stages + 1:
            boundaries_t = torch.tensor(
                boundaries, device=t_norm.device, dtype=t_norm.dtype
            )
            t_start = boundaries_t[stage]
            t_end = boundaries_t[stage + 1]
        else:
            t_start = stage.to(dtype=t_norm.dtype) / float(num_stages)
            t_end = (stage.to(dtype=t_norm.dtype) + 1.0) / float(num_stages)

        gamma_sigma_mode = getattr(self._args, "temporal_pyramid_gamma_sigma_mode", "flow")
        if gamma_sigma_mode == "scheduler":
            idx_start = (t_start * (num_steps - 1)).round().long()
            idx_end = (t_end * (num_steps - 1)).round().long()
            alpha_start = _lookup_alpha_cumprod(noise_scheduler, idx_start)
            alpha_end = _lookup_alpha_cumprod(noise_scheduler, idx_end)
            if alpha_start is None or alpha_end is None:
                if not self._warned_scheduler_mapping:
                    logger.warning(
                        "Temporal pyramid scheduler gamma/sigma mapping unavailable; "
                        "falling back to flow parameterization."
                    )
                    self._warned_scheduler_mapping = True
                gamma_sigma_mode = "flow"
            else:
                gamma_sk = torch.sqrt(alpha_start)
                sigma_sk = torch.sqrt(1.0 - alpha_start)
                gamma_ek = torch.sqrt(alpha_end)
                sigma_ek = torch.sqrt(1.0 - alpha_end)

        if gamma_sigma_mode == "flow":
            gamma_sk = 1.0 - t_start
            sigma_sk = t_start
            gamma_ek = 1.0 - t_end
            sigma_ek = t_end

        gamma_sk = gamma_sk.view(-1, 1, 1, 1, 1)
        sigma_sk = sigma_sk.view(-1, 1, 1, 1, 1)
        gamma_ek = gamma_ek.view(-1, 1, 1, 1, 1)
        sigma_ek = sigma_ek.view(-1, 1, 1, 1, 1)

        x_ek = gamma_ek * x_ek_base + sigma_ek * noise
        sigma_ek_safe = torch.clamp(sigma_ek, min=1e-6)
        eps_k = (x_ek - gamma_ek * x_ek_base) / sigma_ek_safe

        target = eps_k - x_sk_base
        return target.to(device=device, dtype=dtype)


def create_temporal_pyramid_stagewise_target_helper(
    args,
) -> Optional[TemporalPyramidStagewiseTargetHelper]:
    """Create stagewise target helper only when enabled."""
    if getattr(args, "enable_temporal_pyramid_stagewise_target", False):
        helper = TemporalPyramidStagewiseTargetHelper(args)
        helper.setup_hooks()
        return helper
    return None
