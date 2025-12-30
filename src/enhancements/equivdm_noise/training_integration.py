"""
EquiVDM consistent noise integration for video LoRA training.

This helper keeps the standard denoising loss unchanged while swapping
the noise sampling strategy to temporally consistent (warped) noise.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def _permute_to_frame_dim(
    tensor: torch.Tensor, frame_dim: int
) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
    if tensor.dim() <= frame_dim or frame_dim == 2:
        return tensor, None
    perm = list(range(tensor.dim()))
    perm[2], perm[frame_dim] = perm[frame_dim], perm[2]
    return tensor.permute(*perm), tuple(perm)


def _permute_from_frame_dim(
    tensor: torch.Tensor, perm: Optional[Tuple[int, ...]]
) -> torch.Tensor:
    if perm is None:
        return tensor
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return tensor.permute(*inverse)


class EquiVDMConsistentNoiseHelper:
    """Generate temporally consistent noise using precomputed motion flow."""

    def __init__(
        self,
        *,
        enabled: bool,
        beta: float,
        flow_source: str,
        flow_key: str,
        frame_dim_in_batch: int,
        flow_resize_mode: str,
        flow_resize_align_corners: bool,
        flow_spatial_stride: int,
        warp_mode: str,
        warp_jitter: float,
        nte_samples: int,
        flow_quality_check: bool,
        flow_max_magnitude: float,
        flow_stride_check: bool,
        flow_expected_stride: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.beta = float(beta)
        self.flow_source = str(flow_source)
        self.flow_key = str(flow_key)
        self.frame_dim_in_batch = int(frame_dim_in_batch)
        self.flow_resize_mode = str(flow_resize_mode)
        self.flow_resize_align_corners = bool(flow_resize_align_corners)
        self.flow_spatial_stride = int(flow_spatial_stride)
        self.warp_mode = str(warp_mode)
        self.warp_jitter = float(warp_jitter)
        self.nte_samples = int(nte_samples)
        self.flow_quality_check = bool(flow_quality_check)
        self.flow_max_magnitude = float(flow_max_magnitude)
        self.flow_stride_check = bool(flow_stride_check)
        self.flow_expected_stride = int(flow_expected_stride)

        self._warned_missing_flow = False
        self._warned_shape_mismatch = False
        self._warned_quality = False
        self._warned_stride_mismatch = False

        if not self.enabled:
            return
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"equivdm_noise_beta must be in [0, 1], got {beta}")
        if self.flow_source not in {"batch"}:
            raise ValueError(
                "equivdm_flow_source must be 'batch' for EquiVDM consistent noise."
            )
        if self.frame_dim_in_batch < 2:
            raise ValueError(
                "equivdm_flow_frame_dim_in_batch must be >= 2 for EquiVDM noise."
            )
        if self.flow_resize_mode not in {"none", "bilinear", "nearest"}:
            raise ValueError(
                "equivdm_flow_resize_mode must be one of 'none', 'bilinear', 'nearest'."
            )
        if self.flow_spatial_stride == 0:
            raise ValueError("equivdm_flow_spatial_stride must be != 0.")
        if self.warp_mode not in {"grid", "stochastic", "nte"}:
            raise ValueError(
                "equivdm_warp_mode must be one of 'grid', 'stochastic', 'nte'."
            )
        if self.warp_jitter < 0:
            raise ValueError("equivdm_warp_jitter must be >= 0.")
        if self.nte_samples < 1:
            raise ValueError("equivdm_nte_samples must be >= 1.")
        if self.flow_max_magnitude < 0:
            raise ValueError(
                "equivdm_flow_max_magnitude must be >= 0 (0 enables auto threshold)."
            )
        if self.flow_expected_stride == 0:
            raise ValueError("equivdm_flow_expected_stride must be != 0.")
        logger.info(
            "EquiVDM consistent noise enabled: beta=%.3f, flow_source=%s, flow_key=%s, frame_dim=%s, resize=%s, warp=%s",
            self.beta,
            self.flow_source,
            self.flow_key,
            self.frame_dim_in_batch,
            self.flow_resize_mode,
            self.warp_mode,
        )

    @classmethod
    def create_from_args(
        cls, args: Any
    ) -> Optional["EquiVDMConsistentNoiseHelper"]:
        if not bool(getattr(args, "enable_equivdm_consistent_noise", False)):
            return None
        return cls(
            enabled=True,
            beta=float(getattr(args, "equivdm_noise_beta", 0.9)),
            flow_source=str(getattr(args, "equivdm_flow_source", "batch")),
            flow_key=str(getattr(args, "equivdm_flow_key", "optical_flow")),
            frame_dim_in_batch=int(
                getattr(args, "equivdm_flow_frame_dim_in_batch", 2)
            ),
            flow_resize_mode=str(
                getattr(args, "equivdm_flow_resize_mode", "none")
            ),
            flow_resize_align_corners=bool(
                getattr(args, "equivdm_flow_resize_align_corners", True)
            ),
            flow_spatial_stride=int(
                getattr(args, "equivdm_flow_spatial_stride", -1)
            ),
            warp_mode=str(getattr(args, "equivdm_warp_mode", "grid")),
            warp_jitter=float(getattr(args, "equivdm_warp_jitter", 0.5)),
            nte_samples=int(getattr(args, "equivdm_nte_samples", 4)),
            flow_quality_check=bool(
                getattr(args, "equivdm_flow_quality_check", False)
            ),
            flow_max_magnitude=float(
                getattr(args, "equivdm_flow_max_magnitude", 0.0)
            ),
            flow_stride_check=bool(
                getattr(args, "equivdm_flow_stride_check", False)
            ),
            flow_expected_stride=int(
                getattr(args, "equivdm_flow_expected_stride", -1)
            ),
        )

    def setup_hooks(self) -> None:
        """No-op (kept for consistency with enhancement interfaces)."""

    def remove_hooks(self) -> None:
        """No-op (kept for consistency with enhancement interfaces)."""

    def sample_noise(
        self, latents: torch.Tensor, batch: Dict[str, Any]
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.randn_like(latents)

        latents_canonical, perm = _permute_to_frame_dim(
            latents, self.frame_dim_in_batch
        )
        if latents_canonical.dim() <= 2:
            return torch.randn_like(latents)

        frame_count = latents_canonical.shape[2]
        if frame_count <= 1:
            return torch.randn_like(latents)

        flow = self._extract_flow(batch, latents_canonical)
        if flow is None:
            return torch.randn_like(latents)

        consistent = self._build_consistent_noise(latents_canonical, flow)
        if self.beta < 1.0:
            independent = torch.randn_like(consistent)
            consistent = self.beta * consistent + math.sqrt(1.0 - self.beta**2) * independent

        return _permute_from_frame_dim(consistent, perm)

    def _extract_flow(
        self, batch: Dict[str, Any], latents: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.flow_source != "batch":
            return None
        if not isinstance(batch, dict):
            return None
        flow = batch.get(self.flow_key)
        if flow is None:
            if not self._warned_missing_flow:
                available_keys = ", ".join(sorted(str(k) for k in batch.keys()))
                logger.warning(
                    "EquiVDM consistent noise enabled but no flow found in batch key '%s'. "
                    "Available keys: %s",
                    self.flow_key,
                    available_keys if available_keys else "<none>",
                )
                self._warned_missing_flow = True
            return None

        flow = flow.to(device=latents.device, dtype=torch.float32)
        normalized = self._normalize_flow(flow, latents)
        if normalized is None and not self._warned_shape_mismatch:
            logger.warning(
                "EquiVDM flow tensor shape mismatch; falling back to iid noise."
            )
            self._warned_shape_mismatch = True
        if normalized is not None:
            if self.flow_quality_check:
                self._check_flow_quality(normalized, latents)
            if self.flow_stride_check:
                self._check_flow_stride(batch)
        return normalized

    def _normalize_flow(
        self, flow: torch.Tensor, latents: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if flow.dim() == 5:
            if flow.shape[2] == 2:
                flow_bf = flow
            elif flow.shape[1] == 2:
                flow_bf = flow.permute(0, 2, 1, 3, 4)
            else:
                return None
        elif flow.dim() == 4:
            if flow.shape[1] == 2:
                flow_bf = flow.unsqueeze(1)
            else:
                return None
        else:
            return None

        frame_count = latents.shape[2]
        if flow_bf.shape[1] != frame_count - 1:
            return None

        lat_h, lat_w = latents.shape[-2:]
        if flow_bf.shape[-2:] != (lat_h, lat_w):
            flow_bf = self._maybe_resize_flow(flow_bf, lat_h, lat_w)
            if flow_bf is None:
                return None

        if self.flow_spatial_stride > 0:
            stride_scale = 1.0 / float(self.flow_spatial_stride)
            flow_bf[:, :, 0].mul_(stride_scale)
            flow_bf[:, :, 1].mul_(stride_scale)

        return flow_bf

    def _maybe_resize_flow(
        self, flow_bf: torch.Tensor, lat_h: int, lat_w: int
    ) -> Optional[torch.Tensor]:
        if self.flow_resize_mode == "none":
            return None
        flow_h, flow_w = flow_bf.shape[-2:]
        if flow_h <= 0 or flow_w <= 0:
            return None
        scale_y = lat_h / float(flow_h)
        scale_x = lat_w / float(flow_w)
        flow_reshaped = flow_bf.reshape(-1, 2, flow_h, flow_w)
        resized = F.interpolate(
            flow_reshaped,
            size=(lat_h, lat_w),
            mode=self.flow_resize_mode,
            align_corners=(
                self.flow_resize_align_corners
                if self.flow_resize_mode == "bilinear"
                else None
            ),
        )
        resized = resized.reshape(flow_bf.shape[0], flow_bf.shape[1], 2, lat_h, lat_w)
        resized[:, :, 0].mul_(scale_x)
        resized[:, :, 1].mul_(scale_y)
        return resized

    def _build_consistent_noise(
        self, latents: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, frames, height, width = latents.shape
        base = torch.randn(
            (batch, channels, height, width),
            device=latents.device,
            dtype=latents.dtype,
        )
        noise_frames = [base]
        for idx in range(1, frames):
            flow_frame = flow[:, idx - 1]
            warped = self._warp_noise(noise_frames[-1], flow_frame)
            noise_frames.append(warped)
        return torch.stack(noise_frames, dim=2)

    def _check_flow_quality(
        self, flow: torch.Tensor, latents: torch.Tensor
    ) -> None:
        if self._warned_quality:
            return
        finite_ratio = torch.isfinite(flow).float().mean().item()
        if finite_ratio < 1.0:
            logger.warning(
                "EquiVDM flow contains non-finite values (finite_ratio=%.4f).",
                finite_ratio,
            )
            self._warned_quality = True
            return

        lat_h, lat_w = latents.shape[-2:]
        threshold = self.flow_max_magnitude
        if threshold <= 0.0:
            threshold = 0.5 * float(max(lat_h, lat_w))
        magnitude = torch.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        max_mag = float(magnitude.max().item())
        mean_mag = float(magnitude.mean().item())
        if max_mag > threshold:
            logger.warning(
                "EquiVDM flow magnitude high (max=%.3f, mean=%.3f, threshold=%.3f).",
                max_mag,
                mean_mag,
                threshold,
            )
            self._warned_quality = True

    def _check_flow_stride(self, batch: Dict[str, Any]) -> None:
        if self._warned_stride_mismatch:
            return
        stride_tensor = batch.get("optical_flow_stride")
        if stride_tensor is None:
            return
        try:
            stride_vals = stride_tensor.detach().to(torch.int64).view(-1)
            unique = torch.unique(stride_vals).tolist()
        except Exception:
            return
        if not unique:
            return
        expected = self.flow_expected_stride
        if expected <= 0:
            return
        if any(int(v) != expected for v in unique):
            logger.warning(
                "EquiVDM flow stride mismatch: expected=%s, batch=%s",
                expected,
                unique,
            )
            self._warned_stride_mismatch = True

    def _warp_noise(
        self, noise_frame: torch.Tensor, flow_frame: torch.Tensor
    ) -> torch.Tensor:
        _, _, height, width = noise_frame.shape
        flow_frame = flow_frame.to(dtype=noise_frame.dtype)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=flow_frame.device, dtype=noise_frame.dtype),
            torch.arange(width, device=flow_frame.device, dtype=noise_frame.dtype),
            indexing="ij",
        )
        if self.warp_mode == "nte":
            return self._warp_noise_nte(noise_frame, flow_frame, xx, yy)

        jitter_x, jitter_y = self._get_stochastic_jitter(flow_frame)
        grid_x = (xx + flow_frame[:, 0] + jitter_x) / max(width - 1, 1) * 2 - 1
        grid_y = (yy + flow_frame[:, 1] + jitter_y) / max(height - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(
            noise_frame,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _get_stochastic_jitter(
        self, flow_frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.warp_mode != "stochastic" or self.warp_jitter <= 0:
            zeros = torch.zeros_like(flow_frame[:, 0])
            return zeros, zeros
        jitter = self.warp_jitter
        jitter_x = (torch.rand_like(flow_frame[:, 0]) * 2.0 - 1.0) * jitter
        jitter_y = (torch.rand_like(flow_frame[:, 1]) * 2.0 - 1.0) * jitter
        return jitter_x, jitter_y

    def _warp_noise_nte(
        self,
        noise_frame: torch.Tensor,
        flow_frame: torch.Tensor,
        xx: torch.Tensor,
        yy: torch.Tensor,
    ) -> torch.Tensor:
        samples = max(1, self.nte_samples)
        jitter = max(0.0, self.warp_jitter)
        acc = torch.zeros_like(noise_frame)
        for _ in range(samples):
            jitter_x = (torch.rand_like(flow_frame[:, 0]) * 2.0 - 1.0) * jitter
            jitter_y = (torch.rand_like(flow_frame[:, 1]) * 2.0 - 1.0) * jitter
            height = noise_frame.shape[2]
            width = noise_frame.shape[3]
            grid_x = (xx + flow_frame[:, 0] + jitter_x) / max(width - 1, 1) * 2 - 1
            grid_y = (yy + flow_frame[:, 1] + jitter_y) / max(height - 1, 1) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1)
            acc = acc + F.grid_sample(
                noise_frame,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        return acc / math.sqrt(float(samples))
