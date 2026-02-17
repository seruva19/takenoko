"""FlexAM-style training conditioning helpers.

This module provides a train-time, config-gated conditioning path that adapts
FlexAM-like control signals to Takenoko's existing control-concatenation flow.
It is intentionally inference-neutral and disabled by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FlexAMConditioningResult:
    """Prepared conditioning tensors for a single training forward call."""

    control_latents: torch.Tensor
    model_timesteps: torch.Tensor
    density: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)


def is_flexam_training_enabled(args: Any) -> bool:
    """Return whether FlexAM train-time conditioning is enabled."""
    return bool(getattr(args, "enable_flexam_training", False))


def _align_channels(tensor: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Pad/truncate channel dimension to match latent channel count."""
    if tensor.shape[1] == target_channels:
        return tensor
    if tensor.shape[1] < target_channels:
        pad_ch = target_channels - tensor.shape[1]
        pad = torch.zeros(
            (tensor.shape[0], pad_ch, tensor.shape[2], tensor.shape[3], tensor.shape[4]),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)
    return tensor[:, :target_channels, :, :, :]


def _to_bcfhw(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize tensors into B,C,F,H,W and resize to latent resolution."""
    t = tensor.to(device=device, dtype=dtype)

    if t.dim() == 4:
        # (B,F,H,W) -> (B,1,F,H,W)
        if t.shape[0] == batch_size and t.shape[1] == frames:
            t = t.unsqueeze(1)
        # (B,C,H,W) -> (B,C,1,H,W)
        elif t.shape[0] == batch_size:
            t = t.unsqueeze(2)
        else:
            raise ValueError(f"Unsupported 4D tensor shape for FlexAM conditioning: {tuple(t.shape)}")
    elif t.dim() != 5:
        raise ValueError(f"Unsupported tensor rank for FlexAM conditioning: {tuple(t.shape)}")

    if t.shape[0] != batch_size:
        raise ValueError(
            f"FlexAM conditioning batch mismatch: expected {batch_size}, got {t.shape[0]}"
        )

    if (t.shape[2], t.shape[3], t.shape[4]) != (frames, height, width):
        t = F.interpolate(
            t.float(),
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        ).to(dtype=dtype)

    return t


def _extract_reference(
    *,
    args: Any,
    latents: torch.Tensor,
    noisy_model_input: torch.Tensor,
    control_latents: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Build a full-length reference latent from the configured source."""
    source = str(getattr(args, "flexam_reference_source", "latents_first_frame")).lower()
    if source == "control_first_frame" and control_latents is not None:
        ref = control_latents[:, :, :1, :, :]
    elif source == "noisy_first_frame":
        ref = noisy_model_input[:, :, :1, :, :]
    else:
        ref = latents[:, :, :1, :, :]

    if ref is None:
        return None

    dropout_p = float(getattr(args, "flexam_reference_dropout_p", 0.0) or 0.0)
    if dropout_p > 0.0:
        keep = (
            torch.rand((ref.shape[0], 1, 1, 1, 1), device=ref.device, dtype=torch.float32)
            >= dropout_p
        ).to(dtype=ref.dtype)
        ref = ref * keep

    # Expand to all frames so channels stay aligned for control concatenation.
    ref = ref.expand(-1, -1, noisy_model_input.shape[2], -1, -1)
    return ref


def _compute_density_scaled_timesteps(
    *,
    args: Any,
    batch: Dict[str, torch.Tensor],
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally scale model-side timesteps using per-sample density."""
    key = str(getattr(args, "flexam_density_batch_key", "density") or "density")
    default_density = float(getattr(args, "flexam_density_default", 1.0) or 1.0)

    density_value: torch.Tensor
    if key in batch:
        raw = batch[key]
        if isinstance(raw, torch.Tensor):
            density_value = raw
        else:
            density_value = torch.as_tensor(raw)
    else:
        density_value = torch.full(
            (timesteps.shape[0],),
            default_density,
            device=timesteps.device,
            dtype=torch.float32,
        )

    density_value = density_value.to(device=timesteps.device, dtype=torch.float32).reshape(-1)
    if density_value.shape[0] != timesteps.shape[0]:
        density_value = density_value[:1].expand(timesteps.shape[0])

    density_min = float(getattr(args, "flexam_density_min", 0.25) or 0.25)
    density_max = float(getattr(args, "flexam_density_max", 4.0) or 4.0)
    density_value = density_value.clamp(min=density_min, max=density_max)

    if timesteps.dim() == 1:
        scaled = timesteps.to(torch.float32) * density_value
    else:
        expanded = density_value
        while expanded.dim() < timesteps.dim():
            expanded = expanded.unsqueeze(-1)
        scaled = timesteps.to(torch.float32) * expanded

    t_min = float(getattr(args, "min_timestep", 0))
    t_max = float(getattr(args, "max_timestep", 1000))
    scaled = scaled.clamp(min=t_min, max=t_max).to(dtype=timesteps.dtype)
    return scaled, density_value


def _compute_weight_schedule_scale(
    *,
    args: Any,
    global_step: Optional[int],
) -> float:
    """Compute global scaling factor applied to all FlexAM control terms."""
    mode = str(getattr(args, "flexam_weight_schedule", "constant") or "constant").lower()
    if mode == "constant":
        return 1.0

    start_step = int(getattr(args, "flexam_schedule_start_step", 0) or 0)
    end_step = int(getattr(args, "flexam_schedule_end_step", 0) or 0)
    min_scale = float(getattr(args, "flexam_schedule_min_scale", 0.0) or 0.0)
    max_scale = float(getattr(args, "flexam_schedule_max_scale", 1.0) or 1.0)

    if global_step is None or end_step <= start_step:
        return max_scale
    if global_step <= start_step:
        return min_scale
    if global_step >= end_step:
        return max_scale

    progress = float(global_step - start_step) / float(end_step - start_step)
    progress = max(0.0, min(1.0, progress))
    if mode == "cosine_ramp":
        progress = 0.5 - 0.5 * math.cos(progress * math.pi)
    return min_scale + (max_scale - min_scale) * progress


def _resolve_additional_control_keys(args: Any) -> List[str]:
    """Resolve single-key legacy config + list config into ordered unique keys."""
    keys: List[str] = []
    legacy_key = str(getattr(args, "flexam_additional_control_key", "") or "").strip()
    if legacy_key:
        keys.append(legacy_key)

    raw_keys = getattr(args, "flexam_additional_control_keys", [])
    if isinstance(raw_keys, str):
        raw_iter = [part.strip() for part in raw_keys.split(",")]
    elif isinstance(raw_keys, list):
        raw_iter = [str(v).strip() for v in raw_keys]
    else:
        raw_iter = []
    for key in raw_iter:
        if key and key not in keys:
            keys.append(key)
    return keys


def prepare_flexam_conditioning(
    *,
    args: Any,
    batch: Dict[str, torch.Tensor],
    latents: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    control_latents: Optional[torch.Tensor],
    network_dtype: torch.dtype,
    global_step: Optional[int] = None,
) -> FlexAMConditioningResult:
    """Prepare FlexAM-style control latents and optional model-side timesteps.

    The output `control_latents` always matches `latents` shape.
    """
    base = torch.zeros_like(latents, dtype=network_dtype, device=noisy_model_input.device)
    schedule_scale = _compute_weight_schedule_scale(args=args, global_step=global_step)
    metrics: Dict[str, float] = {
        "flexam/enabled": 1.0,
        "flexam/schedule_scale": float(schedule_scale),
    }

    control_weight = float(getattr(args, "flexam_control_weight", 1.0) or 1.0) * schedule_scale
    metrics["flexam/control_weight_effective"] = float(control_weight)
    if control_latents is not None:
        base = base + control_latents.to(device=base.device, dtype=base.dtype) * control_weight
        metrics["flexam/control_signal_used"] = 1.0
    else:
        metrics["flexam/control_signal_used"] = 0.0

    # Optional inpaint-style conditioning from mask_signal.
    mask_weight = float(getattr(args, "flexam_mask_weight", 0.0) or 0.0) * schedule_scale
    metrics["flexam/mask_weight_effective"] = float(mask_weight)
    if mask_weight > 0.0 and "mask_signal" in batch and batch["mask_signal"] is not None:
        try:
            mask = _to_bcfhw(
                batch["mask_signal"],
                batch_size=latents.shape[0],
                frames=latents.shape[2],
                height=latents.shape[3],
                width=latents.shape[4],
                device=base.device,
                dtype=base.dtype,
            )
            if mask.min() < 0.0 or mask.max() > 1.0:
                mask = (mask + 1.0) / 2.0
            mask = mask.clamp(0.0, 1.0)
            if mask.shape[1] != 1:
                mask = mask[:, :1, :, :, :]
            masked = noisy_model_input.to(device=base.device, dtype=base.dtype) * (1.0 - mask) + (
                -1.0 * mask
            )
            masked = _align_channels(masked, base.shape[1])
            base = base + masked * mask_weight
            metrics["flexam/mask_signal_used"] = 1.0
            metrics["flexam/mask_mean"] = float(mask.detach().mean().item())
        except Exception as exc:
            logger.warning("FlexAM mask conditioning skipped: %s", exc)
            metrics["flexam/mask_signal_used"] = 0.0
    else:
        metrics["flexam/mask_signal_used"] = 0.0

    # Optional full-reference style conditioning mapped to channel-concat path.
    ref_weight = float(getattr(args, "flexam_reference_weight", 0.0) or 0.0) * schedule_scale
    metrics["flexam/reference_weight_effective"] = float(ref_weight)
    if ref_weight > 0.0:
        ref = _extract_reference(
            args=args,
            latents=latents,
            noisy_model_input=noisy_model_input,
            control_latents=control_latents,
        )
        if ref is not None:
            ref = _align_channels(ref.to(device=base.device, dtype=base.dtype), base.shape[1])
            base = base + ref * ref_weight
            metrics["flexam/reference_used"] = 1.0
        else:
            metrics["flexam/reference_used"] = 0.0
    else:
        metrics["flexam/reference_used"] = 0.0

    # Optional extra control tensors from batch (e.g., depth/camera/proxy maps).
    extra_weight = float(getattr(args, "flexam_additional_control_weight", 0.0) or 0.0)
    extra_weight *= schedule_scale
    additional_keys = _resolve_additional_control_keys(args)
    reduce_mode = str(getattr(args, "flexam_additional_control_reduce", "sum") or "sum").lower()
    metrics["flexam/additional_weight_effective"] = float(extra_weight)
    metrics["flexam/additional_key_count"] = float(len(additional_keys))
    used_extra_count = 0
    if extra_weight > 0.0 and additional_keys:
        extras = []
        for extra_key in additional_keys:
            if extra_key not in batch or batch[extra_key] is None:
                continue
            try:
                extra = _to_bcfhw(
                    batch[extra_key],
                    batch_size=latents.shape[0],
                    frames=latents.shape[2],
                    height=latents.shape[3],
                    width=latents.shape[4],
                    device=base.device,
                    dtype=base.dtype,
                )
                extras.append(_align_channels(extra, base.shape[1]))
                used_extra_count += 1
            except Exception as exc:
                logger.warning("FlexAM additional control '%s' skipped: %s", extra_key, exc)

        if extras:
            if reduce_mode == "mean":
                combined_extra = torch.stack(extras, dim=0).mean(dim=0)
            else:
                combined_extra = extras[0]
                for extra in extras[1:]:
                    combined_extra = combined_extra + extra
            base = base + combined_extra * extra_weight
    metrics["flexam/additional_keys_used"] = float(used_extra_count)
    metrics["flexam/additional_used"] = 1.0 if used_extra_count > 0 else 0.0

    model_timesteps = timesteps
    density_value = None
    if bool(getattr(args, "flexam_use_density_timestep_scaling", False)):
        model_timesteps, density_value = _compute_density_scaled_timesteps(
            args=args,
            batch=batch,
            timesteps=timesteps,
        )
        if density_value is not None:
            metrics["flexam/density_mean"] = float(density_value.detach().mean().item())
            metrics["flexam/density_std"] = float(
                density_value.detach().float().std(unbiased=False).item()
            )
        step_delta = (
            model_timesteps.to(torch.float32) - timesteps.to(torch.float32)
        ).detach().abs().mean()
        metrics["flexam/timestep_delta_abs_mean"] = float(step_delta.item())

    return FlexAMConditioningResult(
        control_latents=base,
        model_timesteps=model_timesteps,
        density=density_value,
        metrics=metrics,
    )
