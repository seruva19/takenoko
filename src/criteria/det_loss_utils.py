"""Helpers for DeT metric plumbing into LossComponents."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

# Keys emitted by DeTMotionTransferHelper.compute_loss() that map directly to
# LossComponents fields.
DET_LOSS_COMPONENT_FIELDS: Tuple[str, ...] = (
    "det_temporal_kernel_loss",
    "det_dense_tracking_loss",
    "det_external_tracking_loss",
    "det_external_tracking_active_samples",
    "det_external_tracking_active_ratio",
    "det_external_tracking_active_points",
    "det_high_frequency_loss",
    "det_nonlocal_fallback_loss",
    "det_nonlocal_fallback_blend_mean",
    "det_nonlocal_fallback_active_depths",
    "det_controller_sync_enabled",
    "det_controller_sync_applied",
    "det_controller_sync_world_size",
    "det_optimizer_lr_scale",
    "det_optimizer_lr_modulation_active",
    "det_locality_ratio",
    "det_locality_scale",
    "det_attention_locality_ratio",
    "det_attention_locality_scale",
    "det_attention_locality_policy_active",
    "det_unified_controller_scale",
    "det_unified_controller_locality_scale",
    "det_unified_controller_stability_scale",
    "det_unified_controller_spike_ratio",
    "det_unified_controller_cooldown_active",
    "det_per_depth_scale_mean",
    "det_per_depth_scale_min",
    "det_per_depth_scale_max",
    "det_per_depth_cooldown_active_count",
    "det_per_depth_spike_ratio_max",
    "det_schedule_temporal_factor",
    "det_schedule_tracking_factor",
    "det_schedule_external_factor",
    "det_schedule_nonlocal_factor",
    "det_schedule_hf_factor",
    "det_auto_safeguard_active",
    "det_auto_safeguard_risky_step",
    "det_auto_safeguard_bad_streak",
    "det_auto_safeguard_good_streak",
    "det_auto_safeguard_risk_locality",
    "det_auto_safeguard_risk_spike",
    "det_auto_safeguard_risk_cooldown",
    "det_auto_safeguard_local_scale_cap",
    "det_auto_safeguard_nonlocal_boost",
)


def _to_detached_tensor(value: Any, *, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach()
    return reference.detach().new_tensor(float(value))


def extract_det_component_tensors(
    det_logs: Dict[str, Any],
    *,
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert DeT helper logs into tensors consumable by LossComponents."""
    parsed: Dict[str, torch.Tensor] = {}
    for key in DET_LOSS_COMPONENT_FIELDS:
        value = det_logs.get(key)
        if value is None:
            continue
        parsed[key] = _to_detached_tensor(value, reference=reference)
    return parsed

