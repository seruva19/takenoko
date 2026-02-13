"""DeT-style train-time helper for temporal decoupling and tracking losses."""

from __future__ import annotations

import os
import re
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_KERNEL_MODES = {"avg", "gaussian"}
_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)-(\d+)$")


class DeTMotionTransferHelper(nn.Module):
    """Train-only helper inspired by DeT temporal kernel and tracking losses."""
    _RUNTIME_STATE_KEY = "__det_runtime_state__"
    _RUNTIME_STATE_VERSION = 2

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model
        self.args = args

        self.alignment_depths = self._parse_alignment_depths(args)
        self.temporal_kernel_size = int(getattr(args, "det_temporal_kernel_size", 5))
        self.temporal_kernel_mode = str(
            getattr(args, "det_temporal_kernel_mode", "avg")
        ).lower()
        self.temporal_loss_weight = float(
            getattr(args, "det_temporal_kernel_loss_weight", 0.1)
        )
        self.tracking_loss_weight = float(
            getattr(args, "det_dense_tracking_loss_weight", 0.1)
        )
        self.tracking_stride = int(getattr(args, "det_dense_tracking_stride", 1))
        self.tracking_topk_tokens = int(
            getattr(args, "det_dense_tracking_topk_tokens", 64)
        )
        self.local_loss_max_timestep = int(
            getattr(args, "det_local_loss_max_timestep", -1)
        )
        self.detach_temporal_target = bool(
            getattr(args, "det_detach_temporal_target", True)
        )

        self.external_tracking_enabled = bool(
            getattr(args, "det_external_tracking_enabled", False)
        )
        self.external_tracking_loss_weight = float(
            getattr(args, "det_external_tracking_loss_weight", 0.0)
        )
        self.external_tracking_max_timestep = int(
            getattr(args, "det_external_tracking_max_timestep", 400)
        )
        self.trajectory_root = str(getattr(args, "det_trajectory_root", "") or "")
        self.trajectory_subdir = str(
            getattr(args, "det_trajectory_subdir", "trajectories") or ""
        )
        ext = str(getattr(args, "det_trajectory_extension", ".pth") or ".pth")
        self.trajectory_extension = ext if ext.startswith(".") else f".{ext}"
        self.trajectory_use_visibility = bool(
            getattr(args, "det_trajectory_use_visibility", True)
        )
        self.trajectory_max_points = int(getattr(args, "det_trajectory_max_points", 0))
        self.trajectory_cache_size = int(getattr(args, "det_trajectory_cache_size", 256))
        self.external_tracking_min_active_samples = int(
            getattr(args, "det_external_tracking_min_active_samples", 1)
        )
        self.external_tracking_preflight_enabled = bool(
            getattr(args, "det_external_tracking_preflight_enabled", True)
        )
        self.external_tracking_preflight_strict = bool(
            getattr(args, "det_external_tracking_preflight_strict", False)
        )
        self.external_tracking_preflight_min_coverage = float(
            getattr(args, "det_external_tracking_preflight_min_coverage", 0.7)
        )
        self.external_tracking_preflight_max_items = int(
            getattr(args, "det_external_tracking_preflight_max_items", 2048)
        )
        self.external_tracking_preflight_validate_tensors = bool(
            getattr(args, "det_external_tracking_preflight_validate_tensors", False)
        )
        self.external_tracking_bind_item_paths = bool(
            getattr(args, "det_external_tracking_bind_item_paths", True)
        )
        self.external_tracking_use_batch_trajectories = bool(
            getattr(args, "det_external_tracking_use_batch_trajectories", False)
        )
        self.locality_adaptive_weighting_enabled = bool(
            getattr(args, "det_locality_adaptive_weighting_enabled", False)
        )
        self.locality_adaptive_target_ratio = float(
            getattr(args, "det_locality_adaptive_target_ratio", 0.65)
        )
        self.locality_adaptive_min_scale = float(
            getattr(args, "det_locality_adaptive_min_scale", 0.1)
        )
        self.locality_adaptive_ema_momentum = float(
            getattr(args, "det_locality_adaptive_ema_momentum", 0.9)
        )
        self.attention_locality_probe_enabled = bool(
            getattr(args, "det_attention_locality_probe_enabled", False)
        )
        self.attention_locality_probe_interval = int(
            getattr(args, "det_attention_locality_probe_interval", 1)
        )
        self.attention_locality_probe_ema_momentum = float(
            getattr(args, "det_attention_locality_probe_ema_momentum", 0.95)
        )
        self.attention_locality_probe_min_ratio = float(
            getattr(args, "det_attention_locality_probe_min_ratio", 0.65)
        )
        self.attention_locality_auto_policy = str(
            getattr(args, "det_attention_locality_auto_policy", "off")
        ).lower()
        self.attention_locality_auto_scale_min = float(
            getattr(args, "det_attention_locality_auto_scale_min", 0.1)
        )
        self.attention_locality_disable_threshold = float(
            getattr(args, "det_attention_locality_disable_threshold", 0.35)
        )
        self.attention_locality_reenable_threshold = float(
            getattr(args, "det_attention_locality_reenable_threshold", 0.55)
        )
        self.locality_profiler_enabled = bool(
            getattr(args, "det_locality_profiler_enabled", False)
        )
        self.locality_profiler_interval = int(
            getattr(args, "det_locality_profiler_interval", 200)
        )
        self.locality_profiler_bins = int(
            getattr(args, "det_locality_profiler_bins", 16)
        )
        self.locality_profiler_max_depths_in_plot = int(
            getattr(args, "det_locality_profiler_max_depths_in_plot", 8)
        )
        self.locality_profiler_log_prefix = str(
            getattr(args, "det_locality_profiler_log_prefix", "det_locality_profile")
            or "det_locality_profile"
        )
        self.locality_profiler_export_artifacts = bool(
            getattr(args, "det_locality_profiler_export_artifacts", False)
        )
        self.locality_profiler_export_dir = str(
            getattr(args, "det_locality_profiler_export_dir", "") or ""
        )
        self.controller_sync_enabled = bool(
            getattr(args, "det_controller_sync_enabled", False)
        )
        self.controller_sync_interval = int(
            getattr(args, "det_controller_sync_interval", 1)
        )
        self.controller_sync_include_per_depth = bool(
            getattr(args, "det_controller_sync_include_per_depth", True)
        )
        self.auto_safeguard_enabled = bool(
            getattr(args, "det_auto_safeguard_enabled", False)
        )
        self.auto_safeguard_locality_threshold = float(
            getattr(args, "det_auto_safeguard_locality_threshold", 0.45)
        )
        self.auto_safeguard_spike_ratio_threshold = float(
            getattr(args, "det_auto_safeguard_spike_ratio_threshold", 1.8)
        )
        self.auto_safeguard_bad_step_patience = int(
            getattr(args, "det_auto_safeguard_bad_step_patience", 4)
        )
        self.auto_safeguard_recovery_step_patience = int(
            getattr(args, "det_auto_safeguard_recovery_step_patience", 12)
        )
        self.auto_safeguard_local_scale_cap = float(
            getattr(args, "det_auto_safeguard_local_scale_cap", 0.0)
        )
        self.auto_safeguard_force_nonlocal_fallback = bool(
            getattr(args, "det_auto_safeguard_force_nonlocal_fallback", True)
        )
        self.auto_safeguard_nonlocal_min_blend = float(
            getattr(args, "det_auto_safeguard_nonlocal_min_blend", 0.2)
        )
        self.auto_safeguard_nonlocal_weight_boost = float(
            getattr(args, "det_auto_safeguard_nonlocal_weight_boost", 1.5)
        )
        self.unified_controller_enabled = bool(
            getattr(args, "det_unified_controller_enabled", False)
        )
        self.unified_controller_locality_source = str(
            getattr(args, "det_unified_controller_locality_source", "min")
        ).lower()
        self.unified_controller_min_scale = float(
            getattr(args, "det_unified_controller_min_scale", 0.1)
        )
        self.unified_controller_loss_ema_momentum = float(
            getattr(args, "det_unified_controller_loss_ema_momentum", 0.9)
        )
        self.unified_controller_spike_threshold = float(
            getattr(args, "det_unified_controller_spike_threshold", 1.8)
        )
        self.unified_controller_cooldown_steps = int(
            getattr(args, "det_unified_controller_cooldown_steps", 20)
        )
        self.unified_controller_recovery_steps = int(
            getattr(args, "det_unified_controller_recovery_steps", 100)
        )
        self.unified_controller_apply_to_adapter = bool(
            getattr(args, "det_unified_controller_apply_to_adapter", False)
        )
        self.per_depth_adaptive_enabled = bool(
            getattr(args, "det_per_depth_adaptive_enabled", False)
        )
        self.per_depth_adaptive_locality_target_ratio = float(
            getattr(args, "det_per_depth_adaptive_locality_target_ratio", 0.65)
        )
        self.per_depth_adaptive_min_scale = float(
            getattr(args, "det_per_depth_adaptive_min_scale", 0.1)
        )
        self.per_depth_adaptive_ema_momentum = float(
            getattr(args, "det_per_depth_adaptive_ema_momentum", 0.9)
        )
        self.per_depth_adaptive_spike_threshold = float(
            getattr(args, "det_per_depth_adaptive_spike_threshold", 1.8)
        )
        self.per_depth_adaptive_cooldown_steps = int(
            getattr(args, "det_per_depth_adaptive_cooldown_steps", 20)
        )
        self.per_depth_adaptive_recovery_steps = int(
            getattr(args, "det_per_depth_adaptive_recovery_steps", 100)
        )
        self.nonlocal_fallback_enabled = bool(
            getattr(args, "det_nonlocal_fallback_enabled", False)
        )
        self.nonlocal_fallback_loss_weight = float(
            getattr(args, "det_nonlocal_fallback_loss_weight", 0.0)
        )
        self.nonlocal_fallback_trigger_scale = float(
            getattr(args, "det_nonlocal_fallback_trigger_scale", 0.6)
        )
        self.nonlocal_fallback_min_blend = float(
            getattr(args, "det_nonlocal_fallback_min_blend", 0.0)
        )
        self.nonlocal_fallback_stride = int(
            getattr(args, "det_nonlocal_fallback_stride", 1)
        )
        self.nonlocal_fallback_mode = str(
            getattr(args, "det_nonlocal_fallback_mode", "cosine")
        ).lower()
        self.nonlocal_fallback_loss_warmup_steps = int(
            getattr(args, "det_nonlocal_fallback_loss_warmup_steps", 0)
        )
        self.optimizer_modulation_enabled = bool(
            getattr(args, "det_optimizer_modulation_enabled", False)
        )
        self.optimizer_modulation_target = str(
            getattr(args, "det_optimizer_modulation_target", "det_adapter")
        ).lower()
        self.optimizer_modulation_source = str(
            getattr(args, "det_optimizer_modulation_source", "min")
        ).lower()
        self.optimizer_modulation_min_scale = float(
            getattr(args, "det_optimizer_modulation_min_scale", 0.2)
        )
        self.loss_schedule_enabled = bool(
            getattr(args, "det_loss_schedule_enabled", False)
        )
        self.loss_schedule_shape = str(
            getattr(args, "det_loss_schedule_shape", "linear")
        ).lower()
        self.temporal_loss_warmup_steps = int(
            getattr(args, "det_temporal_loss_warmup_steps", 0)
        )
        self.dense_tracking_loss_warmup_steps = int(
            getattr(args, "det_dense_tracking_loss_warmup_steps", 0)
        )
        self.external_tracking_loss_warmup_steps = int(
            getattr(args, "det_external_tracking_loss_warmup_steps", 0)
        )
        self.high_frequency_loss_warmup_steps = int(
            getattr(args, "det_high_frequency_loss_warmup_steps", 0)
        )
        self.high_frequency_loss_enabled = bool(
            getattr(args, "det_high_frequency_loss_enabled", False)
        )
        self.high_frequency_loss_weight = float(
            getattr(args, "det_high_frequency_loss_weight", 0.0)
        )
        self.high_frequency_cutoff_frequency = int(
            getattr(args, "det_high_frequency_cutoff_frequency", 3)
        )
        self.high_frequency_max_timestep = int(
            getattr(args, "det_high_frequency_max_timestep", 400)
        )

        if self.temporal_kernel_mode not in _ALLOWED_KERNEL_MODES:
            logger.warning(
                "DeT helper: unsupported det_temporal_kernel_mode '%s'; using 'avg'.",
                self.temporal_kernel_mode,
            )
            self.temporal_kernel_mode = "avg"

        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_features: Dict[int, torch.Tensor] = {}
        self._attention_probe_ratio_by_depth: Dict[int, torch.Tensor] = {}

        self._trajectory_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._trajectory_cache_order: List[str] = []
        self._warned_missing_trajectories: set[str] = set()
        self._locality_ratio_ema: Optional[torch.Tensor] = None
        self._attention_locality_ema: Optional[torch.Tensor] = None
        self._attention_locality_disabled: bool = False
        self._unified_local_loss_ema: Optional[torch.Tensor] = None
        self._unified_cooldown_remaining = 0
        self._unified_recovery_remaining = 0
        self._per_depth_locality_ema: Dict[int, torch.Tensor] = {}
        self._per_depth_loss_ema: Dict[int, torch.Tensor] = {}
        self._per_depth_cooldown_remaining: Dict[int, int] = {}
        self._per_depth_recovery_remaining: Dict[int, int] = {}
        self._latest_locality_profile: Optional[Dict[str, Any]] = None
        self._auto_safeguard_bad_step_streak = 0
        self._auto_safeguard_good_step_streak = 0
        self._auto_safeguard_active = False
        try:
            setattr(self.args, "_det_attention_locality_scale", 1.0)
            setattr(self.args, "_det_locality_scale", 1.0)
            setattr(self.args, "_det_unified_controller_scale", 1.0)
            setattr(self.args, "_det_optimizer_lr_scale", 1.0)
        except Exception:
            pass

    @staticmethod
    def _serialize_optional_scalar_tensor(value: Optional[torch.Tensor]) -> Optional[float]:
        if value is None:
            return None
        if not torch.is_tensor(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            return numeric if math.isfinite(numeric) else None
        if value.numel() <= 0:
            return None
        scalar = value.detach().to(dtype=torch.float32, device="cpu").reshape(-1)[0]
        numeric = float(scalar.item())
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _deserialize_optional_scalar_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric):
            return None
        return torch.tensor(numeric, dtype=torch.float32, device="cpu")

    @classmethod
    def _serialize_depth_tensor_map(cls, values: Dict[int, torch.Tensor]) -> Dict[str, float]:
        serialized: Dict[str, float] = {}
        for depth, tensor in values.items():
            scalar = cls._serialize_optional_scalar_tensor(tensor)
            if scalar is None:
                continue
            serialized[str(int(depth))] = scalar
        return serialized

    @classmethod
    def _deserialize_depth_tensor_map(cls, values: Any) -> Dict[int, torch.Tensor]:
        if not isinstance(values, dict):
            return {}
        restored: Dict[int, torch.Tensor] = {}
        for raw_depth, raw_value in values.items():
            try:
                depth = int(raw_depth)
            except Exception:
                continue
            tensor = cls._deserialize_optional_scalar_tensor(raw_value)
            if tensor is None:
                continue
            restored[depth] = tensor
        return restored

    @staticmethod
    def _serialize_depth_int_map(values: Dict[int, int]) -> Dict[str, int]:
        serialized: Dict[str, int] = {}
        for depth, count in values.items():
            try:
                serialized[str(int(depth))] = max(0, int(count))
            except Exception:
                continue
        return serialized

    @staticmethod
    def _deserialize_depth_int_map(values: Any) -> Dict[int, int]:
        if not isinstance(values, dict):
            return {}
        restored: Dict[int, int] = {}
        for raw_depth, raw_value in values.items():
            try:
                depth = int(raw_depth)
                count = max(0, int(raw_value))
            except Exception:
                continue
            restored[depth] = count
        return restored

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    @staticmethod
    def _safe_non_negative_int(value: Any, default: int = 0) -> int:
        try:
            numeric = int(value)
        except Exception:
            return max(0, int(default))
        return max(0, numeric)

    def _get_controller_sync_world_size(self) -> int:
        if not self.controller_sync_enabled:
            return 1
        try:
            if not torch.distributed.is_available():
                return 1
            if not torch.distributed.is_initialized():
                return 1
            world_size = int(torch.distributed.get_world_size())
            return world_size if world_size > 1 else 1
        except Exception:
            return 1

    def _should_sync_controller(self, global_step: Optional[int]) -> bool:
        if not self.controller_sync_enabled:
            return False
        if self.controller_sync_interval <= 0:
            return False
        if global_step is None:
            return False
        if (int(global_step) % int(self.controller_sync_interval)) != 0:
            return False
        return self._get_controller_sync_world_size() > 1

    @staticmethod
    def _all_reduce_sum_in_place(value: torch.Tensor) -> bool:
        try:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            return True
        except Exception:
            return False

    @staticmethod
    def _all_reduce_max_in_place(value: torch.Tensor) -> bool:
        try:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
            return True
        except Exception:
            return False

    def _sync_optional_cpu_scalar_state(
        self,
        value: Optional[torch.Tensor],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        packed = torch.zeros(2, dtype=torch.float32, device=device)
        if value is not None and torch.is_tensor(value) and value.numel() > 0:
            scalar = value.detach().to(dtype=torch.float32, device=device).reshape(-1)[0]
            if bool(torch.isfinite(scalar).item()):
                packed[0] = scalar
                packed[1] = 1.0
        if not self._all_reduce_sum_in_place(packed):
            return value
        count = float(packed[1].item())
        if count <= 0.0:
            return None
        mean_value = packed[0] / max(count, 1e-6)
        return mean_value.detach().to(dtype=torch.float32, device="cpu")

    def _sync_int_state_max(
        self,
        value: int,
        *,
        device: torch.device,
    ) -> int:
        packed = torch.tensor(
            [max(0, int(value))],
            dtype=torch.int64,
            device=device,
        )
        if not self._all_reduce_max_in_place(packed):
            return max(0, int(value))
        return max(0, int(packed.item()))

    def _sync_bool_state_any(
        self,
        value: bool,
        *,
        device: torch.device,
    ) -> bool:
        packed = torch.tensor(
            [1 if bool(value) else 0],
            dtype=torch.int64,
            device=device,
        )
        if not self._all_reduce_max_in_place(packed):
            return bool(value)
        return bool(int(packed.item()) > 0)

    def _sync_depth_metric_map_mean(
        self,
        values: Dict[int, torch.Tensor],
        *,
        reference: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        if not values:
            return {}
        synced: Dict[int, torch.Tensor] = {}
        for depth in sorted(values.keys()):
            value = values.get(depth)
            if value is None or not torch.is_tensor(value):
                continue
            packed = torch.zeros(2, dtype=torch.float32, device=reference.device)
            scalar = value.detach().to(dtype=torch.float32, device=reference.device).reshape(-1)[0]
            if bool(torch.isfinite(scalar).item()):
                packed[0] = scalar
                packed[1] = 1.0
            if not self._all_reduce_sum_in_place(packed):
                synced[depth] = scalar.to(device=reference.device, dtype=reference.dtype)
                continue
            count = float(packed[1].item())
            if count <= 0.0:
                continue
            synced_value = (packed[0] / max(count, 1e-6)).to(dtype=reference.dtype)
            synced[depth] = synced_value
        return synced

    def _sync_depth_cpu_scalar_state_map(
        self,
        values: Dict[int, torch.Tensor],
        *,
        depths: List[int],
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        synced: Dict[int, torch.Tensor] = {}
        for depth in depths:
            value = values.get(depth)
            packed = torch.zeros(2, dtype=torch.float32, device=device)
            if value is not None and torch.is_tensor(value) and value.numel() > 0:
                scalar = value.detach().to(dtype=torch.float32, device=device).reshape(-1)[0]
                if bool(torch.isfinite(scalar).item()):
                    packed[0] = scalar
                    packed[1] = 1.0
            if not self._all_reduce_sum_in_place(packed):
                if value is not None and torch.is_tensor(value):
                    synced[depth] = value.detach().to(dtype=torch.float32, device="cpu")
                continue
            count = float(packed[1].item())
            if count <= 0.0:
                continue
            synced[depth] = (packed[0] / max(count, 1e-6)).detach().to(
                dtype=torch.float32,
                device="cpu",
            )
        return synced

    def _sync_depth_int_state_map_max(
        self,
        values: Dict[int, int],
        *,
        depths: List[int],
        device: torch.device,
    ) -> Dict[int, int]:
        synced: Dict[int, int] = {}
        for depth in depths:
            value = max(0, int(values.get(depth, 0)))
            packed = torch.tensor([value], dtype=torch.int64, device=device)
            if not self._all_reduce_max_in_place(packed):
                if value > 0:
                    synced[depth] = value
                continue
            synced_value = max(0, int(packed.item()))
            if synced_value > 0:
                synced[depth] = synced_value
        return synced

    def _sync_controller_runtime_state_if_needed(
        self,
        *,
        global_step: Optional[int],
        reference: torch.Tensor,
    ) -> bool:
        if not self._should_sync_controller(global_step):
            return False

        device = reference.device
        self._locality_ratio_ema = self._sync_optional_cpu_scalar_state(
            self._locality_ratio_ema,
            device=device,
        )
        self._attention_locality_ema = self._sync_optional_cpu_scalar_state(
            self._attention_locality_ema,
            device=device,
        )
        self._attention_locality_disabled = self._sync_bool_state_any(
            self._attention_locality_disabled,
            device=device,
        )
        self._unified_local_loss_ema = self._sync_optional_cpu_scalar_state(
            self._unified_local_loss_ema,
            device=device,
        )
        self._unified_cooldown_remaining = self._sync_int_state_max(
            self._unified_cooldown_remaining,
            device=device,
        )
        self._unified_recovery_remaining = self._sync_int_state_max(
            self._unified_recovery_remaining,
            device=device,
        )
        self._auto_safeguard_bad_step_streak = self._sync_int_state_max(
            self._auto_safeguard_bad_step_streak,
            device=device,
        )
        self._auto_safeguard_good_step_streak = self._sync_int_state_max(
            self._auto_safeguard_good_step_streak,
            device=device,
        )
        self._auto_safeguard_active = self._sync_bool_state_any(
            self._auto_safeguard_active,
            device=device,
        )

        if self.controller_sync_include_per_depth:
            all_depths: List[int] = sorted(
                set(int(d) for d in self.alignment_depths)
                | set(int(d) for d in self._per_depth_locality_ema.keys())
                | set(int(d) for d in self._per_depth_loss_ema.keys())
                | set(int(d) for d in self._per_depth_cooldown_remaining.keys())
                | set(int(d) for d in self._per_depth_recovery_remaining.keys())
            )
            self._per_depth_locality_ema = self._sync_depth_cpu_scalar_state_map(
                self._per_depth_locality_ema,
                depths=all_depths,
                device=device,
            )
            self._per_depth_loss_ema = self._sync_depth_cpu_scalar_state_map(
                self._per_depth_loss_ema,
                depths=all_depths,
                device=device,
            )
            self._per_depth_cooldown_remaining = self._sync_depth_int_state_map_max(
                self._per_depth_cooldown_remaining,
                depths=all_depths,
                device=device,
            )
            self._per_depth_recovery_remaining = self._sync_depth_int_state_map_max(
                self._per_depth_recovery_remaining,
                depths=all_depths,
                device=device,
            )

        return True

    def _export_runtime_state(self) -> Dict[str, Any]:
        return {
            "version": int(self._RUNTIME_STATE_VERSION),
            "locality_ratio_ema": self._serialize_optional_scalar_tensor(
                self._locality_ratio_ema
            ),
            "attention_locality_ema": self._serialize_optional_scalar_tensor(
                self._attention_locality_ema
            ),
            "attention_locality_disabled": bool(self._attention_locality_disabled),
            "unified_local_loss_ema": self._serialize_optional_scalar_tensor(
                self._unified_local_loss_ema
            ),
            "unified_cooldown_remaining": max(0, int(self._unified_cooldown_remaining)),
            "unified_recovery_remaining": max(0, int(self._unified_recovery_remaining)),
            "per_depth_locality_ema": self._serialize_depth_tensor_map(
                self._per_depth_locality_ema
            ),
            "per_depth_loss_ema": self._serialize_depth_tensor_map(
                self._per_depth_loss_ema
            ),
            "per_depth_cooldown_remaining": self._serialize_depth_int_map(
                self._per_depth_cooldown_remaining
            ),
            "per_depth_recovery_remaining": self._serialize_depth_int_map(
                self._per_depth_recovery_remaining
            ),
            "auto_safeguard_bad_step_streak": max(
                0, int(self._auto_safeguard_bad_step_streak)
            ),
            "auto_safeguard_good_step_streak": max(
                0, int(self._auto_safeguard_good_step_streak)
            ),
            "auto_safeguard_active": bool(self._auto_safeguard_active),
            "args_runtime_scales": {
                "det_attention_locality_scale": self._safe_float(
                    getattr(self.args, "_det_attention_locality_scale", 1.0),
                    1.0,
                ),
                "det_locality_scale": self._safe_float(
                    getattr(self.args, "_det_locality_scale", 1.0),
                    1.0,
                ),
                "det_unified_controller_scale": self._safe_float(
                    getattr(self.args, "_det_unified_controller_scale", 1.0),
                    1.0,
                ),
                "det_optimizer_lr_scale": self._safe_float(
                    getattr(self.args, "_det_optimizer_lr_scale", 1.0),
                    1.0,
                ),
            },
        }

    def _import_runtime_state(self, runtime_state: Any) -> None:
        if not isinstance(runtime_state, dict):
            return

        try:
            version = int(runtime_state.get("version", 1))
        except Exception:
            version = 1
        if version > int(self._RUNTIME_STATE_VERSION):
            logger.warning(
                "DeT helper: runtime state version %s is newer than supported version %s; attempting partial restore.",
                version,
                self._RUNTIME_STATE_VERSION,
            )

        self._locality_ratio_ema = self._deserialize_optional_scalar_tensor(
            runtime_state.get("locality_ratio_ema")
        )
        self._attention_locality_ema = self._deserialize_optional_scalar_tensor(
            runtime_state.get("attention_locality_ema")
        )
        self._attention_locality_disabled = bool(
            runtime_state.get("attention_locality_disabled", False)
        )
        self._unified_local_loss_ema = self._deserialize_optional_scalar_tensor(
            runtime_state.get("unified_local_loss_ema")
        )
        self._unified_cooldown_remaining = self._safe_non_negative_int(
            runtime_state.get("unified_cooldown_remaining", 0),
            0,
        )
        self._unified_recovery_remaining = self._safe_non_negative_int(
            runtime_state.get("unified_recovery_remaining", 0),
            0,
        )
        self._per_depth_locality_ema = self._deserialize_depth_tensor_map(
            runtime_state.get("per_depth_locality_ema")
        )
        self._per_depth_loss_ema = self._deserialize_depth_tensor_map(
            runtime_state.get("per_depth_loss_ema")
        )
        self._per_depth_cooldown_remaining = self._deserialize_depth_int_map(
            runtime_state.get("per_depth_cooldown_remaining")
        )
        self._per_depth_recovery_remaining = self._deserialize_depth_int_map(
            runtime_state.get("per_depth_recovery_remaining")
        )
        self._auto_safeguard_bad_step_streak = self._safe_non_negative_int(
            runtime_state.get("auto_safeguard_bad_step_streak", 0),
            0,
        )
        self._auto_safeguard_good_step_streak = self._safe_non_negative_int(
            runtime_state.get("auto_safeguard_good_step_streak", 0),
            0,
        )
        self._auto_safeguard_active = bool(
            runtime_state.get("auto_safeguard_active", False)
        )

        raw_scales = runtime_state.get("args_runtime_scales")
        if isinstance(raw_scales, dict):
            try:
                setattr(
                    self.args,
                    "_det_attention_locality_scale",
                    self._safe_float(
                        raw_scales.get("det_attention_locality_scale", 1.0),
                        1.0,
                    ),
                )
                setattr(
                    self.args,
                    "_det_locality_scale",
                    self._safe_float(raw_scales.get("det_locality_scale", 1.0), 1.0),
                )
                setattr(
                    self.args,
                    "_det_unified_controller_scale",
                    self._safe_float(
                        raw_scales.get("det_unified_controller_scale", 1.0),
                        1.0,
                    ),
                )
                setattr(
                    self.args,
                    "_det_optimizer_lr_scale",
                    self._safe_float(raw_scales.get("det_optimizer_lr_scale", 1.0), 1.0),
                )
            except Exception:
                pass

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        state[self._RUNTIME_STATE_KEY] = self._export_runtime_state()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):  # type: ignore[override]
        runtime_state = state_dict.get(self._RUNTIME_STATE_KEY)
        base_state = {
            key: value
            for key, value in state_dict.items()
            if key != self._RUNTIME_STATE_KEY
        }
        incompatible = super().load_state_dict(base_state, strict=False)
        try:
            self._import_runtime_state(runtime_state)
        except Exception as exc:
            logger.warning("DeT helper: failed to restore runtime state: %s", exc)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for DeTMotionTransferHelper: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return incompatible

    @staticmethod
    def _parse_alignment_depths(args: Any) -> List[int]:
        raw_depths = getattr(args, "det_alignment_depths", None)
        if isinstance(raw_depths, Sequence) and not isinstance(raw_depths, (str, bytes)):
            parsed = [int(v) for v in raw_depths]
        else:
            parsed = [int(getattr(args, "det_alignment_depth", 8))]
        deduped: List[int] = []
        for depth in parsed:
            if depth not in deduped:
                deduped.append(depth)
        return deduped or [8]

    def _locate_blocks(self) -> Sequence[nn.Module]:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("DeT helper: could not locate transformer blocks.")

    @staticmethod
    def _resolve_depth(depth: int, num_blocks: int) -> Optional[int]:
        if -num_blocks <= depth < num_blocks:
            return depth % num_blocks
        return None

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        value = output[0] if isinstance(output, (tuple, list)) and output else output
        if not torch.is_tensor(value):
            return None
        if value.ndim < 3:
            return None
        if value.ndim == 3:
            return value
        batch = int(value.shape[0])
        dim = int(value.shape[-1])
        return value.reshape(batch, -1, dim)

    def _build_hook(self, depth: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None:
                return
            self.captured_features[depth] = tensor

        return hook

    @staticmethod
    def _apply_batched_rotary(
        tensor: torch.Tensor,
        rotary: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, num_heads, head_dim = tensor.shape
        tensor_complex = torch.view_as_complex(
            tensor.float().reshape(batch_size, token_count, num_heads, head_dim // 2, 2)
        )
        rotary_complex = torch.view_as_complex(
            rotary.float().reshape(batch_size, token_count, 1, head_dim // 2, 2)
        )
        output = torch.view_as_real(tensor_complex * rotary_complex).reshape(
            batch_size,
            token_count,
            num_heads,
            head_dim,
        )
        return output.type_as(tensor)

    def _compute_attention_probe_locality_ratio(
        self,
        attn_module: nn.Module,
        inputs: Tuple[Any, ...],
    ) -> Optional[torch.Tensor]:
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None
        if len(inputs) < 4:
            return None

        sparse_attention = bool(inputs[4]) if len(inputs) >= 5 else False
        if sparse_attention:
            return None

        num_heads = getattr(attn_module, "num_heads", None)
        head_dim = getattr(attn_module, "head_dim", None)
        if num_heads is None or head_dim is None:
            return None

        batch_size, token_count = int(x.shape[0]), int(x.shape[1])
        if batch_size <= 0 or token_count <= 0:
            return None

        grid_sizes = inputs[2]
        freqs = inputs[3]
        batched_rotary = inputs[5] if (len(inputs) >= 6 and torch.is_tensor(inputs[5])) else None
        extra_tokens_raw = inputs[6] if len(inputs) >= 7 else 0
        try:
            extra_tokens = max(0, int(extra_tokens_raw))
        except Exception:
            extra_tokens = 0

        usable_tokens = token_count - extra_tokens
        if usable_tokens <= 1:
            return None

        frames = 1
        if torch.is_tensor(grid_sizes) and grid_sizes.ndim >= 2 and grid_sizes.shape[-1] >= 1:
            try:
                grid_frames = int(grid_sizes[0, 0].detach().item())
            except Exception:
                grid_frames = 1
            if grid_frames > 1 and usable_tokens % grid_frames == 0:
                frames = grid_frames
        if frames <= 1:
            return None

        tokens_per_frame = usable_tokens // frames
        if tokens_per_frame <= 0:
            return None

        token_budget = max(
            16,
            min(512, max(1, int(self.tracking_topk_tokens)) * 4),
        )
        spatial_probe_count = max(
            1,
            min(tokens_per_frame, token_budget // max(1, frames)),
        )
        if spatial_probe_count <= 0:
            return None

        if spatial_probe_count >= tokens_per_frame:
            spatial_ids = torch.arange(tokens_per_frame, dtype=torch.long, device=x.device)
        else:
            spatial_ids = torch.linspace(
                0,
                tokens_per_frame - 1,
                steps=spatial_probe_count,
                device=x.device,
            ).round().long()
            spatial_ids = torch.unique(spatial_ids, sorted=True)
        if spatial_ids.numel() <= 0:
            return None

        frame_offsets = (
            torch.arange(frames, dtype=torch.long, device=x.device) * tokens_per_frame
            + int(extra_tokens)
        )
        sampled_indices = (frame_offsets[:, None] + spatial_ids[None, :]).reshape(-1)
        if sampled_indices.numel() <= 1:
            return None

        with torch.no_grad():
            x_detached = x.detach()
            q = attn_module.norm_q(attn_module.q(x_detached)).view(
                batch_size,
                token_count,
                num_heads,
                head_dim,
            )
            k = attn_module.norm_k(attn_module.k(x_detached)).view(
                batch_size,
                token_count,
                num_heads,
                head_dim,
            )

            if batched_rotary is not None:
                q = self._apply_batched_rotary(q, batched_rotary.detach())
                k = self._apply_batched_rotary(k, batched_rotary.detach())
            elif torch.is_tensor(freqs):
                from wan.modules.model import rope_apply, rope_apply_inplace_cached

                use_comfy = bool(getattr(attn_module, "use_comfy_rope", False))
                rope_func = getattr(attn_module, "rope_func", "default")
                rope_on_the_fly = bool(getattr(attn_module, "rope_on_the_fly", False))
                if use_comfy and rope_func == "comfy":
                    try:
                        q, k = attn_module.comfyrope(q, k, freqs)  # type: ignore[attr-defined]
                    except Exception:
                        rope_apply_inplace_cached(
                            q,
                            grid_sizes,
                            freqs,
                            extra_tokens=extra_tokens,
                        )
                        rope_apply_inplace_cached(
                            k,
                            grid_sizes,
                            freqs,
                            extra_tokens=extra_tokens,
                        )
                elif rope_on_the_fly:
                    q = rope_apply(q, grid_sizes, freqs, extra_tokens=extra_tokens)
                    k = rope_apply(k, grid_sizes, freqs, extra_tokens=extra_tokens)
                else:
                    rope_apply_inplace_cached(
                        q,
                        grid_sizes,
                        freqs,
                        extra_tokens=extra_tokens,
                    )
                    rope_apply_inplace_cached(
                        k,
                        grid_sizes,
                        freqs,
                        extra_tokens=extra_tokens,
                    )

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q_sample = q.index_select(dim=2, index=sampled_indices)
            k_sample = k.index_select(dim=2, index=sampled_indices)

            scale = 1.0 / math.sqrt(float(max(head_dim, 1)))
            attn_logits = torch.matmul(
                q_sample.to(torch.float32),
                k_sample.transpose(-2, -1).to(torch.float32),
            ) * scale
            attn_prob = torch.softmax(attn_logits, dim=-1)

            sampled_grid_indices = sampled_indices - int(extra_tokens)
            sampled_frames = torch.div(
                sampled_grid_indices,
                tokens_per_frame,
                rounding_mode="floor",
            )
            sampled_spatial = torch.remainder(sampled_grid_indices, tokens_per_frame)
            temporal_radius = max(1, int(self.temporal_kernel_size // 2))
            temporal_mask = (
                sampled_frames[:, None] - sampled_frames[None, :]
            ).abs() <= temporal_radius
            spatial_mask = sampled_spatial[:, None].eq(sampled_spatial[None, :])
            locality_mask = (temporal_mask & spatial_mask).to(dtype=attn_prob.dtype)
            if locality_mask.numel() <= 0:
                return None

            local_mass = (
                attn_prob
                * locality_mask.view(1, 1, locality_mask.shape[0], locality_mask.shape[1])
            ).sum(dim=-1).clamp(min=0.0, max=1.0)
            ratio = local_mass.mean()
            if not bool(torch.isfinite(ratio).item()):
                return None
            return ratio.detach().to(dtype=torch.float32, device="cpu")

    def _build_attention_probe_hook(self, depth: int, attn_module: nn.Module):
        def hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            try:
                locality_ratio = self._compute_attention_probe_locality_ratio(
                    attn_module,
                    inputs,
                )
            except Exception:
                return
            if locality_ratio is None:
                return
            self._attention_probe_ratio_by_depth[depth] = locality_ratio

        return hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        resolved_depths: List[int] = []
        attention_probe_hook_count = 0
        for depth in self.alignment_depths:
            resolved = self._resolve_depth(depth, num_blocks)
            if resolved is None:
                logger.warning(
                    "DeT helper: alignment depth %s is outside [-%s, %s).",
                    depth,
                    num_blocks,
                    num_blocks,
                )
                continue
            resolved_depths.append(resolved)
            block = blocks[resolved]
            handle = block.register_forward_hook(self._build_hook(resolved))
            self.hook_handles.append(handle)
            if self.attention_locality_probe_enabled:
                attn_module = getattr(block, "self_attn", None)
                if attn_module is not None:
                    attn_handle = attn_module.register_forward_pre_hook(
                        self._build_attention_probe_hook(resolved, attn_module)
                    )
                    self.hook_handles.append(attn_handle)
                    attention_probe_hook_count += 1
        self.alignment_depths = resolved_depths
        if self.hook_handles:
            logger.info(
                "DeT helper: attached hooks to blocks %s.",
                ", ".join(str(idx) for idx in self.alignment_depths),
            )
            if self.attention_locality_probe_enabled and attention_probe_hook_count <= 0:
                logger.warning(
                    "DeT helper: attention locality probe is enabled, but no self_attn modules were found at target depths; probe falls back to tracking locality ratio."
                )
        else:
            logger.warning("DeT helper: no hooks attached; helper is inactive.")

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles = []
        self.captured_features.clear()
        self._attention_probe_ratio_by_depth.clear()

    @staticmethod
    def _infer_temporal_layout(
        token_count: int,
        latents: torch.Tensor,
    ) -> Tuple[int, int]:
        if latents.ndim != 5:
            return 1, token_count
        desired_frames = max(1, int(latents.shape[2]))
        if desired_frames == 1:
            return 1, token_count
        if token_count % desired_frames == 0:
            return desired_frames, token_count // desired_frames
        for frames in range(desired_frames, 0, -1):
            if token_count % frames == 0:
                return frames, token_count // frames
        return 1, token_count

    def _select_topk_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, F, T, D]
        _, frames, token_count, dim = tokens.shape
        k = self.tracking_topk_tokens
        if k <= 0 or k >= token_count:
            return tokens
        if frames <= 1:
            return tokens[:, :, :k, :]
        motion_scores = (tokens[:, 1:] - tokens[:, :-1]).pow(2).mean(dim=(1, 3))
        _, indices = torch.topk(motion_scores, k=k, dim=-1, largest=True, sorted=False)
        gather_index = indices.unsqueeze(1).unsqueeze(-1).expand(-1, frames, -1, dim)
        return torch.gather(tokens, dim=2, index=gather_index)

    def _build_temporal_kernel(
        self,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        kernel_size = self.temporal_kernel_size
        if self.temporal_kernel_mode == "gaussian":
            positions = torch.arange(kernel_size, device=device, dtype=dtype)
            positions = positions - (float(kernel_size - 1) * 0.5)
            sigma = max(float(kernel_size) / 3.0, 1e-6)
            base = torch.exp(-0.5 * (positions / sigma) ** 2)
        else:
            base = torch.ones(kernel_size, device=device, dtype=dtype)
        base = base / base.sum().clamp_min(1e-12)
        return base.view(1, 1, kernel_size).repeat(channels, 1, 1)

    def _compute_temporal_kernel_loss(
        self,
        tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # tokens: [B, F, T, D]
        _, frames, token_count, channels = tokens.shape
        if frames <= 1 or token_count <= 0:
            return None
        sequence = tokens.permute(0, 2, 3, 1).reshape(-1, channels, frames)
        kernel = self._build_temporal_kernel(
            channels=channels,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        smoothed = F.conv1d(
            sequence,
            weight=kernel,
            padding=self.temporal_kernel_size // 2,
            groups=channels,
        )
        target = smoothed.detach() if self.detach_temporal_target else smoothed
        return F.mse_loss(sequence, target, reduction="mean")

    def _compute_dense_tracking_loss(
        self,
        tokens: torch.Tensor,
        *,
        collect_profile: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # tokens: [B, F, T, D]
        batch, frames, token_count, dim = tokens.shape
        stride = max(1, self.tracking_stride)
        if frames <= stride or token_count <= 0:
            return None, None, None

        source = tokens[:, :-stride]
        target = tokens[:, stride:]
        pair_count = source.shape[1]

        source_flat = F.normalize(source.reshape(batch * pair_count, token_count, dim), dim=-1)
        target_flat = F.normalize(target.reshape(batch * pair_count, token_count, dim), dim=-1)

        sim = torch.bmm(source_flat, target_flat.transpose(1, 2))
        best_idx = torch.argmax(sim, dim=-1)
        matched = torch.gather(
            target_flat,
            dim=1,
            index=best_idx.unsqueeze(-1).expand(-1, -1, dim),
        )
        cosine = (source_flat * matched).sum(dim=-1)
        tracking_loss = (1.0 - cosine).mean()

        locality_ratio: Optional[torch.Tensor] = None
        if (
            self.locality_adaptive_weighting_enabled
            or self.attention_locality_probe_enabled
        ):
            best_similarity = sim.max(dim=-1).values
            self_similarity = (source_flat * target_flat).sum(dim=-1)
            denom = best_similarity.mean().abs().clamp_min(1e-6)
            locality_ratio = (self_similarity.mean() / denom).clamp(0.0, 2.0)

        profile_distances: Optional[torch.Tensor] = None
        if collect_profile:
            token_ids = torch.arange(
                token_count,
                device=best_idx.device,
                dtype=torch.long,
            ).view(1, token_count)
            displacement = (best_idx - token_ids).abs().to(dtype=torch.float32)
            norm_denom = float(max(token_count - 1, 1))
            profile_distances = (displacement / norm_denom).reshape(-1)
            finite = torch.isfinite(profile_distances)
            if bool(finite.any().item()):
                profile_distances = profile_distances[finite].clamp(0.0, 1.0)
            else:
                profile_distances = None

        return tracking_loss, locality_ratio, profile_distances

    def _should_collect_locality_profile(self, global_step: Optional[int]) -> bool:
        if not self.locality_profiler_enabled:
            return False
        if self.locality_profiler_interval <= 0:
            return False
        if global_step is None:
            return False
        return (int(global_step) % int(self.locality_profiler_interval)) == 0

    def _build_locality_profile_payload(
        self,
        *,
        distance_by_depth: Dict[int, torch.Tensor],
        global_step: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not distance_by_depth:
            return None

        bins = max(4, min(int(self.locality_profiler_bins), 512))
        depths = sorted(int(depth) for depth in distance_by_depth.keys())
        active_depths: List[int] = []
        hist_rows: List[torch.Tensor] = []
        mean_rows: List[torch.Tensor] = []
        std_rows: List[torch.Tensor] = []
        count_rows: List[float] = []

        for depth in depths:
            distances = distance_by_depth[depth]
            if distances.numel() <= 0:
                continue
            clipped = distances.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
            hist = torch.histc(clipped, bins=bins, min=0.0, max=1.0).to(
                dtype=torch.float32
            )
            total = float(hist.sum().item())
            if total > 0.0:
                hist = hist / total
            active_depths.append(depth)
            hist_rows.append(hist)
            mean_rows.append(clipped.mean())
            std_rows.append(clipped.std(unbiased=False))
            count_rows.append(float(clipped.numel()))

        if not hist_rows:
            return None

        hist_matrix = torch.stack(hist_rows, dim=0).to(dtype=torch.float32, device="cpu")
        mean_by_depth = torch.stack(mean_rows, dim=0).to(dtype=torch.float32, device="cpu")
        std_by_depth = torch.stack(std_rows, dim=0).to(dtype=torch.float32, device="cpu")
        sample_count_by_depth = torch.tensor(
            count_rows,
            dtype=torch.float32,
            device="cpu",
        )
        bin_centers = torch.linspace(
            0.0,
            1.0,
            steps=bins,
            dtype=torch.float32,
            device="cpu",
        )

        return {
            "step": int(global_step) if global_step is not None else -1,
            "depths": active_depths,
            "hist_matrix": hist_matrix,
            "bin_centers": bin_centers,
            "mean_by_depth": mean_by_depth,
            "std_by_depth": std_by_depth,
            "sample_count_by_depth": sample_count_by_depth,
        }

    def consume_locality_profile(
        self,
        *,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        profile = self._latest_locality_profile
        if profile is None:
            return None
        if step is not None:
            profile_step = int(profile.get("step", -1))
            if profile_step != int(step):
                return None
        self._latest_locality_profile = None
        return profile

    def _compute_nonlocal_temporal_loss(self, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        # tokens: [B, F, T, D]
        batch, frames, token_count, dim = tokens.shape
        del batch, dim
        stride = max(1, self.nonlocal_fallback_stride)
        if frames <= stride or token_count <= 0:
            return None

        pooled = tokens.mean(dim=2)  # [B, F, D]
        source = pooled[:, :-stride, :]
        target = pooled[:, stride:, :]
        if self.nonlocal_fallback_mode == "mse":
            return F.mse_loss(source, target, reduction="mean")

        source_norm = F.normalize(source, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cosine = (source_norm * target_norm).sum(dim=-1)
        return (1.0 - cosine).mean()

    def _compute_warmup_factor(
        self,
        *,
        global_step: Optional[int],
        warmup_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.loss_schedule_enabled or warmup_steps <= 0:
            return torch.tensor(1.0, device=device, dtype=dtype)
        if global_step is None:
            return torch.tensor(1.0, device=device, dtype=dtype)
        progress = min(max((float(global_step) + 1.0) / float(warmup_steps), 0.0), 1.0)
        if self.loss_schedule_shape == "cosine":
            factor = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            factor = progress
        return torch.tensor(float(factor), device=device, dtype=dtype)

    def _compute_attention_locality_auto_scale(
        self,
        *,
        locality_ratios: List[torch.Tensor],
        reference: torch.Tensor,
        global_step: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.attention_locality_probe_enabled or not locality_ratios:
            return None, {}

        update_ema = True
        if (
            global_step is not None
            and self.attention_locality_probe_interval > 1
            and (int(global_step) % int(self.attention_locality_probe_interval)) != 0
        ):
            update_ema = False

        ratio_mean = torch.stack(locality_ratios).mean()
        if not torch.isfinite(ratio_mean):
            return None, {}

        ratio_ref = ratio_mean.detach().to(dtype=torch.float32, device="cpu")
        if self._attention_locality_ema is None:
            self._attention_locality_ema = ratio_ref
        elif update_ema:
            momentum = float(self.attention_locality_probe_ema_momentum)
            self._attention_locality_ema = (
                self._attention_locality_ema * momentum
                + ratio_ref * (1.0 - momentum)
            )

        ema_ratio = self._attention_locality_ema.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        scale = reference.new_tensor(1.0)
        policy_active = reference.new_tensor(0.0)
        if self.attention_locality_auto_policy == "scale":
            target_ratio = max(self.attention_locality_probe_min_ratio, 1e-6)
            scale = (ema_ratio / target_ratio).clamp(
                min=self.attention_locality_auto_scale_min,
                max=1.0,
            )
            policy_active = reference.new_tensor(1.0)
        elif self.attention_locality_auto_policy == "disable":
            if self._attention_locality_disabled:
                if float(ema_ratio.item()) >= float(self.attention_locality_reenable_threshold):
                    self._attention_locality_disabled = False
            else:
                if float(ema_ratio.item()) < float(self.attention_locality_disable_threshold):
                    self._attention_locality_disabled = True
            scale = reference.new_tensor(0.0 if self._attention_locality_disabled else 1.0)
            policy_active = reference.new_tensor(1.0 if self._attention_locality_disabled else 0.0)

        metrics: Dict[str, torch.Tensor] = {
            "det_attention_locality_ratio": ema_ratio.detach(),
            "det_attention_locality_scale": scale.detach(),
            "det_attention_locality_policy_active": policy_active.detach(),
        }
        return scale, metrics

    def _resolve_unified_controller_locality_scale(
        self,
        *,
        locality_scale: Optional[torch.Tensor],
        attention_locality_scale: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        one = reference.new_tensor(1.0)

        def _safe_scale(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if not torch.is_tensor(value):
                value = reference.new_tensor(float(value))
            if not bool(torch.isfinite(value).all().item()):
                return None
            return value.clamp(min=0.0, max=1.0)

        locality = _safe_scale(locality_scale)
        attention = _safe_scale(attention_locality_scale)
        source = self.unified_controller_locality_source

        if source == "locality_adaptive":
            return locality if locality is not None else one
        if source == "attention_probe":
            return attention if attention is not None else one
        if locality is not None and attention is not None:
            return torch.minimum(locality, attention)
        if locality is not None:
            return locality
        if attention is not None:
            return attention
        return one

    def _compute_unified_controller_scale(
        self,
        *,
        local_loss_value: Optional[torch.Tensor],
        locality_scale: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        min_scale = min(max(float(self.unified_controller_min_scale), 0.0), 1.0)
        stability_value = 1.0
        spike_ratio_value = 1.0

        if local_loss_value is not None and bool(torch.isfinite(local_loss_value).all().item()):
            detached_loss = local_loss_value.detach().to(dtype=torch.float32, device="cpu")
            prev_ema = self._unified_local_loss_ema
            if prev_ema is not None and bool(torch.isfinite(prev_ema).all().item()):
                denom = prev_ema.abs().clamp_min(1e-6)
                ratio = (detached_loss / denom).clamp(min=0.0, max=1e6)
                spike_ratio_value = float(ratio.item())
                if spike_ratio_value > float(self.unified_controller_spike_threshold):
                    self._unified_cooldown_remaining = max(
                        0,
                        int(self.unified_controller_cooldown_steps),
                    )
                    self._unified_recovery_remaining = max(
                        0,
                        int(self.unified_controller_recovery_steps),
                    )
                momentum = float(self.unified_controller_loss_ema_momentum)
                self._unified_local_loss_ema = (
                    prev_ema * momentum + detached_loss * (1.0 - momentum)
                )
            else:
                self._unified_local_loss_ema = detached_loss

        cooldown_active = self._unified_cooldown_remaining > 0
        if cooldown_active:
            stability_value = min_scale
            self._unified_cooldown_remaining -= 1
        elif self._unified_recovery_remaining > 0:
            total_recovery = max(1, int(self.unified_controller_recovery_steps))
            recovered_steps = total_recovery - self._unified_recovery_remaining + 1
            progress = min(max(float(recovered_steps) / float(total_recovery), 0.0), 1.0)
            stability_value = min_scale + (1.0 - min_scale) * progress
            self._unified_recovery_remaining -= 1
        else:
            stability_value = 1.0

        stability = reference.new_tensor(stability_value).clamp(min=0.0, max=1.0)
        locality = locality_scale.clamp(min=0.0, max=1.0)
        scale = (locality * stability).clamp(min=min_scale, max=1.0)
        metrics: Dict[str, torch.Tensor] = {
            "det_unified_controller_scale": scale.detach(),
            "det_unified_controller_locality_scale": locality.detach(),
            "det_unified_controller_stability_scale": stability.detach(),
            "det_unified_controller_spike_ratio": reference.new_tensor(
                spike_ratio_value
            ).detach(),
            "det_unified_controller_cooldown_active": reference.new_tensor(
                1.0 if cooldown_active else 0.0
            ).detach(),
        }
        return scale, metrics

    def _compute_auto_safeguard_modulation(
        self,
        *,
        local_scale: Optional[torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> Dict[str, Any]:
        if not self.auto_safeguard_enabled:
            return {
                "active": False,
                "force_nonlocal": False,
                "nonlocal_min_blend": 0.0,
                "nonlocal_weight_boost": 1.0,
                "local_scale_cap": 1.0,
                "metrics": {},
            }

        observed_locality = 1.0
        if local_scale is not None and bool(torch.isfinite(local_scale).all().item()):
            observed_locality = float(local_scale.detach().float().mean().item())

        spike_ratio = 1.0
        unified_spike = metrics.get("det_unified_controller_spike_ratio")
        if unified_spike is not None and bool(torch.isfinite(unified_spike).all().item()):
            spike_ratio = float(unified_spike.detach().float().mean().item())

        cooldown_active = False
        unified_cooldown = metrics.get("det_unified_controller_cooldown_active")
        if unified_cooldown is not None and bool(torch.isfinite(unified_cooldown).all().item()):
            cooldown_active = float(unified_cooldown.detach().float().mean().item()) > 0.5

        risk_locality = observed_locality < float(self.auto_safeguard_locality_threshold)
        risk_spike = spike_ratio > float(self.auto_safeguard_spike_ratio_threshold)
        risk_cooldown = cooldown_active
        risky_step = risk_locality or risk_spike or risk_cooldown

        if risky_step:
            self._auto_safeguard_bad_step_streak += 1
            self._auto_safeguard_good_step_streak = 0
        else:
            self._auto_safeguard_good_step_streak += 1
            self._auto_safeguard_bad_step_streak = 0

        if (
            not self._auto_safeguard_active
            and self._auto_safeguard_bad_step_streak
            >= max(1, int(self.auto_safeguard_bad_step_patience))
        ):
            self._auto_safeguard_active = True
            self._auto_safeguard_good_step_streak = 0
        elif (
            self._auto_safeguard_active
            and self._auto_safeguard_good_step_streak
            >= max(1, int(self.auto_safeguard_recovery_step_patience))
        ):
            self._auto_safeguard_active = False
            self._auto_safeguard_bad_step_streak = 0

        configured_cap = min(max(float(self.auto_safeguard_local_scale_cap), 0.0), 1.0)
        local_scale_cap = configured_cap if self._auto_safeguard_active else 1.0
        force_nonlocal = (
            self._auto_safeguard_active
            and self.auto_safeguard_force_nonlocal_fallback
        )
        nonlocal_min_blend = (
            min(max(float(self.auto_safeguard_nonlocal_min_blend), 0.0), 1.0)
            if force_nonlocal
            else 0.0
        )
        nonlocal_boost = (
            max(1.0, float(self.auto_safeguard_nonlocal_weight_boost))
            if force_nonlocal
            else 1.0
        )

        out_metrics: Dict[str, torch.Tensor] = {
            "det_auto_safeguard_active": reference.new_tensor(
                1.0 if self._auto_safeguard_active else 0.0
            ).detach(),
            "det_auto_safeguard_risky_step": reference.new_tensor(
                1.0 if risky_step else 0.0
            ).detach(),
            "det_auto_safeguard_bad_streak": reference.new_tensor(
                float(self._auto_safeguard_bad_step_streak)
            ).detach(),
            "det_auto_safeguard_good_streak": reference.new_tensor(
                float(self._auto_safeguard_good_step_streak)
            ).detach(),
            "det_auto_safeguard_risk_locality": reference.new_tensor(
                1.0 if risk_locality else 0.0
            ).detach(),
            "det_auto_safeguard_risk_spike": reference.new_tensor(
                1.0 if risk_spike else 0.0
            ).detach(),
            "det_auto_safeguard_risk_cooldown": reference.new_tensor(
                1.0 if risk_cooldown else 0.0
            ).detach(),
            "det_auto_safeguard_local_scale_cap": reference.new_tensor(
                float(local_scale_cap)
            ).detach(),
            "det_auto_safeguard_nonlocal_boost": reference.new_tensor(
                float(nonlocal_boost)
            ).detach(),
        }

        return {
            "active": bool(self._auto_safeguard_active),
            "force_nonlocal": bool(force_nonlocal),
            "nonlocal_min_blend": float(nonlocal_min_blend),
            "nonlocal_weight_boost": float(nonlocal_boost),
            "local_scale_cap": float(local_scale_cap),
            "metrics": out_metrics,
        }

    def _compute_per_depth_adaptive_scale(
        self,
        *,
        depth: int,
        depth_local_loss_value: Optional[torch.Tensor],
        depth_locality_ratio: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        min_scale = min(max(float(self.per_depth_adaptive_min_scale), 0.0), 1.0)
        target_ratio = max(float(self.per_depth_adaptive_locality_target_ratio), 1e-6)
        momentum = float(self.per_depth_adaptive_ema_momentum)

        locality_scale_value = 1.0
        if (
            depth_locality_ratio is not None
            and bool(torch.isfinite(depth_locality_ratio).all().item())
        ):
            ratio_ref = depth_locality_ratio.detach().to(dtype=torch.float32, device="cpu")
            prev_locality_ema = self._per_depth_locality_ema.get(depth)
            if prev_locality_ema is None:
                self._per_depth_locality_ema[depth] = ratio_ref
            else:
                self._per_depth_locality_ema[depth] = (
                    prev_locality_ema * momentum + ratio_ref * (1.0 - momentum)
                )
            ema_locality = self._per_depth_locality_ema[depth]
            locality_scale_value = float(
                (ema_locality / target_ratio).clamp(min=min_scale, max=1.0).item()
            )

        spike_ratio_value = 1.0
        if (
            depth_local_loss_value is not None
            and bool(torch.isfinite(depth_local_loss_value).all().item())
        ):
            detached_loss = depth_local_loss_value.detach().to(dtype=torch.float32, device="cpu")
            prev_loss_ema = self._per_depth_loss_ema.get(depth)
            if prev_loss_ema is None:
                self._per_depth_loss_ema[depth] = detached_loss
            else:
                denom = prev_loss_ema.abs().clamp_min(1e-6)
                spike_ratio_value = float((detached_loss / denom).clamp(min=0.0, max=1e6).item())
                if spike_ratio_value > float(self.per_depth_adaptive_spike_threshold):
                    self._per_depth_cooldown_remaining[depth] = max(
                        0, int(self.per_depth_adaptive_cooldown_steps)
                    )
                    self._per_depth_recovery_remaining[depth] = max(
                        0, int(self.per_depth_adaptive_recovery_steps)
                    )
                self._per_depth_loss_ema[depth] = (
                    prev_loss_ema * momentum + detached_loss * (1.0 - momentum)
                )

        cooldown_remaining = int(self._per_depth_cooldown_remaining.get(depth, 0))
        recovery_remaining = int(self._per_depth_recovery_remaining.get(depth, 0))
        cooldown_active = cooldown_remaining > 0
        if cooldown_active:
            stability_scale_value = min_scale
            self._per_depth_cooldown_remaining[depth] = max(cooldown_remaining - 1, 0)
        elif recovery_remaining > 0:
            total_recovery = max(1, int(self.per_depth_adaptive_recovery_steps))
            recovered_steps = total_recovery - recovery_remaining + 1
            progress = min(max(float(recovered_steps) / float(total_recovery), 0.0), 1.0)
            stability_scale_value = min_scale + (1.0 - min_scale) * progress
            self._per_depth_recovery_remaining[depth] = max(recovery_remaining - 1, 0)
        else:
            stability_scale_value = 1.0

        locality_scale = reference.new_tensor(locality_scale_value).clamp(min=0.0, max=1.0)
        stability_scale = reference.new_tensor(stability_scale_value).clamp(min=0.0, max=1.0)
        final_scale = (locality_scale * stability_scale).clamp(min=min_scale, max=1.0)
        metrics: Dict[str, torch.Tensor] = {
            "scale": final_scale.detach(),
            "locality_scale": locality_scale.detach(),
            "stability_scale": stability_scale.detach(),
            "spike_ratio": reference.new_tensor(spike_ratio_value).detach(),
            "cooldown_active": reference.new_tensor(1.0 if cooldown_active else 0.0).detach(),
        }
        return final_scale, metrics

    def _compute_optimizer_lr_scale(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.optimizer_modulation_enabled:
            return None
        source = self.optimizer_modulation_source

        unified = metrics.get("det_unified_controller_scale")
        per_depth = metrics.get("det_per_depth_scale_mean")

        def _as_scale(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if not torch.is_tensor(value):
                value = reference.new_tensor(float(value))
            if not bool(torch.isfinite(value).all().item()):
                return None
            return value.clamp(min=0.0, max=1.0)

        unified_scale = _as_scale(unified)
        per_depth_scale = _as_scale(per_depth)

        selected: Optional[torch.Tensor] = None
        if source == "unified":
            selected = unified_scale
        elif source == "per_depth":
            selected = per_depth_scale
        else:
            if unified_scale is not None and per_depth_scale is not None:
                selected = torch.minimum(unified_scale, per_depth_scale)
            else:
                selected = unified_scale if unified_scale is not None else per_depth_scale

        if selected is None:
            selected = reference.new_tensor(1.0)

        min_scale = min(max(float(self.optimizer_modulation_min_scale), 0.0), 1.0)
        return selected.clamp(min=min_scale, max=1.0)

    @staticmethod
    def _high_frequency_filter(
        latent: torch.Tensor,
        cutoff_frequency: int,
    ) -> torch.Tensor:
        # latent: [B, M, F]
        if latent.ndim != 3:
            return latent
        frame_count = int(latent.shape[-1])
        cutoff = max(1, min(int(cutoff_frequency), max(frame_count // 2, 1)))
        latent_fft = torch.fft.fft(latent, dim=-1)
        latent_fft[..., :cutoff] = 0
        latent_fft[..., -cutoff:] = 0
        return torch.fft.ifft(latent_fft, dim=-1).real

    def _compute_high_frequency_loss(
        self,
        *,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.high_frequency_loss_enabled or self.high_frequency_loss_weight <= 0.0:
            return None
        if model_pred.ndim != 5 or target.ndim != 5:
            return None
        if model_pred.shape != target.shape:
            return None
        if int(model_pred.shape[2]) < 2:
            return None

        pred = model_pred
        ref = target
        if (
            self.high_frequency_max_timestep >= 0
            and torch.is_tensor(timesteps)
            and timesteps.numel() > 0
        ):
            timestep_vec = timesteps.reshape(-1)
            if timestep_vec.numel() >= pred.shape[0]:
                active = (
                    timestep_vec[: pred.shape[0]].to(device=pred.device)
                    <= float(self.high_frequency_max_timestep)
                )
                if bool(active.any().item()):
                    pred = pred[active]
                    ref = ref[active]
                else:
                    return None

        if pred.shape[0] <= 0:
            return None

        pred_seq = pred.permute(0, 1, 3, 4, 2).reshape(pred.shape[0], -1, pred.shape[2]).to(
            dtype=torch.float32
        )
        ref_seq = ref.permute(0, 1, 3, 4, 2).reshape(ref.shape[0], -1, ref.shape[2]).to(
            dtype=torch.float32
        )
        hf_pred = self._high_frequency_filter(pred_seq, self.high_frequency_cutoff_frequency)
        hf_ref = self._high_frequency_filter(ref_seq, self.high_frequency_cutoff_frequency)
        return F.mse_loss(hf_pred, hf_ref, reduction="mean")

    def _iter_dataset_item_infos(
        self,
        dataset_group: Any,
        max_items: int,
    ) -> Iterable[Any]:
        datasets = getattr(dataset_group, "datasets", None)
        if not isinstance(datasets, Sequence):
            return

        count = 0
        seen_keys: set[str] = set()
        for dataset in datasets:
            batch_manager = getattr(dataset, "batch_manager", None)
            buckets = getattr(batch_manager, "buckets", None)
            if not isinstance(buckets, dict):
                continue
            for bucket in buckets.values():
                if not isinstance(bucket, Sequence):
                    continue
                for item_info in bucket:
                    item_key = getattr(item_info, "item_key", None)
                    if not isinstance(item_key, str) or item_key == "":
                        continue
                    canonical = self._strip_window_suffix(item_key)
                    if canonical in seen_keys:
                        continue
                    seen_keys.add(canonical)
                    yield item_info
                    count += 1
                    if max_items > 0 and count >= max_items:
                        return

    def run_external_tracking_preflight(self, dataset_group: Any) -> Dict[str, float]:
        """Validate external trajectory availability from the active training dataset."""
        if not self.external_tracking_enabled:
            return {}
        if not self.external_tracking_preflight_enabled:
            return {}

        max_items = self.external_tracking_preflight_max_items
        total = 0
        resolved = 0
        valid = 0

        for item_info in self._iter_dataset_item_infos(dataset_group, max_items):
            total += 1
            item_key = getattr(item_info, "item_key", None)
            if not isinstance(item_key, str):
                continue

            trajectory_path = self._resolve_trajectory_path(item_key)
            if self.external_tracking_bind_item_paths:
                setattr(item_info, "det_trajectory_path", trajectory_path or "")

            if trajectory_path is None:
                continue
            resolved += 1

            if self.external_tracking_preflight_validate_tensors:
                trajectory = self._load_trajectory_tensor(trajectory_path)
                if trajectory is None:
                    continue
            valid += 1

        coverage = float(valid) / float(total) if total > 0 else 0.0
        summary: Dict[str, float] = {
            "samples_checked": float(total),
            "paths_resolved": float(resolved),
            "valid_trajectories": float(valid),
            "coverage": float(coverage),
        }

        logger.info(
            "DeT trajectory preflight: checked=%d, resolved=%d, valid=%d, coverage=%.3f.",
            total,
            resolved,
            valid,
            coverage,
        )

        if total == 0:
            message = (
                "DeT trajectory preflight found no dataset samples to inspect; "
                "external tracking may stay inactive."
            )
            if self.external_tracking_preflight_strict:
                raise ValueError(message)
            logger.warning(message)

        if coverage < self.external_tracking_preflight_min_coverage:
            message = (
                "DeT trajectory preflight coverage %.3f is below configured minimum %.3f."
                % (coverage, self.external_tracking_preflight_min_coverage)
            )
            if self.external_tracking_preflight_strict:
                raise ValueError(message)
            logger.warning(message)

        return summary

    @staticmethod
    def _strip_window_suffix(item_key: str) -> str:
        body, ext = os.path.splitext(item_key)
        body = _WINDOW_SUFFIX_RE.sub("", body)
        return f"{body}{ext}"

    @staticmethod
    def _dedupe_paths(paths: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for path in paths:
            normalized = os.path.normpath(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _resolve_trajectory_path(self, item_key: str) -> Optional[str]:
        canonical_item = self._strip_window_suffix(item_key)
        item_dir = os.path.dirname(canonical_item)
        item_stem = os.path.splitext(os.path.basename(canonical_item))[0]

        candidates: List[str] = []
        if self.trajectory_root:
            candidates.append(
                os.path.join(self.trajectory_root, f"{item_stem}{self.trajectory_extension}")
            )
            if self.trajectory_subdir:
                candidates.append(
                    os.path.join(
                        self.trajectory_root,
                        self.trajectory_subdir,
                        f"{item_stem}{self.trajectory_extension}",
                    )
                )

        candidates.append(os.path.join(item_dir, f"{item_stem}{self.trajectory_extension}"))
        if self.trajectory_subdir:
            parent_dir = os.path.dirname(item_dir)
            candidates.append(
                os.path.join(
                    parent_dir,
                    self.trajectory_subdir,
                    f"{item_stem}{self.trajectory_extension}",
                )
            )

        if os.path.basename(item_dir).lower() in {"videos", "video"} and self.trajectory_subdir:
            candidates.append(
                os.path.join(
                    os.path.dirname(item_dir),
                    self.trajectory_subdir,
                    f"{item_stem}{self.trajectory_extension}",
                )
            )

        for candidate in self._dedupe_paths(candidates):
            if os.path.exists(candidate):
                return candidate
        return None

    def _cache_put(self, path: str, value: Optional[torch.Tensor]) -> None:
        if path in self._trajectory_cache:
            return
        self._trajectory_cache[path] = value
        self._trajectory_cache_order.append(path)
        while len(self._trajectory_cache_order) > self.trajectory_cache_size:
            evicted = self._trajectory_cache_order.pop(0)
            self._trajectory_cache.pop(evicted, None)

    def _load_trajectory_tensor(self, trajectory_path: str) -> Optional[torch.Tensor]:
        if trajectory_path in self._trajectory_cache:
            return self._trajectory_cache[trajectory_path]

        if not os.path.exists(trajectory_path):
            self._cache_put(trajectory_path, None)
            return None

        trajectory: Optional[torch.Tensor] = None
        try:
            payload = torch.load(trajectory_path, map_location="cpu")
            if torch.is_tensor(payload):
                trajectory = payload
            elif isinstance(payload, dict):
                for value in payload.values():
                    if torch.is_tensor(value):
                        trajectory = value
                        break
            if trajectory is None:
                self._cache_put(trajectory_path, None)
                return None

            trajectory = trajectory.detach().to(dtype=torch.float32, device="cpu")
            if trajectory.ndim == 4 and trajectory.shape[0] == 1:
                trajectory = trajectory.squeeze(0)
            if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
                self._cache_put(trajectory_path, None)
                return None

            # Heuristic: convert [N, T, C] to [T, N, C] when N appears first.
            if trajectory.shape[0] > trajectory.shape[1]:
                trajectory = trajectory.permute(1, 0, 2).contiguous()

            if trajectory.shape[-1] == 2:
                ones = torch.ones(
                    trajectory.shape[0], trajectory.shape[1], 1, dtype=trajectory.dtype
                )
                trajectory = torch.cat([trajectory, ones], dim=-1)

            self._cache_put(trajectory_path, trajectory)
            return trajectory
        except Exception as exc:
            logger.warning(
                "DeT helper: failed to load trajectory '%s': %s",
                trajectory_path,
                exc,
            )
            self._cache_put(trajectory_path, None)
            return None

    @staticmethod
    def _extract_frame_indices(item_info: Any) -> Optional[List[int]]:
        indices = getattr(item_info, "sampled_frame_indices", None)
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            return None
        parsed: List[int] = []
        for value in indices:
            try:
                parsed.append(int(value))
            except Exception:
                continue
        return parsed if parsed else None

    def _select_trajectory_points(
        self,
        coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_count = coords.shape[1]
        max_points = self.trajectory_max_points
        if max_points <= 0 or point_count <= max_points:
            return coords, visibility

        scores = visibility.mean(dim=0)
        topk = torch.topk(scores, k=max_points, largest=True, sorted=False).indices
        return coords[:, topk, :], visibility[:, topk]

    def _sample_trajectory_for_item(
        self,
        trajectory: torch.Tensor,
        *,
        target_frames: int,
        frame_indices: Optional[Sequence[int]],
        original_size: Optional[Tuple[int, int]],
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if trajectory.ndim != 3 or trajectory.shape[0] < 2 or target_frames < 2:
            return None

        total_frames = int(trajectory.shape[0])
        sampled_ids: torch.Tensor
        if frame_indices:
            valid = [max(0, min(int(v), total_frames - 1)) for v in frame_indices]
            if not valid:
                return None
            sampled_ids = torch.tensor(valid, dtype=torch.long)
            if sampled_ids.numel() > target_frames:
                pick = torch.linspace(
                    0,
                    sampled_ids.numel() - 1,
                    target_frames,
                    dtype=torch.float32,
                ).round().long()
                sampled_ids = sampled_ids.index_select(0, pick)
            elif sampled_ids.numel() < target_frames:
                pad_count = target_frames - sampled_ids.numel()
                pad_value = sampled_ids[-1].view(1).expand(pad_count)
                sampled_ids = torch.cat([sampled_ids, pad_value], dim=0)
        else:
            sampled_ids = torch.linspace(
                0,
                total_frames - 1,
                target_frames,
                dtype=torch.float32,
            ).round().long()

        sampled = trajectory.index_select(0, sampled_ids.to(trajectory.device))
        coords = sampled[..., :2]
        visibility = sampled[..., 2] if sampled.shape[-1] >= 3 else None

        if coords.abs().max().item() > 1.5:
            if (
                isinstance(original_size, tuple)
                and len(original_size) == 2
                and original_size[0] > 0
                and original_size[1] > 0
            ):
                width, height = float(original_size[0]), float(original_size[1])
                coords = coords.clone()
                coords[..., 0] = coords[..., 0] / max(width, 1.0)
                coords[..., 1] = coords[..., 1] / max(height, 1.0)

        coords = coords.clamp(0.0, 1.0)
        if visibility is None:
            visibility = torch.ones(
                coords.shape[0],
                coords.shape[1],
                dtype=coords.dtype,
                device=coords.device,
            )
        else:
            visibility = visibility.clamp(0.0, 1.0)

        coords, visibility = self._select_trajectory_points(coords, visibility)
        if coords.shape[1] == 0:
            return None

        return coords.to(device=device), visibility.to(device=device)

    def _load_trajectory_for_item(
        self,
        item_info: Any,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        item_key = getattr(item_info, "item_key", None)
        if not isinstance(item_key, str) or item_key == "":
            return None

        trajectory_path: Optional[str]
        bound_path = getattr(item_info, "det_trajectory_path", None)
        if isinstance(bound_path, str):
            trajectory_path = bound_path if bound_path else None
        else:
            trajectory_path = self._resolve_trajectory_path(item_key)
            if self.external_tracking_bind_item_paths:
                setattr(item_info, "det_trajectory_path", trajectory_path or "")

        if trajectory_path is None:
            stem = os.path.splitext(os.path.basename(self._strip_window_suffix(item_key)))[0]
            if stem not in self._warned_missing_trajectories:
                logger.warning(
                    "DeT helper: trajectory not found for sample '%s'.", stem
                )
                self._warned_missing_trajectories.add(stem)
            return None

        trajectory = self._load_trajectory_tensor(trajectory_path)
        if trajectory is None:
            if trajectory_path not in self._warned_missing_trajectories:
                logger.warning(
                    "DeT helper: failed to parse trajectory file '%s'.",
                    trajectory_path,
                )
                self._warned_missing_trajectories.add(trajectory_path)
            return None
        return trajectory.to(device=device)

    @staticmethod
    def _coerce_trajectory_tensor(value: Any) -> Optional[torch.Tensor]:
        if not torch.is_tensor(value):
            return None
        tensor = value.detach()
        if tensor.ndim == 4 and int(tensor.shape[0]) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            return None
        if int(tensor.shape[0]) < 2 or int(tensor.shape[-1]) < 2:
            return None
        return tensor

    def _load_trajectory_from_batch(
        self,
        batch: Dict[str, Any],
        sample_idx: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.external_tracking_use_batch_trajectories:
            return None

        trajectories = batch.get("trajectories")
        if torch.is_tensor(trajectories):
            if trajectories.ndim != 4:
                return None
            if sample_idx >= int(trajectories.shape[0]):
                return None
            tensor = self._coerce_trajectory_tensor(trajectories[sample_idx])
            return tensor.to(device=device) if tensor is not None else None

        if isinstance(trajectories, Sequence) and not isinstance(trajectories, (str, bytes)):
            if sample_idx >= len(trajectories):
                return None
            tensor = self._coerce_trajectory_tensor(trajectories[sample_idx])
            return tensor.to(device=device) if tensor is not None else None

        return None

    def _compute_external_tracking_loss(
        self,
        *,
        model_pred: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.external_tracking_enabled or self.external_tracking_loss_weight <= 0:
            return None, {}
        if model_pred.ndim != 5:
            return None, {}

        item_infos = batch.get("item_info")
        has_item_infos = isinstance(item_infos, Sequence) and len(item_infos) > 0
        if not has_item_infos and not self.external_tracking_use_batch_trajectories:
            return None, {}

        pred = model_pred.permute(0, 2, 3, 4, 1).to(dtype=torch.float32)
        batch_size, num_frames, height, width, _ = pred.shape
        if num_frames < 2:
            return None, {}

        timestep_vec: Optional[torch.Tensor] = None
        if torch.is_tensor(timesteps):
            timestep_vec = timesteps.reshape(-1)

        sample_losses: List[torch.Tensor] = []
        active_samples = 0
        active_points = 0
        for sample_idx in range(batch_size):
            if timestep_vec is not None and self.external_tracking_max_timestep >= 0:
                if sample_idx < timestep_vec.numel():
                    timestep_value = float(timestep_vec[sample_idx].item())
                    if timestep_value > float(self.external_tracking_max_timestep):
                        continue

            item_info = (
                item_infos[sample_idx]
                if has_item_infos and sample_idx < len(item_infos)
                else None
            )
            trajectory = self._load_trajectory_from_batch(batch, sample_idx, pred.device)
            if trajectory is None and item_info is not None:
                trajectory = self._load_trajectory_for_item(item_info, pred.device)
            if trajectory is None:
                continue

            frame_indices = self._extract_frame_indices(item_info) if item_info is not None else None
            original_size = (
                getattr(item_info, "original_size", None) if item_info is not None else None
            )
            sampled = self._sample_trajectory_for_item(
                trajectory,
                target_frames=num_frames,
                frame_indices=frame_indices,
                original_size=original_size,
                device=pred.device,
            )
            if sampled is None:
                continue

            coords, visibility = sampled
            point_count = coords.shape[1]
            if point_count <= 0:
                continue
            active_samples += 1
            active_points += int(point_count)

            x_idx = (coords[..., 0] * float(width - 1)).round().long().clamp(0, width - 1)
            y_idx = (coords[..., 1] * float(height - 1)).round().long().clamp(0, height - 1)
            frame_ids = (
                torch.arange(num_frames, device=pred.device)
                .unsqueeze(1)
                .expand(num_frames, point_count)
            )
            trajectory_embeddings = pred[sample_idx, frame_ids, y_idx, x_idx]
            tracking_error = (trajectory_embeddings[1:] - trajectory_embeddings[:-1]).pow(2).mean(dim=-1)

            if self.trajectory_use_visibility:
                visibility_pair = visibility[1:].to(dtype=tracking_error.dtype)
                denom = visibility_pair.sum().clamp_min(1.0)
                sample_loss = (tracking_error * visibility_pair).sum() / denom
            else:
                sample_loss = tracking_error.mean()

            if torch.isfinite(sample_loss):
                sample_losses.append(sample_loss)

        metrics: Dict[str, torch.Tensor] = {}
        ref_tensor = pred.new_tensor(0.0)
        metrics["det_external_tracking_active_samples"] = ref_tensor.new_tensor(
            float(active_samples)
        )
        metrics["det_external_tracking_active_ratio"] = ref_tensor.new_tensor(
            float(active_samples) / max(1.0, float(batch_size))
        )
        metrics["det_external_tracking_active_points"] = ref_tensor.new_tensor(
            float(active_points)
        )

        if active_samples < max(1, self.external_tracking_min_active_samples):
            return None, metrics
        if not sample_losses:
            return None, metrics
        return torch.stack(sample_losses).mean(), metrics

    def _publish_runtime_scales(self, metrics: Dict[str, torch.Tensor]) -> None:
        """Publish detached scalar locality scales for other train-time helpers."""
        try:
            attention_scale = metrics.get("det_attention_locality_scale")
            if attention_scale is not None:
                if torch.is_tensor(attention_scale):
                    value = float(attention_scale.detach().float().mean().item())
                else:
                    value = float(attention_scale)
                if math.isfinite(value):
                    setattr(self.args, "_det_attention_locality_scale", value)

            locality_scale = metrics.get("det_locality_scale")
            if locality_scale is not None:
                if torch.is_tensor(locality_scale):
                    value = float(locality_scale.detach().float().mean().item())
                else:
                    value = float(locality_scale)
                if math.isfinite(value):
                    setattr(self.args, "_det_locality_scale", value)

            unified_scale = metrics.get("det_unified_controller_scale")
            if unified_scale is not None:
                if torch.is_tensor(unified_scale):
                    value = float(unified_scale.detach().float().mean().item())
                else:
                    value = float(unified_scale)
                if math.isfinite(value):
                    setattr(self.args, "_det_unified_controller_scale", value)

            optimizer_scale = metrics.get("det_optimizer_lr_scale")
            if optimizer_scale is not None:
                if torch.is_tensor(optimizer_scale):
                    value = float(optimizer_scale.detach().float().mean().item())
                else:
                    value = float(optimizer_scale)
                if math.isfinite(value):
                    setattr(self.args, "_det_optimizer_lr_scale", value)
        except Exception:
            pass

    def apply_optimizer_lr_modulation_before_step(self, optimizer: Any) -> Optional[float]:
        """Apply temporary LR scaling to DeT optimizer param groups before optimizer.step()."""
        if not self.optimizer_modulation_enabled:
            return None
        if self.optimizer_modulation_target != "det_adapter":
            return None

        raw_scale = getattr(self.args, "_det_optimizer_lr_scale", None)
        if raw_scale is None:
            return None
        try:
            scale = float(raw_scale)
        except Exception:
            return None
        if not math.isfinite(scale):
            return None
        min_scale = min(max(float(self.optimizer_modulation_min_scale), 0.0), 1.0)
        scale = max(min_scale, min(1.0, scale))

        target_optimizer = optimizer
        if hasattr(target_optimizer, "optimizer"):
            target_optimizer = target_optimizer.optimizer
        param_groups = getattr(target_optimizer, "param_groups", None)
        if not isinstance(param_groups, list):
            return None

        applied = False
        for group in param_groups:
            if not isinstance(group, dict):
                continue
            if not bool(group.get("det_adapter_group", False)):
                continue
            lr_value = group.get("lr")
            if lr_value is None:
                continue
            try:
                base_lr = float(lr_value)
            except Exception:
                continue
            group["_det_saved_lr"] = base_lr
            group["lr"] = base_lr * scale
            applied = True

        return scale if applied else None

    def restore_optimizer_lr_after_step(self, optimizer: Any) -> None:
        """Restore original LR values after a temporary DeT modulation step."""
        target_optimizer = optimizer
        if hasattr(target_optimizer, "optimizer"):
            target_optimizer = target_optimizer.optimizer
        param_groups = getattr(target_optimizer, "param_groups", None)
        if not isinstance(param_groups, list):
            return

        for group in param_groups:
            if not isinstance(group, dict):
                continue
            if "_det_saved_lr" not in group:
                continue
            try:
                group["lr"] = float(group.pop("_det_saved_lr"))
            except Exception:
                try:
                    group.pop("_det_saved_lr")
                except Exception:
                    pass

    def compute_loss(
        self,
        *,
        latents: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
        model_pred: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        has_internal_features = bool(self.hook_handles and self.captured_features)
        has_external_tracking = (
            self.external_tracking_enabled
            and self.external_tracking_loss_weight > 0.0
            and batch is not None
            and model_pred is not None
        )
        has_high_frequency = (
            self.high_frequency_loss_enabled
            and self.high_frequency_loss_weight > 0.0
            and model_pred is not None
            and target is not None
        )

        if not has_internal_features and not has_external_tracking and not has_high_frequency:
            self._latest_locality_profile = None
            self._attention_probe_ratio_by_depth.clear()
            if self.optimizer_modulation_enabled:
                try:
                    setattr(self.args, "_det_optimizer_lr_scale", 1.0)
                except Exception:
                    pass
            return None, {}

        controller_sync_applied = self._sync_controller_runtime_state_if_needed(
            global_step=global_step,
            reference=latents,
        )
        controller_sync_world_size = self._get_controller_sync_world_size()

        temporal_components_by_depth: Dict[int, torch.Tensor] = {}
        tracking_components_by_depth: Dict[int, torch.Tensor] = {}
        nonlocal_components_by_depth: Dict[int, torch.Tensor] = {}
        locality_ratio_by_depth: Dict[int, torch.Tensor] = {}
        attention_probe_ratio_by_depth: Dict[int, torch.Tensor] = {}
        locality_profile_distance_by_depth: Dict[int, torch.Tensor] = {}
        locality_ratios: List[torch.Tensor] = []
        collect_locality_profile = self._should_collect_locality_profile(global_step)
        local_active_mask: Optional[torch.Tensor] = None
        if self.local_loss_max_timestep >= 0 and torch.is_tensor(timesteps):
            t_vec = timesteps.reshape(-1)
            if t_vec.numel() > 0:
                local_active_mask = (
                    t_vec.to(device=latents.device)
                    <= float(self.local_loss_max_timestep)
                )

        if has_internal_features:
            for depth, features in list(self.captured_features.items()):
                if features.ndim != 3:
                    continue
                batch_size, token_count, hidden_dim = features.shape
                if batch_size <= 0 or token_count <= 0 or hidden_dim <= 0:
                    continue

                frames, tokens_per_frame = self._infer_temporal_layout(token_count, latents)
                if frames <= 1 or tokens_per_frame <= 0:
                    continue

                usable_tokens = frames * tokens_per_frame
                if usable_tokens <= 0:
                    continue
                sequence = features[:, :usable_tokens, :]
                sequence = sequence.reshape(batch_size, frames, tokens_per_frame, hidden_dim)
                sequence = sequence.to(dtype=torch.float32)
                if local_active_mask is not None:
                    valid = local_active_mask[:batch_size]
                    if bool(valid.any().item()):
                        sequence = sequence[valid]
                    else:
                        continue
                sequence = self._select_topk_tokens(sequence)

                temporal_loss = self._compute_temporal_kernel_loss(sequence)
                if temporal_loss is not None:
                    temporal_components_by_depth[depth] = temporal_loss

                tracking_loss, locality_ratio, locality_profile_distances = (
                    self._compute_dense_tracking_loss(
                        sequence,
                        collect_profile=collect_locality_profile,
                    )
                )
                if tracking_loss is not None:
                    tracking_components_by_depth[depth] = tracking_loss
                if locality_ratio is not None and bool(torch.isfinite(locality_ratio).all().item()):
                    locality_ratio_by_depth[depth] = locality_ratio
                    locality_ratios.append(locality_ratio)
                attention_ratio = self._attention_probe_ratio_by_depth.get(depth)
                if attention_ratio is not None:
                    attention_ratio = attention_ratio.to(
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                    if bool(torch.isfinite(attention_ratio).all().item()):
                        attention_probe_ratio_by_depth[depth] = attention_ratio
                if (
                    collect_locality_profile
                    and locality_profile_distances is not None
                    and locality_profile_distances.numel() > 0
                ):
                    locality_profile_distance_by_depth[depth] = (
                        locality_profile_distances.detach().to(device="cpu")
                    )

                if self.nonlocal_fallback_enabled and self.nonlocal_fallback_loss_weight > 0.0:
                    nonlocal_loss = self._compute_nonlocal_temporal_loss(sequence)
                    if nonlocal_loss is not None:
                        nonlocal_components_by_depth[depth] = nonlocal_loss

        self.captured_features.clear()
        self._attention_probe_ratio_by_depth.clear()
        if collect_locality_profile:
            self._latest_locality_profile = self._build_locality_profile_payload(
                distance_by_depth=locality_profile_distance_by_depth,
                global_step=global_step,
            )
        else:
            self._latest_locality_profile = None

        external_tracking_loss = None
        external_tracking_metrics: Dict[str, torch.Tensor] = {}
        if has_external_tracking and model_pred is not None and batch is not None:
            external_tracking_loss, external_tracking_metrics = (
                self._compute_external_tracking_loss(
                model_pred=model_pred,
                batch=batch,
                timesteps=timesteps,
            )
            )

        high_frequency_loss = None
        if has_high_frequency and model_pred is not None and target is not None:
            high_frequency_loss = self._compute_high_frequency_loss(
                model_pred=model_pred,
                target=target,
                timesteps=timesteps,
            )

        weighted_terms: List[torch.Tensor] = []
        metrics: Dict[str, torch.Tensor] = {}
        if self.controller_sync_enabled:
            metrics["det_controller_sync_enabled"] = latents.new_tensor(1.0).detach()
            metrics["det_controller_sync_applied"] = latents.new_tensor(
                1.0 if controller_sync_applied else 0.0
            ).detach()
            metrics["det_controller_sync_world_size"] = latents.new_tensor(
                float(controller_sync_world_size)
            ).detach()

        if controller_sync_applied:
            locality_ratio_by_depth = self._sync_depth_metric_map_mean(
                locality_ratio_by_depth,
                reference=latents,
            )
            locality_ratios = [
                locality_ratio_by_depth[depth]
                for depth in sorted(locality_ratio_by_depth.keys())
            ]
            if self.controller_sync_include_per_depth and attention_probe_ratio_by_depth:
                attention_probe_ratio_by_depth = self._sync_depth_metric_map_mean(
                    attention_probe_ratio_by_depth,
                    reference=latents,
                )

        locality_scale: Optional[torch.Tensor] = None
        if self.locality_adaptive_weighting_enabled and locality_ratios:
            ratio_mean = torch.stack(locality_ratios).mean()
            if torch.isfinite(ratio_mean):
                ratio_ref = ratio_mean.detach().to(dtype=torch.float32, device="cpu")
                if self._locality_ratio_ema is None:
                    self._locality_ratio_ema = ratio_ref
                else:
                    momentum = float(self.locality_adaptive_ema_momentum)
                    self._locality_ratio_ema = (
                        self._locality_ratio_ema * momentum
                        + ratio_ref * (1.0 - momentum)
                    )
                ema_ratio = self._locality_ratio_ema.to(
                    device=ratio_mean.device, dtype=ratio_mean.dtype
                )
                target_ratio = max(self.locality_adaptive_target_ratio, 1e-6)
                locality_scale = (ema_ratio / target_ratio).clamp(
                    min=self.locality_adaptive_min_scale,
                    max=1.0,
                )
                metrics["det_locality_ratio"] = ema_ratio.detach()
                metrics["det_locality_scale"] = locality_scale.detach()
        attention_probe_ratios = [
            attention_probe_ratio_by_depth[depth]
            for depth in sorted(attention_probe_ratio_by_depth.keys())
        ]
        attention_locality_ratios = (
            attention_probe_ratios if attention_probe_ratios else locality_ratios
        )
        attention_locality_scale, attention_metrics = self._compute_attention_locality_auto_scale(
            locality_ratios=attention_locality_ratios,
            reference=latents,
            global_step=global_step,
        )
        for key, value in attention_metrics.items():
            metrics[key] = value.detach()

        combined_local_scale: Optional[torch.Tensor] = None
        if locality_scale is not None:
            combined_local_scale = locality_scale
        if attention_locality_scale is not None:
            combined_local_scale = (
                attention_locality_scale
                if combined_local_scale is None
                else (combined_local_scale * attention_locality_scale)
            )

        temporal_weighted_base_by_depth: Dict[int, torch.Tensor] = {}
        temporal_sched: Optional[torch.Tensor] = None
        if temporal_components_by_depth and self.temporal_loss_weight > 0:
            temporal_reference = next(iter(temporal_components_by_depth.values()))
            temporal_sched = self._compute_warmup_factor(
                global_step=global_step,
                warmup_steps=self.temporal_loss_warmup_steps,
                device=temporal_reference.device,
                dtype=temporal_reference.dtype,
            )
            for depth, temporal_loss in temporal_components_by_depth.items():
                temporal_weighted_base_by_depth[depth] = (
                    temporal_loss * self.temporal_loss_weight * temporal_sched
                )

        tracking_weighted_base_by_depth: Dict[int, torch.Tensor] = {}
        tracking_sched: Optional[torch.Tensor] = None
        if tracking_components_by_depth and self.tracking_loss_weight > 0:
            tracking_reference = next(iter(tracking_components_by_depth.values()))
            tracking_sched = self._compute_warmup_factor(
                global_step=global_step,
                warmup_steps=self.dense_tracking_loss_warmup_steps,
                device=tracking_reference.device,
                dtype=tracking_reference.dtype,
            )
            for depth, tracking_loss in tracking_components_by_depth.items():
                tracking_weighted_base_by_depth[depth] = (
                    tracking_loss * self.tracking_loss_weight * tracking_sched
                )

        nonlocal_weighted_base_by_depth: Dict[int, torch.Tensor] = {}
        nonlocal_sched: Optional[torch.Tensor] = None
        if nonlocal_components_by_depth and self.nonlocal_fallback_loss_weight > 0.0:
            nonlocal_reference = next(iter(nonlocal_components_by_depth.values()))
            nonlocal_sched = self._compute_warmup_factor(
                global_step=global_step,
                warmup_steps=self.nonlocal_fallback_loss_warmup_steps,
                device=nonlocal_reference.device,
                dtype=nonlocal_reference.dtype,
            )
            for depth, nonlocal_loss in nonlocal_components_by_depth.items():
                nonlocal_weighted_base_by_depth[depth] = (
                    nonlocal_loss * self.nonlocal_fallback_loss_weight * nonlocal_sched
                )

        depth_base_terms: Dict[int, torch.Tensor] = {}
        for depth in sorted(
            set(temporal_weighted_base_by_depth.keys())
            | set(tracking_weighted_base_by_depth.keys())
        ):
            branch_terms: List[torch.Tensor] = []
            temporal_term = temporal_weighted_base_by_depth.get(depth)
            tracking_term = tracking_weighted_base_by_depth.get(depth)
            if temporal_term is not None:
                branch_terms.append(temporal_term)
            if tracking_term is not None:
                branch_terms.append(tracking_term)
            if branch_terms:
                depth_base_terms[depth] = torch.stack(branch_terms).sum()

        controller_depth_loss_by_depth: Dict[int, torch.Tensor] = {
            depth: depth_base.detach()
            for depth, depth_base in depth_base_terms.items()
        }
        if controller_sync_applied and self.controller_sync_include_per_depth:
            controller_depth_loss_by_depth = self._sync_depth_metric_map_mean(
                controller_depth_loss_by_depth,
                reference=latents,
            )

        local_loss_for_controller: Optional[torch.Tensor] = None
        if controller_depth_loss_by_depth:
            local_loss_for_controller = torch.stack(
                list(controller_depth_loss_by_depth.values())
            ).mean()

        local_loss_scale: Optional[torch.Tensor] = combined_local_scale
        if self.unified_controller_enabled:
            unified_locality_scale = self._resolve_unified_controller_locality_scale(
                locality_scale=locality_scale,
                attention_locality_scale=attention_locality_scale,
                reference=latents,
            )
            unified_scale, unified_metrics = self._compute_unified_controller_scale(
                local_loss_value=local_loss_for_controller,
                locality_scale=unified_locality_scale,
                reference=latents,
            )
            local_loss_scale = unified_scale
            for key, value in unified_metrics.items():
                metrics[key] = value.detach()

        safeguard_force_nonlocal = False
        safeguard_nonlocal_min_blend = 0.0
        safeguard_nonlocal_weight_boost = 1.0
        if self.auto_safeguard_enabled:
            safeguard_info = self._compute_auto_safeguard_modulation(
                local_scale=local_loss_scale,
                metrics=metrics,
                reference=latents,
            )
            local_cap = float(safeguard_info["local_scale_cap"])
            cap_tensor = latents.new_tensor(local_cap)
            if local_loss_scale is None:
                local_loss_scale = cap_tensor
            else:
                local_loss_scale = torch.minimum(local_loss_scale, cap_tensor)
            safeguard_force_nonlocal = bool(safeguard_info["force_nonlocal"])
            safeguard_nonlocal_min_blend = float(safeguard_info["nonlocal_min_blend"])
            safeguard_nonlocal_weight_boost = float(
                safeguard_info["nonlocal_weight_boost"]
            )
            for key, value in safeguard_info["metrics"].items():
                metrics[key] = value.detach()

        per_depth_scales: List[torch.Tensor] = []
        per_depth_spike_ratios: List[torch.Tensor] = []
        per_depth_cooldown_active: List[torch.Tensor] = []
        per_depth_weighted_terms: List[torch.Tensor] = []
        temporal_weighted_terms: List[torch.Tensor] = []
        tracking_weighted_terms: List[torch.Tensor] = []
        nonlocal_weighted_terms: List[torch.Tensor] = []
        nonlocal_blends: List[torch.Tensor] = []

        for depth, depth_base in depth_base_terms.items():
            depth_scale = depth_base.new_tensor(1.0)
            depth_scale_metrics: Dict[str, torch.Tensor] = {}
            if self.per_depth_adaptive_enabled:
                depth_scale, depth_scale_metrics = self._compute_per_depth_adaptive_scale(
                    depth=depth,
                    depth_local_loss_value=controller_depth_loss_by_depth.get(depth),
                    depth_locality_ratio=locality_ratio_by_depth.get(depth),
                    reference=depth_base,
                )
                per_depth_scales.append(depth_scale.detach())
                per_depth_spike_ratios.append(depth_scale_metrics["spike_ratio"].detach())
                per_depth_cooldown_active.append(depth_scale_metrics["cooldown_active"].detach())

            combined_depth_scale = depth_scale
            if local_loss_scale is not None:
                combined_depth_scale = combined_depth_scale * local_loss_scale
            weighted_depth = depth_base * combined_depth_scale

            nonlocal_depth = nonlocal_weighted_base_by_depth.get(depth)
            if nonlocal_depth is not None and self.nonlocal_fallback_enabled:
                trigger = min(max(float(self.nonlocal_fallback_trigger_scale), 0.0), 1.0)
                scale_value = float(combined_depth_scale.detach().float().mean().item())
                blend_value = 0.0
                if trigger > 0.0 and scale_value < trigger:
                    blend_value = (trigger - scale_value) / max(trigger, 1e-6)
                    blend_value = max(blend_value, float(self.nonlocal_fallback_min_blend))
                    blend_value = min(max(blend_value, 0.0), 1.0)
                if safeguard_force_nonlocal:
                    blend_value = max(blend_value, safeguard_nonlocal_min_blend)
                if blend_value > 0.0:
                    blend = depth_base.new_tensor(blend_value)
                    weighted_nonlocal = (
                        nonlocal_depth
                        * blend
                        * depth_base.new_tensor(safeguard_nonlocal_weight_boost)
                    )
                    weighted_depth = weighted_depth + weighted_nonlocal
                    nonlocal_weighted_terms.append(weighted_nonlocal)
                    nonlocal_blends.append(blend.detach())
            per_depth_weighted_terms.append(weighted_depth)

            temporal_depth = temporal_weighted_base_by_depth.get(depth)
            if temporal_depth is not None:
                temporal_weighted_terms.append(temporal_depth * combined_depth_scale)
            tracking_depth = tracking_weighted_base_by_depth.get(depth)
            if tracking_depth is not None:
                tracking_weighted_terms.append(tracking_depth * combined_depth_scale)

        if per_depth_weighted_terms:
            weighted_terms.append(torch.stack(per_depth_weighted_terms).mean())
        if temporal_weighted_terms:
            metrics["det_temporal_kernel_loss"] = (
                torch.stack(temporal_weighted_terms).mean().detach()
            )
        if tracking_weighted_terms:
            metrics["det_dense_tracking_loss"] = (
                torch.stack(tracking_weighted_terms).mean().detach()
            )
        if nonlocal_weighted_terms:
            metrics["det_nonlocal_fallback_loss"] = (
                torch.stack(nonlocal_weighted_terms).mean().detach()
            )
        if nonlocal_blends:
            blend_stack = torch.stack(nonlocal_blends)
            metrics["det_nonlocal_fallback_blend_mean"] = blend_stack.mean().detach()
            metrics["det_nonlocal_fallback_active_depths"] = blend_stack.new_tensor(
                float(blend_stack.numel())
            ).detach()
        if temporal_sched is not None:
            metrics["det_schedule_temporal_factor"] = temporal_sched.detach()
        if tracking_sched is not None:
            metrics["det_schedule_tracking_factor"] = tracking_sched.detach()
        if nonlocal_sched is not None:
            metrics["det_schedule_nonlocal_factor"] = nonlocal_sched.detach()
        if self.per_depth_adaptive_enabled and per_depth_scales:
            scale_stack = torch.stack(per_depth_scales)
            cooldown_stack = torch.stack(per_depth_cooldown_active)
            spike_stack = torch.stack(per_depth_spike_ratios)
            metrics["det_per_depth_scale_mean"] = scale_stack.mean().detach()
            metrics["det_per_depth_scale_min"] = scale_stack.min().detach()
            metrics["det_per_depth_scale_max"] = scale_stack.max().detach()
            metrics["det_per_depth_cooldown_active_count"] = cooldown_stack.sum().detach()
            metrics["det_per_depth_spike_ratio_max"] = spike_stack.max().detach()

        optimizer_lr_scale = self._compute_optimizer_lr_scale(
            metrics=metrics,
            reference=latents,
        )
        if optimizer_lr_scale is not None:
            metrics["det_optimizer_lr_scale"] = optimizer_lr_scale.detach()
            metrics["det_optimizer_lr_modulation_active"] = latents.new_tensor(
                1.0 if float(optimizer_lr_scale.item()) < 0.999 else 0.0
            ).detach()

        if external_tracking_loss is not None and self.external_tracking_loss_weight > 0:
            weighted_external = external_tracking_loss * self.external_tracking_loss_weight
            external_sched = self._compute_warmup_factor(
                global_step=global_step,
                warmup_steps=self.external_tracking_loss_warmup_steps,
                device=weighted_external.device,
                dtype=weighted_external.dtype,
            )
            weighted_external = weighted_external * external_sched
            weighted_terms.append(weighted_external)
            metrics["det_external_tracking_loss"] = weighted_external.detach()
            metrics["det_schedule_external_factor"] = external_sched.detach()
        for key, value in external_tracking_metrics.items():
            metrics[key] = value.detach()
        if high_frequency_loss is not None and self.high_frequency_loss_weight > 0:
            weighted_hf = high_frequency_loss * self.high_frequency_loss_weight
            hf_sched = self._compute_warmup_factor(
                global_step=global_step,
                warmup_steps=self.high_frequency_loss_warmup_steps,
                device=weighted_hf.device,
                dtype=weighted_hf.dtype,
            )
            weighted_hf = weighted_hf * hf_sched
            weighted_terms.append(weighted_hf)
            metrics["det_high_frequency_loss"] = weighted_hf.detach()
            metrics["det_schedule_hf_factor"] = hf_sched.detach()

        self._publish_runtime_scales(metrics)
        if not weighted_terms:
            return None, metrics

        reference = weighted_terms[0].detach()
        metrics["det_active_depths"] = reference.new_tensor(float(len(self.alignment_depths)))
        total = torch.stack(weighted_terms).sum()
        return total, metrics
