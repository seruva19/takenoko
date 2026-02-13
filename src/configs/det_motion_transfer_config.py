"""Configuration parser for DeT-style motion transfer regularization."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

_ALLOWED_KERNEL_MODES = {"avg", "gaussian"}
_ALLOWED_LOCALITY_POLICIES = {"off", "scale", "disable"}
_ALLOWED_SCHEDULE_SHAPES = {"linear", "cosine"}
_ALLOWED_ADAPTER_LOCALITY_SOURCES = {"attention_probe", "locality_adaptive"}
_ALLOWED_UNIFIED_LOCALITY_SOURCES = {"attention_probe", "locality_adaptive", "min"}
_ALLOWED_NONLOCAL_FALLBACK_MODES = {"cosine", "mse"}
_ALLOWED_OPTIMIZER_MODULATION_TARGETS = {"det_adapter"}
_ALLOWED_OPTIMIZER_MODULATION_SOURCES = {"unified", "per_depth", "min"}


def _parse_depth_list(
    config: Dict[str, Any],
    *,
    list_key: str,
    fallback_depth: int,
) -> List[int]:
    raw_depths = config.get(list_key, None)
    if raw_depths is None:
        return [fallback_depth]
    if not isinstance(raw_depths, Sequence) or isinstance(raw_depths, (str, bytes)):
        raise ValueError(
            f"{list_key} must be a list of integers when provided."
        )
    if len(raw_depths) == 0:
        raise ValueError(f"{list_key} must not be empty when provided.")
    parsed = [int(v) for v in raw_depths]
    deduped: List[int] = []
    for depth in parsed:
        if depth not in deduped:
            deduped.append(depth)
    return deduped


def apply_det_motion_transfer_config(
    args: Any,
    config: Dict[str, Any],
    logger: Any,
) -> None:
    """Parse and validate DeT-style training-only motion transfer settings."""

    args.enable_det_motion_transfer = bool(
        config.get("enable_det_motion_transfer", False)
    )
    args.det_alignment_depth = int(config.get("det_alignment_depth", 8))
    args.det_alignment_depths = _parse_depth_list(
        config,
        list_key="det_alignment_depths",
        fallback_depth=args.det_alignment_depth,
    )
    args.det_temporal_kernel_size = int(config.get("det_temporal_kernel_size", 5))
    args.det_temporal_kernel_mode = str(
        config.get("det_temporal_kernel_mode", "avg")
    ).lower()
    args.det_temporal_kernel_loss_weight = float(
        config.get("det_temporal_kernel_loss_weight", 0.1)
    )
    args.det_dense_tracking_loss_weight = float(
        config.get("det_dense_tracking_loss_weight", 0.1)
    )
    args.det_dense_tracking_stride = int(config.get("det_dense_tracking_stride", 1))
    args.det_dense_tracking_topk_tokens = int(
        config.get("det_dense_tracking_topk_tokens", 64)
    )
    args.det_local_loss_max_timestep = int(
        config.get("det_local_loss_max_timestep", -1)
    )
    args.det_detach_temporal_target = bool(
        config.get("det_detach_temporal_target", True)
    )
    args.det_external_tracking_enabled = bool(
        config.get("det_external_tracking_enabled", False)
    )
    args.det_external_tracking_loss_weight = float(
        config.get("det_external_tracking_loss_weight", 0.0)
    )
    args.det_external_tracking_max_timestep = int(
        config.get("det_external_tracking_max_timestep", 400)
    )
    args.det_trajectory_root = str(config.get("det_trajectory_root", "") or "")
    args.det_trajectory_subdir = str(
        config.get("det_trajectory_subdir", "trajectories") or ""
    )
    args.det_trajectory_extension = str(
        config.get("det_trajectory_extension", ".pth") or ".pth"
    )
    if not args.det_trajectory_extension.startswith("."):
        args.det_trajectory_extension = f".{args.det_trajectory_extension}"
    args.det_trajectory_use_visibility = bool(
        config.get("det_trajectory_use_visibility", True)
    )
    args.det_trajectory_max_points = int(config.get("det_trajectory_max_points", 0))
    args.det_trajectory_cache_size = int(config.get("det_trajectory_cache_size", 256))
    args.det_external_tracking_min_active_samples = int(
        config.get("det_external_tracking_min_active_samples", 1)
    )
    args.det_external_tracking_preflight_enabled = bool(
        config.get("det_external_tracking_preflight_enabled", True)
    )
    args.det_external_tracking_preflight_strict = bool(
        config.get("det_external_tracking_preflight_strict", False)
    )
    args.det_external_tracking_preflight_min_coverage = float(
        config.get("det_external_tracking_preflight_min_coverage", 0.7)
    )
    args.det_external_tracking_preflight_max_items = int(
        config.get("det_external_tracking_preflight_max_items", 2048)
    )
    args.det_external_tracking_preflight_validate_tensors = bool(
        config.get("det_external_tracking_preflight_validate_tensors", False)
    )
    args.det_external_tracking_bind_item_paths = bool(
        config.get("det_external_tracking_bind_item_paths", True)
    )
    args.det_external_tracking_use_batch_trajectories = bool(
        config.get("det_external_tracking_use_batch_trajectories", False)
    )
    args.det_locality_adaptive_weighting_enabled = bool(
        config.get("det_locality_adaptive_weighting_enabled", False)
    )
    args.det_locality_adaptive_target_ratio = float(
        config.get("det_locality_adaptive_target_ratio", 0.65)
    )
    args.det_locality_adaptive_min_scale = float(
        config.get("det_locality_adaptive_min_scale", 0.1)
    )
    args.det_locality_adaptive_ema_momentum = float(
        config.get("det_locality_adaptive_ema_momentum", 0.9)
    )
    args.det_attention_locality_probe_enabled = bool(
        config.get("det_attention_locality_probe_enabled", False)
    )
    args.det_attention_locality_probe_interval = int(
        config.get("det_attention_locality_probe_interval", 1)
    )
    args.det_attention_locality_probe_ema_momentum = float(
        config.get("det_attention_locality_probe_ema_momentum", 0.95)
    )
    args.det_attention_locality_probe_min_ratio = float(
        config.get("det_attention_locality_probe_min_ratio", 0.65)
    )
    args.det_attention_locality_auto_policy = str(
        config.get("det_attention_locality_auto_policy", "off")
    ).lower()
    args.det_attention_locality_auto_scale_min = float(
        config.get("det_attention_locality_auto_scale_min", 0.1)
    )
    args.det_attention_locality_disable_threshold = float(
        config.get("det_attention_locality_disable_threshold", 0.35)
    )
    args.det_attention_locality_reenable_threshold = float(
        config.get("det_attention_locality_reenable_threshold", 0.55)
    )
    args.det_locality_profiler_enabled = bool(
        config.get("det_locality_profiler_enabled", False)
    )
    args.det_locality_profiler_interval = int(
        config.get("det_locality_profiler_interval", 200)
    )
    args.det_locality_profiler_bins = int(
        config.get("det_locality_profiler_bins", 16)
    )
    args.det_locality_profiler_max_depths_in_plot = int(
        config.get("det_locality_profiler_max_depths_in_plot", 8)
    )
    args.det_locality_profiler_log_prefix = str(
        config.get("det_locality_profiler_log_prefix", "det_locality_profile")
        or "det_locality_profile"
    )
    args.det_locality_profiler_export_artifacts = bool(
        config.get("det_locality_profiler_export_artifacts", False)
    )
    args.det_locality_profiler_export_dir = str(
        config.get("det_locality_profiler_export_dir", "") or ""
    )
    args.det_controller_sync_enabled = bool(
        config.get("det_controller_sync_enabled", False)
    )
    args.det_controller_sync_interval = int(
        config.get("det_controller_sync_interval", 1)
    )
    args.det_controller_sync_include_per_depth = bool(
        config.get("det_controller_sync_include_per_depth", True)
    )
    args.det_auto_safeguard_enabled = bool(
        config.get("det_auto_safeguard_enabled", False)
    )
    args.det_auto_safeguard_locality_threshold = float(
        config.get("det_auto_safeguard_locality_threshold", 0.45)
    )
    args.det_auto_safeguard_spike_ratio_threshold = float(
        config.get("det_auto_safeguard_spike_ratio_threshold", 1.8)
    )
    args.det_auto_safeguard_bad_step_patience = int(
        config.get("det_auto_safeguard_bad_step_patience", 4)
    )
    args.det_auto_safeguard_recovery_step_patience = int(
        config.get("det_auto_safeguard_recovery_step_patience", 12)
    )
    args.det_auto_safeguard_local_scale_cap = float(
        config.get("det_auto_safeguard_local_scale_cap", 0.0)
    )
    args.det_auto_safeguard_force_nonlocal_fallback = bool(
        config.get("det_auto_safeguard_force_nonlocal_fallback", True)
    )
    args.det_auto_safeguard_nonlocal_min_blend = float(
        config.get("det_auto_safeguard_nonlocal_min_blend", 0.2)
    )
    args.det_auto_safeguard_nonlocal_weight_boost = float(
        config.get("det_auto_safeguard_nonlocal_weight_boost", 1.5)
    )
    args.det_unified_controller_enabled = bool(
        config.get("det_unified_controller_enabled", False)
    )
    args.det_unified_controller_locality_source = str(
        config.get("det_unified_controller_locality_source", "min")
    ).lower()
    args.det_unified_controller_min_scale = float(
        config.get("det_unified_controller_min_scale", 0.1)
    )
    args.det_unified_controller_loss_ema_momentum = float(
        config.get("det_unified_controller_loss_ema_momentum", 0.9)
    )
    args.det_unified_controller_spike_threshold = float(
        config.get("det_unified_controller_spike_threshold", 1.8)
    )
    args.det_unified_controller_cooldown_steps = int(
        config.get("det_unified_controller_cooldown_steps", 20)
    )
    args.det_unified_controller_recovery_steps = int(
        config.get("det_unified_controller_recovery_steps", 100)
    )
    args.det_unified_controller_apply_to_adapter = bool(
        config.get("det_unified_controller_apply_to_adapter", False)
    )
    args.det_per_depth_adaptive_enabled = bool(
        config.get("det_per_depth_adaptive_enabled", False)
    )
    args.det_per_depth_adaptive_locality_target_ratio = float(
        config.get("det_per_depth_adaptive_locality_target_ratio", 0.65)
    )
    args.det_per_depth_adaptive_min_scale = float(
        config.get("det_per_depth_adaptive_min_scale", 0.1)
    )
    args.det_per_depth_adaptive_ema_momentum = float(
        config.get("det_per_depth_adaptive_ema_momentum", 0.9)
    )
    args.det_per_depth_adaptive_spike_threshold = float(
        config.get("det_per_depth_adaptive_spike_threshold", 1.8)
    )
    args.det_per_depth_adaptive_cooldown_steps = int(
        config.get("det_per_depth_adaptive_cooldown_steps", 20)
    )
    args.det_per_depth_adaptive_recovery_steps = int(
        config.get("det_per_depth_adaptive_recovery_steps", 100)
    )
    args.det_nonlocal_fallback_enabled = bool(
        config.get("det_nonlocal_fallback_enabled", False)
    )
    args.det_nonlocal_fallback_loss_weight = float(
        config.get("det_nonlocal_fallback_loss_weight", 0.0)
    )
    args.det_nonlocal_fallback_trigger_scale = float(
        config.get("det_nonlocal_fallback_trigger_scale", 0.6)
    )
    args.det_nonlocal_fallback_min_blend = float(
        config.get("det_nonlocal_fallback_min_blend", 0.0)
    )
    args.det_nonlocal_fallback_stride = int(
        config.get("det_nonlocal_fallback_stride", 1)
    )
    args.det_nonlocal_fallback_mode = str(
        config.get("det_nonlocal_fallback_mode", "cosine")
    ).lower()
    args.det_nonlocal_fallback_loss_warmup_steps = int(
        config.get("det_nonlocal_fallback_loss_warmup_steps", 0)
    )
    args.det_optimizer_modulation_enabled = bool(
        config.get("det_optimizer_modulation_enabled", False)
    )
    args.det_optimizer_modulation_target = str(
        config.get("det_optimizer_modulation_target", "det_adapter")
    ).lower()
    args.det_optimizer_modulation_source = str(
        config.get("det_optimizer_modulation_source", "min")
    ).lower()
    args.det_optimizer_modulation_min_scale = float(
        config.get("det_optimizer_modulation_min_scale", 0.2)
    )
    args.det_loss_schedule_enabled = bool(
        config.get("det_loss_schedule_enabled", False)
    )
    args.det_loss_schedule_shape = str(
        config.get("det_loss_schedule_shape", "linear")
    ).lower()
    args.det_temporal_loss_warmup_steps = int(
        config.get("det_temporal_loss_warmup_steps", 0)
    )
    args.det_dense_tracking_loss_warmup_steps = int(
        config.get("det_dense_tracking_loss_warmup_steps", 0)
    )
    args.det_external_tracking_loss_warmup_steps = int(
        config.get("det_external_tracking_loss_warmup_steps", 0)
    )
    args.det_high_frequency_loss_warmup_steps = int(
        config.get("det_high_frequency_loss_warmup_steps", 0)
    )
    args.det_high_frequency_loss_enabled = bool(
        config.get("det_high_frequency_loss_enabled", False)
    )
    args.det_high_frequency_loss_weight = float(
        config.get("det_high_frequency_loss_weight", 0.0)
    )
    args.det_high_frequency_cutoff_frequency = int(
        config.get("det_high_frequency_cutoff_frequency", 3)
    )
    args.det_high_frequency_max_timestep = int(
        config.get("det_high_frequency_max_timestep", 400)
    )
    args.enable_det_adapter = bool(config.get("enable_det_adapter", False))
    args.det_adapter_alignment_depth = int(config.get("det_adapter_alignment_depth", 8))
    args.det_adapter_alignment_depths = _parse_depth_list(
        config,
        list_key="det_adapter_alignment_depths",
        fallback_depth=args.det_adapter_alignment_depth,
    )
    args.det_adapter_rank = int(config.get("det_adapter_rank", 128))
    args.det_adapter_kernel_size = int(config.get("det_adapter_kernel_size", 3))
    args.det_adapter_gate_init = float(config.get("det_adapter_gate_init", 0.0))
    args.det_adapter_gate_max = float(config.get("det_adapter_gate_max", 0.25))
    args.det_adapter_lr_scale = float(config.get("det_adapter_lr_scale", 1.0))
    args.det_adapter_require_uniform_grid = bool(
        config.get("det_adapter_require_uniform_grid", True)
    )
    args.det_adapter_allow_sparse_attention = bool(
        config.get("det_adapter_allow_sparse_attention", False)
    )
    args.det_adapter_follow_det_locality_scale = bool(
        config.get("det_adapter_follow_det_locality_scale", False)
    )
    args.det_adapter_locality_scale_source = str(
        config.get("det_adapter_locality_scale_source", "attention_probe")
    ).lower()
    args.det_adapter_locality_min_scale = float(
        config.get("det_adapter_locality_min_scale", 0.0)
    )
    args.det_adapter_gate_warmup_enabled = bool(
        config.get("det_adapter_gate_warmup_enabled", False)
    )
    args.det_adapter_gate_warmup_steps = int(
        config.get("det_adapter_gate_warmup_steps", 0)
    )
    args.det_adapter_gate_warmup_shape = str(
        config.get("det_adapter_gate_warmup_shape", "linear")
    ).lower()

    if args.det_temporal_kernel_size <= 0:
        raise ValueError("det_temporal_kernel_size must be > 0")
    if args.det_temporal_kernel_size % 2 == 0:
        raise ValueError("det_temporal_kernel_size must be odd")
    if args.det_temporal_kernel_mode not in _ALLOWED_KERNEL_MODES:
        raise ValueError(
            "det_temporal_kernel_mode must be one of "
            f"{sorted(_ALLOWED_KERNEL_MODES)}, got {args.det_temporal_kernel_mode!r}"
        )
    if args.det_temporal_kernel_loss_weight < 0:
        raise ValueError("det_temporal_kernel_loss_weight must be >= 0")
    if args.det_dense_tracking_loss_weight < 0:
        raise ValueError("det_dense_tracking_loss_weight must be >= 0")
    if args.det_dense_tracking_stride <= 0:
        raise ValueError("det_dense_tracking_stride must be > 0")
    if args.det_dense_tracking_topk_tokens < 0:
        raise ValueError("det_dense_tracking_topk_tokens must be >= 0")
    if args.det_local_loss_max_timestep < -1:
        raise ValueError("det_local_loss_max_timestep must be >= -1")
    if args.det_external_tracking_loss_weight < 0:
        raise ValueError("det_external_tracking_loss_weight must be >= 0")
    if args.det_external_tracking_max_timestep < -1:
        raise ValueError("det_external_tracking_max_timestep must be >= -1")
    if args.det_trajectory_max_points < 0:
        raise ValueError("det_trajectory_max_points must be >= 0")
    if args.det_trajectory_cache_size <= 0:
        raise ValueError("det_trajectory_cache_size must be > 0")
    if args.det_external_tracking_min_active_samples <= 0:
        raise ValueError("det_external_tracking_min_active_samples must be > 0")
    if args.det_trajectory_extension == "":
        raise ValueError("det_trajectory_extension must not be empty")
    if not (0.0 <= args.det_external_tracking_preflight_min_coverage <= 1.0):
        raise ValueError(
            "det_external_tracking_preflight_min_coverage must be in [0, 1]"
        )
    if args.det_external_tracking_preflight_max_items == 0:
        raise ValueError("det_external_tracking_preflight_max_items must be != 0")
    if args.det_locality_adaptive_target_ratio <= 0.0:
        raise ValueError("det_locality_adaptive_target_ratio must be > 0")
    if not (0.0 <= args.det_locality_adaptive_min_scale <= 1.0):
        raise ValueError("det_locality_adaptive_min_scale must be in [0, 1]")
    if not (0.0 <= args.det_locality_adaptive_ema_momentum < 1.0):
        raise ValueError("det_locality_adaptive_ema_momentum must be in [0, 1)")
    if args.det_attention_locality_probe_interval <= 0:
        raise ValueError("det_attention_locality_probe_interval must be > 0")
    if not (0.0 <= args.det_attention_locality_probe_ema_momentum < 1.0):
        raise ValueError("det_attention_locality_probe_ema_momentum must be in [0, 1)")
    if args.det_attention_locality_probe_min_ratio <= 0.0:
        raise ValueError("det_attention_locality_probe_min_ratio must be > 0")
    if args.det_attention_locality_auto_policy not in _ALLOWED_LOCALITY_POLICIES:
        raise ValueError(
            "det_attention_locality_auto_policy must be one of "
            f"{sorted(_ALLOWED_LOCALITY_POLICIES)}"
        )
    if not (0.0 <= args.det_attention_locality_auto_scale_min <= 1.0):
        raise ValueError("det_attention_locality_auto_scale_min must be in [0, 1]")
    if args.det_attention_locality_disable_threshold < 0.0:
        raise ValueError("det_attention_locality_disable_threshold must be >= 0")
    if args.det_attention_locality_reenable_threshold < 0.0:
        raise ValueError("det_attention_locality_reenable_threshold must be >= 0")
    if (
        args.det_attention_locality_auto_policy == "disable"
        and args.det_attention_locality_reenable_threshold
        < args.det_attention_locality_disable_threshold
    ):
        raise ValueError(
            "det_attention_locality_reenable_threshold must be >= det_attention_locality_disable_threshold when auto_policy='disable'"
        )
    if args.det_locality_profiler_interval <= 0:
        raise ValueError("det_locality_profiler_interval must be > 0")
    if args.det_locality_profiler_bins < 4:
        raise ValueError("det_locality_profiler_bins must be >= 4")
    if args.det_locality_profiler_bins > 512:
        raise ValueError("det_locality_profiler_bins must be <= 512")
    if args.det_locality_profiler_max_depths_in_plot <= 0:
        raise ValueError("det_locality_profiler_max_depths_in_plot must be > 0")
    if args.det_locality_profiler_log_prefix.strip() == "":
        raise ValueError("det_locality_profiler_log_prefix must not be empty")
    if args.det_controller_sync_interval <= 0:
        raise ValueError("det_controller_sync_interval must be > 0")
    if not (0.0 <= args.det_auto_safeguard_locality_threshold <= 1.0):
        raise ValueError("det_auto_safeguard_locality_threshold must be in [0, 1]")
    if args.det_auto_safeguard_spike_ratio_threshold <= 1.0:
        raise ValueError("det_auto_safeguard_spike_ratio_threshold must be > 1.0")
    if args.det_auto_safeguard_bad_step_patience <= 0:
        raise ValueError("det_auto_safeguard_bad_step_patience must be > 0")
    if args.det_auto_safeguard_recovery_step_patience <= 0:
        raise ValueError("det_auto_safeguard_recovery_step_patience must be > 0")
    if not (0.0 <= args.det_auto_safeguard_local_scale_cap <= 1.0):
        raise ValueError("det_auto_safeguard_local_scale_cap must be in [0, 1]")
    if not (0.0 <= args.det_auto_safeguard_nonlocal_min_blend <= 1.0):
        raise ValueError("det_auto_safeguard_nonlocal_min_blend must be in [0, 1]")
    if args.det_auto_safeguard_nonlocal_weight_boost < 1.0:
        raise ValueError("det_auto_safeguard_nonlocal_weight_boost must be >= 1.0")
    if (
        args.det_unified_controller_locality_source
        not in _ALLOWED_UNIFIED_LOCALITY_SOURCES
    ):
        raise ValueError(
            "det_unified_controller_locality_source must be one of "
            f"{sorted(_ALLOWED_UNIFIED_LOCALITY_SOURCES)}"
        )
    if not (0.0 <= args.det_unified_controller_min_scale <= 1.0):
        raise ValueError("det_unified_controller_min_scale must be in [0, 1]")
    if not (0.0 <= args.det_unified_controller_loss_ema_momentum < 1.0):
        raise ValueError("det_unified_controller_loss_ema_momentum must be in [0, 1)")
    if args.det_unified_controller_spike_threshold <= 1.0:
        raise ValueError("det_unified_controller_spike_threshold must be > 1.0")
    if args.det_unified_controller_cooldown_steps < 0:
        raise ValueError("det_unified_controller_cooldown_steps must be >= 0")
    if args.det_unified_controller_recovery_steps < 0:
        raise ValueError("det_unified_controller_recovery_steps must be >= 0")
    if args.det_per_depth_adaptive_locality_target_ratio <= 0.0:
        raise ValueError("det_per_depth_adaptive_locality_target_ratio must be > 0")
    if not (0.0 <= args.det_per_depth_adaptive_min_scale <= 1.0):
        raise ValueError("det_per_depth_adaptive_min_scale must be in [0, 1]")
    if not (0.0 <= args.det_per_depth_adaptive_ema_momentum < 1.0):
        raise ValueError("det_per_depth_adaptive_ema_momentum must be in [0, 1)")
    if args.det_per_depth_adaptive_spike_threshold <= 1.0:
        raise ValueError("det_per_depth_adaptive_spike_threshold must be > 1.0")
    if args.det_per_depth_adaptive_cooldown_steps < 0:
        raise ValueError("det_per_depth_adaptive_cooldown_steps must be >= 0")
    if args.det_per_depth_adaptive_recovery_steps < 0:
        raise ValueError("det_per_depth_adaptive_recovery_steps must be >= 0")
    if args.det_nonlocal_fallback_loss_weight < 0.0:
        raise ValueError("det_nonlocal_fallback_loss_weight must be >= 0")
    if not (0.0 <= args.det_nonlocal_fallback_trigger_scale <= 1.0):
        raise ValueError("det_nonlocal_fallback_trigger_scale must be in [0, 1]")
    if not (0.0 <= args.det_nonlocal_fallback_min_blend <= 1.0):
        raise ValueError("det_nonlocal_fallback_min_blend must be in [0, 1]")
    if args.det_nonlocal_fallback_stride <= 0:
        raise ValueError("det_nonlocal_fallback_stride must be > 0")
    if args.det_nonlocal_fallback_mode not in _ALLOWED_NONLOCAL_FALLBACK_MODES:
        raise ValueError(
            "det_nonlocal_fallback_mode must be one of "
            f"{sorted(_ALLOWED_NONLOCAL_FALLBACK_MODES)}"
        )
    if args.det_nonlocal_fallback_loss_warmup_steps < 0:
        raise ValueError("det_nonlocal_fallback_loss_warmup_steps must be >= 0")
    if args.det_optimizer_modulation_target not in _ALLOWED_OPTIMIZER_MODULATION_TARGETS:
        raise ValueError(
            "det_optimizer_modulation_target must be one of "
            f"{sorted(_ALLOWED_OPTIMIZER_MODULATION_TARGETS)}"
        )
    if args.det_optimizer_modulation_source not in _ALLOWED_OPTIMIZER_MODULATION_SOURCES:
        raise ValueError(
            "det_optimizer_modulation_source must be one of "
            f"{sorted(_ALLOWED_OPTIMIZER_MODULATION_SOURCES)}"
        )
    if not (0.0 <= args.det_optimizer_modulation_min_scale <= 1.0):
        raise ValueError("det_optimizer_modulation_min_scale must be in [0, 1]")
    if args.det_loss_schedule_shape not in _ALLOWED_SCHEDULE_SHAPES:
        raise ValueError(
            "det_loss_schedule_shape must be one of "
            f"{sorted(_ALLOWED_SCHEDULE_SHAPES)}"
        )
    if args.det_temporal_loss_warmup_steps < 0:
        raise ValueError("det_temporal_loss_warmup_steps must be >= 0")
    if args.det_dense_tracking_loss_warmup_steps < 0:
        raise ValueError("det_dense_tracking_loss_warmup_steps must be >= 0")
    if args.det_external_tracking_loss_warmup_steps < 0:
        raise ValueError("det_external_tracking_loss_warmup_steps must be >= 0")
    if args.det_high_frequency_loss_warmup_steps < 0:
        raise ValueError("det_high_frequency_loss_warmup_steps must be >= 0")
    if args.det_high_frequency_loss_weight < 0.0:
        raise ValueError("det_high_frequency_loss_weight must be >= 0")
    if args.det_high_frequency_cutoff_frequency <= 0:
        raise ValueError("det_high_frequency_cutoff_frequency must be > 0")
    if args.det_high_frequency_max_timestep < -1:
        raise ValueError("det_high_frequency_max_timestep must be >= -1")
    if not args.det_alignment_depths:
        raise ValueError("det_alignment_depths must contain at least one depth")
    if args.det_adapter_rank <= 0:
        raise ValueError("det_adapter_rank must be > 0")
    if args.det_adapter_kernel_size <= 0:
        raise ValueError("det_adapter_kernel_size must be > 0")
    if args.det_adapter_kernel_size % 2 == 0:
        raise ValueError("det_adapter_kernel_size must be odd")
    if args.det_adapter_gate_max <= 0.0:
        raise ValueError("det_adapter_gate_max must be > 0")
    if args.det_adapter_lr_scale <= 0.0:
        raise ValueError("det_adapter_lr_scale must be > 0")
    if not args.det_adapter_alignment_depths:
        raise ValueError("det_adapter_alignment_depths must contain at least one depth")
    if args.det_adapter_locality_scale_source not in _ALLOWED_ADAPTER_LOCALITY_SOURCES:
        raise ValueError(
            "det_adapter_locality_scale_source must be one of "
            f"{sorted(_ALLOWED_ADAPTER_LOCALITY_SOURCES)}"
        )
    if not (0.0 <= args.det_adapter_locality_min_scale <= 1.0):
        raise ValueError("det_adapter_locality_min_scale must be in [0, 1]")
    if args.det_adapter_gate_warmup_steps < 0:
        raise ValueError("det_adapter_gate_warmup_steps must be >= 0")
    if args.det_adapter_gate_warmup_shape not in _ALLOWED_SCHEDULE_SHAPES:
        raise ValueError(
            "det_adapter_gate_warmup_shape must be one of "
            f"{sorted(_ALLOWED_SCHEDULE_SHAPES)}"
        )
    if args.det_adapter_gate_warmup_enabled and args.det_adapter_gate_warmup_steps <= 0:
        raise ValueError(
            "det_adapter_gate_warmup_enabled=true requires det_adapter_gate_warmup_steps > 0."
        )
    has_external_tracking_weight = (
        args.det_external_tracking_enabled and args.det_external_tracking_loss_weight > 0.0
    )
    has_high_frequency_weight = (
        args.det_high_frequency_loss_enabled and args.det_high_frequency_loss_weight > 0.0
    )
    if args.enable_det_motion_transfer and (
        args.det_temporal_kernel_loss_weight <= 0.0
        and args.det_dense_tracking_loss_weight <= 0.0
        and not has_external_tracking_weight
        and not has_high_frequency_weight
    ):
        raise ValueError(
            "enable_det_motion_transfer=true requires at least one positive loss weight "
            "(det_temporal_kernel_loss_weight, det_dense_tracking_loss_weight, det_external_tracking_loss_weight when det_external_tracking_enabled=true, or det_high_frequency_loss_weight when det_high_frequency_loss_enabled=true)."
        )
    if args.det_external_tracking_enabled and args.det_external_tracking_loss_weight <= 0.0:
        raise ValueError(
            "det_external_tracking_enabled=true requires det_external_tracking_loss_weight > 0."
        )
    if args.det_high_frequency_loss_enabled and args.det_high_frequency_loss_weight <= 0.0:
        raise ValueError(
            "det_high_frequency_loss_enabled=true requires det_high_frequency_loss_weight > 0."
        )
    if args.det_nonlocal_fallback_enabled and args.det_nonlocal_fallback_loss_weight <= 0.0:
        raise ValueError(
            "det_nonlocal_fallback_enabled=true requires det_nonlocal_fallback_loss_weight > 0."
        )
    if args.det_nonlocal_fallback_enabled and (
        args.det_temporal_kernel_loss_weight <= 0.0
        and args.det_dense_tracking_loss_weight <= 0.0
    ):
        raise ValueError(
            "det_nonlocal_fallback_enabled=true requires at least one positive local loss weight "
            "(det_temporal_kernel_loss_weight or det_dense_tracking_loss_weight)."
        )
    if args.det_optimizer_modulation_enabled and not args.enable_det_adapter:
        raise ValueError(
            "det_optimizer_modulation_enabled=true requires enable_det_adapter=true."
        )

    if args.enable_det_motion_transfer:
        logger.info(
            "DeT motion transfer enabled (depths=%s, kernel=%s/%s, temporal_w=%.4f, tracking_w=%.4f, local_max_timestep=%d).",
            args.det_alignment_depths,
            args.det_temporal_kernel_mode,
            args.det_temporal_kernel_size,
            args.det_temporal_kernel_loss_weight,
            args.det_dense_tracking_loss_weight,
            args.det_local_loss_max_timestep,
        )
        if args.det_external_tracking_enabled:
            logger.info(
                "DeT external trajectory tracking enabled (weight=%.4f, max_timestep=%d, root='%s', subdir='%s', ext='%s', max_points=%d, use_visibility=%s, min_active_samples=%d).",
                args.det_external_tracking_loss_weight,
                args.det_external_tracking_max_timestep,
                args.det_trajectory_root,
                args.det_trajectory_subdir,
                args.det_trajectory_extension,
                args.det_trajectory_max_points,
                args.det_trajectory_use_visibility,
                args.det_external_tracking_min_active_samples,
            )
            if args.det_external_tracking_preflight_enabled:
                logger.info(
                    "DeT trajectory preflight enabled (strict=%s, min_coverage=%.3f, max_items=%d, validate_tensors=%s, bind_item_paths=%s).",
                    args.det_external_tracking_preflight_strict,
                    args.det_external_tracking_preflight_min_coverage,
                    args.det_external_tracking_preflight_max_items,
                    args.det_external_tracking_preflight_validate_tensors,
                    args.det_external_tracking_bind_item_paths,
                )
            if args.det_external_tracking_use_batch_trajectories:
                logger.info(
                    "DeT external tracking will consume in-batch trajectories when available."
                )
        if args.det_locality_adaptive_weighting_enabled:
            logger.info(
                "DeT locality-adaptive scaling enabled (target_ratio=%.3f, min_scale=%.3f, ema_momentum=%.3f).",
                args.det_locality_adaptive_target_ratio,
                args.det_locality_adaptive_min_scale,
                args.det_locality_adaptive_ema_momentum,
            )
        if args.det_attention_locality_probe_enabled:
            logger.info(
                "DeT attention-locality probe enabled (self-attention map based; interval=%d, ema_momentum=%.3f, min_ratio=%.3f, auto_policy=%s, auto_scale_min=%.3f, disable_th=%.3f, reenable_th=%.3f).",
                args.det_attention_locality_probe_interval,
                args.det_attention_locality_probe_ema_momentum,
                args.det_attention_locality_probe_min_ratio,
                args.det_attention_locality_auto_policy,
                args.det_attention_locality_auto_scale_min,
                args.det_attention_locality_disable_threshold,
                args.det_attention_locality_reenable_threshold,
            )
        if args.det_locality_profiler_enabled:
            logger.info(
                "DeT locality profiler enabled (interval=%d, bins=%d, max_depths_in_plot=%d, prefix=%s, export=%s, export_dir='%s').",
                args.det_locality_profiler_interval,
                args.det_locality_profiler_bins,
                args.det_locality_profiler_max_depths_in_plot,
                args.det_locality_profiler_log_prefix,
                args.det_locality_profiler_export_artifacts,
                args.det_locality_profiler_export_dir,
            )
        if args.det_controller_sync_enabled:
            logger.info(
                "DeT controller-state sync enabled (interval=%d, include_per_depth=%s).",
                args.det_controller_sync_interval,
                args.det_controller_sync_include_per_depth,
            )
        if args.det_auto_safeguard_enabled:
            logger.info(
                "DeT auto safeguard enabled (locality_threshold=%.3f, spike_threshold=%.3f, bad_patience=%d, recovery_patience=%d, local_scale_cap=%.3f, force_nonlocal=%s, nonlocal_min_blend=%.3f, nonlocal_boost=%.3f).",
                args.det_auto_safeguard_locality_threshold,
                args.det_auto_safeguard_spike_ratio_threshold,
                args.det_auto_safeguard_bad_step_patience,
                args.det_auto_safeguard_recovery_step_patience,
                args.det_auto_safeguard_local_scale_cap,
                args.det_auto_safeguard_force_nonlocal_fallback,
                args.det_auto_safeguard_nonlocal_min_blend,
                args.det_auto_safeguard_nonlocal_weight_boost,
            )
            if (
                args.det_auto_safeguard_force_nonlocal_fallback
                and not args.det_nonlocal_fallback_enabled
            ):
                logger.warning(
                    "det_auto_safeguard_force_nonlocal_fallback=true but det_nonlocal_fallback_enabled=false; safeguard cannot inject non-local branch."
                )
        if args.det_unified_controller_enabled:
            logger.info(
                "DeT unified controller enabled (source=%s, min_scale=%.3f, ema_momentum=%.3f, spike_threshold=%.3f, cooldown_steps=%d, recovery_steps=%d, apply_to_adapter=%s).",
                args.det_unified_controller_locality_source,
                args.det_unified_controller_min_scale,
                args.det_unified_controller_loss_ema_momentum,
                args.det_unified_controller_spike_threshold,
                args.det_unified_controller_cooldown_steps,
                args.det_unified_controller_recovery_steps,
                args.det_unified_controller_apply_to_adapter,
            )
            if (
                args.det_unified_controller_locality_source == "attention_probe"
                and not args.det_attention_locality_probe_enabled
            ):
                logger.warning(
                    "det_unified_controller_enabled=true with source='attention_probe' "
                    "but det_attention_locality_probe_enabled=false; locality source defaults to 1.0."
                )
            if (
                args.det_unified_controller_locality_source == "locality_adaptive"
                and not args.det_locality_adaptive_weighting_enabled
            ):
                logger.warning(
                    "det_unified_controller_enabled=true with source='locality_adaptive' "
                    "but det_locality_adaptive_weighting_enabled=false; locality source defaults to 1.0."
                )
        if args.det_per_depth_adaptive_enabled:
            logger.info(
                "DeT per-depth adaptive gating enabled (target_ratio=%.3f, min_scale=%.3f, ema_momentum=%.3f, spike_threshold=%.3f, cooldown_steps=%d, recovery_steps=%d).",
                args.det_per_depth_adaptive_locality_target_ratio,
                args.det_per_depth_adaptive_min_scale,
                args.det_per_depth_adaptive_ema_momentum,
                args.det_per_depth_adaptive_spike_threshold,
                args.det_per_depth_adaptive_cooldown_steps,
                args.det_per_depth_adaptive_recovery_steps,
            )
        if args.det_nonlocal_fallback_enabled:
            logger.info(
                "DeT non-local fallback enabled (weight=%.4f, trigger_scale=%.3f, min_blend=%.3f, stride=%d, mode=%s, warmup_steps=%d).",
                args.det_nonlocal_fallback_loss_weight,
                args.det_nonlocal_fallback_trigger_scale,
                args.det_nonlocal_fallback_min_blend,
                args.det_nonlocal_fallback_stride,
                args.det_nonlocal_fallback_mode,
                args.det_nonlocal_fallback_loss_warmup_steps,
            )
        if args.det_optimizer_modulation_enabled:
            logger.info(
                "DeT optimizer modulation enabled (target=%s, source=%s, min_scale=%.3f).",
                args.det_optimizer_modulation_target,
                args.det_optimizer_modulation_source,
                args.det_optimizer_modulation_min_scale,
            )
            if (
                args.det_optimizer_modulation_source in {"unified", "min"}
                and not args.det_unified_controller_enabled
            ):
                logger.warning(
                    "det_optimizer_modulation_source includes 'unified' but det_unified_controller_enabled=false; unified scale source defaults to 1.0."
                )
            if (
                args.det_optimizer_modulation_source in {"per_depth", "min"}
                and not args.det_per_depth_adaptive_enabled
            ):
                logger.warning(
                    "det_optimizer_modulation_source includes 'per_depth' but det_per_depth_adaptive_enabled=false; per-depth scale source defaults to 1.0."
                )
        if args.det_loss_schedule_enabled:
            logger.info(
                "DeT loss schedule enabled (shape=%s, warmups: temporal=%d, dense=%d, external=%d, nonlocal=%d, hf=%d).",
                args.det_loss_schedule_shape,
                args.det_temporal_loss_warmup_steps,
                args.det_dense_tracking_loss_warmup_steps,
                args.det_external_tracking_loss_warmup_steps,
                args.det_nonlocal_fallback_loss_warmup_steps,
                args.det_high_frequency_loss_warmup_steps,
            )
        if args.det_high_frequency_loss_enabled:
            logger.info(
                "DeT high-frequency loss enabled (weight=%.4f, cutoff=%d, max_timestep=%d).",
                args.det_high_frequency_loss_weight,
                args.det_high_frequency_cutoff_frequency,
                args.det_high_frequency_max_timestep,
            )
    if args.enable_det_adapter:
        logger.info(
            "Wan-safe DeT adapter enabled (depths=%s, rank=%d, kernel=%d, gate_init=%.4f, gate_max=%.4f, lr_scale=%.4f).",
            args.det_adapter_alignment_depths,
            args.det_adapter_rank,
            args.det_adapter_kernel_size,
            args.det_adapter_gate_init,
            args.det_adapter_gate_max,
            args.det_adapter_lr_scale,
        )
        if args.det_adapter_follow_det_locality_scale:
            logger.info(
                "Wan-safe DeT adapter locality-follow enabled (source=%s, min_scale=%.3f).",
                args.det_adapter_locality_scale_source,
                args.det_adapter_locality_min_scale,
            )
            if (
                args.det_adapter_locality_scale_source == "attention_probe"
                and not args.det_attention_locality_probe_enabled
            ):
                logger.warning(
                    "det_adapter_follow_det_locality_scale is enabled with source='attention_probe' "
                    "but det_attention_locality_probe_enabled=false; adapter locality scaling will stay at 1.0."
                )
            if (
                args.det_adapter_locality_scale_source == "locality_adaptive"
                and not args.det_locality_adaptive_weighting_enabled
            ):
                logger.warning(
                    "det_adapter_follow_det_locality_scale is enabled with source='locality_adaptive' "
                    "but det_locality_adaptive_weighting_enabled=false; adapter locality scaling will stay at 1.0."
                )
        if args.det_adapter_gate_warmup_enabled:
            logger.info(
                "Wan-safe DeT adapter gate warmup enabled (steps=%d, shape=%s).",
                args.det_adapter_gate_warmup_steps,
                args.det_adapter_gate_warmup_shape,
            )
        if (
            args.det_unified_controller_enabled
            and args.det_unified_controller_apply_to_adapter
        ):
            logger.info(
                "Wan-safe DeT adapter will additionally follow det_unified_controller_scale."
            )
