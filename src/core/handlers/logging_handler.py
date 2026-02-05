"""Advanced logging and metrics collection for training loop."""

import argparse
from typing import Dict, Any, Optional, List
import math
import torch
import logging
from common.performance_logger import (
    get_timing_metrics,
    get_model_statistics,
    log_performance_summary,
)
from optimizers.enhanced_logging import (
    get_histogram_data,
    is_supported,
)
from scheduling.timestep_logging import (
    log_live_timestep_distribution,
    log_loss_scatterplot,
)
from utils.lora_weight_stats import (
    initialize_lora_stats_tracker,
    log_lora_weight_histograms,
    get_lora_weight_metrics,
)
from utils.activation_stats import (
    collect_activation_stats,
    get_activation_metrics_after_forward,
)

logger = logging.getLogger(__name__)

# Advanced metrics tracker (lazy initialization, only loaded if enabled)
_advanced_metrics_tracker = None
_advanced_metrics_initialized = False

# LoRA weight stats tracker (lazy initialization)
_lora_stats_tracker = None
_lora_stats_initialized = False

# Activation stats tracker (initialized in wan_network_trainer.py)
_activation_stats_enabled = False


def collect_and_log_training_metrics(
    args: argparse.Namespace,
    accelerator: Any,
    current_loss: float,
    avr_loss: float,
    lr_scheduler: Any,
    lr_descriptions: List[str],
    optimizer: Optional[Any],
    keys_scaled: Optional[int],
    mean_norm: Optional[float],
    maximum_norm: Optional[float],
    ema_loss_value: float,
    ema_loss_debiased: Optional[float],
    network: Any,
    global_step: int,
    per_source_losses: Dict[str, float],
    gradient_norm: Optional[float],
    model_pred: torch.Tensor,
    target: torch.Tensor,
    network_dtype: torch.dtype,
    timesteps: torch.Tensor,
    loss_components: Any,
    noise: torch.Tensor,
    noise_scheduler: Any,
    adaptive_manager: Optional[Any],
    loss_computer: Any,
) -> None:
    """Collect and log all training metrics to TensorBoard and other trackers.

    This function handles the complete logging pipeline including:
    - Basic step logs generation
    - Performance metrics
    - Model statistics
    - Component losses
    - Extra training metrics
    - Attention metrics
    - Adaptive timestep metrics
    - Optimizer histograms
    - Timestep distributions
    """
    if not (accelerator.is_main_process and len(accelerator.trackers) > 0):
        return

    # Import here to avoid circular imports
    from core.handlers.metrics_utils import generate_step_logs

    # Generate basic step logs
    logs = generate_step_logs(
        args,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer,
        keys_scaled,
        mean_norm,
        maximum_norm,
        ema_loss_value,
        ema_loss_debiased,
        network,  # Pass the model for parameter stats
        global_step,  # Pass global_step for parameter stats
        per_source_losses,  # Pass per-source losses
        gradient_norm,  # Pass gradient norm
    )

    # Add performance metrics
    if accelerator.sync_gradients:
        # Get timing metrics
        timing_metrics = get_timing_metrics()
        logs.update(timing_metrics)

        # Get model statistics
        model_stats = get_model_statistics(
            model_pred.to(network_dtype),
            target,
            accelerator.is_main_process,
            timesteps,
            global_step,
        )
        logs.update(model_stats)

        # Log performance summary
        log_performance_summary(global_step, timing_metrics)

    # Attach GGPO metrics if available
    try:
        if hasattr(network, "grad_norms") and hasattr(network, "combined_weight_norms"):
            gn = accelerator.unwrap_model(network).grad_norms()
            wn = accelerator.unwrap_model(network).combined_weight_norms()
            if gn is not None:
                logs["norm/avg_grad_norm"] = float(gn.item())
            if wn is not None:
                logs["norm/avg_combined_norm"] = float(wn.item())
    except Exception:
        pass

    # Attach component losses if available
    base_loss_val = getattr(loss_components, "base_loss", None)
    if base_loss_val is not None:
        logs["loss/mse"] = float(base_loss_val.item())
    if loss_components.dispersive_loss is not None:
        logs["loss/dispersive"] = float(loss_components.dispersive_loss.item())
    if loss_components.dop_loss is not None:
        logs["loss/dop"] = float(loss_components.dop_loss.item())
    if getattr(loss_components, "blank_prompt_loss", None) is not None:
        logs["loss/blank_preservation"] = float(
            loss_components.blank_prompt_loss.item()
        )
    if loss_components.optical_flow_loss is not None:
        logs["loss/optical_flow"] = float(loss_components.optical_flow_loss.item())
    if getattr(loss_components, "video_consistency_distance_loss", None) is not None:
        logs["loss/vcd"] = float(loss_components.video_consistency_distance_loss.item())
    if getattr(loss_components, "layer_sync_loss", None) is not None:
        logs["loss/layer_sync"] = float(loss_components.layer_sync_loss.item())
        logs["layersync_loss"] = float(loss_components.layer_sync_loss.item())
    if getattr(loss_components, "layer_sync_similarity", None) is not None:
        logs["layersync_similarity"] = float(
            loss_components.layer_sync_similarity.item()
        )
    if getattr(loss_components, "internal_guidance_loss", None) is not None:
        logs["loss/internal_guidance"] = float(
            loss_components.internal_guidance_loss.item()
        )
    if getattr(loss_components, "self_transcendence_loss", None) is not None:
        logs["loss/self_transcendence"] = float(
            loss_components.self_transcendence_loss.item()
        )
    if loss_components.repa_loss is not None:
        logs["loss/repa"] = float(loss_components.repa_loss.item())
    if getattr(loss_components, "bfm_semfeat_loss", None) is not None:
        logs["loss/bfm_semfeat"] = float(loss_components.bfm_semfeat_loss.item())
    if getattr(loss_components, "bfm_semfeat_similarity", None) is not None:
        logs["bfm_semfeat_similarity"] = float(
            loss_components.bfm_semfeat_similarity.item()
        )
    if getattr(loss_components, "bfm_frn_loss", None) is not None:
        logs["loss/bfm_frn"] = float(loss_components.bfm_frn_loss.item())
    if getattr(loss_components, "bfm_segment_losses", None):
        for key, value in loss_components.bfm_segment_losses.items():
            if value is not None:
                logs[f"loss/bfm_{key}"] = float(value.item())
    if getattr(loss_components, "reg_align_loss", None) is not None:
        logs["loss/reg_align"] = float(loss_components.reg_align_loss.item())
    if getattr(loss_components, "reg_cls_loss", None) is not None:
        logs["loss/reg_cls"] = float(loss_components.reg_cls_loss.item())
    if getattr(loss_components, "reg_similarity", None) is not None:
        logs["reg_similarity"] = float(loss_components.reg_similarity.item())
    if getattr(loss_components, "haste_attn_loss", None) is not None:
        logs["loss/haste_attn"] = float(loss_components.haste_attn_loss.item())
    if getattr(loss_components, "haste_proj_loss", None) is not None:
        logs["loss/haste_proj"] = float(loss_components.haste_proj_loss.item())
    if getattr(loss_components, "contrastive_attn_loss", None) is not None:
        logs["loss/contrastive_attn"] = float(
            loss_components.contrastive_attn_loss.item()
        )
    if getattr(loss_components, "contrastive_attn_diversity_loss", None) is not None:
        logs["loss/contrastive_attn_diversity"] = float(
            loss_components.contrastive_attn_diversity_loss.item()
        )
    if getattr(loss_components, "contrastive_attn_consistency_loss", None) is not None:
        logs["loss/contrastive_attn_consistency"] = float(
            loss_components.contrastive_attn_consistency_loss.item()
        )
    if getattr(loss_components, "contrastive_attn_subject_overlap_loss", None) is not None:
        logs["loss/contrastive_attn_subject_overlap"] = float(
            loss_components.contrastive_attn_subject_overlap_loss.item()
        )
    if getattr(loss_components, "contrastive_attn_subject_entropy_loss", None) is not None:
        logs["loss/contrastive_attn_subject_entropy"] = float(
            loss_components.contrastive_attn_subject_entropy_loss.item()
        )
    if getattr(loss_components, "contrastive_attn_subject_temporal_loss", None) is not None:
        logs["loss/contrastive_attn_subject_temporal"] = float(
            loss_components.contrastive_attn_subject_temporal_loss.item()
        )
    if getattr(loss_components, "crepa_loss", None) is not None:
        logs["loss/crepa"] = float(loss_components.crepa_loss.item())
        logs["crepa_loss"] = float(loss_components.crepa_loss.item())
    if getattr(loss_components, "crepa_similarity", None) is not None:
        logs["crepa_similarity"] = float(loss_components.crepa_similarity.item())
    if getattr(loss_components, "wanvideo_cfm_loss", None) is not None:
        logs["loss/wanvideo_cfm"] = float(loss_components.wanvideo_cfm_loss.item())
    if getattr(loss_components, "mhc_identity_reg", None) is not None:
        logs["loss/mhc_identity_reg"] = float(
            loss_components.mhc_identity_reg.item()
        )
    if getattr(loss_components, "mhc_entropy_reg", None) is not None:
        logs["loss/mhc_entropy_reg"] = float(
            loss_components.mhc_entropy_reg.item()
        )
    mhc_stats = None
    mhc_log_interval = int(getattr(args, "mhc_mix_log_interval", 100))
    mhc_hist_interval = int(getattr(args, "mhc_mix_histogram_interval", 500))
    mhc_should_log = (
        getattr(args, "network_module", "") == "networks.mhc_lora"
        and global_step % mhc_log_interval == 0
    )
    mhc_should_hist = (
        getattr(args, "network_module", "") == "networks.mhc_lora"
        and global_step % mhc_hist_interval == 0
    )
    if (mhc_should_log or mhc_should_hist) and hasattr(network, "get_mhc_mixing_stats"):
        try:
            mhc_stats = network.get_mhc_mixing_stats()
            if isinstance(mhc_stats, dict) and mhc_should_log:
                logs.update(mhc_stats)
        except Exception:
            mhc_stats = None
    if getattr(loss_components, "memflow_guidance_loss", None) is not None:
        logs["loss/memflow_guidance"] = float(
            loss_components.memflow_guidance_loss.item()
        )

    # Optionally compute extra training metrics periodically
    try:
        if (
            getattr(args, "log_extra_train_metrics", True)
            and (args.train_metrics_interval or 0) > 0
            and (global_step % int(args.train_metrics_interval) == 0)
        ):
            extra_metrics = loss_computer.compute_extra_train_metrics(
                model_pred=model_pred,
                target=target,
                noise=noise,
                timesteps=timesteps,
                noise_scheduler=noise_scheduler,
                accelerator=accelerator,
            )
            if extra_metrics:
                logs.update(extra_metrics)
    except Exception:
        pass

    # Log scalar attention metrics and optional heatmap via helper
    try:
        from common.attention_logging import (
            attach_attention_metrics_and_maybe_heatmap as _attn_log_helper,
        )

        _attn_log_helper(accelerator, args, logs, global_step)
    except Exception:
        pass

    try:
        from utils.tensorboard_utils import (
            apply_direction_hints_to_logs as _adh,
        )

        logs = _adh(args, logs)
    except Exception:
        pass

    # Log adaptive timestep sampling statistics
    if adaptive_manager and adaptive_manager.enabled:
        try:
            adaptive_stats = adaptive_manager.get_stats()

            # Core metrics
            logs.update(
                {
                    "adaptive_timestep/total_important": adaptive_stats.get(
                        "total_important", 0
                    ),
                    "adaptive_timestep/avg_importance": adaptive_stats.get(
                        "avg_importance", 0
                    ),
                    "adaptive_timestep/timestep_coverage": adaptive_stats.get(
                        "timestep_coverage", 0
                    ),
                    "adaptive_timestep/importance_updates": adaptive_stats.get(
                        "importance_updates", 0
                    ),
                }
            )

            # Video-specific metrics
            if (
                hasattr(adaptive_manager, "video_specific_categories")
                and adaptive_manager.video_specific_categories
            ):
                logs.update(
                    {
                        "adaptive_timestep/motion_timesteps": adaptive_stats.get(
                            "motion_timesteps", 0
                        ),
                        "adaptive_timestep/detail_timesteps": adaptive_stats.get(
                            "detail_timesteps", 0
                        ),
                        "adaptive_timestep/temporal_timesteps": adaptive_stats.get(
                            "temporal_timesteps", 0
                        ),
                    }
                )

            # Boundary information
            logs.update(
                {
                    "adaptive_timestep/min_boundary": getattr(
                        adaptive_manager, "min_timestep", 0
                    ),
                    "adaptive_timestep/max_boundary": getattr(
                        adaptive_manager, "max_timestep", 1000
                    ),
                    "adaptive_timestep/boundary_range": getattr(
                        adaptive_manager, "max_timestep", 1000
                    )
                    - getattr(adaptive_manager, "min_timestep", 0),
                }
            )

            # Warmup progress
            warmup_remaining = adaptive_stats.get("warmup_remaining", 0)
            warmup_steps = getattr(adaptive_manager, "warmup_steps", 500)
            if warmup_remaining > 0:
                warmup_progress = (
                    max(0.0, 1.0 - (warmup_remaining / warmup_steps))
                    if warmup_steps > 0
                    else 1.0
                )
                logs["adaptive_timestep/warmup_progress"] = warmup_progress
            else:
                logs["adaptive_timestep/warmup_progress"] = 1.0

        except Exception as e:
            logger.debug(f"Error logging adaptive timestep metrics: {e}")

    accelerator.log(logs, step=global_step)

    # mHC mixing histograms and health warnings
    if mhc_stats is not None and mhc_should_log:
        try:
            warn_entropy_min = float(getattr(args, "mhc_mix_warn_entropy_min", 0.05))
            warn_offdiag_max = float(getattr(args, "mhc_mix_warn_offdiag_max", 0.5))
            entropy_val = float(mhc_stats.get("mhc/mixing_entropy", math.nan))
            offdiag_val = float(mhc_stats.get("mhc/mixing_offdiag_mean", math.nan))
            if not math.isnan(entropy_val) and entropy_val < warn_entropy_min:
                logger.warning(
                    "mHC mixing entropy low: %.4f < %.4f",
                    entropy_val,
                    warn_entropy_min,
                )
            if not math.isnan(offdiag_val) and offdiag_val > warn_offdiag_max:
                logger.warning(
                    "mHC off-diagonal mixing high: %.4f > %.4f",
                    offdiag_val,
                    warn_offdiag_max,
                )
        except Exception:
            pass
        if mhc_should_hist and hasattr(network, "get_mhc_mixing_histogram"):
            try:
                hist_data = network.get_mhc_mixing_histogram()
                if isinstance(hist_data, torch.Tensor) and hist_data.numel() > 0:
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            tracker.writer.add_histogram(
                                "mhc/mixing_weights",
                                hist_data.detach().cpu(),
                                global_step,
                            )
                            break
            except Exception:
                pass

    # Enhanced optimizer-specific histogram logging
    if optimizer is not None and is_supported(optimizer):
        try:
            histogram_data = get_histogram_data(optimizer)
            if histogram_data:
                metric_name, tensor_data = histogram_data
                # Make sure tensor_data is not empty
                if tensor_data.numel() > 0:
                    # Log histogram directly to TensorBoard writer
                    # TensorBoard expects histograms to be logged via add_histogram
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            tracker.writer.add_histogram(
                                metric_name, tensor_data, global_step
                            )
                            break
        except Exception as e:
            logger.debug(f"Failed to log enhanced optimizer histogram: {e}")

    # Periodic live histogram of used timesteps (1..1000)
    try:
        log_live_timestep_distribution(accelerator, args, timesteps, global_step)
    except Exception:
        pass

    # Periodic loss-vs-timestep scatter figure
    try:
        log_loss_scatterplot(
            accelerator,
            args,
            timesteps,
            model_pred,
            target,
            global_step,
        )
    except Exception:
        pass

    # Advanced metrics tracking (lazy initialization, zero overhead when disabled)
    global _advanced_metrics_tracker, _advanced_metrics_initialized

    if not _advanced_metrics_initialized:
        _advanced_metrics_initialized = True

        # Only initialize if enabled in config
        if getattr(args, "enable_advanced_metrics", False):
            try:
                from core.advanced_metrics_tracker import AdvancedMetricsTracker

                # Get dual_model_manager from args (if it was stored there)
                dual_model_manager = getattr(args, "dual_model_manager", None)

                # Parse features from config (default: all features if None)
                feature_list = getattr(args, "advanced_metrics_features", None)
                features = set(feature_list) if feature_list else None

                _advanced_metrics_tracker = AdvancedMetricsTracker(
                    enabled=True,
                    features=features,
                    max_history=getattr(args, "advanced_metrics_max_history", 10000),
                    dual_model_manager=dual_model_manager,
                    gradient_watch_threshold=getattr(
                        args, "gradient_watch_threshold", 0.5
                    ),
                    gradient_stability_window=getattr(
                        args, "gradient_stability_window", 10
                    ),
                    convergence_window_sizes=getattr(
                        args, "convergence_window_sizes", [10, 25, 50, 100]
                    ),
                )

                logger.info(
                    f"Advanced metrics enabled with features: {_advanced_metrics_tracker.features}"
                )
            except Exception as e:
                logger.debug(f"Advanced metrics initialization failed, disabling: {e}")
                _advanced_metrics_tracker = None

    # Track and log advanced metrics if initialized
    if _advanced_metrics_tracker is not None:
        try:
            advanced_logs = _advanced_metrics_tracker.track_step(
                step=global_step,
                loss=current_loss,
                gradient_norm=gradient_norm,
            )

            if advanced_logs:
                # Merge advanced metrics into logs dict for this step
                logs.update(advanced_logs)
                # Log the updated metrics
                accelerator.log(advanced_logs, step=global_step)
        except Exception as e:
            logger.debug(f"Advanced metrics tracking error: {e}")

    # LoRA weight statistics tracking (lazy initialization)
    global _lora_stats_tracker, _lora_stats_initialized

    if not _lora_stats_initialized:
        _lora_stats_initialized = True

        if getattr(args, "log_lora_weight_histograms", False):
            try:
                _lora_stats_tracker = initialize_lora_stats_tracker(
                    log_interval=getattr(args, "lora_weight_histogram_interval", 100),
                    log_separate_matrices=getattr(
                        args, "lora_weight_log_separate_matrices", False
                    ),
                )
                # Initialize baseline stats
                _lora_stats_tracker.initialize_baseline(network)
                logger.info(
                    f"LoRA weight histogram logging enabled "
                    f"(interval: {_lora_stats_tracker.log_interval})"
                )
            except Exception as e:
                logger.debug(f"LoRA weight stats initialization failed: {e}")
                _lora_stats_tracker = None

    # Log LoRA weight histograms and scalar metrics
    if _lora_stats_tracker is not None:
        try:
            # Log histograms to TensorBoard
            log_lora_weight_histograms(
                accelerator, network, global_step, _lora_stats_tracker
            )

            # Add scalar metrics to logs (always, not just at histogram interval)
            if global_step % 10 == 0:  # Log scalars more frequently
                lora_metrics = get_lora_weight_metrics(network, _lora_stats_tracker)
                if lora_metrics:
                    accelerator.log(lora_metrics, step=global_step)
        except Exception as e:
            logger.debug(f"LoRA weight stats logging error: {e}")

    # Log activation stats (hooks registered in wan_network_trainer.py)
    # Stats are collected during forward pass via hooks, retrieve them here
    try:
        activation_metrics = get_activation_metrics_after_forward()
        if activation_metrics:
            accelerator.log(activation_metrics, step=global_step)
    except Exception as e:
        logger.debug(f"Activation stats logging error: {e}")
