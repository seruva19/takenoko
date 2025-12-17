"""Advanced logging and metrics collection for training loop."""

import argparse
from typing import Dict, Any, Optional, List
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

logger = logging.getLogger(__name__)

# Advanced metrics tracker (lazy initialization, only loaded if enabled)
_advanced_metrics_tracker = None
_advanced_metrics_initialized = False


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
    if getattr(loss_components, "layer_sync_loss", None) is not None:
        logs["loss/layer_sync"] = float(loss_components.layer_sync_loss.item())
    if loss_components.repa_loss is not None:
        logs["loss/repa"] = float(loss_components.repa_loss.item())
    if getattr(loss_components, "wanvideo_cfm_loss", None) is not None:
        logs["loss/wanvideo_cfm"] = float(
            loss_components.wanvideo_cfm_loss.item()
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
        if getattr(args, 'enable_advanced_metrics', False):
            try:
                from core.advanced_metrics_tracker import AdvancedMetricsTracker

                # Get dual_model_manager from args (if it was stored there)
                dual_model_manager = getattr(args, 'dual_model_manager', None)

                # Parse features from config (default: all features if None)
                feature_list = getattr(args, 'advanced_metrics_features', None)
                features = set(feature_list) if feature_list else None

                _advanced_metrics_tracker = AdvancedMetricsTracker(
                    enabled=True,
                    features=features,
                    max_history=getattr(args, 'advanced_metrics_max_history', 10000),
                    dual_model_manager=dual_model_manager,
                    gradient_watch_threshold=getattr(args, 'gradient_watch_threshold', 0.5),
                    gradient_stability_window=getattr(args, 'gradient_stability_window', 10),
                    convergence_window_sizes=getattr(args, 'convergence_window_sizes', [10, 25, 50, 100]),
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
