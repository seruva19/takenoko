"""Metrics utilities for training.

This module centralizes metric computations and logging helpers used by the
training loop to keep `training_core.py` lean while preserving functionality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import torch

from common.logger import get_logger

# Enhanced optimizer logging
from optimizers.enhanced_logging import (
    get_enhanced_metrics,
    is_supported,
)


logger = get_logger(__name__)


class ThroughputTracker:
    """Track training throughput metrics for samples per second and steps per second."""

    def __init__(self, window_size: int = 100):
        """Initialize throughput tracker.

        Args:
            window_size: Number of recent measurements to keep for averaging
        """
        self.window_size = window_size
        self.step_times: List[float] = []
        self.batch_sizes: List[int] = []
        self.start_time = time.perf_counter()
        self.last_step_time = self.start_time

    def record_step(self, batch_size: int) -> None:
        """Record a training step completion.

        Args:
            batch_size: Number of samples in the current batch
        """
        current_time = time.perf_counter()
        step_duration = current_time - self.last_step_time

        self.step_times.append(step_duration)
        self.batch_sizes.append(batch_size)

        # Keep only recent measurements
        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)
            self.batch_sizes.pop(0)

        self.last_step_time = current_time

    def get_throughput_metrics(self) -> Dict[str, float]:
        """Calculate current throughput metrics.

        Returns:
            Dictionary containing samples_per_sec and steps_per_sec
        """
        if not self.step_times:
            return {"samples_per_sec": 0.0, "steps_per_sec": 0.0}

        # Calculate average step time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

        # Calculate average batch size
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
        samples_per_sec = steps_per_sec * avg_batch_size

        return {
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "avg_step_time": avg_step_time,
            "avg_batch_size": avg_batch_size,
        }

    def get_total_runtime(self) -> float:
        """Get total runtime since initialization.

        Returns:
            Total runtime in seconds
        """
        return time.perf_counter() - self.start_time


# Global throughput tracker instance
_throughput_tracker = ThroughputTracker()


def initialize_throughput_tracker(window_size: int = 100) -> None:
    """Initialize the global throughput tracker with a specific window size.

    Args:
        window_size: Number of recent measurements to keep for averaging
    """
    global _throughput_tracker
    _throughput_tracker = ThroughputTracker(window_size)


def record_training_step(batch_size: int) -> None:
    """Record a training step for throughput calculation.

    Args:
        batch_size: Number of samples in the current batch
    """
    _throughput_tracker.record_step(batch_size)


def get_throughput_metrics() -> Dict[str, float]:
    """Get current throughput metrics.

    Returns:
        Dictionary containing throughput metrics
    """
    return _throughput_tracker.get_throughput_metrics()


def get_total_runtime() -> float:
    """Get total training runtime.

    Returns:
        Total runtime in seconds
    """
    return _throughput_tracker.get_total_runtime()


def generate_parameter_stats(
    model: Any,
    global_step: int,
    log_every_n_steps: int = 100,
    max_params_to_log: int = 20,
) -> Dict[str, float]:
    """Generate parameter statistics for weight drift and gradient monitoring.

    Args:
        model: Model to analyze. Must implement `named_parameters()`.
        global_step: Current training step.
        log_every_n_steps: Log at most every N steps.
        max_params_to_log: Number of largest-norm parameters to log.

    Returns:
        Mapping of metric name to value.
    """
    # Only log periodically to avoid TensorBoard spam
    if getattr(generate_parameter_stats, "_last_log_step", -1) >= 0:
        if (
            global_step - getattr(generate_parameter_stats, "_last_log_step")
            < log_every_n_steps
        ):
            return {}

    setattr(generate_parameter_stats, "_last_log_step", global_step)
    param_stats: Dict[str, float] = {}

    try:
        param_info: List[Dict[str, float]] = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                param_norm = param.data.norm().item()
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                param_info.append(
                    {
                        "name": name,
                        "param_norm": float(param_norm),
                        "grad_norm": float(grad_norm),
                        "size": float(param.numel()),
                    }
                )

        param_info.sort(key=lambda x: x["param_norm"], reverse=True)
        top_params = param_info[:max_params_to_log]

        for info in top_params:
            name_str = str(info["name"])  # ensure string
            clean_name = name_str.replace(".", "/")
            param_stats[f"param_norm/{clean_name}"] = float(info["param_norm"])
            param_stats[f"grad_norm/{clean_name}"] = float(info["grad_norm"])

        if param_info:
            total_param_norm = float(sum(info["param_norm"] for info in param_info))
            total_grad_norm = float(sum(info["grad_norm"] for info in param_info))
            avg_param_norm = total_param_norm / len(param_info)
            avg_grad_norm = total_grad_norm / len(param_info)

            param_stats.update(
                {
                    "param_stats/total_param_norm": total_param_norm,
                    "param_stats/avg_param_norm": avg_param_norm,
                    "param_stats/total_grad_norm": total_grad_norm,
                    "param_stats/avg_grad_norm": avg_grad_norm,
                    "param_stats/num_params": float(len(param_info)),
                    "param_stats/largest_param_norm": (
                        float(top_params[0]["param_norm"]) if top_params else 0.0
                    ),
                    "param_stats/largest_grad_norm": float(
                        max(info["grad_norm"] for info in param_info)
                    ),
                }
            )

    except Exception as err:
        logger.warning(f"Failed to generate parameter statistics: {err}")

    return param_stats


def compute_per_source_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict[str, Any],
    weighting: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute loss per data source (video vs image) if source information is available."""
    per_source_losses: Dict[str, float] = {}

    try:
        source_key: Optional[str] = None
        for key in [
            "source",
            "data_source",
            "source_type",
            "video_source",
            "media_type",
        ]:
            if key in batch:
                source_key = key
                break

        if source_key is None:
            # Try to infer from existing batch structure
            sources: Optional[List[str]]
            if "item_info" in batch:
                try:
                    item_infos = batch["item_info"]
                    sources = []
                    for item_info in item_infos:
                        if (
                            hasattr(item_info, "frame_count")
                            and item_info.frame_count
                            and item_info.frame_count > 1
                        ):
                            sources.append("video")
                        else:
                            sources.append("image")
                except Exception:
                    sources = None
            elif "latents" in batch:
                try:
                    latents = batch["latents"]
                    if latents.dim() == 5:
                        frame_counts = latents.shape[2]
                        sources = [
                            "video" if frame_counts > 1 else "image"
                        ] * latents.shape[0]
                    else:
                        sources = ["image"] * latents.shape[0]
                except Exception:
                    sources = None
            elif any("video" in str(k).lower() for k in batch.keys()):
                has_video_keys = any(
                    "video" in str(k).lower() or "frame" in str(k).lower()
                    for k in batch.keys()
                )
                has_image_keys = any(
                    "image" in str(k).lower() or "img" in str(k).lower()
                    for k in batch.keys()
                )
                if has_video_keys and not has_image_keys:
                    sources = ["video"] * model_pred.shape[0]
                elif has_image_keys and not has_video_keys:
                    sources = ["image"] * model_pred.shape[0]
                else:
                    sources = None
            else:
                sources = None

            if sources is None:
                return {}
        else:
            sources = batch[source_key]
            if torch.is_tensor(sources):
                sources = sources.cpu().tolist()

        assert sources is not None
        unique_sources = list(set(sources))
        for source in unique_sources:
            if isinstance(sources, list):
                indices = [i for i, s in enumerate(sources) if s == source]
            else:
                # In practice we always convert to list above
                indices = torch.where(sources == source)[0]

            if len(indices) == 0:
                continue

            source_pred = model_pred[indices]
            source_target = target[indices]

            source_loss = torch.nn.functional.mse_loss(
                source_pred, source_target, reduction="none"
            )

            if sample_weights is not None:
                source_sample_weights = sample_weights[indices]
                while source_sample_weights.dim() < source_loss.dim():
                    source_sample_weights = source_sample_weights.unsqueeze(-1)
                source_loss = source_loss * source_sample_weights

            if weighting is not None:
                source_weighting = weighting[indices]
                source_loss = source_loss * source_weighting

            per_source_losses[f"loss/{source}"] = float(source_loss.mean().item())

    except Exception as err:
        logger.debug(f"Could not compute per-source loss: {err}")

    return per_source_losses


def compute_gradient_norm(
    model: Any,
    max_norm: Optional[float] = None,  # kept for API compatibility
    norm_type: float = 2.0,
) -> float:
    """Compute gradient norm for monitoring gradient flow."""
    try:
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return float(total_norm.item())
    except Exception as err:
        logger.debug(f"Could not compute gradient norm: {err}")
        return 0.0


def generate_step_logs(
    args: Any,
    current_loss: float,
    avr_loss: float,
    lr_scheduler: Any,
    lr_descriptions: Optional[List[str]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    keys_scaled: Optional[int] = None,
    mean_norm: Optional[float] = None,
    maximum_norm: Optional[float] = None,
    ema_loss: Optional[float] = None,
    model: Optional[Any] = None,
    global_step: Optional[int] = None,
    per_source_losses: Optional[Dict[str, float]] = None,
    gradient_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate scalar logs for a training step."""
    network_train_unet_only = True
    logs: Dict[str, Any] = {"loss/current": current_loss, "loss/average": avr_loss}

    # Add throughput metrics if enabled
    if getattr(args, "log_throughput_metrics", True):
        throughput_metrics = get_throughput_metrics()
        logs.update(
            {
                "train/samples_per_sec": throughput_metrics["samples_per_sec"],
                "train/steps_per_second": throughput_metrics["steps_per_sec"],
                "train/runtime": get_total_runtime(),
            }
        )

    if ema_loss is not None:
        logs["loss/ema"] = ema_loss

    if per_source_losses is not None:
        logs.update(per_source_losses)

    if gradient_norm is not None:
        logs["grad_norm"] = gradient_norm

    # Include max-norm regularization statistics if provided
    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/average_key_norm"] = mean_norm
        logs["max_norm/max_key_norm"] = maximum_norm

    if (
        hasattr(args, "log_param_stats")
        and args.log_param_stats
        and model is not None
        and global_step is not None
    ):
        param_stats_interval = getattr(args, "param_stats_every_n_steps", 100)
        max_params = getattr(args, "max_param_stats_logged", 20)
        param_stats = generate_parameter_stats(
            model, global_step, param_stats_interval, max_params
        )
        logs.update(param_stats)

    # Try to resolve the underlying optimizer to read param group fields like 'd'
    actual_optimizer: Optional[torch.optim.Optimizer]
    actual_optimizer = None
    if optimizer is not None:
        actual_optimizer = (
            optimizer.optimizer  # type: ignore[attr-defined]
            if hasattr(optimizer, "optimizer")
            else optimizer
        )
    elif hasattr(lr_scheduler, "optimizer"):
        try:
            actual_optimizer = getattr(lr_scheduler, "optimizer")
        except Exception:
            actual_optimizer = None
    elif hasattr(lr_scheduler, "optimizers"):
        try:
            actual_optimizer = lr_scheduler.optimizers[-1]
        except Exception:
            actual_optimizer = None

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                lr_desc = f"group{idx}" if len(lrs) > 2 else "unet"
        logs[f"lr/{lr_desc}"] = lr

        if (
            args.optimizer_type.lower().startswith("DAdapt".lower())
            or args.optimizer_type.lower() == "Prodigy".lower()
        ):
            try:
                if actual_optimizer is not None:
                    d_val = actual_optimizer.param_groups[i].get("d", None)
                    if d_val is not None:
                        logs[f"lr/d/{lr_desc}"] = d_val
                        logs[f"lr/d*lr/{lr_desc}"] = (
                            d_val * actual_optimizer.param_groups[i]["lr"]
                        )
            except Exception:
                pass
        if (
            args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
            and optimizer is not None
        ):
            actual_optimizer = (
                optimizer.optimizer  # type: ignore[attr-defined]
                if hasattr(optimizer, "optimizer")
                else optimizer
            )
            logs["lr/d*lr"] = (
                actual_optimizer.param_groups[0]["d"]
                * actual_optimizer.param_groups[0]["lr"]
            )
    else:
        idx = 0
        if not network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1
        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if (
                args.optimizer_type.lower().startswith("DAdapt".lower())
                or args.optimizer_type.lower() == "Prodigy".lower()
            ):
                try:
                    if actual_optimizer is not None:
                        d_val = actual_optimizer.param_groups[i].get("d", None)
                        if d_val is not None:
                            logs[f"lr/d/group{i}"] = d_val
                            logs[f"lr/d*lr/group{i}"] = (
                                d_val * actual_optimizer.param_groups[i]["lr"]
                            )
                except Exception:
                    pass
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
                and optimizer is not None
            ):
                actual_optimizer = (
                    optimizer.optimizer  # type: ignore[attr-defined]
                    if hasattr(optimizer, "optimizer")
                    else optimizer
                )
                logs[f"lr/d*lr/group{i}"] = (
                    actual_optimizer.param_groups[i]["d"]
                    * actual_optimizer.param_groups[i]["lr"]
                )

    if optimizer is not None and is_supported(optimizer):
        try:
            enhanced_metrics = get_enhanced_metrics(optimizer)
            logs.update(enhanced_metrics)
        except Exception as err:
            logger.debug(f"Failed to log enhanced optimizer metrics: {err}")

    return logs


def generate_safe_progress_metrics(
    args: Any,
    current_loss: float,
    avr_loss: float,
    lr_scheduler: Any,
    epoch: int,
    global_step: int,
    keys_scaled: Optional[int] = None,
    mean_norm: Optional[float] = None,
    maximum_norm: Optional[float] = None,
    current_step_in_epoch: Optional[int] = None,
    total_steps_in_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate safe progress bar metrics that won't interfere with training."""
    try:
        metrics: Dict[str, Any] = {
            "loss": f"{current_loss:.4f}",
            "avg": f"{avr_loss:.4f}",
        }

        if current_step_in_epoch is not None and total_steps_in_epoch is not None:
            try:
                steps_remaining = total_steps_in_epoch - current_step_in_epoch
                if steps_remaining >= 0:
                    metrics["left"] = f"{steps_remaining} (ep{epoch + 1})"
            except (TypeError, ValueError):
                pass

        try:
            lrs = lr_scheduler.get_last_lr()
            if lrs and len(lrs) > 0:
                metrics["lr"] = f"{lrs[0]:.1e}"
        except (AttributeError, IndexError, TypeError):
            pass

        try:
            if hasattr(args, "max_train_steps") and args.max_train_steps > 0:
                _ = (global_step / args.max_train_steps) * 100
        except (AttributeError, ZeroDivisionError, TypeError):
            pass

        try:
            if avr_loss > 0:
                _ = current_loss / avr_loss
        except (ZeroDivisionError, TypeError, ValueError):
            pass

        if getattr(args, "scale_weight_norms", False) and keys_scaled is not None:
            try:
                metrics["scaled"] = str(keys_scaled)
                if mean_norm is not None:
                    metrics["norm"] = f"{mean_norm:.3f}"
                if maximum_norm is not None:
                    metrics["max_norm"] = f"{maximum_norm:.3f}"
            except (TypeError, ValueError):
                pass

        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                torch.cuda.reset_peak_memory_stats(device)
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
                if peak_allocated > 0.1:
                    metrics["peak"] = f"{peak_allocated:.2f} GiB"
                try:
                    import pynvml  # type: ignore

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["peak"] = f"{float(meminfo.used)/1024**3:.2f} GiB"
                    metrics["util"] = f"{utilization.gpu}%"
                    memory_util = (float(meminfo.used) / float(meminfo.total)) * 100
                    metrics["mem_util"] = f"{memory_util:.1f}%"
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except (RuntimeError, AttributeError):
            pass

        return metrics
    except Exception:
        return {
            "loss": f"{current_loss:.4f}",
            "avg": f"{avr_loss:.4f}",
        }
