"""Enhanced progress bar processing for training loop."""

import argparse
from typing import Dict, Any, Optional, Tuple
from common.performance_logger import (
    get_last_train_iter_ms,
    get_last_total_step_ms,
    get_hardware_metrics,
)


def process_enhanced_progress_bar(
    args: argparse.Namespace,
    current_loss: float,
    avr_loss: float,
    lr_scheduler: Any,
    epoch: int,
    global_step: int,
    keys_scaled: Optional[int],
    mean_norm: Optional[float],
    maximum_norm: Optional[float],
    current_perf_display_toggle: bool,
    current_iter_time_ema_sec: Optional[float],
    iter_time_ema_beta: float,
    progress_bar: Any,
    max_mean_logs: Optional[Dict[str, Any]] = None,
    current_step_in_epoch: Optional[int] = None,
    total_steps_in_epoch: Optional[int] = None,
) -> Tuple[Dict[str, Any], bool, Optional[float]]:
    """Process enhanced progress bar metrics and update display.

    Args:
        args: Training arguments
        current_loss: Current loss value
        avr_loss: Average loss
        lr_scheduler: Learning rate scheduler
        epoch: Current epoch
        global_step: Current global step
        keys_scaled: Number of keys scaled (if applicable)
        mean_norm: Mean norm (if applicable)
        maximum_norm: Maximum norm (if applicable)
        current_perf_display_toggle: Current toggle state
        current_iter_time_ema_sec: Current EMA iteration time
        iter_time_ema_beta: EMA beta for iteration time
        progress_bar: Progress bar object
        max_mean_logs: Additional logs from weight scaling

    Returns:
        Tuple of (enhanced_logs, new_perf_display_toggle, new_iter_time_ema_sec)
    """
    if not getattr(args, "enhanced_progress_bar", True):
        # Use original simple progress bar
        logs = {"avr_loss": avr_loss}
        if max_mean_logs and args.scale_weight_norms:
            progress_bar.set_postfix({**max_mean_logs, **logs})
        else:
            progress_bar.set_postfix(logs)
        return logs, current_perf_display_toggle, current_iter_time_ema_sec

    try:
        # Import here to avoid circular imports
        from core.handlers.metrics_utils import generate_safe_progress_metrics

        enhanced_logs = generate_safe_progress_metrics(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            epoch,
            global_step,
            keys_scaled,
            mean_norm,
            maximum_norm,
            current_step_in_epoch,
            total_steps_in_epoch,
        )

        new_iter_time_ema_sec = current_iter_time_ema_sec
        new_perf_display_toggle = current_perf_display_toggle

        # Alternate between last iteration ms and peak VRAM/util
        try:
            # Prefer actual train iteration time (fwd+bwd+opt),
            # fallback to total step ms if iter time not available
            iter_ms = get_last_train_iter_ms()
            if iter_ms > 0:
                # Convert to seconds and update EMA
                last_iter_s = iter_ms / 1000.0
                # Update EMA
                if new_iter_time_ema_sec is None:
                    new_iter_time_ema_sec = float(last_iter_s)
                else:
                    new_iter_time_ema_sec = iter_time_ema_beta * float(
                        new_iter_time_ema_sec
                    ) + (1 - iter_time_ema_beta) * float(last_iter_s)

                # Expose with 1 decimal in postfix for readability
                enhanced_logs["iter_s"] = f"{last_iter_s:.1f}"
                enhanced_logs["avg_iter_s"] = f"{new_iter_time_ema_sec:.1f}"
                try:
                    # Show label as "avg.s (last.s)" with one decimal
                    progress_bar.set_description(
                        f"{new_iter_time_ema_sec:.1f}s ({last_iter_s:.1f}s)"
                    )
                except Exception:
                    pass
            else:
                total_ms = get_last_total_step_ms()
                if total_ms > 0:
                    last_step_s = total_ms / 1000.0
                    # Update EMA
                    if new_iter_time_ema_sec is None:
                        new_iter_time_ema_sec = float(last_step_s)
                    else:
                        new_iter_time_ema_sec = iter_time_ema_beta * float(
                            new_iter_time_ema_sec
                        ) + (1 - iter_time_ema_beta) * float(last_step_s)

                    enhanced_logs["step_s"] = f"{last_step_s:.1f}"
                    enhanced_logs["avg_iter_s"] = f"{new_iter_time_ema_sec:.1f}"
                    try:
                        progress_bar.set_description(
                            f"{new_iter_time_ema_sec:.1f}s ({last_step_s:.1f}s)"
                        )
                    except Exception:
                        pass

            if getattr(args, "alternate_perf_postfix", True):
                # Alternate between timing and hardware each step
                new_perf_display_toggle = not new_perf_display_toggle
                if not new_perf_display_toggle:
                    hardware_metrics = get_hardware_metrics()
                    enhanced_logs.update(hardware_metrics)
            else:
                # Show both timing and hardware every step
                hardware_metrics = get_hardware_metrics()
                enhanced_logs.update(hardware_metrics)
        except Exception:
            pass

        progress_bar.set_postfix(enhanced_logs)
        return enhanced_logs, new_perf_display_toggle, new_iter_time_ema_sec

    except Exception:
        # Fallback to original simple display if enhanced metrics fail
        logs = {"avr_loss": avr_loss}
        if max_mean_logs and args.scale_weight_norms:
            progress_bar.set_postfix({**max_mean_logs, **logs})
        else:
            progress_bar.set_postfix(logs)
        return logs, current_perf_display_toggle, current_iter_time_ema_sec
