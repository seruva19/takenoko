"""Training metrics utilities that manage state externally."""

import argparse
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def should_generate_parameter_stats(
    global_step: int,
    last_param_log_step: int,
    log_every_n_steps: int
) -> bool:
    """Check if parameter statistics should be generated at this step."""
    return global_step - last_param_log_step >= log_every_n_steps


def generate_parameter_stats(
    model: Any,
    global_step: int,
    last_param_log_step: int,
    log_every_n_steps: int = 100,
    max_params_to_log: int = 20,
) -> Dict[str, float]:
    """Generate parameter statistics if enough steps have passed.
    
    Args:
        model: The model to analyze
        global_step: Current global step
        last_param_log_step: Last step when params were logged
        log_every_n_steps: Steps between parameter logging
        max_params_to_log: Maximum parameters to include
        
    Returns:
        Dict of parameter statistics (empty if not time to log)
    """
    if not should_generate_parameter_stats(global_step, last_param_log_step, log_every_n_steps):
        return {}
    
    # Delegate to existing implementation
    from core.metrics import generate_parameter_stats as _gps
    return _gps(model, global_step, log_every_n_steps, max_params_to_log)


def compute_per_source_loss(
    model_pred: Any,
    target: Any,
    batch: Dict[str, Any],
    weighting: Optional[Any] = None,
    sample_weights: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute per-source loss statistics.
    
    This is a pure function that doesn't use instance state.
    """
    from core.metrics import compute_per_source_loss as _cpsl
    return _cpsl(model_pred, target, batch, weighting, sample_weights)


def compute_gradient_norm(
    model: Any, 
    max_norm: Optional[float] = None, 
    norm_type: float = 2.0
) -> float:
    """Compute gradient norm for the model.
    
    This is a pure function that doesn't use instance state.
    """
    from core.metrics import compute_gradient_norm as _cgn
    return _cgn(model, max_norm, norm_type)


def generate_step_logs(
    args: argparse.Namespace,
    current_loss: float,
    avr_loss: float,
    lr_scheduler: Any,
    lr_descriptions: list,
    optimizer: Optional[Any] = None,
    keys_scaled: Optional[int] = None,
    mean_norm: Optional[float] = None,
    maximum_norm: Optional[float] = None,
    ema_loss: Optional[float] = None,
    model: Optional[Any] = None,
    global_step: Optional[int] = None,
    per_source_losses: Optional[Dict[str, float]] = None,
    gradient_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate step logging information.
    
    This is a pure function that doesn't use instance state.
    """
    from core.metrics import generate_step_logs as _gsl
    return _gsl(
        args,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer,
        keys_scaled,
        mean_norm,
        maximum_norm,
        ema_loss,
        model,
        global_step,
        per_source_losses,
        gradient_norm,
    )


def generate_safe_progress_metrics(
    args: argparse.Namespace,
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
    """Generate safe progress metrics.
    
    This is a pure function that doesn't use instance state.
    """
    from core.metrics import generate_safe_progress_metrics as _gspm
    return _gspm(
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