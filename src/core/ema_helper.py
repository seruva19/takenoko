from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from utils.ema import ExponentialMovingAverage

from common.logger import get_logger
from core.handlers.ema_utils import validate_ema_beta

logger = get_logger(__name__)


def initialize_weight_ema(
    owner: Any,
    args: Any,
    accelerator: Any,
    network: Any,
    register_checkpoint: bool = True,
) -> None:
    """Optionally initialize model weight EMA for eval/saving."""
    if getattr(owner, "_weight_ema_initialized", False):
        return

    owner.weight_ema_start_step = max(int(getattr(args, "weight_ema_start_step", 0)), 0)
    owner.weight_ema_use_for_eval = bool(getattr(args, "weight_ema_use_for_eval", True))
    owner.weight_ema_update_interval = max(int(getattr(args, "weight_ema_update_interval", 1)), 1)
    owner.weight_ema_device = str(getattr(args, "weight_ema_device", "accelerator")).lower()
    owner.weight_ema_eval_mode = str(getattr(args, "weight_ema_eval_mode", "ema")).lower()
    owner.weight_ema_save_separately = bool(getattr(args, "weight_ema_save_separately", False))

    if owner.weight_ema_device not in ("accelerator", "cpu"):
        logger.warning(
            "Unknown weight_ema_device=%s; defaulting to accelerator", owner.weight_ema_device
        )
        owner.weight_ema_device = "accelerator"

    owner.weight_ema = None
    owner._weight_ema_initialized = True

    if not getattr(args, "enable_weight_ema", False):
        return

    try:
        decay = float(getattr(args, "weight_ema_decay", 0.999))
        validate_ema_beta(decay)
    except Exception as exc:
        logger.warning("Disabling weight EMA due to invalid decay: %s", exc)
        return

    trainable_only = bool(getattr(args, "weight_ema_trainable_only", True))
    try:
        params = list(accelerator.unwrap_model(network).parameters())
    except Exception:
        params = list(network.parameters())
    if trainable_only:
        params = [p for p in params if p.requires_grad]

    if not params:
        logger.warning("Weight EMA requested but no parameters were selected; skipping EMA setup.")
        return

    device = accelerator.device if owner.weight_ema_device == "accelerator" else torch.device("cpu")
    try:
        owner.weight_ema = ExponentialMovingAverage(params, decay=decay)
        try:
            owner.weight_ema.to(device=device)
        except Exception as exc:
            logger.warning("Failed to move weight EMA to %s: %s", device, exc)
        if register_checkpoint:
            try:
                accelerator.register_for_checkpointing(owner.weight_ema)
            except Exception as exc:  # pragma: no cover
                logger.debug("Could not register weight EMA for checkpointing: %s", exc)
        logger.info(
            "Weight EMA enabled (decay=%s, start_step=%s, update_interval=%s, device=%s, params=%s, trainable_only=%s).",
            decay,
            owner.weight_ema_start_step,
            owner.weight_ema_update_interval,
            owner.weight_ema_device,
            len(params),
            trainable_only,
        )
    except Exception as exc:
        logger.warning("Failed to initialize weight EMA: %s", exc)
        owner.weight_ema = None


def weight_ema_eval_context(owner: Any):
    """Return context manager that swaps EMA weights in for eval/saving when enabled."""
    if (
        getattr(owner, "weight_ema", None) is None
        or getattr(owner, "weight_ema_eval_mode", "off") == "off"
        or not getattr(owner, "weight_ema_use_for_eval", True)
    ):
        return nullcontext()
    return owner.weight_ema.average_parameters()


def update_weight_ema_if_needed(owner: Any, accelerator: Any, global_step: int) -> None:
    """Update weight EMA if enabled and at the correct interval."""
    weight_ema = getattr(owner, "weight_ema", None)
    if weight_ema is None:
        return

    try:
        if not getattr(accelerator, "sync_gradients", True):
            return

        start_step = max(getattr(owner, "weight_ema_start_step", 0), 0)
        update_interval = max(getattr(owner, "weight_ema_update_interval", 1), 1)
        if global_step < start_step:
            return
        if (global_step - start_step) % update_interval != 0:
            return

        weight_ema.update()
    except Exception as exc:
        logger.warning("Disabling weight EMA due to update failure: %s", exc)
        owner.weight_ema = None


def set_ema_beta(owner: Any, beta: float) -> None:
    validate_ema_beta(beta)
    owner.ema_beta = beta
    logger.info("EMA beta set to %s", beta)


def configure_ema_from_args(owner: Any, args: Any) -> None:
    ema_beta, ema_bias_warmup_steps = getattr(args, "weight_ema_decay", None), getattr(
        args, "ema_bias_warmup_steps", None
    )
    try:
        if ema_beta is not None:
            set_ema_beta(owner, float(ema_beta))
        if ema_bias_warmup_steps is not None:
            owner.ema_bias_warmup_steps = ema_bias_warmup_steps
    except Exception as exc:
        logger.warning("EMA config from args failed; keeping defaults: %s", exc)
