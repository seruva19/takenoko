from __future__ import annotations

import argparse
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from common.logger import get_logger
from enhancements.reflexflow.reflexflow_helper import ReflexFlowHelper, ReflexFlowState

logger = get_logger(__name__, level=logging.INFO)


def maybe_apply_reflexflow(
    *,
    args: argparse.Namespace,
    reflexflow_helper: Optional[ReflexFlowHelper],
    noisy_model_input: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    global_step: Optional[int],
    eqm_enabled: bool,
    warned_reflexflow_eqm: bool,
) -> Tuple[torch.Tensor, Optional[ReflexFlowState], Optional[torch.Tensor], bool]:
    if reflexflow_helper is None or not reflexflow_helper.enabled:
        return noisy_model_input, None, None, warned_reflexflow_eqm

    if eqm_enabled:
        if not warned_reflexflow_eqm:
            logger.warning("ReflexFlow skipped: EqM mode active.")
            warned_reflexflow_eqm = True
        return noisy_model_input, None, None, warned_reflexflow_eqm

    clean_noisy_model_input = noisy_model_input.detach().clone()
    noisy_model_input, reflexflow_state = reflexflow_helper.apply_to_inputs(
        noisy_model_input=noisy_model_input,
        latents=latents,
        noise=noise,
        global_step=global_step,
    )
    return noisy_model_input, reflexflow_state, clean_noisy_model_input, warned_reflexflow_eqm


def build_reflexflow_context(
    *,
    args: argparse.Namespace,
    reflexflow_state: Optional[ReflexFlowState],
    reflexflow_clean_noisy_input: Optional[torch.Tensor],
    get_clean_prediction: Callable[[], Optional[torch.Tensor]],
    warned_reflexflow_clean_pred: bool,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not bool(getattr(args, "enable_reflexflow", False)):
        return None, warned_reflexflow_clean_pred

    clean_pred = None
    if (
        reflexflow_state is not None
        and bool(getattr(reflexflow_state, "applied", False))
        and reflexflow_clean_noisy_input is not None
    ):
        try:
            clean_pred = get_clean_prediction()
        except Exception as exc:
            if not warned_reflexflow_clean_pred:
                logger.warning(
                    "ReflexFlow clean prediction unavailable (%s); FC term will be skipped.",
                    exc,
                )
                warned_reflexflow_clean_pred = True

    return (
        {
            "applied": bool(
                reflexflow_state is not None
                and getattr(reflexflow_state, "applied", False)
            ),
            "clean_pred": clean_pred,
        },
        warned_reflexflow_clean_pred,
    )


def maybe_log_reflexflow_metrics(
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    reflexflow_helper: Optional[ReflexFlowHelper],
    reflexflow_state: Optional[ReflexFlowState],
    global_step: int,
) -> None:
    if (
        not accelerator.is_main_process
        or not accelerator.trackers
        or not accelerator.sync_gradients
        or reflexflow_helper is None
        or reflexflow_state is None
    ):
        return

    log_interval = int(getattr(args, "reflexflow_log_interval", 50) or 50)
    if global_step % max(log_interval, 1) != 0:
        return

    metrics = reflexflow_helper.state_to_metrics(reflexflow_state)
    if metrics:
        accelerator.log(metrics, step=global_step)
