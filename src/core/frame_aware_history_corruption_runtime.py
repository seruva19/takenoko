from __future__ import annotations

from typing import Optional, Tuple

import torch

from common.logger import get_logger
from enhancements.frame_aware_history_corruption.frame_aware_history_corruption_helper import (
    FrameAwareHistoryCorruptionHelper,
    FrameAwareHistoryCorruptionState,
)

logger = get_logger(__name__)


def maybe_apply_frame_aware_history_corruption(
    *,
    frame_aware_history_corruption_helper: Optional[FrameAwareHistoryCorruptionHelper],
    noisy_model_input: torch.Tensor,
    global_step: Optional[int],
    eqm_enabled: bool,
    warned_frame_aware_history_corruption_eqm: bool,
) -> Tuple[torch.Tensor, Optional[FrameAwareHistoryCorruptionState], bool]:
    if (
        frame_aware_history_corruption_helper is None
        or not frame_aware_history_corruption_helper.enabled
    ):
        return (
            noisy_model_input,
            None,
            warned_frame_aware_history_corruption_eqm,
        )

    if eqm_enabled:
        if not warned_frame_aware_history_corruption_eqm:
            logger.warning("Frame-aware history corruption skipped: EqM mode active.")
            warned_frame_aware_history_corruption_eqm = True
        return (
            noisy_model_input,
            None,
            warned_frame_aware_history_corruption_eqm,
        )

    noisy_model_input, frame_aware_history_corruption_state = (
        frame_aware_history_corruption_helper.apply_to_inputs(
            noisy_model_input=noisy_model_input,
            global_step=global_step,
        )
    )
    return (
        noisy_model_input,
        frame_aware_history_corruption_state,
        warned_frame_aware_history_corruption_eqm,
    )
