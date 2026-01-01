"""BFM sampling utilities for inference-time conditioning and segment routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import logging

from common.logger import get_logger
from enhancements.blockwise_flow_matching.conditioning import BFMConditioningHelper
from enhancements.blockwise_flow_matching.segment_utils import (
    build_segment_boundaries,
    normalize_timesteps,
    segment_index_for_timesteps,
)

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class BFMSamplingState:
    bfm_inference_enabled: bool
    bfm_inference_semfeat: bool
    bfm_inference_segment: bool
    bfm_segment_blocks_enabled: bool
    bfm_semfeat_model_injection_enabled: bool
    bfm_inference_refresh: str
    bfm_inference_use_frn: bool
    bfm_helper: Optional[BFMConditioningHelper]
    bfm_cached_semfeat: Optional[torch.Tensor]
    bfm_cached_segment_idx: Optional[int]
    bfm_boundaries: Optional[torch.Tensor]


def setup_bfm_sampling(
    args: Any,
    *,
    context: torch.Tensor,
    device: torch.device,
    vae: Optional[Any],
) -> BFMSamplingState:
    bfm_inference_enabled = bool(
        getattr(args, "bfm_inference_enabled", False)
    )
    bfm_inference_semfeat = bool(
        getattr(args, "bfm_inference_semfeat_enabled", False)
    )
    bfm_inference_segment = bool(
        getattr(args, "bfm_inference_segment_enabled", False)
    )
    bfm_segment_blocks_enabled = bool(
        getattr(args, "bfm_segment_blocks_enabled", False)
    )
    bfm_semfeat_model_injection_enabled = bool(
        getattr(args, "bfm_semfeat_model_injection_enabled", False)
    )
    bfm_inference_refresh = str(
        getattr(args, "bfm_inference_semfeat_refresh", "per_segment")
    ).lower()
    bfm_inference_use_frn = bool(
        getattr(args, "bfm_inference_use_frn", False)
    )
    bfm_helper: Optional[BFMConditioningHelper] = None
    bfm_cached_semfeat: Optional[torch.Tensor] = None
    bfm_cached_segment_idx: Optional[int] = None
    bfm_boundaries: Optional[torch.Tensor] = None

    bfm_need_helper = bool(
        bfm_inference_semfeat
        or bfm_inference_segment
        or bfm_semfeat_model_injection_enabled
    )
    if bfm_need_helper:
        logger.warning(
            "BFM inference/model features enabled; outputs and performance will differ."
        )
        bfm_helper = BFMConditioningHelper(args, context.shape[-1], device)
        bfm_helper.set_inference_overrides(
            semfeat_scale=float(
                getattr(args, "bfm_inference_semfeat_scale", 1.0)
            ),
            segment_scale=float(
                getattr(args, "bfm_inference_segment_scale", 1.0)
            ),
        )
        if bfm_inference_use_frn and not getattr(
            bfm_helper, "frn_enabled", False
        ):
            logger.warning(
                "BFM inference FRN requested but FRN is disabled; falling back to VAE SemFeat."
            )
            bfm_inference_use_frn = False

    if bfm_inference_segment or bfm_segment_blocks_enabled:
        bfm_boundaries = build_segment_boundaries(
            num_segments=int(getattr(args, "bfm_num_segments", 6)),
            min_t=float(getattr(args, "bfm_segment_min_t", 0.0)),
            max_t=float(getattr(args, "bfm_segment_max_t", 1.0)),
            device=device,
            dtype=context.dtype,
        )

    if (bfm_inference_semfeat or bfm_semfeat_model_injection_enabled) and vae is None:
        if bfm_inference_use_frn:
            logger.info(
                "BFM SemFeat inference using FRN; skipping VAE requirement."
            )
        else:
            logger.warning(
                "BFM SemFeat inference enabled but VAE is unavailable; disabling."
            )
            bfm_inference_semfeat = False

    return BFMSamplingState(
        bfm_inference_enabled=bfm_inference_enabled,
        bfm_inference_semfeat=bfm_inference_semfeat,
        bfm_inference_segment=bfm_inference_segment,
        bfm_segment_blocks_enabled=bfm_segment_blocks_enabled,
        bfm_semfeat_model_injection_enabled=bfm_semfeat_model_injection_enabled,
        bfm_inference_refresh=bfm_inference_refresh,
        bfm_inference_use_frn=bfm_inference_use_frn,
        bfm_helper=bfm_helper,
        bfm_cached_semfeat=bfm_cached_semfeat,
        bfm_cached_segment_idx=bfm_cached_segment_idx,
        bfm_boundaries=bfm_boundaries,
    )


def apply_bfm_sampling_step(
    state: BFMSamplingState,
    *,
    context: torch.Tensor,
    context_null: Optional[torch.Tensor],
    timestep: torch.Tensor,
    latent: torch.Tensor,
    vae: Optional[Any],
    arg_c: dict,
    arg_null: dict,
) -> Tuple[dict, dict, Optional[torch.Tensor]]:
    arg_c_step = arg_c
    arg_null_step = arg_null

    segment_idx = None
    segment_idx_value = None
    if state.bfm_boundaries is not None and state.bfm_segment_blocks_enabled:
        t_norm = normalize_timesteps(timestep.view(-1))
        segment_idx = segment_index_for_timesteps(t_norm, state.bfm_boundaries)
        if segment_idx.numel() > 0:
            segment_idx_value = int(segment_idx[0].item())

    if state.bfm_helper is not None and (
        state.bfm_inference_semfeat
        or state.bfm_inference_segment
        or state.bfm_semfeat_model_injection_enabled
        or state.bfm_segment_blocks_enabled
    ):
        context_step = context
        context_null_step = context_null
        semfeat_tokens = None
        segment_tokens = None

        if state.bfm_boundaries is not None and state.bfm_inference_segment:
            if segment_idx is None:
                t_norm = normalize_timesteps(timestep.view(-1))
                segment_idx = segment_index_for_timesteps(
                    t_norm, state.bfm_boundaries
                )
            if segment_idx is not None and segment_idx.numel() > 0:
                segment_idx_value = int(segment_idx[0].item())
            segment_tokens = state.bfm_helper.compute_segment_tokens(timestep)

        if (
            state.bfm_inference_semfeat
            or state.bfm_semfeat_model_injection_enabled
        ):
            refresh = state.bfm_inference_refresh == "per_step"
            if (
                not refresh
                and segment_idx_value is not None
                and segment_idx_value != state.bfm_cached_segment_idx
            ):
                refresh = True
            if refresh or state.bfm_cached_semfeat is None:
                if state.bfm_inference_use_frn and state.bfm_helper is not None:
                    semfeat_tokens = state.bfm_helper.predict_frn_tokens(
                        latent.unsqueeze(0),
                        timestep,
                    )
                elif vae is not None:
                    with torch.no_grad():
                        decoded = vae.decode(
                            latent.unsqueeze(0).to(
                                device=latent.device, dtype=vae.dtype
                            )
                        )[0]
                    semfeat_tokens = (
                        state.bfm_helper.compute_semantic_tokens_from_pixels(
                            decoded.unsqueeze(0)
                        )
                    )
                state.bfm_cached_semfeat = semfeat_tokens
            else:
                semfeat_tokens = state.bfm_cached_semfeat

        if segment_idx_value is not None:
            state.bfm_cached_segment_idx = segment_idx_value

        if state.bfm_inference_semfeat and semfeat_tokens is not None:
            context_step = torch.cat([context_step, semfeat_tokens[0]], dim=0)
            if context_null_step is not None:
                context_null_step = torch.cat(
                    [context_null_step, semfeat_tokens[0]], dim=0
                )

        if state.bfm_inference_segment and segment_tokens is not None:
            context_step = torch.cat([context_step, segment_tokens[0]], dim=0)
            if context_null_step is not None:
                context_null_step = torch.cat(
                    [context_null_step, segment_tokens[0]], dim=0
                )

        if state.bfm_inference_semfeat or state.bfm_inference_segment:
            arg_c_step = dict(arg_c)
            arg_c_step["context"] = [context_step]
            if context_null_step is not None:
                arg_null_step = dict(arg_null)
                arg_null_step["context"] = [context_null_step]

        if segment_idx is not None:
            arg_c_step["segment_idx"] = segment_idx
            if context_null_step is not None:
                arg_null_step["segment_idx"] = segment_idx

        if semfeat_tokens is not None:
            arg_c_step["bfm_semfeat_tokens"] = semfeat_tokens
            if context_null_step is not None:
                arg_null_step["bfm_semfeat_tokens"] = semfeat_tokens

    if segment_idx is not None and "segment_idx" not in arg_c_step:
        arg_c_step = dict(arg_c_step)
        arg_c_step["segment_idx"] = segment_idx
        if context_null is not None:
            arg_null_step = dict(arg_null_step)
            arg_null_step["segment_idx"] = segment_idx

    return arg_c_step, arg_null_step, segment_idx
