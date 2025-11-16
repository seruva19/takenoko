"""TREAD routing handlers for model.py to reduce code complexity."""

import torch
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from .tread_helpers import (
    handle_time_embedding_routing,
    recompute_rotary_freqs,
    cleanup_routing_variables,
    ensure_freqs_list_for_routing,
)
from .tread_row import (
    pack_row_routed_tokens,
    reconstruct_row_routed_tokens,
    RowRouteState,
    SpatialAutoRouteState,
)
from .tread_row import pack_spatial_auto_tokens, reconstruct_spatial_auto_tokens

try:
    from .tread_frame import pack_frame_routed_tokens, reconstruct_frame_routed_tokens, FrameRouteState
    _FRAME_ROUTING_AVAILABLE = True
except ImportError:
    _FRAME_ROUTING_AVAILABLE = False
    FrameRouteState = None

logger = logging.getLogger(__name__)


class TREADRoutingState:
    """Container for TREAD routing state variables."""

    def __init__(self):
        # Frame-based routing state
        self.frame_state: Optional[FrameRouteState] = None
        self.orig_freqs_list: Optional[List[torch.Tensor]] = None
        self.e0_full_saved: Optional[torch.Tensor] = None

        # Row-based routing state
        self.row_state: Optional[RowRouteState] = None
        self.row_orig_freqs_list: Optional[List[torch.Tensor]] = None
        self.row_e0_full_saved: Optional[torch.Tensor] = None

        # Spatial auto routing state
        self.spatial_auto_state: Optional[
            Union[RowRouteState, FrameRouteState, SpatialAutoRouteState]
        ] = None
        self.spatial_auto_orig_freqs_list: Optional[List[torch.Tensor]] = None
        self.spatial_auto_e0_full_saved: Optional[torch.Tensor] = None

        # Content-aware routing state
        self.tread_mask_info = None
        self.saved_tokens: Optional[torch.Tensor] = None

        # Common state
        self.routing_now: bool = False

    def cleanup_all(self):
        """Clean up all routing state variables."""
        cleanup_routing_variables(
            self,
            "frame_state",
            "row_state",
            "spatial_auto_state",
            "orig_freqs_list",
            "row_orig_freqs_list",
            "spatial_auto_orig_freqs_list",
            "e0_full_saved",
            "row_e0_full_saved",
            "spatial_auto_e0_full_saved",
            "tread_mask_info",
            "saved_tokens",
        )
        self.routing_now = False


def handle_routing_start(
    model,
    x: torch.Tensor,
    kwargs: Dict[str, Any],
    state: TREADRoutingState,
    route_config: Dict[str, Any],
    router,
    force_keep_mask=None,
    freqs_list=None
) -> torch.Tensor:
    """Handle TREAD routing start at configured layer index.

    Parameters
    ----------
    model
        Model instance with TREAD configuration
    x: torch.Tensor
        Input token sequence
    kwargs: Dict[str, Any]
        Model forward kwargs (modified in-place)
    state: TREADRoutingState
        Routing state container
    route_config: Dict[str, Any]
        Current route configuration with start/end layers and selection ratio
    router
        TREAD router instance
    force_keep_mask
        Optional mask for forced token keeping
    freqs_list
        Rotary frequency list

    Returns
    -------
    torch.Tensor
        Processed token sequence (potentially routed)
    """
    mask_ratio = float(route_config["selection_ratio"])
    tread_mode = str(getattr(model, "_tread_mode", "full"))

    # ─────────────────────────────────────────────────────────────────────────
    # FRAME-BASED ROUTING: Temporal video routing (F>1 frames)
    # ─────────────────────────────────────────────────────────────────────────
    if tread_mode.startswith("frame_"):
        if not _FRAME_ROUTING_AVAILABLE:
            raise ImportError("Frame routing not available")

        keep_ratio = max(0.0, min(1.0, 1.0 - mask_ratio))
        mode = "contiguous" if tread_mode == "frame_contiguous" else "stride"

        x_proc, state.frame_state = pack_frame_routed_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, mode
        )

        # Update kwargs with processed tensors
        x = x_proc
        kwargs["seq_lens"] = state.frame_state.seq_lens_proc
        kwargs["grid_sizes"] = state.frame_state.grid_sizes_proc

        # Recompute rotary frequencies
        state.orig_freqs_list = recompute_rotary_freqs(model, kwargs, state.frame_state.grid_sizes_proc)

        # Handle time embedding routing
        state.e0_full_saved = handle_time_embedding_routing(kwargs, state.frame_state.idx_proc_pad, "frame")

        state.routing_now = True

    # ─────────────────────────────────────────────────────────────────────────
    # ROW-BASED ROUTING: Spatial image routing (F=1 frame)
    # ─────────────────────────────────────────────────────────────────────────
    elif tread_mode.startswith("row_"):
        keep_ratio = max(0.0, min(1.0, 1.0 - mask_ratio))

        # Determine row mode
        if tread_mode == "row_contiguous":
            row_mode = "contiguous"
        elif tread_mode == "row_stride":
            row_mode = "stride"
        elif tread_mode == "row_random":
            row_mode = "random"
        else:
            row_mode = "contiguous"  # fallback

        # Get configuration
        router_seed = getattr(router, 'seed', None) if router else None
        auto_fallback = getattr(model, 'row_tread_auto_fallback', True)

        x_proc, state.row_state = pack_row_routed_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, row_mode, router_seed, auto_fallback
        )

        # Update kwargs with processed tensors
        x = x_proc
        kwargs["seq_lens"] = state.row_state.seq_lens_proc
        kwargs["grid_sizes"] = state.row_state.grid_sizes_proc

        # Recompute rotary frequencies
        state.row_orig_freqs_list = recompute_rotary_freqs(model, kwargs, state.row_state.grid_sizes_proc)

        # Handle time embedding routing
        state.row_e0_full_saved = handle_time_embedding_routing(kwargs, state.row_state.idx_proc_pad, "row")

        state.routing_now = True

    # ─────────────────────────────────────────────────────────────────────────
    # SPATIAL AUTO ROUTING: Hybrid mode (F=1→rows, F>1→frames)
    # ─────────────────────────────────────────────────────────────────────────
    elif tread_mode == "spatial_auto":
        keep_ratio = max(0.0, min(1.0, 1.0 - mask_ratio))

        # Determine mode from tread_mode suffix (default to contiguous)
        if "_stride" in tread_mode:
            auto_mode = "stride"
        elif "_random" in tread_mode:
            auto_mode = "random"
        else:
            auto_mode = "contiguous"

        router_seed = getattr(router, 'seed', None) if router else None

        x_proc, state.spatial_auto_state = pack_spatial_auto_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, auto_mode, router_seed
        )

        # Update kwargs with processed tensors
        x = x_proc
        kwargs["seq_lens"] = state.spatial_auto_state.seq_lens_proc
        kwargs["grid_sizes"] = state.spatial_auto_state.grid_sizes_proc

        # Recompute rotary frequencies
        state.spatial_auto_orig_freqs_list = recompute_rotary_freqs(model, kwargs, state.spatial_auto_state.grid_sizes_proc)

        # Handle time embedding routing for spatial_auto
        if isinstance(state.spatial_auto_state, RowRouteState):
            idx_proc_pad = state.spatial_auto_state.idx_proc_pad
        else:
            idx_proc_pad = state.spatial_auto_state.idx_proc_pad
        state.spatial_auto_e0_full_saved = handle_time_embedding_routing(kwargs, idx_proc_pad, "spatial_auto")

        state.routing_now = True

    # ─────────────────────────────────────────────────────────────────────────
    # CONTENT-AWARE ROUTING: Importance-based token routing
    # ─────────────────────────────────────────────────────────────────────────
    else:
        # Only route video tokens (x). Text/context stays full
        state.tread_mask_info = router.get_mask(
            x, mask_ratio=mask_ratio, force_keep=force_keep_mask
        )
        state.saved_tokens = x
        x = router.start_route(x, state.tread_mask_info)
        state.routing_now = True

        # Build a batched rotary tensor for routed tokens
        effective_freqs = ensure_freqs_list_for_routing(
            model, kwargs["grid_sizes"], freqs_list
        )
        if not effective_freqs:
            raise RuntimeError(
                "TREAD routing requires rotary caches but none were available. "
                "Disable rope_on_the_fly or ensure grid caches are built."
            )
        try:
            from .tread_frame import build_batched_rotary_from_freqs
        except ImportError:
            # Fallback import path
            from utils.tread_token import build_batched_rotary_from_freqs

        B = x.size(0)
        S_keep = x.size(1)
        shuf = build_batched_rotary_from_freqs(
            effective_freqs, state.tread_mask_info.ids_shuffle
        )
        batched_rotary = shuf[:, :S_keep, :]
        kwargs["batched_rotary"] = batched_rotary

        # Slice per-token e to kept tokens (Wan 2.2 path) unless broadcasting is enabled
        try:
            from .tread_frame import slice_e0_for_token_route
        except ImportError:
            # Fallback import path
            from utils.tread_token import slice_e0_for_token_route

        e_arg = kwargs.get("e")
        saved_e, e_proc = slice_e0_for_token_route(
            e_arg if isinstance(e_arg, torch.Tensor) else torch.tensor([]),
            state.tread_mask_info.ids_shuffle,
            S_keep,
            model.broadcast_time_embed,
            model.strict_e_slicing_checks,
        )
        if saved_e is not None:
            state.e0_full_saved = saved_e
            kwargs["e"] = e_proc

    return x


def handle_routing_end(
    model,
    x: torch.Tensor,
    kwargs: Dict[str, Any],
    state: TREADRoutingState,
    freqs_list=None
) -> torch.Tensor:
    """Handle TREAD routing end at configured layer index.

    Parameters
    ----------
    model
        Model instance with TREAD configuration
    x: torch.Tensor
        Processed token sequence from routing
    kwargs: Dict[str, Any]
        Model forward kwargs (modified in-place)
    state: TREADRoutingState
        Routing state container
    freqs_list
        Original rotary frequency list

    Returns
    -------
    torch.Tensor
        Reconstructed full token sequence
    """
    tread_mode = str(getattr(model, "_tread_mode", "full"))

    if tread_mode.startswith("frame_"):
        assert state.frame_state is not None
        x = reconstruct_frame_routed_tokens(x, state.frame_state)

        # Restore original tensors
        kwargs["seq_lens"] = state.frame_state.seq_lens_orig
        kwargs["grid_sizes"] = state.frame_state.grid_sizes_orig

        # Restore original rotary freqs
        if not model.rope_on_the_fly and state.orig_freqs_list is not None:
            kwargs["freqs"] = state.orig_freqs_list

        # Restore original per-token e0 if it was sliced
        if state.e0_full_saved is not None:
            kwargs["e"] = state.e0_full_saved

        # Clean up state
        state.orig_freqs_list = None
        state.e0_full_saved = None
        state.frame_state = None

    elif tread_mode.startswith("row_"):
        assert state.row_state is not None
        x = reconstruct_row_routed_tokens(x, state.row_state)

        # Restore original tensors
        kwargs["seq_lens"] = state.row_state.seq_lens_orig
        kwargs["grid_sizes"] = state.row_state.grid_sizes_orig

        # Restore original rotary freqs
        if not model.rope_on_the_fly and state.row_orig_freqs_list is not None:
            kwargs["freqs"] = state.row_orig_freqs_list

        # Restore original time embedding
        if state.row_e0_full_saved is not None:
            kwargs["e"] = state.row_e0_full_saved

        # Clean up state
        state.row_e0_full_saved = None
        state.row_state = None

    elif tread_mode == "spatial_auto":
        assert state.spatial_auto_state is not None
        x = reconstruct_spatial_auto_tokens(x, state.spatial_auto_state)

        # Restore original tensors
        kwargs["seq_lens"] = state.spatial_auto_state.seq_lens_orig
        kwargs["grid_sizes"] = state.spatial_auto_state.grid_sizes_orig

        # Restore original rotary freqs
        if not model.rope_on_the_fly and state.spatial_auto_orig_freqs_list is not None:
            kwargs["freqs"] = state.spatial_auto_orig_freqs_list

        # Restore original time embedding
        if state.spatial_auto_e0_full_saved is not None:
            kwargs["e"] = state.spatial_auto_e0_full_saved

        # Clean up state
        state.spatial_auto_e0_full_saved = None
        state.spatial_auto_state = None

    else:
        # Content-aware routing
        assert state.tread_mask_info is not None and state.saved_tokens is not None
        x = model._tread_router.end_route(x, state.tread_mask_info, original_x=state.saved_tokens)
        kwargs.pop("batched_rotary", None)

        # Restore full rotary embeddings when not on-the-fly
        if not model.rope_on_the_fly:
            kwargs["freqs"] = freqs_list

        # Restore original per-token e0 if it was sliced in token-routing path
        if state.e0_full_saved is not None:
            kwargs["e"] = state.e0_full_saved
            state.e0_full_saved = None

        # Release references early to help GC
        state.saved_tokens = None
        state.tread_mask_info = None

    state.routing_now = False
    return x
