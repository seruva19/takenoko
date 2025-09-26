"""TREAD routing manager for clean separation of routing logic."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
import torch
import logging

if TYPE_CHECKING:
    from .tread_router import TREADRouter, MaskInfo
    from .tread_frame import FrameRouteState
    from .tread_row import RowRouteState

from utils.tread_frame import (
    FrameRouteState,
    pack_frame_routed_tokens,
    reconstruct_frame_routed_tokens,
)
from utils.tread_row import (
    RowRouteState,
    pack_row_routed_tokens,
    reconstruct_row_routed_tokens,
    pack_spatial_auto_tokens,
    reconstruct_spatial_auto_tokens,
)
from utils.tread_token import (
    build_batched_rotary_from_freqs,
    slice_e0_for_token_route,
    restore_e0_after_route,
)

logger = logging.getLogger(__name__)


@dataclass
class TREADRoutingState:
    """Consolidated state for all TREAD routing modes."""

    # Common state
    routing_now: bool = False
    tread_mode: str = "full"

    # Content-aware routing state
    tread_mask_info: Optional["MaskInfo"] = None
    saved_tokens: Optional[torch.Tensor] = None

    # Frame-based routing state
    frame_state: Optional[FrameRouteState] = None
    orig_freqs_list: Optional[List[torch.Tensor]] = None
    e0_full_saved: Optional[torch.Tensor] = None

    # Row-based routing state
    row_state: Optional[RowRouteState] = None
    row_orig_freqs_list: Optional[List[torch.Tensor]] = None
    row_e0_full_saved: Optional[torch.Tensor] = None

    # Spatial auto routing state
    spatial_auto_state: Optional[Union[RowRouteState, FrameRouteState]] = None
    spatial_auto_orig_freqs_list: Optional[List[torch.Tensor]] = None
    spatial_auto_e0_full_saved: Optional[torch.Tensor] = None


class TREADRoutingManager:
    """Manager class for TREAD routing operations."""

    def __init__(self, model):
        self.model = model
        self.state = TREADRoutingState()

    def start_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        tread_mode: str,
        mask_ratio: float,
        router: Optional["TREADRouter"] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        freqs_list: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Start TREAD routing based on the specified mode.

        Parameters
        ----------
        x: torch.Tensor
            Input token tensor
        kwargs: Dict[str, Any]
            Model forward kwargs (will be modified in-place)
        tread_mode: str
            TREAD routing mode
        mask_ratio: float
            Masking ratio for routing
        router: Optional[TREADRouter]
            Router instance for content-aware routing
        force_keep_mask: Optional[torch.Tensor]
            Forced keep mask for content-aware routing
        freqs_list: Optional[List[torch.Tensor]]
            Frequency list for rotary embeddings

        Returns
        -------
        x: torch.Tensor
            Processed token tensor
        kwargs: Dict[str, Any]
            Updated kwargs
        """
        self.state.tread_mode = tread_mode
        self.state.routing_now = True

        if tread_mode.startswith("frame_"):
            return self._start_frame_routing(x, kwargs, tread_mode, mask_ratio)
        elif tread_mode.startswith("row_"):
            return self._start_row_routing(x, kwargs, tread_mode, mask_ratio, router)
        elif tread_mode == "spatial_auto":
            return self._start_spatial_auto_routing(x, kwargs, mask_ratio, router)
        else:  # "full" - content-aware routing
            return self._start_content_aware_routing(x, kwargs, mask_ratio, router, force_keep_mask, freqs_list)

    def _start_frame_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        tread_mode: str,
        mask_ratio: float
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Start frame-based routing."""
        keep_ratio = max(0.0, min(1.0, 1.0 - mask_ratio))
        mode = "contiguous" if tread_mode == "frame_contiguous" else "stride"

        x_proc, self.state.frame_state = pack_frame_routed_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, mode
        )

        # Update kwargs
        x = x_proc
        kwargs["seq_lens"] = self.state.frame_state.seq_lens_proc
        kwargs["grid_sizes"] = self.state.frame_state.grid_sizes_proc

        # Recompute rotary freqs if needed
        if not self.model.rope_on_the_fly:
            self.state.orig_freqs_list = kwargs.get("freqs")
            freqs_list_proc: List[torch.Tensor] = []
            for fhw_tensor in self.state.frame_state.grid_sizes_proc:
                fhw = tuple(int(v) for v in fhw_tensor.tolist())
                if fhw not in self.model.freqs_fhw:
                    c = self.model.dim // self.model.num_heads // 2
                    # Import here to avoid circular imports
                    from wan.modules.model import calculate_freqs_i
                    self.model.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.model.freqs)
                freqs_list_proc.append(self.model.freqs_fhw[fhw])
            kwargs["freqs"] = freqs_list_proc

        # Handle time embedding routing
        self._handle_time_embedding_routing(kwargs, self.state.frame_state.idx_proc_pad, "frame")

        return x, kwargs

    def _start_row_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        tread_mode: str,
        mask_ratio: float,
        router: Optional["TREADRouter"] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Start row-based routing."""
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

        # Get routing parameters
        router_seed = getattr(router, 'seed', None) if router else None
        auto_fallback = getattr(self.model, 'row_tread_auto_fallback', True)

        x_proc, self.state.row_state = pack_row_routed_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, row_mode, router_seed, auto_fallback
        )

        # Update kwargs
        x = x_proc
        kwargs["seq_lens"] = self.state.row_state.seq_lens_proc
        kwargs["grid_sizes"] = self.state.row_state.grid_sizes_proc

        # Recompute rotary freqs if needed
        if not self.model.rope_on_the_fly:
            self.state.row_orig_freqs_list = kwargs.get("freqs")
            freqs_list_proc: List[torch.Tensor] = []
            for fhw_tensor in self.state.row_state.grid_sizes_proc:
                fhw = tuple(int(v) for v in fhw_tensor.tolist())
                if fhw not in self.model.freqs_fhw:
                    c = self.model.dim // self.model.num_heads // 2
                    # Import here to avoid circular imports
                    from wan.modules.model import calculate_freqs_i
                    self.model.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.model.freqs)
                freqs_list_proc.append(self.model.freqs_fhw[fhw])
            kwargs["freqs"] = freqs_list_proc

        # Handle time embedding routing
        self._handle_time_embedding_routing(kwargs, self.state.row_state.idx_proc_pad, "row")

        return x, kwargs

    def _start_spatial_auto_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        mask_ratio: float,
        router: Optional["TREADRouter"] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Start spatial auto routing."""
        keep_ratio = max(0.0, min(1.0, 1.0 - mask_ratio))
        router_seed = getattr(router, 'seed', None) if router else None

        x_proc, self.state.spatial_auto_state = pack_spatial_auto_tokens(
            x, kwargs["seq_lens"], kwargs["grid_sizes"], keep_ratio, "contiguous", router_seed
        )

        # Update kwargs based on state type
        x = x_proc
        if isinstance(self.state.spatial_auto_state, RowRouteState):
            kwargs["seq_lens"] = self.state.spatial_auto_state.seq_lens_proc
            kwargs["grid_sizes"] = self.state.spatial_auto_state.grid_sizes_proc
            idx_proc_pad = self.state.spatial_auto_state.idx_proc_pad
        else:
            kwargs["seq_lens"] = self.state.spatial_auto_state.seq_lens_proc
            kwargs["grid_sizes"] = self.state.spatial_auto_state.grid_sizes_proc
            idx_proc_pad = self.state.spatial_auto_state.idx_proc_pad

        # Recompute rotary freqs if needed
        if not self.model.rope_on_the_fly:
            self.state.spatial_auto_orig_freqs_list = kwargs.get("freqs")
            freqs_list_proc: List[torch.Tensor] = []
            grid_sizes_proc = (
                self.state.spatial_auto_state.grid_sizes_proc
                if hasattr(self.state.spatial_auto_state, 'grid_sizes_proc')
                else self.state.spatial_auto_state.grid_sizes_proc
            )
            for fhw_tensor in grid_sizes_proc:
                fhw = tuple(int(v) for v in fhw_tensor.tolist())
                if fhw not in self.model.freqs_fhw:
                    c = self.model.dim // self.model.num_heads // 2
                    # Import here to avoid circular imports
                    from wan.modules.model import calculate_freqs_i
                    self.model.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.model.freqs)
                freqs_list_proc.append(self.model.freqs_fhw[fhw])
            kwargs["freqs"] = freqs_list_proc

        # Handle time embedding routing
        self._handle_time_embedding_routing(kwargs, idx_proc_pad, "spatial_auto")

        return x, kwargs

    def _start_content_aware_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        mask_ratio: float,
        router: Optional["TREADRouter"] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        freqs_list: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Start content-aware routing."""
        if router is None:
            raise ValueError("Router required for content-aware routing")

        # Only route video tokens (x). Text/context stays full
        self.state.tread_mask_info = router.get_mask(
            x, mask_ratio=mask_ratio, force_keep=force_keep_mask
        )
        self.state.saved_tokens = x
        x = router.start_route(x, self.state.tread_mask_info)

        # Build batched rotary tensor for routed tokens
        if freqs_list is not None:
            B = x.size(0)
            S_keep = x.size(1)
            shuf = build_batched_rotary_from_freqs(freqs_list, self.state.tread_mask_info.ids_shuffle)
            batched_rotary = shuf[:, :S_keep, :]
            kwargs["batched_rotary"] = batched_rotary

        # Slice per-token time embedding
        e_arg = kwargs.get("e")
        saved_e, e_proc = slice_e0_for_token_route(
            e_arg if isinstance(e_arg, torch.Tensor) else torch.tensor([]),
            self.state.tread_mask_info.ids_shuffle,
            x.size(1),
            self.model.broadcast_time_embed,
            self.model.strict_e_slicing_checks,
        )
        if saved_e is not None:
            self.state.e0_full_saved = saved_e
            kwargs["e"] = e_proc

        return x, kwargs

    def _handle_time_embedding_routing(
        self,
        kwargs: Dict[str, Any],
        idx_proc_pad: torch.Tensor,
        routing_type: str
    ):
        """Handle time embedding routing for spatial modes."""
        e_param = kwargs.get("e")
        if not isinstance(e_param, torch.Tensor) or e_param.dim() != 4:
            return

        # Validate indices are within bounds
        proc_valid = idx_proc_pad >= 0
        if proc_valid.any():
            max_proc = (
                torch.where(
                    proc_valid,
                    idx_proc_pad,
                    idx_proc_pad.new_zeros(()).expand_as(idx_proc_pad),
                )
                .max()
                .item()
            )
        else:
            max_proc = -1

        max_allowed = e_param.size(1) - 1
        if max_proc > max_allowed:
            logger.error(
                f"{routing_type} routing index out of bounds: max_proc={max_proc}, max_allowed={max_allowed}"
            )

        # Save original and create processed embedding
        if routing_type == "frame":
            self.state.e0_full_saved = e_param
        elif routing_type == "row":
            self.state.row_e0_full_saved = e_param
        elif routing_type == "spatial_auto":
            self.state.spatial_auto_e0_full_saved = e_param

        B = e_param.size(0)
        Lpmax = int(idx_proc_pad.size(1))
        C6 = e_param.size(2)
        C = e_param.size(3)
        e_proc = e_param.new_zeros((B, Lpmax, C6, C), dtype=e_param.dtype)

        for b in range(B):
            mask = idx_proc_pad[b] >= 0
            idx = idx_proc_pad[b, mask]
            Lpi = int(idx.numel())
            if Lpi > 0:
                e_proc[b, :Lpi, :, :] = e_param[b, idx, :, :]

        kwargs["e"] = e_proc

    def end_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        router: Optional["TREADRouter"] = None,
        freqs_list: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """End TREAD routing and reconstruct full tensors."""

        tread_mode = self.state.tread_mode

        if tread_mode.startswith("frame_"):
            return self._end_frame_routing(x, kwargs)
        elif tread_mode.startswith("row_"):
            return self._end_row_routing(x, kwargs)
        elif tread_mode == "spatial_auto":
            return self._end_spatial_auto_routing(x, kwargs)
        else:  # "full" - content-aware routing
            return self._end_content_aware_routing(x, kwargs, router, freqs_list)

    def _end_frame_routing(self, x: torch.Tensor, kwargs: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """End frame-based routing."""
        assert self.state.frame_state is not None
        x = reconstruct_frame_routed_tokens(x, self.state.frame_state)

        # Restore original seq/grid sizes
        kwargs["seq_lens"] = self.state.frame_state.seq_lens_orig
        kwargs["grid_sizes"] = self.state.frame_state.grid_sizes_orig

        # Restore original rotary freqs
        if not self.model.rope_on_the_fly and self.state.orig_freqs_list is not None:
            kwargs["freqs"] = self.state.orig_freqs_list

        # Restore original time embedding
        if self.state.e0_full_saved is not None:
            kwargs["e"] = self.state.e0_full_saved

        # Cleanup
        self._cleanup_frame_state()
        return x, kwargs

    def _end_row_routing(self, x: torch.Tensor, kwargs: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """End row-based routing."""
        assert self.state.row_state is not None
        x = reconstruct_row_routed_tokens(x, self.state.row_state)

        # Restore original seq/grid sizes
        kwargs["seq_lens"] = self.state.row_state.seq_lens_orig
        kwargs["grid_sizes"] = self.state.row_state.grid_sizes_orig

        # Restore original rotary freqs
        if not self.model.rope_on_the_fly and self.state.row_orig_freqs_list is not None:
            kwargs["freqs"] = self.state.row_orig_freqs_list

        # Restore original time embedding
        if self.state.row_e0_full_saved is not None:
            kwargs["e"] = self.state.row_e0_full_saved

        # Cleanup
        self._cleanup_row_state()
        return x, kwargs

    def _end_spatial_auto_routing(self, x: torch.Tensor, kwargs: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """End spatial auto routing."""
        assert self.state.spatial_auto_state is not None
        x = reconstruct_spatial_auto_tokens(x, self.state.spatial_auto_state)

        # Restore original seq/grid sizes based on state type
        if isinstance(self.state.spatial_auto_state, RowRouteState):
            kwargs["seq_lens"] = self.state.spatial_auto_state.seq_lens_orig
            kwargs["grid_sizes"] = self.state.spatial_auto_state.grid_sizes_orig
        else:
            kwargs["seq_lens"] = self.state.spatial_auto_state.seq_lens_orig
            kwargs["grid_sizes"] = self.state.spatial_auto_state.grid_sizes_orig

        # Restore original rotary freqs
        if not self.model.rope_on_the_fly and self.state.spatial_auto_orig_freqs_list is not None:
            kwargs["freqs"] = self.state.spatial_auto_orig_freqs_list

        # Restore original time embedding
        if self.state.spatial_auto_e0_full_saved is not None:
            kwargs["e"] = self.state.spatial_auto_e0_full_saved

        # Cleanup
        self._cleanup_spatial_auto_state()
        return x, kwargs

    def _end_content_aware_routing(
        self,
        x: torch.Tensor,
        kwargs: Dict[str, Any],
        router: Optional["TREADRouter"] = None,
        freqs_list: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """End content-aware routing."""
        if router is None or self.state.tread_mask_info is None or self.state.saved_tokens is None:
            raise ValueError("Invalid state for content-aware routing")

        x = router.end_route(x, self.state.tread_mask_info, original_x=self.state.saved_tokens)
        kwargs.pop("batched_rotary", None)

        # Restore full rotary embeddings when not on-the-fly
        if not self.model.rope_on_the_fly and freqs_list is not None:
            kwargs["freqs"] = freqs_list

        # Restore original per-token e0
        if self.state.e0_full_saved is not None:
            kwargs["e"] = self.state.e0_full_saved

        # Cleanup
        self._cleanup_content_aware_state()
        return x, kwargs

    def _cleanup_frame_state(self):
        """Cleanup frame routing state."""
        self.state.frame_state = None
        self.state.orig_freqs_list = None
        self.state.e0_full_saved = None

    def _cleanup_row_state(self):
        """Cleanup row routing state."""
        self.state.row_state = None
        self.state.row_orig_freqs_list = None
        self.state.row_e0_full_saved = None

    def _cleanup_spatial_auto_state(self):
        """Cleanup spatial auto routing state."""
        self.state.spatial_auto_state = None
        self.state.spatial_auto_orig_freqs_list = None
        self.state.spatial_auto_e0_full_saved = None

    def _cleanup_content_aware_state(self):
        """Cleanup content-aware routing state."""
        self.state.saved_tokens = None
        self.state.tread_mask_info = None
        self.state.e0_full_saved = None

    def cleanup_all(self):
        """Final cleanup of all routing state."""
        self._cleanup_frame_state()
        self._cleanup_row_state()
        self._cleanup_spatial_auto_state()
        self._cleanup_content_aware_state()
        self.state.routing_now = False
        self.state.tread_mode = "full"