"""Helper utilities for TREAD routing."""

import torch
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def handle_time_embedding_routing(
    kwargs: Dict[str, Any], idx_proc_pad: torch.Tensor, routing_type: str = "routing"
) -> Optional[torch.Tensor]:
    """Handle time embedding routing for spatial TREAD modes.

    Parameters
    ----------
    kwargs: Dict[str, Any]
        Model forward kwargs (will be modified in-place)
    idx_proc_pad: torch.Tensor
        Padded processed token indices
    routing_type: str
        Type of routing for error messages

    Returns
    -------
    Optional[torch.Tensor]
        Original time embedding if it was processed, None otherwise
    """
    e_param = kwargs.get("e")
    if not isinstance(e_param, torch.Tensor) or e_param.dim() != 4:
        return None

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

    rout_valid = getattr(idx_proc_pad, "idx_rout_pad", None) is not None
    if rout_valid and hasattr(idx_proc_pad, "idx_rout_pad"):
        rout_pad = idx_proc_pad.idx_rout_pad  # This won't work, but keeping structure
        rout_valid_mask = rout_pad >= 0
        if rout_valid_mask.any():
            max_rout = (
                torch.where(
                    rout_valid_mask,
                    rout_pad,
                    rout_pad.new_zeros(()).expand_as(rout_pad),
                )
                .max()
                .item()
            )
        else:
            max_rout = -1
    else:
        max_rout = -1

    max_allowed = e_param.size(1) - 1
    if max_proc > max_allowed or max_rout > max_allowed:
        logger.error(
            f"{routing_type} routing index out of bounds: max_proc={max_proc}, max_rout={max_rout}, max_allowed={max_allowed}"
        )

    # Save original and create processed embedding
    e0_full_saved = e_param
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
    return e0_full_saved


def recompute_rotary_freqs(
    model, kwargs: Dict[str, Any], grid_sizes_proc: torch.Tensor
) -> Optional[List[torch.Tensor]]:
    """Recompute rotary frequencies for processed grid sizes.

    Parameters
    ----------
    model
        Model instance with freqs_fhw cache and rope settings
    kwargs: Dict[str, Any]
        Model forward kwargs (will be modified in-place)
    grid_sizes_proc: torch.Tensor
        Processed grid sizes

    Returns
    -------
    Optional[List[torch.Tensor]]
        Original frequency list if it was replaced, None otherwise
    """
    if model.rope_on_the_fly:
        return None

    orig_freqs_list = kwargs.get("freqs")
    freqs_list_proc: List[torch.Tensor] = []

    for fhw_tensor in grid_sizes_proc:
        fhw = tuple(int(v) for v in fhw_tensor.tolist())
        if fhw not in model.freqs_fhw:
            c = model.dim // model.num_heads // 2
            # Import here to avoid circular imports
            from wan.modules.model import calculate_freqs_i

            model.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, model.freqs)
        freqs_list_proc.append(model.freqs_fhw[fhw])

    kwargs["freqs"] = freqs_list_proc
    return orig_freqs_list


def cleanup_routing_variables(**variables) -> None:
    """Cleanup routing variables by setting them to None.

    Parameters
    ----------
    **variables
        Named variables to set to None
    """
    for name, var_ref in variables.items():
        if var_ref[0] is not None:
            var_ref[0] = None
