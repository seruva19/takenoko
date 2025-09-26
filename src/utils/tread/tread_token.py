import torch
from typing import Tuple


def build_batched_rotary_from_freqs(
    freqs_list: list[torch.Tensor], ids_shuffle: torch.Tensor
) -> torch.Tensor:
    """Construct a batched rotary tensor after applying ids_shuffle.

    freqs_list: list of [L, 1, D] per sample (cached per-(F,H,W) rotary multipliers)
    ids_shuffle: [B, L_full] long indices mapping full-token order to kept-token order
    Returns: [B, S_keep, D] rotary tensor aligned to routed tokens
    """
    B = ids_shuffle.size(0)
    full_rope = [f.squeeze(1) for f in freqs_list]  # (L, D)
    full_rope = torch.stack(full_rope, dim=0).to(device=ids_shuffle.device)  # (B, L, D)
    shuf = torch.take_along_dim(
        full_rope,
        ids_shuffle.unsqueeze(-1).expand(B, -1, full_rope.size(-1)),
        dim=1,
    )
    return shuf


def slice_e0_for_token_route(
    e0: torch.Tensor,
    ids_shuffle: torch.Tensor,
    s_keep: int,
    broadcast_time_embed: bool,
    strict_checks: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """Slice per-token e0 to kept tokens using ids_shuffle.

    e0: [B, L, 6, C] or [B, 1, 6, C] if broadcasted
    ids_shuffle: [B, L_full]
    s_keep: kept routed tokens length
    broadcast_time_embed: when True, returns e0 unchanged and None as saved
    strict_checks: when True, validates dtypes/shapes/bounds
    Returns: (saved_full_e0 or None, e0_proc)
    """
    if not isinstance(e0, torch.Tensor) or e0.dim() != 4:
        return None, e0
    if broadcast_time_embed:
        return None, e0
    if strict_checks:
        assert e0.dtype == torch.float32
        assert e0.size(2) == 6
        max_idx = int(ids_shuffle.max().item())
        assert max_idx < e0.size(1)
    saved = e0
    B, L_full, C6, C = e0.size()
    # Ensure indices live on the same device as the tensor we gather from
    ids_shuffle = ids_shuffle.to(device=e0.device)
    gather_idx = ids_shuffle.unsqueeze(-1).unsqueeze(-1).expand(B, L_full, C6, C)
    e_shuf = torch.take_along_dim(e0, gather_idx, dim=1)
    e_proc = e_shuf[:, :s_keep, :, :]
    return saved, e_proc


def restore_e0_after_route(
    saved_full_e0: torch.Tensor | None, e_kwarg: torch.Tensor
) -> torch.Tensor:
    """Restore original e0 if we sliced it; otherwise keep current.

    saved_full_e0: original [B, L, 6, C] or None
    e_kwarg: current e argument in kwargs
    """
    if saved_full_e0 is not None:
        return saved_full_e0
    return e_kwarg


def normalize_routes_with_neg_indices(
    routes: list[dict], total_layers: int
) -> list[dict]:
    """Convert negative start/end indices to positive layer indices in-place.
    Returns a new normalized list.
    """

    def _to_pos(idx: int) -> int:
        return idx if idx >= 0 else total_layers + idx

    return [
        {
            **r,
            "start_layer_idx": _to_pos(int(r["start_layer_idx"])),
            "end_layer_idx": _to_pos(int(r["end_layer_idx"])),
        }
        for r in routes
    ]
