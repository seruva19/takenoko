from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from wan.modules.self_resampling_routing_attention import history_branch_attention
from wan.modules.self_resampling_routing_config import history_routing_in_range


def project_self_attn_qkv_with_rollout_cache(
    *,
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    block_index: int,
    extra_tokens: int,
    sparse_attention: bool,
    batched_rotary: Optional[torch.Tensor],
    enable_rollout_self_attn_kv_cache: bool,
    rollout_self_attn_kv_cache: Optional[dict],
    rollout_history_frame_count: int,
    feature_dim: int,
    q_layer: Any,
    k_layer: Any,
    v_layer: Any,
    norm_q_layer: Any,
    norm_k_layer: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s = x.shape[:2]
    use_self_attn_kv_cache = (
        enable_rollout_self_attn_kv_cache
        and rollout_self_attn_kv_cache is not None
        and int(block_index) >= 0
        and extra_tokens == 0
        and not sparse_attention
        and batched_rotary is None
        and int(rollout_history_frame_count) > 0
    )
    cache_key = f"self_attn_block_{int(block_index)}"
    cache_prefix_tokens = 0
    tokens_per_frame = 0
    k_prefix_cached = None
    v_prefix_cached = None
    if use_self_attn_kv_cache:
        same_grid = (
            grid_sizes.shape[0] > 0
            and bool(torch.all(grid_sizes[:, 1] == grid_sizes[0, 1]).item())
            and bool(torch.all(grid_sizes[:, 2] == grid_sizes[0, 2]).item())
        )
        same_seq = bool(torch.all(seq_lens == seq_lens[0]).item())
        if same_grid and same_seq:
            h0 = int(grid_sizes[0, 1].item())
            w0 = int(grid_sizes[0, 2].item())
            tokens_per_frame = int(h0 * w0)
            cache_prefix_tokens = max(
                0,
                min(
                    int(seq_lens[0].item()),
                    int(rollout_history_frame_count) * tokens_per_frame,
                ),
            )
            if cache_prefix_tokens > 0 and cache_prefix_tokens < s:
                cached = rollout_self_attn_kv_cache.get(cache_key)
                if isinstance(cached, dict):
                    k_c = cached.get("k_prefix")
                    v_c = cached.get("v_prefix")
                    cached_len = int(cached.get("prefix_tokens", 0))
                    if (
                        torch.is_tensor(k_c)
                        and torch.is_tensor(v_c)
                        and cached_len >= cache_prefix_tokens
                        and k_c.dim() == 3
                        and v_c.dim() == 3
                        and k_c.shape[0] == b
                        and v_c.shape[0] == b
                        and k_c.shape[2] == feature_dim
                        and v_c.shape[2] == feature_dim
                        and k_c.device == x.device
                        and v_c.device == x.device
                        and k_c.dtype == x.dtype
                        and v_c.dtype == x.dtype
                    ):
                        k_prefix_cached = k_c[:, :cache_prefix_tokens, :]
                        v_prefix_cached = v_c[:, :cache_prefix_tokens, :]

    q = q_layer(x)
    if k_prefix_cached is not None and v_prefix_cached is not None:
        k_suffix = k_layer(x[:, cache_prefix_tokens:, :])
        v_suffix = v_layer(x[:, cache_prefix_tokens:, :])
        k = torch.cat([k_prefix_cached, k_suffix], dim=1)
        v = torch.cat([v_prefix_cached, v_suffix], dim=1)
    else:
        k = k_layer(x)
        v = v_layer(x)
    q = norm_q_layer(q)
    k = norm_k_layer(k)

    if (
        use_self_attn_kv_cache
        and tokens_per_frame > 0
        and rollout_self_attn_kv_cache is not None
    ):
        cache_store_tokens = max(
            0,
            min(
                int(seq_lens[0].item()),
                (int(rollout_history_frame_count) + 1) * tokens_per_frame,
            ),
        )
        if cache_store_tokens > 0:
            rollout_self_attn_kv_cache[cache_key] = {
                "k_prefix": k[:, :cache_store_tokens, :].detach(),
                "v_prefix": v[:, :cache_store_tokens, :].detach(),
                "prefix_tokens": cache_store_tokens,
                "tokens_per_frame": tokens_per_frame,
            }

    return q, k, v


def maybe_route_history_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    history_routing_config: Optional[Dict[str, Any]],
    training: bool,
    sparse_attention: bool,
    batched_rotary: Optional[torch.Tensor],
    block_index: int,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    extra_tokens: int,
    backend_default: str,
    attn_mode: str,
    split_attn: bool,
    window_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    if (
        history_routing_config is None
        or not training
        or sparse_attention
        or batched_rotary is not None
        or not history_routing_in_range(history_routing_config, block_index)
    ):
        return None

    return history_branch_attention(
        q,
        k,
        v,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        top_k_frames=int(history_routing_config.get("top_k_frames", 5)),
        always_keep_first_frame=bool(
            history_routing_config.get("always_keep_first_frame", True)
        ),
        always_keep_last_frame=bool(
            history_routing_config.get("always_keep_last_frame", False)
        ),
        extra_tokens=extra_tokens,
        backend=str(history_routing_config.get("backend", backend_default)),
        attn_mode=attn_mode,
        split_attn=split_attn,
        window_size=window_size,
    )


def project_cross_attn_qkv_with_rollout_cache(
    *,
    x: torch.Tensor,
    context: torch.Tensor,
    block_index: int,
    enable_rollout_kv_cache: bool,
    rollout_kv_cache: Optional[dict],
    q_layer: Any,
    k_layer: Any,
    v_layer: Any,
    norm_q_layer: Any,
    norm_k_layer: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_layer(x)
    k = None
    v = None
    cached_normed = False
    cache_key = f"cross_attn_block_{int(block_index)}"
    if enable_rollout_kv_cache and rollout_kv_cache is not None:
        cached = rollout_kv_cache.get(cache_key)
        if isinstance(cached, dict):
            k = cached.get("k")
            v = cached.get("v")
            cached_normed = bool(cached.get("k_normed", False))
            if (
                torch.is_tensor(k)
                and torch.is_tensor(v)
                and k.device == context.device
                and v.device == context.device
                and k.dtype == context.dtype
                and v.dtype == context.dtype
            ):
                k = k
                v = v
            else:
                k = None
                v = None

    if k is None or v is None:
        k = k_layer(context)
        v = v_layer(context)
        k = norm_k_layer(k)
        if enable_rollout_kv_cache and rollout_kv_cache is not None:
            rollout_kv_cache[cache_key] = {
                "k": k.detach(),
                "v": v.detach(),
                "k_normed": True,
            }
    q = norm_q_layer(q)
    if not cached_normed:
        k = norm_k_layer(k)
    return q, k, v
