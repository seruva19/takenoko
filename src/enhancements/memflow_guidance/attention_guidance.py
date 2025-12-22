from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import types

import torch
import torch.nn.functional as F

from common.logger import get_logger
from wan.modules.attention import attention as wan_attention

from .collector import MemFlowGuidanceCollector
from .config import MemFlowGuidanceConfig

logger = get_logger(__name__)


def install_memflow_guidance_on_attention(
    model: torch.nn.Module,
    config: MemFlowGuidanceConfig,
    collector: MemFlowGuidanceCollector,
) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "WanSelfAttention":
            continue
        if getattr(module, "_memflow_guidance_installed", False):
            continue
        module._memflow_guidance_installed = True
        module._memflow_guidance_config = config
        module._memflow_guidance_collector = collector
        module._memflow_guidance_original_forward = module.forward
        module.forward = types.MethodType(_memflow_guidance_forward, module)
        patched += 1
    return patched


def remove_memflow_guidance_from_attention(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        if not getattr(module, "_memflow_guidance_installed", False):
            continue
        original = getattr(module, "_memflow_guidance_original_forward", None)
        if original is not None:
            module.forward = original
            restored += 1
        for attr in (
            "_memflow_guidance_installed",
            "_memflow_guidance_config",
            "_memflow_guidance_collector",
            "_memflow_guidance_original_forward",
        ):
            if hasattr(module, attr):
                delattr(module, attr)
    return restored


def _memflow_guidance_forward(
    self,
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    sparse_attention: bool = False,
    batched_rotary: Optional[torch.Tensor] = None,
):
    base_out = self._memflow_guidance_original_forward(
        x,
        seq_lens,
        grid_sizes,
        freqs,
        sparse_attention=sparse_attention,
        batched_rotary=batched_rotary,
    )

    collector: Optional[MemFlowGuidanceCollector] = getattr(
        self, "_memflow_guidance_collector", None
    )
    config: Optional[MemFlowGuidanceConfig] = getattr(
        self, "_memflow_guidance_config", None
    )

    if (
        collector is None
        or config is None
        or not collector.enabled
        or not config.enable_memflow_guidance
        or not self.training
        or sparse_attention
        or batched_rotary is not None
    ):
        return base_out

    try:
        guidance_loss = _compute_memflow_guidance_loss(
            self=self,
            x=x,
            grid_sizes=grid_sizes,
            freqs=freqs,
            base_out=base_out,
            config=config,
        )
        if guidance_loss is not None:
            collector.record(guidance_loss)
    except Exception as exc:
        if collector.warn_once():
            logger.warning("MemFlow guidance skipped due to error: %s", exc)
    return base_out


def _compute_memflow_guidance_loss(
    *,
    self,
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    base_out: torch.Tensor,
    config: MemFlowGuidanceConfig,
) -> Optional[torch.Tensor]:
    if grid_sizes is None or grid_sizes.numel() == 0:
        return None
    if grid_sizes.dim() != 2 or grid_sizes.size(1) != 3:
        return None
    if torch.any(grid_sizes[0] != grid_sizes).item():
        return None

    t_frames, grid_h, grid_w = [int(v) for v in grid_sizes[0].tolist()]
    if t_frames < config.memflow_guidance_min_frames:
        return None

    frame_tokens = grid_h * grid_w
    if frame_tokens <= 0:
        return None

    bsz, seq_tokens = x.shape[:2]
    if seq_tokens != t_frames * frame_tokens:
        return None

    num_heads = int(self.num_heads)
    head_dim = int(self.head_dim)

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    q = q.view(bsz, seq_tokens, num_heads, head_dim)
    k = k.view(bsz, seq_tokens, num_heads, head_dim)
    v = v.view(bsz, seq_tokens, num_heads, head_dim)

    from wan.modules.model import rope_apply, rope_apply_inplace_cached

    if (
        bool(getattr(self, "use_comfy_rope", False))
        and getattr(self, "rope_func", "default") == "comfy"
    ):
        try:
            q_rope, k_rope = self.comfyrope(q, k, freqs)  # type: ignore[attr-defined]
            q, k = q_rope, k_rope
        except Exception:
            rope_apply_inplace_cached(q, grid_sizes, freqs)
            rope_apply_inplace_cached(k, grid_sizes, freqs)
    elif getattr(self, "rope_on_the_fly", False):
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
    else:
        rope_apply_inplace_cached(q, grid_sizes, freqs)
        rope_apply_inplace_cached(k, grid_sizes, freqs)

    mem_out = _memory_guided_attention(
        q=q,
        k=k,
        v=v,
        t_frames=t_frames,
        frame_tokens=frame_tokens,
        config=config,
        attn_mode=self.attn_mode,
        split_attn=self.split_attn,
    )
    if mem_out is None:
        return None

    mem_out = mem_out.flatten(2)
    mem_out = self.o(mem_out)

    base_out = base_out.to(dtype=mem_out.dtype)
    return F.mse_loss(mem_out, base_out, reduction="mean")


def _memory_guided_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t_frames: int,
    frame_tokens: int,
    config: MemFlowGuidanceConfig,
    attn_mode: str,
    split_attn: bool,
) -> Optional[torch.Tensor]:
    bsz = q.shape[0]
    output = torch.zeros_like(q)

    local_window = config.memflow_guidance_local_attn_size
    record_interval = max(1, int(config.memflow_guidance_record_interval))
    bank_size = max(1, int(config.memflow_guidance_bank_size))
    top_k = max(0, int(config.memflow_guidance_top_k))

    for t in range(t_frames):
        q_frame = _slice_frame(q, t, frame_tokens)
        local_frames = _local_frame_indices(t, local_window, t_frames)
        bank_frames = _bank_frame_indices(
            t, record_interval=record_interval, bank_size=bank_size, local_frames=local_frames
        )
        if top_k > 0 and len(bank_frames) > top_k:
            bank_frames = _select_topk_frames(
                q_frame=q_frame, k=k, frame_indices=bank_frames, frame_tokens=frame_tokens, top_k=top_k
            )

        k_cat, v_cat = _gather_bank_and_local(
            k=k,
            v=v,
            bank_frames=bank_frames,
            local_frames=local_frames,
            frame_tokens=frame_tokens,
        )
        if k_cat is None or v_cat is None:
            return None

        y_frame = wan_attention(
            [q_frame, k_cat, v_cat],
            k_lens=None,
            window_size=(-1, -1),
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        output[:, t * frame_tokens : (t + 1) * frame_tokens] = y_frame

    return output


def _slice_frame(tensor: torch.Tensor, t: int, frame_tokens: int) -> torch.Tensor:
    start = t * frame_tokens
    end = start + frame_tokens
    return tensor[:, start:end]


def _local_frame_indices(t: int, local_window: int, t_frames: int) -> List[int]:
    if local_window == -1:
        return list(range(0, t + 1))
    if local_window < 1:
        return [t]
    start = max(0, t - local_window + 1)
    return list(range(start, t + 1))


def _bank_frame_indices(
    t: int,
    *,
    record_interval: int,
    bank_size: int,
    local_frames: Iterable[int],
) -> List[int]:
    candidates = [idx for idx in range(0, t) if idx % record_interval == 0]
    local_set = set(local_frames)
    filtered = [idx for idx in candidates if idx not in local_set]
    if bank_size <= 0:
        return []
    return filtered[-bank_size:]


def _select_topk_frames(
    *,
    q_frame: torch.Tensor,
    k: torch.Tensor,
    frame_indices: List[int],
    frame_tokens: int,
    top_k: int,
) -> List[int]:
    if top_k <= 0 or len(frame_indices) <= top_k:
        return frame_indices
    q_mean = q_frame.mean(dim=1)  # [B, H, D]
    scores = []
    for idx in frame_indices:
        k_frame = _slice_frame(k, idx, frame_tokens)
        k_mean = k_frame.mean(dim=1)
        score = (q_mean * k_mean).sum(dim=-1).mean(dim=1).mean()
        scores.append(score)
    scores_tensor = torch.stack(scores)
    topk = torch.topk(scores_tensor, k=top_k, largest=True).indices.tolist()
    return [frame_indices[i] for i in sorted(topk)]


def _gather_bank_and_local(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    bank_frames: List[int],
    local_frames: List[int],
    frame_tokens: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    frame_slices = []
    for idx in bank_frames + local_frames:
        start = idx * frame_tokens
        end = start + frame_tokens
        frame_slices.append(slice(start, end))
    if not frame_slices:
        return None, None
    k_cat = torch.cat([k[:, s] for s in frame_slices], dim=1)
    v_cat = torch.cat([v[:, s] for s in frame_slices], dim=1)
    return k_cat, v_cat
