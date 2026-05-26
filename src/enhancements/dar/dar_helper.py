from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from common.logger import get_logger


logger = get_logger(__name__)


class _SourceRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.to(torch.float32)
        normed = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed * self.weight.to(normed.dtype)).to(x.dtype)


class _LowRankQuery(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(1, int(rank))
        self.down = nn.Linear(int(dim), rank, bias=False)
        self.up = nn.Linear(rank, int(dim), bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(float(dim)))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class DiffusionAdaptiveRouting(nn.Module):
    """Chunked depth router for Wan transformer residual streams.

    The router stores per-forward sublayer outputs v_i, computes RMS-normalized
    source keys, and replaces the next hidden stream with a softmax-weighted
    aggregation over prior source vectors. Wan blocks expose three residual
    sublayers: self-attention, cross-attention, and FFN.
    """

    sublayers_per_block = 3

    def __init__(self, num_blocks: int, dim: int, args: Any) -> None:
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.dim = int(dim)
        self.total_sublayers = self.num_blocks * self.sublayers_per_block
        self.query_mode = str(getattr(args, "dar_query_mode", "static_timestep"))
        self.chunk_size = int(getattr(args, "dar_chunk_size", 4))
        self.residual_blend = float(getattr(args, "dar_residual_blend", 1.0))
        self.query_temperature = float(getattr(args, "dar_query_temperature", 1.0))
        self.final_aggregate_enabled = bool(
            getattr(args, "dar_final_aggregate", True)
        )
        self.apply_in_eval = bool(getattr(args, "dar_apply_in_eval", False))
        self.norm_eps = float(getattr(args, "dar_norm_eps", 1e-6))
        dynamic_rank = int(getattr(args, "dar_dynamic_rank", 16))

        target_count = self.total_sublayers + 2
        source_count = self.total_sublayers + 1
        self.static_queries = nn.Parameter(torch.empty(target_count, self.dim))
        nn.init.normal_(self.static_queries, std=1.0 / math.sqrt(float(self.dim)))
        self.source_norms = nn.ModuleList(
            [_SourceRMSNorm(self.dim, self.norm_eps) for _ in range(source_count)]
        )
        self.dynamic_queries = nn.ModuleList(
            [_LowRankQuery(self.dim, dynamic_rank) for _ in range(target_count)]
        )

        self._sources: list[torch.Tensor] = []
        self._source_ids: list[int] = []
        self._active = False
        self._last_aggregate_target: Optional[int] = None
        self._warned_lean = False

    @property
    def is_active(self) -> bool:
        return self._active

    def should_apply(self) -> bool:
        return self.apply_in_eval or torch.is_grad_enabled()

    def begin_forward(self, initial_hidden: torch.Tensor) -> None:
        if not self.should_apply():
            self._active = False
            self._sources = []
            self._source_ids = []
            self._last_aggregate_target = None
            return
        self._active = True
        self._sources = [initial_hidden]
        self._source_ids = [0]
        self._last_aggregate_target = None

    def end_forward(self) -> None:
        self._active = False
        self._sources = []
        self._source_ids = []
        self._last_aggregate_target = None

    def append_source(self, source_id: int, value: torch.Tensor) -> None:
        if not self._active:
            return
        source_id = int(source_id)
        if source_id < 1 or source_id > self.total_sublayers:
            raise ValueError(
                f"DAR source id {source_id} is outside [1, {self.total_sublayers}]"
            )
        self._sources.append(value)
        self._source_ids.append(source_id)
        self._last_aggregate_target = None

    def aggregate(
        self,
        target_id: int,
        current_hidden: torch.Tensor,
        timestep_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self._active or not self._sources:
            return current_hidden
        target_id = int(target_id)
        if target_id == self._last_aggregate_target:
            return current_hidden

        selected_positions = self._select_source_positions(target_id)
        if not selected_positions:
            return current_hidden
        if len(selected_positions) == 1 and self.residual_blend < 1.0:
            return current_hidden

        sources = [self._sources[pos] for pos in selected_positions]
        source_ids = [self._source_ids[pos] for pos in selected_positions]
        keys = [
            self.source_norms[min(source_id, len(self.source_norms) - 1)](source)
            for source_id, source in zip(source_ids, sources)
        ]
        stacked_sources = torch.stack(sources, dim=0)
        stacked_keys = torch.stack(keys, dim=0)
        query = self._query_tensor(target_id, current_hidden, timestep_embedding)

        logits = (
            (stacked_keys.to(torch.float32) * query.to(torch.float32).unsqueeze(0))
            .sum(dim=-1)
            .mul(1.0 / math.sqrt(float(self.dim)))
        )
        logits = logits / self.query_temperature
        weights = torch.softmax(logits, dim=0).to(stacked_sources.dtype)
        routed = (weights.unsqueeze(-1) * stacked_sources).sum(dim=0)
        routed = routed.to(current_hidden.dtype)

        if self.residual_blend >= 1.0:
            output = routed
        elif self.residual_blend <= 0.0:
            output = current_hidden
        else:
            output = current_hidden.lerp(routed, self.residual_blend)
        self._last_aggregate_target = target_id
        return output

    def final_aggregate(
        self,
        current_hidden: torch.Tensor,
        timestep_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.final_aggregate_enabled:
            return current_hidden
        return self.aggregate(
            self.total_sublayers + 1,
            current_hidden,
            timestep_embedding,
        )

    def _select_source_positions(self, target_id: int) -> list[int]:
        if self.chunk_size <= 1:
            return [
                pos
                for pos, source_id in enumerate(self._source_ids)
                if 0 <= source_id < target_id
            ]

        current_chunk = max(0, (target_id - 1) // self.chunk_size)
        selected: list[int] = []
        for pos, source_id in enumerate(self._source_ids):
            if source_id < 0 or source_id >= target_id:
                continue
            if source_id == 0:
                selected.append(pos)
                continue
            source_chunk = (source_id - 1) // self.chunk_size
            if source_chunk == current_chunk:
                selected.append(pos)
            elif source_id % self.chunk_size == 0:
                selected.append(pos)
        return selected

    def _query_tensor(
        self,
        target_id: int,
        current_hidden: torch.Tensor,
        timestep_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_idx = min(max(0, target_id - 1), self.static_queries.shape[0] - 1)
        if self.query_mode == "dynamic":
            query = self.dynamic_queries[query_idx](current_hidden)
        else:
            query = self.static_queries[query_idx].view(1, 1, self.dim).to(
                device=current_hidden.device,
                dtype=current_hidden.dtype,
            )
            query = query.expand_as(current_hidden)
            if self.query_mode == "static_timestep":
                timestep_query = self._coerce_timestep_embedding(
                    timestep_embedding,
                    current_hidden,
                )
                if timestep_query is not None:
                    query = query + timestep_query
        return query

    @staticmethod
    def _coerce_timestep_embedding(
        timestep_embedding: Optional[torch.Tensor],
        current_hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if timestep_embedding is None or not torch.is_tensor(timestep_embedding):
            return None
        value = timestep_embedding
        if value.dim() == 4:
            value = value.mean(dim=2)
        elif value.dim() == 3 and value.shape[1] != current_hidden.shape[1]:
            value = value.mean(dim=1)
        if value.dim() == 2:
            value = value.unsqueeze(1)
        if value.dim() != 3 or value.shape[-1] != current_hidden.shape[-1]:
            return None
        if value.shape[1] == 1:
            value = value.expand(-1, current_hidden.shape[1], -1)
        elif value.shape[1] != current_hidden.shape[1]:
            return None
        return value.to(device=current_hidden.device, dtype=current_hidden.dtype)

    def forward_block(
        self,
        block: nn.Module,
        block_index: int,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Any,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        sparse_attention: bool = False,
        batched_rotary: Optional[torch.Tensor] = None,
        extra_tokens: int = 0,
        history_routing_config: Optional[dict[str, Any]] = None,
        enable_rollout_kv_cache: bool = False,
        rollout_kv_cache: Optional[dict[str, Any]] = None,
        enable_rollout_self_attn_kv_cache: bool = False,
        rollout_self_attn_kv_cache: Optional[dict[str, Any]] = None,
        rollout_history_frame_count: int = 0,
        rope_offsets: Optional[torch.Tensor] = None,
        reference_frame_token_counts: Optional[torch.Tensor] = None,
        dynamic_rope_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(getattr(block, "_lean_attn_math", False)) and not self._warned_lean:
            logger.warning(
                "DAR is active; using the standard Wan block path instead of "
                "lean_attn_math."
            )
            self._warned_lean = True

        x_orig_dtype = x.dtype
        s0, s1, s2, s3, s4, s5 = block.get_modulation(e)
        block_offset = int(block_index) * self.sublayers_per_block

        x = self.aggregate(block_offset + 1, x, e)
        q_in = block.norm1(x).to(block.attention_dtype, copy=False)
        fi1 = q_in.addcmul(q_in, s1).add(s0).contiguous()
        y = block.self_attn(
            fi1,
            seq_lens,
            grid_sizes,
            freqs if batched_rotary is None else None,
            sparse_attention=sparse_attention,
            batched_rotary=batched_rotary,
            extra_tokens=extra_tokens,
            rope_offsets=rope_offsets,
            reference_frame_token_counts=reference_frame_token_counts,
            dynamic_rope_scales=dynamic_rope_scales,
            history_routing_config=history_routing_config,
            block_index=block_index,
            enable_rollout_self_attn_kv_cache=enable_rollout_self_attn_kv_cache,
            rollout_self_attn_kv_cache=rollout_self_attn_kv_cache,
            rollout_history_frame_count=rollout_history_frame_count,
        )
        delta = (y * s2).to(x_orig_dtype, copy=False)
        x = x + delta
        self.append_source(block_offset + 1, delta)
        del y, delta

        x = self.aggregate(block_offset + 2, x, e)
        delta = block.cross_attn(
            block.norm3(x).to(block.attention_dtype, copy=False),
            context,
            context_lens,
            enable_rollout_kv_cache=enable_rollout_kv_cache,
            rollout_kv_cache=rollout_kv_cache,
            block_index=block_index,
        ).to(x_orig_dtype, copy=False)
        x = x + delta
        self.append_source(block_offset + 2, delta)
        del delta

        x = self.aggregate(block_offset + 3, x, e)
        ff_in = block.norm2(x).to(block.attention_dtype, copy=False)
        y = block.ffn(
            (ff_in * (1 + s4) + s3).contiguous().to(x_orig_dtype, copy=False)
        )
        delta = (y * s5).to(x_orig_dtype, copy=False)
        x = x + delta
        self.append_source(block_offset + 3, delta)
        del y, delta

        return x.to(x_orig_dtype, copy=False)


class DarRoutingHelper(nn.Module):
    def __init__(self, transformer: nn.Module, args: Any) -> None:
        super().__init__()
        self.args = args
        self.enabled = bool(getattr(args, "enable_dar", False))
        self.transformer = self._unwrap_model(transformer)
        self.router: Optional[DiffusionAdaptiveRouting] = None
        self._attached_blocks: list[nn.Module] = []

        if not self.enabled:
            return
        blocks = self._locate_blocks(self.transformer)
        dim = int(getattr(self.transformer, "dim", 0) or 0)
        if dim <= 0 and blocks:
            dim = int(getattr(blocks[0], "dim", 0) or 0)
        if dim <= 0:
            raise ValueError("DAR could not infer Wan hidden dimension.")
        self.router = DiffusionAdaptiveRouting(len(blocks), dim, args)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _locate_blocks(model: nn.Module) -> list[nn.Module]:
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("DAR requires a WanModel with a .blocks ModuleList.")
        return list(blocks)

    @staticmethod
    def _iter_block_lists(model: nn.Module) -> list[Sequence[nn.Module]]:
        block_lists: list[Sequence[nn.Module]] = []
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            block_lists.append(blocks)
        segment_blocks = getattr(model, "bfm_segment_blocks", None)
        if segment_blocks is not None:
            for segment in segment_blocks:
                block_lists.append(segment)
        return block_lists

    def setup_hooks(self) -> None:
        if not self.enabled or self.router is None:
            return
        self.remove_hooks()
        try:
            device = next(self.transformer.parameters()).device
            self.router.to(device=device)
        except StopIteration:
            pass
        setattr(self.transformer, "_dar_router", self.router)
        for block_list in self._iter_block_lists(self.transformer):
            for idx, block in enumerate(block_list):
                setattr(block, "_dar_router", self.router)
                setattr(block, "_dar_block_index", idx)
                if bool(getattr(self.args, "dar_disable_block_checkpointing", True)):
                    try:
                        block.disable_gradient_checkpointing()
                    except Exception:
                        setattr(block, "gradient_checkpointing", False)
                self._attached_blocks.append(block)
        logger.info(
            "DAR routing attached to %d Wan blocks (query_mode=%s, chunk_size=%d).",
            len(self._attached_blocks),
            self.router.query_mode,
            self.router.chunk_size,
        )

    def remove_hooks(self) -> None:
        for block in self._attached_blocks:
            if getattr(block, "_dar_router", None) is self.router:
                try:
                    delattr(block, "_dar_router")
                except Exception:
                    pass
        self._attached_blocks = []
        if getattr(self.transformer, "_dar_router", None) is self.router:
            try:
                delattr(self.transformer, "_dar_router")
            except Exception:
                pass

    def get_trainable_params(self) -> list[nn.Parameter]:
        if not self.enabled or self.router is None:
            return []
        return [param for param in self.router.parameters() if param.requires_grad]


def maybe_add_dar_params(
    trainable_params: list[Any],
    lr_descriptions: list[Any],
    helper: Optional[DarRoutingHelper],
    args: Any,
) -> None:
    if helper is None:
        return
    params = helper.get_trainable_params()
    if not params:
        logger.warning("DAR enabled but no trainable router parameters were found.")
        return
    lr = float(getattr(args, "learning_rate", 0.0)) * float(
        getattr(args, "dar_lr_scale", 1.0)
    )
    trainable_params.append({"params": params, "lr": lr})
    lr_descriptions.append("dar_router")
