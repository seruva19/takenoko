from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class ContrastiveAttentionHelper:
    def __init__(self, transformer: Any, args: Any, device: torch.device) -> None:
        self.enabled = bool(getattr(args, "enable_contrastive_attention", False))
        self.transformer = transformer
        self.device = device
        self.layer_start = int(getattr(args, "contrastive_attention_layer_start", 0))
        self.layer_end = int(getattr(args, "contrastive_attention_layer_end", 0))
        self.head_limit = int(getattr(args, "contrastive_attention_head_limit", 0))
        self.max_queries = int(getattr(args, "contrastive_attention_max_queries", 128))
        self.temperature = float(
            getattr(args, "contrastive_attention_temperature", 0.1)
        )
        self.interval = int(getattr(args, "contrastive_attention_interval", 1))
        self.weight = float(getattr(args, "contrastive_attention_weight", 0.0))
        self.diversity_weight = float(
            getattr(args, "contrastive_attention_diversity_weight", 0.0)
        )
        self.consistency_weight = float(
            getattr(args, "contrastive_attention_consistency_weight", 0.0)
        )
        self.layer_agg = str(getattr(args, "contrastive_attention_layer_agg", "mean"))
        self.weight_ramp_start = int(
            getattr(args, "contrastive_attention_weight_ramp_start", 0)
        )
        self.weight_ramp_end = int(
            getattr(args, "contrastive_attention_weight_ramp_end", 0)
        )
        self.weight_ramp_type = str(
            getattr(args, "contrastive_attention_weight_ramp_type", "linear")
        ).lower()
        self.focus_tokens = bool(
            getattr(args, "contrastive_attention_focus_tokens", False)
        )
        self.focus_renorm = bool(
            getattr(args, "contrastive_attention_focus_renorm", True)
        )
        self.spatial_focus = bool(
            getattr(args, "contrastive_attention_spatial_focus", False)
        )
        self.spatial_focus_power = float(
            getattr(args, "contrastive_attention_spatial_focus_power", 1.0)
        )
        token_map = getattr(args, "contrastive_attention_token_indices", None)
        self.token_indices_by_concept: Dict[int, list[int]] = (
            token_map if isinstance(token_map, dict) else {}
        )
        self._hooks: list[Any] = []
        self._attn_summaries: Dict[int, torch.Tensor] = {}
        self._attn_summaries_raw: Dict[int, torch.Tensor] = {}
        self._query_concept_maps: Dict[int, torch.Tensor] = {}
        self._summary_counts: Dict[int, int] = {}
        self._active_step = False
        self._warned_capture = False
        self._warned_focus = False
        self._concept_ids: Optional[torch.Tensor] = None
        self._consistency_ema: Dict[int, torch.Tensor] = {}
        self._consistency_decay = float(
            getattr(args, "contrastive_attention_consistency_decay", 0.9)
        )
        self.enable_subject_masks = bool(
            getattr(args, "enable_contrastive_attention_subject_masks", False)
        )
        self.subject_overlap_weight = float(
            getattr(args, "contrastive_attention_subject_overlap_weight", 0.0)
        )
        self.subject_entropy_weight = float(
            getattr(args, "contrastive_attention_subject_entropy_weight", 0.0)
        )
        self.subject_temporal_weight = float(
            getattr(args, "contrastive_attention_subject_temporal_weight", 0.0)
        )
        self.subject_mask_ema_decay = float(
            getattr(args, "contrastive_attention_subject_mask_ema_decay", 0.9)
        )
        self.subject_mask_min_token_count = int(
            getattr(args, "contrastive_attention_subject_mask_min_token_count", 1)
        )
        self._subject_mask_ema: Dict[int, torch.Tensor] = {}
        self._concept_order: list[int] = []

    def setup_hooks(self) -> None:
        if not self.enabled:
            return
        blocks = getattr(self.transformer, "blocks", None)
        if not blocks:
            logger.warning("Contrastive attention: transformer has no blocks.")
            return
        total = len(blocks)
        start = max(0, min(self.layer_start, total))
        end = max(start, min(self.layer_end, total))
        if end <= start:
            logger.warning(
                "Contrastive attention: invalid layer range (%d, %d).", start, end
            )
            return
        for idx in range(start, end):
            block = blocks[idx]
            attn = getattr(block, "cross_attn", None)
            if attn is None:
                continue
            handle = attn.register_forward_hook(
                self._capture_attention(idx, attn)
            )
            self._hooks.append(handle)
        logger.info(
            "Contrastive attention hooks attached (layers %d-%d).",
            start,
            end - 1,
        )

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks = []
        self._attn_summaries = {}
        self._attn_summaries_raw = {}
        self._query_concept_maps = {}
        self._summary_counts = {}
        self._concept_order = []

    def begin_step(self, global_step: int) -> None:
        if not self.enabled or self.interval <= 0:
            self._active_step = False
            self._attn_summaries = {}
            self._attn_summaries_raw = {}
            self._query_concept_maps = {}
            self._summary_counts = {}
            self._concept_ids = None
            self._concept_order = []
            return
        self._active_step = (global_step % self.interval) == 0
        if not self._active_step:
            self._attn_summaries = {}
            self._attn_summaries_raw = {}
            self._query_concept_maps = {}
            self._summary_counts = {}
            self._concept_ids = None
            self._concept_order = []

    def set_concept_ids(self, concept_ids: Optional[torch.Tensor]) -> None:
        if concept_ids is None:
            self._concept_ids = None
            return
        if torch.is_tensor(concept_ids):
            self._concept_ids = concept_ids.detach()
        else:
            self._concept_ids = torch.tensor(concept_ids, device=self.device)

    def _capture_attention(self, layer_idx: int, attn_module: Any):
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if not self._active_step:
                return
            if (
                self.weight <= 0.0
                and self.diversity_weight <= 0.0
                and self.consistency_weight <= 0.0
                and self.subject_overlap_weight <= 0.0
                and self.subject_entropy_weight <= 0.0
                and self.subject_temporal_weight <= 0.0
            ):
                return
            try:
                result = self._compute_attention_summary(attn_module, inputs)
            except Exception as exc:
                if not self._warned_capture:
                    logger.warning(
                        "Contrastive attention capture failed at layer %d (%s).",
                        layer_idx,
                        exc,
                    )
                    self._warned_capture = True
                return
            if result is not None:
                summary, raw_summary, query_concept_map = result
                self._record_summary(
                    layer_idx,
                    summary,
                    raw_summary,
                    query_concept_map=query_concept_map,
                )

        return hook

    def _compute_attention_summary(
        self, attn_module: Any, inputs: tuple[Any, ...]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        if len(inputs) < 2:
            return None
        x = inputs[0]
        context = inputs[1]
        context_lens = inputs[2] if len(inputs) > 2 else None
        if not torch.is_tensor(x) or not torch.is_tensor(context):
            return None

        num_heads = getattr(attn_module, "num_heads", None)
        head_dim = getattr(attn_module, "head_dim", None)
        if num_heads is None or head_dim is None:
            return None

        q = attn_module.q(x)
        k = attn_module.k(context)
        q = attn_module.norm_q(q)
        k = attn_module.norm_k(k)
        bsz = q.shape[0]
        q = q.view(bsz, -1, num_heads, head_dim)
        k = k.view(bsz, -1, num_heads, head_dim)

        if self.head_limit > 0 and num_heads > self.head_limit:
            q = q[:, :, : self.head_limit]
            k = k[:, :, : self.head_limit]
            num_heads = self.head_limit

        max_q = max(1, min(self.max_queries, q.shape[1]))
        if max_q < q.shape[1]:
            idx = (
                torch.linspace(0, q.shape[1] - 1, steps=max_q, device=q.device)
                .round()
                .long()
            )
            q = q[:, idx]

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (
            1.0 / math.sqrt(float(head_dim))
        )
        if context_lens is not None:
            lens = context_lens.to(scores.device)
            lk = scores.shape[-1]
            mask = torch.arange(lk, device=scores.device).unsqueeze(0) >= lens.unsqueeze(
                1
            )
            scores = scores.masked_fill(
                mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        probs = torch.softmax(scores, dim=-1)
        raw_summary = probs.mean(dim=2).mean(dim=1)
        query_concept_map = self._compute_query_concept_map(probs)
        if self.spatial_focus and self.focus_tokens and self.token_indices_by_concept:
            probs = self._apply_token_focus_on_probs(
                probs, self._concept_ids, renorm=self.focus_renorm
            )
            return (
                self._weighted_query_summary(probs, self._concept_ids),
                raw_summary,
                query_concept_map,
            )
        return raw_summary, raw_summary, query_concept_map

    def compute_loss(
        self, concept_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0 or not self._active_step:
            return torch.tensor(0.0, device=self.device)
        if not self._attn_summaries:
            return torch.tensor(0.0, device=self.device)
        if concept_ids is None:
            self._attn_summaries = {}
            return torch.tensor(0.0, device=self.device)

        summaries = [v for _, v in sorted(self._attn_summaries.items())]
        if self.layer_agg == "max":
            attn = torch.stack(summaries, dim=0).max(dim=0).values
        else:
            attn = torch.stack(summaries, dim=0).mean(dim=0)
        raw_summaries = [v for _, v in sorted(self._attn_summaries_raw.items())]
        if raw_summaries:
            raw_attn = torch.stack(raw_summaries, dim=0).mean(dim=0)
        else:
            raw_attn = attn
        self._attn_summaries = {}
        self._attn_summaries_raw = {}
        self._query_concept_maps = {}
        self._summary_counts = {}
        self._concept_order = []

        if not torch.is_tensor(concept_ids):
            concept_ids = torch.tensor(
                concept_ids, device=attn.device, dtype=torch.long
            )
        else:
            concept_ids = concept_ids.to(attn.device).long()

        valid = concept_ids >= 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=attn.device)
        attn = attn[valid]
        labels = concept_ids[valid]
        if self.focus_tokens and not self.spatial_focus:
            attn = self._apply_token_focus(attn, labels)

        loss = self._supervised_contrastive(attn, labels)
        return loss

    def compute_diversity_loss(
        self, concept_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if (
            not self.enabled
            or not self._active_step
            or not self.focus_tokens
            or not self.token_indices_by_concept
        ):
            return torch.tensor(0.0, device=self.device)
        if not self._attn_summaries_raw:
            return torch.tensor(0.0, device=self.device)
        summaries = [v for _, v in sorted(self._attn_summaries_raw.items())]
        raw_attn = torch.stack(summaries, dim=0).mean(dim=0)
        if concept_ids is None:
            return torch.tensor(0.0, device=raw_attn.device)
        labels = (
            concept_ids.to(raw_attn.device).long()
            if torch.is_tensor(concept_ids)
            else torch.tensor(concept_ids, device=raw_attn.device, dtype=torch.long)
        )
        valid = labels >= 0
        if valid.sum() < 1:
            return torch.tensor(0.0, device=raw_attn.device)
        raw_attn = raw_attn[valid]
        labels = labels[valid]
        losses = []
        eps = 1e-8
        for i in range(raw_attn.size(0)):
            label = int(labels[i].item())
            focus_indices = self.token_indices_by_concept.get(label, [])
            if not focus_indices:
                continue
            mask = torch.ones(raw_attn.size(1), device=raw_attn.device, dtype=torch.bool)
            for idx in focus_indices:
                if 0 <= idx < raw_attn.size(1):
                    mask[idx] = False
            non_target = raw_attn[i][mask]
            if non_target.numel() < 2:
                continue
            probs = non_target / non_target.sum().clamp(min=eps)
            entropy = -(probs * (probs + eps).log()).sum()
            losses.append(-entropy)
        if not losses:
            return torch.tensor(0.0, device=raw_attn.device)
        return torch.stack(losses).mean()

    def compute_consistency_loss(
        self, concept_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not self.enabled or not self._active_step:
            return torch.tensor(0.0, device=self.device)
        if not self._attn_summaries:
            return torch.tensor(0.0, device=self.device)
        summaries = [v for _, v in sorted(self._attn_summaries.items())]
        attn = torch.stack(summaries, dim=0).mean(dim=0)
        if concept_ids is None:
            return torch.tensor(0.0, device=attn.device)
        labels = (
            concept_ids.to(attn.device).long()
            if torch.is_tensor(concept_ids)
            else torch.tensor(concept_ids, device=attn.device, dtype=torch.long)
        )
        valid = labels >= 0
        if valid.sum() < 1:
            return torch.tensor(0.0, device=attn.device)
        attn = attn[valid]
        labels = labels[valid]
        losses = []
        for i in range(attn.size(0)):
            label = int(labels[i].item())
            ema = self._consistency_ema.get(label)
            current = attn[i]
            if ema is not None:
                losses.append(F.mse_loss(current, ema))
            decay = max(0.0, min(self._consistency_decay, 0.999))
            self._consistency_ema[label] = (
                current.detach()
                if ema is None
                else ema * decay + current.detach() * (1.0 - decay)
            )
        if not losses:
            return torch.tensor(0.0, device=attn.device)
        return torch.stack(losses).mean()

    def get_weight(self, global_step: Optional[int]) -> float:
        base = float(self.weight)
        if global_step is None:
            return base
        if self.weight_ramp_end <= self.weight_ramp_start:
            return base
        progress = (global_step - self.weight_ramp_start) / float(
            self.weight_ramp_end - self.weight_ramp_start
        )
        progress = max(0.0, min(1.0, progress))
        if self.weight_ramp_type == "cosine":
            factor = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            factor = progress
        return base * factor

    def _record_summary(
        self,
        layer_idx: int,
        summary: torch.Tensor,
        raw_summary: torch.Tensor,
        *,
        query_concept_map: Optional[torch.Tensor],
    ) -> None:
        if layer_idx not in self._attn_summaries:
            self._attn_summaries[layer_idx] = summary
            self._attn_summaries_raw[layer_idx] = raw_summary
            if query_concept_map is not None:
                self._query_concept_maps[layer_idx] = query_concept_map
            self._summary_counts[layer_idx] = 1
            return
        count = self._summary_counts.get(layer_idx, 1)
        prev = self._attn_summaries[layer_idx]
        self._attn_summaries[layer_idx] = (prev * count + summary) / float(
            count + 1
        )
        prev_raw = self._attn_summaries_raw[layer_idx]
        self._attn_summaries_raw[layer_idx] = (
            (prev_raw * count + raw_summary) / float(count + 1)
        )
        if query_concept_map is not None:
            prev_query_map = self._query_concept_maps.get(layer_idx)
            if (
                prev_query_map is not None
                and prev_query_map.shape == query_concept_map.shape
            ):
                self._query_concept_maps[layer_idx] = (
                    prev_query_map * count + query_concept_map
                ) / float(count + 1)
            else:
                self._query_concept_maps[layer_idx] = query_concept_map
        self._summary_counts[layer_idx] = count + 1

    def _compute_query_concept_map(self, probs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_subject_masks or not self.token_indices_by_concept:
            return None
        concept_order = sorted(int(k) for k in self.token_indices_by_concept.keys())
        if not concept_order:
            return None
        avg_probs = probs.mean(dim=1)
        concept_maps: list[torch.Tensor] = []
        for concept_id in concept_order:
            indices = self.token_indices_by_concept.get(concept_id, [])
            valid = [i for i in indices if 0 <= i < avg_probs.size(-1)]
            if len(valid) < self.subject_mask_min_token_count:
                concept_maps.append(
                    torch.zeros(
                        avg_probs.size(0),
                        avg_probs.size(1),
                        device=avg_probs.device,
                        dtype=avg_probs.dtype,
                    )
                )
                continue
            concept_maps.append(avg_probs[:, :, valid].sum(dim=-1))
        if not concept_maps:
            return None
        stacked = torch.stack(concept_maps, dim=-1)
        denom = stacked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self._concept_order = concept_order
        return stacked / denom

    def compute_subject_mask_losses(
        self, concept_ids: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        if (
            not self.enabled
            or not self._active_step
            or not self.enable_subject_masks
            or not self._query_concept_maps
            or not self._concept_order
        ):
            return zero, zero, zero
        if concept_ids is None:
            return zero, zero, zero

        query_maps = [v for _, v in sorted(self._query_concept_maps.items())]
        query_map = torch.stack(query_maps, dim=0).mean(dim=0)  # [B, Q, C]
        labels = (
            concept_ids.to(query_map.device).long()
            if torch.is_tensor(concept_ids)
            else torch.tensor(concept_ids, device=query_map.device, dtype=torch.long)
        )
        valid = labels >= 0
        if valid.sum() < 1:
            return zero, zero, zero
        query_map = query_map[valid]
        labels = labels[valid]

        overlap_losses: list[torch.Tensor] = []
        entropy_losses: list[torch.Tensor] = []
        temporal_losses: list[torch.Tensor] = []
        eps = 1e-8
        concept_index_map = {
            concept_id: idx for idx, concept_id in enumerate(self._concept_order)
        }

        for i in range(query_map.size(0)):
            concept_id = int(labels[i].item())
            concept_idx = concept_index_map.get(concept_id)
            if concept_idx is None:
                continue
            probs = query_map[i]
            target = probs[:, concept_idx]
            non_target = probs.sum(dim=-1) - target
            overlap_losses.append((target * non_target).mean())
            entropy_losses.append(
                -(probs * (probs + eps).log()).sum(dim=-1).mean()
            )

            target_dist = target / target.sum().clamp(min=eps)
            ema = self._subject_mask_ema.get(concept_id)
            if ema is not None and ema.shape == target_dist.shape:
                temporal_losses.append(F.mse_loss(target_dist, ema))
            decay = max(0.0, min(self.subject_mask_ema_decay, 0.999))
            self._subject_mask_ema[concept_id] = (
                target_dist.detach()
                if ema is None or ema.shape != target_dist.shape
                else ema * decay + target_dist.detach() * (1.0 - decay)
            )

        overlap_loss = (
            torch.stack(overlap_losses).mean() if overlap_losses else zero.clone()
        )
        entropy_loss = (
            torch.stack(entropy_losses).mean() if entropy_losses else zero.clone()
        )
        temporal_loss = (
            torch.stack(temporal_losses).mean() if temporal_losses else zero.clone()
        )
        return overlap_loss, entropy_loss, temporal_loss

    def _apply_token_focus_on_probs(
        self,
        probs: torch.Tensor,
        labels: Optional[torch.Tensor],
        *,
        renorm: bool,
    ) -> torch.Tensor:
        if labels is None:
            return probs
        labels = labels.to(probs.device).long()
        focused = probs
        for concept_id, indices in self.token_indices_by_concept.items():
            if not indices:
                continue
            mask = labels == int(concept_id)
            if not torch.any(mask):
                continue
            token_mask = torch.zeros(
                probs.size(-1), device=probs.device, dtype=probs.dtype
            )
            valid_indices = [i for i in indices if 0 <= i < probs.size(-1)]
            if not valid_indices:
                continue
            token_mask[valid_indices] = 1.0
            focused = focused.clone()
            focused[mask] = focused[mask] * token_mask
        if renorm:
            denom = focused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            focused = focused / denom
        return focused

    def _weighted_query_summary(
        self, probs: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if labels is None:
            return probs.mean(dim=2).mean(dim=1)
        labels = labels.to(probs.device).long()
        summaries = []
        for i in range(probs.size(0)):
            label = int(labels[i].item())
            indices = self.token_indices_by_concept.get(label, [])
            if not indices:
                summaries.append(probs[i].mean(dim=1).mean(dim=0))
                continue
            valid_indices = [idx for idx in indices if 0 <= idx < probs.size(-1)]
            if not valid_indices:
                summaries.append(probs[i].mean(dim=1).mean(dim=0))
                continue
            focus = probs[i, :, :, valid_indices].sum(dim=-1)
            focus = focus.mean(dim=0)
            focus = focus.clamp(min=0.0) ** max(self.spatial_focus_power, 0.0)
            denom = focus.sum().clamp(min=1e-8)
            weights = (focus / denom).view(1, -1, 1)
            weighted = (probs[i] * weights).sum(dim=1).mean(dim=0)
            summaries.append(weighted)
        return torch.stack(summaries, dim=0)

    def _apply_token_focus(
        self, attn: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if not self.token_indices_by_concept:
            if not self._warned_focus:
                logger.warning(
                    "Contrastive attention focus enabled, but no token indices configured."
                )
                self._warned_focus = True
            return attn

        focused = attn.clone()
        used_mask = torch.zeros(attn.size(0), device=attn.device, dtype=torch.bool)
        for concept_id, indices in self.token_indices_by_concept.items():
            try:
                concept_id_int = int(concept_id)
            except Exception:
                continue
            mask = labels == concept_id_int
            if not torch.any(mask):
                continue
            valid_indices = [i for i in indices if 0 <= i < attn.size(1)]
            if not valid_indices:
                continue
            token_mask = torch.zeros(attn.size(1), device=attn.device)
            token_mask[valid_indices] = 1.0
            focused[mask] = attn[mask] * token_mask
            used_mask |= mask

        if not torch.all(used_mask):
            focused[~used_mask] = attn[~used_mask]
        if self.focus_renorm:
            denom = focused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            focused = focused / denom
        return focused

    def _supervised_contrastive(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)
        embeddings = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / max(self.temperature, 1e-8)

        mask = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -float("inf"))
        labels = labels.view(-1, 1)
        positive = labels.eq(labels.T) & ~mask
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)
        pos = (exp_sim * positive).sum(dim=1)
        valid = pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return (-torch.log(pos[valid] / denom[valid])).mean()
