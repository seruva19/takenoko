from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch


@dataclass
class AttentionMetricsConfig:
    enabled: bool = False
    interval: int = 1000
    max_layers_per_step: int = 1
    max_queries_per_head: int = 1024
    topk: int = 16
    log_prefix: str = "attn"
    # Heatmap logging (optional)
    log_heatmap: bool = False
    heatmap_max_heads: int = 1
    heatmap_max_queries: int = 64
    heatmap_log_prefix: str = "attn_hm"


class AttentionMetricsCollector:
    """Lightweight, gated attention metrics collector.

    - Does not modify forward path outputs
    - Computes metrics from detached Q/K for cross-attention
    - Subsamples queries to bound memory
    - Aggregates per-step and exposes scalars for logging
    """

    def __init__(self) -> None:
        self.cfg = AttentionMetricsConfig()
        self._active_this_step: bool = False
        self._layers_collected: int = 0
        self._step_metrics: Dict[str, float] = {}
        self._heatmap_tensor: Optional[torch.Tensor] = None

    def configure_from_args(self, args: object) -> None:
        try:
            self.cfg.enabled = bool(getattr(args, "enable_attention_metrics", False))
            self.cfg.interval = int(
                getattr(args, "attention_metrics_interval", 1000) or 1000
            )
            self.cfg.max_layers_per_step = int(
                getattr(args, "attention_metrics_max_layers", 1) or 1
            )
            self.cfg.max_queries_per_head = int(
                getattr(args, "attention_metrics_max_queries", 1024) or 1024
            )
            self.cfg.topk = int(getattr(args, "attention_metrics_topk", 16) or 16)
            self.cfg.log_prefix = str(
                getattr(args, "attention_metrics_log_prefix", "attn") or "attn"
            )
            # Heatmap-specific options
            self.cfg.log_heatmap = bool(
                getattr(args, "attention_metrics_log_heatmap", False)
            )
            self.cfg.heatmap_max_heads = int(
                getattr(args, "attention_metrics_heatmap_max_heads", 1) or 1
            )
            self.cfg.heatmap_max_queries = int(
                getattr(args, "attention_metrics_heatmap_max_queries", 64) or 64
            )
            self.cfg.heatmap_log_prefix = str(
                getattr(args, "attention_metrics_heatmap_log_prefix", "attn_hm")
                or "attn_hm"
            )
        except Exception:
            # Keep defaults on any parsing issue
            pass

    def begin_step(self, global_step: int) -> None:
        if (
            self.cfg.enabled
            and self.cfg.interval > 0
            and (global_step % self.cfg.interval == 0)
        ):
            self._active_this_step = True
            self._layers_collected = 0
            self._step_metrics = {}
            self._heatmap_tensor = None
        else:
            self._active_this_step = False
            self._layers_collected = 0
            self._step_metrics = {}
            self._heatmap_tensor = None

    def should_collect(self) -> bool:
        if not self._active_this_step:
            return False
        if self._layers_collected >= self.cfg.max_layers_per_step:
            return False
        return True

    @torch.no_grad()
    def collect_cross_attention(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Collect metrics for a cross-attention call.

        Parameters
        ----------
        q : Tensor
            Shape [B, Lq, H, D]
        k : Tensor
            Shape [B, Lk, H, D]
        """
        if not self.should_collect():
            return

        try:
            # Use first batch element; WAN training usually has uniform shapes per batch
            q0 = q[0].detach()  # [Lq, H, D]
            k0 = k[0].detach()  # [Lk, H, D]

            # Bring to [H, L, D]
            q0 = q0.permute(1, 0, 2).contiguous()
            k0 = k0.permute(1, 0, 2).contiguous()

            num_heads, lq, d = q0.shape
            lk = k0.shape[1]

            # Subsample queries evenly up to max_queries_per_head to bound memory
            max_q = max(1, min(self.cfg.max_queries_per_head, lq))
            if max_q < lq:
                # Evenly spaced indices for stability
                idx = (
                    torch.linspace(0, lq - 1, steps=max_q, device=q0.device)
                    .round()
                    .long()
                )
                q_sel = q0[:, idx, :]
            else:
                q_sel = q0

            # Compute scaled dot-product scores and softmax over text tokens
            scale = 1.0 / math.sqrt(float(d))
            # [H, M, D] @ [H, D, Lk] -> [H, M, Lk]
            scores = (
                torch.matmul(
                    q_sel.to(torch.float32), k0.transpose(1, 2).to(torch.float32)
                )
                * scale
            )
            probs = torch.softmax(scores, dim=-1)

            # Metrics per head, then average across heads
            eps = 1e-12
            # Entropy
            entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean(dim=-1)  # [H]
            entropy = entropy.mean().item()
            # Normalize entropy by log(lk) for comparability
            norm_entropy = (
                -(probs * (probs + eps).log()).sum(dim=-1)
                / max(1.0, math.log(max(1, lk)))
            ).mean(dim=-1)
            norm_entropy = norm_entropy.mean().item()

            # Top-k mass
            k_val = max(1, min(self.cfg.topk, lk))
            topk_mass = probs.topk(k_val, dim=-1).values.sum(dim=-1).mean(dim=-1)
            topk_mass = topk_mass.mean().item()

            # Token focus: average of max probability over text tokens
            token_focus = probs.max(dim=-1).values.mean(dim=-1)
            token_focus = token_focus.mean().item()

            # Aggregate into step metrics (average if called multiple layers)
            def _acc(name: str, v: float) -> None:
                if name in self._step_metrics:
                    # Running mean across layers collected this step
                    self._step_metrics[name] = (
                        self._step_metrics[name] * self._layers_collected + v
                    ) / (self._layers_collected + 1)
                else:
                    self._step_metrics[name] = v

            p = self.cfg.log_prefix
            _acc(f"{p}/entropy", float(entropy))
            _acc(f"{p}/entropy_norm", float(norm_entropy))
            _acc(f"{p}/topk_mass@{k_val}", float(topk_mass))
            _acc(f"{p}/token_focus", float(token_focus))

            self._layers_collected += 1

            # Optionally capture a small heatmap once per step (first collected layer)
            try:
                if self.cfg.log_heatmap and self._heatmap_tensor is None:
                    # Average across up to K heads, leaving [M, Lk]
                    heads_to_use = max(
                        1, min(int(self.cfg.heatmap_max_heads), probs.size(0))
                    )
                    heat = probs[:heads_to_use].mean(dim=0)  # [M, Lk]

                    # Downsample queries dimension to at most heatmap_max_queries
                    m = int(heat.size(0))
                    max_q = max(1, min(int(self.cfg.heatmap_max_queries), m))
                    if max_q < m:
                        idx_hm = (
                            torch.linspace(0, m - 1, steps=max_q, device=heat.device)
                            .round()
                            .long()
                        )
                        heat = heat.index_select(0, idx_hm)

                    # Store as CPU float tensor in [0,1]
                    self._heatmap_tensor = (
                        heat.detach().to(torch.float32).clamp(0, 1).cpu()
                    )
            except Exception:
                # Never let heatmap capture affect training
                self._heatmap_tensor = None
        except Exception:
            # Swallow any metric error to avoid interfering with training
            pass

    def get_and_clear_latest_metrics(self) -> Dict[str, float]:
        if not self._active_this_step or self._layers_collected == 0:
            self._step_metrics = {}
            return {}
        out = dict(self._step_metrics)
        self._step_metrics = {}
        return out

    def get_and_clear_latest_heatmap(self) -> Optional[torch.Tensor]:
        if not self._active_this_step or self._heatmap_tensor is None:
            self._heatmap_tensor = None
            return None
        out = self._heatmap_tensor
        self._heatmap_tensor = None
        return out


# Singleton collector used across modules
collector = AttentionMetricsCollector()


def configure_from_args(args: object) -> None:
    collector.configure_from_args(args)


def begin_step(global_step: int) -> None:
    collector.begin_step(global_step)


def should_collect() -> bool:
    return collector.should_collect()


@torch.no_grad()
def collect_cross_attention(q: torch.Tensor, k: torch.Tensor) -> None:
    collector.collect_cross_attention(q, k)


def get_and_clear_latest_metrics() -> Dict[str, float]:
    return collector.get_and_clear_latest_metrics()


def get_and_clear_latest_heatmap() -> Optional[torch.Tensor]:
    return collector.get_and_clear_latest_heatmap()
