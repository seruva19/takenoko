"""LoRWeB-style LoRA basis mixing module for WAN training.

This module implements a LoRA basis where each adapter layer contains multiple
rank-r LoRA components. A per-sample query produces mixing coefficients over
the basis, and the mixed low-rank update is applied at runtime.

The feature is fully opt-in through `network_module = "networks.lorweb_lora_wan"`
and network_args keys prefixed with `lorweb_`.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from networks.lora_wan import LoRANetwork, WAN_TARGET_REPLACE_MODULES

logger = get_logger(__name__)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
        return default
    return bool(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_patterns(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return None


class LoRWeBLoRAModule(nn.Module):
    """LoRA basis module with query-conditioned basis mixing."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        lorweb_basis_size: int = 32,
        lorweb_keys_dim: int = 128,
        lorweb_heads: int = 1,
        lorweb_softmax: bool = True,
        lorweb_mixing_coeffs_type: str = "mean",
        lorweb_query_projection_type: str = "linear",
        lorweb_query_pooling: str = "avg",
        lorweb_query_mode: str = "all_tokens",
        lorweb_query_l2_normalize: bool = False,
        lorweb_external_query: bool = False,
        lorweb_external_query_dim: int = 128,
        lorweb_external_query_mode: str = "global",
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.lora_name = lora_name
        self.enabled = True

        in_dim = getattr(org_module, "in_features", None)
        out_dim = getattr(org_module, "out_features", None)
        if in_dim is None or out_dim is None:
            raise RuntimeError(
                "LoRWeBLoRA only supports Linear-like modules with "
                "`in_features`/`out_features`."
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.lora_dim = int(max(1, int(lora_dim)))
        if isinstance(alpha, torch.Tensor):
            alpha = float(alpha.detach().float().item())
        alpha = float(self.lora_dim if alpha is None or float(alpha) == 0.0 else alpha)
        self.scale = float(alpha) / float(self.lora_dim)
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = float(multiplier)
        self.dropout = float(dropout) if dropout is not None else None
        self.rank_dropout = (
            float(rank_dropout) if rank_dropout is not None else None
        )
        self.module_dropout = (
            float(module_dropout) if module_dropout is not None else None
        )

        self.lorweb_basis_size = int(max(1, int(lorweb_basis_size)))
        self.lorweb_keys_dim = int(max(1, int(lorweb_keys_dim)))
        self.lorweb_heads = int(max(1, int(lorweb_heads)))
        if self.lorweb_keys_dim % self.lorweb_heads != 0:
            raise ValueError(
                f"lorweb_keys_dim ({self.lorweb_keys_dim}) must be divisible by "
                f"lorweb_heads ({self.lorweb_heads})."
            )
        self.lorweb_softmax = bool(lorweb_softmax)
        self.lorweb_mixing_coeffs_type = str(lorweb_mixing_coeffs_type)
        self.lorweb_query_projection_type = str(lorweb_query_projection_type)
        self.lorweb_query_pooling = str(lorweb_query_pooling)
        self.lorweb_query_mode = str(lorweb_query_mode)
        self.lorweb_query_l2_normalize = bool(lorweb_query_l2_normalize)
        self.lorweb_external_query = bool(lorweb_external_query)
        self.lorweb_external_query_mode = str(lorweb_external_query_mode)

        query_dim = self.in_dim
        if self.lorweb_external_query:
            query_dim = int(max(1, int(lorweb_external_query_dim)))
            if self.lorweb_external_query_mode in {
                "triplet_concat",
                "cat-aa'b",
                "cat-paa'b",
                "cat-paa'b3",
            }:
                query_dim *= 3
        self.query_dim = query_dim

        self.lora_down = nn.Parameter(
            torch.empty(self.lorweb_basis_size, self.in_dim, self.lora_dim)
        )
        self.lora_up = nn.Parameter(
            torch.empty(self.lorweb_basis_size, self.lora_dim, self.out_dim)
        )
        self.lora_keys = nn.Parameter(
            torch.empty(self.lorweb_basis_size, self.lorweb_keys_dim)
        )
        if self.lorweb_query_projection_type == "none":
            if self.query_dim != self.lorweb_keys_dim:
                raise ValueError(
                    "lorweb_query_projection_type='none' requires "
                    "query_dim == lorweb_keys_dim."
                )
            self.query_proj: nn.Module = nn.Identity()
        elif self.lorweb_query_projection_type == "linear":
            self.query_proj = nn.Linear(self.query_dim, self.lorweb_keys_dim, bias=False)
        else:
            raise ValueError(
                "lorweb_query_projection_type must be one of {'linear', 'none'}."
            )

        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up)
        nn.init.normal_(self.lora_keys, mean=0.0, std=1.0 / math.sqrt(self.lorweb_keys_dim))

        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.external_query_tensor: Optional[torch.Tensor] = None
        self._last_mixing: Optional[torch.Tensor] = None
        self._last_mixing_live: Optional[torch.Tensor] = None

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def set_external_query(self, query: Optional[torch.Tensor]) -> None:
        if query is None:
            self.external_query_tensor = None
            return
        if query.dim() == 2:
            query = query.unsqueeze(1)
        elif query.dim() != 3:
            self.external_query_tensor = None
            return
        self.external_query_tensor = query

    def _select_internal_tokens(self, x: torch.Tensor, wtext: bool = False) -> torch.Tensor:
        if x.dim() != 3:
            return x
        tokens = x
        if wtext:
            # Mirrors the legacy context split behavior in WAN LoRA modules.
            text_tokens = min(512, int(tokens.shape[1]))
            tokens = tokens[:, text_tokens:, :]
        if tokens.shape[1] <= 1:
            return tokens
        mode = self.lorweb_query_mode
        if mode in {"first_half"}:
            return tokens[:, : tokens.shape[1] // 2, :]
        if mode in {"last_half", "caa'bb'"}:
            return tokens[:, tokens.shape[1] // 2 :, :]
        if mode in {"first_three_quarters"}:
            end = max(1, (tokens.shape[1] * 3) // 4)
            return tokens[:, :end, :]
        if mode == "caa'b":
            selected = tokens[:, tokens.shape[1] // 2 :, :]
            end = max(1, (selected.shape[1] * 3) // 4)
            return selected[:, :end, :]
        if mode == "caa'":
            selected = tokens[:, tokens.shape[1] // 2 :, :]
            end = max(1, selected.shape[1] // 2)
            return selected[:, :end, :]
        return tokens

    def _build_query(self, x: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        if self.lorweb_external_query and self.external_query_tensor is not None:
            return self.external_query_tensor

        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() != 3:
            return None

        selected = self._select_internal_tokens(
            x,
            wtext=bool(kwargs.get("wtext", False)),
        )
        if selected.shape[1] == 0:
            return None
        if self.lorweb_query_pooling == "max":
            pooled = selected.max(dim=1).values
        else:
            pooled = selected.mean(dim=1)
        return pooled.unsqueeze(1)

    def _compute_mixing(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, n_queries, _ = query.shape

        q = query.to(dtype=self.lora_down.dtype, device=self.lora_down.device)
        q = self.query_proj(q)
        k = self.lora_keys.to(dtype=q.dtype, device=q.device)

        if self.lorweb_query_l2_normalize:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        head_dim = self.lorweb_keys_dim // self.lorweb_heads
        q = q.view(batch_size, n_queries, self.lorweb_heads, head_dim).transpose(1, 2)
        keys = k.view(self.lorweb_basis_size, self.lorweb_heads, head_dim)
        keys = keys.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        logits = torch.matmul(q, keys.transpose(-2, -1)) * (head_dim**-0.5)
        if self.lorweb_softmax:
            coeffs = torch.softmax(logits, dim=-1)
        else:
            coeffs = torch.tanh(logits)

        if self.lorweb_mixing_coeffs_type == "sum":
            mixing = coeffs.sum(dim=(1, 2))
            if self.lorweb_softmax:
                mixing = mixing / float(max(1, self.lorweb_heads * n_queries))
            else:
                mixing = mixing.clamp(min=-1.0, max=1.0)
        else:
            mixing = coeffs.mean(dim=(1, 2))
        return mixing

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.enabled:
            return self.org_forward(x, *args, **kwargs)
        org_forwarded = self.org_forward(x, *args, **kwargs)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        query = self._build_query(x, **kwargs)
        if query is None:
            return org_forwarded
        mixing = self._compute_mixing(query)
        self._last_mixing_live = mixing
        self._last_mixing = mixing.detach()

        lora_input = x
        if self.dropout is not None and self.training:
            lora_input = F.dropout(lora_input, p=self.dropout)

        if lora_input.dim() == 3:
            lora_out = torch.einsum(
                "bsi,nir,bn,nro->bso",
                lora_input.to(self.lora_down.dtype),
                self.lora_down,
                mixing,
                self.lora_up,
            )
        elif lora_input.dim() == 2:
            lora_out = torch.einsum(
                "bi,nir,bn,nro->bo",
                lora_input.to(self.lora_down.dtype),
                self.lora_down,
                mixing,
                self.lora_up,
            )
        else:
            return org_forwarded

        scale = self.scale
        if self.rank_dropout is not None and self.training and self.rank_dropout > 0.0:
            lora_out = F.dropout(lora_out, p=self.rank_dropout)
            scale = scale * (1.0 / max(1.0 - self.rank_dropout, 1e-6))

        lora_out = lora_out.to(dtype=org_forwarded.dtype, device=org_forwarded.device)
        return org_forwarded + lora_out * self.multiplier * scale

    def _normalized_mixing(self) -> Optional[torch.Tensor]:
        if self._last_mixing_live is None or self._last_mixing_live.numel() == 0:
            return None
        probs = torch.clamp(self._last_mixing_live, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    def mixing_entropy(self) -> Optional[torch.Tensor]:
        probs = self._normalized_mixing()
        if probs is None:
            return None
        return -(probs * torch.log(probs)).sum(dim=-1).mean()

    def usage_diversity_regularization(self) -> Optional[torch.Tensor]:
        probs = self._normalized_mixing()
        if probs is None:
            return None
        usage = probs.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / float(max(1, usage.numel())))
        return F.mse_loss(usage, uniform)

    def get_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        coeffs: torch.Tensor
        if self._last_mixing is not None and self._last_mixing.numel() > 0:
            coeffs = self._last_mixing.detach().float().mean(dim=0)
        else:
            coeffs = torch.full(
                (self.lorweb_basis_size,),
                fill_value=1.0 / float(self.lorweb_basis_size),
                device=self.lora_down.device,
                dtype=torch.float32,
            )
        delta_io = torch.einsum(
            "n,nir,nro->io",
            coeffs,
            self.lora_down.float(),
            self.lora_up.float(),
        )
        weight = delta_io.transpose(0, 1) * float(self.scale)
        if multiplier is None:
            multiplier = self.multiplier
        return weight * float(multiplier)


class LoRWeBLoRAInfModule(LoRWeBLoRAModule):
    """Inference-safe wrapper with enable/disable toggle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled = True
        self.network = None

    def set_network(self, network: Any) -> None:
        self.network = network

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.enabled:
            return self.org_forward(x, *args, **kwargs)
        return super().forward(x, *args, **kwargs)


class LoRWeBLoRANetwork(LoRANetwork):
    """LoRA network that instantiates LoRWeB-style basis modules."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        lorweb_basis_size: int = 32,
        lorweb_keys_dim: int = 128,
        lorweb_heads: int = 1,
        lorweb_softmax: bool = True,
        lorweb_mixing_coeffs_type: str = "mean",
        lorweb_query_projection_type: str = "linear",
        lorweb_query_pooling: str = "avg",
        lorweb_query_mode: str = "all_tokens",
        lorweb_query_l2_normalize: bool = False,
        lorweb_external_query: bool = False,
        lorweb_external_query_dim: int = 128,
        lorweb_external_query_mode: str = "global",
        lorweb_runtime_source: str = "auto",
        lorweb_external_query_encoder: str = "none",
        lorweb_external_query_model_name: str = "",
        lorweb_external_query_image_size: int = 224,
        lorweb_external_query_cache: bool = False,
        lorweb_external_query_cache_size: int = 512,
        lorweb_external_query_encode_batch_size: int = 0,
        lorweb_external_query_autocast_dtype: str = "none",
        lorweb_external_query_encoder_offload_interval: int = 0,
        lorweb_video_query_frame_strategy: str = "mean",
        lorweb_entropy_reg_lambda: float = 0.0,
        lorweb_entropy_reg_target: Optional[float] = None,
        lorweb_diversity_reg_lambda: float = 0.0,
        lorweb_layer_stats_limit: int = 0,
        lorweb_topk_basis: int = 0,
        module_class: Optional[type] = None,
        **kwargs,
    ) -> None:
        self.lorweb_basis_size = int(max(1, int(lorweb_basis_size)))
        self.lorweb_keys_dim = int(max(1, int(lorweb_keys_dim)))
        self.lorweb_heads = int(max(1, int(lorweb_heads)))
        self.lorweb_softmax = bool(lorweb_softmax)
        self.lorweb_mixing_coeffs_type = str(lorweb_mixing_coeffs_type)
        self.lorweb_query_projection_type = str(lorweb_query_projection_type)
        self.lorweb_query_pooling = str(lorweb_query_pooling)
        self.lorweb_query_mode = str(lorweb_query_mode)
        self.lorweb_query_l2_normalize = bool(lorweb_query_l2_normalize)
        self.lorweb_external_query = bool(lorweb_external_query)
        self.lorweb_external_query_dim = int(max(1, int(lorweb_external_query_dim)))
        self.lorweb_external_query_mode = str(lorweb_external_query_mode)
        self.lorweb_runtime_source = str(lorweb_runtime_source)
        self.lorweb_external_query_encoder = str(lorweb_external_query_encoder).lower()
        self.lorweb_external_query_model_name = str(
            lorweb_external_query_model_name or ""
        )
        self.lorweb_external_query_image_size = int(
            max(8, int(lorweb_external_query_image_size))
        )
        self.lorweb_external_query_cache = bool(lorweb_external_query_cache)
        self.lorweb_external_query_cache_size = int(
            max(1, int(lorweb_external_query_cache_size))
        )
        self.lorweb_external_query_encode_batch_size = int(
            max(0, int(lorweb_external_query_encode_batch_size))
        )
        self.lorweb_external_query_autocast_dtype = str(
            lorweb_external_query_autocast_dtype
        ).strip().lower()
        self.lorweb_external_query_encoder_offload_interval = int(
            max(0, int(lorweb_external_query_encoder_offload_interval))
        )
        self.lorweb_video_query_frame_strategy = str(
            lorweb_video_query_frame_strategy
        ).strip().lower()
        self.lorweb_entropy_reg_lambda = float(lorweb_entropy_reg_lambda)
        self.lorweb_entropy_reg_target = (
            None
            if lorweb_entropy_reg_target is None
            else float(lorweb_entropy_reg_target)
        )
        self.lorweb_diversity_reg_lambda = float(lorweb_diversity_reg_lambda)
        self.lorweb_layer_stats_limit = int(max(0, int(lorweb_layer_stats_limit)))
        self.lorweb_topk_basis = int(max(0, int(lorweb_topk_basis)))
        self._lorweb_encoder_failed = False
        self._lorweb_image_encoder: Optional[nn.Module] = None
        self._lorweb_image_processor: Any = None
        self._lorweb_query_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lorweb_query_cache_hits: int = 0
        self._lorweb_query_cache_misses: int = 0
        self._lorweb_external_query_encode_calls: int = 0

        super().__init__(
            target_replace_modules=target_replace_modules,
            prefix=prefix,
            text_encoders=text_encoders,
            unet=unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            module_class=module_class or self._create_lorweb_lora_module,
            **kwargs,
        )

    def _create_lorweb_lora_module(
        self,
        lora_name,
        org_module,
        multiplier,
        lora_dim,
        alpha,
        **kwargs,
    ):
        return LoRWeBLoRAModule(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            lorweb_basis_size=self.lorweb_basis_size,
            lorweb_keys_dim=self.lorweb_keys_dim,
            lorweb_heads=self.lorweb_heads,
            lorweb_softmax=self.lorweb_softmax,
            lorweb_mixing_coeffs_type=self.lorweb_mixing_coeffs_type,
            lorweb_query_projection_type=self.lorweb_query_projection_type,
            lorweb_query_pooling=self.lorweb_query_pooling,
            lorweb_query_mode=self.lorweb_query_mode,
            lorweb_query_l2_normalize=self.lorweb_query_l2_normalize,
            lorweb_external_query=self.lorweb_external_query,
            lorweb_external_query_dim=self.lorweb_external_query_dim,
            lorweb_external_query_mode=self.lorweb_external_query_mode,
            **kwargs,
        )

    @staticmethod
    def _pool_condition_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.dim() == 5:
            return value.mean(dim=(2, 3, 4))
        if value.dim() == 4:
            return value.mean(dim=(2, 3))
        if value.dim() == 3:
            return value.mean(dim=1)
        if value.dim() == 2:
            return value
        return None

    @staticmethod
    def _pool_triplet_panels(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None or value.dim() not in {4, 5}:
            return None
        if value.dim() == 5:
            # B, C, F, H, W
            _, _, _, h, w = value.shape
            if h < 2 or w < 2:
                return None
            h2 = h // 2
            w2 = w // 2
            a = value[:, :, :, :h2, :w2].mean(dim=(2, 3, 4))
            at = value[:, :, :, :h2, w2:].mean(dim=(2, 3, 4))
            b = value[:, :, :, h2:, :w2].mean(dim=(2, 3, 4))
            return torch.cat([a, at, b], dim=-1)
        # B, C, H, W
        _, _, h, w = value.shape
        if h < 2 or w < 2:
            return None
        h2 = h // 2
        w2 = w // 2
        a = value[:, :, :h2, :w2].mean(dim=(2, 3))
        at = value[:, :, :h2, w2:].mean(dim=(2, 3))
        b = value[:, :, h2:, :w2].mean(dim=(2, 3))
        return torch.cat([a, at, b], dim=-1)

    @staticmethod
    def _fit_last_dim(value: torch.Tensor, target_dim: int) -> torch.Tensor:
        if value.shape[-1] == target_dim:
            return value
        if value.shape[-1] > target_dim:
            return value[..., :target_dim]
        pad = target_dim - value.shape[-1]
        return F.pad(value, (0, pad))

    @staticmethod
    def _mean_frame_from_video(source: torch.Tensor) -> torch.Tensor:
        return source.mean(dim=2)

    def _select_video_frame_from_source(self, source: torch.Tensor) -> torch.Tensor:
        # source shape: [B, C, F, H, W]
        strategy = self.lorweb_video_query_frame_strategy
        frames = source.shape[2]
        if frames <= 1:
            return source[:, :, 0, :, :]
        if strategy == "first":
            return source[:, :, 0, :, :]
        if strategy == "middle":
            return source[:, :, frames // 2, :, :]
        if strategy == "last":
            return source[:, :, -1, :, :]
        if strategy == "motion_weighted":
            diffs = (source[:, :, 1:, :, :] - source[:, :, :-1, :, :]).abs()
            diffs = diffs.mean(dim=(1, 3, 4))  # [B, F-1]
            scores = source.new_zeros((source.shape[0], frames))
            scores[:, 1:] += diffs
            scores[:, :-1] += diffs
            scores = scores + 1e-6
            weights = scores / scores.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return (source * weights[:, None, :, None, None]).sum(dim=2)
        return self._mean_frame_from_video(source)

    def _source_to_images(self, source: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if source is None:
            return None
        x = source.detach()
        if x.dim() == 5:
            # B,C,F,H,W -> B,C,H,W using configured frame strategy.
            x = self._select_video_frame_from_source(x)
        if x.dim() != 4:
            return None
        x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]
        max_val = float(x.max().item())
        min_val = float(x.min().item())
        if min_val < 0.0 or max_val > 1.0:
            x = x.clamp(-1.0, 1.0)
            x = (x + 1.0) * 0.5
        else:
            x = x.clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _split_triplet_panels_images(
        images: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if images.dim() != 4:
            return None
        _, _, h, w = images.shape
        if h < 2 or w < 2:
            return None
        h2 = h // 2
        w2 = w // 2
        a = images[:, :, :h2, :w2]
        at = images[:, :, :h2, w2:]
        b = images[:, :, h2:, :w2]
        return a, at, b

    @staticmethod
    def _sanitize_box(
        box: torch.Tensor,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x0 = int(torch.floor(box[0]).item())
        y0 = int(torch.floor(box[1]).item())
        x1 = int(torch.ceil(box[2]).item())
        y1 = int(torch.ceil(box[3]).item())
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))
        return x0, y0, x1, y1

    def _get_analogy_boxes(
        self,
        images: torch.Tensor,
        analogy_boxes: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if analogy_boxes is None:
            return None
        if analogy_boxes.dim() != 3 or analogy_boxes.shape[1] < 3 or analogy_boxes.shape[2] != 4:
            return None
        if analogy_boxes.shape[0] != images.shape[0]:
            return None
        return analogy_boxes.to(device=images.device, dtype=torch.float32)

    def _split_triplet_panels_by_boxes(
        self,
        images: torch.Tensor,
        analogy_boxes: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        panels_a: List[torch.Tensor] = []
        panels_at: List[torch.Tensor] = []
        panels_b: List[torch.Tensor] = []
        _, _, h, w = images.shape
        for i in range(images.shape[0]):
            box_a = self._sanitize_box(analogy_boxes[i, 0], width=w, height=h)
            box_at = self._sanitize_box(analogy_boxes[i, 1], width=w, height=h)
            box_b = self._sanitize_box(analogy_boxes[i, 2], width=w, height=h)
            xa0, ya0, xa1, ya1 = box_a
            xt0, yt0, xt1, yt1 = box_at
            xb0, yb0, xb1, yb1 = box_b
            panels_a.append(images[i : i + 1, :, ya0:ya1, xa0:xa1])
            panels_at.append(images[i : i + 1, :, yt0:yt1, xt0:xt1])
            panels_b.append(images[i : i + 1, :, yb0:yb1, xb0:xb1])
        if not panels_a:
            return None

        def _stack_resized(panels: List[torch.Tensor]) -> torch.Tensor:
            target_h = max(int(p.shape[-2]) for p in panels)
            target_w = max(int(p.shape[-1]) for p in panels)
            resized = []
            for panel in panels:
                if panel.shape[-2] != target_h or panel.shape[-1] != target_w:
                    panel = F.interpolate(
                        panel,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                resized.append(panel)
            return torch.cat(resized, dim=0)

        return _stack_resized(panels_a), _stack_resized(panels_at), _stack_resized(panels_b)

    @staticmethod
    def _normalize_external_query_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized == "triplet_concat":
            return "cat-aa'b"
        return normalized

    @staticmethod
    def _is_concat_external_mode(mode: str) -> bool:
        return mode in {"cat-aa'b", "cat-paa'b", "cat-paa'b3"}

    def _external_query_target_dim(self) -> int:
        mode = self._normalize_external_query_mode(self.lorweb_external_query_mode)
        if self._is_concat_external_mode(mode):
            return self.lorweb_external_query_dim * 3
        return self.lorweb_external_query_dim

    def _ensure_external_query_encoder(self, device: torch.device) -> bool:
        if self.lorweb_external_query_encoder in {"", "none"}:
            return False
        if self._lorweb_encoder_failed:
            return False
        if self._lorweb_image_encoder is not None:
            return True
        try:
            if self.lorweb_external_query_encoder == "clip":
                from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

                model_name = (
                    self.lorweb_external_query_model_name
                    or "openai/clip-vit-large-patch14"
                )
                encoder = CLIPVisionModelWithProjection.from_pretrained(model_name)
                processor = CLIPImageProcessor.from_pretrained(model_name)
            elif self.lorweb_external_query_encoder == "siglip2":
                from transformers import Siglip2ImageProcessor, Siglip2VisionModel

                model_name = (
                    self.lorweb_external_query_model_name
                    or "google/siglip2-base-patch16-224"
                )
                encoder = Siglip2VisionModel.from_pretrained(model_name)
                processor = Siglip2ImageProcessor.from_pretrained(model_name)
            else:
                logger.warning(
                    "Unknown lorweb_external_query_encoder=%s; disabling external encoder path.",
                    self.lorweb_external_query_encoder,
                )
                self._lorweb_encoder_failed = True
                return False
            encoder.eval()
            encoder.requires_grad_(False)
            self._lorweb_image_encoder = encoder.to(device=device)
            self._lorweb_image_processor = processor
            return True
        except Exception as exc:
            logger.warning(
                "Failed to initialize LoRWeB external query encoder (%s): %s",
                self.lorweb_external_query_encoder,
                exc,
            )
            self._lorweb_encoder_failed = True
            return False

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        images = self._source_to_images(images)
        if images is None:
            return None
        if not self._ensure_external_query_encoder(images.device):
            return None
        encoder = self._lorweb_image_encoder
        processor = self._lorweb_image_processor
        if encoder is None or processor is None:
            return None
        try:
            encoder = encoder.to(device=images.device)
            # Resize before processor to keep memory predictable for large inputs.
            if (
                images.shape[-2] != self.lorweb_external_query_image_size
                or images.shape[-1] != self.lorweb_external_query_image_size
            ):
                images = F.interpolate(
                    images,
                    size=(
                        self.lorweb_external_query_image_size,
                        self.lorweb_external_query_image_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if not self.lorweb_external_query_cache:
                encoded = self._encode_images_batch(
                    images=images,
                    encoder=encoder,
                    processor=processor,
                )
            else:
                encoded = self._encode_images_with_cache(
                    images=images,
                    encoder=encoder,
                    processor=processor,
                )
            self._lorweb_external_query_encode_calls += 1
            self._maybe_offload_external_query_encoder()
            return encoded
        except Exception as exc:
            logger.warning("LoRWeB external image encoding failed: %s", exc)
            return None

    @staticmethod
    def _extract_encoder_embeddings(outputs: Any) -> Optional[torch.Tensor]:
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state.mean(dim=1)
        return None

    def _get_external_query_autocast_dtype(self) -> Optional[torch.dtype]:
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.lorweb_external_query_autocast_dtype)

    def _maybe_offload_external_query_encoder(self) -> None:
        if self.lorweb_external_query_encoder_offload_interval <= 0:
            return
        if self._lorweb_image_encoder is None:
            return
        if (
            self._lorweb_external_query_encode_calls
            % self.lorweb_external_query_encoder_offload_interval
            == 0
        ):
            try:
                self._lorweb_image_encoder = self._lorweb_image_encoder.to(device="cpu")
            except Exception:
                pass

    def _encode_images_batch(
        self,
        images: torch.Tensor,
        encoder: nn.Module,
        processor: Any,
    ) -> Optional[torch.Tensor]:
        batch_size = int(images.shape[0])
        chunk = (
            self.lorweb_external_query_encode_batch_size
            if self.lorweb_external_query_encode_batch_size > 0
            else batch_size
        )
        encoded_parts: List[torch.Tensor] = []
        autocast_dtype = self._get_external_query_autocast_dtype()
        for start in range(0, batch_size, chunk):
            end = min(start + chunk, batch_size)
            sub_images = images[start:end]
            inputs = processor(
                images=[img for img in sub_images],  # list of CHW tensors
                return_tensors="pt",
                do_rescale=False,
            )
            inputs = {k: v.to(device=images.device) for k, v in inputs.items()}
            if autocast_dtype is not None and images.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                ):
                    outputs = encoder(**inputs)
            else:
                outputs = encoder(**inputs)
            embeddings = self._extract_encoder_embeddings(outputs)
            if embeddings is None:
                return None
            encoded_parts.append(embeddings)
        if not encoded_parts:
            return None
        return torch.cat(encoded_parts, dim=0)

    def _make_query_cache_key(self, image: torch.Tensor) -> str:
        img = (image.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).cpu().contiguous()
        digest = hashlib.sha1(img.numpy().tobytes()).hexdigest()
        return (
            f"{self.lorweb_external_query_encoder}|"
            f"{self.lorweb_external_query_model_name}|"
            f"{self.lorweb_external_query_image_size}|"
            f"{tuple(img.shape)}|{digest}"
        )

    def _cache_lookup(
        self,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value = self._lorweb_query_cache.get(key)
        if value is None:
            self._lorweb_query_cache_misses += 1
            return None
        self._lorweb_query_cache.move_to_end(key)
        self._lorweb_query_cache_hits += 1
        return value.to(device=device, dtype=dtype)

    def _cache_store(self, key: str, value: torch.Tensor) -> None:
        self._lorweb_query_cache[key] = value.detach().float().cpu()
        self._lorweb_query_cache.move_to_end(key)
        while len(self._lorweb_query_cache) > self.lorweb_external_query_cache_size:
            self._lorweb_query_cache.popitem(last=False)

    def _encode_images_with_cache(
        self,
        images: torch.Tensor,
        encoder: nn.Module,
        processor: Any,
    ) -> Optional[torch.Tensor]:
        cached_outputs: List[Optional[torch.Tensor]] = [None] * int(images.shape[0])
        cache_keys: List[str] = []
        missing_indices: List[int] = []

        for i in range(images.shape[0]):
            key = self._make_query_cache_key(images[i])
            cache_keys.append(key)
            cached = self._cache_lookup(key, device=images.device, dtype=images.dtype)
            if cached is None:
                missing_indices.append(i)
            else:
                cached_outputs[i] = cached

        if missing_indices:
            uncached = images[missing_indices]
            encoded_uncached = self._encode_images_batch(
                images=uncached,
                encoder=encoder,
                processor=processor,
            )
            if encoded_uncached is None:
                return None
            for local_idx, batch_idx in enumerate(missing_indices):
                encoded = encoded_uncached[local_idx]
                cached_outputs[batch_idx] = encoded
                self._cache_store(cache_keys[batch_idx], encoded)

        if any(v is None for v in cached_outputs):
            return None
        stacked = torch.stack(
            [v for v in cached_outputs if v is not None],  # type: ignore[arg-type]
            dim=0,
        )
        return stacked.to(device=images.device)

    def _build_external_query_from_images(
        self,
        images: torch.Tensor,
        analogy_boxes: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        mode = self._normalize_external_query_mode(self.lorweb_external_query_mode)
        if mode in {"global", "aa'bb'", "caa'bb'"}:
            return self._encode_images(images)

        if mode == "caa'":
            half_h = images.shape[-2] // 2
            if half_h <= 0:
                return None
            return self._encode_images(images[:, :, :half_h, :])

        if mode in {"caa'b", "paa'b", "paa'b3"}:
            half_h = images.shape[-2] // 2
            half_w = images.shape[-1] // 2
            if half_h <= 0 or half_w <= 0:
                return None
            masked = images.clone()
            masked[:, :, half_h:, half_w:] = 0.0
            return self._encode_images(masked)

        boxes = self._get_analogy_boxes(images, analogy_boxes)
        if boxes is not None:
            triplets = self._split_triplet_panels_by_boxes(images, boxes)
        else:
            triplets = self._split_triplet_panels_images(images)
        if triplets is None:
            return None
        panel_a, panel_atag, panel_b = triplets
        emb_a = self._encode_images(panel_a)
        emb_atag = self._encode_images(panel_atag)
        emb_b = self._encode_images(panel_b)
        if emb_a is None or emb_atag is None or emb_b is None:
            return None

        if mode == "ca'-ca":
            return emb_atag - emb_a
        if mode == "ca'-ca+cb":
            return emb_atag - emb_a + emb_b
        if self._is_concat_external_mode(mode):
            return torch.cat([emb_a, emb_atag, emb_b], dim=-1)
        return None

    def _pool_external_query_from_source(
        self,
        source: torch.Tensor,
        analogy_boxes: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        mode = self._normalize_external_query_mode(self.lorweb_external_query_mode)
        if mode in {"global", "aa'bb'", "caa'bb'"}:
            return self._pool_condition_tensor(source)
        if mode == "caa'":
            if source.dim() == 5:
                half_h = source.shape[-2] // 2
                if half_h <= 0:
                    return None
                return self._pool_condition_tensor(source[:, :, :, :half_h, :])
            if source.dim() == 4:
                half_h = source.shape[-2] // 2
                if half_h <= 0:
                    return None
                return self._pool_condition_tensor(source[:, :, :half_h, :])
            return None
        if mode in {"caa'b", "paa'b", "paa'b3"}:
            masked = source.clone()
            half_h = masked.shape[-2] // 2
            half_w = masked.shape[-1] // 2
            if half_h <= 0 or half_w <= 0:
                return None
            if masked.dim() == 5:
                masked[:, :, :, half_h:, half_w:] = 0
            elif masked.dim() == 4:
                masked[:, :, half_h:, half_w:] = 0
            else:
                return None
            return self._pool_condition_tensor(masked)

        pooled_triplet: Optional[torch.Tensor]
        if (
            analogy_boxes is not None
            and source.dim() in {4, 5}
            and analogy_boxes.dim() == 3
            and analogy_boxes.shape[1] >= 3
            and analogy_boxes.shape[2] == 4
            and analogy_boxes.shape[0] == source.shape[0]
        ):
            pooled_values: List[torch.Tensor] = []
            h = source.shape[-2]
            w = source.shape[-1]
            for panel_idx in range(3):
                panel_means: List[torch.Tensor] = []
                for i in range(source.shape[0]):
                    x0, y0, x1, y1 = self._sanitize_box(
                        analogy_boxes[i, panel_idx].to(device=source.device),
                        width=w,
                        height=h,
                    )
                    if source.dim() == 5:
                        panel = source[i : i + 1, :, :, y0:y1, x0:x1].mean(dim=(2, 3, 4))
                    else:
                        panel = source[i : i + 1, :, y0:y1, x0:x1].mean(dim=(2, 3))
                    panel_means.append(panel)
                pooled_values.append(torch.cat(panel_means, dim=0))
            pooled_triplet = torch.cat(pooled_values, dim=-1)
        else:
            pooled_triplet = self._pool_triplet_panels(source)
        if pooled_triplet is None:
            return None
        dim = pooled_triplet.shape[-1] // 3
        part_a = pooled_triplet[:, :dim]
        part_atag = pooled_triplet[:, dim : 2 * dim]
        part_b = pooled_triplet[:, 2 * dim :]
        if mode == "ca'-ca":
            return part_atag - part_a
        if mode == "ca'-ca+cb":
            return part_atag - part_a + part_b
        if self._is_concat_external_mode(mode):
            return pooled_triplet
        return None

    @torch.no_grad()
    def set_lorweb_external_query(self, query: Optional[torch.Tensor]) -> None:
        for lora in getattr(self, "unet_loras", []):
            if hasattr(lora, "set_external_query"):
                lora.set_external_query(query)

    @torch.no_grad()
    def clear_lorweb_external_query(self) -> None:
        self.set_lorweb_external_query(None)

    @torch.no_grad()
    def set_lorweb_runtime_condition(
        self,
        latents: Optional[torch.Tensor],
        control_signal: Optional[torch.Tensor] = None,
        pixels: Optional[torch.Tensor] = None,
        analogy_boxes: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> None:
        del timesteps
        if not self.lorweb_external_query:
            self.clear_lorweb_external_query()
            return

        source: Optional[torch.Tensor] = None
        mode = self.lorweb_runtime_source
        if mode == "control_signal":
            source = control_signal
        elif mode == "pixels":
            source = pixels
        elif mode == "latents":
            source = latents
        else:
            if pixels is not None:
                source = pixels
            elif control_signal is not None:
                source = control_signal
            else:
                source = latents

        if source is None:
            self.clear_lorweb_external_query()
            return
        if analogy_boxes is not None:
            try:
                analogy_boxes = analogy_boxes.to(device=source.device)
            except Exception:
                analogy_boxes = None

        if self.lorweb_external_query_encoder not in {"", "none"}:
            encoded_query = None
            source_images = self._source_to_images(source)
            if source_images is not None:
                encoded_query = self._build_external_query_from_images(
                    source_images,
                    analogy_boxes=analogy_boxes,
                )
            if encoded_query is not None:
                encoded_query = self._fit_last_dim(
                    encoded_query,
                    self._external_query_target_dim(),
                )
                self.set_lorweb_external_query(encoded_query)
                return

        pooled = self._pool_external_query_from_source(
            source,
            analogy_boxes=analogy_boxes,
        )
        if pooled is None:
            self.clear_lorweb_external_query()
            return
        pooled = self._fit_last_dim(pooled, self._external_query_target_dim())
        self.set_lorweb_external_query(pooled)

    def get_lorweb_mixing_stats(self) -> Optional[Dict[str, float]]:
        entropies: List[float] = []
        maxima: List[float] = []
        per_layer_usage: List[torch.Tensor] = []
        per_layer_entropy: List[float] = []
        per_layer_max: List[float] = []
        for lora in getattr(self, "unet_loras", []):
            mixing = getattr(lora, "_last_mixing", None)
            if mixing is None:
                continue
            with torch.no_grad():
                m = mixing.detach().float()
                if m.numel() == 0:
                    continue
                probs = torch.clamp(m, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
                entropies.append(float(entropy.item()))
                maxima.append(float(probs.max(dim=-1).values.mean().item()))
                per_layer_usage.append(probs.mean(dim=0))
                per_layer_entropy.append(float(entropy.item()))
                per_layer_max.append(float(probs.max(dim=-1).values.mean().item()))
        stats: Dict[str, float] = {}
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            stats["lorweb/mixing_entropy"] = mean_entropy
            stats["lorweb/mixing_max_coeff"] = float(sum(maxima) / len(maxima))
            stats["lorweb/mixing_perplexity"] = float(math.exp(mean_entropy))

        if self.lorweb_layer_stats_limit > 0 and per_layer_entropy:
            limit = min(self.lorweb_layer_stats_limit, len(per_layer_entropy))
            for idx in range(limit):
                stats[f"lorweb/layer_{idx}_entropy"] = per_layer_entropy[idx]
                stats[f"lorweb/layer_{idx}_max_coeff"] = per_layer_max[idx]

        if self.lorweb_topk_basis > 0 and per_layer_usage:
            usage = torch.stack(per_layer_usage, dim=0).mean(dim=0)
            k = min(self.lorweb_topk_basis, int(usage.numel()))
            if k > 0:
                top_vals, top_idx = torch.topk(usage, k=k)
                for rank in range(k):
                    stats[f"lorweb/top_basis_{rank}"] = float(top_idx[rank].item())
                    stats[f"lorweb/top_basis_coeff_{rank}"] = float(
                        top_vals[rank].item()
                    )

        if self.lorweb_external_query_cache:
            total = self._lorweb_query_cache_hits + self._lorweb_query_cache_misses
            hit_rate = float(self._lorweb_query_cache_hits / total) if total > 0 else 0.0
            stats["lorweb/query_cache_hit_rate"] = hit_rate
            stats["lorweb/query_cache_size"] = float(len(self._lorweb_query_cache))
        return stats if stats else None

    def get_lorweb_mixing_histogram(self) -> Optional[torch.Tensor]:
        values: List[torch.Tensor] = []
        for lora in getattr(self, "unet_loras", []):
            mixing = getattr(lora, "_last_mixing", None)
            if mixing is None:
                continue
            values.append(mixing.detach().flatten().float())
        if not values:
            return None
        return torch.cat(values, dim=0)


def _extract_module_dims(
    weights_sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], int, int]:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, torch.Tensor] = {}
    basis_size = 32
    keys_dim = 128
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif key.endswith(".lora_down"):
            if value.dim() == 3:
                modules_dim[lora_name] = int(value.shape[-1])
                basis_size = int(value.shape[0])
        elif key.endswith(".lora_keys"):
            if value.dim() == 2:
                keys_dim = int(value.shape[-1])
    return modules_dim, modules_alpha, basis_size, keys_dim


def create_lorweb_lora_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRWeBLoRANetwork:
    del vae
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        conv_alpha = float(conv_alpha) if conv_alpha is not None else 1.0

    rank_dropout = _parse_optional_float(kwargs.get("rank_dropout", None))
    module_dropout = _parse_optional_float(kwargs.get("module_dropout", None))
    loraplus_lr_ratio = _parse_optional_float(kwargs.get("loraplus_lr_ratio", None))

    include_patterns = _parse_patterns(kwargs.get("include_patterns", None))
    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns", None))
    extra_include_patterns = _parse_patterns(kwargs.get("extra_include_patterns", None))
    extra_exclude_patterns = _parse_patterns(kwargs.get("extra_exclude_patterns", None))

    include_time_modules = _parse_bool(kwargs.get("include_time_modules", False), False)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    network = LoRWeBLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="lorweb_lora_unet",
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=float(network_alpha),
        dropout=float(neuron_dropout) if neuron_dropout is not None else None,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        lorweb_basis_size=int(kwargs.get("lorweb_basis_size", 32)),
        lorweb_keys_dim=int(kwargs.get("lorweb_keys_dim", 128)),
        lorweb_heads=int(kwargs.get("lorweb_heads", 1)),
        lorweb_softmax=_parse_bool(kwargs.get("lorweb_softmax", True), True),
        lorweb_mixing_coeffs_type=str(
            kwargs.get("lorweb_mixing_coeffs_type", "mean")
        ),
        lorweb_query_projection_type=str(
            kwargs.get("lorweb_query_projection_type", "linear")
        ),
        lorweb_query_pooling=str(kwargs.get("lorweb_query_pooling", "avg")),
        lorweb_query_mode=str(kwargs.get("lorweb_query_mode", "all_tokens")),
        lorweb_query_l2_normalize=_parse_bool(
            kwargs.get("lorweb_query_l2_normalize", False), False
        ),
        lorweb_external_query=_parse_bool(
            kwargs.get("lorweb_external_query", False), False
        ),
        lorweb_external_query_dim=int(kwargs.get("lorweb_external_query_dim", 128)),
        lorweb_external_query_mode=str(
            kwargs.get("lorweb_external_query_mode", "global")
        ),
        lorweb_runtime_source=str(kwargs.get("lorweb_runtime_source", "auto")),
        lorweb_external_query_encoder=str(
            kwargs.get("lorweb_external_query_encoder", "none")
        ),
        lorweb_external_query_model_name=str(
            kwargs.get("lorweb_external_query_model_name", "")
        ),
        lorweb_external_query_image_size=int(
            kwargs.get("lorweb_external_query_image_size", 224)
        ),
        lorweb_external_query_cache=_parse_bool(
            kwargs.get("lorweb_external_query_cache", False), False
        ),
        lorweb_external_query_cache_size=int(
            kwargs.get("lorweb_external_query_cache_size", 512)
        ),
        lorweb_external_query_encode_batch_size=int(
            kwargs.get("lorweb_external_query_encode_batch_size", 0)
        ),
        lorweb_external_query_autocast_dtype=str(
            kwargs.get("lorweb_external_query_autocast_dtype", "none")
        ),
        lorweb_external_query_encoder_offload_interval=int(
            kwargs.get("lorweb_external_query_encoder_offload_interval", 0)
        ),
        lorweb_video_query_frame_strategy=str(
            kwargs.get("lorweb_video_query_frame_strategy", "mean")
        ),
        lorweb_entropy_reg_lambda=float(
            kwargs.get("lorweb_entropy_reg_lambda", 0.0)
        ),
        lorweb_entropy_reg_target=_parse_optional_float(
            kwargs.get("lorweb_entropy_reg_target", None)
        ),
        lorweb_diversity_reg_lambda=float(
            kwargs.get("lorweb_diversity_reg_lambda", 0.0)
        ),
        lorweb_layer_stats_limit=int(kwargs.get("lorweb_layer_stats_limit", 0)),
        lorweb_topk_basis=int(kwargs.get("lorweb_topk_basis", 0)),
    )

    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    logger.info(
        "LoRWeB-LoRA initialized: rank=%s, basis=%s, keys_dim=%s, heads=%s, external_query=%s, frame_strategy=%s",
        int(network_dim),
        network.lorweb_basis_size,
        network.lorweb_keys_dim,
        network.lorweb_heads,
        network.lorweb_external_query,
        network.lorweb_video_query_frame_strategy,
    )
    return network


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRWeBLoRANetwork:
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
    excluded_parts = ["patch_embedding", "text_embedding", "norm", "head"]
    include_time_modules = _parse_bool(kwargs.get("include_time_modules", False), False)
    if not include_time_modules:
        excluded_parts.extend(["time_embedding", "time_projection"])
    exclude_patterns.append(r".*(" + "|".join(excluded_parts) + r").*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_lorweb_lora_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRWeBLoRANetwork:
    modules_dim, modules_alpha, inferred_basis, inferred_keys = _extract_module_dims(
        weights_sd
    )
    lorweb_basis_size = int(kwargs.get("lorweb_basis_size", inferred_basis))
    lorweb_keys_dim = int(kwargs.get("lorweb_keys_dim", inferred_keys))
    lorweb_heads = int(kwargs.get("lorweb_heads", 1))
    lorweb_softmax = _parse_bool(kwargs.get("lorweb_softmax", True), True)
    lorweb_mixing_coeffs_type = str(kwargs.get("lorweb_mixing_coeffs_type", "mean"))
    lorweb_query_projection_type = str(
        kwargs.get("lorweb_query_projection_type", "linear")
    )
    lorweb_query_pooling = str(kwargs.get("lorweb_query_pooling", "avg"))
    lorweb_query_mode = str(kwargs.get("lorweb_query_mode", "all_tokens"))
    lorweb_query_l2_normalize = _parse_bool(
        kwargs.get("lorweb_query_l2_normalize", False), False
    )
    lorweb_external_query = _parse_bool(kwargs.get("lorweb_external_query", False), False)
    lorweb_external_query_dim = int(kwargs.get("lorweb_external_query_dim", 128))
    lorweb_external_query_mode = str(kwargs.get("lorweb_external_query_mode", "global"))
    lorweb_runtime_source = str(kwargs.get("lorweb_runtime_source", "auto"))
    lorweb_external_query_encoder = str(
        kwargs.get("lorweb_external_query_encoder", "none")
    )
    lorweb_external_query_model_name = str(
        kwargs.get("lorweb_external_query_model_name", "")
    )
    lorweb_external_query_image_size = int(
        kwargs.get("lorweb_external_query_image_size", 224)
    )
    lorweb_external_query_cache = _parse_bool(
        kwargs.get("lorweb_external_query_cache", False), False
    )
    lorweb_external_query_cache_size = int(
        kwargs.get("lorweb_external_query_cache_size", 512)
    )
    lorweb_external_query_encode_batch_size = int(
        kwargs.get("lorweb_external_query_encode_batch_size", 0)
    )
    lorweb_external_query_autocast_dtype = str(
        kwargs.get("lorweb_external_query_autocast_dtype", "none")
    )
    lorweb_external_query_encoder_offload_interval = int(
        kwargs.get("lorweb_external_query_encoder_offload_interval", 0)
    )
    lorweb_video_query_frame_strategy = str(
        kwargs.get("lorweb_video_query_frame_strategy", "mean")
    )
    lorweb_entropy_reg_lambda = float(kwargs.get("lorweb_entropy_reg_lambda", 0.0))
    lorweb_entropy_reg_target = _parse_optional_float(
        kwargs.get("lorweb_entropy_reg_target", None)
    )
    lorweb_diversity_reg_lambda = float(
        kwargs.get("lorweb_diversity_reg_lambda", 0.0)
    )
    lorweb_layer_stats_limit = int(kwargs.get("lorweb_layer_stats_limit", 0))
    lorweb_topk_basis = int(kwargs.get("lorweb_topk_basis", 0))

    def _module_factory(lora_name, org_module, module_multiplier, lora_dim, alpha, **kw):
        module_cls = LoRWeBLoRAInfModule if for_inference else LoRWeBLoRAModule
        return module_cls(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=module_multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            lorweb_basis_size=lorweb_basis_size,
            lorweb_keys_dim=lorweb_keys_dim,
            lorweb_heads=lorweb_heads,
            lorweb_softmax=lorweb_softmax,
            lorweb_mixing_coeffs_type=lorweb_mixing_coeffs_type,
            lorweb_query_projection_type=lorweb_query_projection_type,
            lorweb_query_pooling=lorweb_query_pooling,
            lorweb_query_mode=lorweb_query_mode,
            lorweb_query_l2_normalize=lorweb_query_l2_normalize,
            lorweb_external_query=lorweb_external_query,
            lorweb_external_query_dim=lorweb_external_query_dim,
            lorweb_external_query_mode=lorweb_external_query_mode,
            **kw,
        )

    network = LoRWeBLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="lorweb_lora_unet",
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        lora_dim=1,
        alpha=1.0,
        module_class=_module_factory,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        lorweb_basis_size=lorweb_basis_size,
        lorweb_keys_dim=lorweb_keys_dim,
        lorweb_heads=lorweb_heads,
        lorweb_softmax=lorweb_softmax,
        lorweb_mixing_coeffs_type=lorweb_mixing_coeffs_type,
        lorweb_query_projection_type=lorweb_query_projection_type,
        lorweb_query_pooling=lorweb_query_pooling,
        lorweb_query_mode=lorweb_query_mode,
        lorweb_query_l2_normalize=lorweb_query_l2_normalize,
        lorweb_external_query=lorweb_external_query,
        lorweb_external_query_dim=lorweb_external_query_dim,
        lorweb_external_query_mode=lorweb_external_query_mode,
        lorweb_runtime_source=lorweb_runtime_source,
        lorweb_external_query_encoder=lorweb_external_query_encoder,
        lorweb_external_query_model_name=lorweb_external_query_model_name,
        lorweb_external_query_image_size=lorweb_external_query_image_size,
        lorweb_external_query_cache=lorweb_external_query_cache,
        lorweb_external_query_cache_size=lorweb_external_query_cache_size,
        lorweb_external_query_encode_batch_size=lorweb_external_query_encode_batch_size,
        lorweb_external_query_autocast_dtype=lorweb_external_query_autocast_dtype,
        lorweb_external_query_encoder_offload_interval=lorweb_external_query_encoder_offload_interval,
        lorweb_video_query_frame_strategy=lorweb_video_query_frame_strategy,
        lorweb_entropy_reg_lambda=lorweb_entropy_reg_lambda,
        lorweb_entropy_reg_target=lorweb_entropy_reg_target,
        lorweb_diversity_reg_lambda=lorweb_diversity_reg_lambda,
        lorweb_layer_stats_limit=lorweb_layer_stats_limit,
        lorweb_topk_basis=lorweb_topk_basis,
    )
    return network


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRWeBLoRANetwork:
    return create_arch_network(
        multiplier=multiplier,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRWeBLoRANetwork:
    return create_arch_network_from_weights(
        multiplier=multiplier,
        weights_sd=weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )
