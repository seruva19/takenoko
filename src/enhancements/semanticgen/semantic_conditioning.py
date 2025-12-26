from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from enhancements.semanticgen.semantic_encoder import SemanticEncoder, sample_frames
from dataset.image_video_dataset import TARGET_FPS_WAN
from utils.model_utils import str_to_dtype

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class SemanticConditioningOutput:
    context: List[torch.Tensor]
    kl_loss: Optional[torch.Tensor]
    semantic_tokens: Optional[torch.Tensor]


class SemanticConditioningHelper(nn.Module):
    def __init__(self, args: Any, text_dim: int, device: torch.device) -> None:
        super().__init__()
        self.args = args
        self.text_dim = text_dim
        self.device = device
        self._warned_missing_inputs = False
        self._warned_mode = False

        self.encoder = SemanticEncoder(
            model_name=getattr(args, "semantic_encoder_name", "dinov2-vit-b14"),
            encoder_type=_normalize_encoder_type(
                getattr(args, "semantic_encoder_type", "repa")
            ),
            device=str(device),
            dtype=str_to_dtype(
                getattr(args, "semantic_encoder_dtype", "float16")
            ),
            input_resolution=int(getattr(args, "semantic_encoder_resolution", 256)),
            cache_dir=getattr(args, "model_cache_dir", "models"),
        )

        self.embed_dim = int(getattr(args, "semantic_embed_dim", 1024))
        self.compress_dim = int(getattr(args, "semantic_compress_dim", 256))
        self.semantic_noise_std = float(getattr(args, "semantic_noise_std", 0.0))
        self.semantic_kl_weight = float(getattr(args, "semantic_kl_weight", 0.1))
        self.semantic_context_mode = getattr(
            args, "semantic_context_mode", "concat_text"
        )
        self.semantic_context_scale = float(
            getattr(args, "semantic_context_scale", 1.0)
        )
        self.semantic_condition_dropout = float(
            getattr(args, "semantic_condition_dropout", 0.0)
        )
        self.semantic_condition_anneal_steps = int(
            getattr(args, "semantic_condition_anneal_steps", 0)
        )
        self.semantic_condition_min_scale = float(
            getattr(args, "semantic_condition_min_scale", 0.0)
        )

        self._mlp: Optional[nn.Linear] = nn.Linear(
            self.embed_dim, self.compress_dim * 2
        ).to(self.device)
        self._proj: Optional[nn.Linear] = nn.Linear(
            self.compress_dim, self.text_dim
        ).to(self.device)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._mlp is not None:
            params.extend(list(self._mlp.parameters()))
        if self._proj is not None:
            params.extend(list(self._proj.parameters()))
        return params

    def build_context(
        self,
        context: List[torch.Tensor],
        batch: dict,
        global_step: Optional[int],
    ) -> SemanticConditioningOutput:
        # SPEC:semanticgen_lora:conditioning - inject compressed semantics to guide training without inference changes.
        semantic_tokens = self._resolve_semantic_tokens(batch)
        if semantic_tokens is None:
            return SemanticConditioningOutput(context, None, None)
        if semantic_tokens.dim() == 2:
            semantic_tokens = semantic_tokens.unsqueeze(1)

        scale = self._compute_scale(global_step)
        if scale <= 0:
            return SemanticConditioningOutput(context, None, semantic_tokens)

        if self._maybe_drop_condition():
            return SemanticConditioningOutput(context, None, semantic_tokens)

        semantic_tokens = semantic_tokens.to(device=self.device)
        compressed, kl_loss = self._compress_tokens(semantic_tokens)
        if self.semantic_noise_std > 0:
            compressed = compressed + torch.randn_like(compressed) * self.semantic_noise_std

        projected = self._project_tokens(compressed)
        projected = projected * scale

        updated_context = self._concat_context(context, projected)
        if kl_loss is not None and self.semantic_kl_weight > 0:
            kl_loss = kl_loss * self.semantic_kl_weight
        else:
            kl_loss = None

        return SemanticConditioningOutput(updated_context, kl_loss, projected)

    def get_semantic_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        return self._resolve_semantic_tokens(batch)

    def _resolve_semantic_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        if "semantic_embeddings" in batch and batch["semantic_embeddings"] is not None:
            embeddings = batch["semantic_embeddings"]
            if isinstance(embeddings, list):
                embeddings = _pad_and_stack(embeddings)
            return embeddings

        if "pixels" not in batch:
            if not self._warned_missing_inputs:
                logger.warning(
                    "SemanticGen LoRA enabled but no semantic cache or pixels in batch; skipping semantic conditioning."
                )
                self._warned_missing_inputs = True
            return None

        pixels = torch.stack(batch["pixels"], dim=0).to(self.device)
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(2)
        bsz, _, frames, _, _ = pixels.shape
        sampled: List[torch.Tensor] = []
        for idx in range(bsz):
            cf_hw = pixels[idx]
            frame_stack = cf_hw.permute(1, 0, 2, 3).contiguous()
            frame_stack = sample_frames(
                frame_stack,
                target_fps=float(getattr(self.args, "semantic_encoder_target_fps", TARGET_FPS_WAN)),
                semantic_fps=float(getattr(self.args, "semantic_encoder_fps", 2.0)),
                stride=int(getattr(self.args, "semantic_encoder_stride", 1)),
                frame_limit=_optional_int(getattr(self.args, "semantic_encoder_frame_limit", None)),
            )
            sampled.append(frame_stack)
        embeds = []
        for frame_stack in sampled:
            frame_stack = _to_uint8(frame_stack)
            embed = self.encoder.encode_frames(frame_stack)
            embeds.append(embed)
        return _pad_and_stack(embeds)

    def _compress_tokens(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, dim = tokens.shape
        if self._mlp is None or dim != self.embed_dim:
            # SPEC:semanticgen_lora:conditioning - ensure projections match encoder output dims.
            self._mlp = nn.Linear(dim, self.compress_dim * 2).to(self.device)
        mlp_out = self._mlp(tokens)
        mu, logvar = torch.chunk(mlp_out, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._proj is None or self._proj.out_features != self.text_dim:
            # SPEC:semanticgen_lora:conditioning - adapt projection to text context width.
            self._proj = nn.Linear(tokens.shape[-1], self.text_dim).to(self.device)
        return self._proj(tokens)

    def _concat_context(
        self, context: List[torch.Tensor], tokens: torch.Tensor
    ) -> List[torch.Tensor]:
        if self.semantic_context_mode not in ("concat_text", "concat_tokens"):
            if not self._warned_mode:
                logger.warning(
                    "Unknown semantic_context_mode '%s'; falling back to concat_text.",
                    self.semantic_context_mode,
                )
                self._warned_mode = True
            mode = "concat_text"
        else:
            mode = self.semantic_context_mode

        updated: List[torch.Tensor] = []
        for idx, ctx in enumerate(context):
            token_slice = tokens[idx]
            if mode == "concat_tokens":
                combined = torch.cat([token_slice, ctx], dim=0)
            else:
                combined = torch.cat([ctx, token_slice], dim=0)
            updated.append(combined)
        return updated

    def _compute_scale(self, global_step: Optional[int]) -> float:
        # SPEC:semanticgen_lora:curriculum - anneal semantic strength to preserve inference behavior.
        scale = self.semantic_context_scale
        if global_step is None or self.semantic_condition_anneal_steps <= 0:
            return max(scale, self.semantic_condition_min_scale)
        progress = min(1.0, global_step / float(self.semantic_condition_anneal_steps))
        annealed = (1.0 - progress) * scale
        return max(annealed, self.semantic_condition_min_scale)

    def _maybe_drop_condition(self) -> bool:
        if self.semantic_condition_dropout <= 0:
            return False
        return torch.rand(1).item() < self.semantic_condition_dropout


def _normalize_encoder_type(encoder_type: str) -> str:
    mapping = {
        "qwen_vl": "hf",
        "hf": "hf",
        "repa": "repa",
    }
    return mapping.get(encoder_type, encoder_type)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_uint8(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        return frames
    frames = frames.float()
    frames = (frames + 1.0) * 127.5
    return frames.clamp(0, 255).to(torch.uint8)


def _pad_and_stack(items: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(item.shape[0] for item in items)
    embed_dim = items[0].shape[-1]
    padded = []
    for item in items:
        if item.shape[0] == max_len:
            padded.append(item)
            continue
        pad_len = max_len - item.shape[0]
        pad = torch.zeros(pad_len, embed_dim, device=item.device, dtype=item.dtype)
        padded.append(torch.cat([item, pad], dim=0))
    return torch.stack(padded, dim=0)
