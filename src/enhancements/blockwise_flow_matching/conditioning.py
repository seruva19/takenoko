from __future__ import annotations

from typing import Any, List, Optional

import logging
import torch
import torch.nn as nn

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from enhancements.blockwise_flow_matching.segment_utils import (
    build_segment_boundaries,
    normalize_timesteps,
    segment_index_for_timesteps,
)

logger = get_logger(__name__, level=logging.INFO)


def infer_text_context_dim(transformer: Any) -> int:
    if hasattr(transformer, "text_embedding") and transformer.text_embedding:
        layer = transformer.text_embedding[0]
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
    return int(getattr(transformer, "dim", 1024))


def _extract_patch_tokens(features: Any) -> Optional[torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"]
        for value in features.values():
            if torch.is_tensor(value) and value.dim() == 3:
                return value
        return None
    if torch.is_tensor(features):
        if features.dim() == 3:
            return features
        if features.dim() == 2:
            return features.unsqueeze(1)
    return None


class BFMConditioningHelper(nn.Module):
    """Training-only BFM conditioning helper (semantic + segment tokens)."""

    def __init__(
        self,
        args: Any,
        text_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.text_dim = int(text_dim)
        self._warned_missing_pixels = False
        self._warned_token_mismatch = False

        self.semfeat_conditioning_enabled = bool(
            getattr(args, "bfm_semfeat_conditioning_enabled", False)
            or getattr(args, "bfm_inference_semfeat_enabled", False)
        )
        self.semfeat_conditioning_scale = float(
            getattr(args, "bfm_semfeat_conditioning_scale", 1.0)
        )
        self.semfeat_conditioning_dropout = float(
            getattr(args, "bfm_semfeat_conditioning_dropout", 0.0)
        )

        self.segment_conditioning_enabled = bool(
            getattr(args, "bfm_segment_conditioning_enabled", False)
            or getattr(args, "bfm_inference_segment_enabled", False)
        )
        self.segment_conditioning_scale = float(
            getattr(args, "bfm_segment_conditioning_scale", 1.0)
        )
        self.num_segments = int(getattr(args, "bfm_num_segments", 6))
        self.segment_min_t = float(getattr(args, "bfm_segment_min_t", 0.0))
        self.segment_max_t = float(getattr(args, "bfm_segment_max_t", 1.0))

        self.encoder_name = str(
            getattr(args, "bfm_semfeat_encoder_name", "dinov2-vit-b14")
        )
        self.input_resolution = int(
            getattr(args, "bfm_semfeat_resolution", 256)
        )
        self.frn_enabled = bool(getattr(args, "bfm_frn_enabled", False))
        self.frn_loss_weight = float(getattr(args, "bfm_frn_loss_weight", 0.1))
        self.frn_hidden_dim = int(getattr(args, "bfm_frn_hidden_dim", 1024))

        self._projection: Optional[nn.Linear] = None
        self._segment_embed: Optional[nn.Embedding] = None
        self._frn: Optional[nn.Module] = None

        if self.semfeat_conditioning_enabled:
            self.encoder_manager = EncoderManager(self.device)
            self.encoders, self.encoder_types, _ = (
                self.encoder_manager.load_encoders(
                    self.encoder_name, resolution=self.input_resolution
                )
            )
            if not self.encoders:
                raise ValueError("BFM conditioning: no encoders loaded.")
            encoder_dim = int(getattr(self.encoders[0], "embed_dim", 1024))
            self._projection = nn.Linear(encoder_dim, self.text_dim).to(
                self.device
            )

        if self.segment_conditioning_enabled:
            self._segment_embed = nn.Embedding(
                self.num_segments, self.text_dim
            ).to(self.device)
        if self.frn_enabled:
            self._frn = nn.Sequential(
                nn.Linear(self.text_dim + 1, self.frn_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.frn_hidden_dim, self.text_dim),
            ).to(self.device)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._projection is not None:
            params.extend(list(self._projection.parameters()))
        if self._segment_embed is not None:
            params.extend(list(self._segment_embed.parameters()))
        if self._frn is not None:
            params.extend(list(self._frn.parameters()))
        return params

    def set_inference_overrides(
        self,
        *,
        semfeat_scale: Optional[float] = None,
        segment_scale: Optional[float] = None,
    ) -> None:
        if semfeat_scale is not None:
            self.semfeat_conditioning_scale = float(semfeat_scale)
        if segment_scale is not None:
            self.segment_conditioning_scale = float(segment_scale)
        self.semfeat_conditioning_dropout = 0.0

    def apply_context(
        self,
        context: List[torch.Tensor],
        batch: dict,
        timesteps: torch.Tensor,
    ) -> List[torch.Tensor]:
        updated = list(context)
        if self.semfeat_conditioning_enabled:
            tokens = self._compute_semantic_tokens(batch)
            if tokens is not None:
                tokens = tokens * self.semfeat_conditioning_scale
                updated = self._concat_context(updated, tokens)
        if self.segment_conditioning_enabled:
            tokens = self._compute_segment_tokens(timesteps)
            if tokens is not None:
                tokens = tokens * self.segment_conditioning_scale
                updated = self._concat_context(updated, tokens)
        return updated

    def compute_semantic_tokens_from_batch(
        self, batch: dict
    ) -> Optional[torch.Tensor]:
        return self._compute_semantic_tokens(batch)

    def _compute_semantic_tokens(
        self, batch: dict
    ) -> Optional[torch.Tensor]:
        if self._projection is None:
            return None
        if self.semfeat_conditioning_dropout > 0:
            if torch.rand(1).item() < self.semfeat_conditioning_dropout:
                return None
        if "pixels" not in batch:
            if not self._warned_missing_pixels:
                logger.warning(
                    "BFM conditioning enabled but no pixels in batch; skipping."
                )
                self._warned_missing_pixels = True
            return None
        pixels = batch.get("pixels")
        if isinstance(pixels, list):
            pixels = torch.stack(pixels, dim=0)
        if not torch.is_tensor(pixels):
            return None
        if pixels.dim() == 5:
            pixels = pixels[:, :, 0, :, :]
        pixels = pixels.to(device=self.device)
        images = (pixels + 1.0) / 2.0
        images = images * 255.0

        encoder = self.encoders[0]
        enc_type = self.encoder_types[0]
        with torch.no_grad():
            prepped = preprocess_raw_image(images, enc_type)
            features = encoder.forward_features(prepped)
        patch_tokens = _extract_patch_tokens(features)
        if patch_tokens is None:
            return None
        token = patch_tokens.mean(dim=1, keepdim=True)
        return self._projection(token)

    def compute_semantic_tokens_from_pixels(
        self, pixels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self._projection is None:
            return None
        if pixels.dim() == 5:
            pixels = pixels[:, :, 0, :, :]
        pixels = pixels.to(device=self.device)
        images = (pixels + 1.0) / 2.0
        images = images * 255.0

        encoder = self.encoders[0]
        enc_type = self.encoder_types[0]
        with torch.no_grad():
            prepped = preprocess_raw_image(images, enc_type)
            features = encoder.forward_features(prepped)
        patch_tokens = _extract_patch_tokens(features)
        if patch_tokens is None:
            return None
        token = patch_tokens.mean(dim=1, keepdim=True)
        return self._projection(token)

    def predict_frn_tokens(
        self, latents: torch.Tensor, timesteps: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self._frn is None:
            return None
        pooled = latents.float().mean(dim=tuple(range(2, latents.dim())))
        t_norm = normalize_timesteps(timesteps.view(-1)).unsqueeze(1)
        frn_in = torch.cat([pooled, t_norm.to(pooled)], dim=1)
        return self._frn(frn_in).unsqueeze(1)

    def compute_frn_loss(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self._frn is None:
            return None
        pred = self.predict_frn_tokens(latents, timesteps)
        if pred is None:
            return None
        target = target_tokens.detach()
        if pred.shape != target.shape:
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len]
            target = target[:, :min_len]
        loss = torch.mean((pred - target) ** 2)
        return loss

    def _compute_segment_tokens(
        self, timesteps: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self._segment_embed is None:
            return None
        t = timesteps.view(-1)
        boundaries = build_segment_boundaries(
            num_segments=self.num_segments,
            min_t=self.segment_min_t,
            max_t=self.segment_max_t,
            device=t.device,
            dtype=t.dtype,
        )
        t_norm = normalize_timesteps(t)
        segment_idx = segment_index_for_timesteps(t_norm, boundaries)
        tokens = self._segment_embed(segment_idx)
        return tokens.unsqueeze(1)

    def compute_segment_tokens(
        self, timesteps: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return self._compute_segment_tokens(timesteps)

    def _concat_context(
        self, context: List[torch.Tensor], tokens: torch.Tensor
    ) -> List[torch.Tensor]:
        if len(context) != tokens.size(0):
            if not self._warned_token_mismatch:
                logger.warning(
                    "BFM conditioning token count mismatch; skipping conditioning."
                )
                self._warned_token_mismatch = True
            return context
        updated: List[torch.Tensor] = []
        for idx, ctx in enumerate(context):
            updated.append(torch.cat([ctx, tokens[idx]], dim=0))
        return updated
