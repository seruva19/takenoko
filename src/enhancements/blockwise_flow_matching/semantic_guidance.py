from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from enhancements.repa.enhanced_repa_helper import (
    MultiEncoderProjectionHead,
    interpolate_features_spatial,
)

logger = get_logger(__name__, level=logging.INFO)


def _extract_patch_tokens(
    features: Any,
) -> Optional[torch.Tensor]:
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


class SemFeatAlignmentHelper(nn.Module):
    """Training-only semantic feature guidance helper (BFM SemFeat)."""

    def __init__(
        self,
        diffusion_model: nn.Module,
        args: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.device = device or next(diffusion_model.parameters()).device
        self.enabled = bool(getattr(args, "bfm_semfeat_enabled", False))
        self.alignment_depth = int(
            getattr(args, "bfm_semfeat_alignment_depth", 8)
        )
        self.encoder_name = str(
            getattr(args, "bfm_semfeat_encoder_name", "dinov2-vit-b14")
        )
        self.input_resolution = int(
            getattr(args, "bfm_semfeat_resolution", 256)
        )
        self.spatial_align = bool(
            getattr(args, "bfm_semfeat_spatial_align", True)
        )
        self._hook_handle: Optional[Any] = None
        self._captured_features: Optional[torch.Tensor] = None

        self.encoder_manager = EncoderManager(self.device)
        self.encoders, self.encoder_types, _ = self.encoder_manager.load_encoders(
            self.encoder_name, resolution=self.input_resolution
        )
        self.encoder_dims = [enc.embed_dim for enc in self.encoders]
        self.diffusion_hidden_dim = self._infer_diffusion_hidden_dim()

        projection_dim = int(
            getattr(args, "bfm_semfeat_projection_dim", self.encoder_dims[0])
        )
        self.projection_heads = MultiEncoderProjectionHead(
            diffusion_hidden_dim=self.diffusion_hidden_dim,
            encoder_dims=[
                projection_dim for _ in range(len(self.encoder_dims))
            ],
            ensemble_mode="individual",
            shared_projection=False,
            projection_type="mlp",
        )
        self._projection_out_dim = projection_dim

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning(
            "SemFeat: Could not determine diffusion hidden dim; using 1024."
        )
        return 1024

    def _get_blocks(self) -> Optional[list[nn.Module]]:
        if hasattr(self.diffusion_model, "blocks"):
            return list(self.diffusion_model.blocks)
        if hasattr(self.diffusion_model, "layers"):
            return list(self.diffusion_model.layers)
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return list(self.diffusion_model.transformer_blocks)
        return None

    def _get_hook(self):
        def hook(_module: nn.Module, _inp: Any, output: Any) -> None:
            features = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(features):
                self._captured_features = features

        return hook

    def setup_hooks(self) -> None:
        if not self.enabled:
            return
        blocks = self._get_blocks()
        if not blocks:
            raise ValueError("SemFeat: Could not resolve diffusion blocks.")
        max_idx = len(blocks) - 1
        if self.alignment_depth < 0 or self.alignment_depth > max_idx:
            raise ValueError(
                f"bfm_semfeat_alignment_depth {self.alignment_depth} out of range "
                f"[0, {max_idx}]"
            )
        self._hook_handle = blocks[self.alignment_depth].register_forward_hook(
            self._get_hook()
        )
        logger.info(
            "SemFeat hook attached to diffusion block %d.", self.alignment_depth
        )

    def remove_hooks(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured_features = None

    def get_trainable_params(self) -> list[nn.Parameter]:
        return list(self.projection_heads.parameters())

    def compute_loss(
        self, clean_pixels: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.enabled or self._captured_features is None:
            return None, None

        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]

        images = (clean_pixels + 1) / 2.0
        images = images * 255.0

        diffusion_features = self._captured_features
        if diffusion_features.dim() == 4:
            diffusion_features = diffusion_features.flatten(2).transpose(1, 2)
        if diffusion_features.dim() != 3:
            return None, None

        losses = []
        similarities = []
        for idx, (encoder, enc_type) in enumerate(
            zip(self.encoders, self.encoder_types)
        ):
            with torch.no_grad():
                prepped = preprocess_raw_image(images, enc_type)
                target_features = encoder.forward_features(prepped)
            patch_tokens = _extract_patch_tokens(target_features)
            if patch_tokens is None:
                continue

            projected = self.projection_heads._project_with_layer(  # type: ignore[attr-defined]
                diffusion_features,
                self.projection_heads.projections[idx],
                self.projection_heads.fallback_linears[idx]
                if self.projection_heads.fallback_linears is not None
                else None,
            )
            if projected.shape[-1] != self._projection_out_dim:
                projected = projected[..., : self._projection_out_dim]
            if self.spatial_align and projected.shape[1] != patch_tokens.shape[1]:
                projected = interpolate_features_spatial(
                    projected, patch_tokens.shape[1]
                )
            elif projected.shape[1] != patch_tokens.shape[1]:
                min_tokens = min(projected.shape[1], patch_tokens.shape[1])
                projected = projected[:, :min_tokens]
                patch_tokens = patch_tokens[:, :min_tokens]

            projected = F.normalize(projected, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)
            similarity = (projected * patch_tokens).sum(dim=-1).mean()
            loss = 1.0 - similarity
            losses.append(loss)
            similarities.append(similarity)

        if not losses:
            return None, None
        loss = torch.stack(losses).mean()
        similarity = torch.stack(similarities).mean()
        return loss, similarity
