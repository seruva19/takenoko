"""MOALIGN helper for motion-centric representation alignment during training."""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image

logger = get_logger(__name__)


def _interpolate_token_count(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token features to match a target token count."""
    bf, token_count, dim = tokens.shape
    if token_count == target_tokens:
        return tokens

    src_side = int(math.isqrt(token_count))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == token_count and tgt_side * tgt_side == target_tokens:
        x = tokens.permute(0, 2, 1).reshape(bf, dim, src_side, src_side)
        x = F.interpolate(x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False)
        return x.reshape(bf, dim, target_tokens).permute(0, 2, 1)

    x_1d = tokens.permute(0, 2, 1)
    x_1d = F.interpolate(x_1d, size=target_tokens, mode="linear", align_corners=False)
    return x_1d.permute(0, 2, 1)


class MoAlignHelper(nn.Module):
    """Train-time MOALIGN adaptation using motion-centric relational alignment."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []
        self._shape_warning_logged = False

        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = getattr(args, "model_cache_dir", "models")
        resolution = int(getattr(args, "moalign_input_resolution", 256))
        encoder_spec = getattr(args, "moalign_encoder_name", "dinov2-vit-b")

        manager = EncoderManager(device=device, cache_dir=cache_dir)
        encoders, encoder_types, _ = manager.load_encoders(encoder_spec, resolution=resolution)
        self.encoder = encoders[0]
        self.encoder_type = encoder_types[0]
        if len(encoders) > 1:
            logger.warning(
                "MOALIGN received multiple encoders; only the first one is used: %s",
                encoder_spec.split(",")[0].strip(),
            )
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        raw_depths = getattr(args, "moalign_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            self.alignment_depths = [int(depth) for depth in raw_depths]
        else:
            self.alignment_depths = [int(getattr(args, "moalign_alignment_depth", 18))]
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        proj_hidden = int(getattr(args, "moalign_projection_hidden_dim", 256))
        proj_out = int(getattr(args, "moalign_projection_out_dim", 64))
        self.use_stage1_teacher = bool(getattr(args, "moalign_use_stage1_teacher", False))
        self.stage1_checkpoint = str(getattr(args, "moalign_stage1_checkpoint", "") or "")
        if self.use_stage1_teacher:
            proj_hidden, proj_out = self._resolve_stage1_projection_dims(
                checkpoint_path=self.stage1_checkpoint,
                fallback_hidden=proj_hidden,
                fallback_out=proj_out,
            )

        self.diffusion_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, proj_hidden),
            nn.SiLU(),
            nn.Linear(proj_hidden, proj_out),
        )
        self.stage1_motion_projector: Optional[nn.Module] = None
        self.motion_projector: Optional[nn.Module] = None
        if self.use_stage1_teacher:
            self.stage1_motion_projector = self._load_stage1_motion_projector(
                checkpoint_path=self.stage1_checkpoint,
                hidden_dim=proj_hidden,
                output_dim=proj_out,
            )
            logger.info(
                "MOALIGN: using frozen Stage-1 teacher projector from %s.",
                self.stage1_checkpoint,
            )
        else:
            self.motion_projector = nn.Sequential(
                nn.Linear(self.encoder.embed_dim, proj_hidden),
                nn.SiLU(),
                nn.Linear(proj_hidden, proj_out),
            )

        self.motion_target_mode = str(
            getattr(args, "moalign_motion_target_mode", "delta")
        ).lower()
        self.spatial_weight = float(getattr(args, "moalign_spatial_weight", 1.0))
        self.temporal_weight = float(getattr(args, "moalign_temporal_weight", 1.0))
        self.temporal_tau = float(getattr(args, "moalign_temporal_tau", 10.0))
        self.max_tokens = int(getattr(args, "moalign_max_tokens", 256))
        self.loss_lambda = float(getattr(args, "moalign_loss_lambda", 0.5))
        self.spatial_align = bool(getattr(args, "moalign_spatial_align", True))
        self.detach_targets = bool(getattr(args, "moalign_detach_targets", False))

    def _resolve_stage1_projection_dims(
        self,
        checkpoint_path: str,
        fallback_hidden: int,
        fallback_out: int,
    ) -> Tuple[int, int]:
        if checkpoint_path == "" or not os.path.exists(checkpoint_path):
            return fallback_hidden, fallback_out
        try:
            payload = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(payload, dict):
                hidden = int(payload.get("hidden_dim", fallback_hidden))
                out = int(payload.get("output_dim", fallback_out))
                if hidden > 0 and out > 0:
                    if hidden != fallback_hidden or out != fallback_out:
                        logger.info(
                            "MOALIGN: overriding projection dims from Stage-1 checkpoint (hidden=%d, out=%d).",
                            hidden,
                            out,
                        )
                    return hidden, out
        except Exception as exc:
            logger.warning(
                "MOALIGN: could not read projection dims from Stage-1 checkpoint: %s",
                exc,
            )
        return fallback_hidden, fallback_out

    def _load_stage1_motion_projector(
        self,
        checkpoint_path: str,
        hidden_dim: int,
        output_dim: int,
    ) -> nn.Module:
        from enhancements.moalign.stage1_model import Stage1MotionProjector

        if checkpoint_path == "":
            raise ValueError(
                "MOALIGN: moalign_stage1_checkpoint is required when moalign_use_stage1_teacher is true."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"MOALIGN: Stage-1 checkpoint not found: {checkpoint_path}"
            )

        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                "MOALIGN: Stage-1 checkpoint must be a dict payload with motion_projector weights."
            )

        state_dict = payload.get("motion_projector")
        if not isinstance(state_dict, dict):
            # Allow direct state dict checkpoints as fallback.
            if all(isinstance(v, torch.Tensor) for v in payload.values()):
                state_dict = payload
            else:
                raise ValueError(
                    "MOALIGN: could not locate motion_projector weights in Stage-1 checkpoint."
                )

        projector = Stage1MotionProjector(
            input_dim=int(self.encoder.embed_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(output_dim),
        )
        missing_keys, unexpected_keys = projector.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            logger.warning(
                "MOALIGN: Stage-1 checkpoint load had key mismatches (missing=%d, unexpected=%d).",
                len(missing_keys),
                len(unexpected_keys),
            )
        projector.eval()
        projector.requires_grad_(False)
        return projector

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("MOALIGN: falling back to hidden_dim=1024")
        return 1024

    def _get_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        try:
            for i, depth in enumerate(self.alignment_depths):
                if hasattr(self.diffusion_model, "blocks"):
                    target_module = self.diffusion_model.blocks[depth]
                elif hasattr(self.diffusion_model, "layers"):
                    target_module = self.diffusion_model.layers[depth]
                elif hasattr(self.diffusion_model, "transformer_blocks"):
                    target_module = self.diffusion_model.transformer_blocks[depth]
                else:
                    raise ValueError("MOALIGN: could not locate transformer block list")
                handle = target_module.register_forward_hook(self._get_hook(i))
                self.hook_handles.append(handle)
                logger.info("MOALIGN: hook attached to layer %d.", depth)
        except Exception as exc:
            logger.error("MOALIGN: failed to attach hooks: %s", exc)
            raise

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()
        self.captured_features = [None] * len(self.alignment_depths)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.diffusion_projector.parameters())
        if self.motion_projector is not None:
            params.extend(self.motion_projector.parameters())
        return params

    def _extract_encoder_tokens(self, clean_pixels: torch.Tensor) -> torch.Tensor:
        """Return encoder tokens shaped [B, F, N, D]."""
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        bsz, channels, frames, height, width = clean_pixels.shape
        images = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        with torch.no_grad():
            images = ((images + 1.0) / 2.0).clamp(0, 1) * 255.0
            images = preprocess_raw_image(images, self.encoder_type)
            features = self.encoder.forward_features(images)
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    features = features["x_norm_patchtokens"]
                elif "x_norm_clstoken" in features:
                    features = features["x_norm_clstoken"].unsqueeze(1)
                else:
                    tensor_candidate = None
                    for value in features.values():
                        if isinstance(value, torch.Tensor):
                            tensor_candidate = value
                            break
                    if tensor_candidate is None:
                        raise ValueError("MOALIGN: encoder output did not contain tensor features")
                    features = tensor_candidate
            if not isinstance(features, torch.Tensor):
                raise ValueError("MOALIGN: unsupported encoder output type")
            if features.dim() == 2:
                features = features.unsqueeze(1)
            elif features.dim() > 3:
                n_s, c_feat, h_feat, w_feat = features.shape
                features = features.view(n_s, c_feat, h_feat * w_feat).transpose(1, 2)

        return features.view(bsz, frames, features.shape[1], features.shape[2])

    def _build_motion_targets(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        if self.stage1_motion_projector is not None:
            with torch.no_grad():
                motion_tokens, _ = self.stage1_motion_projector.forward_tokens(encoder_tokens)
            if self.detach_targets:
                motion_tokens = motion_tokens.detach()
            return motion_tokens

        if self.motion_target_mode == "delta":
            delta = encoder_tokens[:, 1:] - encoder_tokens[:, :-1]
            head = torch.zeros_like(encoder_tokens[:, :1])
            motion_tokens = torch.cat([head, delta], dim=1)
        else:
            motion_tokens = encoder_tokens
        if self.motion_projector is None:
            raise ValueError("MOALIGN: motion_projector is not initialized.")
        projected = self.motion_projector(motion_tokens)
        if self.detach_targets:
            projected = projected.detach()
        return projected

    def _align_tokens(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align source and target token grids to a common token count."""
        bsz, frames, src_tokens, dim = source.shape
        _, tgt_frames, tgt_tokens, _ = target.shape
        if frames != tgt_frames:
            raise ValueError("MOALIGN: source and target must have identical frame count")

        source_2d = source.reshape(bsz * frames, src_tokens, dim)
        target_2d = target.reshape(bsz * frames, tgt_tokens, dim)

        if src_tokens != tgt_tokens:
            if self.spatial_align:
                if src_tokens > tgt_tokens:
                    source_2d = _interpolate_token_count(source_2d, tgt_tokens)
                else:
                    target_2d = _interpolate_token_count(target_2d, src_tokens)
            else:
                min_tokens = min(src_tokens, tgt_tokens)
                source_2d = source_2d[:, :min_tokens]
                target_2d = target_2d[:, :min_tokens]

        if source_2d.shape[1] > self.max_tokens:
            source_2d = _interpolate_token_count(source_2d, self.max_tokens)
            target_2d = _interpolate_token_count(target_2d, self.max_tokens)

        aligned_tokens = source_2d.shape[1]
        source = source_2d.view(bsz, frames, aligned_tokens, dim)
        target = target_2d.view(bsz, frames, aligned_tokens, dim)
        return source, target

    def _temporal_weights(self, frame_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(frame_count, device=device)
        distance = (idx[:, None] - idx[None, :]).abs().to(dtype=dtype)
        weights = torch.exp(-distance / self.temporal_tau)
        weights = weights * (distance > 0).to(dtype=dtype)
        return weights

    def _relational_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MOALIGN soft relational loss over spatial + temporal relations."""
        source, target = self._align_tokens(source, target)

        source_norm = F.normalize(source, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        source_spatial = torch.matmul(source_norm, source_norm.transpose(-1, -2))
        target_spatial = torch.matmul(target_norm, target_norm.transpose(-1, -2))
        spatial_loss = F.l1_loss(source_spatial, target_spatial)

        frame_count = source.shape[1]
        if frame_count > 1 and self.temporal_weight > 0.0:
            source_frame = source_norm.mean(dim=2)
            target_frame = target_norm.mean(dim=2)
            source_temporal = torch.matmul(source_frame, source_frame.transpose(-1, -2))
            target_temporal = torch.matmul(target_frame, target_frame.transpose(-1, -2))
            weights = self._temporal_weights(
                frame_count=frame_count,
                device=source.device,
                dtype=source.dtype,
            )
            temporal_loss = F.l1_loss(source_temporal * weights, target_temporal * weights)
        else:
            temporal_loss = source.new_tensor(0.0)

        return self.spatial_weight * spatial_loss + self.temporal_weight * temporal_loss

    def get_moalign_loss(self, clean_pixels: torch.Tensor, vae: Optional[Any] = None) -> torch.Tensor:
        """Compute weighted MOALIGN loss from captured diffusion features."""
        del vae
        if not any(feat is not None for feat in self.captured_features):
            return clean_pixels.new_tensor(0.0)

        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        bsz, _, frames, _, _ = clean_pixels.shape

        encoder_tokens = self._extract_encoder_tokens(clean_pixels)
        target_tokens = self._build_motion_targets(encoder_tokens)

        losses: List[torch.Tensor] = []
        for diffusion_features in self.captured_features:
            if diffusion_features is None:
                continue

            projected = self.diffusion_projector(diffusion_features)
            if projected.dim() != 3 or projected.shape[0] != bsz:
                if not self._shape_warning_logged:
                    logger.warning(
                        "MOALIGN: unexpected diffusion feature shape %s; skipping loss for this step.",
                        tuple(projected.shape),
                    )
                    self._shape_warning_logged = True
                continue

            seq_len = projected.shape[1]
            if frames > 0 and seq_len % frames == 0:
                tokens_per_frame = seq_len // frames
                source_tokens = projected.view(bsz, frames, tokens_per_frame, projected.shape[-1])
                target_for_layer = target_tokens
            else:
                if not self._shape_warning_logged:
                    logger.warning(
                        "MOALIGN: seq_len (%d) not divisible by frame_count (%d); using pooled-frame fallback.",
                        seq_len,
                        frames,
                    )
                    self._shape_warning_logged = True
                source_tokens = projected.unsqueeze(1)
                target_for_layer = target_tokens.mean(dim=1, keepdim=True)

            losses.append(self._relational_loss(source_tokens, target_for_layer))

        self.captured_features = [None] * len(self.alignment_depths)
        if not losses:
            return clean_pixels.new_tensor(0.0)

        base_loss = torch.stack(losses).mean()
        return base_loss * self.loss_lambda
