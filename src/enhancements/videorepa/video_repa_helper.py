"""VideoREPA helper for token-relation distillation during training."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from enhancements.videorepa.native_teachers import (
    is_native_teacher_spec,
    load_native_teacher,
)

logger = get_logger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _interpolate_token_count(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token features to match a target token count."""
    if tokens.shape[1] == target_tokens:
        return tokens

    bf, src_tokens, dim = tokens.shape
    src_side = int(math.isqrt(src_tokens))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_tokens:
        x = tokens.permute(0, 2, 1).reshape(bf, dim, src_side, src_side)
        x = F.interpolate(x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False)
        return x.reshape(bf, dim, target_tokens).permute(0, 2, 1)

    x = tokens.permute(0, 2, 1)
    x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


class VideoRepaHelper(nn.Module):
    """Token-relation distillation helper inspired by VideoREPA."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []
        self._shape_warning_logged = False

        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = getattr(args, "model_cache_dir", "models")
        resolution = int(getattr(args, "videorepa_input_resolution", 256))
        encoder_spec = str(getattr(args, "videorepa_encoder_name", "dinov2-vit-b"))
        self.teacher_kind = "image"
        self.encoder_type = "image"
        self.video_teacher_patch_size = int(
            getattr(args, "videorepa_video_teacher_patch_size", 16)
        )
        self.video_teacher_tubelet_size = int(
            getattr(args, "videorepa_video_teacher_tubelet_size", 2)
        )
        teacher_image_size = int(getattr(args, "videorepa_video_teacher_image_size", 224))
        self.video_teacher_input_size = (teacher_image_size, teacher_image_size)
        self.video_teacher_drop_first_frame = bool(
            getattr(args, "videorepa_video_teacher_drop_first_frame", True)
        )

        if is_native_teacher_spec(encoder_spec):
            native_bundle = load_native_teacher(
                spec=encoder_spec,
                args=args,
                device=torch.device(device),
            )
            self.encoder = native_bundle.model
            self.teacher_kind = "video"
            self.encoder_type = native_bundle.teacher_type
            self.video_teacher_patch_size = int(native_bundle.patch_size)
            self.video_teacher_tubelet_size = int(native_bundle.tubelet_size)
            self.video_teacher_input_size = native_bundle.input_size
            logger.info(
                "VideoREPA: loaded native %s teacher (patch=%d, tubelet=%d, input=%s).",
                self.encoder_type,
                self.video_teacher_patch_size,
                self.video_teacher_tubelet_size,
                self.video_teacher_input_size,
            )
        else:
            manager = EncoderManager(device=device, cache_dir=cache_dir)
            encoders, encoder_types, _ = manager.load_encoders(
                encoder_spec, resolution=resolution
            )
            self.encoder = encoders[0]
            self.encoder_type = encoder_types[0]
            if len(encoders) > 1:
                logger.warning(
                    "VideoREPA received multiple encoders; using the first one: %s",
                    encoder_spec.split(",")[0].strip(),
                )
            self.encoder.eval()
            self.encoder.requires_grad_(False)

        raw_depths = getattr(args, "videorepa_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            self.alignment_depths = [int(v) for v in raw_depths]
        else:
            self.alignment_depths = [int(getattr(args, "videorepa_alignment_depth", 18))]
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        self.align_dim = int(getattr(args, "videorepa_align_dim", 768))
        self.projector_hidden_dim = int(
            getattr(args, "videorepa_projector_hidden_dim", 2048)
        )
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.projector_hidden_dim, self.align_dim),
                )
                for _ in self.alignment_depths
            ]
        )

        self.loss_lambda = float(getattr(args, "videorepa_loss_lambda", 0.5))
        self.margin = float(getattr(args, "videorepa_margin", 0.1))
        self.margin_matrix = float(
            getattr(args, "videorepa_margin_matrix", self.margin)
        )
        self.relation_mode = str(
            getattr(args, "videorepa_relation_mode", "token_relation_distillation")
        ).lower()
        self.max_spatial_tokens = int(getattr(args, "videorepa_max_spatial_tokens", -1))
        self.spatial_align = bool(getattr(args, "videorepa_spatial_align", True))
        self.temporal_align = bool(getattr(args, "videorepa_temporal_align", True))
        self.temporal_exclude_same_frame = bool(
            getattr(args, "videorepa_temporal_exclude_same_frame", True)
        )
        self.detach_teacher = bool(getattr(args, "videorepa_detach_teacher", True))
        self.encoder_chunk_size = int(getattr(args, "videorepa_encoder_chunk_size", 0))

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("VideoREPA: falling back to hidden_dim=1024")
        return 1024

    def _get_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def _locate_blocks(self) -> Any:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("VideoREPA: could not locate transformer block list")

    def setup_hooks(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        try:
            for i, depth in enumerate(self.alignment_depths):
                if depth >= num_blocks:
                    raise ValueError(
                        f"VideoREPA alignment depth {depth} exceeds available blocks ({num_blocks})"
                    )
                handle = blocks[depth].register_forward_hook(self._get_hook(i))
                self.hook_handles.append(handle)
                logger.info("VideoREPA: hook attached to layer %d.", depth)
        except Exception:
            self.remove_hooks()
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
        return list(self.projectors.parameters())

    @staticmethod
    def _maybe_strip_cls(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            return tokens
        token_count = tokens.shape[1]
        is_square = int(math.isqrt(token_count)) ** 2 == token_count
        is_square_minus_one = int(math.isqrt(token_count - 1)) ** 2 == (token_count - 1)
        if is_square_minus_one and not is_square:
            return tokens[:, 1:, :]
        return tokens

    @staticmethod
    def _coerce_encoder_tokens(features: Any) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            tensor_candidate = None
            for value in features:
                if isinstance(value, torch.Tensor):
                    tensor_candidate = value
                    break
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, torch.Tensor):
                            tensor_candidate = nested
                            break
                if tensor_candidate is not None:
                    break
            if tensor_candidate is None:
                raise ValueError(
                    "VideoREPA: encoder tuple/list output did not contain tensor features"
                )
            features = tensor_candidate

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
                    raise ValueError("VideoREPA: encoder output did not contain tensor features")
                features = tensor_candidate

        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state

        if not isinstance(features, torch.Tensor):
            raise ValueError("VideoREPA: unsupported encoder output type")
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() == 5:
            bsz, channels, frames, height, width = features.shape
            features = features.view(
                bsz, channels, frames * height * width
            ).transpose(1, 2)
        elif features.dim() == 4:
            bsz, channels, height, width = features.shape
            features = features.view(bsz, channels, height * width).transpose(1, 2)
        elif features.dim() != 3:
            raise ValueError(
                f"VideoREPA: expected encoder features with 2-5 dims, got {features.dim()}"
            )
        return VideoRepaHelper._maybe_strip_cls(features)

    def _extract_teacher_tokens(self, clean_pixels: torch.Tensor) -> torch.Tensor:
        if self.teacher_kind == "video":
            return self._extract_teacher_tokens_video(clean_pixels)

        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        bsz, channels, frames, height, width = clean_pixels.shape

        images = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
        images = preprocess_raw_image(images, self.encoder_type)

        chunks = [images]
        if self.encoder_chunk_size > 0 and images.shape[0] > self.encoder_chunk_size:
            chunks = list(torch.split(images, self.encoder_chunk_size, dim=0))

        token_chunks: List[torch.Tensor] = []
        for chunk in chunks:
            with torch.no_grad():
                encoded = self.encoder.forward_features(chunk)
                token_chunks.append(self._coerce_encoder_tokens(encoded))
        tokens = torch.cat(token_chunks, dim=0)
        if tokens.shape[-1] != self.align_dim:
            tokens = F.interpolate(
                tokens, size=self.align_dim, mode="linear", align_corners=False
            )
        if self.max_spatial_tokens > 0 and tokens.shape[1] > self.max_spatial_tokens:
            tokens = _interpolate_token_count(tokens, self.max_spatial_tokens)
        tokens = tokens.view(bsz, frames, tokens.shape[1], tokens.shape[2])
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    def _normalize_for_video_teacher(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            _IMAGENET_MEAN,
            device=frames_bchw.device,
            dtype=frames_bchw.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            _IMAGENET_STD,
            device=frames_bchw.device,
            dtype=frames_bchw.dtype,
        ).view(1, 3, 1, 1)
        return (frames_bchw - mean) / std

    def _extract_teacher_tokens_video(self, clean_pixels: torch.Tensor) -> torch.Tensor:
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)

        if self.video_teacher_drop_first_frame and clean_pixels.shape[2] > 1:
            clean_pixels = clean_pixels[:, :, 1:, :, :]

        bsz, channels, frames, height, width = clean_pixels.shape
        frames_bchw = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        frames_bchw = ((frames_bchw + 1.0) / 2.0).clamp(0.0, 1.0)
        if (
            frames_bchw.shape[-2] != self.video_teacher_input_size[0]
            or frames_bchw.shape[-1] != self.video_teacher_input_size[1]
        ):
            frames_bchw = F.interpolate(
                frames_bchw,
                size=self.video_teacher_input_size,
                mode="bicubic",
                align_corners=False,
            )
        frames_bchw = self._normalize_for_video_teacher(frames_bchw)
        video = frames_bchw.view(
            bsz,
            frames,
            channels,
            self.video_teacher_input_size[0],
            self.video_teacher_input_size[1],
        ).permute(0, 2, 1, 3, 4)

        if self.encoder_chunk_size > 0 and bsz > self.encoder_chunk_size:
            video_chunks = list(torch.split(video, self.encoder_chunk_size, dim=0))
        else:
            video_chunks = [video]

        token_chunks: List[torch.Tensor] = []
        for chunk in video_chunks:
            with torch.no_grad():
                encoded = self.encoder(chunk)
            token_chunks.append(self._coerce_encoder_tokens(encoded))
        tokens = torch.cat(token_chunks, dim=0)

        if tokens.shape[-1] != self.align_dim:
            tokens = F.interpolate(
                tokens, size=self.align_dim, mode="linear", align_corners=False
            )

        spatial_tokens = (
            self.video_teacher_input_size[0] // self.video_teacher_patch_size
        ) * (self.video_teacher_input_size[1] // self.video_teacher_patch_size)
        if spatial_tokens > 0:
            if tokens.shape[1] % spatial_tokens == 0:
                frame_tokens = tokens.shape[1] // spatial_tokens
                tokens = tokens.view(bsz, frame_tokens, spatial_tokens, tokens.shape[-1])
            elif (tokens.shape[1] - 1) > 0 and (tokens.shape[1] - 1) % spatial_tokens == 0:
                tokens = tokens[:, 1:, :]
                frame_tokens = tokens.shape[1] // spatial_tokens
                tokens = tokens.view(bsz, frame_tokens, spatial_tokens, tokens.shape[-1])
            else:
                if not self._shape_warning_logged:
                    logger.warning(
                        "VideoREPA: native teacher tokens (%d) not divisible by spatial token count (%d); using single-frame fallback.",
                        tokens.shape[1],
                        spatial_tokens,
                    )
                    self._shape_warning_logged = True
                tokens = tokens.unsqueeze(1)
        else:
            tokens = tokens.unsqueeze(1)

        if self.max_spatial_tokens > 0 and tokens.shape[2] > self.max_spatial_tokens:
            tokens_2d = tokens.reshape(
                tokens.shape[0] * tokens.shape[1], tokens.shape[2], tokens.shape[3]
            )
            tokens_2d = _interpolate_token_count(tokens_2d, self.max_spatial_tokens)
            tokens = tokens_2d.view(
                tokens.shape[0], tokens.shape[1], self.max_spatial_tokens, tokens.shape[3]
            )

        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    @staticmethod
    def _interpolate_frames(tokens: torch.Tensor, target_frames: int) -> torch.Tensor:
        if tokens.shape[1] == target_frames:
            return tokens
        bsz, frames, token_count, dim = tokens.shape
        x = tokens.permute(0, 2, 3, 1).reshape(bsz * token_count, dim, frames)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        return x.reshape(bsz, token_count, dim, target_frames).permute(0, 3, 1, 2)

    def _reshape_source_tokens(
        self,
        projected: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = projected.shape
        if target_frames > 0 and seq_len % target_frames == 0:
            frames = target_frames
        else:
            frames = 1
            if target_frames > 1:
                for candidate in range(target_frames, 0, -1):
                    if seq_len % candidate == 0:
                        frames = candidate
                        break

        tokens_per_frame = seq_len // max(1, frames)
        source = projected.view(bsz, frames, tokens_per_frame, projected.shape[-1])
        if self.temporal_align and frames != target_frames and target_frames > 0:
            source = self._interpolate_frames(source, target_frames=target_frames)
        return source

    def _match_temporal_and_spatial(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.shape[1] != target.shape[1]:
            if self.temporal_align:
                source = self._interpolate_frames(source, target_frames=target.shape[1])
            else:
                shared_frames = min(source.shape[1], target.shape[1])
                source = source[:, :shared_frames]
                target = target[:, :shared_frames]

        src_tokens = source.shape[2]
        tgt_tokens = target.shape[2]
        if src_tokens != tgt_tokens:
            source_2d = source.reshape(source.shape[0] * source.shape[1], src_tokens, source.shape[3])
            target_2d = target.reshape(target.shape[0] * target.shape[1], tgt_tokens, target.shape[3])
            if self.spatial_align:
                if src_tokens > tgt_tokens:
                    source_2d = _interpolate_token_count(source_2d, tgt_tokens)
                else:
                    target_2d = _interpolate_token_count(target_2d, src_tokens)
            else:
                min_tokens = min(src_tokens, tgt_tokens)
                source_2d = source_2d[:, :min_tokens]
                target_2d = target_2d[:, :min_tokens]
            source = source_2d.view(source.shape[0], source.shape[1], source_2d.shape[1], source.shape[3])
            target = target_2d.view(target.shape[0], target.shape[1], target_2d.shape[1], target.shape[3])

        if self.max_spatial_tokens > 0 and source.shape[2] > self.max_spatial_tokens:
            source_2d = source.reshape(source.shape[0] * source.shape[1], source.shape[2], source.shape[3])
            target_2d = target.reshape(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])
            source_2d = _interpolate_token_count(source_2d, self.max_spatial_tokens)
            target_2d = _interpolate_token_count(target_2d, self.max_spatial_tokens)
            source = source_2d.view(source.shape[0], source.shape[1], self.max_spatial_tokens, source.shape[3])
            target = target_2d.view(target.shape[0], target.shape[1], self.max_spatial_tokens, target.shape[3])
        return source, target

    def _relation_error(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        margin: Optional[float] = None,
    ) -> torch.Tensor:
        effective_margin = self.margin if margin is None else float(margin)
        return F.relu((source - target).abs() - effective_margin).mean()

    def _trd_full_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bsz, frames, tokens, dim = source.shape
        source_frame = source.reshape(bsz * frames, tokens, dim)
        target_frame = target.reshape(bsz * frames, tokens, dim)

        source_video = source.reshape(bsz, frames * tokens, dim)
        target_video = target.reshape(bsz, frames * tokens, dim)
        source_video = (
            source_video.unsqueeze(1)
            .expand(-1, frames, -1, -1)
            .reshape(bsz * frames, frames * tokens, dim)
        )
        target_video = (
            target_video.unsqueeze(1)
            .expand(-1, frames, -1, -1)
            .reshape(bsz * frames, frames * tokens, dim)
        )

        source_sim = torch.bmm(source_frame, source_video.transpose(1, 2))
        target_sim = torch.bmm(target_frame, target_video.transpose(1, 2))
        return self._relation_error(source_sim, target_sim)

    def _trd_spatial_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bsz, frames, tokens, dim = source.shape
        source_bf = source.reshape(bsz * frames, tokens, dim)
        target_bf = target.reshape(bsz * frames, tokens, dim)
        source_sim = torch.bmm(source_bf, source_bf.transpose(1, 2))
        target_sim = torch.bmm(target_bf, target_bf.transpose(1, 2))
        return self._relation_error(
            source_sim,
            target_sim,
            margin=self.margin_matrix,
        )

    def _trd_temporal_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bsz, frames, tokens, dim = source.shape
        if frames <= 1:
            return source.new_tensor(0.0)
        source_flat = source.reshape(bsz, frames * tokens, dim)
        target_flat = target.reshape(bsz, frames * tokens, dim)
        source_sim = torch.bmm(source_flat, source_flat.transpose(1, 2))
        target_sim = torch.bmm(target_flat, target_flat.transpose(1, 2))
        diff = F.relu((source_sim - target_sim).abs() - self.margin_matrix)

        if not self.temporal_exclude_same_frame:
            return diff.mean()
        token_index = torch.arange(frames * tokens, device=source.device)
        frame_index = token_index // tokens
        mask = frame_index[:, None] != frame_index[None, :]
        if not bool(mask.any()):
            return source.new_tensor(0.0)
        return diff[:, mask].mean()

    @staticmethod
    def _cosine_similarity_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (-(source * target).sum(dim=-1)).mean()

    def get_repa_loss(self, clean_pixels: torch.Tensor, vae: Optional[Any] = None) -> torch.Tensor:
        del vae
        if not any(feat is not None for feat in self.captured_features):
            return clean_pixels.new_tensor(0.0)

        target_tokens = self._extract_teacher_tokens(clean_pixels)
        losses: List[torch.Tensor] = []
        for idx, features in enumerate(self.captured_features):
            if features is None:
                continue
            if not isinstance(features, torch.Tensor):
                continue
            if features.dim() != 3:
                if not self._shape_warning_logged:
                    logger.warning(
                        "VideoREPA: expected diffusion features shape [B, Seq, C], got %s. Skipping layer loss.",
                        tuple(features.shape),
                    )
                    self._shape_warning_logged = True
                continue

            projected = self.projectors[idx](features)
            source_tokens = self._reshape_source_tokens(
                projected,
                target_frames=target_tokens.shape[1],
            )
            source_tokens, aligned_target = self._match_temporal_and_spatial(
                source=source_tokens,
                target=target_tokens,
            )
            source_tokens = F.normalize(source_tokens, dim=-1)
            aligned_target = F.normalize(aligned_target, dim=-1)

            if self.relation_mode == "token_relation_distillation":
                layer_loss = self._trd_full_loss(source_tokens, aligned_target)
            elif self.relation_mode == "token_relation_distillation_only_spatial":
                layer_loss = self._trd_spatial_loss(source_tokens, aligned_target)
            elif self.relation_mode == "token_relation_distillation_only_temporal":
                layer_loss = self._trd_temporal_loss(source_tokens, aligned_target)
            elif self.relation_mode == "cosine_similarity":
                layer_loss = self._cosine_similarity_loss(source_tokens, aligned_target)
            else:
                raise ValueError(f"Unsupported videorepa_relation_mode: {self.relation_mode}")
            losses.append(layer_loss)

        self.captured_features = [None] * len(self.alignment_depths)
        if not losses:
            return clean_pixels.new_tensor(0.0)
        total_loss = torch.stack(losses).mean()
        return total_loss * self.loss_lambda
