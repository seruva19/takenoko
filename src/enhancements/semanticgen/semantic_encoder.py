from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image

logger = get_logger(__name__, level=logging.INFO)

_HF_CACHE: dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


def _to_pil_list(frames: torch.Tensor) -> List[Image.Image]:
    frames = frames.detach()
    if frames.dtype != torch.uint8:
        frames = frames.float()
        frames = (frames + 1.0) * 127.5
        frames = frames.clamp(0, 255).to(torch.uint8)
    np_frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(frame) for frame in np_frames]


def _infer_stride(target_fps: float, semantic_fps: float, fallback_stride: int) -> int:
    if semantic_fps <= 0:
        return max(1, fallback_stride)
    stride_from_fps = max(1, int(round(target_fps / semantic_fps)))
    return max(1, fallback_stride, stride_from_fps)


class SemanticEncoder:
    """Unified semantic encoder wrapper with REPA-style and HF backends."""

    def __init__(
        self,
        model_name: str,
        encoder_type: str,
        device: str,
        dtype: torch.dtype,
        input_resolution: int,
        cache_dir: str,
    ) -> None:
        self.model_name = model_name
        self.encoder_type = encoder_type
        self.device = device
        self.dtype = dtype
        self.input_resolution = input_resolution
        self.cache_dir = cache_dir
        self._repa_encoder: Optional[torch.nn.Module] = None
        self._repa_encoder_type: Optional[str] = None
        self._hf_processor: Optional[Any] = None
        self._hf_model: Optional[Any] = None

        if self.encoder_type == "repa":
            self._init_repa_encoder()
        else:
            self._init_hf_encoder()

    def _init_repa_encoder(self) -> None:
        encoder_manager = EncoderManager(self.device, self.cache_dir)
        encoders, encoder_types, _ = encoder_manager.load_encoders(
            self.model_name, resolution=self.input_resolution
        )
        self._repa_encoder = encoders[0]
        self._repa_encoder_type = encoder_types[0]
        self._repa_encoder.eval()
        logger.info(
            "Semantic encoder (REPA) loaded: %s (type=%s)",
            self.model_name,
            self._repa_encoder_type,
        )

    def _init_hf_encoder(self) -> None:
        cache_key = (self.model_name, self.encoder_type, self.device)
        if cache_key in _HF_CACHE:
            self._hf_processor, self._hf_model = _HF_CACHE[cache_key]
            return
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(
            self.device, dtype=self.dtype
        )
        model.eval()
        _HF_CACHE[cache_key] = (processor, model)
        self._hf_processor = processor
        self._hf_model = model
        logger.info(
            "Semantic encoder (HF) loaded: %s (type=%s)",
            self.model_name,
            self.encoder_type,
        )

    def encode_frames(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        # SPEC:semanticgen_lora:encoder - unify semantic encoder outputs for conditioning and caching.
        if self.encoder_type == "repa":
            return self._encode_repa(frames)
        return self._encode_hf(frames)

    def _encode_repa(self, frames: torch.Tensor) -> torch.Tensor:
        if self._repa_encoder is None or self._repa_encoder_type is None:
            raise RuntimeError("REPA encoder not initialized.")
        images = preprocess_raw_image(frames, self._repa_encoder_type)
        with torch.no_grad():
            features = self._repa_encoder.forward_features(images)
        tokens = _extract_encoder_tokens(features)
        if tokens is None:
            raise RuntimeError("REPA encoder did not return usable tokens.")
        if tokens.dim() == 3:
            tokens = tokens.mean(dim=1)
        return tokens

    def _encode_hf(self, frames: torch.Tensor) -> torch.Tensor:
        if self._hf_processor is None or self._hf_model is None:
            raise RuntimeError("HF encoder not initialized.")
        pil_frames = _to_pil_list(frames)
        inputs = self._hf_processor(images=pil_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            embeds = outputs.image_embeds
        elif hasattr(self._hf_model, "get_image_features"):
            embeds = self._hf_model.get_image_features(**inputs)
        elif hasattr(outputs, "last_hidden_state"):
            embeds = outputs.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("HF encoder output does not expose image features.")
        return embeds


def _extract_encoder_tokens(features: Any) -> Optional[torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"]
        for value in features.values():
            if torch.is_tensor(value) and value.dim() == 3:
                return value
        return None
    if torch.is_tensor(features):
        if features.dim() == 2:
            return features
        if features.dim() == 3:
            return features
    return None


def sample_frames(
    frames: torch.Tensor,
    *,
    target_fps: float,
    semantic_fps: float,
    stride: int,
    frame_limit: Optional[int],
) -> torch.Tensor:
    frame_count = frames.shape[0]
    effective_stride = _infer_stride(target_fps, semantic_fps, stride)
    indices = list(range(0, frame_count, effective_stride))
    if frame_limit is not None and frame_limit > 0:
        indices = indices[:frame_limit]
    if not indices:
        indices = [0]
    return frames[indices]
