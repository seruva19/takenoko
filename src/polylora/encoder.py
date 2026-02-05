from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoProcessor
import numpy as np

from .utils import resolve_device

_ENCODER_CACHE: Dict[Tuple[str, str, str], Tuple[AutoProcessor, AutoModel]] = {}
_FACE_APP_CACHE: Dict[Tuple[str, str], object] = {}


def encode_style_frames(
    frames: List,
    model_name: str = "openai/clip-vit-large-patch14-336",
    encoder_type: str = "clip",  # "clip" or "video"
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode frames with either an image (CLIP) or video encoder and mean-pool to a single embedding.
    Falls back to CPU if CUDA is unavailable or video path fails.
    """
    device = resolve_device(device)
    key = (model_name, encoder_type, device)
    if key in _ENCODER_CACHE:
        processor, model = _ENCODER_CACHE[key]
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        _ENCODER_CACHE[key] = (processor, model)
    try:
        if encoder_type == "video" and hasattr(processor, "videos"):
            inputs = processor(videos=[frames], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            if hasattr(outputs, "video_features"):
                feats = outputs.video_features
            elif hasattr(outputs, "last_hidden_state"):
                feats = outputs.last_hidden_state.mean(dim=1)
            else:
                feats = model.get_video_features(**inputs)  # type: ignore[attr-defined]
            emb = feats.squeeze(0)
        else:
            inputs = processor(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)  # [N, D]
            emb = feats.mean(dim=0)
        emb = emb / emb.norm()
        return emb
    except Exception:
        # Fallback to image path if video encoding fails
        inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        emb = feats.mean(dim=0)
        emb = emb / emb.norm()
        return emb


def encode_style_frames_multi(
    frames: List,
    encoders: List[Tuple[str, str]],  # list of (model_name, encoder_type)
    device: str = "cuda",
    fusion_mode: str = "mean",
) -> torch.Tensor:
    """Encode frames with multiple encoders and fuse outputs."""
    if not encoders:
        raise ValueError("encode_style_frames_multi requires at least one encoder.")
    embs: List[torch.Tensor] = []
    for model_name, enc_type in encoders:
        embs.append(encode_style_frames(frames, model_name=model_name, encoder_type=enc_type, device=device))
    if fusion_mode == "mean":
        stacked = torch.stack(embs, dim=0)
        fused = stacked.mean(dim=0)
    elif fusion_mode == "concat":
        fused = torch.cat(embs, dim=-1)
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
    fused = fused / fused.norm()
    return fused


def encode_identity_frames(
    frames: List,
    model_name: str = "antelopev2",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode identity embeddings using an InsightFace/ArcFace-style encoder.
    This expects frames to already be cropped/aligned; we do not run face detection here.
    """
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "insightface is required for identity encoding; install insightface and provide a compatible model."
        ) from exc

    device = resolve_device(device)
    app_key = (model_name, device)
    if app_key in _FACE_APP_CACHE:
        app = _FACE_APP_CACHE[app_key]  # type: ignore[assignment]
    else:
        app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"])
        app.prepare(ctx_id=0 if device == "cuda" else -1)
        _FACE_APP_CACHE[app_key] = app  # type: ignore[assignment]

    embs: List[torch.Tensor] = []
    for img in frames:
        np_img = np.array(img)
        faces = app.get(np_img)
        if not faces:
            continue
        # Use the first detected face; align and embed
        face = faces[0]
        embs.append(torch.tensor(face.normed_embedding, device=device))
    if not embs:
        raise RuntimeError("No faces detected for identity encoding; provide cropped/aligned faces or different frames.")
    fused = torch.stack(embs, dim=0).mean(dim=0)
    fused = fused / fused.norm()
    return fused
