from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from polylora.encoder import encode_identity_frames, encode_style_frames
from polylora.model import PolyLoRANetwork
from polylora.predict import load_polylora_checkpoint, predict_lora_state_dict
from polylora.spec import load_spec_file
from polylora.utils import resolve_device


def _load_frames(frame_paths: list[str]) -> list:
    frames = []
    for p in frame_paths:
        with Image.open(Path(p)) as img:
            frames.append(img.convert("RGB"))
    return frames


def predict_lora_from_args(
    args,
    device: Optional[str] = None,
    include_base: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    frame_list = getattr(args, "polylora_predict_frames", None)
    if not frame_list:
        raise ValueError("polylora_predict_frames must be set for live apply.")
    if isinstance(frame_list, str):
        frame_list = [frame_list]
    device = resolve_device(device or "cuda")
    spec_path = getattr(args, "polylora_spec", None)
    if not spec_path:
        raise ValueError("polylora_spec must be set for live apply.")
    spec = load_spec_file(Path(spec_path))
    embed_dim = getattr(args, "polylora_embed_dim", None)
    if embed_dim is None:
        raise ValueError("polylora_embed_dim must be set for live apply.")
    encoder_name = getattr(args, "polylora_encoder", "openai/clip-vit-large-patch14-336")
    encoder_type = getattr(args, "polylora_encoder_type", "clip")
    use_identity = bool(getattr(args, "polylora_use_identity", False))
    identity_encoder = getattr(args, "polylora_identity_encoder", "antelopev2")
    use_perceiver = bool(getattr(args, "polylora_use_perceiver_frontend", False))
    use_base = bool(getattr(args, "polylora_dual_lora_heads", False))
    ckpt_path = getattr(args, "polylora_ckpt", None)
    if not ckpt_path:
        raise ValueError("polylora_ckpt must be set for live apply.")

    frames = _load_frames(frame_list)
    embedding = encode_style_frames(
        frames,
        model_name=encoder_name,
        encoder_type=encoder_type,
        device=device,
    )
    identity_emb = None
    if use_identity:
        identity_emb = encode_identity_frames(frames, model_name=identity_encoder, device=device)
    if getattr(args, "polylora_si_fusion_mode", "style_only") != "style_only" and use_identity and identity_emb is None:
        raise ValueError("Identity fusion requested but identity embeddings were not produced.")

    model = PolyLoRANetwork(
        embed_dim=int(embed_dim),
        target_specs=spec,
        head_mode=getattr(args, "polylora_head_mode", "trunk"),
        fusion_mode=getattr(args, "polylora_si_fusion_mode", "style_only"),
        identity_dim=identity_emb.shape[-1] if identity_emb is not None else None,
        use_perceiver_frontend=use_perceiver,
        enable_base_branch=use_base,
    )
    load_polylora_checkpoint(Path(ckpt_path), model)
    model.eval()
    pred = predict_lora_state_dict(
        model,
        embedding,
        identity=identity_emb,
        use_perceiver=use_perceiver,
        include_base=use_base if include_base is None else include_base,
    )
    return pred


def merge_lora_into_network(
    network,
    transformer,
    weights_sd: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    non_blocking: bool = False,
) -> None:
    if not hasattr(network, "merge_to"):
        raise ValueError("Network does not support merge_to; cannot apply PolyLoRA.")
    try:
        network.merge_to(None, transformer, weights_sd, dtype, device, non_blocking)
        return
    except TypeError:
        pass
    network.merge_to(weights_sd, dtype, device, non_blocking)
