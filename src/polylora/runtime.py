from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from polylora.encoder import (
    encode_identity_frames,
    encode_style_frames,
    encode_style_frames_multi,
)
from polylora.model import PolyLoRANetwork
from polylora.predict import (
    load_polylora_checkpoint,
    load_polylora_metadata,
    predict_lora_state_dict,
    weighted_merge_state_dicts,
)
from polylora.spec import load_spec_file
from polylora.utils import resolve_device


def _load_frames(frame_paths: list[str]) -> list:
    frames = []
    for p in frame_paths:
        with Image.open(Path(p)) as img:
            frames.append(img.convert("RGB"))
    return frames


def _get_list_setting(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _encode_from_settings(
    frames: list,
    device: str,
    encoder_name: str,
    encoder_type: str,
    encoders: list[str],
    fusion_mode: str,
) -> torch.Tensor:
    if encoders:
        return encode_style_frames_multi(
            frames,
            encoders=[(name, encoder_type) for name in encoders],
            device=device,
            fusion_mode=fusion_mode,
        )
    return encode_style_frames(
        frames,
        model_name=encoder_name,
        encoder_type=encoder_type,
        device=device,
    )


def _build_prediction_from_metadata(
    frames: list,
    ckpt_path: str,
    metadata: Dict[str, Any],
    device: str,
    include_base: bool,
) -> Dict[str, torch.Tensor]:
    spec = load_spec_file(Path(metadata["spec_path"]))
    embed_dim = int(metadata["embed_dim"])
    encoder_name = str(
        metadata.get("encoder") or "openai/clip-vit-large-patch14-336"
    )
    encoder_type = str(metadata.get("encoder_type") or "clip")
    encoders = _get_list_setting(metadata.get("encoders"))
    fusion_mode = str(metadata.get("encoder_fusion_mode") or "mean")
    use_identity = bool(metadata.get("use_identity", False))
    identity_encoder = str(metadata.get("identity_encoder") or "antelopev2")
    use_perceiver = bool(metadata.get("use_perceiver_frontend", False))
    use_residual = bool(metadata.get("use_residual_branch", False))
    residual_encoders = _get_list_setting(metadata.get("residual_encoders"))
    residual_encoder_type = str(
        metadata.get("residual_encoder_type") or encoder_type
    )
    residual_fusion_mode = str(
        metadata.get("residual_fusion_mode") or "concat"
    )
    residual_scale = float(metadata.get("residual_scale", 0.05))
    use_base = bool(metadata.get("use_base_branch", False))

    embedding = _encode_from_settings(
        frames,
        device=device,
        encoder_name=encoder_name,
        encoder_type=encoder_type,
        encoders=encoders,
        fusion_mode=fusion_mode,
    )

    identity_emb = None
    if use_identity:
        identity_emb = encode_identity_frames(
            frames, model_name=identity_encoder, device=device
        )
    residual_emb = None
    if use_residual:
        if not residual_encoders:
            raise ValueError(
                f"PolyLoRA metadata for {ckpt_path} enables residual branch without residual_encoders"
            )
        residual_emb = encode_style_frames_multi(
            frames,
            encoders=[(name, residual_encoder_type) for name in residual_encoders],
            device=device,
            fusion_mode=residual_fusion_mode,
        )

    model = PolyLoRANetwork(
        embed_dim=embed_dim,
        target_specs=spec,
        head_mode=str(metadata.get("head_mode") or "trunk"),
        fusion_mode=str(metadata.get("fusion_mode") or "style_only"),
        identity_dim=identity_emb.shape[-1] if identity_emb is not None else None,
        use_perceiver_frontend=use_perceiver,
        use_residual=use_residual,
        residual_dim=residual_emb.shape[-1] if residual_emb is not None else None,
        residual_scale=residual_scale,
        enable_base_branch=use_base,
    )
    load_polylora_checkpoint(Path(ckpt_path), model)
    model.to(device)
    model.eval()
    return predict_lora_state_dict(
        model,
        embedding,
        identity=identity_emb,
        residual=residual_emb,
        use_perceiver=use_perceiver,
        include_base=include_base and use_base,
    )


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
    head_ckpts = _get_list_setting(getattr(args, "polylora_head_ckpts", None))
    head_weights = getattr(args, "polylora_head_weights", None)
    include_base_final = (
        bool(getattr(args, "polylora_live_include_base", False))
        if include_base is None
        else bool(include_base)
    )
    frames = _load_frames(frame_list)
    if head_ckpts:
        metadata_paths = _get_list_setting(
            getattr(args, "polylora_head_metadata_paths", None)
        )
        if metadata_paths and len(metadata_paths) != len(head_ckpts):
            raise ValueError(
                "polylora_head_metadata_paths must match polylora_head_ckpts length"
            )
        predictions = []
        for idx, ckpt in enumerate(head_ckpts):
            metadata = (
                load_polylora_metadata(Path(metadata_paths[idx]))
                if metadata_paths
                else load_polylora_metadata(Path(ckpt))
            )
            predictions.append(
                _build_prediction_from_metadata(
                    frames,
                    ckpt_path=ckpt,
                    metadata=metadata,
                    device=device,
                    include_base=include_base_final,
                )
            )
        return weighted_merge_state_dicts(predictions, head_weights)

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
    use_residual = bool(getattr(args, "polylora_use_residual_branch", False))
    use_base = bool(getattr(args, "polylora_dual_lora_heads", False))
    ckpt_path = getattr(args, "polylora_ckpt", None)
    if not ckpt_path:
        raise ValueError("polylora_ckpt must be set for live apply.")

    embedding = encode_style_frames(
        frames,
        model_name=encoder_name,
        encoder_type=encoder_type,
        device=device,
    )
    identity_emb = None
    residual_emb = None
    if use_identity:
        identity_emb = encode_identity_frames(frames, model_name=identity_encoder, device=device)
    if getattr(args, "polylora_si_fusion_mode", "style_only") != "style_only" and use_identity and identity_emb is None:
        raise ValueError("Identity fusion requested but identity embeddings were not produced.")
    if use_residual:
        residual_encoders = getattr(args, "polylora_residual_encoders", None)
        if not residual_encoders:
            raise ValueError(
                "polylora_use_residual_branch=true requires polylora_residual_encoders."
            )
        if isinstance(residual_encoders, str):
            residual_encoders = [residual_encoders]
        residual_encoder_type = getattr(
            args, "polylora_residual_encoder_type", encoder_type
        )
        residual_fusion_mode = getattr(
            args, "polylora_residual_fusion_mode", "concat"
        )
        residual_emb = encode_style_frames_multi(
            frames,
            encoders=[(name, residual_encoder_type) for name in residual_encoders],
            device=device,
            fusion_mode=residual_fusion_mode,
        )

    model = PolyLoRANetwork(
        embed_dim=int(embed_dim),
        target_specs=spec,
        head_mode=getattr(args, "polylora_head_mode", "trunk"),
        fusion_mode=getattr(args, "polylora_si_fusion_mode", "style_only"),
        identity_dim=identity_emb.shape[-1] if identity_emb is not None else None,
        use_perceiver_frontend=use_perceiver,
        use_residual=use_residual,
        residual_dim=residual_emb.shape[-1] if residual_emb is not None else None,
        residual_scale=float(getattr(args, "polylora_residual_scale", 0.05)),
        enable_base_branch=use_base,
    )
    load_polylora_checkpoint(Path(ckpt_path), model)
    model.to(device)
    model.eval()
    pred = predict_lora_state_dict(
        model,
        embedding,
        identity=identity_emb,
        residual=residual_emb,
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
