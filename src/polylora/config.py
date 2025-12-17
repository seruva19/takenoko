from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PolyLoRAConfig:
    enable: bool = False
    ckpt: Optional[str] = None
    spec: Optional[str] = None
    encoder: str = "openai/clip-vit-large-patch14-336"
    encoder_type: str = "clip"
    encoders: Optional[list[str]] = None
    fusion_mode: str = "mean"
    head_mode: str = "trunk"
    use_identity: bool = False
    identity_encoder: str = "antelopev2"
    si_fusion_mode: str = "style_only"
    use_perceiver_frontend: bool = False
    dual_lora_heads: bool = False
    base_attenuate: float = 0.5
    use_cached_embeddings: bool = True
    smoke_command: Optional[str] = None
    merge_target: Optional[str] = None
    embed_dim: Optional[int] = None
    frames_root: Optional[str] = None
    pairs_out: Optional[str] = None
    lora_paths: Optional[list[str]] = None
    base_lora_paths: Optional[list[str]] = None
    predict_frames: Optional[list[str]] = None
    predict_out: str = "predicted_lora.pt"
    train_lr: float = 1e-3
    train_epochs: int = 5
    train_batch_size: int = 4
    cosine_loss_weight: float = 0.0
    sample_every_epochs: int = 0
    sample_command: Optional[str] = None
    sample_dir: str = "polylora_samples"
    sample_merge_target: Optional[str] = None


def parse_polylora_config(raw: Dict[str, Any]) -> PolyLoRAConfig:
    cfg = PolyLoRAConfig()
    cfg.enable = bool(raw.get("enable_polylora", False))
    cfg.ckpt = raw.get("polylora_ckpt")
    cfg.spec = raw.get("polylora_spec")
    cfg.encoder = raw.get("polylora_encoder", cfg.encoder)
    cfg.encoder_type = raw.get("polylora_encoder_type", cfg.encoder_type)
    cfg.encoders = raw.get("polylora_encoders", None)
    cfg.fusion_mode = str(raw.get("polylora_fusion_mode", cfg.fusion_mode)).lower()
    if cfg.fusion_mode not in ("mean", "concat"):
        raise ValueError("polylora_fusion_mode must be 'mean' or 'concat'")
    cfg.head_mode = str(raw.get("polylora_head_mode", cfg.head_mode)).lower()
    if cfg.head_mode not in ("trunk", "per_tensor"):
        raise ValueError("polylora_head_mode must be 'trunk' or 'per_tensor'")
    cfg.use_identity = bool(raw.get("polylora_use_identity", False))
    cfg.identity_encoder = raw.get("polylora_identity_encoder", cfg.identity_encoder)
    cfg.si_fusion_mode = str(raw.get("polylora_si_fusion_mode", cfg.si_fusion_mode)).lower()
    if cfg.si_fusion_mode not in ("style_only", "mean", "gated", "concat"):
        raise ValueError("polylora_si_fusion_mode must be one of style_only|mean|gated|concat")
    cfg.use_perceiver_frontend = bool(raw.get("polylora_use_perceiver_frontend", False))
    cfg.dual_lora_heads = bool(raw.get("polylora_dual_lora_heads", False))
    cfg.base_attenuate = float(raw.get("polylora_base_attenuate", cfg.base_attenuate))
    cfg.use_cached_embeddings = bool(raw.get("polylora_use_cached_embeddings", True))
    cfg.smoke_command = raw.get("polylora_smoke_command")
    cfg.merge_target = raw.get("polylora_merge_target")
    cfg.embed_dim = raw.get("polylora_embed_dim")
    cfg.frames_root = raw.get("polylora_frames_root")
    cfg.pairs_out = raw.get("polylora_pairs_out")
    cfg.lora_paths = raw.get("polylora_lora_paths")
    cfg.base_lora_paths = raw.get("polylora_base_lora_paths")
    cfg.predict_frames = raw.get("polylora_predict_frames")
    cfg.predict_out = raw.get("polylora_predict_out", cfg.predict_out)
    cfg.train_lr = float(raw.get("polylora_train_lr", cfg.train_lr))
    cfg.train_epochs = int(raw.get("polylora_train_epochs", cfg.train_epochs))
    cfg.train_batch_size = int(raw.get("polylora_train_batch_size", cfg.train_batch_size))
    cfg.cosine_loss_weight = float(raw.get("polylora_cosine_loss_weight", cfg.cosine_loss_weight))
    cfg.sample_every_epochs = int(raw.get("polylora_sample_every_epochs", cfg.sample_every_epochs))
    cfg.sample_command = raw.get("polylora_sample_command")
    cfg.sample_dir = raw.get("polylora_sample_dir", cfg.sample_dir)
    cfg.sample_merge_target = raw.get("polylora_sample_merge_target")

    if cfg.enable:
        if not isinstance(cfg.ckpt, str) or not isinstance(cfg.spec, str):
            raise ValueError("enable_polylora requires polylora_ckpt and polylora_spec to be set")
        if cfg.embed_dim is None:
            raise ValueError("enable_polylora requires polylora_embed_dim to match the encoder output dimension")
        if cfg.dual_lora_heads and not cfg.base_lora_paths:
            # Heuristic fallback exists, but warn early if not supplied
            pass
    return cfg


def apply_polylora_to_args(args: Any, raw: Dict[str, Any]) -> Any:
    cfg = parse_polylora_config(raw)
    args.enable_polylora = cfg.enable
    args.polylora_ckpt = cfg.ckpt
    args.polylora_spec = cfg.spec
    args.polylora_encoder = cfg.encoder
    args.polylora_encoder_type = cfg.encoder_type
    args.polylora_encoders = cfg.encoders
    args.polylora_fusion_mode = cfg.fusion_mode
    args.polylora_head_mode = cfg.head_mode
    args.polylora_use_identity = cfg.use_identity
    args.polylora_identity_encoder = cfg.identity_encoder
    args.polylora_si_fusion_mode = cfg.si_fusion_mode
    args.polylora_use_perceiver_frontend = cfg.use_perceiver_frontend
    args.polylora_dual_lora_heads = cfg.dual_lora_heads
    args.polylora_base_attenuate = cfg.base_attenuate
    args.polylora_use_cached_embeddings = cfg.use_cached_embeddings
    args.polylora_smoke_command = cfg.smoke_command
    args.polylora_merge_target = cfg.merge_target
    args.polylora_embed_dim = cfg.embed_dim
    args.polylora_frames_root = cfg.frames_root
    args.polylora_pairs_out = cfg.pairs_out
    args.polylora_lora_paths = cfg.lora_paths
    args.polylora_base_lora_paths = cfg.base_lora_paths
    args.polylora_predict_frames = cfg.predict_frames
    args.polylora_predict_out = cfg.predict_out
    args.polylora_train_lr = cfg.train_lr
    args.polylora_train_epochs = cfg.train_epochs
    args.polylora_train_batch_size = cfg.train_batch_size
    args.polylora_cosine_loss_weight = cfg.cosine_loss_weight
    args.polylora_sample_every_epochs = cfg.sample_every_epochs
    args.polylora_sample_command = cfg.sample_command
    args.polylora_sample_dir = cfg.sample_dir
    args.polylora_sample_merge_target = cfg.sample_merge_target
    return args
