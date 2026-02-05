"""
PolyLoRA integration helpers for Takenoko trainer.
Encapsulates menu actions and non-interactive flows.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import torch
from PIL import Image

from polylora.dataset import PairSample, PolyLoRAPairDataset, save_sharded_samples
from polylora.encoder import encode_style_frames, encode_style_frames_multi, encode_identity_frames
from polylora.model import PolyLoRANetwork
from polylora.predict import load_polylora_checkpoint, predict_lora_state_dict
from polylora.spec import (
    LoRATargetSpec,
    collect_lora_specs,
    ensure_specs_consistent,
    load_lora_state_dict,
    load_spec_file,
    validate_lora_matches_spec,
)
from polylora.train import TrainConfig, train_polylora
from polylora.utils import resolve_device


class PolyLoRAController:
    def __init__(self, trainer):
        self.trainer = trainer

    def _prompt_list(self, prompt: str, default: list[str] | None = None) -> list[str]:
        raw = input(prompt).strip()
        if not raw and default:
            return default
        return [s.strip() for s in raw.split(",") if s.strip()]

    def _prompt_value(
        self,
        prompt: str,
        default=None,
        required: bool = False,
        cast=None,
    ):
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                return default
            if required:
                print("? Value is required.")
                return None
            return None
        try:
            return cast(raw) if cast else raw
        except Exception:
            print(f"? Invalid value for {prompt}")
            return None

    def collect_spec(self, non_interactive: bool = False) -> bool:
        args = self.trainer.args
        if non_interactive:
            lora_paths = getattr(args, "polylora_lora_paths", None)
            if not lora_paths:
                print("? polylora_lora_paths must be set in config for non-interactive spec collection.")
                return False
            if isinstance(lora_paths, str):
                lora_paths = [lora_paths]
        else:
            lora_paths = self._prompt_list(
                "Enter LoRA checkpoint paths (comma-separated; .pt or .safetensors): "
            )
        if not lora_paths:
            print("? No LoRA paths provided.")
            return False
        default_out = getattr(args, "polylora_spec", "specs/polylora_spec.json")
        out_path = (
            default_out if non_interactive else self._prompt_value(f"Spec output path [{default_out}]: ", default=default_out)
        )
        if not out_path:
            print("? Spec output path is required.")
            return False
        first_spec = None
        expected_names: list[str] = []
        for path in lora_paths:
            sd = load_lora_state_dict(path)
            specs = collect_lora_specs(sd)
            if first_spec is None:
                first_spec = specs
                expected_names = [spec.name for spec in specs]
            else:
                names = [spec.name for spec in specs]
                if names != expected_names:
                    raise ValueError(f"Spec mismatch for {path}: {names} != {expected_names}")
        from polylora.spec import dump_spec_file

        dump_spec_file(first_spec or [], out_path)
        print(f"[polylora] wrote spec with {len(first_spec or [])} targets to {out_path}")
        return True

    def build_pairs(self, non_interactive: bool = False) -> bool:
        args = self.trainer.args
        try:
            if non_interactive:
                frames_root = getattr(args, "polylora_frames_root", None)
                lora_paths = getattr(args, "polylora_lora_paths", None)
                base_lora_paths = getattr(args, "polylora_base_lora_paths", None)
            else:
                frames_root = self._prompt_value(
                    "Frames root (folders named after LoRA checkpoint stems): ",
                    default=getattr(args, "polylora_frames_root", None),
                )
                lora_paths = self._prompt_list("LoRA checkpoints (comma-separated): ")
                base_lora_paths = self._prompt_list(
                    "Base LoRA checkpoints (comma-separated, optional for dual-head): ",
                    default=getattr(args, "polylora_base_lora_paths", None),
                )

            if not frames_root:
                print("? Frames root is required.")
                return False
            if not lora_paths:
                print("? No LoRA paths provided.")
                return False
            if isinstance(lora_paths, str):
                lora_paths = [lora_paths]
            default_spec = getattr(args, "polylora_spec", "specs/polylora_spec.json")
            spec_path = (
                default_spec
                if non_interactive
                else self._prompt_value(f"Spec path [{default_spec}]: ", default=default_spec)
            )
            spec = load_spec_file(Path(spec_path))
            out_dir = (
                getattr(args, "polylora_pairs_out", None)
                if non_interactive
                else self._prompt_value(
                    "Output directory for pair samples (e.g., data/polylora_pairs): ",
                    default=getattr(args, "polylora_pairs_out", None),
                )
            )
            if not out_dir:
                print("? Output directory is required.")
                return False
            use_cached_embeddings = bool(getattr(args, "polylora_use_cached_embeddings", True))
            cache_dir = Path(out_dir) / "embedding_cache" if use_cached_embeddings else None
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
            encoder_name = (
                getattr(args, "polylora_encoder", "openai/clip-vit-large-patch14-336")
                if non_interactive
                else self._prompt_value(
                    f"Encoder model name [{getattr(args, 'polylora_encoder', 'openai/clip-vit-large-patch14-336')}]: ",
                    default=getattr(args, "polylora_encoder", "openai/clip-vit-large-patch14-336"),
                )
            )
            encoder_type = (
                getattr(args, "polylora_encoder_type", "clip")
                if non_interactive
                else self._prompt_value(
                    f"Encoder type [clip|video] [{getattr(args, 'polylora_encoder_type', 'clip')}]: ",
                    default=getattr(args, "polylora_encoder_type", "clip"),
                )
            )
            fusion_mode = getattr(args, "polylora_fusion_mode", "mean")
            use_identity = bool(getattr(args, "polylora_use_identity", False))
            use_base = bool(getattr(args, "polylora_dual_lora_heads", False))
            identity_encoder = getattr(args, "polylora_identity_encoder", "antelopev2")
            enc_fusion_cfg = getattr(args, "polylora_encoders", None)
            encoder_fusion = None
            if enc_fusion_cfg:
                if isinstance(enc_fusion_cfg, str):
                    enc_fusion_cfg = [enc_fusion_cfg]
                encoder_fusion = [(name, encoder_type) for name in enc_fusion_cfg]  # type: ignore[list-item]
            device = "cuda" if non_interactive else self._prompt_value("Device [cuda]: ", default="cuda")
            device = resolve_device(device)
            base_map = {}
            if base_lora_paths:
                if isinstance(base_lora_paths, str):
                    base_lora_paths = [base_lora_paths]
                for p in base_lora_paths:
                    base_map[Path(p).stem] = p
            auto_base = use_base and not base_map

            samples: list[PairSample] = []
            frames_root_path = Path(frames_root)
            for lora_path_str in lora_paths:
                lora_path = Path(lora_path_str)
                stem = lora_path.stem
                frame_dir = frames_root_path / stem
                if not frame_dir.exists():
                    print(f"[polylora] skipping {lora_path} (no frames at {frame_dir})")
                    continue
                frame_paths = sorted(
                    [p for p in frame_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}]
                )
                if not frame_paths:
                    print(f"[polylora] skipping {lora_path} (no images in {frame_dir})")
                    continue
                frames = []
                for fp in frame_paths:
                    with Image.open(fp) as img:
                        frames.append(img.convert("RGB"))
                identity_emb = None
                cache_hit = False
                cache_path = None
                if cache_dir is not None:
                    cache_path = cache_dir / f"{stem}.pt"
                    if cache_path.exists():
                        try:
                            cached = torch.load(cache_path, map_location="cpu")
                            embedding = cached.get("embedding")
                            identity_emb = cached.get("identity")
                            if embedding is not None:
                                cache_hit = True
                        except Exception as exc:
                            print(f"[polylora] cache read failed for {cache_path}: {exc}")
                if not cache_hit:
                    if encoder_fusion:
                        embedding = encode_style_frames_multi(
                            frames, encoders=encoder_fusion, device=device, fusion_mode=fusion_mode
                        )
                    else:
                        embedding = encode_style_frames(
                            frames,
                            model_name=encoder_name,
                            encoder_type=encoder_type,
                            device=device,
                        )
                    if use_identity:
                        try:
                            identity_emb = encode_identity_frames(frames, model_name=identity_encoder, device=device)
                        except Exception as exc:
                            print(f"[polylora] identity encoding failed for {lora_path}: {exc}")
                    if cache_path is not None:
                        try:
                            record = {"embedding": embedding.cpu()}
                            if identity_emb is not None:
                                record["identity"] = identity_emb.cpu()
                            torch.save(record, cache_path)
                        except Exception as exc:
                            print(f"[polylora] cache write failed for {cache_path}: {exc}")
                lora_sd = load_lora_state_dict(lora_path)
                validate_lora_matches_spec(lora_sd, spec or collect_lora_specs(lora_sd))
                base_lora = None
                if use_base:
                    base_path = base_map.get(lora_path.stem)
                    if base_path is None and auto_base:
                        print(
                            f"[polylora] dual-heads: auto-deriving base LoRA for {lora_path.stem} (attenuate={getattr(args, 'polylora_base_attenuate', 0.5)})"
                        )
                        base_lora = {
                            k: v * getattr(args, "polylora_base_attenuate", 0.5)
                            if k.endswith(("lora_down.weight", "lora_up.weight"))
                            else v
                            for k, v in lora_sd.items()
                        }
                    elif base_path is None:
                        raise ValueError(f"dual_lora_heads enabled but no base LoRA provided for stem {lora_path.stem}")
                    else:
                        base_lora = load_lora_state_dict(base_path)
                    if base_lora is not None:
                        validate_lora_matches_spec(base_lora, spec or collect_lora_specs(lora_sd))
                samples.append(PairSample(embedding=embedding, lora=lora_sd, identity=identity_emb, base_lora=base_lora))

            if not samples:
                print("? No samples created; check inputs.")
                return False

            save_sharded_samples(samples, Path(out_dir), shard_size=32)
            print(f"[polylora] wrote {len(samples)} samples to {out_dir}")
            return True
        except Exception as e:
            print(f"? Error building pairs: {e}")
            return False

    def train_network(self, non_interactive: bool = False) -> bool:
        args = self.trainer.args
        try:
            spec_path = (
                getattr(args, "polylora_spec", "specs/polylora_spec.json")
                if non_interactive
                else self._prompt_value(
                    f"Spec path [{getattr(args, 'polylora_spec', 'specs/polylora_spec.json')}]: ",
                    default=getattr(args, "polylora_spec", "specs/polylora_spec.json"),
                )
            )
            spec: List[LoRATargetSpec] = load_spec_file(Path(spec_path))
            data_dir = (
                getattr(args, "polylora_pairs_out", None)
                if non_interactive
                else self._prompt_value(
                    "Pair data directory or files (comma-separated): ",
                    default=getattr(args, "polylora_pairs_out", None),
                )
            )
            if not data_dir:
                print("? Data directory or files are required.")
                return False
            data_items = [s.strip() for s in data_dir.split(",") if s.strip()]
            shard_paths: list[Path] = []
            for item in data_items:
                p = Path(item)
                if p.is_dir():
                    shard_paths.extend(sorted([f for f in p.iterdir() if f.suffix == ".pt"]))
                else:
                    shard_paths.append(p)
            if not shard_paths:
                print("? No shard files found for training.")
                return False
            embed_dim = getattr(args, "polylora_embed_dim", None)
            embed_dim_input = (
                embed_dim
                if non_interactive
                else self._prompt_value(
                    f"Embedding dimension [{embed_dim or 'auto from shard'}]: ",
                    default=embed_dim,
                )
            )
            if embed_dim_input is not None:
                embed_dim = int(embed_dim_input)
            lr = float(
                getattr(args, "polylora_train_lr", 1e-3)
                if non_interactive
                else self._prompt_value(
                    "Learning rate [1e-3]: ",
                    default=getattr(args, "polylora_train_lr", 1e-3),
                )
            )
            epochs = int(
                getattr(args, "polylora_train_epochs", 5)
                if non_interactive
                else self._prompt_value(
                    "Epochs [5]: ", default=getattr(args, "polylora_train_epochs", 5)
                )
            )
            batch_size = int(
                getattr(args, "polylora_train_batch_size", 4)
                if non_interactive
                else self._prompt_value(
                    "Batch size [4]: ",
                    default=getattr(args, "polylora_train_batch_size", 4),
                )
            )
            cosine_w = float(
                getattr(args, "polylora_cosine_loss_weight", 0.0)
                if non_interactive
                else self._prompt_value(
                    "Cosine loss weight [0.0]: ",
                    default=getattr(args, "polylora_cosine_loss_weight", 0.0),
                )
            )
            device = "cuda" if non_interactive else self._prompt_value("Device [cuda]: ", default="cuda")
            ckpt_out = (
                getattr(args, "polylora_ckpt", "checkpoints/polylora.pt")
                if non_interactive
                else self._prompt_value(
                    f"Checkpoint output path [{getattr(args, 'polylora_ckpt', 'checkpoints/polylora.pt')}]: ",
                    default=getattr(args, "polylora_ckpt", "checkpoints/polylora.pt"),
                )
            )
            sample_every_val = (
                getattr(args, "polylora_sample_every_epochs", 0)
                if non_interactive
                else self._prompt_value(
                    "Run sampling every N epochs (0 to disable) [0]: ",
                    default=getattr(args, "polylora_sample_every_epochs", 0),
                )
            )
            sample_every = int(sample_every_val or 0)
            sample_cmd = (
                getattr(args, "polylora_sample_command", None)
                if non_interactive
                else self._prompt_value(
                    "Sample command (use {lora} / {merged} / {epoch} placeholders, empty to skip): ",
                    default=getattr(args, "polylora_sample_command", None),
                )
            )
            sample_cmd = sample_cmd or None
            sample_dir = (
                getattr(args, "polylora_sample_dir", "polylora_samples")
                if non_interactive
                else self._prompt_value(
                    f"Sample output dir [{getattr(args, 'polylora_sample_dir', 'polylora_samples')}]: ",
                    default=getattr(args, "polylora_sample_dir", "polylora_samples"),
                )
            )
            sample_merge_target = (
                getattr(args, "polylora_sample_merge_target", None)
                if non_interactive
                else self._prompt_value(
                    "Optional base checkpoint to merge predicted LoRA for sampling (empty to skip): ",
                    default=getattr(args, "polylora_sample_merge_target", None),
                )
            )
            sample_merge_target = sample_merge_target or None

            fusion_mode = getattr(args, "polylora_si_fusion_mode", "style_only")
            use_perceiver = bool(getattr(args, "polylora_use_perceiver_frontend", False))
            use_identity = bool(getattr(args, "polylora_use_identity", False))
            use_base = bool(getattr(args, "polylora_dual_lora_heads", False))
            dataset = PolyLoRAPairDataset(shard_paths, expected_spec=spec)
            if len(dataset) == 0:
                print("? No samples found in shard files.")
                return False
            first_emb, _, first_id, _ = dataset[0]
            if embed_dim is None:
                embed_dim = first_emb.shape[-1]
            identity_dim = first_id.shape[-1] if first_id is not None else None
            if fusion_mode != "style_only" and use_identity and identity_dim is None:
                raise ValueError("Identity fusion requested but shards do not contain identity embeddings.")
            model = PolyLoRANetwork(
                embed_dim=embed_dim,
                target_specs=spec,
                head_mode=getattr(args, "polylora_head_mode", "trunk"),
                fusion_mode=fusion_mode,
                identity_dim=identity_dim,
                use_perceiver_frontend=use_perceiver,
                enable_base_branch=use_base,
            )
            stats = train_polylora(
                model,
                dataset,
                TrainConfig(
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    weight_decay=float(getattr(args, "polylora_train_weight_decay", 0.0)),
                    val_split=float(getattr(args, "polylora_train_val_split", 0.1)),
                    amp=bool(getattr(args, "polylora_train_amp", True)),
                    grad_clip=getattr(args, "polylora_train_grad_clip", 1.0),
                    cosine_loss_weight=cosine_w,
                    sample_every_epochs=sample_every,
                    sample_command=sample_cmd,
                    sample_dir=sample_dir,
                    sample_merge_target=sample_merge_target,
                    use_identity=use_identity,
                    use_perceiver_frontend=use_perceiver,
                    use_base_branch=use_base,
                    base_loss_weight=float(getattr(args, "polylora_base_loss_weight", 1.0)),
                    use_ema=bool(getattr(args, "polylora_use_ema", False)),
                    ema_decay=float(getattr(args, "polylora_ema_decay", 0.995)),
                ),
                device=device,
            )
            torch.save(model.state_dict(), ckpt_out)
            print(f"[polylora] saved checkpoint to {ckpt_out}")
            if stats:
                print(f"[polylora] stats: {stats}")
            if bool(getattr(args, "polylora_save_metadata", True)):
                try:
                    spec_bytes = Path(spec_path).read_bytes()
                    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
                except Exception:
                    spec_hash = "unknown"
                meta = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ckpt_path": str(ckpt_out),
                    "spec_path": str(spec_path),
                    "spec_sha256": spec_hash,
                    "num_samples": len(dataset),
                    "embed_dim": embed_dim,
                    "identity_dim": identity_dim,
                    "encoder": getattr(args, "polylora_encoder", None),
                    "encoder_type": getattr(args, "polylora_encoder_type", None),
                    "fusion_mode": getattr(args, "polylora_si_fusion_mode", None),
                    "head_mode": getattr(args, "polylora_head_mode", None),
                    "use_identity": use_identity,
                    "use_perceiver_frontend": use_perceiver,
                    "use_base_branch": use_base,
                    "train_config": {
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "weight_decay": float(getattr(args, "polylora_train_weight_decay", 0.0)),
                        "val_split": float(getattr(args, "polylora_train_val_split", 0.1)),
                        "amp": bool(getattr(args, "polylora_train_amp", True)),
                        "grad_clip": getattr(args, "polylora_train_grad_clip", 1.0),
                        "cosine_loss_weight": cosine_w,
                        "base_loss_weight": float(getattr(args, "polylora_base_loss_weight", 1.0)),
                        "use_ema": bool(getattr(args, "polylora_use_ema", False)),
                        "ema_decay": float(getattr(args, "polylora_ema_decay", 0.995)),
                    },
                    "stats": stats or {},
                    "seed": getattr(args, "seed", None),
                    "torch_version": torch.__version__,
                }
                meta_out = getattr(args, "polylora_metadata_out", None) or f"{ckpt_out}.meta.json"
                Path(meta_out).write_text(json.dumps(meta, indent=2))
                print(f"[polylora] wrote metadata to {meta_out}")
            return True
        except Exception as e:
            print(f"? Error training hyper-LoRA: {e}")
            return False

    def predict_sample(self, non_interactive: bool = False) -> bool:
        args = self.trainer.args
        try:
            if non_interactive:
                frame_list = getattr(args, "polylora_predict_frames", None)
                if frame_list is None:
                    print("? polylora_predict_frames must be set for non-interactive predict.")
                    return False
                if isinstance(frame_list, str):
                    frame_list = [frame_list]
            else:
                frame_list = self._prompt_list("Paths to style frames (comma-separated): ")
            if not frame_list:
                print("? No frame paths provided.")
                return False
            frames = [Image.open(Path(p)).convert("RGB") for p in frame_list]
            spec_path = (
                getattr(args, "polylora_spec", "specs/polylora_spec.json")
                if non_interactive
                else self._prompt_value(
                    f"Spec path [{getattr(args, 'polylora_spec', 'specs/polylora_spec.json')}]: ",
                    default=getattr(args, "polylora_spec", "specs/polylora_spec.json"),
                )
            )
            spec = load_spec_file(Path(spec_path))
            embed_dim = getattr(args, "polylora_embed_dim", None)
            embed_dim_input = (
                embed_dim
                if non_interactive
                else self._prompt_value(
                    f"Embedding dimension [{embed_dim or 'required'}]: ",
                    default=embed_dim,
                )
            )
            embed_dim = int(embed_dim_input) if embed_dim_input is not None else 0
            if embed_dim <= 0:
                print("? Embedding dimension must be > 0.")
                return False
            encoder_name = (
                getattr(args, "polylora_encoder", "openai/clip-vit-large-patch14-336")
                if non_interactive
                else self._prompt_value(
                    f"Encoder model name [{getattr(args, 'polylora_encoder', 'openai/clip-vit-large-patch14-336')}]: ",
                    default=getattr(args, "polylora_encoder", "openai/clip-vit-large-patch14-336"),
                )
            )
            encoder_type = (
                getattr(args, "polylora_encoder_type", "clip")
                if non_interactive
                else self._prompt_value(
                    f"Encoder type [clip|video] [{getattr(args, 'polylora_encoder_type', 'clip')}]: ",
                    default=getattr(args, "polylora_encoder_type", "clip"),
                )
            )
            use_identity = bool(getattr(args, "polylora_use_identity", False))
            identity_encoder = getattr(args, "polylora_identity_encoder", "antelopev2")
            use_base = bool(getattr(args, "polylora_dual_lora_heads", False))
            device = "cuda" if non_interactive else self._prompt_value("Device [cuda]: ", default="cuda")
            device = resolve_device(device)
            device = resolve_device(device)
            ckpt_path = (
                getattr(args, "polylora_ckpt", "checkpoints/polylora.pt")
                if non_interactive
                else self._prompt_value(
                    f"PolyLoRA checkpoint [{getattr(args, 'polylora_ckpt', 'checkpoints/polylora.pt')}]: ",
                    default=getattr(args, "polylora_ckpt", "checkpoints/polylora.pt"),
                )
            )
            out_path = (
                getattr(args, "polylora_predict_out", "predicted_lora.pt")
                if non_interactive
                else self._prompt_value(
                    "Output path for predicted LoRA (.pt) [predicted_lora.pt]: ",
                    default="predicted_lora.pt",
                )
            )

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
                embed_dim=embed_dim,
                target_specs=spec,
                head_mode=getattr(args, "polylora_head_mode", "trunk"),
                fusion_mode=getattr(args, "polylora_si_fusion_mode", "style_only"),
                identity_dim=identity_emb.shape[-1] if identity_emb is not None else None,
                use_perceiver_frontend=bool(getattr(args, "polylora_use_perceiver_frontend", False)),
                enable_base_branch=use_base,
            )
            load_polylora_checkpoint(Path(ckpt_path), model)
            model.eval()
            pred = predict_lora_state_dict(
                model,
                embedding,
                identity=identity_emb,
                use_perceiver=bool(getattr(args, "polylora_use_perceiver_frontend", False)),
                include_base=use_base,
            )
            validate_lora_matches_spec(pred, spec)
            torch.save(pred, out_path)
            merge_target = getattr(args, "polylora_merge_target", None)
            merged_path = None
            if merge_target:
                try:
                    model_sd = torch.load(merge_target, map_location="cpu")
                    missing_pred = [k for k in pred if k not in model_sd]
                    extra_base = [
                        k
                        for k in model_sd
                        if k.endswith("lora_down.weight") and k not in pred
                    ]
                    if missing_pred:
                        print(
                            f"[polylora] merge warning: {len(missing_pred)} predicted keys missing in merge target (first 3: {missing_pred[:3]})"
                        )
                    if extra_base:
                        print(
                            f"[polylora] merge warning: {len(extra_base)} merge target LoRA keys not predicted (first 3: {extra_base[:3]})"
                        )
                    if not any(k in model_sd for k in pred):
                        print("[polylora] merge skipped: no overlapping LoRA keys in merge target.")
                    else:
                        for k, v in pred.items():
                            model_sd[k] = v
                        merged_path = Path(out_path).with_name(Path(out_path).stem + "_merged.pt")
                        torch.save(model_sd, merged_path)
                        print(f"[polylora] merged predicted LoRA into {merged_path}")
                except Exception as e:
                    print(f"[polylora] merge failed: {e}")
            print(f"[polylora] wrote predicted LoRA to {out_path}")
            smoke_cmd = getattr(args, "polylora_smoke_command", None)
            if smoke_cmd:
                cmd = smoke_cmd.format(
                    lora=str(merged_path if merged_path else out_path),
                    merged=str(merged_path) if merged_path else "",
                )
                print(f"[polylora] running smoke command: {cmd}")
                try:
                    res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    print(f"[polylora] smoke command completed (stdout below):\n{res.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"[polylora] smoke command failed (exit {e.returncode}): {e.stderr}")
            return True
        except Exception as e:
            print(f"? Error predicting hyper-LoRA: {e}")
            return False

    def qa_corpus(self, non_interactive: bool = False) -> bool:
        args = self.trainer.args
        try:
            if non_interactive:
                lora_paths = getattr(args, "polylora_lora_paths", None)
                frames_root = getattr(args, "polylora_frames_root", None)
            else:
                lora_paths = self._prompt_list("LoRA checkpoints for QA (comma-separated): ")
                frames_root = self._prompt_value(
                    "Frames root for QA (optional, empty to skip frame checks): ",
                    default=getattr(args, "polylora_frames_root", None),
                )
            if not lora_paths:
                print("? No LoRA paths provided for QA.")
                return False
            if isinstance(lora_paths, str):
                lora_paths = [lora_paths]
            specs = []
            for p in lora_paths:
                sd = load_lora_state_dict(Path(p))
                specs.append(collect_lora_specs(sd))
            try:
                ensure_specs_consistent(specs)
                print(f"[polylora][qa] spec consistent across {len(lora_paths)} checkpoints")
            except Exception as e:
                print(f"[polylora][qa] spec inconsistency: {e}")
                return False
            if frames_root:
                frames_root_path = Path(frames_root)
                missing = []
                zero = []
                for lp in lora_paths:
                    frame_dir = frames_root_path / Path(lp).stem
                    if not frame_dir.exists():
                        missing.append(lp)
                        continue
                    frames = [p for p in frame_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}]
                    if not frames:
                        zero.append(lp)
                if missing:
                    print(f"[polylora][qa] missing frame dirs for {len(missing)} loras: {missing[:5]}{' ...' if len(missing)>5 else ''}")
                if zero:
                    print(f"[polylora][qa] zero frames for {len(zero)} loras: {zero[:5]}{' ...' if len(zero)>5 else ''}")
                if not missing and not zero:
                    print("[polylora][qa] frame folders present with >0 images for all LoRAs")
            return True
        except Exception as e:
            print(f"? Error during PolyLoRA QA: {e}")
            return False
