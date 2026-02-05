"""PolyLoRA CLI entrypoint (collect-spec, build-pairs, train, predict, qa)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from polylora.dataset import PolyLoRAPairDataset, PairSample, save_sharded_samples
from polylora.encoder import (
    encode_identity_frames,
    encode_style_frames,
    encode_style_frames_multi,
)
from polylora.model import PolyLoRANetwork
from polylora.predict import load_polylora_checkpoint, predict_lora_state_dict
from polylora.spec import (
    LoRATargetSpec,
    collect_lora_specs,
    dump_spec_file,
    ensure_specs_consistent,
    load_lora_state_dict,
    load_spec_file,
    validate_lora_matches_spec,
)
from polylora.train import TrainConfig, train_polylora


def _iter_lora_paths(items: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(
                sorted([f for f in p.iterdir() if p.suffix in {".pt", ".safetensors"}])
            )
        else:
            paths.append(p)
    return paths


def _find_images(frame_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in exts])


def _build_encoder_list(
    encoders: Optional[List[str]], encoder_type: str
) -> Optional[List[Tuple[str, str]]]:
    if not encoders:
        return None
    return [(name, encoder_type) for name in encoders]


def cmd_collect_spec(args: argparse.Namespace) -> None:
    first_spec = None
    expected_names: List[str] = []
    for path in args.lora:
        sd = load_lora_state_dict(Path(path))
        specs = collect_lora_specs(sd)
        if first_spec is None:
            first_spec = specs
            expected_names = [spec.name for spec in specs]
        else:
            names = [spec.name for spec in specs]
            if names != expected_names:
                raise ValueError(
                    f"Spec mismatch for {path}: {names} != {expected_names}"
                )
    dump_spec_file(first_spec or [], Path(args.out))
    print(f"[polylora] wrote spec with {len(first_spec or [])} targets to {args.out}")


def _derive_base_lora(lora_sd: dict[str, torch.Tensor], attenuate: float = 0.5) -> dict[str, torch.Tensor]:
    """Heuristic base LoRA: scale down weights to reduce ID dominance."""
    base_sd: dict[str, torch.Tensor] = {}
    for k, v in lora_sd.items():
        if k.endswith(("lora_down.weight", "lora_up.weight")):
            base_sd[k] = v * attenuate
        else:
            base_sd[k] = v
    return base_sd


def cmd_build_pairs(args: argparse.Namespace) -> None:
    specs_list = []
    samples: List[PairSample] = []
    device = args.device
    encoder_fusion = _build_encoder_list(args.encoders, args.encoder_type)
    base_lora_paths = _iter_lora_paths(args.base_lora) if args.base_lora else []
    base_map = {p.stem: p for p in base_lora_paths}
    auto_base = args.dual_heads and not base_map
    for lora_path in _iter_lora_paths(args.lora):
        lora_sd = load_lora_state_dict(lora_path)
        specs_list.append(collect_lora_specs(lora_sd))
        frame_dir = Path(args.frames) / lora_path.stem
        frame_paths = _find_images(frame_dir)
        if not frame_paths:
            print(f"[polylora] skipping {lora_path} (no images in {frame_dir})")
            continue
        frames = [Image.open(fp).convert("RGB") for fp in frame_paths]
        identity_emb = None
        if encoder_fusion:
            embedding = encode_style_frames_multi(
                frames,
                encoders=encoder_fusion,
                device=device,
                fusion_mode=args.fusion_mode,
            )
        else:
            embedding = encode_style_frames(
                frames,
                model_name=args.encoder,
                encoder_type=args.encoder_type,
                device=device,
            )
        base_lora = None
        identity_emb = None
        if args.use_identity:
            try:
                identity_emb = encode_identity_frames(
                    frames,
                    model_name=args.identity_encoder,
                    device=args.identity_device or args.device,
                )
            except Exception as exc:
                print(f"[polylora] identity encoding failed for {lora_path}: {exc}")
        if args.dual_heads:
            base_path = base_map.get(lora_path.stem)
            if base_path is None and auto_base:
                print(f"[polylora] dual-heads: auto-deriving base LoRA for {lora_path.stem} with attenuate={args.base_attenuate}")
                base_lora = _derive_base_lora(lora_sd, attenuate=args.base_attenuate)
            elif base_path is None:
                raise ValueError(f"--dual-heads set but no base LoRA provided for stem {lora_path.stem}")
            else:
                base_lora = load_lora_state_dict(base_path)
                validate_lora_matches_spec(base_lora, collect_lora_specs(base_lora))
        samples.append(
            PairSample(embedding=embedding, lora=lora_sd, identity=identity_emb, base_lora=base_lora)
        )
    spec = ensure_specs_consistent(specs_list)
    validate_lora_matches_spec(samples[0].lora, spec)
    save_sharded_samples(samples, Path(args.out), shard_size=args.shard_size)
    dump_spec_file(spec, Path(args.out) / "spec.json")
    print(f"[polylora] wrote {len(samples)} samples to {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    spec: List[LoRATargetSpec] = load_spec_file(Path(args.spec))
    ds = PolyLoRAPairDataset(args.data, expected_spec=spec)
    if len(ds) == 0:
        raise ValueError("No samples found in shard files.")
    first_emb, _, first_id, _ = ds[0]
    embed_dim = args.embed_dim or first_emb.shape[-1]
    identity_dim = first_id.shape[-1] if first_id is not None else None
    if (
        args.si_fusion_mode != "style_only"
        and identity_dim is None
        and args.use_identity
    ):
        raise ValueError(
            "Identity fusion requested but no identity embeddings found in shards."
        )
    model = PolyLoRANetwork(
        embed_dim=embed_dim,
        target_specs=spec,
        head_mode=args.head_mode,
        fusion_mode=args.si_fusion_mode,
        identity_dim=identity_dim,
        use_perceiver_frontend=args.use_perceiver_frontend,
        enable_base_branch=args.dual_heads,
    )
    stats = train_polylora(
        model,
        ds,
        TrainConfig(
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            val_split=args.val_split,
            amp=args.amp,
            grad_clip=args.grad_clip,
            cosine_loss_weight=args.cosine_loss_weight,
            sample_every_epochs=args.sample_every_epochs,
            sample_command=args.sample_command,
            sample_dir=args.sample_dir,
            sample_merge_target=args.sample_merge_target,
            use_identity=args.use_identity,
            use_perceiver_frontend=args.use_perceiver_frontend,
            use_base_branch=args.dual_heads,
            base_loss_weight=args.base_loss_weight,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
        ),
        device=args.device,
    )
    torch.save(model.state_dict(), args.save)
    print(f"[polylora] saved checkpoint to {args.save}")
    if stats:
        print(f"[polylora] stats: {stats}")
    if args.save_metadata:
        import hashlib
        import json
        from datetime import datetime, timezone

        try:
            spec_bytes = Path(args.spec).read_bytes()
            spec_hash = hashlib.sha256(spec_bytes).hexdigest()
        except Exception:
            spec_hash = "unknown"
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ckpt_path": str(args.save),
            "spec_path": str(args.spec),
            "spec_sha256": spec_hash,
            "num_samples": len(ds),
            "embed_dim": embed_dim,
            "identity_dim": identity_dim,
            "encoder": args.encoder if hasattr(args, "encoder") else None,
            "encoder_type": args.encoder_type if hasattr(args, "encoder_type") else None,
            "fusion_mode": args.si_fusion_mode,
            "head_mode": args.head_mode,
            "use_identity": args.use_identity,
            "use_perceiver_frontend": args.use_perceiver_frontend,
            "use_base_branch": args.dual_heads,
            "train_config": {
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "val_split": args.val_split,
                "amp": args.amp,
                "grad_clip": args.grad_clip,
                "cosine_loss_weight": args.cosine_loss_weight,
                "base_loss_weight": args.base_loss_weight,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
            },
            "stats": stats or {},
            "torch_version": torch.__version__,
        }
        meta_out = args.metadata_out or f"{args.save}.meta.json"
        Path(meta_out).write_text(json.dumps(meta, indent=2))
        print(f"[polylora] wrote metadata to {meta_out}")


def cmd_predict(args: argparse.Namespace) -> None:
    frames = [Image.open(Path(p)).convert("RGB") for p in args.frames]
    embedding = encode_style_frames(
        frames,
        model_name=args.encoder,
        encoder_type=args.encoder_type,
        device=args.device,
    )
    identity_emb = None
    if args.use_identity:
        identity_emb = encode_identity_frames(
            frames, model_name=args.identity_encoder, device=args.device
        )
    model = PolyLoRANetwork(
        embed_dim=args.embed_dim,
        target_specs=load_spec_file(Path(args.spec)),
        fusion_mode=args.si_fusion_mode,
        identity_dim=identity_emb.shape[-1] if identity_emb is not None else None,
        use_perceiver_frontend=args.use_perceiver_frontend,
        enable_base_branch=args.dual_heads,
    )
    load_polylora_checkpoint(Path(args.ckpt), model)
    pred = predict_lora_state_dict(
        model,
        embedding,
        identity=identity_emb,
        use_perceiver=args.use_perceiver_frontend,
        include_base=args.dual_heads,
    )
    out_path = Path(args.out)
    torch.save(pred, out_path)
    print(f"[polylora] wrote predicted LoRA to {out_path}")
    if args.merge:
        merge_path = Path(args.merge)
        try:
            sd = torch.load(merge_path, map_location="cpu")
            missing = [k for k in pred if k not in sd]
            extra = [k for k in sd if "lora_down" in k and k not in pred]
            if missing:
                print(
                    f"[polylora] merge check: {len(missing)} predicted keys missing in target (first 3: {missing[:3]})"
                )
            if extra:
                print(
                    f"[polylora] merge check: {len(extra)} target LoRA keys not predicted (first 3: {extra[:3]})"
                )
            sd.update(pred)
            merged_out = out_path.with_name(out_path.stem + "_merged.pt")
            torch.save(sd, merged_out)
            print(f"[polylora] merged predicted LoRA into {merged_out}")
        except Exception as e:
            print(f"[polylora] merge failed: {e}")
    if args.smoke:
        cmd = args.smoke.format(lora=str(out_path), merged="")
        print(f"[polylora] running smoke command: {cmd}")
        try:
            res = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
            if res.stdout:
                print(f"[polylora] smoke stdout:\n{res.stdout}")
            if res.stderr:
                print(f"[polylora] smoke stderr:\n{res.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"[polylora] smoke command failed (exit {e.returncode}): {e.stderr}")


def cmd_qa(args: argparse.Namespace) -> None:
    paths = _iter_lora_paths(args.lora)
    specs = [collect_lora_specs(load_lora_state_dict(p)) for p in paths]
    ensure_specs_consistent(specs)
    print(
        f"[polylora][qa] spec consistent across {len(paths)} checkpoints; targets={len(specs[0])}"
    )
    frames_root = Path(args.frames) if args.frames else None
    if frames_root:
        missing, zero = [], []
        for p in paths:
            frame_dir = frames_root / p.stem
            imgs = _find_images(frame_dir)
            if not frame_dir.exists():
                missing.append(str(frame_dir))
            elif not imgs:
                zero.append(str(frame_dir))
        if missing:
            print(
                f"[polylora][qa] missing frame dirs for {len(missing)} loras: {missing[:5]}{' ...' if len(missing)>5 else ''}"
            )
        if zero:
            print(
                f"[polylora][qa] zero frames for {len(zero)} loras: {zero[:5]}{' ...' if len(zero)>5 else ''}"
            )
        if not (missing or zero):
            print("[polylora][qa] frame folders present with >0 images for all LoRAs")
    print("[polylora][qa] done")


def cmd_sample_wan(args: argparse.Namespace) -> None:
    from pathlib import Path as _Path

    import accelerate
    import importlib
    import toml
    import torch

    from config_parser import create_args_from_config
    from core.model_manager import ModelManager
    from core.sampling_manager import SamplingManager
    from utils.model_utils import str_to_dtype

    config_path = _Path(args.config)
    lora_path = _Path(args.lora)

    raw_config = toml.loads(config_path.read_text())
    cfg_text = config_path.read_text()
    wan_args = create_args_from_config(raw_config, str(config_path), cfg_text)

    accelerator = accelerate.Accelerator()
    wan_args.device = str(accelerator.device)

    model_manager = ModelManager()

    vae_dtype = (
        torch.float16
        if wan_args.vae_dtype is None
        else str_to_dtype(wan_args.vae_dtype)
    )
    vae = model_manager.load_vae(wan_args, vae_dtype=vae_dtype, vae_path=wan_args.vae)

    attn_mode = model_manager.get_attention_mode(wan_args)
    dit_weight_dtype = model_manager.detect_wan_sd_dtype(wan_args.dit)
    transformer, dual_model_manager = model_manager.load_transformer(
        accelerator=accelerator,
        args=wan_args,
        dit_path=wan_args.dit,
        attn_mode=attn_mode,
        split_attn=wan_args.split_attn,
        loading_device=accelerator.device,
        dit_weight_dtype=dit_weight_dtype,
        config=raw_config,
    )
    transformer.eval()
    transformer.requires_grad_(False)

    network_module = importlib.import_module(wan_args.network_module)
    weights_sd = torch.load(lora_path, map_location="cpu")
    module = network_module.create_arch_network_from_weights(
        1.0, weights_sd, unet=transformer, for_inference=True
    )
    module.merge_to(None, transformer, weights_sd, torch.float32, "cpu")

    sampling_manager = SamplingManager(raw_config)
    sampling_manager.set_vae_config(
        {"args": wan_args, "vae_dtype": vae_dtype, "vae_path": wan_args.vae}
    )

    sample_params = getattr(wan_args, "sample_prompts", None)
    if not sample_params:
        raise SystemExit("No sample_prompts found in config; cannot run sampling.")

    sampling_manager.sample_images(
        accelerator=accelerator,
        args=wan_args,
        epoch=None,
        steps=0,
        vae=vae,
        transformer=transformer,
        sample_parameters=sample_params,
        dit_dtype=transformer.dtype if hasattr(transformer, "dtype") else None,
        dual_model_manager=dual_model_manager,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyLoRA CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser(
        "collect-spec", help="Collect LoRA target spec from checkpoints"
    )
    collect.add_argument("--lora", nargs="+", required=True)
    collect.add_argument("--out", required=True)
    collect.set_defaults(func=cmd_collect_spec)

    pairs = sub.add_parser("build-pairs", help="Build (embedding, lora) pairs")
    pairs.add_argument("--lora", nargs="+", required=True)
    pairs.add_argument("--frames", required=True)
    pairs.add_argument("--encoder", default="openai/clip-vit-large-patch14-336")
    pairs.add_argument("--encoder-type", default="clip", choices=["clip", "video"])
    pairs.add_argument(
        "--encoders",
        nargs="*",
        default=None,
        help="Optional list of encoder model names for fusion",
    )
    pairs.add_argument("--fusion-mode", default="mean", choices=["mean", "concat"])
    pairs.add_argument(
        "--use-identity",
        action="store_true",
        help="Encode identity embeddings (insightface).",
    )
    pairs.add_argument("--identity-encoder", default="antelopev2")
    pairs.add_argument(
        "--identity-device",
        default=None,
        help="Identity encoder device (defaults to --device).",
    )
    pairs.add_argument("--dual-heads", action="store_true", help="Attach base LoRA branch; requires --base-lora.")
    pairs.add_argument("--base-lora", nargs="*", default=None, help="Base LoRA paths aligned by stem to --lora.")
    pairs.add_argument(
        "--base-attenuate",
        type=float,
        default=0.5,
        help="If dual-heads and no base-lora provided, scale main LoRA by this factor to derive base.",
    )
    pairs.add_argument("--out", required=True)
    pairs.add_argument("--shard-size", type=int, default=32)
    pairs.add_argument("--device", default="cuda")
    pairs.set_defaults(func=cmd_build_pairs)

    train = sub.add_parser("train", help="Train PolyLoRA")
    train.add_argument(
        "--data", nargs="+", required=True, help="Pair shards (.pt) or directories"
    )
    train.add_argument("--spec", required=True)
    train.add_argument("--embed-dim", type=int, default=None)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--val-split", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable AMP during training.",
    )
    train.set_defaults(amp=True)
    train.add_argument("--cosine-loss-weight", type=float, default=0.0)
    train.add_argument("--base-loss-weight", type=float, default=1.0)
    train.add_argument("--use-ema", action="store_true")
    train.add_argument("--ema-decay", type=float, default=0.995)
    train.add_argument("--head-mode", default="trunk", choices=["trunk", "per_tensor"])
    train.add_argument(
        "--dual-heads",
        action="store_true",
        help="Emit base branch (keys prefixed with base.)",
    )
    train.add_argument(
        "--si-fusion-mode",
        default="style_only",
        choices=["style_only", "mean", "gated", "concat"],
        help="Fusion for style + identity embeddings.",
    )
    train.add_argument(
        "--use-identity",
        action="store_true",
        help="Use identity embeddings if present in shards.",
    )
    train.add_argument(
        "--use-perceiver-front-end",
        dest="use_perceiver_frontend",
        action="store_true",
        help="Enable Perceiver mixer.",
    )
    train.add_argument("--sample-every-epochs", type=int, default=0)
    train.add_argument("--sample-command", default=None)
    train.add_argument("--sample-dir", default="polylora_samples")
    train.add_argument("--sample-merge-target", default=None)
    train.add_argument("--device", default="cuda")
    train.add_argument("--save", required=True)
    train.add_argument("--save-metadata", action="store_true")
    train.add_argument("--metadata-out", default=None)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Predict LoRA weights from frames")
    predict.add_argument("--ckpt", required=True)
    predict.add_argument("--spec", required=True)
    predict.add_argument("--encoder", default="openai/clip-vit-large-patch14-336")
    predict.add_argument("--encoder-type", default="clip", choices=["clip", "video"])
    predict.add_argument("--embed-dim", type=int, required=True)
    predict.add_argument("--frames", nargs="+", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--merge", default=None)
    predict.add_argument("--smoke", default=None)
    predict.add_argument(
        "--use-identity",
        action="store_true",
        help="Encode identity embeddings for conditioning.",
    )
    predict.add_argument("--identity-encoder", default="antelopev2")
    predict.add_argument(
        "--si-fusion-mode",
        default="style_only",
        choices=["style_only", "mean", "gated", "concat"],
    )
    predict.add_argument(
        "--use-perceiver-front-end", dest="use_perceiver_frontend", action="store_true"
    )
    predict.add_argument(
        "--dual-heads",
        action="store_true",
        help="Enable base branch (returned keys prefixed with base.)",
    )
    predict.add_argument("--device", default="cuda")
    predict.set_defaults(func=cmd_predict)

    qa = sub.add_parser("qa", help="Quick spec/frame QA")
    qa.add_argument("--lora", nargs="+", required=True)
    qa.add_argument("--frames", default=None)
    qa.set_defaults(func=cmd_qa)

    sample_wan = sub.add_parser(
        "sample-wan",
        help="Sample using a predicted LoRA and an existing WAN config (merges LoRA then runs SamplingManager)",
    )
    sample_wan.add_argument(
        "--config",
        required=True,
        help="Path to WAN TOML config with sampling settings.",
    )
    sample_wan.add_argument(
        "--lora", required=True, help="Predicted LoRA checkpoint (.pt/.safetensors)."
    )
    sample_wan.set_defaults(func=cmd_sample_wan)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
