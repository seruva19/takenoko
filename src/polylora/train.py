from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from .model import PolyLoRANetwork, lora_loss, predict_lora_state_dict
from .spec import LoRATargetSpec


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 4
    weight_decay: float = 0.0
    val_split: float = 0.1
    cosine_loss_weight: float = 0.0
    amp: bool = True
    grad_clip: Optional[float] = 1.0
    sample_every_epochs: int = 0
    sample_command: Optional[str] = None
    sample_dir: str = "polylora_samples"
    sample_merge_target: Optional[str] = None
    use_identity: bool = False
    use_perceiver_frontend: bool = False
    use_base_branch: bool = False
    base_loss_weight: float = 1.0


def _split_dataset(
    dataset, val_split: float, seed: int = 42
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if val_split <= 0 or val_split >= 1:
        return dataset, None
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    return random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))


def train_polylora(
    model: PolyLoRANetwork,
    dataset,
    config: TrainConfig,
    device: str = "cuda",
    num_workers: int = 0,
) -> Dict[str, float]:
    def _collate(batch):
        embs, loras, ids, base_loras = zip(*batch)
        emb_tensor = torch.stack(embs, dim=0)
        lora_keys = loras[0].keys()
        lora_tensor = {k: torch.stack([l[k] for l in loras], dim=0) for k in lora_keys}
        id_tensor = None
        if any(id_val is not None for id_val in ids):
            ref_id = next(id_val for id_val in ids if id_val is not None)
            id_tensor = torch.stack(
                [id_val if id_val is not None else torch.zeros_like(ref_id) for id_val in ids],
                dim=0,
            )
        base_lora_tensor = None
        if any(bl is not None for bl in base_loras):
            ref = next(bl for bl in base_loras if bl is not None)
            ref_keys = ref.keys()
            base_lora_tensor = {
                k: torch.stack(
                    [
                        bl[k] if bl is not None else torch.zeros_like(ref[k])
                        for bl in base_loras
                    ],
                    dim=0,
                )
                for k in ref_keys
            }
        return emb_tensor, lora_tensor, id_tensor, base_lora_tensor

    def _run_sampling_hook(epoch_idx: int) -> None:
        if config.sample_every_epochs <= 0:
            return
        if (epoch_idx + 1) % config.sample_every_epochs != 0:
            return
        try:
            if len(dataset) == 0:
                return
            sample_idx = epoch_idx % len(dataset)
            emb, _, id_emb, _ = dataset[sample_idx]  # type: ignore[index]
            embedding = emb.unsqueeze(0).to(device)
            if id_emb is not None and config.use_identity:
                id_emb = id_emb.unsqueeze(0).to(device)
            model.eval()
            pred = predict_lora_state_dict(
                model,
                embedding,
                identity=id_emb if (id_emb is not None and config.use_identity) else None,
                include_base=False,
            )
            out_dir = Path(config.sample_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / f"predicted_epoch{epoch_idx + 1:03d}.pt"
            torch.save(pred, pred_path)
            merged_path = None
            if config.sample_merge_target:
                try:
                    base_sd = torch.load(config.sample_merge_target, map_location="cpu")
                    missing_pred = [k for k in pred if k not in base_sd]
                    extra_base = [k for k in base_sd if k.endswith("lora_down.weight") and k not in pred]
                    if missing_pred:
                        print(f"[polylora][sample] warning: {len(missing_pred)} predicted keys missing in merge target (first 3: {missing_pred[:3]})")
                    if extra_base:
                        print(f"[polylora][sample] warning: {len(extra_base)} merge target LoRA keys not predicted (first 3: {extra_base[:3]})")
                    base_sd.update(pred)
                    merged_path = pred_path.with_name(pred_path.stem + "_merged.pt")
                    torch.save(base_sd, merged_path)
                except Exception as merge_err:
                    print(f"[polylora][sample] merge failed: {merge_err}")
            if config.sample_command:
                cmd = config.sample_command.format(
                    lora=str(merged_path if merged_path else pred_path),
                    merged=str(merged_path) if merged_path else "",
                    epoch=epoch_idx + 1,
                )
                print(f"[polylora][sample] running: {cmd}")
                try:
                    res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    if res.stdout:
                        print(f"[polylora][sample] stdout:\n{res.stdout}")
                    if res.stderr:
                        print(f"[polylora][sample] stderr:\n{res.stderr}")
                except subprocess.CalledProcessError as e:
                    print(f"[polylora][sample] command failed (exit {e.returncode}): {e.stderr}")
            else:
                summary_path = pred_path.with_suffix(".summary.txt")
                try:
                    merged_note = f"\nmerged_path: {merged_path}" if merged_path else ""
                    summary = (
                        f"epoch: {epoch_idx + 1}\n"
                        f"pred_path: {pred_path}\n"
                        f"num_keys: {len(pred)}{merged_note}\n"
                    )
                    summary_path.write_text(summary)
                    print(f"[polylora][sample] wrote summary to {summary_path}")
                except Exception as write_err:
                    print(f"[polylora][sample] summary write failed: {write_err}")
        except Exception as e:
            print(f"[polylora][sample] skipped due to error: {e}")

    model.to(device)
    train_ds, val_ds = _split_dataset(dataset, config.val_split)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate
        )

    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.amp)
    best_val = None

    for epoch in range(config.epochs):
        model.train()
        for embedding, lora, id_emb, base_lora in train_loader:
            embedding = embedding.to(device)
            lora = {k: v.to(device) for k, v in lora.items()}
            id_emb = id_emb.to(device) if id_emb is not None else None
            with autocast(enabled=config.amp):
                pred = model(
                    embedding,
                    identity=id_emb if config.use_identity else None,
                    use_perceiver=config.use_perceiver_frontend,
                )
                loss = lora_loss(
                    pred,
                    lora,
                    base_target=base_lora if config.use_base_branch and base_lora is not None else None,
                    cosine_weight=config.cosine_loss_weight,
                    base_weight=config.base_loss_weight,
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            if config.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()

        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for embedding, lora, id_emb, base_lora in val_loader:
                    embedding = embedding.to(device)
                    lora = {k: v.to(device) for k, v in lora.items()}
                    id_emb = id_emb.to(device) if id_emb is not None else None
                    pred = model(
                        embedding,
                        identity=id_emb if config.use_identity else None,
                        use_perceiver=config.use_perceiver_frontend,
                    )
                    val_loss = lora_loss(
                        pred,
                        lora,
                        base_target=base_lora if config.use_base_branch and base_lora is not None else None,
                        cosine_weight=config.cosine_loss_weight,
                        base_weight=config.base_loss_weight,
                    )
                    val_losses.append(val_loss.item())
            best_val = min(val_losses) if val_losses else None
        _run_sampling_hook(epoch)
    stats = {"best_val_loss": best_val} if best_val is not None else {}
    return stats
