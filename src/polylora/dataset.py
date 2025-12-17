from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .spec import LoRATargetSpec, validate_lora_matches_spec


@dataclass
class PairSample:
    embedding: torch.Tensor
    lora: Dict[str, torch.Tensor]
    identity: Optional[torch.Tensor] = None
    base_lora: Optional[Dict[str, torch.Tensor]] = None


class PolyLoRAPairDataset(Dataset):
    """Dataset of precomputed (embedding, lora_state_dict) pairs stored as .pt shards."""

    def __init__(self, items: Iterable[Path], expected_spec: Optional[Sequence[LoRATargetSpec]] = None):
        super().__init__()
        self.items: List[Tuple[Path, Optional[int]]] = []
        for p in items:
            path = Path(p)
            rec = torch.load(path, map_location="cpu")
            if isinstance(rec, dict) and "samples" in rec and isinstance(rec["samples"], list):
                for idx in range(len(rec["samples"])):
                    self.items.append((path, idx))
            else:
                self.items.append((path, None))
        self.expected_spec = list(expected_spec) if expected_spec else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        path, sample_idx = self.items[idx]
        record = torch.load(path, map_location="cpu")
        if sample_idx is not None and "samples" in record:
            sample = record["samples"][sample_idx]
        else:
            sample = record
        embedding = sample["embedding"].float()
        lora = {k: v.float() for k, v in sample["lora"].items()}
        identity = sample.get("identity")
        if identity is not None:
            identity = identity.float()
        base_lora_raw = sample.get("base_lora")
        base_lora = None
        if base_lora_raw:
            base_lora = {k: v.float() for k, v in base_lora_raw.items()}
        if self.expected_spec:
            validate_lora_matches_spec(lora, self.expected_spec)
        return embedding, lora, identity, base_lora


def save_sharded_samples(
    samples: List[PairSample],
    output_dir: Path,
    shard_size: int = 32,
) -> List[Path]:
    """Persist samples into shard files (up to shard_size per file); returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    shard: List[Dict[str, Dict[str, torch.Tensor]]] = []
    shard_idx = 0
    for sample in samples:
        record: Dict[str, Dict[str, torch.Tensor] | torch.Tensor] = {
            "embedding": sample.embedding.cpu(),
            "lora": {k: v.cpu() for k, v in sample.lora.items()},
        }
        if sample.identity is not None:
            record["identity"] = sample.identity.cpu()
        if sample.base_lora is not None:
            record["base_lora"] = {k: v.cpu() for k, v in sample.base_lora.items()}
        shard.append(record)
        if len(shard) >= shard_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save({"samples": shard}, shard_path)
            paths.append(shard_path)
            shard_idx += 1
            shard = []
    if shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({"samples": shard}, shard_path)
        paths.append(shard_path)
    manifest = {
        "num_samples": len(samples),
        "shard_size": shard_size,
        "paths": [str(p.name) for p in paths],
        "fields": ["embedding", "lora"] + (["identity"] if any(s.identity is not None for s in samples) else []) + (["base_lora"] if any(s.base_lora is not None for s in samples) else []),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths
