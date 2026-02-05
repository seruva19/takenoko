from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from safetensors.torch import load_file as safetensors_load


@dataclass
class LoRATargetSpec:
    name: str
    down_shape: torch.Size
    up_shape: torch.Size

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "down_shape": list(self.down_shape),
            "up_shape": list(self.up_shape),
        }

    @classmethod
    def from_json(cls, obj: Dict[str, object]) -> "LoRATargetSpec":
        return cls(
            name=str(obj["name"]),
            down_shape=torch.Size(obj["down_shape"]),
            up_shape=torch.Size(obj["up_shape"]),
        )


def load_lora_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a LoRA checkpoint from safetensors or regular torch format."""
    path = Path(path)
    if path.suffix in {".safetensors", ".safe"}:
        return safetensors_load(str(path))
    return torch.load(path, map_location="cpu")


def collect_lora_specs(lora_sd: Dict[str, torch.Tensor]) -> List[LoRATargetSpec]:
    """Derive LoRA target shapes from a state dict (expects lora_down / lora_up)."""
    specs: List[LoRATargetSpec] = []
    for key, value in lora_sd.items():
        if not key.endswith(".lora_down.weight"):
            continue
        base = key[: -len(".lora_down.weight")]
        up_key = f"{base}.lora_up.weight"
        if up_key not in lora_sd:
            continue
        specs.append(
            LoRATargetSpec(
                name=base,
                down_shape=value.shape,
                up_shape=lora_sd[up_key].shape,
            )
        )
    return sorted(specs, key=lambda spec: spec.name)


def ensure_specs_consistent(spec_lists: List[List[LoRATargetSpec]]) -> List[LoRATargetSpec]:
    """Validate that all provided spec lists have identical names/shapes."""
    if not spec_lists:
        raise ValueError("No specs provided for consistency check.")
    base = sorted(spec_lists[0], key=lambda spec: spec.name)
    for idx, specs in enumerate(spec_lists[1:], start=1):
        specs = sorted(specs, key=lambda spec: spec.name)
        if len(specs) != len(base):
            raise ValueError(f"Spec count mismatch between spec[0] and spec[{idx}]")
        for a, b in zip(base, specs):
            if a.name != b.name or a.down_shape != b.down_shape or a.up_shape != b.up_shape:
                raise ValueError(f"Spec mismatch at position {idx}: {a} vs {b}")
    return base


def validate_lora_matches_spec(
    lora_sd: Dict[str, torch.Tensor], target_specs: Iterable[LoRATargetSpec]
) -> None:
    """Raise if state dict does not contain the expected keys/shapes."""
    missing: List[str] = []
    mismatched: List[str] = []
    for spec in target_specs:
        down_key = f"{spec.name}.lora_down.weight"
        up_key = f"{spec.name}.lora_up.weight"
        if down_key not in lora_sd or up_key not in lora_sd:
            missing.append(spec.name)
            continue
        if lora_sd[down_key].shape != spec.down_shape:
            mismatched.append(f"{down_key}: {lora_sd[down_key].shape} != {spec.down_shape}")
        if lora_sd[up_key].shape != spec.up_shape:
            mismatched.append(f"{up_key}: {lora_sd[up_key].shape} != {spec.up_shape}")
    if missing or mismatched:
        msgs = []
        if missing:
            msgs.append(f"missing {len(missing)} modules: {missing}")
        if mismatched:
            msgs.append(f"shape mismatches: {mismatched}")
        raise ValueError("LoRA spec validation failed: " + "; ".join(msgs))


def load_spec_file(path: Path) -> List[LoRATargetSpec]:
    data = json.loads(Path(path).read_text())
    return [LoRATargetSpec.from_json(item) for item in data]


def dump_spec_file(specs: List[LoRATargetSpec], path: Path) -> None:
    serialized = [spec.to_json() for spec in specs]
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serialized, indent=2))
