from __future__ import annotations

import torch


def resolve_device(device: str | None) -> str:
    if not device:
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device
