from typing import Optional, Union

import torch


def _normalize_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
    """Normalize device inputs to a torch.device instance.

    Accepts ``None`` and string identifiers so that callers can pass their
    accelerator device directly without manual conversion.
    """

    if device is None:
        return None

    if isinstance(device, str):
        return torch.device(device)

    return device


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    """Release allocator caches for the specified device if applicable."""

    normalized = _normalize_device(device)
    if normalized is None:
        return

    if normalized.type == "cuda":
        torch.cuda.empty_cache()
    elif normalized.type == "cpu":
        # CPU cache clearing is a no-op intentionally
        return
    elif normalized.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif normalized.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    """Synchronize the specified compute device if synchronization is supported."""

    normalized = _normalize_device(device)
    if normalized is None:
        return

    if normalized.type == "cuda":
        torch.cuda.synchronize()
    elif normalized.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif normalized.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
