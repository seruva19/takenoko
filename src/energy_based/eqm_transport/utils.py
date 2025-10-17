from __future__ import annotations

from typing import Any, Dict, Iterable

import torch


class EasyDict(dict):
    """Dictionary with attribute-style access used by the EqM utilities."""

    def __init__(self, values: Dict[str, Any]) -> None:
        super().__init__(values)
        for key, value in values.items():
            setattr(self, key, value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - mirrored dict behaviour
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        object.__setattr__(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    def update(self, values: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        if values is None:
            values = {}
        merged = dict(values)
        merged.update(kwargs)
        for key, value in merged.items():
            self.__setattr__(key, value)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    if tensor.ndim <= 1:
        return tensor.mean()
    reduce_dims = tuple(range(1, tensor.ndim))
    return torch.mean(tensor, dim=reduce_dims)


def log_state(state: Dict[str, Any]) -> str:
    """Pretty-print helper used by the original EqM logging."""
    lines = []
    for key in sorted(state.keys()):
        value = state[key]
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(map(str, value))}]")
        elif hasattr(value, "__class__") and "object at" in str(value):
            lines.append(f"{key}: [{value.__class__.__name__}]")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
