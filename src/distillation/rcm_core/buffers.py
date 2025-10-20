"""Replay buffer utilities for RCM-style distillation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__)


def _move_nested(value: Any, device: torch.device) -> Any:
    """Recursively move tensors contained in ``value`` to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_nested(val, device) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_move_nested(v, device) for v in value]
        return type(value)(converted)  # preserve list/tuple type
    return value


def _tensor_to_cpu(value: Any) -> Any:
    """Detach tensors and move them to CPU for serialisation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_to_cpu(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_tensor_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tensor_to_cpu(v) for v in value)
    return value


@dataclass(slots=True)
class RCMReplaySample:
    """Container for a single teacher/student supervision pair."""

    observations: torch.Tensor
    actions: torch.Tensor
    teacher_logits: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: torch.device) -> "RCMReplaySample":
        """Return a copy of the sample moved to ``device``."""
        return RCMReplaySample(
            observations=self.observations.to(device, non_blocking=True),
            actions=self.actions.to(device, non_blocking=True),
            teacher_logits=self.teacher_logits.to(device, non_blocking=True),
            metadata=dict(self.metadata),
            payload=_move_nested(self.payload, device),
        )


class RCMReplayBuffer:
    """Simple ring buffer that stores :class:`RCMReplaySample` instances."""

    def __init__(self, capacity: int = 2048, *, device: torch.device | None = None) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self._capacity = capacity
        self._device = device
        self._storage: Deque[RCMReplaySample] = deque(maxlen=capacity)
        self._sealed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device | None:
        return self._device

    def insert(self, sample: RCMReplaySample) -> None:
        """Insert a new sample into the buffer."""
        if self._sealed:
            logger.warning("Attempted to insert into a sealed RCMReplayBuffer; ignoring.")
            return

        if self._device is not None:
            sample = sample.to_device(self._device)
        self._storage.append(sample)

    def extend(self, samples: Iterable[RCMReplaySample]) -> None:
        """Insert multiple samples."""
        for sample in samples:
            self.insert(sample)

    def finalize(self) -> None:
        """Prevent further writes so downstream code can count on a static snapshot."""
        self._sealed = True

    def clear(self) -> None:
        self._storage.clear()
        self._sealed = False

    def __len__(self) -> int:
        return len(self._storage)

    def iter_samples(self) -> Iterator[RCMReplaySample]:
        """Yield samples in insertion order."""
        return iter(list(self._storage))

    def iter_batches(
        self, batch_size: int, *, drop_last: bool = False
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
        """Yield mini-batches of tensors suitable for loss computation."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        observations: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        meta: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []

        for sample in self._storage:
            observations.append(sample.observations)
            actions.append(sample.actions)
            logits.append(sample.teacher_logits)
            meta.append(sample.metadata)
            payloads.append(sample.payload)

            if len(observations) == batch_size:
                yield (
                    torch.stack(observations, dim=0),
                    torch.stack(actions, dim=0),
                    torch.stack(logits, dim=0),
                    list(meta),
                    list(payloads),
                )
                observations.clear()
                actions.clear()
                logits.clear()
                meta = []
                payloads = []

        if observations and not drop_last:
            yield (
                torch.stack(observations, dim=0),
                torch.stack(actions, dim=0),
                torch.stack(logits, dim=0),
                list(meta),
                list(payloads),
            )

    def state_dict(self) -> Dict[str, Any]:
        """Serialize buffer contents for checkpointing."""
        return {
            "capacity": self._capacity,
            "sealed": self._sealed,
            "samples": [
                {
                    "observations": sample.observations.cpu(),
                    "actions": sample.actions.cpu(),
                    "teacher_logits": sample.teacher_logits.cpu(),
                    "metadata": sample.metadata,
                    "payload": _tensor_to_cpu(sample.payload),
                }
                for sample in self._storage
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore buffer contents from :meth:`state_dict` output."""
        samples = []
        for entry in state.get("samples", []):
            samples.append(
                RCMReplaySample(
                    observations=entry["observations"],
                    actions=entry["actions"],
                    teacher_logits=entry["teacher_logits"],
                    metadata=dict(entry.get("metadata", {})),
                    payload=dict(entry.get("payload", {})),
                )
            )
        self.clear()
        self.extend(samples)
        self._sealed = bool(state.get("sealed", False))
