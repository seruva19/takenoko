from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


@contextmanager
def ivon_sampled_params(args: Any, optimizer: Any) -> Iterator[None]:
    if getattr(args, "optimizer_type", "").lower() != "ivon":
        with nullcontext():
            yield
        return

    target = optimizer
    if not hasattr(target, "sampled_params") and hasattr(target, "optimizer"):
        target = target.optimizer
    if not hasattr(target, "sampled_params") and hasattr(target, "_optimizer"):
        target = target._optimizer

    if hasattr(target, "sampled_params"):
        with target.sampled_params(train=True):
            yield
    else:
        with nullcontext():
            yield
