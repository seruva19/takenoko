"""Lazy torchao-backed low-precision AdamW wrappers."""

from __future__ import annotations

import importlib
from typing import Any, Type


_TORCHAO_INSTALL_HINT = (
    "torchao is required for this optimizer. Install the repository's pinned "
    "torchao version or an equivalent build that exposes low-precision AdamW "
    "optimizers."
)


def _load_torchao_optimizer_class(class_name: str) -> Type[Any]:
    """Return ``torchao.optim.<class_name>`` with a clear optional-dep error."""

    try:
        torchao_optim = importlib.import_module("torchao.optim")
    except ModuleNotFoundError as err:
        raise ImportError(_TORCHAO_INSTALL_HINT) from err

    optimizer_class = getattr(torchao_optim, class_name, None)
    if optimizer_class is None:
        raise ImportError(
            f"torchao.optim.{class_name} is unavailable in the installed torchao "
            "package. Upgrade torchao or use a build that exposes this optimizer."
        )
    return optimizer_class


class TorchAOAdamW8bit:
    """Alias for ``torchao.optim.AdamW8bit``."""

    def __new__(
        cls,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ):
        optimizer_class = _load_torchao_optimizer_class("AdamW8bit")
        return optimizer_class(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )


class TorchAOAdamW4bit:
    """Alias for ``torchao.optim.AdamW4bit``."""

    def __new__(
        cls,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=128,
        bf16_stochastic_round=False,
    ):
        optimizer_class = _load_torchao_optimizer_class("AdamW4bit")
        return optimizer_class(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )


class TorchAOAdamWFp8:
    """Alias for ``torchao.optim.AdamWFp8``."""

    def __new__(
        cls,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ):
        optimizer_class = _load_torchao_optimizer_class("AdamWFp8")
        return optimizer_class(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )


__all__ = [
    "TorchAOAdamW8bit",
    "TorchAOAdamW4bit",
    "TorchAOAdamWFp8",
]
