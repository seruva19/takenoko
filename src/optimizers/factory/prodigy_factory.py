"""Prodigy optimizer creation helper for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch


def create_prodigy_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    # Prodigy optimizer from prodigyopt
    try:
        from prodigyopt import Prodigy  # type: ignore
    except Exception as err:  # pragma: no cover - import-time failure
        raise ImportError(
            "Prodigy not available. Please install with `pip install prodigyopt`."
        ) from err

    # Map commonly used kwargs with sensible defaults
    # - d_coef: multiplicative factor D (logged as `d` in param_groups)
    # - decouple: decoupled weight decay
    # - betas: 2 or 3 beta values are accepted by prodigyopt
    # - use_bias_correction / safeguard_warmup: stability toggles
    d_coef = optimizer_kwargs.get("d_coef", 1.5)
    decouple = optimizer_kwargs.get("decouple", True)
    weight_decay = optimizer_kwargs.get("weight_decay", 0.1)
    betas = optimizer_kwargs.get("betas", (0.9, 0.999))
    use_bias_correction = optimizer_kwargs.get("use_bias_correction", False)
    safeguard_warmup = optimizer_kwargs.get("safeguard_warmup", False)

    # Ensure tuple for betas
    if isinstance(betas, list):
        betas = tuple(betas)

    logger.info(
        "using Prodigy optimizer | d_coef=%s, decouple=%s, weight_decay=%s, betas=%s, use_bias_correction=%s, safeguard_warmup=%s",
        d_coef,
        decouple,
        weight_decay,
        betas,
        use_bias_correction,
        safeguard_warmup,
    )

    optimizer_class = Prodigy
    optimizer = optimizer_class(
        trainable_params,
        lr=lr,
        d_coef=d_coef,
        decouple=decouple,
        weight_decay=weight_decay,
        betas=betas,  # type: ignore[arg-type]
        use_bias_correction=use_bias_correction,
        safeguard_warmup=safeguard_warmup,
    )

    return optimizer_class, optimizer
