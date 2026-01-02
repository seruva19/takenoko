"""Custom optimizer creation helpers for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch


def create_ivon_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using IVON optimizer | {optimizer_kwargs}")

    from vendor.ivon.ivon import IVON

    # Allow users to pass standard Adam-style betas=(b1,b2)
    if "betas" in optimizer_kwargs and "beta1" not in optimizer_kwargs and "beta2" not in optimizer_kwargs:
        betas_value = optimizer_kwargs.pop("betas")
        if isinstance(betas_value, list):
            betas_value = tuple(betas_value)
        if not isinstance(betas_value, (tuple, list)) or len(betas_value) != 2:
            raise ValueError(
                "IVON betas must be a length-2 sequence when provided as betas=(beta1,beta2). "
                f"Received: {betas_value}"
            )
        optimizer_kwargs["beta1"], optimizer_kwargs["beta2"] = betas_value

    ess = optimizer_kwargs.pop("ess", None)
    if ess is None:
        ess = getattr(args, "ivon_ess", None)
    if ess is None:
        raise ValueError(
            (
                "IVON requires an effective sample size 'ess'. Provide ivon_ess in the TOML "
                'or optimizer_args=["ess=..."].'
            )
        )

    optimizer_class = IVON
    optimizer = optimizer_class(
        trainable_params,
        lr=lr,
        ess=float(ess),
        **optimizer_kwargs,
    )
    return optimizer_class, optimizer


def create_automagic_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using Automagic optimizer | {optimizer_kwargs}")

    from optimizers.automagic import Automagic

    optimizer_class = Automagic
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_adamw_optimi_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using optimi.AdamW optimizer | {optimizer_kwargs}")

    from optimi import AdamW

    optimizer_class = AdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_lion_optimi_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using optimi.Lion optimizer | {optimizer_kwargs}")

    from optimi import Lion

    optimizer_class = Lion
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_sophiag_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using SophiaG optimizer | {optimizer_kwargs}")

    from optimizers.sophia import SophiaG

    optimizer_class = SophiaG
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_soap_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using Soap optimizer | {optimizer_kwargs}")

    from optimizers.soap import SOAP

    optimizer_class = SOAP
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_temporal_adamw_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using TemporalAdamW optimizer | {optimizer_kwargs}")

    # Import our custom optimizer
    from optimizers.temporal_adamw import TemporalAdamW

    optimizer_class = TemporalAdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_raven_adamw_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using RavenAdamW optimizer | {optimizer_kwargs}")

    from optimizers.raven import RavenAdamW

    optimizer_class = RavenAdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer
