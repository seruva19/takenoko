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


def create_sinksgd_adv_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using SinkSGDAdv optimizer | {optimizer_kwargs}")

    try:
        from adv_optm.optim import SinkSGD_adv
    except ImportError as err:
        raise ImportError(
            "SinkSGDAdv requires adv-optm==2.4.dev17 or newer with "
            "adv_optm.optim.SinkSGD_adv available."
        ) from err

    sinksgd_kwargs = dict(optimizer_kwargs)
    legacy_keys = sorted(
        key for key in sinksgd_kwargs if isinstance(key, str) and key.startswith("sinksgd_adv_")
    )
    if legacy_keys:
        raise ValueError(
            "SinkSGDAdv optimizer_args must use upstream argument names, "
            f"not prefixed aliases: {legacy_keys}"
        )

    sinksgd_lr = float(sinksgd_kwargs.pop("lr", 1.0e-2))
    if sinksgd_lr <= 0.0:
        raise ValueError("SinkSGDAdv optimizer_args lr must be > 0")

    reference_batch_size = int(sinksgd_kwargs.pop("reference_batch_size", 4))
    if reference_batch_size < 1:
        raise ValueError("SinkSGDAdv optimizer_args reference_batch_size must be >= 1")
    batch_scale_mode = str(sinksgd_kwargs.pop("batch_scale_mode", "sqrt")).lower()
    if batch_scale_mode not in {"off", "sqrt", "linear"}:
        raise ValueError(
            "SinkSGDAdv optimizer_args batch_scale_mode must be one of: off, sqrt, linear"
        )

    effective_batch_size = getattr(args, "effective_batch_size", None)
    if effective_batch_size is None:
        effective_batch_size = reference_batch_size
    effective_batch_size = max(float(effective_batch_size), 1.0)
    batch_ratio = effective_batch_size / float(reference_batch_size)
    if batch_scale_mode == "linear":
        lr_scale = batch_ratio
    elif batch_scale_mode == "sqrt":
        lr_scale = batch_ratio**0.5
    else:
        lr_scale = 1.0
    effective_lr = sinksgd_lr * lr_scale

    sinksgd_kwargs.setdefault("orthogonal_sinkhorn", True)
    sinksgd_kwargs.setdefault("spectral_normalization", True)
    sinksgd_kwargs.setdefault("sinkhorn_iterations", 5)

    base_lr = float(lr if lr is not None else getattr(args, "learning_rate", sinksgd_lr))
    scaled_trainable_params: List[Any] = []
    for param_group in trainable_params:
        if not isinstance(param_group, dict):
            scaled_trainable_params.append(param_group)
            continue
        scaled_group = dict(param_group)
        group_lr = scaled_group.get("lr")
        if group_lr is None or base_lr == 0.0:
            scaled_group["lr"] = effective_lr
        else:
            scaled_group["lr"] = effective_lr * (float(group_lr) / base_lr)
        scaled_trainable_params.append(scaled_group)

    optimizer_class = SinkSGD_adv
    optimizer = optimizer_class(scaled_trainable_params, lr=effective_lr, **sinksgd_kwargs)

    logger.info("SinkSGDAdv configuration:")
    logger.info(f"  - Base optimizer_args lr: {sinksgd_lr}")
    logger.info(f"  - Effective LR after batch scaling: {effective_lr}")
    logger.info(f"  - Batch scale mode: {batch_scale_mode}")
    logger.info(f"  - Reference batch size: {reference_batch_size}")
    logger.info(f"  - Effective batch size: {effective_batch_size}")
    logger.info("  - Orthogonal Sinkhorn: %s", sinksgd_kwargs.get("orthogonal_sinkhorn"))
    logger.info(
        "  - Spectral normalization: %s",
        sinksgd_kwargs.get("spectral_normalization"),
    )
    logger.info(
        "  - Sinkhorn iterations: %s",
        sinksgd_kwargs.get("sinkhorn_iterations"),
    )

    return optimizer_class, optimizer
