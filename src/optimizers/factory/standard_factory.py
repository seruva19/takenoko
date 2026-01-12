"""Standard optimizer creation helpers for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch
import transformers


def create_adamw_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using AdamW optimizer | {optimizer_kwargs}")
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_stochastic_adamw_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using StochasticAdamW optimizer | {optimizer_kwargs}")
    from optimizers.stochastic_adamw import StochasticAdamW
    
    optimizer_class = StochasticAdamW
    
    # Extract specific kwargs for StochasticAdamW if mixed with general ones, 
    # though usually they are passed directly.
    # We ensure defaults are respected if keys are missing.
    
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_adafactor_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer, float]:
    # Adafactor: check relative_step and warmup_init
    if "relative_step" not in optimizer_kwargs:
        optimizer_kwargs["relative_step"] = True  # default
    if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
        logger.info("set relative_step to True because warmup_init is True")
        optimizer_kwargs["relative_step"] = True
    logger.info(f"using Adafactor optimizer | {optimizer_kwargs}")

    if optimizer_kwargs["relative_step"]:
        logger.info("relative_step is true")
        if lr != 0.0:
            logger.warning(
                "The specified learning rate will be used as initial_lr for Adafactor with relative_step=True."
            )
        args.learning_rate = None

        if args.lr_scheduler != "adafactor":
            logger.info("using adafactor_scheduler")
        args.lr_scheduler = f"adafactor:{lr}"

        lr = None
    else:
        if args.max_grad_norm != 0.0:
            logger.warning(
                "max_grad_norm is set, so gradient clipping is enabled. Consider setting it to 0 to disable clipping."
            )
        if args.lr_scheduler != "constant_with_warmup":
            logger.warning(
                "It is recommended to use the 'constant_with_warmup' scheduler with Adafactor when relative_step is False."
            )
        if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
            logger.warning("It is recommended to set clip_threshold=1.0 for Adafactor.")

    optimizer_class = transformers.optimization.Adafactor
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer, lr
