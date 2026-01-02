"""Muon-family optimizer creation helpers for WAN network trainer."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def create_muon_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [str, str, List[Any], List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using Muon optimizer | {optimizer_kwargs}")

    # Use SingleDeviceMuonWithAuxAdam for single-GPU training (avoids distributed training requirements)
    from optimizers.muon import SingleDeviceMuonWithAuxAdam

    # Separate trainable parameters by dimensionality
    # Muon should be applied to hidden weights (>=2D parameters) - Linear layers
    # AdamW should be applied to biases/gains (<2D) and other parameters
    all_params = extract_params(trainable_params)

    # Separate by dimensionality
    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    log_param_structure(
        "Muon",
        "AdamW",
        trainable_params,
        all_params,
        hidden_weights,
        hidden_gains_biases,
    )

    # Validate that we have parameters to optimize
    if len(hidden_weights) == 0 and len(hidden_gains_biases) == 0:
        raise ValueError("No trainable parameters found for Muon optimizer!")

    if len(hidden_weights) == 0:
        logger.warning(
            "No hidden weight parameters (>=2D) found for Muon. Consider using a different optimizer."
        )

    if len(hidden_gains_biases) == 0:
        logger.info(
            "No bias/gain parameters (<2D) found. This is normal for WAN LoRA networks."
        )

    # Use learning rate from args, with Muon group using higher LR as recommended
    muon_lr = optimizer_kwargs.get("muon_lr", 0.001)  # Conservative Muon LR for LoRA
    adam_lr = optimizer_kwargs.get("adam_lr", lr)  # Use specified LR for AdamW
    weight_decay = optimizer_kwargs.get("weight_decay", 0.001)  # Lower weight decay for LoRA
    betas = optimizer_kwargs.get("betas", (0.9, 0.95))

    # Muon-specific parameters based on theory
    momentum = optimizer_kwargs.get("momentum", 0.9)  # Lower momentum for stability
    ns_steps = optimizer_kwargs.get("ns_steps", 3)  # Fewer Newton-Schulz steps
    nesterov = optimizer_kwargs.get("nesterov", True)
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    # Only include parameter groups that have parameters
    param_groups = []

    if len(hidden_weights) > 0:
        param_groups.append(
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=muon_lr,
                weight_decay=weight_decay,
                momentum=momentum,  # Add momentum for Muon group
                ns_steps=ns_steps,
                nesterov=nesterov,
                initial_lr=muon_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(hidden_gains_biases) > 0:
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=adam_lr,
                betas=betas,
                weight_decay=weight_decay,
                initial_lr=adam_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    # Ensure we have at least one parameter group
    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for Muon optimizer!")

    optimizer_class = SingleDeviceMuonWithAuxAdam
    optimizer = optimizer_class(param_groups)

    # Log configuration for transparency
    logger.info("Muon configuration:")
    logger.info(f"  - Muon LR: {muon_lr}")
    logger.info(f"  - AdamW LR: {adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Momentum: {momentum}")
    logger.info(f"  - Newton-Schulz steps: {ns_steps}")

    return optimizer_class, optimizer


def create_normuon_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [str, str, List[Any], List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using NorMuon optimizer | {optimizer_kwargs}")

    from optimizers.normuon import (
        SingleDeviceNorMuonWithAuxAdam,
        apply_normuon_config_overrides,
    )

    # NorMuon uses the same parameter partitioning strategy as Muon
    all_params = extract_params(trainable_params)

    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    log_param_structure(
        "NorMuon",
        "Aux Adam",
        trainable_params,
        all_params,
        hidden_weights,
        hidden_gains_biases,
    )

    if len(hidden_weights) == 0 and len(hidden_gains_biases) == 0:
        raise ValueError("No trainable parameters found for NorMuon optimizer!")

    if len(hidden_weights) == 0:
        logger.warning(
            "No hidden weight parameters (>=2D) found for NorMuon. Consider using a different optimizer."
        )

    if len(hidden_gains_biases) == 0:
        logger.info(
            "No bias/gain parameters (<2D) found for NorMuon. Optimizer will only update matrix weights."
        )

    apply_normuon_config_overrides(args, optimizer_kwargs)

    normuon_lr = optimizer_kwargs.get("normuon_lr", optimizer_kwargs.get("muon_lr", 0.001))
    normuon_adam_lr = optimizer_kwargs.get("normuon_adam_lr", optimizer_kwargs.get("adam_lr", lr))
    weight_decay = optimizer_kwargs.get(
        "normuon_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
    )
    betas_value = optimizer_kwargs.get("normuon_betas", optimizer_kwargs.get("betas", (0.9, 0.95)))
    if isinstance(betas_value, list):
        betas = tuple(betas_value)
    else:
        betas = betas_value
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(
            f"NorMuon auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
        )
    betas = tuple(betas)
    momentum = optimizer_kwargs.get("normuon_momentum", optimizer_kwargs.get("momentum", 0.9))
    beta2 = optimizer_kwargs.get("normuon_beta2", 0.95)
    eps = optimizer_kwargs.get("normuon_eps", 1e-10)
    ns_steps = optimizer_kwargs.get("normuon_ns_steps", optimizer_kwargs.get("ns_steps", 3))
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    param_groups = []

    if len(hidden_weights) > 0:
        param_groups.append(
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=normuon_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                beta2=beta2,
                eps=eps,
                ns_steps=ns_steps,
                initial_lr=normuon_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(hidden_gains_biases) > 0:
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=normuon_adam_lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
                initial_lr=normuon_adam_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for NorMuon optimizer!")

    optimizer_class = SingleDeviceNorMuonWithAuxAdam
    optimizer = optimizer_class(param_groups)

    logger.info("NorMuon configuration:")
    logger.info(f"  - NorMuon LR: {normuon_lr}")
    logger.info(f"  - Aux Adam LR: {normuon_adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Momentum (beta1): {momentum}")
    logger.info(f"  - Beta2: {beta2}")
    logger.info(f"  - Epsilon: {eps}")
    logger.info(f"  - Newton-Schulz steps: {ns_steps}")
    logger.info(f"  - Aux Adam betas: {betas}")

    return optimizer_class, optimizer


def create_adamuon_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [str, str, List[Any], List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using AdaMuon optimizer | {optimizer_kwargs}")

    from optimizers.adamuon import (
        SingleDeviceAdaMuonWithAuxAdam,
        apply_adamuon_config_overrides,
    )

    all_params = extract_params(trainable_params)

    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    log_param_structure(
        "AdaMuon",
        "Aux Adam",
        trainable_params,
        all_params,
        hidden_weights,
        hidden_gains_biases,
    )

    if len(hidden_weights) == 0 and len(hidden_gains_biases) == 0:
        raise ValueError("No trainable parameters found for AdaMuon optimizer!")

    if len(hidden_weights) == 0:
        logger.warning(
            "No hidden weight parameters (>=2D) found for AdaMuon. Consider using a different optimizer."
        )

    if len(hidden_gains_biases) == 0:
        logger.info(
            "No bias/gain parameters (<2D) found for AdaMuon. Optimizer will only update matrix weights."
        )

    apply_adamuon_config_overrides(args, optimizer_kwargs)

    adamuon_lr = optimizer_kwargs.get("adamuon_lr", optimizer_kwargs.get("muon_lr", 0.001))
    adamuon_adam_lr = optimizer_kwargs.get("adamuon_adam_lr", optimizer_kwargs.get("adam_lr", lr))
    weight_decay = optimizer_kwargs.get(
        "adamuon_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
    )
    betas_value = optimizer_kwargs.get("adamuon_betas", optimizer_kwargs.get("betas", (0.9, 0.95)))
    if isinstance(betas_value, list):
        betas = tuple(betas_value)
    else:
        betas = betas_value
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(
            f"AdaMuon auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
        )
    betas = tuple(betas)

    momentum = optimizer_kwargs.get("adamuon_momentum", optimizer_kwargs.get("momentum", 0.95))
    beta2 = optimizer_kwargs.get("adamuon_beta2", 0.95)
    eps = optimizer_kwargs.get("adamuon_eps", 1e-8)
    ns_steps = optimizer_kwargs.get("adamuon_ns_steps", optimizer_kwargs.get("ns_steps", 5))
    scale_factor = optimizer_kwargs.get("adamuon_scale_factor", 0.2)
    nesterov = optimizer_kwargs.get("adamuon_nesterov", True)
    sign_stabilization = optimizer_kwargs.get("adamuon_sign_stabilization", True)
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    param_groups = []

    if len(hidden_weights) > 0:
        param_groups.append(
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=adamuon_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                beta2=beta2,
                eps=eps,
                ns_steps=ns_steps,
                scale_factor=scale_factor,
                nesterov=nesterov,
                sign_stabilization=sign_stabilization,
                initial_lr=adamuon_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(hidden_gains_biases) > 0:
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=adamuon_adam_lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
                initial_lr=adamuon_adam_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for AdaMuon optimizer!")

    optimizer_class = SingleDeviceAdaMuonWithAuxAdam
    optimizer = optimizer_class(param_groups)

    logger.info("AdaMuon configuration:")
    logger.info(f"  - AdaMuon LR: {adamuon_lr}")
    logger.info(f"  - Aux Adam LR: {adamuon_adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Momentum (beta1): {momentum}")
    logger.info(f"  - Beta2: {beta2}")
    logger.info(f"  - Epsilon: {eps}")
    logger.info(f"  - Newton-Schulz steps: {ns_steps}")
    logger.info(f"  - Scale factor: {scale_factor}")
    logger.info(f"  - Nesterov: {nesterov}")
    logger.info(f"  - Sign stabilization: {sign_stabilization}")
    logger.info(f"  - Aux Adam betas: {betas}")

    return optimizer_class, optimizer


def create_muonclip_optimizer(
    transformer: Optional[torch.nn.Module],
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [str, str, List[Any], List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using MuonClip optimizer | {optimizer_kwargs}")

    from optimizers.muonclip import (
        SingleDeviceMuonClipWithAuxAdam,
        auto_detect_attention_params,
    )

    all_params = extract_params(trainable_params)

    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    log_param_structure(
        "MuonClip",
        "Aux Adam",
        trainable_params,
        all_params,
        hidden_weights,
        hidden_gains_biases,
    )

    if len(hidden_weights) == 0 and len(hidden_gains_biases) == 0:
        raise ValueError("No trainable parameters found for MuonClip optimizer!")

    if len(hidden_weights) == 0:
        logger.warning(
            "No hidden weight parameters (>=2D) found for MuonClip. Consider using a different optimizer."
        )

    if len(hidden_gains_biases) == 0:
        logger.info(
            "No bias/gain parameters (<2D) found for MuonClip. Optimizer will only update matrix weights."
        )

    # MuonClip-specific parameters
    muonclip_lr = optimizer_kwargs.get("muonclip_lr", optimizer_kwargs.get("muon_lr", 0.001))
    muonclip_adam_lr = optimizer_kwargs.get("muonclip_adam_lr", optimizer_kwargs.get("adam_lr", lr))
    weight_decay = optimizer_kwargs.get(
        "muonclip_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
    )
    betas_value = optimizer_kwargs.get("muonclip_betas", optimizer_kwargs.get("betas", (0.9, 0.95)))
    if isinstance(betas_value, list):
        betas = tuple(betas_value)
    else:
        betas = betas_value
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(
            f"MuonClip auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
        )
    betas = tuple(betas)

    momentum = optimizer_kwargs.get("muonclip_momentum", optimizer_kwargs.get("momentum", 0.95))
    tau = optimizer_kwargs.get("muonclip_tau", optimizer_kwargs.get("tau", 100.0))
    ns_steps = optimizer_kwargs.get("muonclip_ns_steps", optimizer_kwargs.get("ns_steps", 5))
    nesterov = optimizer_kwargs.get("muonclip_nesterov", optimizer_kwargs.get("nesterov", True))
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    # Auto-detect attention parameters if requested
    auto_detect = optimizer_kwargs.get("muonclip_auto_detect_attention", True)

    param_groups = []

    if len(hidden_weights) > 0:
        param_groups.append(
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=muonclip_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                tau=tau,
                ns_steps=ns_steps,
                nesterov=nesterov,
                initial_lr=muonclip_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(hidden_gains_biases) > 0:
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=muonclip_adam_lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
                initial_lr=muonclip_adam_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for MuonClip optimizer!")

    optimizer_class = SingleDeviceMuonClipWithAuxAdam
    optimizer = optimizer_class(param_groups)

    # Auto-detect and register attention parameters if enabled
    if auto_detect and transformer is not None:
        try:
            attention_params = auto_detect_attention_params(transformer, all_params)
            if attention_params:
                optimizer.register_attention_params(attention_params)
                logger.info("QK-Clip attention stabilization enabled")
            else:
                logger.info("No attention parameters detected - MuonClip will work as standard Muon")
        except Exception as e:
            logger.warning(
                f"Failed to auto-detect attention parameters: {e}. MuonClip will work as standard Muon."
            )

    logger.info("MuonClip configuration:")
    logger.info(f"  - MuonClip LR: {muonclip_lr}")
    logger.info(f"  - Aux Adam LR: {muonclip_adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Momentum: {momentum}")
    logger.info(f"  - QK-Clip tau: {tau}")
    logger.info(f"  - Newton-Schulz steps: {ns_steps}")
    logger.info(f"  - Nesterov: {nesterov}")
    logger.info(f"  - Aux Adam betas: {betas}")

    return optimizer_class, optimizer


def create_manifoldmuon_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    log_param_structure: Callable[
        [str, str, List[Any], List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]],
        None,
    ],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using ManifoldMuon optimizer | {optimizer_kwargs}")

    from optimizers.manifoldmuon import (
        SingleDeviceManifoldMuonWithAuxAdam,
        apply_manifoldmuon_config_overrides,
    )

    all_params = extract_params(trainable_params)

    hidden_weights = [p for p in all_params if p.ndim >= 2]
    hidden_gains_biases = [p for p in all_params if p.ndim < 2]

    log_param_structure(
        "ManifoldMuon",
        "Aux Adam",
        trainable_params,
        all_params,
        hidden_weights,
        hidden_gains_biases,
    )

    if len(hidden_weights) == 0 and len(hidden_gains_biases) == 0:
        raise ValueError("No trainable parameters found for ManifoldMuon optimizer!")

    if len(hidden_weights) == 0:
        logger.warning(
            "No hidden weight parameters (>=2D) found for ManifoldMuon. Consider using a different optimizer."
        )

    if len(hidden_gains_biases) == 0:
        logger.info(
            "No bias/gain parameters (<2D) found for ManifoldMuon. Optimizer will only update matrix weights."
        )

    apply_manifoldmuon_config_overrides(args, optimizer_kwargs)

    manifoldmuon_lr = optimizer_kwargs.get(
        "manifoldmuon_lr", optimizer_kwargs.get("muon_lr", 0.001)
    )
    manifoldmuon_adam_lr = optimizer_kwargs.get(
        "manifoldmuon_adam_lr", optimizer_kwargs.get("adam_lr", lr)
    )
    weight_decay = optimizer_kwargs.get(
        "manifoldmuon_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
    )
    betas_value = optimizer_kwargs.get(
        "manifoldmuon_betas", optimizer_kwargs.get("betas", (0.9, 0.95))
    )
    if isinstance(betas_value, list):
        betas = tuple(betas_value)
    else:
        betas = betas_value
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(
            f"ManifoldMuon auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
        )
    betas = tuple(betas)

    eta = optimizer_kwargs.get("manifoldmuon_eta", 0.1)
    alpha = optimizer_kwargs.get("manifoldmuon_alpha", 0.01)
    dual_steps = optimizer_kwargs.get("manifoldmuon_dual_steps", 100)
    tolerance = optimizer_kwargs.get("manifoldmuon_tolerance", 1e-6)
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    param_groups = []

    if len(hidden_weights) > 0:
        param_groups.append(
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=manifoldmuon_lr,
                eta=eta,
                alpha=alpha,
                dual_steps=dual_steps,
                tolerance=tolerance,
                weight_decay=weight_decay,
                initial_lr=manifoldmuon_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(hidden_gains_biases) > 0:
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=manifoldmuon_adam_lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
                initial_lr=manifoldmuon_adam_lr,
                weight_decay_type=weight_decay_type,
            )
        )

    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for ManifoldMuon optimizer!")

    optimizer_class = SingleDeviceManifoldMuonWithAuxAdam
    optimizer = optimizer_class(param_groups)

    logger.info("ManifoldMuon configuration:")
    logger.info(f"  - ManifoldMuon LR: {manifoldmuon_lr}")
    logger.info(f"  - Aux Adam LR: {manifoldmuon_adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Eta (step size): {eta}")
    logger.info(f"  - Alpha (dual update rate): {alpha}")
    logger.info(f"  - Dual ascent steps: {dual_steps}")
    logger.info(f"  - Tolerance: {tolerance}")
    logger.info(f"  - Aux Adam betas: {betas}")

    return optimizer_class, optimizer


def create_riemannion_optimizer(
    args: Any,
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using Riemannion optimizer | {optimizer_kwargs}")

    from optimizers.riemannion import (
        SingleDeviceRiemannionWithAuxAdam,
        apply_riemannion_config_overrides,
    )

    apply_riemannion_config_overrides(args, optimizer_kwargs)

    riemannion_lr = optimizer_kwargs.get("riemannion_lr", optimizer_kwargs.get("muon_lr", 0.001))
    riemannion_adam_lr = optimizer_kwargs.get("riemannion_adam_lr", optimizer_kwargs.get("adam_lr", lr))
    weight_decay = optimizer_kwargs.get(
        "riemannion_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
    )
    betas_value = optimizer_kwargs.get(
        "riemannion_betas", optimizer_kwargs.get("betas", (0.9, 0.95))
    )
    if isinstance(betas_value, list):
        betas = tuple(betas_value)
    else:
        betas = betas_value
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError(
            "Riemannion auxiliary Adam betas must be a length-2 sequence. "
            f"Received: {betas_value}"
        )
    betas = tuple(betas)
    momentum = optimizer_kwargs.get("riemannion_momentum", 0.9)
    ns_steps = optimizer_kwargs.get("riemannion_ns_steps", 3)
    nesterov = optimizer_kwargs.get("riemannion_nesterov", True)
    max_elements = optimizer_kwargs.get("riemannion_max_elements", 20000000)
    weight_decay_type = optimizer_kwargs.get("weight_decay_type", "default")

    param_groups = []
    if (
        isinstance(trainable_params, list)
        and len(trainable_params) > 0
        and isinstance(trainable_params[0], dict)
    ):
        for group in trainable_params:
            if not isinstance(group, dict):
                continue
            if "lora_down" in group and "lora_up" in group:
                param_groups.append(
                    dict(
                        params=group.get("params", []),
                        lora_down=group.get("lora_down"),
                        lora_up=group.get("lora_up"),
                        pair_name=group.get("pair_name"),
                        rank=group.get("rank"),
                        use_riemannion=True,
                        lr=group.get("lr", riemannion_lr),
                        momentum=momentum,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                        weight_decay=weight_decay,
                        max_elements=max_elements,
                        initial_lr=group.get("lr", riemannion_lr),
                        weight_decay_type=weight_decay_type,
                    )
                )
            else:
                param_groups.append(
                    dict(
                        params=group.get("params", []),
                        use_riemannion=False,
                        lr=group.get("lr", riemannion_adam_lr),
                        betas=betas,
                        eps=optimizer_kwargs.get("eps", 1e-10),
                        weight_decay=weight_decay,
                        initial_lr=group.get("lr", riemannion_adam_lr),
                        weight_decay_type=weight_decay_type,
                    )
                )
    else:
        all_params = extract_params(trainable_params)
        matrix_params = [p for p in all_params if p.ndim >= 2]
        scalar_params = [p for p in all_params if p.ndim < 2]
        if matrix_params:
            param_groups.append(
                dict(
                    params=matrix_params,
                    use_riemannion=True,
                    lr=riemannion_lr,
                    momentum=momentum,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                    weight_decay=weight_decay,
                    max_elements=max_elements,
                    initial_lr=riemannion_lr,
                    weight_decay_type=weight_decay_type,
                )
            )
        if scalar_params:
            param_groups.append(
                dict(
                    params=scalar_params,
                    use_riemannion=False,
                    lr=riemannion_adam_lr,
                    betas=betas,
                    eps=optimizer_kwargs.get("eps", 1e-10),
                    weight_decay=weight_decay,
                    initial_lr=riemannion_adam_lr,
                    weight_decay_type=weight_decay_type,
                )
            )

    if len(param_groups) == 0:
        raise ValueError("No parameter groups created for Riemannion optimizer!")

    optimizer_class = SingleDeviceRiemannionWithAuxAdam
    optimizer = optimizer_class(param_groups)

    logger.info("Riemannion configuration:")
    logger.info(f"  - Riemannion LR: {riemannion_lr}")
    logger.info(f"  - Aux Adam LR: {riemannion_adam_lr}")
    logger.info(f"  - Weight decay: {weight_decay}")
    logger.info(f"  - Momentum: {momentum}")
    logger.info(f"  - Newton-Schulz steps: {ns_steps}")
    logger.info(f"  - Max delta elements: {max_elements}")
    logger.info(f"  - Nesterov: {nesterov}")
    logger.info(f"  - Aux Adam betas: {betas}")

    return optimizer_class, optimizer
