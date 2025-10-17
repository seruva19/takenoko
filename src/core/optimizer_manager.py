"""Optimizer and scheduler management for WAN network trainer.

This module handles all optimizer and learning rate scheduler creation, configuration,
and logging functionality. Extracted from wan_network_trainer.py to improve code
organization and maintainability.
"""

import ast
import importlib
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class OptimizerManager:
    """Handles optimizer and learning rate scheduler creation and management."""

    def __init__(self):
        pass

    @staticmethod
    def generate_step_logs(
        args: argparse.Namespace,
        current_loss: float,
        avr_loss: float,
        lr_scheduler: Any,
        lr_descriptions: Optional[List[str]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        keys_scaled: Optional[int] = None,
        mean_norm: Optional[float] = None,
        maximum_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate step logs for training metrics.

        Delegates to the shared implementation in core.metrics.generate_step_logs
        to keep behavior consistent across the codebase.
        """
        from core.metrics import generate_step_logs as _gsl

        return _gsl(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            None,  # ema_loss not used here
            None,  # model not available here
            None,  # global_step not available here
            None,  # per_source_losses not available here
            None,  # gradient_norm not available here
        )

    @staticmethod
    def get_optimizer(
        args: argparse.Namespace,
        transformer: torch.nn.Module,
        trainable_params: List[torch.nn.Parameter],
    ) -> Tuple[str, str, torch.optim.Optimizer, Callable, Callable]:
        """Create and configure the optimizer based on arguments.

        Returns:
            Tuple of (optimizer_name, optimizer_args_str, optimizer, train_fn, eval_fn)
        """
        # adamw, adamw8bit, adafactor
        optimizer_type = args.optimizer_type.lower()

        # split optimizer_type and optimizer_args
        optimizer_kwargs = {}
        if args.optimizer_args is not None and len(args.optimizer_args) > 0:
            logger.info(f"Processing optimizer args: {args.optimizer_args}")
            for arg in args.optimizer_args:
                key, value = arg.split("=", 1)  # Split only on first '='
                try:
                    # Try to parse as literal first
                    parsed_value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If that fails, treat as string (remove quotes if present)
                    parsed_value = value.strip("'\"")
                optimizer_kwargs[key] = parsed_value
                logger.info(
                    f"  Parsed: {key} = {parsed_value} (type: {type(parsed_value)})"
                )

        def extract_params(params_list: List[Any]) -> List[torch.nn.Parameter]:
            """Extract individual parameters from parameter groups or parameter lists."""
            extracted_params: List[torch.nn.Parameter] = []
            for i, item in enumerate(params_list):
                if isinstance(item, dict) and "params" in item:
                    logger.debug(
                        f"Parameter group {i}: {len(item['params'])} parameters found"
                    )
                    extracted_params.extend(list(item["params"]))
                elif isinstance(item, torch.nn.Parameter):
                    logger.debug(
                        f"Parameter {i}: shape {item.shape}, ndim {item.ndim}"
                    )
                    extracted_params.append(item)
                else:
                    logger.debug(f"Skipping item {i}: {type(item)}")
            return extracted_params

        def log_param_structure(
            matrix_label: str,
            scalar_label: str,
            trainable_items: List[Any],
            all_params: List[torch.nn.Parameter],
            matrix_params: List[torch.nn.Parameter],
            scalar_params: List[torch.nn.Parameter],
        ) -> None:
            logger.info(f"Total trainable parameters: {len(trainable_items)}")
            for i, param_item in enumerate(trainable_items):
                if isinstance(param_item, dict):
                    logger.info(f"Parameter group {i}:")
                    logger.info("  - Type: Parameter group (dictionary)")
                    logger.info(f"  - Keys: {list(param_item.keys())}")
                    if "params" in param_item:
                        params_list = list(param_item["params"])  # type: ignore[index]
                        logger.info(f"  - Number of parameters: {len(params_list)}")
                        if params_list:
                            logger.info(
                                "  - Parameter types: "
                                f"{[type(p).__name__ for p in params_list[:5]]}..."
                            )
                            logger.info(
                                "  - Parameter shapes: "
                                f"{[p.shape for p in params_list[:5]]}..."
                            )
                    if "lr" in param_item:
                        logger.info(f"  - Learning rate: {param_item['lr']}")  # type: ignore[index]
                    if "weight_decay" in param_item:
                        logger.info(
                            f"  - Weight decay: {param_item['weight_decay']}"
                        )  # type: ignore[index]
                elif isinstance(param_item, torch.nn.Parameter):
                    logger.info(f"Parameter {i}:")
                    logger.info("  - Type: Individual parameter")
                    logger.info(f"  - Shape: {param_item.shape}")
                    logger.info(f"  - Dtype: {param_item.dtype}")
                else:
                    logger.info(f"Item {i}:")
                    logger.info(f"  - Type: {type(param_item).__name__}")
                    logger.info(f"  - Content: {param_item}")

            logger.info(
                f"{matrix_label}: {len(matrix_params)} hidden weight parameters (?2D)"
            )
            logger.info(
                f"{scalar_label}: {len(scalar_params)} bias/gain parameters (<2D)"
            )

        lr = args.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type == "AdamW8bit".lower():
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "bitsandbytes is not installed. Please install bitsandbytes to use 8-bit optimizers."
                )

            logger.info(f"using AdamW8bit optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Adafactor".lower():
            # Adafactor: check relative_step and warmup_init
            if "relative_step" not in optimizer_kwargs:
                optimizer_kwargs["relative_step"] = True  # default
            if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get(
                "warmup_init", False
            ):
                logger.info("set relative_step to True because warmup_init is True")
                optimizer_kwargs["relative_step"] = True
            logger.info(f"using Adafactor optimizer | {optimizer_kwargs}")

            if optimizer_kwargs["relative_step"]:
                logger.info(f"relative_step is true")
                if lr != 0.0:
                    logger.warning(
                        "The specified learning rate will be used as initial_lr for Adafactor with relative_step=True."
                    )
                args.learning_rate = None

                if args.lr_scheduler != "adafactor":
                    logger.info(f"using adafactor_scheduler")
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
                    logger.warning(
                        "It is recommended to set clip_threshold=1.0 for Adafactor."
                    )

            optimizer_class = transformers.optimization.Adafactor
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamW".lower():
            logger.info(f"using AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "CAME8Bit".lower():
            # https://github.com/NVlabs/Sana/commit/dd38c12744ac652b01e9b653412fc76c355798bd
            try:
                from optimizers.sana_optimizer import CAME8BitWrapper

                optimizer_class = CAME8BitWrapper
                logger.info(
                    "using CamE8Bit optimizer (SANA implementation) | %s",
                    optimizer_kwargs,
                )
                optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
            except Exception as err:
                logger.warning(
                    "⚠️ Failed to import CamE8Bit implementation (%s).",
                    err,
                )
                raise ImportError("CamE8Bit implementation could not be used") from err

        elif optimizer_type == "Automagic".lower():
            logger.info(f"using Automagic optimizer | {optimizer_kwargs}")

            from optimizers.automagic import Automagic

            optimizer_class = Automagic
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamW8bitKahan".lower():
            logger.info(f"using AdamW8bitKahan optimizer | {optimizer_kwargs}")

            from optimizers.adamw_8bit_kahan import AdamW8bitKahan

            optimizer_class = AdamW8bitKahan
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamWOptimi".lower():
            logger.info(f"using optimi.AdamW optimizer | {optimizer_kwargs}")

            from optimi import AdamW

            optimizer_class = AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "LionOptimi".lower():
            logger.info(f"using optimi.Lion optimizer | {optimizer_kwargs}")

            from optimi import Lion

            optimizer_class = Lion
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Fira".lower():
            logger.info(f"using Fira optimizer | {optimizer_kwargs}")

            from optimizers.fira_optimizer import FiraOptimizerManager
            from vendor.fira.fira_adamw import FiraAdamW

            optimizer_class = FiraAdamW
            optimizer, functions = FiraOptimizerManager.create_fira_optimizer(
                args, transformer, trainable_params, lr, optimizer_kwargs
            )
            train_fn = functions["train_fn"]
            eval_fn = functions["eval_fn"]

        elif optimizer_type == "SophiaG".lower():
            logger.info(f"using SophiaG optimizer | {optimizer_kwargs}")

            from optimizers.sophia import SophiaG

            optimizer_class = SophiaG
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Soap".lower():
            logger.info(f"using Soap optimizer | {optimizer_kwargs}")

            from optimizers.soap import SOAP

            optimizer_class = SOAP
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "TemporalAdamW".lower():
            logger.info(f"using TemporalAdamW optimizer | {optimizer_kwargs}")

            # Import our custom optimizer
            from optimizers.temporal_adamw import TemporalAdamW

            optimizer_class = TemporalAdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "TemporalAdamW8bit".lower():
            logger.info(f"using TemporalAdamW8bit optimizer | {optimizer_kwargs}")

            try:
                from optimizers.temporal_adamw_8bit import TemporalAdamW8bit
            except Exception as err:
                raise ImportError(
                    "TemporalAdamW8bit requires bitsandbytes. Please install it."
                ) from err

            optimizer_class = TemporalAdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Muon".lower():
            logger.info(f"using Muon optimizer | {optimizer_kwargs}")

            # Use SingleDeviceMuonWithAuxAdam for single-GPU training (avoids distributed training requirements)
            from optimizers.muon import SingleDeviceMuonWithAuxAdam

            # Separate trainable parameters by dimensionality
            # Muon should be applied to hidden weights (?2D parameters) - Linear layers
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
                    "No hidden weight parameters (≥2D) found for Muon. Consider using a different optimizer."
                )

            if len(hidden_gains_biases) == 0:
                logger.info(
                    "No bias/gain parameters (<2D) found. This is normal for WAN LoRA networks."
                )

            # Use learning rate from args, with Muon group using higher LR as recommended
            muon_lr = optimizer_kwargs.get(
                "muon_lr", 0.001
            )  # Conservative Muon LR for LoRA
            adam_lr = optimizer_kwargs.get("adam_lr", lr)  # Use specified LR for AdamW
            weight_decay = optimizer_kwargs.get(
                "weight_decay", 0.001
            )  # Lower weight decay for LoRA
            betas = optimizer_kwargs.get("betas", (0.9, 0.95))

            # Muon-specific parameters based on theory
            momentum = optimizer_kwargs.get(
                "momentum", 0.9
            )  # Lower momentum for stability
            ns_steps = optimizer_kwargs.get("ns_steps", 3)  # Fewer Newton-Schulz steps

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
                    )
                )

            # Ensure we have at least one parameter group
            if len(param_groups) == 0:
                raise ValueError("No parameter groups created for Muon optimizer!")

            optimizer_class = SingleDeviceMuonWithAuxAdam
            optimizer = optimizer_class(param_groups)

            # Log configuration for transparency
            logger.info(f"Muon configuration:")
            logger.info(f"  - Muon LR: {muon_lr}")
            logger.info(f"  - AdamW LR: {adam_lr}")
            logger.info(f"  - Weight decay: {weight_decay}")
            logger.info(f"  - Momentum: {momentum}")
            logger.info(f"  - Newton-Schulz steps: {ns_steps}")

        elif optimizer_type == "Normuon".lower():
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
                    "No hidden weight parameters (?2D) found for NorMuon. Consider using a different optimizer."
                )

            if len(hidden_gains_biases) == 0:
                logger.info(
                    "No bias/gain parameters (<2D) found for NorMuon. Optimizer will only update matrix weights."
                )

            apply_normuon_config_overrides(args, optimizer_kwargs)

            normuon_lr = optimizer_kwargs.get(
                "normuon_lr", optimizer_kwargs.get("muon_lr", 0.001)
            )
            normuon_adam_lr = optimizer_kwargs.get(
                "normuon_adam_lr", optimizer_kwargs.get("adam_lr", lr)
            )
            weight_decay = optimizer_kwargs.get(
                "normuon_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
            )
            betas_value = optimizer_kwargs.get(
                "normuon_betas", optimizer_kwargs.get("betas", (0.9, 0.95))
            )
            if isinstance(betas_value, list):
                betas = tuple(betas_value)
            else:
                betas = betas_value
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError(
                    f"NorMuon auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
                )
            betas = tuple(betas)
            momentum = optimizer_kwargs.get(
                "normuon_momentum", optimizer_kwargs.get("momentum", 0.9)
            )
            beta2 = optimizer_kwargs.get("normuon_beta2", 0.95)
            eps = optimizer_kwargs.get("normuon_eps", 1e-10)
            ns_steps = optimizer_kwargs.get(
                "normuon_ns_steps", optimizer_kwargs.get("ns_steps", 3)
            )

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

        elif optimizer_type == "Prodigy".lower():
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

        elif optimizer_type == "Scion".lower():
            logger.info(f"using Scion optimizer | {optimizer_kwargs}")

            from optimizers.scion import Scion

            # Default parameters
            momentum = optimizer_kwargs.get("momentum", 0.1)
            scale = optimizer_kwargs.get("scale", 1.0)
            norm = optimizer_kwargs.get("norm", "Auto")
            norm_kwargs = optimizer_kwargs.get("norm_kwargs", {})
            unconstrained = optimizer_kwargs.get("unconstrained", False)

            # Check if user provided parameter groups with different norms
            # If trainable_params is already a list of dicts with 'norm' key, use it directly
            if (
                isinstance(trainable_params, list)
                and len(trainable_params) > 0
                and isinstance(trainable_params[0], dict)
                and "norm" in trainable_params[0]
            ):
                logger.info("Using custom parameter groups for Scion")
                optimizer_class = Scion
                optimizer = optimizer_class(
                    trainable_params, lr=lr, momentum=momentum
                )
            else:
                # Single parameter group with specified norm
                logger.info(
                    f"Scion config: norm={norm}, scale={scale}, momentum={momentum}, unconstrained={unconstrained}"
                )
                optimizer_class = Scion
                optimizer = optimizer_class(
                    trainable_params,
                    lr=lr,
                    momentum=momentum,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    scale=scale,
                    unconstrained=unconstrained,
                )

        elif optimizer_type == "ScionLight".lower():
            logger.info(f"using ScionLight optimizer (memory-efficient) | {optimizer_kwargs}")

            from optimizers.scion import ScionLight

            # Default parameters
            momentum = optimizer_kwargs.get("momentum", 0.1)
            scale = optimizer_kwargs.get("scale", 1.0)
            norm = optimizer_kwargs.get("norm", "Auto")
            norm_kwargs = optimizer_kwargs.get("norm_kwargs", {})
            unconstrained = optimizer_kwargs.get("unconstrained", False)

            # Check if user provided parameter groups with different norms
            if (
                isinstance(trainable_params, list)
                and len(trainable_params) > 0
                and isinstance(trainable_params[0], dict)
                and "norm" in trainable_params[0]
            ):
                logger.info("Using custom parameter groups for ScionLight")
                optimizer_class = ScionLight
                optimizer = optimizer_class(
                    trainable_params, lr=lr, momentum=momentum
                )
            else:
                # Single parameter group with specified norm
                logger.info(
                    f"ScionLight config: norm={norm}, scale={scale}, momentum={momentum}, unconstrained={unconstrained}"
                )
                optimizer_class = ScionLight
                optimizer = optimizer_class(
                    trainable_params,
                    lr=lr,
                    momentum=momentum,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    scale=scale,
                    unconstrained=unconstrained,
                )

        if optimizer is None:
            case_sensitive_optimizer_type = args.optimizer_type  # not lower
            logger.info(f"using {case_sensitive_optimizer_type} | {optimizer_kwargs}")

            if "." not in case_sensitive_optimizer_type:  # from torch.optim
                optimizer_module = torch.optim
            else:  # from other library
                values = case_sensitive_optimizer_type.split(".")
                optimizer_module = importlib.import_module(".".join(values[:-1]))
                case_sensitive_optimizer_type = values[-1]

            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        # for logging - fix potential None issue
        if optimizer_class is not None:
            optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        else:
            optimizer_name = "unknown"
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        # get train and eval functions
        if hasattr(optimizer, "train") and callable(optimizer.train):  # type: ignore
            train_fn = optimizer.train  # type: ignore
            eval_fn = optimizer.eval  # type: ignore
        else:
            train_fn = lambda: None
            eval_fn = lambda: None

        return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn

    @staticmethod
    def is_schedulefree_optimizer(
        optimizer: torch.optim.Optimizer, args: argparse.Namespace
    ) -> bool:
        """Check if the optimizer is a schedulefree optimizer."""
        return args.optimizer_type.lower().endswith(
            "schedulefree".lower()
        )  # or args.optimizer_schedulefree_wrapper

    @staticmethod
    def get_dummy_scheduler(optimizer: torch.optim.Optimizer) -> Any:
        """Get a dummy scheduler for schedulefree optimizer.

        This scheduler supports only empty step(), get_last_lr() and optimizers.
        This scheduler is used for logging only.
        This isn't wrapped by accelerator because this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
        """

        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    @staticmethod
    def get_lr_scheduler(
        args: argparse.Namespace, optimizer: torch.optim.Optimizer, num_processes: int
    ) -> Any:
        """Unified API to get any scheduler from its name."""
        # if schedulefree optimizer, return dummy scheduler
        if OptimizerManager.is_schedulefree_optimizer(optimizer, args):
            return OptimizerManager.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = (
            args.max_train_steps * num_processes
        )  # * args.gradient_accumulation_steps
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps)
            if isinstance(args.lr_warmup_steps, float)
            else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps)
            if isinstance(args.lr_decay_steps, float)
            else args.lr_decay_steps
        )

        # Fix potential None issues
        if num_warmup_steps is None:
            num_warmup_steps = 0
        if num_decay_steps is None:
            num_decay_steps = 0

        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(
                    f"{name} does not require `num_warmup_steps`. Set None or 0."
                )
            return return_vals

        # using any lr_scheduler from other library
        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type

            # Built-in aliases for custom schedulers
            alias_map = {
                # Short alias → fully-qualified class path
                "per_cycle_cosine": "optimizers.custom_schedulers.per_cycle_cosine.PerCycleWarmupCosineWithFloor",
                "ema_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.EMAAdaptiveScheduler",
                "noise_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.NoiseAdaptiveScheduler",
                "hybrid_adaptive": "optimizers.custom_schedulers.adaptive_schedulers.HybridAdaptiveScheduler",
                "adaptive_per_cycle_cosine": "optimizers.custom_schedulers.adaptive_schedulers.AdaptivePerCycleWarmupCosineScheduler",
                "cycle_adaptive_per_cycle": "optimizers.custom_schedulers.adaptive_schedulers.CycleAdaptivePerCycleScheduler",
                "rex": "optimizers.custom_schedulers.rex_scheduler.RexLR",
            }

            if lr_scheduler_type in alias_map:
                fqcn = alias_map[lr_scheduler_type]
                module_path, class_name = fqcn.rsplit(".", 1)
                logger.info(
                    f"using alias '{lr_scheduler_type}' → {fqcn} | {lr_scheduler_kwargs} as lr_scheduler"
                )
                lr_scheduler_module = importlib.import_module(module_path)
                lr_scheduler_class = getattr(lr_scheduler_module, class_name)

                # Special handling for REX scheduler to auto-populate parameters
                if lr_scheduler_type == "rex":
                    # Set default parameters if not provided
                    if "max_lr" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["max_lr"] = args.learning_rate
                    if "num_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["num_steps"] = num_training_steps
                    if "num_warmup_steps" not in lr_scheduler_kwargs:
                        lr_scheduler_kwargs["num_warmup_steps"] = num_warmup_steps
                    if (
                        "min_lr_ratio" not in lr_scheduler_kwargs
                        and "min_lr" not in lr_scheduler_kwargs
                    ):
                        lr_scheduler_kwargs["min_lr_ratio"] = (
                            min_lr_ratio if min_lr_ratio is not None else 0.01
                        )

                return lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

            logger.info(
                f"using {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler"
            )
            if "." not in lr_scheduler_type:  # default to use torch.optim
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert (
                type(optimizer) == transformers.optimization.Adafactor
            ), f"adafactor scheduler must be used with Adafactor optimizer"
            initial_lr = float(name.split(":")[1])
            # logger.info(f"adafactor scheduler init lr {initial_lr}")
            return wrap_check_needless_num_warmup_steps(
                transformers.optimization.AdafactorSchedule(optimizer, initial_lr)
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(
                optimizer, **lr_scheduler_kwargs
            )  # step_rules and last_epoch are given as kwargs

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(
                schedule_func(optimizer, **lr_scheduler_kwargs)
            )

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(
                f"{name} requires `num_warmup_steps`, please provide that argument."
            )

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs
            )

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                timescale=timescale,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(
                f"{name} requires `num_training_steps`, please provide that argument."
            )

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        # these schedulers do not require `num_decay_steps`
        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_decay_steps`
        if num_decay_steps is None:
            raise ValueError(
                f"{name} requires `num_decay_steps`, please provide that argument."
            )
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        )
