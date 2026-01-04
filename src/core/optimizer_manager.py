"""Optimizer management for WAN network trainer.

This module handles optimizer creation, configuration, and logging functionality.
Extracted from wan_network_trainer.py to improve code
organization and maintainability.
"""

import argparse
import ast
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class OptimizerManager:
    """Handles optimizer creation and management."""

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
                    logger.debug(f"Parameter {i}: shape {item.shape}, ndim {item.ndim}")
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

        from optimizers.factory.galore_factory import (
            is_q_galore_optimizer_type,
            prepare_galore_trainable_params,
        )

        trainable_params, optimizer_kwargs = prepare_galore_trainable_params(
            args,
            transformer,
            trainable_params,
            optimizer_kwargs,
            optimizer_type,
            extract_params=extract_params,
            logger=logger,
        )

        lr = args.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type == "AdamW8bit".lower():
            from optimizers.factory.bitsandbytes_factory import (
                create_adamw8bit_optimizer,
            )

            optimizer_class, optimizer = create_adamw8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Adafactor".lower():
            from optimizers.factory.standard_factory import create_adafactor_optimizer

            optimizer_class, optimizer, lr = create_adafactor_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "AdamW".lower():
            from optimizers.factory.standard_factory import create_adamw_optimizer

            optimizer_class, optimizer = create_adamw_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif is_q_galore_optimizer_type(optimizer_type):
            from optimizers.factory.q_galore_factory import (
                create_q_galore_adamw8bit_optimizer,
            )

            optimizer_class, optimizer = create_q_galore_adamw8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type in {"galoreadamw", "galore_adamw"}:
            from optimizers.factory.galore_factory import create_galore_adamw_optimizer

            optimizer_class, optimizer = create_galore_adamw_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type in {"galoreadamw8bit", "galore_adamw8bit"}:
            from optimizers.factory.galore_factory import (
                create_galore_adamw8bit_optimizer,
            )

            optimizer_class, optimizer = create_galore_adamw8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type in {"galoreadafactor", "galore_adafactor"}:
            from optimizers.factory.galore_factory import (
                create_galore_adafactor_optimizer,
            )

            optimizer_class, optimizer = create_galore_adafactor_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "IVON".lower():
            from optimizers.factory.custom_factory import create_ivon_optimizer

            optimizer_class, optimizer = create_ivon_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "CAME8Bit".lower():
            from optimizers.factory.bitsandbytes_factory import (
                create_came8bit_optimizer,
            )

            optimizer_class, optimizer = create_came8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Automagic".lower():
            from optimizers.factory.custom_factory import create_automagic_optimizer

            optimizer_class, optimizer = create_automagic_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "AdamW8bitKahan".lower():
            from optimizers.factory.bitsandbytes_factory import (
                create_adamw8bitkahan_optimizer,
            )

            optimizer_class, optimizer = create_adamw8bitkahan_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "AdamWOptimi".lower():
            from optimizers.factory.custom_factory import create_adamw_optimi_optimizer

            optimizer_class, optimizer = create_adamw_optimi_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "LionOptimi".lower():
            from optimizers.factory.custom_factory import create_lion_optimi_optimizer

            optimizer_class, optimizer = create_lion_optimi_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Fira".lower():
            from optimizers.fira_optimizer import create_fira_optimizer_init

            optimizer_class, optimizer, functions = create_fira_optimizer_init(
                args,
                transformer,
                trainable_params,
                lr,
                optimizer_kwargs,
            )
            train_fn = functions["train_fn"]
            eval_fn = functions["eval_fn"]

        elif optimizer_type == "SophiaG".lower():
            from optimizers.factory.custom_factory import create_sophiag_optimizer

            optimizer_class, optimizer = create_sophiag_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Soap".lower():
            from optimizers.factory.custom_factory import create_soap_optimizer

            optimizer_class, optimizer = create_soap_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "SPlus".lower():
            from optimizers.splus import SPlus

            logger.info(f"using SPlus optimizer | {optimizer_kwargs}")
            optimizer_class = SPlus
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "TemporalAdamW".lower():
            from optimizers.factory.custom_factory import (
                create_temporal_adamw_optimizer,
            )

            optimizer_class, optimizer = create_temporal_adamw_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "TemporalAdamW8bit".lower():
            from optimizers.factory.bitsandbytes_factory import (
                create_temporal_adamw8bit_optimizer,
            )

            optimizer_class, optimizer = create_temporal_adamw8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "RavenAdamW".lower():
            from optimizers.factory.custom_factory import create_raven_adamw_optimizer

            optimizer_class, optimizer = create_raven_adamw_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "RavenAdamW8bit".lower():
            from optimizers.factory.bitsandbytes_factory import (
                create_raven_adamw8bit_optimizer,
            )

            optimizer_class, optimizer = create_raven_adamw8bit_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Muon".lower():
            from optimizers.factory.muon_factory import create_muon_optimizer

            optimizer_class, optimizer = create_muon_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                log_param_structure=log_param_structure,
                logger=logger,
            )

        elif optimizer_type == "Normuon".lower():
            from optimizers.factory.muon_factory import create_normuon_optimizer

            optimizer_class, optimizer = create_normuon_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                log_param_structure=log_param_structure,
                logger=logger,
            )

        elif optimizer_type == "Adamuon".lower():
            from optimizers.factory.muon_factory import create_adamuon_optimizer

            optimizer_class, optimizer = create_adamuon_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                log_param_structure=log_param_structure,
                logger=logger,
            )

        elif optimizer_type == "MuonClip".lower():
            from optimizers.factory.muon_factory import create_muonclip_optimizer

            optimizer_class, optimizer = create_muonclip_optimizer(
                transformer,
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                log_param_structure=log_param_structure,
                logger=logger,
            )

        elif optimizer_type == "ManifoldMuon".lower():
            from optimizers.factory.muon_factory import create_manifoldmuon_optimizer

            optimizer_class, optimizer = create_manifoldmuon_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                log_param_structure=log_param_structure,
                logger=logger,
            )

        elif optimizer_type == "Riemannion".lower():
            from optimizers.factory.muon_factory import create_riemannion_optimizer

            optimizer_class, optimizer = create_riemannion_optimizer(
                args,
                trainable_params,
                lr,
                optimizer_kwargs,
                extract_params=extract_params,
                logger=logger,
            )

        elif optimizer_type == "Prodigy".lower():
            from optimizers.factory.prodigy_factory import create_prodigy_optimizer

            optimizer_class, optimizer = create_prodigy_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "Scion".lower():
            from optimizers.factory.scion_factory import create_scion_optimizer

            optimizer_class, optimizer = create_scion_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        elif optimizer_type == "ScionLight".lower():
            from optimizers.factory.scion_factory import create_scionlight_optimizer

            optimizer_class, optimizer = create_scionlight_optimizer(
                trainable_params,
                lr,
                optimizer_kwargs,
                logger,
            )

        # TensorFlow-ported optimizers
        elif optimizer_type == "Kron".lower():
            from optimizers.kron import Kron

            optimizer_class, optimizer = Kron, Kron(
                trainable_params,
                lr=lr,
                **optimizer_kwargs,
            )

        elif optimizer_type == "Conda".lower():
            from optimizers.conda import Conda

            optimizer_class, optimizer = Conda, Conda(
                trainable_params,
                lr=lr,
                **optimizer_kwargs,
            )

        elif optimizer_type in {"vsgd", "VSGD", "Vsgd"}:
            from optimizers.vsgd import VSGD

            optimizer_class, optimizer = VSGD, VSGD(
                trainable_params,
                lr=lr,
                **optimizer_kwargs,
            )

        elif optimizer_type in {"rangerva", "RangerVA", "RANGERVA"}:
            from optimizers.rangerva import RangerVA

            optimizer_class, optimizer = RangerVA, RangerVA(
                trainable_params,
                lr=lr,
                **optimizer_kwargs,
            )

        elif optimizer_type in {"nvnovograd", "NvNovoGrad", "NVNOVOGRAD"}:
            from optimizers.nvnovograd import NvNovoGrad

            optimizer_class, optimizer = NvNovoGrad, NvNovoGrad(
                trainable_params,
                lr=lr,
                **optimizer_kwargs,
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
