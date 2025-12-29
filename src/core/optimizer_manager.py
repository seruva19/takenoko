"""Optimizer and scheduler management for WAN network trainer.

This module handles all optimizer and learning rate scheduler creation, configuration,
and logging functionality. Extracted from wan_network_trainer.py to improve code
organization and maintainability.
"""

import ast
import fnmatch
import importlib
import argparse
import re
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
    def _apply_galore_param_groups(
        args: argparse.Namespace,
        trainable_params: List[Any],
        galore_settings: Dict[str, Any],
        param_name_map: Optional[Dict[int, str]] = None,
    ) -> List[Any]:
        apply_to = str(galore_settings.get("galore_apply_to", "matrix_only")).lower()
        if apply_to not in {"matrix_only", "matrix_and_tensor"}:
            raise ValueError(
                "galore_apply_to must be 'matrix_only' or 'matrix_and_tensor'"
            )

        if not trainable_params:
            return trainable_params

        if (
            isinstance(trainable_params, list)
            and len(trainable_params) > 0
            and isinstance(trainable_params[0], dict)
        ):
            base_groups = trainable_params
        else:
            base_groups = [{"params": trainable_params}]

        if any(isinstance(group, dict) and "rank" in group for group in base_groups):
            logger.info(
                "GaLore: using existing parameter groups that already define rank."
            )
            return base_groups

        try:
            rank = int(galore_settings.get("galore_rank", 128))
        except (TypeError, ValueError):
            raise ValueError("galore_rank must be an int") from None
        if rank <= 0:
            raise ValueError("galore_rank must be > 0")

        try:
            update_proj_gap = int(galore_settings.get("galore_update_proj_gap", 200))
        except (TypeError, ValueError):
            raise ValueError("galore_update_proj_gap must be an int") from None
        if update_proj_gap < 1:
            raise ValueError("galore_update_proj_gap must be >= 1")

        try:
            scale = float(galore_settings.get("galore_scale", 0.25))
        except (TypeError, ValueError):
            raise ValueError("galore_scale must be a float") from None
        if scale <= 0:
            raise ValueError("galore_scale must be > 0")

        proj_type = str(galore_settings.get("galore_proj_type", "std"))
        allowed_proj_types = {"std", "reverse_std", "right", "left", "full"}
        if proj_type not in allowed_proj_types:
            raise ValueError(
                f"galore_proj_type must be one of {sorted(allowed_proj_types)}"
            )

        group_by = str(galore_settings.get("galore_group_by", "none")).lower()
        allowed_group_by = {"none", "layer", "param"}
        if group_by not in allowed_group_by:
            raise ValueError(
                f"galore_group_by must be one of {sorted(allowed_group_by)}"
            )

        overrides = galore_settings.get("galore_group_overrides")
        if overrides is None:
            override_rules: List[Dict[str, Any]] = []
        elif isinstance(overrides, dict):
            override_rules = [overrides]
        elif isinstance(overrides, list):
            override_rules = [rule for rule in overrides if isinstance(rule, dict)]
        else:
            raise ValueError("galore_group_overrides must be a dict or list of dicts")

        logger.info(
            "GaLore enabled: apply_to=%s rank=%d update_proj_gap=%d scale=%.3f proj_type=%s group_by=%s",
            apply_to,
            rank,
            update_proj_gap,
            scale,
            proj_type,
            group_by,
        )

        def clone_group(group: Dict[str, Any], params: List[torch.nn.Parameter]) -> Dict[str, Any]:
            new_group = {key: value for key, value in group.items() if key != "params"}
            new_group["params"] = params
            return new_group

        def clone_galore_group(
            group: Dict[str, Any],
            params: List[torch.nn.Parameter],
            dim: int,
            group_settings: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            group_settings = group_settings or {}
            new_group = clone_group(group, params)
            new_group.update(
                {
                    "rank": group_settings.get("rank", rank),
                    "update_proj_gap": group_settings.get(
                        "update_proj_gap", update_proj_gap
                    ),
                    "scale": group_settings.get("scale", scale),
                    "proj_type": group_settings.get("proj_type", proj_type),
                    "dim": dim,
                }
            )
            return new_group

        def resolve_group_key(param: torch.nn.Parameter) -> str:
            if group_by == "none":
                return "all"
            if param_name_map is None:
                return "unmapped"
            name = param_name_map.get(id(param))
            if not name:
                return "unmapped"
            if group_by == "param":
                return name
            return name.rsplit(".", 1)[0] if "." in name else name

        def resolve_param_name(param: torch.nn.Parameter) -> Optional[str]:
            if param_name_map is None:
                return None
            return param_name_map.get(id(param))

        def resolve_group_settings(param_name: Optional[str]) -> Dict[str, Any]:
            if not param_name:
                return {}
            for rule in override_rules:
                pattern = rule.get("pattern")
                if not isinstance(pattern, str):
                    continue
                if fnmatch.fnmatchcase(param_name, pattern):
                    return {
                        key: rule[key]
                        for key in ("rank", "update_proj_gap", "scale", "proj_type")
                        if key in rule
                    }
            return {}

        def group_params(
            params: List[torch.nn.Parameter],
        ) -> List[Tuple[List[torch.nn.Parameter], Dict[str, Any]]]:
            if group_by == "none":
                rep_name = resolve_param_name(params[0]) if params else None
                return [(params, resolve_group_settings(rep_name))]
            grouped: Dict[str, List[torch.nn.Parameter]] = {}
            for param in params:
                key = resolve_group_key(param)
                grouped.setdefault(key, []).append(param)
            grouped_params: List[Tuple[List[torch.nn.Parameter], Dict[str, Any]]] = []
            for grouped_list in grouped.values():
                rep_name = resolve_param_name(grouped_list[0]) if grouped_list else None
                grouped_params.append((grouped_list, resolve_group_settings(rep_name)))
            return grouped_params

        new_groups: List[Dict[str, Any]] = []
        for group in base_groups:
            if not isinstance(group, dict):
                raise TypeError("Optimizer parameter groups must be dictionaries.")

            params = list(group.get("params", []))
            if not params:
                continue

            matrix_params = [p for p in params if p.ndim == 2]
            tensor_params = [p for p in params if p.ndim > 2]
            scalar_params = [p for p in params if p.ndim < 2]

            if apply_to == "matrix_only":
                if matrix_params:
                    grouped_matrices = group_params(matrix_params)
                    for grouped_params, group_settings in grouped_matrices:
                        new_groups.append(
                            clone_galore_group(
                                group,
                                grouped_params,
                                dim=2,
                                group_settings=group_settings,
                            )
                        )
                if tensor_params or scalar_params:
                    remaining = tensor_params + scalar_params
                    new_groups.append(clone_group(group, remaining))
            else:
                if matrix_params:
                    grouped_matrices = group_params(matrix_params)
                    for grouped_params, group_settings in grouped_matrices:
                        new_groups.append(
                            clone_galore_group(
                                group,
                                grouped_params,
                                dim=2,
                                group_settings=group_settings,
                            )
                        )
                if tensor_params:
                    grouped_tensors = group_params(tensor_params)
                    for grouped_params, group_settings in grouped_tensors:
                        max_dim = max(p.ndim for p in grouped_params)
                        new_groups.append(
                            clone_galore_group(
                                group,
                                grouped_params,
                                dim=max_dim,
                                group_settings=group_settings,
                            )
                        )
                if scalar_params:
                    new_groups.append(clone_group(group, scalar_params))

        if not new_groups:
            return trainable_params

        return new_groups

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
        galore_types = {
            "galoreadamw",
            "galore_adamw",
            "galoreadamw8bit",
            "galore_adamw8bit",
            "galoreadafactor",
            "galore_adafactor",
        }
        is_galore_optimizer = optimizer_type in galore_types
        q_galore_types = {
            "qgaloreadamw8bit",
            "qgalore_adamw8bit",
            "q_galore_adamw8bit",
            "qgaloreadamw8bitlayerwise",
            "qgalore_adamw8bit_layerwise",
            "q_galore_adamw8bit_layerwise",
        }
        is_q_galore_optimizer = optimizer_type in q_galore_types

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

        galore_settings: Dict[str, Any] = {}
        for key in (
            "galore_rank",
            "galore_update_proj_gap",
            "galore_scale",
            "galore_proj_type",
            "galore_apply_to",
            "galore_group_by",
            "galore_group_overrides",
        ):
            if key in optimizer_kwargs:
                galore_settings[key] = optimizer_kwargs.pop(key)

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

        def _looks_like_regex(pattern: str) -> bool:
            return any(ch in pattern for ch in ".^$*+?{}[]|()\\")

        def _is_all_linear(target_modules: Union[str, List[str], None]) -> bool:
            if isinstance(target_modules, str):
                return target_modules.replace("_", "-") == "all-linear"
            if isinstance(target_modules, list):
                return any(
                    isinstance(item, str) and item.replace("_", "-") == "all-linear"
                    for item in target_modules
                )
            return False

        def _matches_target_module(
            target_modules: Union[str, List[str], None],
            module_name: str,
        ) -> Tuple[bool, bool]:
            if target_modules is None:
                return False, False
            if isinstance(target_modules, list):
                for target in target_modules:
                    if not isinstance(target, str):
                        continue
                    if target.replace("_", "-") == "all-linear":
                        continue
                    if _looks_like_regex(target):
                        try:
                            if re.search(target, module_name):
                                return True, True
                        except re.error:
                            logger.warning(
                                "Invalid regex in q_galore_target_modules: %s", target
                            )
                    elif target in module_name:
                        return True, False
                return False, False
            if isinstance(target_modules, str):
                if target_modules.replace("_", "-") == "all-linear":
                    return False, False
                if _looks_like_regex(target_modules):
                    try:
                        return re.search(target_modules, module_name) is not None, True
                    except re.error:
                        logger.warning(
                            "Invalid regex in q_galore_target_modules: %s", target_modules
                        )
                        return False, False
                return target_modules in module_name, False
            return False, False

        def _get_linear_weight(module: torch.nn.Module) -> Optional[torch.nn.Parameter]:
            if isinstance(module, torch.nn.Linear):
                return module.weight
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.nn.Parameter) and weight.ndim == 2:
                return weight
            return None

        if is_q_galore_optimizer:
            if optimizer_type.endswith("layerwise"):
                raise NotImplementedError(
                    "Layer-wise Q-GaLore is not supported in Takenoko yet. "
                    "Use optimizer_type='q_galore_adamw8bit' instead."
                )

            q_galore_target_modules = getattr(args, "q_galore_target_modules", None)
            if q_galore_target_modules is None:
                q_galore_target_modules = optimizer_kwargs.pop(
                    "q_galore_target_modules", None
                )
            if q_galore_target_modules is None:
                raise ValueError(
                    "q_galore_target_modules must be set when using Q-GaLore optimizers."
                )
            if not isinstance(q_galore_target_modules, (list, str)):
                raise ValueError(
                    "q_galore_target_modules must be a string or list of strings."
                )

            if transformer is None:
                raise ValueError(
                    "You need to pass a model in order to initialize Q-GaLore optimizers."
                )

            trainable_param_list = extract_params(trainable_params)
            if not trainable_param_list:
                raise ValueError("No trainable parameters available for Q-GaLore.")
            trainable_param_ids = {id(p) for p in trainable_param_list}
            allow_non_trainable = bool(getattr(args, "q_galore_weight_quant", False))

            all_linear = _is_all_linear(q_galore_target_modules)
            q_galore_params: List[torch.nn.Parameter] = []

            for module_name, module in transformer.named_modules():
                matched, used_regex = _matches_target_module(
                    q_galore_target_modules, module_name
                )

                weight = _get_linear_weight(module)
                if weight is None:
                    if matched and not used_regex:
                        logger.warning(
                            "%s matched q_galore_target_modules but is not a Linear layer.",
                            module_name,
                        )
                    continue

                if not matched and not all_linear:
                    continue

                if id(weight) not in trainable_param_ids and not allow_non_trainable:
                    continue

                q_galore_params.append(weight)

            if not q_galore_params:
                raise ValueError(
                    "None of the q_galore_target_modules were found in trainable "
                    f"parameters ({q_galore_target_modules})."
                )

            def _pop_q_galore_setting(
                primary_key: str, fallback_key: str, default: Any
            ) -> Any:
                if primary_key in optimizer_kwargs:
                    return optimizer_kwargs.pop(primary_key)
                if fallback_key in optimizer_kwargs:
                    return optimizer_kwargs.pop(fallback_key)
                return default

            rank = int(_pop_q_galore_setting("q_galore_rank", "rank", 256))
            if rank <= 0:
                raise ValueError("q_galore_rank must be > 0")

            update_proj_gap = int(
                _pop_q_galore_setting("q_galore_update_proj_gap", "update_proj_gap", 200)
            )
            if update_proj_gap < 1:
                raise ValueError("q_galore_update_proj_gap must be >= 1")

            scale = float(_pop_q_galore_setting("q_galore_scale", "scale", 0.25))
            if scale <= 0:
                raise ValueError("q_galore_scale must be > 0")

            proj_type = str(
                _pop_q_galore_setting("q_galore_proj_type", "proj_type", "std")
            )

            quant_value = _pop_q_galore_setting("q_galore_quant", "quant", True)
            if isinstance(quant_value, str):
                quant_value = quant_value.strip().lower() in {"1", "true", "yes", "y"}
            quant = bool(quant_value)

            quant_n_bit = int(
                _pop_q_galore_setting("q_galore_quant_n_bit", "quant_n_bit", 4)
            )
            if quant_n_bit <= 0:
                raise ValueError("q_galore_quant_n_bit must be > 0")

            quant_group_size = int(
                _pop_q_galore_setting(
                    "q_galore_quant_group_size", "quant_group_size", 256
                )
            )
            if quant_group_size <= 0:
                raise ValueError("q_galore_quant_group_size must be > 0")

            cos_threshold = float(
                _pop_q_galore_setting("q_galore_cos_threshold", "cos_threshold", 0.4)
            )
            if not 0.0 <= cos_threshold <= 1.0:
                raise ValueError("q_galore_cos_threshold must be between 0 and 1")

            gamma_proj = float(
                _pop_q_galore_setting("q_galore_gamma_proj", "gamma_proj", 2)
            )
            if gamma_proj <= 0:
                raise ValueError("q_galore_gamma_proj must be > 0")

            queue_size = int(
                _pop_q_galore_setting("q_galore_queue_size", "queue_size", 5)
            )
            if queue_size <= 0:
                raise ValueError("q_galore_queue_size must be > 0")

            q_galore_optim_kwargs = {
                "rank": rank,
                "update_proj_gap": update_proj_gap,
                "scale": scale,
                "proj_type": proj_type,
                "quant": quant,
                "quant_n_bit": quant_n_bit,
                "quant_group_size": quant_group_size,
                "cos_threshold": cos_threshold,
                "gamma_proj": gamma_proj,
                "queue_size": queue_size,
            }

            q_galore_param_ids = {id(p) for p in q_galore_params}
            non_q_galore_params = [
                p for p in trainable_param_list if id(p) not in q_galore_param_ids
            ]
            param_groups: List[Dict[str, Any]] = []
            if non_q_galore_params:
                param_groups.append({"params": non_q_galore_params})
            param_groups.append({"params": q_galore_params, **q_galore_optim_kwargs})
            trainable_params = param_groups

            logger.info(
                "Q-GaLore enabled: target_modules=%s rank=%d update_proj_gap=%d "
                "scale=%.3f proj_type=%s quant=%s",
                q_galore_target_modules,
                rank,
                update_proj_gap,
                scale,
                proj_type,
                quant,
            )

        if is_galore_optimizer:
            param_name_map = {id(p): name for name, p in transformer.named_parameters()}
            trainable_params = OptimizerManager._apply_galore_param_groups(
                args, trainable_params, galore_settings, param_name_map
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

        elif optimizer_type in q_galore_types:
            logger.info(f"using QGaLoreAdamW8bit optimizer | {optimizer_kwargs}")
            try:
                from vendor.q_galore_torch.q_galore_adamw8bit import (
                    AdamW8bit as QGaLoreAdamW8bit,
                )
            except Exception as err:
                try:
                    from q_galore_torch import QGaLoreAdamW8bit
                except Exception as err2:
                    raise ImportError(
                        "QGaLoreAdamW8bit requires q-galore-torch and bitsandbytes. "
                        "Install with `pip install q-galore`."
                    ) from err2
            optimizer_class = QGaLoreAdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type in {"galoreadamw", "galore_adamw"}:
            logger.info(f"using GaLoreAdamW optimizer | {optimizer_kwargs}")
            try:
                from galore_torch import GaLoreAdamW
            except Exception as err:
                raise ImportError(
                    "GaLoreAdamW requires galore-torch. Install with `pip install galore-torch`."
                ) from err
            optimizer_class = GaLoreAdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type in {"galoreadamw8bit", "galore_adamw8bit"}:
            logger.info(f"using GaLoreAdamW8bit optimizer | {optimizer_kwargs}")
            try:
                from galore_torch import GaLoreAdamW8bit
            except Exception as err:
                raise ImportError(
                    "GaLoreAdamW8bit requires galore-torch and bitsandbytes."
                ) from err
            optimizer_class = GaLoreAdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type in {"galoreadafactor", "galore_adafactor"}:
            logger.info(f"using GaLoreAdafactor optimizer | {optimizer_kwargs}")
            try:
                from galore_torch import GaLoreAdafactor
            except Exception as err:
                raise ImportError(
                    "GaLoreAdafactor requires galore-torch. Install with `pip install galore-torch`."
                ) from err
            optimizer_class = GaLoreAdafactor
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "IVON".lower():
            logger.info(f"using IVON optimizer | {optimizer_kwargs}")

            from vendor.ivon.ivon import IVON

            # Allow users to pass standard Adam-style betas=(b1,b2)
            if (
                "betas" in optimizer_kwargs
                and "beta1" not in optimizer_kwargs
                and "beta2" not in optimizer_kwargs
            ):
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

        elif optimizer_type == "CAME8Bit".lower():
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

        elif optimizer_type == "RavenAdamW".lower():
            logger.info(f"using RavenAdamW optimizer | {optimizer_kwargs}")

            from optimizers.raven import RavenAdamW

            optimizer_class = RavenAdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "RavenAdamW8bit".lower():
            logger.info(f"using RavenAdamW8bit optimizer | {optimizer_kwargs}")

            try:
                from optimizers.raven_8bit import RavenAdamW8bit
            except Exception as err:
                raise ImportError(
                    "RavenAdamW8bit requires bitsandbytes. Please install it:\n"
                    "pip install bitsandbytes"
                ) from err

            optimizer_class = RavenAdamW8bit
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

        elif optimizer_type == "Adamuon".lower():
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
                    "No hidden weight parameters (?2D) found for AdaMuon. Consider using a different optimizer."
                )

            if len(hidden_gains_biases) == 0:
                logger.info(
                    "No bias/gain parameters (<2D) found for AdaMuon. Optimizer will only update matrix weights."
                )

            apply_adamuon_config_overrides(args, optimizer_kwargs)

            adamuon_lr = optimizer_kwargs.get(
                "adamuon_lr", optimizer_kwargs.get("muon_lr", 0.001)
            )
            adamuon_adam_lr = optimizer_kwargs.get(
                "adamuon_adam_lr", optimizer_kwargs.get("adam_lr", lr)
            )
            weight_decay = optimizer_kwargs.get(
                "adamuon_weight_decay", optimizer_kwargs.get("weight_decay", 0.001)
            )
            betas_value = optimizer_kwargs.get(
                "adamuon_betas", optimizer_kwargs.get("betas", (0.9, 0.95))
            )
            if isinstance(betas_value, list):
                betas = tuple(betas_value)
            else:
                betas = betas_value
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError(
                    f"AdaMuon auxiliary Adam betas must be a length-2 sequence. Received: {betas_value}"
                )
            betas = tuple(betas)

            momentum = optimizer_kwargs.get(
                "adamuon_momentum", optimizer_kwargs.get("momentum", 0.95)
            )
            beta2 = optimizer_kwargs.get("adamuon_beta2", 0.95)
            eps = optimizer_kwargs.get("adamuon_eps", 1e-8)
            ns_steps = optimizer_kwargs.get(
                "adamuon_ns_steps", optimizer_kwargs.get("ns_steps", 5)
            )
            scale_factor = optimizer_kwargs.get("adamuon_scale_factor", 0.2)
            nesterov = optimizer_kwargs.get("adamuon_nesterov", True)
            sign_stabilization = optimizer_kwargs.get(
                "adamuon_sign_stabilization", True
            )
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
                optimizer = optimizer_class(trainable_params, lr=lr, momentum=momentum)
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
            logger.info(
                f"using ScionLight optimizer (memory-efficient) | {optimizer_kwargs}"
            )

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
                optimizer = optimizer_class(trainable_params, lr=lr, momentum=momentum)
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
