"""GaLore optimizer creation helpers for WAN network trainer."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import fnmatch
import re

import torch


def prepare_galore_trainable_params(
    args: Any,
    transformer: torch.nn.Module,
    trainable_params: List[Any],
    optimizer_kwargs: Dict[str, Any],
    optimizer_type: str,
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    logger: Any,
) -> Tuple[List[Any], Dict[str, Any]]:
    galore_types = {
        "galoreadamw",
        "galore_adamw",
        "galoreadamw8bit",
        "galore_adamw8bit",
        "galoreadafactor",
        "galore_adafactor",
    }
    q_galore_types = {
        "qgaloreadamw8bit",
        "qgalore_adamw8bit",
        "q_galore_adamw8bit",
        "qgaloreadamw8bitlayerwise",
        "qgalore_adamw8bit_layerwise",
        "q_galore_adamw8bit_layerwise",
    }
    is_galore_optimizer = optimizer_type in galore_types
    is_q_galore_optimizer = optimizer_type in q_galore_types

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

    if is_q_galore_optimizer:
        trainable_params = _apply_q_galore_param_groups(
            args,
            transformer,
            trainable_params,
            optimizer_kwargs,
            optimizer_type,
            extract_params,
            logger,
        )

    if is_galore_optimizer:
        param_name_map = {id(p): name for name, p in transformer.named_parameters()}
        trainable_params = _apply_galore_param_groups(
            trainable_params, galore_settings, param_name_map, logger
        )

    return trainable_params, optimizer_kwargs


def is_q_galore_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type in {
        "qgaloreadamw8bit",
        "qgalore_adamw8bit",
        "q_galore_adamw8bit",
        "qgaloreadamw8bitlayerwise",
        "qgalore_adamw8bit_layerwise",
        "q_galore_adamw8bit_layerwise",
    }


def create_galore_adamw_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using GaLoreAdamW optimizer | {optimizer_kwargs}")
    try:
        from galore_torch import GaLoreAdamW
    except Exception as err:
        raise ImportError(
            "GaLoreAdamW requires galore-torch. Install with `pip install galore-torch`."
        ) from err
    optimizer_class = GaLoreAdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_galore_adamw8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using GaLoreAdamW8bit optimizer | {optimizer_kwargs}")
    try:
        from galore_torch import GaLoreAdamW8bit
    except Exception as err:
        raise ImportError("GaLoreAdamW8bit requires galore-torch and bitsandbytes.") from err
    optimizer_class = GaLoreAdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def create_galore_adafactor_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using GaLoreAdafactor optimizer | {optimizer_kwargs}")
    try:
        from galore_torch import GaLoreAdafactor
    except Exception as err:
        raise ImportError(
            "GaLoreAdafactor requires galore-torch. Install with `pip install galore-torch`."
        ) from err
    optimizer_class = GaLoreAdafactor
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _apply_galore_param_groups(
    trainable_params: List[Any],
    galore_settings: Dict[str, Any],
    param_name_map: Optional[Dict[int, str]],
    logger: Any,
) -> List[Any]:
    apply_to = str(galore_settings.get("galore_apply_to", "matrix_only")).lower()
    if apply_to not in {"matrix_only", "matrix_and_tensor"}:
        raise ValueError("galore_apply_to must be 'matrix_only' or 'matrix_and_tensor'")

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
        logger.info("GaLore: using existing parameter groups that already define rank.")
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
        raise ValueError(f"galore_proj_type must be one of {sorted(allowed_proj_types)}")

    group_by = str(galore_settings.get("galore_group_by", "none")).lower()
    allowed_group_by = {"none", "layer", "param"}
    if group_by not in allowed_group_by:
        raise ValueError(f"galore_group_by must be one of {sorted(allowed_group_by)}")

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
                "update_proj_gap": group_settings.get("update_proj_gap", update_proj_gap),
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


def _apply_q_galore_param_groups(
    args: Any,
    transformer: torch.nn.Module,
    trainable_params: List[Any],
    optimizer_kwargs: Dict[str, Any],
    optimizer_type: str,
    extract_params: Callable[[List[Any]], List[torch.nn.Parameter]],
    logger: Any,
) -> List[Any]:
    if optimizer_type.endswith("layerwise"):
        raise NotImplementedError(
            "Layer-wise Q-GaLore is not supported in Takenoko yet. "
            "Use optimizer_type='q_galore_adamw8bit' instead."
        )

    q_galore_target_modules = getattr(args, "q_galore_target_modules", None)
    if q_galore_target_modules is None:
        q_galore_target_modules = optimizer_kwargs.pop("q_galore_target_modules", None)
    if q_galore_target_modules is None:
        raise ValueError(
            "q_galore_target_modules must be set when using Q-GaLore optimizers."
        )
    if not isinstance(q_galore_target_modules, (list, str)):
        raise ValueError("q_galore_target_modules must be a string or list of strings.")

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
        matched, used_regex = _matches_target_module(q_galore_target_modules, module_name, logger)

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
            "None of the q_galore_target_modules were found in trainable parameters "
            f"({q_galore_target_modules})."
        )

    def _pop_q_galore_setting(primary_key: str, fallback_key: str, default: Any) -> Any:
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

    proj_type = str(_pop_q_galore_setting("q_galore_proj_type", "proj_type", "std"))

    quant_value = _pop_q_galore_setting("q_galore_quant", "quant", True)
    if isinstance(quant_value, str):
        quant_value = quant_value.strip().lower() in {"1", "true", "yes", "y"}
    quant = bool(quant_value)

    quant_n_bit = int(_pop_q_galore_setting("q_galore_quant_n_bit", "quant_n_bit", 4))
    if quant_n_bit <= 0:
        raise ValueError("q_galore_quant_n_bit must be > 0")

    quant_group_size = int(
        _pop_q_galore_setting("q_galore_quant_group_size", "quant_group_size", 256)
    )
    if quant_group_size <= 0:
        raise ValueError("q_galore_quant_group_size must be > 0")

    cos_threshold = float(
        _pop_q_galore_setting("q_galore_cos_threshold", "cos_threshold", 0.4)
    )
    if not 0.0 <= cos_threshold <= 1.0:
        raise ValueError("q_galore_cos_threshold must be between 0 and 1")

    gamma_proj = float(_pop_q_galore_setting("q_galore_gamma_proj", "gamma_proj", 2))
    if gamma_proj <= 0:
        raise ValueError("q_galore_gamma_proj must be > 0")

    queue_size = int(_pop_q_galore_setting("q_galore_queue_size", "queue_size", 5))
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
        "Q-GaLore enabled: target_modules=%s rank=%d update_proj_gap=%d scale=%.3f proj_type=%s quant=%s",
        q_galore_target_modules,
        rank,
        update_proj_gap,
        scale,
        proj_type,
        quant,
    )

    return trainable_params


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
    logger: Any,
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
                    logger.warning("Invalid regex in q_galore_target_modules: %s", target)
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
