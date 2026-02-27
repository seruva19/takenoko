"""QLoRA network entrypoint.

This module keeps adapter behavior identical to `networks.lora_wan` while
exposing a dedicated network_module for QLoRA-backed 4-bit base quantization.
"""

from __future__ import annotations

import ast
import re
import types
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from common.logger import get_logger
from networks import lora_wan
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES

logger = get_logger(__name__)


def _parse_string_list(raw: Any, key_name: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"{key_name} contains invalid list literal: {raw!r}") from exc
            return _parse_string_list(parsed, key_name)
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    raise ValueError(
        f"{key_name} must be list[str] or comma-separated string, got {type(raw).__name__}"
    )


def _parse_match_mode(raw: Any, key_name: str) -> str:
    mode = str(raw if raw is not None else "exact").strip().lower()
    if mode not in {"contains", "regex", "exact"}:
        raise ValueError(
            f"{key_name} must be one of ['contains', 'regex', 'exact'], got {mode!r}"
        )
    return mode


def _parse_optional_float(raw: Any, key_name: str) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key_name} must be a float or null, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{key_name} must be > 0 when provided, got {value}")
    return value


def _match_name(full_name: str, patterns: List[str], mode: str) -> bool:
    if not patterns:
        return False
    name = full_name.strip()
    lowered_name = name.lower()
    lowered_patterns = [p.lower() for p in patterns]

    if mode == "contains":
        return any(pattern in lowered_name for pattern in lowered_patterns)
    if mode == "exact":
        return lowered_name in set(lowered_patterns)
    # regex
    for pattern in patterns:
        try:
            if re.search(pattern, name) is not None:
                return True
        except re.error as exc:
            raise ValueError(f"Invalid regex in modules_to_save pattern {pattern!r}: {exc}") from exc
    return False


def _sanitize_alias_key(name: str) -> str:
    return (
        name.replace(".", "__")
        .replace("[", "_")
        .replace("]", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
    )


def _collect_modules_to_save(
    unet: nn.Module,
    patterns: List[str],
    match_mode: str,
) -> List[Tuple[str, nn.Module]]:
    selected: List[Tuple[str, nn.Module]] = []
    for module_name, module in unet.named_modules():
        if not module_name:
            continue
        if _match_name(module_name, patterns, match_mode):
            selected.append((module_name, module))
    return selected


def _attach_modules_to_save(
    network: LoRANetwork,
    unet: Optional[nn.Module],
    kwargs: dict[str, Any],
) -> None:
    if unet is None:
        return

    raw_patterns = kwargs.get("qlora_modules_to_save", [])
    patterns = _parse_string_list(raw_patterns, "qlora_modules_to_save")
    if not patterns:
        return

    match_mode = _parse_match_mode(
        kwargs.get("qlora_modules_to_save_match", "exact"),
        "qlora_modules_to_save_match",
    )
    modules_to_save_lr = _parse_optional_float(
        kwargs.get("qlora_modules_to_save_lr", None),
        "qlora_modules_to_save_lr",
    )

    selected_modules = _collect_modules_to_save(unet, patterns, match_mode)
    if not selected_modules:
        raise ValueError(
            "qlora_modules_to_save did not match any modules in the WAN transformer. "
            f"patterns={patterns}, match_mode={match_mode}"
        )

    selected_param_aliases: List[str] = []
    selected_params: List[nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for module_name, module in selected_modules:
        for param_name, param in module.named_parameters(recurse=True):
            if not param.requires_grad:
                param.requires_grad_(True)
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            selected_params.append(param)

            alias_name = (
                f"modules_to_save__{_sanitize_alias_key(module_name)}__"
                f"{_sanitize_alias_key(param_name)}"
            )
            selected_param_aliases.append(alias_name)
            if alias_name not in network._parameters:
                network.register_parameter(alias_name, param)

    if not selected_params:
        raise ValueError(
            "qlora_modules_to_save matched modules but found no parameters to optimize."
        )

    setattr(network, "qlora_modules_to_save_patterns", patterns)
    setattr(network, "qlora_modules_to_save_match", match_mode)
    setattr(network, "qlora_modules_to_save_lr", modules_to_save_lr)
    setattr(
        network,
        "qlora_modules_to_save_selected_modules",
        [name for name, _ in selected_modules],
    )
    setattr(network, "qlora_modules_to_save_param_aliases", selected_param_aliases)
    setattr(network, "qlora_modules_to_save_params", selected_params)

    original_prepare = network.prepare_optimizer_params

    def _prepare_optimizer_params_with_modules_to_save(
        self: LoRANetwork,
        unet_lr: float = 1e-4,
        input_lr_scale: float = 1.0,
        **prep_kwargs: Any,
    ):
        all_params, lr_descriptions = original_prepare(
            unet_lr=unet_lr,
            input_lr_scale=input_lr_scale,
            **prep_kwargs,
        )
        extra_params: List[nn.Parameter] = []
        seen_ids: set[int] = set()
        for group in all_params:
            params_obj = group.get("params", [])
            for param in params_obj:
                seen_ids.add(id(param))

        for param in getattr(self, "qlora_modules_to_save_params", []):
            if not isinstance(param, nn.Parameter):
                continue
            if not param.requires_grad:
                continue
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            extra_params.append(param)

        if extra_params:
            modules_lr = getattr(self, "qlora_modules_to_save_lr", None)
            param_group = {
                "params": extra_params,
                "lr": float(unet_lr if modules_lr is None else modules_lr),
            }
            all_params.append(param_group)
            lr_descriptions.append("unet modules_to_save")

        return all_params, lr_descriptions

    network.prepare_optimizer_params = types.MethodType(  # type: ignore[method-assign]
        _prepare_optimizer_params_with_modules_to_save,
        network,
    )

    logger.info(
        "QLoRA modules_to_save enabled: matched_modules=%s, trainable_params=%s, match_mode=%s, modules_lr=%s",
        len(selected_modules),
        len(selected_params),
        match_mode,
        modules_to_save_lr,
    )


def _resolve_unet(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[nn.Module]:
    if "unet" in kwargs and isinstance(kwargs["unet"], nn.Module):
        return kwargs["unet"]
    # create_arch_network/create_network call path passes unet as the 6th positional arg.
    if len(args) >= 6 and isinstance(args[5], nn.Module):
        return args[5]
    return None


def create_arch_network(*args: Any, **kwargs: Any) -> LoRANetwork:
    network = lora_wan.create_arch_network(*args, **kwargs)
    _attach_modules_to_save(network, _resolve_unet(args, kwargs), kwargs)
    return network


def create_network(*args: Any, **kwargs: Any) -> LoRANetwork:
    network = lora_wan.create_network(*args, **kwargs)
    _attach_modules_to_save(network, _resolve_unet(args, kwargs), kwargs)
    return network


def create_arch_network_from_weights(*args: Any, **kwargs: Any) -> Any:
    network = lora_wan.create_arch_network_from_weights(*args, **kwargs)
    if isinstance(network, LoRANetwork):
        _attach_modules_to_save(network, _resolve_unet(args, kwargs), kwargs)
    return network


def create_network_from_weights(*args: Any, **kwargs: Any) -> Any:
    return lora_wan.create_network_from_weights(*args, **kwargs)
