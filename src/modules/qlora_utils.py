"""QLoRA 4-bit linear replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Pattern, Tuple

import torch
import torch.nn as nn


_DTYPE_ALIASES = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class QLoRASettings:
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    quant_storage: str = "uint8"
    target_modules: Optional[List[str]] = None
    skip_modules: Optional[List[str]] = None
    target_match_mode: str = "contains"
    skip_match_mode: str = "contains"
    upcast_layernorm: bool = True
    layernorm_patterns: Optional[List[str]] = None
    keep_in_fp32_modules: Optional[List[str]] = None
    fail_on_replacement_error: bool = False
    min_replaced_modules: int = 1


@dataclass
class QLoRAApplyResult:
    replaced_count: int
    skipped_count: int
    failed: List[Tuple[str, str]]
    replaced_modules: List[str]
    upcast_layernorm_count: int
    upcast_fp32_module_count: int


def _normalize_patterns(raw: Optional[List[str]]) -> List[str]:
    if raw is None:
        return []
    return [str(chunk).strip() for chunk in raw if str(chunk).strip()]


def _to_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype alias {name!r}. Expected one of {sorted(_DTYPE_ALIASES)}."
        )
    return _DTYPE_ALIASES[normalized]


def _compile_patterns(patterns: List[str]) -> List[Pattern[str]]:
    compiled: List[Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc
    return compiled


def _match_name(
    full_name: str,
    patterns: List[str],
    mode: str,
    compiled_patterns: Optional[List[Pattern[str]]] = None,
) -> bool:
    if not patterns:
        return False
    name = full_name.strip()
    lowered_name = name.lower()
    normalized_mode = str(mode).strip().lower()

    if normalized_mode == "contains":
        return any(pattern.lower() in lowered_name for pattern in patterns)
    if normalized_mode == "exact":
        lowered_patterns = {pattern.lower() for pattern in patterns}
        return lowered_name in lowered_patterns
    if normalized_mode == "regex":
        if compiled_patterns is None:
            compiled_patterns = _compile_patterns(patterns)
        return any(regex.search(name) is not None for regex in compiled_patterns)
    raise ValueError(
        f"Unsupported match mode {mode!r}. Expected one of ['contains', 'exact', 'regex']."
    )


def _should_replace(
    full_name: str,
    include: List[str],
    exclude: List[str],
    include_mode: str,
    exclude_mode: str,
    include_regex: Optional[List[Pattern[str]]] = None,
    exclude_regex: Optional[List[Pattern[str]]] = None,
) -> bool:
    if include:
        lowered_tokens = {token.lower() for token in include}
        include_all_linear = "all-linear" in lowered_tokens or "all_linear" in lowered_tokens
        if not include_all_linear and not _match_name(
            full_name,
            include,
            include_mode,
            include_regex,
        ):
            return False

    if not exclude:
        return True
    return not _match_name(full_name, exclude, exclude_mode, exclude_regex)


def _is_quantizable_linear(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    return bool(
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and callable(getattr(module, "state_dict", None))
    )


def _upcast_module_to_fp32(module: nn.Module) -> bool:
    casted = False
    for param in module.parameters(recurse=False):
        if param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            casted = True
    for name, buffer in module.named_buffers(recurse=False):
        if torch.is_floating_point(buffer) and buffer.dtype != torch.float32:
            module.register_buffer(name, buffer.to(torch.float32), persistent=True)
            casted = True
    return casted


def _prepare_model_for_kbit_training(
    model: nn.Module,
    settings: QLoRASettings,
    logger: object,
) -> Tuple[int, int]:
    upcast_layernorm_count = 0
    upcast_fp32_module_count = 0

    normalized_keep_patterns = _normalize_patterns(settings.keep_in_fp32_modules)
    normalized_norm_patterns = _normalize_patterns(settings.layernorm_patterns)

    for module_name, module in model.named_modules():
        is_norm_module = (
            isinstance(module, (nn.LayerNorm, nn.GroupNorm))
            or "norm" in module.__class__.__name__.lower()
        )
        if settings.upcast_layernorm and is_norm_module:
            if not normalized_norm_patterns or _match_name(
                module_name,
                normalized_norm_patterns,
                "contains",
            ):
                if _upcast_module_to_fp32(module):
                    upcast_layernorm_count += 1

        if normalized_keep_patterns and _match_name(
            module_name,
            normalized_keep_patterns,
            "contains",
        ):
            if _upcast_module_to_fp32(module):
                upcast_fp32_module_count += 1

    if upcast_layernorm_count > 0:
        logger.info(
            "QLoRA prep: upcasted %s normalization modules to fp32.",
            upcast_layernorm_count,
        )
    if upcast_fp32_module_count > 0:
        logger.info(
            "QLoRA prep: upcasted %s user-targeted modules to fp32.",
            upcast_fp32_module_count,
        )

    return upcast_layernorm_count, upcast_fp32_module_count


def _make_linear4bit(
    linear: nn.Module,
    settings: QLoRASettings,
    bnb: object,
) -> nn.Module:
    in_features = int(getattr(linear, "in_features"))
    out_features = int(getattr(linear, "out_features"))
    has_bias = getattr(linear, "bias", None) is not None
    quant_storage = _to_dtype(settings.quant_storage)
    compute_dtype = _to_dtype(settings.compute_dtype)
    try:
        new_layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            compress_statistics=bool(settings.use_double_quant),
            quant_type=str(settings.quant_type),
            quant_storage=quant_storage,
        )
    except TypeError:
        new_layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            compress_statistics=bool(settings.use_double_quant),
            quant_type=str(settings.quant_type),
        )
    state_dict = linear.state_dict()
    new_layer.load_state_dict(state_dict, strict=False)
    new_layer = new_layer.to(device=getattr(linear, "weight").device)
    new_layer.weight.requires_grad = False
    if getattr(new_layer, "bias", None) is not None:
        new_layer.bias.requires_grad = False
    return new_layer


def apply_qlora_quantization(
    model: nn.Module,
    settings: QLoRASettings,
    logger: object,
) -> QLoRAApplyResult:
    include = _normalize_patterns(settings.target_modules)
    exclude = _normalize_patterns(settings.skip_modules)
    include_mode = str(settings.target_match_mode).strip().lower()
    exclude_mode = str(settings.skip_match_mode).strip().lower()
    include_regex = _compile_patterns(include) if include_mode == "regex" else None
    exclude_regex = _compile_patterns(exclude) if exclude_mode == "regex" else None

    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "QLoRA requires bitsandbytes. Install bitsandbytes and ensure CUDA is available."
        ) from exc

    replaced_modules: List[str] = []
    failed: List[Tuple[str, str]] = []
    skipped_count = 0
    upcast_layernorm_count, upcast_fp32_module_count = _prepare_model_for_kbit_training(
        model,
        settings,
        logger,
    )

    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if not _is_quantizable_linear(child):
                continue

            if not _should_replace(
                full_name,
                include,
                exclude,
                include_mode,
                exclude_mode,
                include_regex,
                exclude_regex,
            ):
                skipped_count += 1
                continue

            try:
                quantized_layer = _make_linear4bit(child, settings, bnb)
                setattr(module, child_name, quantized_layer)
                replaced_modules.append(full_name)
            except Exception as exc:
                failed.append((full_name, str(exc)))

    min_replaced = max(1, int(settings.min_replaced_modules))
    if len(replaced_modules) < min_replaced:
        raise RuntimeError(
            "QLoRA replaced fewer modules than required "
            f"(replaced={len(replaced_modules)}, required={min_replaced}). "
            "Adjust qlora_target_modules / qlora_skip_modules for this architecture."
        )

    if failed:
        if bool(settings.fail_on_replacement_error):
            preview = failed[0] if failed else ("<none>", "<none>")
            raise RuntimeError(
                "QLoRA replacement encountered failures and strict mode is enabled. "
                f"First failure: {preview}"
            )
        logger.warning(
            "QLoRA replaced %s linear layers with %s failures. First failure: %s",
            len(replaced_modules),
            len(failed),
            failed[0],
        )
    else:
        logger.info("QLoRA replaced %s linear layers.", len(replaced_modules))

    return QLoRAApplyResult(
        replaced_count=len(replaced_modules),
        skipped_count=skipped_count,
        failed=failed,
        replaced_modules=replaced_modules,
        upcast_layernorm_count=upcast_layernorm_count,
        upcast_fp32_module_count=upcast_fp32_module_count,
    )
