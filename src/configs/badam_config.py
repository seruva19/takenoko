"""Configuration parsing for BAdam block-coordinate optimization.

BAdam wrapper kwargs are passed via ``optimizer_args`` (a list of ``key=value``
strings, mirroring the way other optimizers receive their kwargs). The wrapped
base optimizer's own kwargs are passed via the separate ``base_optimizer_args``
list. The wrapper's ``base_optimizer_type`` key inside ``optimizer_args``
selects which optimizer to wrap.

When ``optimizer_type`` is not BAdam, ``optimizer_args`` keeps its conventional
meaning (kwargs for the named optimizer) and ``base_optimizer_args`` is unused.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable


_BADAM_ALIASES = {"badam", "blockadam", "block_optimizer", "blockoptimizer"}
_BADAM_RATIO_ALIASES = {"badamratio", "badam_ratio", "blockadamratio"}
_BADAM_SWITCH_MODES = {"random", "ascending", "descending", "fixed"}
_BADAM_PREFIX_MODES = {"auto", "wan_blocks", "custom"}
_BADAM_RATIO_MASK_MODES = {"adjacent", "scatter"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_string_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a string or list of strings")
    parsed = list(value)
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{name} must contain only strings")
    return parsed


def _validate_block_prefixes(value: Any) -> list[str | list[str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("block_prefixes must be a list")
    normalized: list[str | list[str]] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, list) and all(isinstance(prefix, str) for prefix in item):
            normalized.append(list(item))
        else:
            raise ValueError(
                "block_prefixes entries must be strings or lists of strings"
            )
    return normalized


def _parse_optimizer_arg_value(value: str) -> Any:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return normalized.strip("'\"")


def _parse_kv_list(items: Iterable[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in items or []:
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError(
                f"optimizer_args entries must be 'key=value' strings, got {entry!r}"
            )
        key, _, value = entry.partition("=")
        out[key.strip()] = _parse_optimizer_arg_value(value)
    return out


def _pop_int(kwargs: dict[str, Any], key: str, default: int) -> int:
    if key not in kwargs:
        return default
    return int(kwargs.pop(key))


def _pop_float(kwargs: dict[str, Any], key: str, default: float) -> float:
    if key not in kwargs:
        return default
    return float(kwargs.pop(key))


def _pop_bool(kwargs: dict[str, Any], key: str, default: bool) -> bool:
    if key not in kwargs:
        return default
    return _as_bool(kwargs.pop(key))


def _pop_str(kwargs: dict[str, Any], key: str, default: str) -> str:
    if key not in kwargs:
        return default
    return str(kwargs.pop(key))


def apply_badam_config(args: Any, config: dict[str, Any], logger: Any) -> Any:
    """Parse BAdam wrapper settings from ``args.optimizer_args``.

    On the BAdam path, ``args.optimizer_args`` is treated as wrapper kwargs:
    the recognized keys populate ``args.badam_*`` attributes and the rest are
    rejected. ``args.base_optimizer_args`` (read here from ``config``) becomes
    the new ``args.optimizer_args`` so the downstream optimizer factory builds
    the wrapped base optimizer with its own kwargs.
    """

    optimizer_type = str(getattr(args, "optimizer_type", "") or "")
    optimizer_type_lower = optimizer_type.lower()
    optimizer_requests_badam = optimizer_type_lower in _BADAM_ALIASES
    optimizer_requests_badam_ratio = optimizer_type_lower in _BADAM_RATIO_ALIASES

    args.badam_enabled = optimizer_requests_badam or optimizer_requests_badam_ratio
    args.badam_ratio_mode = optimizer_requests_badam_ratio

    # When invoked directly with a config dict (e.g. tests), pick up base_optimizer_args
    # from the config; the surrounding config_parser.py already mirrors this onto args
    # before reaching us, so this is a no-op on the normal path.
    if "base_optimizer_args" in config:
        args.base_optimizer_args = list(config.get("base_optimizer_args") or [])
    elif not hasattr(args, "base_optimizer_args"):
        args.base_optimizer_args = []

    if not args.badam_enabled:
        return args

    wrapper_kwargs = _parse_kv_list(getattr(args, "optimizer_args", None))

    args.badam_base_optimizer = _pop_str(wrapper_kwargs, "base_optimizer_type", "AdamW")
    if args.badam_base_optimizer.lower() in _BADAM_ALIASES:
        raise ValueError(
            "base_optimizer_type must name the wrapped optimizer, not BAdam"
        )

    args.badam_switch_block_every = _pop_int(wrapper_kwargs, "switch_block_every", 100)
    if args.badam_switch_block_every < 1:
        raise ValueError("switch_block_every must be >= 1")

    args.badam_switch_mode = _pop_str(wrapper_kwargs, "switch_mode", "random").lower()
    if args.badam_switch_mode not in _BADAM_SWITCH_MODES:
        raise ValueError(
            f"switch_mode must be one of {sorted(_BADAM_SWITCH_MODES)}, "
            f"got {args.badam_switch_mode!r}"
        )

    if "start_block" in wrapper_kwargs:
        raw = wrapper_kwargs.pop("start_block")
        args.badam_start_block = int(raw) if raw is not None else None
    else:
        args.badam_start_block = None
    if args.badam_start_block is not None and args.badam_start_block < 0:
        raise ValueError("start_block must be >= 0 when set")

    args.badam_block_prefix_mode = _pop_str(
        wrapper_kwargs, "block_prefix_mode", "wan_blocks"
    ).lower()
    if args.badam_block_prefix_mode not in _BADAM_PREFIX_MODES:
        raise ValueError(
            f"block_prefix_mode must be one of {sorted(_BADAM_PREFIX_MODES)}, "
            f"got {args.badam_block_prefix_mode!r}"
        )

    args.badam_block_prefixes = _validate_block_prefixes(
        wrapper_kwargs.pop("block_prefixes", [])
    )
    if args.badam_block_prefix_mode == "custom" and not args.badam_block_prefixes:
        raise ValueError("block_prefix_mode='custom' requires block_prefixes")

    args.badam_include_non_block = _pop_bool(wrapper_kwargs, "include_non_block", True)
    args.badam_include_embedding = _pop_bool(wrapper_kwargs, "include_embedding", False)
    args.badam_include_lm_head = _pop_bool(wrapper_kwargs, "include_lm_head", False)
    args.badam_always_active_prefixes = _as_string_list(
        wrapper_kwargs.pop("always_active_prefixes", []),
        name="always_active_prefixes",
    )
    args.badam_active_modules = _as_string_list(
        wrapper_kwargs.pop("active_modules", []),
        name="active_modules",
    )
    args.badam_use_fp32_active_copy = _pop_bool(
        wrapper_kwargs, "use_fp32_active_copy", True
    )
    args.badam_purge_inactive_state = _pop_bool(
        wrapper_kwargs, "purge_inactive_state", True
    )
    args.badam_reset_state_on_switch = _pop_bool(
        wrapper_kwargs, "reset_state_on_switch", True
    )
    args.badam_use_gradient_release = _pop_bool(
        wrapper_kwargs, "use_gradient_release", False
    )
    if args.badam_use_gradient_release and not args.badam_use_fp32_active_copy:
        raise ValueError(
            "use_gradient_release=true requires use_fp32_active_copy=true; "
            "the per-parameter step path operates on HP fp32 copies."
        )

    args.badam_update_ratio = _pop_float(wrapper_kwargs, "update_ratio", 0.1)
    if not 0.0 < args.badam_update_ratio <= 1.0:
        raise ValueError("update_ratio must be in (0, 1]")

    args.badam_ratio_mask_mode = _pop_str(
        wrapper_kwargs, "ratio_mask_mode", "adjacent"
    ).lower()
    if args.badam_ratio_mask_mode not in _BADAM_RATIO_MASK_MODES:
        raise ValueError(
            f"ratio_mask_mode must be one of {sorted(_BADAM_RATIO_MASK_MODES)}, "
            f"got {args.badam_ratio_mask_mode!r}"
        )

    args.badam_ratio_preserve_threshold = _pop_int(
        wrapper_kwargs, "ratio_preserve_threshold", 100
    )
    if args.badam_ratio_preserve_threshold < 0:
        raise ValueError("ratio_preserve_threshold must be >= 0")

    args.badam_ratio_keep_mask = _pop_bool(wrapper_kwargs, "ratio_keep_mask", True)
    args.badam_allow_distributed = _pop_bool(wrapper_kwargs, "allow_distributed", False)
    args.badam_allow_unmatched_params = _pop_bool(
        wrapper_kwargs, "allow_unmatched_params", False
    )
    args.badam_verbose = _pop_int(wrapper_kwargs, "verbose", 1)
    if not 0 <= args.badam_verbose <= 2:
        raise ValueError("verbose must be 0, 1, or 2")

    if wrapper_kwargs:
        raise ValueError(
            f"Unknown optimizer_args keys for BAdam: {sorted(wrapper_kwargs)}"
        )

    # Hand the base optimizer kwargs down to the standard optimizer factory by
    # repurposing args.optimizer_args. The original wrapper kwargs are now fully
    # absorbed into args.badam_* attributes.
    args.optimizer_args = list(getattr(args, "base_optimizer_args", []) or [])

    if logger is not None:
        logger.info(
            "BAdam enabled (base=%s, switch_every=%d, mode=%s, prefix_mode=%s).",
            args.badam_base_optimizer,
            args.badam_switch_block_every,
            args.badam_switch_mode,
            args.badam_block_prefix_mode,
        )

    return args
