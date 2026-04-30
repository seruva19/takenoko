"""Configuration parsing for BAdam block-coordinate optimization."""

from __future__ import annotations

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
        raise ValueError("badam_block_prefixes must be a list")
    normalized: list[str | list[str]] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, list) and all(isinstance(prefix, str) for prefix in item):
            normalized.append(list(item))
        else:
            raise ValueError(
                "badam_block_prefixes entries must be strings or lists of strings"
            )
    return normalized


def apply_badam_config(args: Any, config: dict[str, Any], logger: Any) -> Any:
    """Parse BAdam optimizer-wrapper settings into ``args``."""

    optimizer_type = str(getattr(args, "optimizer_type", "") or "")
    optimizer_type_lower = optimizer_type.lower()
    optimizer_requests_badam = optimizer_type_lower in _BADAM_ALIASES
    optimizer_requests_badam_ratio = optimizer_type_lower in _BADAM_RATIO_ALIASES

    args.badam_enabled = optimizer_requests_badam or optimizer_requests_badam_ratio or _as_bool(
        config.get("badam_enabled", False)
    )
    args.badam_ratio_mode = optimizer_requests_badam_ratio or _as_bool(
        config.get("badam_ratio_mode", False)
    )
    args.badam_base_optimizer = str(config.get("badam_base_optimizer", "AdamW"))
    if args.badam_base_optimizer.lower() in _BADAM_ALIASES:
        raise ValueError("badam_base_optimizer must name the wrapped optimizer, not BAdam")

    args.badam_switch_block_every = int(
        config.get("badam_switch_block_every", 100)
    )
    if args.badam_switch_block_every < 1:
        raise ValueError("badam_switch_block_every must be >= 1")

    args.badam_switch_mode = str(
        config.get("badam_switch_mode", "random")
    ).lower()
    if args.badam_switch_mode not in _BADAM_SWITCH_MODES:
        raise ValueError(
            "badam_switch_mode must be one of "
            f"{sorted(_BADAM_SWITCH_MODES)}, got {args.badam_switch_mode!r}"
        )

    start_block = config.get("badam_start_block")
    args.badam_start_block = int(start_block) if start_block is not None else None
    if args.badam_start_block is not None and args.badam_start_block < 0:
        raise ValueError("badam_start_block must be >= 0 when set")

    args.badam_block_prefix_mode = str(
        config.get("badam_block_prefix_mode", "wan_blocks")
    ).lower()
    if args.badam_block_prefix_mode not in _BADAM_PREFIX_MODES:
        raise ValueError(
            "badam_block_prefix_mode must be one of "
            f"{sorted(_BADAM_PREFIX_MODES)}, got {args.badam_block_prefix_mode!r}"
        )

    args.badam_block_prefixes = _validate_block_prefixes(
        config.get("badam_block_prefixes", [])
    )
    if args.badam_block_prefix_mode == "custom" and not args.badam_block_prefixes:
        raise ValueError(
            "badam_block_prefix_mode='custom' requires badam_block_prefixes"
        )

    args.badam_include_non_block = _as_bool(
        config.get("badam_include_non_block", True)
    )
    args.badam_include_embedding = _as_bool(
        config.get("badam_include_embedding", False)
    )
    args.badam_include_lm_head = _as_bool(
        config.get("badam_include_lm_head", False)
    )
    args.badam_always_active_prefixes = _as_string_list(
        config.get("badam_always_active_prefixes", []),
        name="badam_always_active_prefixes",
    )
    args.badam_active_modules = _as_string_list(
        config.get("badam_active_modules", []),
        name="badam_active_modules",
    )
    args.badam_use_fp32_active_copy = _as_bool(
        config.get("badam_use_fp32_active_copy", True)
    )
    args.badam_purge_inactive_state = _as_bool(
        config.get("badam_purge_inactive_state", True)
    )
    args.badam_reset_state_on_switch = _as_bool(
        config.get("badam_reset_state_on_switch", True)
    )
    args.badam_use_gradient_release = _as_bool(
        config.get("badam_use_gradient_release", False)
    )
    if args.badam_use_gradient_release and not args.badam_use_fp32_active_copy:
        raise ValueError(
            "badam_use_gradient_release=true requires badam_use_fp32_active_copy=true; "
            "the per-parameter step path operates on HP fp32 copies."
        )
    args.badam_update_ratio = float(config.get("badam_update_ratio", 0.1))
    if not 0.0 < args.badam_update_ratio <= 1.0:
        raise ValueError("badam_update_ratio must be in (0, 1]")
    args.badam_ratio_mask_mode = str(
        config.get("badam_ratio_mask_mode", "adjacent")
    ).lower()
    if args.badam_ratio_mask_mode not in _BADAM_RATIO_MASK_MODES:
        raise ValueError(
            "badam_ratio_mask_mode must be one of "
            f"{sorted(_BADAM_RATIO_MASK_MODES)}, got {args.badam_ratio_mask_mode!r}"
        )
    args.badam_ratio_preserve_threshold = int(
        config.get("badam_ratio_preserve_threshold", 100)
    )
    if args.badam_ratio_preserve_threshold < 0:
        raise ValueError("badam_ratio_preserve_threshold must be >= 0")
    args.badam_ratio_keep_mask = _as_bool(
        config.get("badam_ratio_keep_mask", True)
    )
    args.badam_allow_distributed = _as_bool(
        config.get("badam_allow_distributed", False)
    )
    args.badam_allow_unmatched_params = _as_bool(
        config.get("badam_allow_unmatched_params", False)
    )
    args.badam_verbose = int(config.get("badam_verbose", 1))
    if not 0 <= args.badam_verbose <= 2:
        raise ValueError("badam_verbose must be 0, 1, or 2")

    if args.badam_enabled and logger is not None:
        logger.info(
            "BAdam enabled (base=%s, switch_every=%d, mode=%s, prefix_mode=%s).",
            args.badam_base_optimizer if optimizer_requests_badam else optimizer_type,
            args.badam_switch_block_every,
            args.badam_switch_mode,
            args.badam_block_prefix_mode,
        )

    return args
