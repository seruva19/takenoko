from __future__ import annotations

import logging
from typing import Any, Dict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def _parse_int_map_string(
    value: str, *, item_sep: str = ";", kv_sep: str = ":", list_sep: str = ","
) -> Dict[int, list[int]]:
    result: Dict[int, list[int]] = {}
    for chunk in value.split(item_sep):
        chunk = chunk.strip()
        if not chunk:
            continue
        if kv_sep not in chunk:
            raise ValueError(
                f"Invalid mapping entry '{chunk}'; expected '{kv_sep}' separator."
            )
        key_str, values_str = chunk.split(kv_sep, 1)
        key_str = key_str.strip()
        values_str = values_str.strip()
        if not key_str:
            raise ValueError("Empty concept_id in mapping string.")
        try:
            concept_id = int(key_str)
        except Exception as exc:
            raise ValueError(
                f"Mapping key '{key_str}' is not an int."
            ) from exc
        if not values_str:
            result[concept_id] = []
            continue
        indices: list[int] = []
        for item in values_str.split(list_sep):
            item = item.strip()
            if not item:
                continue
            try:
                index = int(item)
            except Exception as exc:
                raise ValueError(
                    f"Mapping value '{item}' is not an int."
                ) from exc
            if index < 0:
                raise ValueError(
                    f"Mapping index for concept {concept_id} must be >= 0."
                )
            indices.append(index)
        result[concept_id] = indices
    return result


def _parse_float_map_string(
    value: str, *, item_sep: str = ";", kv_sep: str = ":"
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for chunk in value.split(item_sep):
        chunk = chunk.strip()
        if not chunk:
            continue
        if kv_sep not in chunk:
            raise ValueError(
                f"Invalid mapping entry '{chunk}'; expected '{kv_sep}' separator."
            )
        key_str, value_str = chunk.split(kv_sep, 1)
        key_str = key_str.strip()
        value_str = value_str.strip()
        if not key_str:
            raise ValueError("Empty concept_id in mapping string.")
        try:
            concept_id = int(key_str)
        except Exception as exc:
            raise ValueError(
                f"Mapping key '{key_str}' is not an int."
            ) from exc
        try:
            multiplier = float(value_str)
        except Exception as exc:
            raise ValueError(
                f"Mapping value '{value_str}' is not a float."
            ) from exc
        if multiplier <= 0:
            raise ValueError(
                f"Mapping multiplier for concept {concept_id} must be > 0."
            )
        result[concept_id] = multiplier
    return result


def parse_contrastive_attention_config(config: Dict[str, Any], args: Any) -> None:
    args.enable_contrastive_attention = bool(
        config.get("enable_contrastive_attention", False)
    )
    args.contrastive_attention_weight = float(
        config.get("contrastive_attention_weight", 0.0)
    )
    if args.contrastive_attention_weight < 0:
        raise ValueError(
            "contrastive_attention_weight must be >= 0, got "
            f"{args.contrastive_attention_weight}"
        )
    args.contrastive_attention_temperature = float(
        config.get("contrastive_attention_temperature", 0.1)
    )
    if args.contrastive_attention_temperature <= 0:
        raise ValueError(
            "contrastive_attention_temperature must be > 0, got "
            f"{args.contrastive_attention_temperature}"
        )
    args.contrastive_attention_layer_start = int(
        config.get("contrastive_attention_layer_start", 10)
    )
    if args.contrastive_attention_layer_start < 0:
        raise ValueError(
            "contrastive_attention_layer_start must be >= 0, got "
            f"{args.contrastive_attention_layer_start}"
        )
    args.contrastive_attention_layer_end = int(
        config.get("contrastive_attention_layer_end", 20)
    )
    if args.contrastive_attention_layer_end <= args.contrastive_attention_layer_start:
        raise ValueError(
            "contrastive_attention_layer_end must be greater than "
            "contrastive_attention_layer_start (start="
            f"{args.contrastive_attention_layer_start}, "
            f"end={args.contrastive_attention_layer_end})"
        )
    args.contrastive_attention_head_limit = int(
        config.get("contrastive_attention_head_limit", 4)
    )
    if args.contrastive_attention_head_limit < 0:
        raise ValueError(
            "contrastive_attention_head_limit must be >= 0, got "
            f"{args.contrastive_attention_head_limit}"
        )
    args.contrastive_attention_max_queries = int(
        config.get("contrastive_attention_max_queries", 128)
    )
    if args.contrastive_attention_max_queries <= 0:
        raise ValueError(
            "contrastive_attention_max_queries must be > 0, got "
            f"{args.contrastive_attention_max_queries}"
        )
    args.contrastive_attention_layer_agg = str(
        config.get("contrastive_attention_layer_agg", "mean")
    ).lower()
    if args.contrastive_attention_layer_agg not in {"mean", "max"}:
        raise ValueError(
            "contrastive_attention_layer_agg must be 'mean' or 'max', got "
            f"{args.contrastive_attention_layer_agg}"
        )
    args.contrastive_attention_interval = int(
        config.get("contrastive_attention_interval", 1)
    )
    if args.contrastive_attention_interval <= 0:
        raise ValueError(
            "contrastive_attention_interval must be > 0, got "
            f"{args.contrastive_attention_interval}"
        )
    args.contrastive_attention_focus_tokens = bool(
        config.get("contrastive_attention_focus_tokens", False)
    )
    args.contrastive_attention_focus_renorm = bool(
        config.get("contrastive_attention_focus_renorm", True)
    )
    args.contrastive_attention_spatial_focus = bool(
        config.get("contrastive_attention_spatial_focus", False)
    )
    args.contrastive_attention_spatial_focus_power = float(
        config.get("contrastive_attention_spatial_focus_power", 1.0)
    )
    if args.contrastive_attention_spatial_focus_power <= 0:
        raise ValueError(
            "contrastive_attention_spatial_focus_power must be > 0, got "
            f"{args.contrastive_attention_spatial_focus_power}"
        )
    token_map = config.get("contrastive_attention_token_indices", {})
    parsed_map: Dict[int, list[int]] = {}
    if token_map:
        if isinstance(token_map, str):
            parsed_map = _parse_int_map_string(token_map)
        elif isinstance(token_map, dict):
            for key, value in token_map.items():
                try:
                    concept_id = int(key)
                except Exception as exc:
                    raise ValueError(
                        f"contrastive_attention_token_indices key '{key}' is not an int"
                    ) from exc
                if not isinstance(value, list):
                    raise ValueError(
                        f"contrastive_attention_token_indices[{key}] must be a list"
                    )
                indices: list[int] = []
                for item in value:
                    try:
                        index = int(item)
                    except Exception as exc:
                        raise ValueError(
                            f"contrastive_attention_token_indices[{key}] item '{item}' is not an int"
                        ) from exc
                    if index < 0:
                        raise ValueError(
                            f"contrastive_attention_token_indices[{key}] has negative index {index}"
                        )
                    indices.append(index)
                parsed_map[concept_id] = indices
        else:
            raise ValueError(
                "contrastive_attention_token_indices must be a dict or string mapping."
            )
    args.contrastive_attention_token_indices = parsed_map
    args.contrastive_attention_diversity_weight = float(
        config.get("contrastive_attention_diversity_weight", 0.0)
    )
    if args.contrastive_attention_diversity_weight < 0:
        raise ValueError(
            "contrastive_attention_diversity_weight must be >= 0, got "
            f"{args.contrastive_attention_diversity_weight}"
        )
    args.contrastive_attention_consistency_weight = float(
        config.get("contrastive_attention_consistency_weight", 0.0)
    )
    if args.contrastive_attention_consistency_weight < 0:
        raise ValueError(
            "contrastive_attention_consistency_weight must be >= 0, got "
            f"{args.contrastive_attention_consistency_weight}"
        )
    args.contrastive_attention_consistency_decay = float(
        config.get("contrastive_attention_consistency_decay", 0.9)
    )
    if not 0.0 <= args.contrastive_attention_consistency_decay <= 1.0:
        raise ValueError(
            "contrastive_attention_consistency_decay must be in [0, 1], got "
            f"{args.contrastive_attention_consistency_decay}"
        )
    args.contrastive_attention_weight_ramp_start = int(
        config.get("contrastive_attention_weight_ramp_start", 0)
    )
    args.contrastive_attention_weight_ramp_end = int(
        config.get("contrastive_attention_weight_ramp_end", 0)
    )
    if args.contrastive_attention_weight_ramp_start < 0:
        raise ValueError(
            "contrastive_attention_weight_ramp_start must be >= 0, got "
            f"{args.contrastive_attention_weight_ramp_start}"
        )
    if args.contrastive_attention_weight_ramp_end < 0:
        raise ValueError(
            "contrastive_attention_weight_ramp_end must be >= 0, got "
            f"{args.contrastive_attention_weight_ramp_end}"
        )
    if (
        args.contrastive_attention_weight_ramp_end
        < args.contrastive_attention_weight_ramp_start
    ):
        raise ValueError(
            "contrastive_attention_weight_ramp_end must be >= "
            "contrastive_attention_weight_ramp_start"
        )
    args.contrastive_attention_weight_ramp_type = str(
        config.get("contrastive_attention_weight_ramp_type", "linear")
    ).lower()
    if args.contrastive_attention_weight_ramp_type not in {"linear", "cosine"}:
        raise ValueError(
            "contrastive_attention_weight_ramp_type must be 'linear' or 'cosine', got "
            f"{args.contrastive_attention_weight_ramp_type}"
        )
    multiplier_map = config.get("contrastive_attention_concept_multipliers", {})
    parsed_multiplier_map: Dict[int, float] = {}
    if multiplier_map:
        if isinstance(multiplier_map, str):
            parsed_multiplier_map = _parse_float_map_string(multiplier_map)
        elif isinstance(multiplier_map, dict):
            for key, value in multiplier_map.items():
                try:
                    concept_id = int(key)
                except Exception as exc:
                    raise ValueError(
                        f"contrastive_attention_concept_multipliers key '{key}' is not an int"
                    ) from exc
                try:
                    multiplier = float(value)
                except Exception as exc:
                    raise ValueError(
                        f"contrastive_attention_concept_multipliers[{key}] value '{value}' is not a float"
                    ) from exc
                if multiplier <= 0:
                    raise ValueError(
                        f"contrastive_attention_concept_multipliers[{key}] must be > 0, got {multiplier}"
                    )
                parsed_multiplier_map[concept_id] = multiplier
        else:
            raise ValueError(
                "contrastive_attention_concept_multipliers must be a dict or string mapping."
            )
    args.contrastive_attention_concept_multipliers = parsed_multiplier_map
    args.contrastive_attention_extra_prompt_passes = int(
        config.get("contrastive_attention_extra_prompt_passes", 0)
    )
    if args.contrastive_attention_extra_prompt_passes < 0:
        raise ValueError(
            "contrastive_attention_extra_prompt_passes must be >= 0, got "
            f"{args.contrastive_attention_extra_prompt_passes}"
        )
    args.contrastive_attention_extra_prompt_strategy = str(
        config.get("contrastive_attention_extra_prompt_strategy", "shuffle")
    ).lower()
    if args.contrastive_attention_extra_prompt_strategy not in {"shuffle", "roll"}:
        raise ValueError(
            "contrastive_attention_extra_prompt_strategy must be 'shuffle' or 'roll', got "
            f"{args.contrastive_attention_extra_prompt_strategy}"
        )
    args.contrastive_attention_latent_update = bool(
        config.get("contrastive_attention_latent_update", False)
    )
    args.contrastive_attention_latent_update_steps = int(
        config.get("contrastive_attention_latent_update_steps", 0)
    )
    if args.contrastive_attention_latent_update_steps < 0:
        raise ValueError(
            "contrastive_attention_latent_update_steps must be >= 0, got "
            f"{args.contrastive_attention_latent_update_steps}"
        )
    args.contrastive_attention_latent_update_step_size = float(
        config.get("contrastive_attention_latent_update_step_size", 0.1)
    )
    if args.contrastive_attention_latent_update_step_size <= 0:
        raise ValueError(
            "contrastive_attention_latent_update_step_size must be > 0, got "
            f"{args.contrastive_attention_latent_update_step_size}"
        )
    args.contrastive_attention_latent_update_interval = int(
        config.get("contrastive_attention_latent_update_interval", 1)
    )
    if args.contrastive_attention_latent_update_interval <= 0:
        raise ValueError(
            "contrastive_attention_latent_update_interval must be > 0, got "
            f"{args.contrastive_attention_latent_update_interval}"
        )

    if args.enable_contrastive_attention:
        logger.info(
            "Contrastive attention enabled (layers %d-%d, weight=%.3f, temp=%.3f).",
            args.contrastive_attention_layer_start,
            args.contrastive_attention_layer_end,
            args.contrastive_attention_weight,
            args.contrastive_attention_temperature,
        )
