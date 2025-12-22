import logging
from typing import Any, List


def parse_layer_sync_config(config: Any, args: Any, logger: logging.Logger) -> None:
    """
    Populate LayerSync-related args from the TOML config with validation.

    This keeps LayerSync parsing isolated while preserving the existing defaults
    and safety checks used by the main config parser.
    """
    try:
        args.enable_layer_sync = bool(
            config.get("enable_layer_sync", config.get("layersync_enabled", False))
        )
        weight_raw = config.get("layer_sync_weight", config.get("layersync_lambda", None))
        if weight_raw is None:
            args.layer_sync_weight = 0.2 if args.enable_layer_sync else 0.0
        else:
            args.layer_sync_weight = float(weight_raw)

        args.layer_sync_source_block = int(
            config.get(
                "layer_sync_source_block", config.get("layersync_student_block", 8)
            )
        )
        target_raw = config.get(
            "layer_sync_target_block", config.get("layersync_teacher_block", None)
        )
        if target_raw is None:
            args.layer_sync_target_block = args.layer_sync_source_block
        else:
            args.layer_sync_target_block = int(target_raw)
        args.layer_sync_pairs = config.get("layer_sync_pairs", None)
        args.layer_sync_pair_weights = config.get("layer_sync_pair_weights", None)
        args.layer_sync_detach_guidance = bool(
            config.get("layer_sync_detach_guidance", config.get("layersync_detach_guidance", True))
        )
        args.layer_sync_normalization = str(
            config.get("layer_sync_normalization", config.get("layersync_normalization", "cosine"))
        ).lower()

        if args.layer_sync_weight < 0:
            raise ValueError("layer_sync_weight must be non-negative")
        if args.enable_layer_sync and args.layer_sync_weight <= 0:
            raise ValueError("layer_sync_weight must be > 0 when LayerSync is enabled")
        if args.layer_sync_source_block < 0 or args.layer_sync_target_block < 0:
            raise ValueError("LayerSync block indices must be >= 0")
        if args.layer_sync_source_block > args.layer_sync_target_block:
            raise ValueError(
                "layer_sync_target_block must be >= layer_sync_source_block"
            )

        if args.layer_sync_pairs is not None:
            if not isinstance(args.layer_sync_pairs, list):
                raise ValueError(
                    "layer_sync_pairs must be a list of [source, target] pairs"
                )

            parsed_pairs: List[List[int]] = []
            for pair in args.layer_sync_pairs:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and all(isinstance(v, (int, float)) for v in pair)
                ):
                    src_block, tgt_block = int(pair[0]), int(pair[1])
                    if src_block < 0 or tgt_block < 0:
                        raise ValueError(
                            f"LayerSync pair values must be >=0, got {pair}"
                        )
                    if src_block >= tgt_block:
                        raise ValueError(
                            f"LayerSync pair must satisfy source < target, got {pair}"
                        )
                    parsed_pairs.append([src_block, tgt_block])
                else:
                    raise ValueError(
                        f"LayerSync pair must be a 2-element list, got {pair}"
                    )
            args.layer_sync_pairs = parsed_pairs

            if args.layer_sync_pair_weights is not None:
                if not isinstance(args.layer_sync_pair_weights, list):
                    raise ValueError(
                        "layer_sync_pair_weights must be a list of numeric weights"
                    )
                if len(args.layer_sync_pair_weights) != len(args.layer_sync_pairs):
                    raise ValueError(
                        "layer_sync_pair_weights length must match layer_sync_pairs length"
                    )
                parsed_weights: List[float] = []
                for w in args.layer_sync_pair_weights:
                    try:
                        weight_val = float(w)
                    except Exception as exc:
                        raise ValueError(
                            f"LayerSync pair weight must be numeric, got {w}"
                        ) from exc
                    if weight_val < 0:
                        raise ValueError(
                            f"LayerSync pair weights must be non-negative, got {weight_val}"
                        )
                    parsed_weights.append(weight_val)
                args.layer_sync_pair_weights = parsed_weights
            else:
                args.layer_sync_pair_weights = [1.0 for _ in args.layer_sync_pairs]
        elif args.layer_sync_pair_weights is not None:
            raise ValueError(
                "layer_sync_pair_weights provided without layer_sync_pairs; specify pairs first"
            )

        if args.layer_sync_normalization not in ("cosine",):
            raise ValueError(
                f"Unsupported layer_sync_normalization '{args.layer_sync_normalization}'. "
                "Supported values: 'cosine'."
            )

        if args.enable_layer_sync:
            logger.info(
                "LayerSync enabled (weight=%.4f, pairs=%s, detach_guidance=%s)",
                args.layer_sync_weight,
                (
                    args.layer_sync_pairs
                    if args.layer_sync_pairs is not None
                    else [(args.layer_sync_source_block, args.layer_sync_target_block)]
                ),
                args.layer_sync_detach_guidance,
            )
    except Exception as e:
        logger.exception(f"LayerSync config parsing failed: {e}")
        raise
