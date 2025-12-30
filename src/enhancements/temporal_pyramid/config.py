"""Config parsing for temporal pyramid training features.

Keys:
- temporal_pyramid_num_stages: stage count (>0).
- temporal_pyramid_stage_weights: per-stage sampling weights, length = num_stages.
- temporal_pyramid_stage_boundaries: optional [0,1] boundaries, length = num_stages + 1.
- enable_temporal_pyramid_training: apply temporal downsample/upsample per stage.
- temporal_pyramid_stride_base: base stride (stride_base ** stage).
- temporal_pyramid_max_stride: optional stride cap.
- enable_temporal_pyramid_data_noise_alignment: align noise to latents (SciPy or greedy).
- enable_temporal_pyramid_stagewise_target: enable stage-wise target override.
- temporal_pyramid_gamma_sigma_mode: "flow" or "scheduler" parameterization.
"""

from __future__ import annotations

from typing import Any, Dict


def parse_temporal_pyramid_config(config: Dict[str, Any], args: Any, logger) -> None:
    """Parse temporal pyramid settings into args with validation and logging."""
    # Temporal pyramid (training-only, TPDiff-inspired)
    args.enable_temporal_pyramid_training = bool(
        config.get("enable_temporal_pyramid_training", False)
    )
    args.temporal_pyramid_num_stages = int(
        config.get("temporal_pyramid_num_stages", 3)
    )
    args.temporal_pyramid_stride_base = int(
        config.get("temporal_pyramid_stride_base", 2)
    )
    args.temporal_pyramid_max_stride = config.get("temporal_pyramid_max_stride", 8)
    if args.temporal_pyramid_max_stride is not None:
        args.temporal_pyramid_max_stride = int(args.temporal_pyramid_max_stride)
    args.temporal_pyramid_stage_weights = config.get(
        "temporal_pyramid_stage_weights", [1.0, 1.0, 1.0]
    )
    args.temporal_pyramid_stage_boundaries = config.get(
        "temporal_pyramid_stage_boundaries", None
    )
    args.temporal_pyramid_gamma_sigma_mode = config.get(
        "temporal_pyramid_gamma_sigma_mode", "flow"
    )
    args.enable_temporal_pyramid_data_noise_alignment = bool(
        config.get("enable_temporal_pyramid_data_noise_alignment", False)
    )
    args.enable_temporal_pyramid_stagewise_target = bool(
        config.get("enable_temporal_pyramid_stagewise_target", False)
    )

    if args.temporal_pyramid_num_stages < 1:
        raise ValueError("temporal_pyramid_num_stages must be >= 1.")
    if args.temporal_pyramid_stride_base < 1:
        raise ValueError("temporal_pyramid_stride_base must be >= 1.")
    if (
        args.temporal_pyramid_max_stride is not None
        and args.temporal_pyramid_max_stride < 1
    ):
        raise ValueError("temporal_pyramid_max_stride must be >= 1.")

    if args.temporal_pyramid_stage_weights is None:
        args.temporal_pyramid_stage_weights = [
            1.0 for _ in range(args.temporal_pyramid_num_stages)
        ]
    if not isinstance(args.temporal_pyramid_stage_weights, (list, tuple)):
        raise ValueError("temporal_pyramid_stage_weights must be a list.")
    if len(args.temporal_pyramid_stage_weights) != args.temporal_pyramid_num_stages:
        if all(
            float(v) == 1.0 for v in args.temporal_pyramid_stage_weights
        ) and args.temporal_pyramid_num_stages != len(
            args.temporal_pyramid_stage_weights
        ):
            args.temporal_pyramid_stage_weights = [
                1.0 for _ in range(args.temporal_pyramid_num_stages)
            ]
            logger.info(
                "Temporal pyramid stage weights reset to uniform for %s stages.",
                args.temporal_pyramid_num_stages,
            )
        else:
            raise ValueError(
                "temporal_pyramid_stage_weights must match temporal_pyramid_num_stages."
            )
    args.temporal_pyramid_stage_weights = [
        float(v) for v in args.temporal_pyramid_stage_weights
    ]
    if sum(args.temporal_pyramid_stage_weights) <= 0.0:
        raise ValueError("temporal_pyramid_stage_weights must sum to > 0.")

    if args.temporal_pyramid_stage_boundaries is not None:
        if not isinstance(args.temporal_pyramid_stage_boundaries, (list, tuple)):
            raise ValueError("temporal_pyramid_stage_boundaries must be a list.")
        if len(args.temporal_pyramid_stage_boundaries) != args.temporal_pyramid_num_stages + 1:
            raise ValueError(
                "temporal_pyramid_stage_boundaries must have num_stages + 1 entries."
            )
        boundaries = [float(v) for v in args.temporal_pyramid_stage_boundaries]
        if boundaries[0] != 0.0 or boundaries[-1] != 1.0:
            raise ValueError(
                "temporal_pyramid_stage_boundaries must start at 0.0 and end at 1.0."
            )
        for left, right in zip(boundaries, boundaries[1:]):
            if right <= left:
                raise ValueError(
                    "temporal_pyramid_stage_boundaries must be strictly increasing."
                )
        args.temporal_pyramid_stage_boundaries = boundaries

    if args.temporal_pyramid_gamma_sigma_mode not in ("flow", "scheduler"):
        raise ValueError(
            "temporal_pyramid_gamma_sigma_mode must be 'flow' or 'scheduler'."
        )

    if args.timestep_sampling == "temporal_pyramid":
        logger.info(
            "Temporal pyramid stage weights: %s",
            ",".join(f"{v:.3f}" for v in args.temporal_pyramid_stage_weights),
        )
        if args.temporal_pyramid_stage_boundaries is not None:
            logger.info(
                "Temporal pyramid stage boundaries: %s",
                ",".join(f"{v:.3f}" for v in args.temporal_pyramid_stage_boundaries),
            )
    if args.enable_temporal_pyramid_training:
        logger.info(
            "Temporal pyramid training enabled (stages=%s, stride_base=%s).",
            args.temporal_pyramid_num_stages,
            args.temporal_pyramid_stride_base,
        )
    if args.enable_temporal_pyramid_data_noise_alignment:
        logger.info("Temporal pyramid data-noise alignment enabled.")
    if args.enable_temporal_pyramid_stagewise_target:
        logger.info("Temporal pyramid stagewise targets enabled.")
    if args.timestep_sampling == "fopp" and args.enable_temporal_pyramid_training:
        logger.warning(
            "Temporal pyramid training is enabled with fopp timesteps; "
            "temporal resampling is skipped in the FoPP branch."
        )
