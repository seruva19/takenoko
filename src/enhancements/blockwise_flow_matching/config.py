from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BlockwiseFlowMatchingConfig:
    enabled: bool
    num_segments: int
    segment_sampling: str
    segment_min_t: float
    segment_max_t: float
    log_segment_losses: bool
    use_segment_objective: bool
    semfeat_enabled: bool
    semfeat_encoder_name: str
    semfeat_resolution: int
    semfeat_alignment_depth: int
    semfeat_loss_weight: float
    semfeat_spatial_align: bool
    semfeat_projection_dim: int
    semfeat_conditioning_enabled: bool
    semfeat_conditioning_scale: float
    semfeat_conditioning_dropout: float
    segment_conditioning_enabled: bool
    segment_conditioning_scale: float
    segment_blocks_enabled: bool
    segment_block_mode: str
    semfeat_model_injection_enabled: bool
    semfeat_model_injection_scale: float
    inference_enabled: bool
    inference_semfeat_enabled: bool
    inference_segment_enabled: bool
    inference_semfeat_refresh: str
    inference_semfeat_scale: float
    inference_segment_scale: float
    frn_enabled: bool
    frn_loss_weight: float
    frn_hidden_dim: int
    inference_use_frn: bool

    @classmethod
    def from_args(cls, args: Any) -> "BlockwiseFlowMatchingConfig":
        enabled = bool(getattr(args, "enable_blockwise_flow_matching", False))
        num_segments = int(getattr(args, "bfm_num_segments", 6))
        segment_sampling = str(getattr(args, "bfm_segment_sampling", "stratified"))
        segment_min_t = float(getattr(args, "bfm_segment_min_t", 0.0))
        segment_max_t = float(getattr(args, "bfm_segment_max_t", 1.0))
        log_segment_losses = bool(getattr(args, "bfm_log_segment_losses", True))
        use_segment_objective = bool(
            getattr(args, "bfm_use_segment_objective", True)
        )
        semfeat_enabled = bool(getattr(args, "bfm_semfeat_enabled", False))
        semfeat_encoder_name = str(
            getattr(args, "bfm_semfeat_encoder_name", "dinov2-vit-b14")
        )
        semfeat_resolution = int(getattr(args, "bfm_semfeat_resolution", 256))
        semfeat_alignment_depth = int(
            getattr(args, "bfm_semfeat_alignment_depth", 8)
        )
        semfeat_loss_weight = float(
            getattr(args, "bfm_semfeat_loss_weight", 0.05)
        )
        semfeat_spatial_align = bool(
            getattr(args, "bfm_semfeat_spatial_align", True)
        )
        semfeat_projection_dim = int(
            getattr(args, "bfm_semfeat_projection_dim", 1024)
        )
        semfeat_conditioning_enabled = bool(
            getattr(args, "bfm_semfeat_conditioning_enabled", False)
        )
        semfeat_conditioning_scale = float(
            getattr(args, "bfm_semfeat_conditioning_scale", 1.0)
        )
        semfeat_conditioning_dropout = float(
            getattr(args, "bfm_semfeat_conditioning_dropout", 0.0)
        )
        segment_conditioning_enabled = bool(
            getattr(args, "bfm_segment_conditioning_enabled", False)
        )
        segment_conditioning_scale = float(
            getattr(args, "bfm_segment_conditioning_scale", 1.0)
        )
        segment_blocks_enabled = bool(
            getattr(args, "bfm_segment_blocks_enabled", False)
        )
        segment_block_mode = str(
            getattr(args, "bfm_segment_block_mode", "shared")
        )
        semfeat_model_injection_enabled = bool(
            getattr(args, "bfm_semfeat_model_injection_enabled", False)
        )
        semfeat_model_injection_scale = float(
            getattr(args, "bfm_semfeat_model_injection_scale", 1.0)
        )
        inference_enabled = bool(
            getattr(args, "bfm_inference_enabled", False)
        )
        inference_semfeat_enabled = bool(
            getattr(args, "bfm_inference_semfeat_enabled", False)
        )
        inference_segment_enabled = bool(
            getattr(args, "bfm_inference_segment_enabled", False)
        )
        inference_semfeat_refresh = str(
            getattr(args, "bfm_inference_semfeat_refresh", "per_segment")
        )
        inference_semfeat_scale = float(
            getattr(args, "bfm_inference_semfeat_scale", 1.0)
        )
        inference_segment_scale = float(
            getattr(args, "bfm_inference_segment_scale", 1.0)
        )
        frn_enabled = bool(getattr(args, "bfm_frn_enabled", False))
        frn_loss_weight = float(getattr(args, "bfm_frn_loss_weight", 0.1))
        frn_hidden_dim = int(getattr(args, "bfm_frn_hidden_dim", 1024))
        inference_use_frn = bool(getattr(args, "bfm_inference_use_frn", False))

        _validate_segment_settings(
            enabled=enabled,
            num_segments=num_segments,
            segment_sampling=segment_sampling,
            segment_min_t=segment_min_t,
            segment_max_t=segment_max_t,
        )
        _validate_semfeat_settings(
            semfeat_enabled=semfeat_enabled,
            semfeat_resolution=semfeat_resolution,
            semfeat_alignment_depth=semfeat_alignment_depth,
            semfeat_loss_weight=semfeat_loss_weight,
            semfeat_projection_dim=semfeat_projection_dim,
        )
        if segment_blocks_enabled:
            allowed_modes = {"shared", "replicated"}
            if segment_block_mode not in allowed_modes:
                raise ValueError(
                    "bfm_segment_block_mode must be 'shared' or 'replicated'."
                )
        if semfeat_model_injection_enabled and semfeat_model_injection_scale < 0.0:
            raise ValueError(
                "bfm_semfeat_model_injection_scale must be >= 0."
            )

        return cls(
            enabled=enabled,
            num_segments=num_segments,
            segment_sampling=segment_sampling,
            segment_min_t=segment_min_t,
            segment_max_t=segment_max_t,
            log_segment_losses=log_segment_losses,
            use_segment_objective=use_segment_objective,
            semfeat_enabled=semfeat_enabled,
            semfeat_encoder_name=semfeat_encoder_name,
            semfeat_resolution=semfeat_resolution,
            semfeat_alignment_depth=semfeat_alignment_depth,
            semfeat_loss_weight=semfeat_loss_weight,
            semfeat_spatial_align=semfeat_spatial_align,
            semfeat_projection_dim=semfeat_projection_dim,
            semfeat_conditioning_enabled=semfeat_conditioning_enabled,
            semfeat_conditioning_scale=semfeat_conditioning_scale,
            semfeat_conditioning_dropout=semfeat_conditioning_dropout,
            segment_conditioning_enabled=segment_conditioning_enabled,
            segment_conditioning_scale=segment_conditioning_scale,
            segment_blocks_enabled=segment_blocks_enabled,
            segment_block_mode=segment_block_mode,
            semfeat_model_injection_enabled=semfeat_model_injection_enabled,
            semfeat_model_injection_scale=semfeat_model_injection_scale,
            inference_enabled=inference_enabled,
            inference_semfeat_enabled=inference_semfeat_enabled,
            inference_segment_enabled=inference_segment_enabled,
            inference_semfeat_refresh=inference_semfeat_refresh,
            inference_semfeat_scale=inference_semfeat_scale,
            inference_segment_scale=inference_segment_scale,
            frn_enabled=frn_enabled,
            frn_loss_weight=frn_loss_weight,
            frn_hidden_dim=frn_hidden_dim,
            inference_use_frn=inference_use_frn,
        )


def parse_blockwise_flow_matching_config(
    config: Dict[str, Any],
    args: Any,
    logger: Any,
) -> None:
    args.enable_blockwise_flow_matching = bool(
        config.get("enable_blockwise_flow_matching", False)
    )
    args.bfm_num_segments = int(config.get("bfm_num_segments", 6))
    args.bfm_segment_sampling = str(
        config.get("bfm_segment_sampling", "stratified")
    )
    args.bfm_segment_min_t = float(config.get("bfm_segment_min_t", 0.0))
    args.bfm_segment_max_t = float(config.get("bfm_segment_max_t", 1.0))
    args.bfm_log_segment_losses = bool(
        config.get("bfm_log_segment_losses", True)
    )
    args.bfm_use_segment_objective = bool(
        config.get("bfm_use_segment_objective", True)
    )
    args.bfm_semfeat_enabled = bool(config.get("bfm_semfeat_enabled", False))
    args.bfm_semfeat_encoder_name = str(
        config.get("bfm_semfeat_encoder_name", "dinov2-vit-b14")
    )
    args.bfm_semfeat_resolution = int(
        config.get("bfm_semfeat_resolution", 256)
    )
    args.bfm_semfeat_alignment_depth = int(
        config.get("bfm_semfeat_alignment_depth", 8)
    )
    args.bfm_semfeat_loss_weight = float(
        config.get("bfm_semfeat_loss_weight", 0.05)
    )
    args.bfm_semfeat_spatial_align = bool(
        config.get("bfm_semfeat_spatial_align", True)
    )
    args.bfm_semfeat_projection_dim = int(
        config.get("bfm_semfeat_projection_dim", 1024)
    )
    args.bfm_semfeat_conditioning_enabled = bool(
        config.get("bfm_semfeat_conditioning_enabled", False)
    )
    args.bfm_semfeat_conditioning_scale = float(
        config.get("bfm_semfeat_conditioning_scale", 1.0)
    )
    args.bfm_semfeat_conditioning_dropout = float(
        config.get("bfm_semfeat_conditioning_dropout", 0.0)
    )
    args.bfm_segment_conditioning_enabled = bool(
        config.get("bfm_segment_conditioning_enabled", False)
    )
    args.bfm_segment_conditioning_scale = float(
        config.get("bfm_segment_conditioning_scale", 1.0)
    )
    args.bfm_segment_blocks_enabled = bool(
        config.get("bfm_segment_blocks_enabled", False)
    )
    args.bfm_segment_block_mode = str(
        config.get("bfm_segment_block_mode", "shared")
    )
    args.bfm_semfeat_model_injection_enabled = bool(
        config.get("bfm_semfeat_model_injection_enabled", False)
    )
    args.bfm_semfeat_model_injection_scale = float(
        config.get("bfm_semfeat_model_injection_scale", 1.0)
    )
    args.bfm_inference_enabled = bool(
        config.get("bfm_inference_enabled", False)
    )
    args.bfm_inference_semfeat_enabled = bool(
        config.get("bfm_inference_semfeat_enabled", False)
    )
    args.bfm_inference_segment_enabled = bool(
        config.get("bfm_inference_segment_enabled", False)
    )
    args.bfm_inference_semfeat_refresh = str(
        config.get("bfm_inference_semfeat_refresh", "per_segment")
    )
    args.bfm_inference_semfeat_scale = float(
        config.get("bfm_inference_semfeat_scale", 1.0)
    )
    args.bfm_inference_segment_scale = float(
        config.get("bfm_inference_segment_scale", 1.0)
    )
    args.bfm_frn_enabled = bool(
        config.get("bfm_frn_enabled", False)
    )
    args.bfm_frn_loss_weight = float(
        config.get("bfm_frn_loss_weight", 0.1)
    )
    args.bfm_frn_hidden_dim = int(
        config.get("bfm_frn_hidden_dim", 1024)
    )
    args.bfm_inference_use_frn = bool(
        config.get("bfm_inference_use_frn", False)
    )

    _validate_segment_settings(
        enabled=args.enable_blockwise_flow_matching,
        num_segments=args.bfm_num_segments,
        segment_sampling=args.bfm_segment_sampling,
        segment_min_t=args.bfm_segment_min_t,
        segment_max_t=args.bfm_segment_max_t,
    )
    _validate_semfeat_settings(
        semfeat_enabled=args.bfm_semfeat_enabled,
        semfeat_resolution=args.bfm_semfeat_resolution,
        semfeat_alignment_depth=args.bfm_semfeat_alignment_depth,
        semfeat_loss_weight=args.bfm_semfeat_loss_weight,
        semfeat_projection_dim=args.bfm_semfeat_projection_dim,
    )

    if (
        getattr(args, "timestep_sampling", None) == "blockwise"
        and not args.enable_blockwise_flow_matching
    ):
        logger.info(
            "blockwise timestep sampling requested; enabling blockwise flow matching."
        )
        args.enable_blockwise_flow_matching = True
    if (
        (args.bfm_semfeat_conditioning_enabled or args.bfm_segment_conditioning_enabled)
        and not args.enable_blockwise_flow_matching
    ):
        logger.info(
            "BFM conditioning requested; enabling blockwise flow matching."
        )
        args.enable_blockwise_flow_matching = True
    if (
        (args.bfm_inference_semfeat_enabled or args.bfm_inference_segment_enabled)
        and not args.bfm_inference_enabled
    ):
        logger.info(
            "BFM inference features requested; enabling bfm_inference_enabled."
        )
        args.bfm_inference_enabled = True
    if args.enable_blockwise_flow_matching:
        logger.info(
            "Blockwise flow matching enabled (segments=%d, sampling=%s).",
            args.bfm_num_segments,
            args.bfm_segment_sampling,
        )
        if not args.bfm_use_segment_objective:
            logger.info("BFM segment objective disabled; using standard target.")
    if args.bfm_semfeat_enabled:
        logger.info(
            "BFM SemFeat enabled (encoder=%s, depth=%d, weight=%.3f).",
            args.bfm_semfeat_encoder_name,
            args.bfm_semfeat_alignment_depth,
            args.bfm_semfeat_loss_weight,
        )
    if args.bfm_semfeat_conditioning_enabled:
        if not (0.0 <= args.bfm_semfeat_conditioning_dropout <= 1.0):
            raise ValueError(
                "bfm_semfeat_conditioning_dropout must be within [0, 1]."
            )
        if args.bfm_semfeat_conditioning_scale < 0.0:
            raise ValueError(
                "bfm_semfeat_conditioning_scale must be >= 0."
            )
        logger.info(
            "BFM SemFeat conditioning enabled (scale=%.3f, dropout=%.2f).",
            args.bfm_semfeat_conditioning_scale,
            args.bfm_semfeat_conditioning_dropout,
        )
    if args.bfm_segment_conditioning_enabled:
        if args.bfm_segment_conditioning_scale < 0.0:
            raise ValueError(
                "bfm_segment_conditioning_scale must be >= 0."
            )
        logger.info(
            "BFM segment conditioning enabled (scale=%.3f).",
            args.bfm_segment_conditioning_scale,
        )
    if args.bfm_segment_blocks_enabled:
        allowed_modes = {"shared", "replicated"}
        if args.bfm_segment_block_mode not in allowed_modes:
            raise ValueError(
                "bfm_segment_block_mode must be 'shared' or 'replicated'."
            )
        logger.warning(
            "BFM segment blocks enabled (mode=%s); inference will change.",
            args.bfm_segment_block_mode,
        )
    if args.bfm_semfeat_model_injection_enabled:
        if args.bfm_semfeat_model_injection_scale < 0.0:
            raise ValueError(
                "bfm_semfeat_model_injection_scale must be >= 0."
            )
        logger.warning(
            "BFM SemFeat model injection enabled (scale=%.3f); inference will change.",
            args.bfm_semfeat_model_injection_scale,
        )
    if args.bfm_inference_enabled:
        allowed_refresh = {"per_segment", "per_step"}
        if args.bfm_inference_semfeat_refresh not in allowed_refresh:
            raise ValueError(
                "bfm_inference_semfeat_refresh must be 'per_segment' or 'per_step'."
            )
        if args.bfm_inference_semfeat_scale < 0.0:
            raise ValueError(
                "bfm_inference_semfeat_scale must be >= 0."
            )
        if args.bfm_inference_segment_scale < 0.0:
            raise ValueError(
                "bfm_inference_segment_scale must be >= 0."
            )
        logger.warning(
            "BFM inference enabled (semfeat=%s, segment=%s, refresh=%s).",
            args.bfm_inference_semfeat_enabled,
            args.bfm_inference_segment_enabled,
            args.bfm_inference_semfeat_refresh,
        )
        if args.bfm_inference_use_frn:
            logger.warning("BFM inference FRN enabled; inference will change.")
    if args.bfm_frn_enabled:
        if args.bfm_frn_loss_weight < 0.0:
            raise ValueError("bfm_frn_loss_weight must be >= 0.")
        if args.bfm_frn_hidden_dim <= 0:
            raise ValueError("bfm_frn_hidden_dim must be > 0.")
        logger.info(
            "BFM FRN training enabled (loss_weight=%.3f, hidden=%d).",
            args.bfm_frn_loss_weight,
            args.bfm_frn_hidden_dim,
        )


def _validate_segment_settings(
    *,
    enabled: bool,
    num_segments: int,
    segment_sampling: str,
    segment_min_t: float,
    segment_max_t: float,
) -> None:
    allowed_sampling = {"stratified", "uniform"}
    if segment_sampling not in allowed_sampling:
        raise ValueError(
            f"bfm_segment_sampling must be one of {sorted(allowed_sampling)}, "
            f"got {segment_sampling!r}"
        )
    if num_segments < 2:
        raise ValueError(
            f"bfm_num_segments must be >= 2, got {num_segments}"
        )
    if segment_min_t < 0.0 or segment_max_t > 1.0:
        raise ValueError(
            "bfm_segment_min_t and bfm_segment_max_t must be within [0, 1]."
        )
    if segment_min_t >= segment_max_t:
        raise ValueError(
            "bfm_segment_min_t must be < bfm_segment_max_t."
        )


def _validate_semfeat_settings(
    *,
    semfeat_enabled: bool,
    semfeat_resolution: int,
    semfeat_alignment_depth: int,
    semfeat_loss_weight: float,
    semfeat_projection_dim: int,
) -> None:
    if semfeat_resolution not in {256, 512}:
        raise ValueError(
            f"bfm_semfeat_resolution must be 256 or 512, got {semfeat_resolution}"
        )
    if semfeat_alignment_depth < 0:
        raise ValueError(
            "bfm_semfeat_alignment_depth must be >= 0."
        )
    if semfeat_loss_weight < 0.0:
        raise ValueError(
            "bfm_semfeat_loss_weight must be >= 0."
        )
    if semfeat_projection_dim <= 0:
        raise ValueError(
            "bfm_semfeat_projection_dim must be > 0."
        )
