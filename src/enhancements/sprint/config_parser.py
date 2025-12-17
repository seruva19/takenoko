"""
Sprint Configuration Parser

Handles parsing and validation of Sprint configuration from TOML config files.
"""

import argparse
import warnings
from typing import Dict, Any, Optional

from .exceptions import SprintConfigurationError, SprintCompatibilityError
from .device_utils import (
    validate_sprint_memory_requirements,
    estimate_sprint_memory_usage,
)


def parse_sprint_config(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Parse Sprint configuration from config dict and validate.

    Args:
        config: Configuration dictionary from TOML file
        args: Argparse namespace to populate with Sprint settings

    Raises:
        ValueError: If configuration values are invalid
    """
    # Sprint basic settings
    args.enable_sprint = config.get("enable_sprint", False)
    args.sprint_token_drop_ratio = float(config.get("sprint_token_drop_ratio", 0.75))
    args.sprint_encoder_layers = config.get("sprint_encoder_layers", None)
    args.sprint_middle_layers = config.get("sprint_middle_layers", None)
    args.sprint_sampling_strategy = config.get(
        "sprint_sampling_strategy", "temporal_coherent"
    )

    # Sprint advanced settings
    args.sprint_path_drop_prob = float(config.get("sprint_path_drop_prob", 0.1))
    args.sprint_partitioning_strategy = config.get(
        "sprint_partitioning_strategy", "percentage"
    )
    args.sprint_encoder_ratio = float(config.get("sprint_encoder_ratio", 0.25))
    args.sprint_middle_ratio = float(config.get("sprint_middle_ratio", 0.50))

    # Sprint two-stage training settings
    args.sprint_pretrain_steps = int(config.get("sprint_pretrain_steps", 0))
    args.sprint_finetune_steps = int(config.get("sprint_finetune_steps", 0))
    args.sprint_warmup_steps = int(config.get("sprint_warmup_steps", 0))
    args.sprint_cooldown_steps = int(config.get("sprint_cooldown_steps", 100))

    # Sprint diagnostics
    args.sprint_enable_diagnostics = bool(
        config.get("sprint_enable_diagnostics", False)
    )

    # Sprint mask token behavior
    args.sprint_use_learnable_mask_token = bool(
        config.get("sprint_use_learnable_mask_token", False)
    )

    # Validate Sprint configuration if enabled
    if args.enable_sprint:
        _validate_sprint_config(args)
        _validate_two_stage_training_config(args)
        _validate_sprint_compatibility(args)


def _validate_sprint_config(args: argparse.Namespace) -> None:
    """
    Validate Sprint configuration values.

    Args:
        args: Argparse namespace with Sprint settings

    Raises:
        ValueError: If any configuration value is invalid
    """
    # Validate token drop ratio
    if not 0.0 <= args.sprint_token_drop_ratio <= 1.0:
        raise ValueError(
            f"sprint_token_drop_ratio must be in [0.0, 1.0], got {args.sprint_token_drop_ratio}"
        )

    # Validate sampling strategy
    valid_strategies = ["uniform", "temporal_coherent", "spatial_coherent"]
    if args.sprint_sampling_strategy not in valid_strategies:
        raise ValueError(
            f"sprint_sampling_strategy must be one of {valid_strategies}, "
            f"got {args.sprint_sampling_strategy}"
        )

    # Validate encoder/middle layers if specified
    if args.sprint_encoder_layers is not None and args.sprint_encoder_layers < 1:
        raise ValueError(
            f"sprint_encoder_layers must be >= 1 if specified, got {args.sprint_encoder_layers}"
        )
    if args.sprint_middle_layers is not None and args.sprint_middle_layers < 1:
        raise ValueError(
            f"sprint_middle_layers must be >= 1 if specified, got {args.sprint_middle_layers}"
        )

    # Validate path-drop learning probability
    if not 0.0 <= args.sprint_path_drop_prob <= 1.0:
        raise ValueError(
            f"sprint_path_drop_prob must be in [0.0, 1.0], got {args.sprint_path_drop_prob}"
        )

    # Validate partitioning strategy
    valid_partitioning_strategies = ["percentage", "fixed"]
    if args.sprint_partitioning_strategy not in valid_partitioning_strategies:
        raise ValueError(
            f"sprint_partitioning_strategy must be one of {valid_partitioning_strategies}, "
            f"got '{args.sprint_partitioning_strategy}'"
        )

    # Validate encoder/middle ratios (for percentage strategy)
    if not 0.0 < args.sprint_encoder_ratio < 1.0:
        raise ValueError(
            f"sprint_encoder_ratio must be in (0.0, 1.0), got {args.sprint_encoder_ratio}"
        )
    if not 0.0 < args.sprint_middle_ratio < 1.0:
        raise ValueError(
            f"sprint_middle_ratio must be in (0.0, 1.0), got {args.sprint_middle_ratio}"
        )

    # Validate that ratios don't exceed 100%
    if args.sprint_encoder_ratio + args.sprint_middle_ratio >= 1.0:
        raise ValueError(
            f"sprint_encoder_ratio ({args.sprint_encoder_ratio}) + "
            f"sprint_middle_ratio ({args.sprint_middle_ratio}) must be < 1.0 "
            f"(need room for decoder blocks)"
        )

    # Validate training steps
    if args.sprint_pretrain_steps < 0:
        raise ValueError(
            f"sprint_pretrain_steps must be >= 0, got {args.sprint_pretrain_steps}"
        )
    if args.sprint_finetune_steps < 0:
        raise ValueError(
            f"sprint_finetune_steps must be >= 0, got {args.sprint_finetune_steps}"
        )
    if args.sprint_warmup_steps < 0:
        raise ValueError(
            f"sprint_warmup_steps must be >= 0, got {args.sprint_warmup_steps}"
        )
    if args.sprint_cooldown_steps < 0:
        raise ValueError(
            f"sprint_cooldown_steps must be >= 0, got {args.sprint_cooldown_steps}"
        )


def _validate_two_stage_training_config(args: argparse.Namespace) -> None:
    """
    Advanced validation for two-stage Sprint training configuration.

    Args:
        args: Argparse namespace with Sprint settings

    Raises:
        SprintConfigurationError: If two-stage training config is invalid
    """
    # Validate two-stage training consistency
    total_sprint_steps = args.sprint_pretrain_steps + args.sprint_finetune_steps
    if total_sprint_steps == 0:
        warnings.warn(
            "Sprint is enabled but no pretrain or finetune steps configured. "
            "Sprint will not be effective. Consider setting sprint_pretrain_steps or sprint_finetune_steps."
        )
    elif args.sprint_warmup_steps > total_sprint_steps:
        warnings.warn(
            f"sprint_warmup_steps ({args.sprint_warmup_steps}) exceeds total Sprint steps "
            f"({total_sprint_steps}). Warmup will be truncated."
        )

    # Validate drop ratio vs two-stage training
    if args.sprint_pretrain_steps > 0 and args.sprint_finetune_steps > 0:
        # Two-stage training - validate that drop ratio makes sense
        if args.sprint_token_drop_ratio < 0.3:
            warnings.warn(
                f"Low token drop ratio ({args.sprint_token_drop_ratio:.2f}) for two-stage training. "
                "Consider using a higher ratio (0.5-0.8) for better Sprint efficiency."
            )
        elif args.sprint_token_drop_ratio > 0.9:
            warnings.warn(
                f"Very high token drop ratio ({args.sprint_token_drop_ratio:.2f}) may impact training quality. "
                "Consider using a lower ratio (0.5-0.8) for better results."
            )


def _validate_sprint_compatibility(args: argparse.Namespace) -> None:
    """
    Validate Sprint compatibility with model configuration and hardware.

    Args:
        args: Argparse namespace with Sprint settings

    Raises:
        SprintCompatibilityError: If Sprint is incompatible with current setup
    """
    # Check for incompatible features
    if hasattr(args, "cpu_offload") and args.cpu_offload:
        raise SprintCompatibilityError(
            "Sprint is incompatible with CPU offload. Please disable cpu_offload to use Sprint.",
            incompatible_feature="cpu_offload",
            alternative="Disable CPU offload in config",
        )

    if hasattr(args, "enable_tread") and args.enable_tread:
        raise SprintCompatibilityError(
            "Sprint is incompatible with TREAD routing. Please disable enable_tread to use Sprint.",
            incompatible_feature="tread_routing",
            alternative="Disable TREAD routing in config",
        )

    if hasattr(args, "controlnet") and args.controlnet:
        raise SprintCompatibilityError(
            "Sprint is incompatible with ControlNet. Please disable controlnet to use Sprint.",
            incompatible_feature="controlnet",
            alternative="Disable ControlNet in config",
        )

    # Check rope_on_the_fly compatibility
    if hasattr(args, "rope_on_the_fly") and args.rope_on_the_fly:
        raise SprintCompatibilityError(
            "Sprint requires cached rotary position embeddings (RoPE). rope_on_the_fly=True is incompatible.",
            incompatible_feature="rope_on_the_fly",
            alternative="Set rope_on_the_fly=False in config",
        )

    # Memory validation (if we can estimate model size)
    if hasattr(args, "hidden_size") and hasattr(args, "num_blocks"):
        try:
            model_config = {
                "hidden_size": args.hidden_size,
                "num_blocks": args.num_blocks,
                "max_sequence_length": getattr(args, "max_sequence_length", 4096),
                "batch_size": getattr(args, "batch_size", 1),
            }

            sprint_config = {
                "token_drop_ratio": args.sprint_token_drop_ratio,
                "encoder_layers": args.sprint_encoder_layers or 6,
                "middle_layers": args.sprint_middle_layers or 6,
            }

            # Validate memory requirements
            validate_sprint_memory_requirements(sprint_config, model_config)

        except Exception as e:
            # Memory validation failed, warn but don't block
            warnings.warn(f"Could not validate Sprint memory requirements: {e}")


def validate_sprint_block_partitioning(
    encoder_layers: Optional[int], middle_layers: Optional[int], total_blocks: int
) -> None:
    """
    Validate Sprint block partitioning against model size.

    Args:
        encoder_layers: Number of encoder layers
        middle_layers: Number of middle layers
        total_blocks: Total number of transformer blocks

    Raises:
        SprintConfigurationError: If block partitioning is invalid
    """
    if encoder_layers is None:
        encoder_layers = max(1, total_blocks // 3)
    if middle_layers is None:
        middle_layers = max(1, total_blocks // 4)

    # Validate partitioning
    total_sprint_layers = encoder_layers + middle_layers

    if total_sprint_layers >= total_blocks:
        raise SprintConfigurationError(
            f"Sprint layers ({total_sprint_layers}) must be less than total blocks ({total_blocks}). "
            f"Reduce encoder_layers ({encoder_layers}) or middle_layers ({middle_layers})",
            config_key="block_partitioning",
            config_value=f"encoder={encoder_layers}, middle={middle_layers}",
            expected=f"total < {total_blocks}",
        )

    # Validate minimum layers
    if encoder_layers < 1:
        raise SprintConfigurationError(
            "encoder_layers must be at least 1 for Sprint to function",
            config_key="encoder_layers",
            config_value=encoder_layers,
            expected=">= 1",
        )

    if middle_layers < 0:
        raise SprintConfigurationError(
            "middle_layers cannot be negative",
            config_key="middle_layers",
            config_value=middle_layers,
            expected=">= 0",
        )

    # Validate reasonable partitioning ratios
    encoder_ratio = encoder_layers / total_blocks
    middle_ratio = middle_layers / total_blocks

    if encoder_ratio > 0.7:
        warnings.warn(
            f"Large encoder ratio ({encoder_ratio:.1%}) may reduce Sprint efficiency. "
            f"Consider reducing encoder_layers from {encoder_layers} to ~{int(total_blocks * 0.5)}."
        )

    if middle_ratio > 0.5:
        warnings.warn(
            f"Large middle ratio ({middle_ratio:.1%}) may reduce Sprint efficiency. "
            f"Consider reducing middle_layers from {middle_layers} to ~{int(total_blocks * 0.3)}."
        )
