"""Minimal configuration helpers for WAN network trainer.

All TOML parsing is handled by `takenoko.py`. This module only provides
helpers for metadata creation and sanitized config extraction.
"""

from typing import Any, Dict, Optional
import os
import argparse


class TrainerConfig:
    """Lightweight helper for training metadata and logging config."""

    def __init__(self) -> None:
        pass

    def create_training_metadata(
        self,
        args: argparse.Namespace,
        session_id: int,
        training_started_at: float,
        train_dataset_group: Any,
        num_train_epochs: int,
        num_batches_per_epoch: int,
        optimizer_name: str,
        optimizer_args: str,
    ) -> Dict[str, str]:
        """Create training metadata dictionary."""
        from utils import train_utils

        metadata = {
            "takenoko_session_id": session_id,
            "takenoko_training_started_at": training_started_at,
            "takenoko_output_name": args.output_name,
            "takenoko_learning_rate": args.learning_rate,
            "takenoko_num_train_items": train_dataset_group.num_train_items,
            "takenoko_num_batches_per_epoch": num_batches_per_epoch,
            "takenoko_num_epochs": num_train_epochs,
            "takenoko_gradient_checkpointing": args.gradient_checkpointing,
            "takenoko_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "takenoko_max_train_steps": args.max_train_steps,
            "takenoko_lr_warmup_steps": args.lr_warmup_steps,
            "takenoko_lr_scheduler": args.lr_scheduler,
            train_utils.TAKENOKO_METADATA_KEY_BASE_MODEL_VERSION: args.target_model,
            train_utils.TAKENOKO_METADATA_KEY_NETWORK_MODULE: args.network_module,
            train_utils.TAKENOKO_METADATA_KEY_NETWORK_DIM: args.network_dim,
            train_utils.TAKENOKO_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
            "takenoko_network_dropout": args.network_dropout,
            "takenoko_mixed_precision": args.mixed_precision,
            "takenoko_seed": args.seed,
            "takenoko_training_comment": args.training_comment,
            "takenoko_optimizer": optimizer_name
            + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "takenoko_max_grad_norm": args.max_grad_norm,
        }

        # Add model hashes/names
        if args.dit is not None:
            dit_name = args.dit
            if os.path.exists(dit_name):
                dit_name = os.path.basename(dit_name)
            metadata["takenoko_dit_name"] = dit_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                vae_name = os.path.basename(vae_name)
            metadata["takenoko_vae_name"] = vae_name

        # Add serialized config content to metadata
        if hasattr(args, "config_content") and args.config_content is not None:
            # Check if config embedding is enabled (default to True)
            embed_config = getattr(args, "embed_config_in_metadata", True)
            if embed_config:
                metadata["takenoko_config_content"] = args.config_content
                if hasattr(args, "config_file") and args.config_file is not None:
                    metadata["takenoko_config_file"] = os.path.basename(
                        args.config_file
                    )

        # Convert all values to strings
        metadata = {k: str(v) for k, v in metadata.items()}

        return metadata

    def get_sanitized_config_or_none(
        self, args: argparse.Namespace
    ) -> Optional[Dict[str, Any]]:
        """Get sanitized config for logging, or None if logging is disabled."""
        from utils import train_utils

        return train_utils.get_sanitized_config_or_none(args)
