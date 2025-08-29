"""Checkpoint and resume management for WAN network trainer.

This module handles all checkpoint saving, loading, resuming, and state management.
Extracted from wan_network_trainer.py to improve code organization and maintainability.
"""

import argparse
import json
import os
import pathlib
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from accelerate import Accelerator
from safetensors.torch import save_file, load_file
import safetensors
import ast

import logging
from common.logger import get_logger
from common import sai_model_spec
from utils import train_utils

logger = get_logger(__name__, level=logging.INFO)


class CheckpointManager:
    """Handles checkpoint saving, loading, and resume operations."""

    def __init__(self):
        pass

    def check_control_lora_resume_compatibility(
        self, args: argparse.Namespace, transformer: Any, control_signal_processor: Any
    ) -> bool:
        """Check if control LoRA resume is compatible and apply necessary model modifications."""
        if not args.resume:
            return True

        # Check for control LoRA metadata in the resume state
        control_metadata_path = os.path.join(args.resume, "control_lora_metadata.json")
        control_metadata = None

        if os.path.exists(control_metadata_path):
            try:
                with open(control_metadata_path, "r") as f:
                    control_metadata = json.load(f)
                logger.info(
                    f"🔍 Found control LoRA metadata in resume state: {control_metadata_path}"
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to load control LoRA metadata: {e}")
                return True  # Continue without metadata

        current_is_control_lora = getattr(args, "enable_control_lora", False)
        saved_is_control_lora = control_metadata is not None and control_metadata.get(
            "enabled", False
        )

        logger.info(f"🔍 Control LoRA compatibility check:")
        logger.info(f"   Current config: control_lora={current_is_control_lora}")
        logger.info(f"   Saved state: control_lora={saved_is_control_lora}")

        if current_is_control_lora and saved_is_control_lora:
            logger.info(
                "✅ Both current and saved are control LoRA - checking model compatibility"
            )

            # Ensure model is modified for control LoRA BEFORE resuming
            if not getattr(transformer, "_control_lora_patched", False):
                logger.info("🔧 Applying control LoRA model modification for resume...")
                control_signal_processor.modify_model_for_control_lora(
                    transformer, args
                )

            # Validate compatibility
            saved_channels = control_metadata.get("patch_embedding_channels")  # type: ignore
            current_channels = (
                getattr(transformer.patch_embedding, "in_channels", None)
                if hasattr(transformer, "patch_embedding")
                else None
            )

            if (
                saved_channels
                and current_channels
                and saved_channels != current_channels
            ):
                logger.error(
                    f"❌ Control LoRA resume incompatibility: "
                    f"saved model has {saved_channels} patch embedding channels "
                    f"but current model has {current_channels} channels"
                )
                return False

            logger.info("✅ Control LoRA resume compatibility verified")

        elif current_is_control_lora and not saved_is_control_lora:
            logger.error(
                "❌ Cannot resume regular LoRA state into control LoRA training! "
                "The saved state does not contain control LoRA modifications."
            )
            return False

        elif not current_is_control_lora and saved_is_control_lora:
            logger.error(
                "❌ Cannot resume control LoRA state into regular LoRA training! "
                "The saved state contains control LoRA modifications."
            )
            return False
        else:
            logger.info("✅ Both current and saved are regular LoRA")

        return True

    def resume_from_local_if_specified(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: Optional[Any] = None,
        control_signal_processor: Optional[Any] = None,
    ) -> Optional[int]:
        """
        Loads state and returns the restored step count if available, else None.
        """
        if not args.resume and not args.auto_resume:
            return None

        # If auto_resume is enabled and no specific resume path is provided,
        # automatically find the most recent state folder
        if args.auto_resume and not args.resume:
            latest_state_path = self.find_latest_state_folder(args)
            if latest_state_path:
                args.resume = latest_state_path
                logger.info(
                    f"Auto-resume: Found latest state folder: {latest_state_path}"
                )
            else:
                logger.info(
                    "Auto-resume: No existing state folders found, starting fresh training"
                )
                return None

        if not args.resume:
            return None

        logger.info(f"resume training from local state: {args.resume}")

        # Check control LoRA compatibility and apply model modifications if needed
        if transformer is not None and control_signal_processor is not None:
            if not self.check_control_lora_resume_compatibility(
                args, transformer, control_signal_processor
            ):
                logger.error("❌ Control LoRA resume compatibility check failed")
                if args.auto_resume:
                    logger.info(
                        "Auto-resume: Compatibility check failed, starting fresh training"
                    )
                    args.resume = None  # Clear resume path to start fresh
                    return None
                else:
                    raise RuntimeError("Control LoRA resume compatibility check failed")

                # Add error handling for state loading
        try:
            # Add custom optimizer classes to safe globals for PyTorch 2.6+ compatibility
            from optimizers.safe_globals_manager import SafeGlobalsManager

            SafeGlobalsManager.add_custom_optimizer_safe_globals()

            accelerator.load_state(args.resume)
            logger.info(f"Successfully loaded state from: {args.resume}")
            # Try to read step from step.txt
            from utils.train_utils import read_step_from_state_dir

            step = read_step_from_state_dir(args.resume)
            if step is not None:
                logger.info(f"Restored step from step.txt: {step}")
                return step
            # Fallback: parse from directory name
            import re
            import os

            dir_name = os.path.basename(args.resume)
            match = re.search(r"step(\d+)", dir_name)
            if match:
                step = int(match.group(1))
                logger.info(f"Restored step from directory name: {step}")
                return step
            match = re.search(r"-(\d+)-state", dir_name)
            if match:
                step = int(match.group(1))
                logger.info(f"Restored step from directory name: {step}")
                return step
            logger.warning(
                "Could not determine step from state directory; starting from 0"
            )
            return 0
        except Exception as e:
            logger.error(f"Failed to load state from {args.resume}: {e}")
            if args.auto_resume:
                logger.info(
                    "Auto-resume: State loading failed, starting fresh training"
                )
                return None
            else:
                # If manual resume failed, re-raise the exception
                raise

    def find_latest_state_folder(self, args: argparse.Namespace) -> Optional[str]:
        """
        Find the most recent state folder based on the naming convention.
        State folders follow the pattern: {output_name}-{step_number}-state or {output_name}-state
        Returns the path to the most recent state folder, or None if none found.
        """
        output_dir = Path(args.output_dir)
        output_name = args.output_name

        if not output_dir.exists():
            return None

        # Patterns to match numbered state folders
        # 1. Old style: {output_name}-{step_number}-state → wan21_lora-000001-state
        # 2. Explicit step: {output_name}-step{step_number}-state → wan21_lora-step000001-state
        # 3. Explicit epoch: {output_name}-epoch{epoch_number}-state → wan21_lora-epoch000001-state

        numbered_state_pattern = re.compile(  # old style (implicit step)
            rf"^{re.escape(output_name)}-(\d+)-state$"
        )
        step_state_pattern = re.compile(rf"^{re.escape(output_name)}-step(\d+)-state$")
        epoch_state_pattern = re.compile(
            rf"^{re.escape(output_name)}-epoch(\d+)-state$"
        )

        # Pattern to match non-numbered state folders: {output_name}-state
        simple_state_pattern = re.compile(rf"^{re.escape(output_name)}-state$")

        latest_state = None
        latest_step = -1
        simple_state_folder = None

        for item in output_dir.iterdir():
            if item.is_dir():
                # Try all numbered patterns
                numbered_match = numbered_state_pattern.match(item.name)
                step_match = step_state_pattern.match(item.name)
                epoch_match = epoch_state_pattern.match(item.name)

                matched = numbered_match or step_match or epoch_match

                if matched is not None:
                    step_number = int(matched.group(1))
                    if step_number > latest_step:
                        # Validate that the state folder contains the necessary files
                        if self.is_valid_state_folder(item):
                            latest_step = step_number
                            latest_state = str(item)
                        else:
                            logger.warning(
                                f"Skipping invalid numbered state folder: {item.name}"
                            )
                # Check for simple state folders (without numbers)
                elif simple_state_pattern.match(item.name):
                    if self.is_valid_state_folder(item):
                        simple_state_folder = str(item)
                        logger.debug(f"Found simple state folder: {item.name}")
                    else:
                        logger.warning(
                            f"Skipping invalid simple state folder: {item.name}"
                        )

        # Prioritize numbered state folders over simple ones
        if latest_state is not None:
            logger.info(
                f"Found latest numbered state folder: {Path(latest_state).name}"
            )
            return latest_state
        elif simple_state_folder is not None:
            logger.info(f"Found simple state folder: {Path(simple_state_folder).name}")
            return simple_state_folder
        else:
            logger.debug("No valid state folders found")
            return None

    def is_valid_state_folder(self, state_folder: pathlib.Path) -> bool:
        """
        Check if a state folder contains the necessary files for resuming training.
        Returns True if the folder appears to be a valid state folder.
        """
        try:
            # Check if the folder contains the basic accelerate state files
            required_files = [
                "model.safetensors.safetensors",
                "model_1.safetensors",
                "optimizer.bin",
                "random_states_0.pkl",
                "scaler.pt",
                "scheduler.bin",
            ]

            # At least one of the required files should exist
            has_required = any(
                (state_folder / file).exists() for file in required_files
            )

            # Check if it's a directory and not empty
            is_valid = state_folder.is_dir() and has_required

            if not is_valid:
                logger.debug(
                    f"State folder {state_folder.name} appears invalid (missing required files)"
                )

            return is_valid
        except Exception as e:
            logger.debug(f"Error validating state folder {state_folder.name}: {e}")
            return False

    def create_save_model_hook(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: Any,
        network: Any,
    ) -> Callable:
        """Create the save model hook function for accelerator."""

        def save_model_hook(models, weights, output_dir):
            """Enhanced save hook for control LoRA support.

            For control LoRA, we save both the modified transformer model and the LoRA network,
            plus control LoRA metadata for proper resumption.
            For regular LoRA, we save only the LoRA network.
            """

            if not accelerator.is_main_process:
                return

            is_control_lora = getattr(args, "enable_control_lora", False)

            if is_control_lora:
                logger.info(
                    "🔧 Control LoRA save: Saving only LoRA network and modified patch embedding"
                )

                # Save control LoRA metadata for proper resumption
                control_metadata_path = os.path.join(
                    output_dir, "control_lora_metadata.json"
                )
                control_metadata = {
                    "enabled": True,
                    "control_lora_type": getattr(args, "control_lora_type", "tile"),
                    "control_preprocessing": getattr(
                        args, "control_preprocessing", "blur"
                    ),
                    "control_blur_kernel_size": getattr(
                        args, "control_blur_kernel_size", 15
                    ),
                    "control_blur_sigma": getattr(args, "control_blur_sigma", 4.0),
                    "control_scale_factor": getattr(args, "control_scale_factor", 1.0),
                    "input_lr_scale": getattr(args, "input_lr_scale", 1.0),
                    "control_concatenation_dim": getattr(
                        args, "control_concatenation_dim", 0
                    ),
                    "model_modified": getattr(
                        transformer, "_control_lora_patched", False
                    ),
                    "patch_embedding_channels": (
                        getattr(transformer.patch_embedding, "in_channels", None)
                        if hasattr(transformer, "patch_embedding")
                        else None
                    ),
                    "model_in_dim": getattr(transformer, "in_dim", None),
                    "network_module": getattr(
                        args, "network_module", "networks.control_lora_wan"
                    ),
                    "network_dim": getattr(args, "network_dim", 64),
                    "network_alpha": getattr(args, "network_alpha", 64),
                }

                with open(control_metadata_path, "w") as f:
                    json.dump(control_metadata, f, indent=2)

                logger.info(
                    f"💾 Saved control LoRA metadata to: {control_metadata_path}"
                )
                logger.info(f"   Model modified: {control_metadata['model_modified']}")
                logger.info(
                    f"   Patch embedding channels: {control_metadata['patch_embedding_channels']}"
                )

                # Save only the modified patch embedding layer weights
                if hasattr(transformer, "patch_embedding") and getattr(
                    transformer, "_control_lora_patched", False
                ):
                    patch_embedding_path = os.path.join(
                        output_dir, "control_patch_embedding.safetensors"
                    )

                    # Prepare tensors for safetensors
                    patch_embedding_tensors = {
                        "weight": transformer.patch_embedding.weight.detach().cpu(),
                    }

                    # Add bias if it exists
                    if transformer.patch_embedding.bias is not None:
                        patch_embedding_tensors["bias"] = (
                            transformer.patch_embedding.bias.detach().cpu()
                        )

                    # Prepare metadata (safetensors can handle string metadata)
                    metadata = {
                        "in_channels": str(transformer.patch_embedding.in_channels),
                        "out_channels": str(transformer.patch_embedding.out_channels),
                        "kernel_size": str(
                            list(transformer.patch_embedding.kernel_size)
                        ),
                        "stride": str(list(transformer.patch_embedding.stride)),
                        "padding": str(list(transformer.patch_embedding.padding)),
                        "has_bias": str(transformer.patch_embedding.bias is not None),
                    }

                    save_file(
                        patch_embedding_tensors, patch_embedding_path, metadata=metadata
                    )
                    logger.info(
                        f"💾 Saved control patch embedding weights to: {patch_embedding_path}"
                    )
                    logger.info(
                        f"   Channels: {metadata['in_channels']} -> {metadata['out_channels']}"
                    )
                    logger.info(f"   Kernel size: {metadata['kernel_size']}")

                # Remove transformer from models to save, keep only LoRA network
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

                logger.info(
                    "🔧 Removed transformer from save, keeping only LoRA network"
                )
                return

            # Original behaviour (LoRA-only checkpoint) for regular LoRA
            logger.info("🔧 Regular LoRA save: Keeping only LoRA network")
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)

        return save_model_hook

    def create_load_model_hook(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: Any,
        network: Any,
    ) -> Callable:
        """Create the load model hook function for accelerator."""

        def load_model_hook(models, input_dir):
            """Enhanced load hook for control LoRA support."""
            logger.info(f"🔄 Loading state from: {input_dir}")
            logger.info(f"📦 Found {len(models)} models in state")

            # Check for control LoRA metadata first
            control_metadata_path = os.path.join(
                input_dir, "control_lora_metadata.json"
            )
            control_metadata = None

            if os.path.exists(control_metadata_path):
                try:
                    with open(control_metadata_path, "r") as f:
                        control_metadata = json.load(f)
                    logger.info(
                        f"📋 Found control LoRA metadata: {control_metadata_path}"
                    )
                    logger.info(
                        f"   Saved model was control LoRA: {control_metadata.get('enabled', False)}"
                    )
                    logger.info(
                        f"   Saved patch embedding channels: {control_metadata.get('patch_embedding_channels', 'unknown')}"
                    )
                    logger.info(
                        f"   Saved model_in_dim: {control_metadata.get('model_in_dim', 'unknown')}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load control LoRA metadata: {e}")
                    control_metadata = None

            # Log what models we found
            for i, model in enumerate(models):
                model_type = type(model).__name__
                logger.info(f"   Model {i}: {model_type}")

            # Handle control LoRA state loading
            current_is_control_lora = getattr(args, "enable_control_lora", False)
            saved_is_control_lora = (
                control_metadata is not None and control_metadata.get("enabled", False)
            )

            logger.info(f"🔍 Control LoRA state analysis:")
            logger.info(
                f"   Current training is control LoRA: {current_is_control_lora}"
            )
            logger.info(f"   Saved state is control LoRA: {saved_is_control_lora}")

            if current_is_control_lora and not saved_is_control_lora:
                logger.warning(
                    "⚠️  Current training is control LoRA but saved state is not! "
                    "This may cause issues during loading."
                )
            elif not current_is_control_lora and saved_is_control_lora:
                logger.warning(
                    "⚠️  Saved state is control LoRA but current training is not! "
                    "This may cause issues during loading."
                )
            elif current_is_control_lora and saved_is_control_lora:
                logger.info("✅ Both current and saved states are control LoRA")

                # Validate model consistency for control LoRA
                if hasattr(transformer, "patch_embedding"):
                    current_channels = transformer.patch_embedding.in_channels
                    saved_channels = control_metadata.get("patch_embedding_channels")  # type: ignore

                    logger.info(
                        f"   Current patch embedding channels: {current_channels}"
                    )
                    logger.info(f"   Saved patch embedding channels: {saved_channels}")

                    if saved_channels and current_channels != saved_channels:
                        logger.error(
                            f"❌ Patch embedding channel mismatch! "
                            f"Current: {current_channels}, Saved: {saved_channels}"
                        )
                        logger.error(
                            "This indicates model modification wasn't applied consistently."
                        )
                    else:
                        logger.info("✅ Patch embedding channels match")

                    # Check model.in_dim consistency
                    current_in_dim = getattr(transformer, "in_dim", None)
                    saved_in_dim = control_metadata.get("model_in_dim")  # type: ignore

                    if saved_in_dim and current_in_dim != saved_in_dim:
                        logger.error(
                            f"❌ Model in_dim mismatch! "
                            f"Current: {current_in_dim}, Saved: {saved_in_dim}"
                        )
                    else:
                        logger.info("✅ Model in_dim matches")
            else:
                logger.info("✅ Both current and saved states are regular LoRA")

            # Determine how to handle model loading based on control LoRA state
            if current_is_control_lora:
                # For control LoRA, we only need the LoRA network
                # The patch embedding will be loaded separately
                logger.info(
                    "🔧 Control LoRA load: Keeping only LoRA network, loading patch embedding separately"
                )

                # Check if we have the patch embedding weights file
                patch_embedding_path = os.path.join(
                    input_dir, "control_patch_embedding.safetensors"
                )
                if os.path.exists(patch_embedding_path):
                    logger.info(
                        f"📦 Found patch embedding weights: {patch_embedding_path}"
                    )

                    # Load the patch embedding weights
                    try:
                        # Load tensors and metadata
                        patch_embedding_tensors = load_file(patch_embedding_path)

                        # Get metadata from the file
                        with safetensors.safe_open(
                            patch_embedding_path, framework="pt"
                        ) as f:  # type: ignore
                            metadata = f.metadata()

                        # Reconstruct the patch embedding layer
                        if hasattr(transformer, "patch_embedding"):
                            # Get current device and dtype from transformer
                            current_device = next(transformer.parameters()).device
                            current_dtype = next(transformer.parameters()).dtype

                            in_cls = transformer.patch_embedding.__class__

                            # Parse metadata
                            in_channels = int(metadata["in_channels"])
                            out_channels = int(metadata["out_channels"])
                            kernel_size = tuple(
                                ast.literal_eval(metadata["kernel_size"])
                            )
                            stride = tuple(ast.literal_eval(metadata["stride"]))
                            padding = tuple(ast.literal_eval(metadata["padding"]))
                            has_bias = metadata["has_bias"] == "True"

                            # NEW: if the existing patch_embedding already has the same geometry, load
                            # the saved weights directly into it so the previously attached LoRA hooks
                            # remain valid.  If the geometry differs we fall back to the original logic
                            # that rebuilds a fresh Conv3d.
                            pe = (
                                transformer.patch_embedding
                                if hasattr(transformer, "patch_embedding")
                                else None
                            )
                            same_geometry = (
                                pe is not None
                                and pe.in_channels == in_channels
                                and pe.out_channels == out_channels
                                and pe.kernel_size == kernel_size
                                and pe.stride == stride
                                and pe.padding == padding
                                and ((pe.bias is not None) == has_bias)
                            )

                            if same_geometry:
                                logger.info(
                                    "🛠️  Loading weights into existing patch_embedding to keep LoRA hooks intact"
                                )
                                # copy weights (and bias if it exists)
                                pe.weight.data.copy_(  # type: ignore
                                    patch_embedding_tensors["weight"].to(
                                        device=current_device, dtype=current_dtype
                                    )
                                )
                                if has_bias and "bias" in patch_embedding_tensors:
                                    pe.bias.data.copy_(  # type: ignore
                                        patch_embedding_tensors["bias"].to(
                                            device=current_device, dtype=current_dtype
                                        )
                                    )
                                new_patch_embedding = (
                                    pe  # so downstream logging works unmodified
                                )
                                transformer._control_lora_patched = True
                            else:
                                logger.warning(
                                    "Patch embedding geometry differs. Rebuilding layer and re-applying LoRA."
                                )

                                # Create new patch embedding with the loaded parameters as before
                                new_patch_embedding = in_cls(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=has_bias,
                                ).to(device=current_device, dtype=current_dtype)

                                # Load the saved weights
                                new_patch_embedding.weight.data = (
                                    patch_embedding_tensors["weight"].to(
                                        device=current_device, dtype=current_dtype
                                    )
                                )

                                if has_bias and "bias" in patch_embedding_tensors:
                                    new_patch_embedding.bias.data = (
                                        patch_embedding_tensors["bias"].to(
                                            device=current_device, dtype=current_dtype
                                        )
                                    )

                                # Replace the patch embedding and update model dimensions
                                transformer.patch_embedding = new_patch_embedding
                                transformer.in_dim = in_channels

                                # CRITICAL FIX: Re-apply the network to the new patch_embedding
                                # This re-attaches the LoRA hooks to the new layer.
                                network.apply_to(
                                    None,
                                    transformer.patch_embedding,
                                    apply_text_encoder=False,
                                    apply_unet=True,
                                )

                                transformer._control_lora_patched = True

                            # Ensure HuggingFace config stays in sync
                            if hasattr(transformer, "register_to_config"):
                                transformer.register_to_config(
                                    in_dim=transformer.in_dim
                                )

                            logger.info(
                                f"✅ Patch embedding restored: {in_channels} channels (bias={has_bias})"
                            )
                            logger.info(
                                f"   Kernel size: {kernel_size}, Stride: {stride}, Padding: {padding}"
                            )
                        else:
                            logger.warning(
                                "⚠️  Transformer has no patch_embedding attribute"
                            )
                    except Exception as e:
                        logger.error(f"❌ Failed to load patch embedding weights: {e}")
                        logger.error(
                            "Will proceed with current patch embedding configuration"
                        )
                else:
                    logger.warning(
                        f"⚠️  No patch embedding weights found at: {patch_embedding_path}"
                    )
                    logger.warning(
                        "Will proceed with current patch embedding configuration"
                    )

                # Remove everything except the network
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)

            else:
                # For regular LoRA, remove everything except the network
                logger.info("🔧 Regular LoRA load: Keeping only LoRA network")
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)

            logger.info(f"🗑️  Removing {len(remove_indices)} models from loading")
            for i in reversed(remove_indices):
                removed_model = (
                    type(models[i]).__name__ if i < len(models) else "unknown"
                )
                logger.info(f"   Removed: {removed_model}")
                models.pop(i)

            logger.info(f"✅ {len(models)} models will be loaded")

            # Log network state information if available
            for i, model in enumerate(models):
                model_type = type(model).__name__
                logger.info(f"📋 Model {i} ({model_type}):")

                if hasattr(model, "state_dict"):
                    try:
                        state_keys = list(model.state_dict().keys())
                        logger.info(f"Contains {len(state_keys)} state keys 👇")
                        if len(state_keys) > 0:
                            logger.info(f"Sample keys: {state_keys[:3]}")

                        # Log control LoRA specific information for network models
                        if hasattr(model, "control_config"):
                            logger.info(f"Control config: {model.control_config}")
                    except Exception as e:
                        logger.warning(f"❌ Could not inspect state: {e}")

        return load_model_hook

    def create_save_model_function(
        self,
        args: argparse.Namespace,
        metadata: Dict[str, str],
        minimum_metadata: Dict[str, str],
        dit_dtype: torch.dtype,
    ) -> Callable:
        """Create the save model function."""
        save_dtype = dit_dtype

        def save_model(
            ckpt_name: str,
            unwrapped_nw: Any,
            steps: int,
            epoch_no: int,
            force_sync_upload: bool = False,
        ):
            # Safety check for output_dir
            if not args.output_dir or not args.output_dir.strip():
                logger.error(
                    f"args.output_dir is empty or None: '{args.output_dir}'. Cannot save model."
                )
                return

            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            logger.info(f"saving checkpoint: {ckpt_file}")
            metadata["takenoko_training_finished_at"] = str(time.time())
            metadata["takenoko_steps"] = str(steps)
            metadata["takenoko_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = (
                args.metadata_title
                if args.metadata_title is not None
                else args.output_name
            )
            if args.min_timestep is not None or args.max_timestep is not None:
                min_time_step = (
                    args.min_timestep if args.min_timestep is not None else 0
                )
                max_time_step = (
                    args.max_timestep if args.max_timestep is not None else 1000
                )
                md_timesteps = (min_time_step, max_time_step)
            else:
                md_timesteps = None

            sai_metadata = sai_model_spec.build_metadata(
                None,
                time.time(),
                title,
                None,
                args.metadata_author,
                args.metadata_description,
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
            )

            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        return save_model

    def create_remove_model_function(self, args: argparse.Namespace) -> Callable:
        """Create the remove model function."""

        def remove_model(old_ckpt_name: str):
            # Safety check for output_dir
            if not args.output_dir or not args.output_dir.strip():
                logger.error(
                    f"args.output_dir is empty or None: '{args.output_dir}'. Cannot remove model."
                )
                return

            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                logger.info(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        return remove_model

    def register_hooks(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: Any,
        network: Any,
    ) -> None:
        """Register save and load hooks with the accelerator."""
        save_hook = self.create_save_model_hook(accelerator, args, transformer, network)
        load_hook = self.create_load_model_hook(accelerator, args, transformer, network)

        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)
