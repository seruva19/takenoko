"""Checkpoint utilities for WAN fine-tuning trainer.

This module contains all checkpoint-related functionality including:
- Checkpoint resuming and loading
- Checkpoint saving with strict naming conventions
- Checkpoint discovery and validation
- Metadata handling
"""

import argparse
import json
import os
import re
import signal
import time
import logging
from typing import Any, Optional, Tuple
import torch
from accelerate import Accelerator

try:
    from common.logger import get_logger
except ImportError:
    # Fallback for testing or different import contexts
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger


logger = get_logger(__name__, level=logging.INFO)


class CheckpointUtils:
    """Utility class for checkpoint operations."""

    @staticmethod
    def resume_from_local_if_specified(
        accelerator: Accelerator, args: argparse.Namespace
    ) -> Optional[int]:
        """Simple local checkpoint resume with auto-resume support."""
        # Handle auto-resume: find latest checkpoint
        if getattr(args, "auto_resume", False) and not args.resume:
            latest_checkpoint = CheckpointUtils._find_latest_checkpoint(args)
            if latest_checkpoint:
                args.resume = latest_checkpoint
                logger.info(
                    f"🔍 Auto-resume: Found latest checkpoint: {latest_checkpoint}"
                )
            else:
                logger.info("🔍 Auto-resume: No existing checkpoints found")
                return None

        if not args.resume:
            return None

        logger.info(f"🔄 Loading checkpoint from: {args.resume}")

        # First, check if the checkpoint directory exists and has the expected files
        if not os.path.exists(args.resume):
            logger.error(f"❌ Checkpoint directory does not exist: {args.resume}")
            return None

        # Validate checkpoint consistency to catch step/epoch mixing issues early
        if not CheckpointUtils.validate_checkpoint_consistency(args.resume):
            logger.error(
                f"❌ Checkpoint validation failed - refusing to load potentially corrupted checkpoint"
            )
            return None

        logger.info(f"🔄 Checkpoint directory contents:")
        try:
            for item in os.listdir(args.resume):
                item_path = os.path.join(args.resume, item)
                size = (
                    os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
                )
                logger.info(f"   - {item} ({size} bytes)")
        except Exception as e:
            logger.warning(f"Could not list checkpoint directory: {e}")

        try:
            logger.info("🔄 Calling accelerator.load_state()...")

            # Try loading with a timeout mechanism
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    "accelerator.load_state() timed out after 60 seconds"
                )

            # Set up timeout (60 seconds)
            if hasattr(signal, "SIGALRM"):  # Unix systems
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)

            start_time = time.time()
            accelerator.load_state(args.resume)
            end_time = time.time()

            if hasattr(signal, "SIGALRM"):  # Cancel timeout
                signal.alarm(0)

            logger.info(
                f"🔄 accelerator.load_state() completed successfully in {end_time - start_time:.2f} seconds"
            )

            # Try to read step from step.txt first (matching LoRA approach exactly)
            try:
                from utils.train_utils import read_step_from_state_dir

                step = read_step_from_state_dir(args.resume)
                if step is not None:
                    logger.info(f"Restored step from step.txt: {step}")
                    return step
            except ImportError:
                logger.warning(
                    "⚠️ read_step_from_state_dir not available, using fallback"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to read step from step.txt: {e}")

            # Fallback: manual step.txt reading
            step_file = os.path.join(args.resume, "step.txt")
            if os.path.exists(step_file):
                try:
                    with open(step_file, "r") as f:
                        step = int(f.read().strip())
                    logger.info(f"Restored step from step.txt (manual): {step}")
                    return step
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read step.txt manually: {e}")

            logger.info("🔄 Extracting checkpoint information from directory name...")
            checkpoint_number, checkpoint_type = (
                CheckpointUtils._extract_checkpoint_info(args.resume)
            )
            logger.info(f"🔄 Extracted {checkpoint_type}: {checkpoint_number}")

            # Store both pieces of information for proper handling
            restored_step = checkpoint_number  # Keep for compatibility
            checkpoint_info = (checkpoint_number, checkpoint_type)

            logger.info("✅ Successfully loaded checkpoint")
            return restored_step
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.exception("Full exception details:")

            # If epoch-100 fails, try falling back to earlier checkpoints
            if "epoch-100" in args.resume:
                logger.info("🔄 Trying to fall back to earlier checkpoints...")
                base_path = args.resume.replace("-epoch-100", "")

                for fallback_epoch in [50, 20, 10]:
                    fallback_path = f"{base_path}-epoch-{fallback_epoch}"
                    if os.path.exists(fallback_path):
                        logger.info(f"🔄 Trying fallback checkpoint: {fallback_path}")
                        try:
                            start_time = time.time()
                            accelerator.load_state(fallback_path)
                            end_time = time.time()
                            logger.info(
                                f"✅ Successfully loaded fallback checkpoint from epoch {fallback_epoch} in {end_time - start_time:.2f} seconds"
                            )
                            return fallback_epoch
                        except Exception as fallback_e:
                            logger.warning(
                                f"❌ Fallback checkpoint also failed: {fallback_e}"
                            )
                            continue

                logger.error("❌ All checkpoint fallbacks failed")

            # Don't return None - let's see if we can continue without the checkpoint
            logger.warning(
                "⚠️ Continuing with fresh training due to checkpoint loading failure"
            )
            return None

    @staticmethod
    def handle_full_model_checkpoint_resume(
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: torch.nn.Module,
        text_encoder: Optional[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
    ) -> Optional[int]:
        """
        Handle checkpoint resume for model fine-tuning.
        This loads the complete model state, optimizer, and scheduler.
        """
        if not args.resume and not args.auto_resume:
            return None

        # Auto-resume: find latest checkpoint
        if args.auto_resume and not args.resume:
            latest_checkpoint = CheckpointUtils._find_latest_full_model_checkpoint(args)
            if latest_checkpoint:
                args.resume = latest_checkpoint
                logger.info(
                    f"Auto-resume: Found latest checkpoint: {latest_checkpoint}"
                )
            else:
                logger.info(
                    "Auto-resume: No existing checkpoints found, starting fresh"
                )
                return None

        if not args.resume:
            return None

        logger.info(f"🔄 Loading FULL MODEL checkpoint from: {args.resume}")

        try:
            # Load the checkpoint using accelerate's built-in mechanism
            accelerator.load_state(args.resume)

            # Extract step number from checkpoint directory name
            restored_step = CheckpointUtils._extract_step_from_checkpoint_path(
                args.resume
            )

            logger.info(f"✅ Successfully loaded full model checkpoint")
            logger.info(f"   - Transformer parameters: ✅ Loaded")
            if text_encoder is not None:
                logger.info(f"   - Text encoder parameters: ✅ Loaded")
            logger.info(f"   - Optimizer state: ✅ Loaded")
            logger.info(f"   - Scheduler state: ✅ Loaded")

            return restored_step

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("Continuing with fresh training...")
            return None

    @staticmethod
    def _find_latest_checkpoint(args: argparse.Namespace) -> Optional[str]:
        """Find the latest checkpoint directory (both epoch-based and step-based)."""
        if not args.output_dir or not os.path.exists(args.output_dir):
            return None

        checkpoint_candidates = []

        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if not os.path.isdir(item_path):
                continue

            # Use the new strict checkpoint info extraction
            try:
                number, checkpoint_type = CheckpointUtils._extract_checkpoint_info(item)
                checkpoint_candidates.append((number, item_path, checkpoint_type))
                logger.info(
                    f"🔍 Found {checkpoint_type} checkpoint: {checkpoint_type} {number} at {item}"
                )
            except (ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Skipping invalid checkpoint directory: {item} ({e})")
                continue

        if not checkpoint_candidates:
            logger.info("🔍 No checkpoint directories found for auto-resume")
            return None

        # Separate step and epoch checkpoints for proper handling
        step_checkpoints = [
            (num, path, ctype)
            for num, path, ctype in checkpoint_candidates
            if ctype == "step"
        ]
        epoch_checkpoints = [
            (num, path, ctype)
            for num, path, ctype in checkpoint_candidates
            if ctype == "epoch"
        ]

        logger.info(f"📊 Checkpoint summary:")
        logger.info(f"   - Step checkpoints found: {len(step_checkpoints)}")
        logger.info(f"   - Epoch checkpoints found: {len(epoch_checkpoints)}")

        # SAFER: Prefer the highest-numbered checkpoint within each type, then by modification time
        # This prevents issues where a step-100 checkpoint might be newer than epoch-50 but represents less training

        # Group by checkpoint type and find the highest number in each type
        step_candidates = [
            (num, path, ctype, os.path.getmtime(path))
            for num, path, ctype in checkpoint_candidates
            if ctype == "step"
        ]
        epoch_candidates = [
            (num, path, ctype, os.path.getmtime(path))
            for num, path, ctype in checkpoint_candidates
            if ctype == "epoch"
        ]

        # Find the best candidate from each type
        best_step = (
            max(step_candidates, key=lambda x: (x[0], x[3]))
            if step_candidates
            else None
        )  # highest step, then newest
        best_epoch = (
            max(epoch_candidates, key=lambda x: (x[0], x[3]))
            if epoch_candidates
            else None
        )  # highest epoch, then newest

        # Choose between step and epoch based on logical progression and training time
        candidates_to_compare = []
        if best_step:
            candidates_to_compare.append(best_step)
        if best_epoch:
            candidates_to_compare.append(best_epoch)

        if not candidates_to_compare:
            logger.error("❌ No accessible checkpoints found")
            return None

        # Select the most recent by modification time from the best candidates
        latest_num, latest_path, checkpoint_type, _ = max(
            candidates_to_compare, key=lambda x: x[3]
        )

        logger.info(f"🔍 Selected latest {checkpoint_type}-based checkpoint:")
        logger.info(f"   - Type: {checkpoint_type}")
        logger.info(f"   - Number: {latest_num}")
        logger.info(f"   - Path: {latest_path}")

        return latest_path

    @staticmethod
    def _find_latest_full_model_checkpoint(args: argparse.Namespace) -> Optional[str]:
        """Find the latest full model checkpoint directory (legacy method)."""
        # Delegate to the new comprehensive method
        return CheckpointUtils._find_latest_checkpoint(args)

    @staticmethod
    def _extract_checkpoint_info(checkpoint_path: str) -> Tuple[int, str]:
        """
        Extract checkpoint information with STRICT separation between steps and epochs.

        Returns:
            tuple[int, str]: (number, type) where type is 'step' or 'epoch'

        NEW CLEAR FORMATS:
        - Step checkpoints: {output_name}-step-{number} (e.g., model-step-000700)
        - Epoch checkpoints: {output_name}-epoch-{number} (e.g., model-epoch-100)
        """
        # STRICT: Step-based checkpoints must have "-step-" pattern
        step_match = re.search(r"-step-(\d+)", checkpoint_path)
        if step_match:
            return int(step_match.group(1)), "step"

        # STRICT: Epoch-based checkpoints must have "-epoch-" pattern
        epoch_match = re.search(r"-epoch-(\d+)", checkpoint_path)
        if epoch_match:
            return int(epoch_match.group(1)), "epoch"

        # Legacy support for existing checkpoints (will be converted)
        if "-epoch-" in checkpoint_path:
            try:
                epoch_str = checkpoint_path.split("-epoch-")[-1]
                return int(epoch_str), "epoch"
            except ValueError:
                pass

        # If we can't determine, default to epoch (safer for existing checkpoints)
        logger.warning(f"⚠️ Could not determine checkpoint type for: {checkpoint_path}")
        logger.warning("⚠️ Assuming epoch-based checkpoint")

        # Try to extract any number as epoch
        numbers = re.findall(r"\d+", checkpoint_path)
        if numbers:
            return int(numbers[-1]), "epoch"

        return 0, "epoch"

    @staticmethod
    def validate_checkpoint_consistency(
        checkpoint_path: str, expected_type: str = None
    ) -> bool:
        """
        Validate checkpoint consistency and detect potential step/epoch mixing issues.

        Args:
            checkpoint_path: Path to checkpoint directory
            expected_type: Expected checkpoint type ('step' or 'epoch'), None for auto-detect

        Returns:
            bool: True if checkpoint is consistent, False if issues detected
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint path does not exist: {checkpoint_path}")
            return False

        try:
            # Extract checkpoint info from path
            checkpoint_number, detected_type = CheckpointUtils._extract_checkpoint_info(
                checkpoint_path
            )

            # Check for metadata file
            metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Could not read metadata: {e}")

            # Validate type consistency
            if expected_type and detected_type != expected_type:
                logger.error(f"❌ Checkpoint type mismatch!")
                logger.error(f"   Expected: {expected_type}")
                logger.error(f"   Detected from path: {detected_type}")
                return False

            # Check metadata consistency
            if metadata:
                metadata_type = metadata.get("checkpoint_type")
                if metadata_type and metadata_type != detected_type:
                    logger.error(f"❌ Metadata type mismatch!")
                    logger.error(f"   Path indicates: {detected_type}")
                    logger.error(f"   Metadata says: {metadata_type}")
                    return False

                # Check number consistency
                if detected_type == "epoch":
                    metadata_epoch = metadata.get("epoch", 0)
                    if metadata_epoch != checkpoint_number:
                        logger.warning(
                            f"⚠️ Epoch number mismatch: path={checkpoint_number}, metadata={metadata_epoch}"
                        )
                elif detected_type == "step":
                    metadata_step = metadata.get("global_step", 0)
                    if metadata_step != checkpoint_number:
                        logger.warning(
                            f"⚠️ Step number mismatch: path={checkpoint_number}, metadata={metadata_step}"
                        )

            # Check for required accelerate files
            required_files = [
                "model.safetensors",
                "optimizer.bin",
                "scheduler.bin",
                "random_states_0.pkl",
            ]
            missing_files = []
            for required_file in required_files:
                if not os.path.exists(os.path.join(checkpoint_path, required_file)):
                    missing_files.append(required_file)

            if missing_files:
                logger.warning(f"⚠️ Missing checkpoint files: {missing_files}")
                # Don't fail for missing files, as they might use different names

            logger.info(f"✅ Checkpoint validation passed:")
            logger.info(f"   Type: {detected_type}")
            logger.info(f"   Number: {checkpoint_number}")
            logger.info(f"   Metadata consistent: {bool(metadata)}")

            return True

        except Exception as e:
            logger.error(f"❌ Checkpoint validation failed: {e}")
            return False

    @staticmethod
    def save_full_model_checkpoint(
        accelerator: Accelerator,
        args: argparse.Namespace,
        epoch: int,
        global_step: int,
        checkpoint_type: str = "epoch",
    ) -> None:
        """
        Save full model checkpoint with STRICT naming conventions.
        Uses LoRA-style approach with accelerator.save_state() for consistency.

        Args:
            checkpoint_type: 'epoch' or 'step' - determines naming format
        """
        if not accelerator.is_main_process:
            return

        try:
            # Create checkpoint directory with STRICT naming
            if checkpoint_type == "epoch":
                checkpoint_dir = os.path.join(
                    args.output_dir, f"{args.output_name}-epoch-{epoch}"
                )
                logger.info(f"💾 Saving EPOCH checkpoint to: {checkpoint_dir}")
                logger.info(f"   - Epoch: {epoch}, Global step: {global_step}")
            elif checkpoint_type == "step":
                checkpoint_dir = os.path.join(
                    args.output_dir, f"{args.output_name}-step-{global_step:06d}"
                )
                logger.info(f"💾 Saving STEP checkpoint to: {checkpoint_dir}")
                logger.info(f"   - Step: {global_step}, Epoch: {epoch}")
            else:
                raise ValueError(
                    f"Invalid checkpoint_type: {checkpoint_type}. Must be 'epoch' or 'step'"
                )

            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save using accelerate's built-in mechanism (handles all states)
            # This is the SAME approach as LoRA - accelerator.save_state() with hooks
            accelerator.save_state(checkpoint_dir)

            # Save comprehensive metadata with checkpoint type
            # This matches LoRA's metadata approach
            metadata = {
                "checkpoint_type": checkpoint_type,
                "epoch": epoch,
                "global_step": global_step,
                "training_type": "full_finetune",
                "model_type": "WanModel",
                "timestamp": time.time(),
                "format_version": "2.0",  # New strict format
            }

            metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save step info for easy reading (matching LoRA approach)
            if checkpoint_type == "step":
                step_file = os.path.join(checkpoint_dir, "step.txt")
                with open(step_file, "w") as f:
                    f.write(str(global_step))

            # Also save epoch info in a separate file for easy reading
            if checkpoint_type == "epoch":
                epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
                with open(epoch_file, "w") as f:
                    f.write(str(epoch))

            # Always save step.txt file for consistency with LoRA approach
            # This enables proper step restoration during resume
            step_file = os.path.join(checkpoint_dir, "step.txt")
            with open(step_file, "w") as f:
                f.write(str(global_step))
            logger.info(f"📝 Saved step information: {global_step}")

            logger.info(f"✅ {checkpoint_type.upper()} checkpoint saved successfully")

            # Cleanup old checkpoints according to save_last_n_epochs/save_last_n_steps
            CheckpointUtils._cleanup_old_checkpoints(
                args, epoch, global_step, checkpoint_type
            )

        except Exception as e:
            logger.error(f"❌ Failed to save {checkpoint_type} checkpoint: {e}")

    @staticmethod
    def _cleanup_old_checkpoints(
        args: argparse.Namespace,
        current_epoch: int,
        current_step: int,
        checkpoint_type: str,
    ) -> None:
        """
        Clean up old checkpoint directories according to save_last_n_epochs/save_last_n_steps.

        This function removes old full checkpoint directories (not just the model files)
        to respect the user's save_last_n_epochs configuration.
        """
        if not hasattr(args, "output_dir") or not args.output_dir:
            return

        try:
            if checkpoint_type == "epoch":
                save_last_n = getattr(args, "save_last_n_epochs", None)
                save_every_n = getattr(args, "save_every_n_epochs", None)

                if not save_last_n or not save_every_n:
                    return

                # Only clean up if we're at a checkpoint saving epoch
                if current_epoch % save_every_n != 0:
                    return

                # Calculate which epoch checkpoint to remove
                # We want to keep the last N checkpoints, so remove the (N+1)th oldest
                remove_epoch = current_epoch - (save_every_n * save_last_n)

                # Ensure the remove epoch is also a valid checkpoint epoch
                if remove_epoch <= 0 or remove_epoch % save_every_n != 0:
                    return  # Nothing to remove yet

                # Find old checkpoint directory to remove
                old_checkpoint_dir = os.path.join(
                    args.output_dir, f"{args.output_name}-epoch-{remove_epoch}"
                )

                if os.path.exists(old_checkpoint_dir):
                    import shutil

                    shutil.rmtree(old_checkpoint_dir)
                    logger.info(f"🗑️ Removed old epoch checkpoint: {old_checkpoint_dir}")

            elif checkpoint_type == "step":
                save_last_n = getattr(args, "save_last_n_steps", None)
                save_every_n = getattr(args, "save_every_n_steps", None)

                if not save_last_n or not save_every_n:
                    return

                # Only clean up if we're at a checkpoint saving step
                if current_step % save_every_n != 0:
                    return

                # Calculate which step checkpoint to remove
                # We want to keep the last N checkpoints, so remove the (N+1)th oldest
                remove_step = current_step - (save_every_n * save_last_n)

                # Ensure the remove step is also a valid checkpoint step
                if remove_step <= 0 or remove_step % save_every_n != 0:
                    return  # Nothing to remove yet

                # Find old checkpoint directory to remove
                old_checkpoint_dir = os.path.join(
                    args.output_dir, f"{args.output_name}-step-{remove_step:06d}"
                )

                if os.path.exists(old_checkpoint_dir):
                    import shutil

                    shutil.rmtree(old_checkpoint_dir)
                    logger.info(f"🗑️ Removed old step checkpoint: {old_checkpoint_dir}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup old checkpoint: {e}")
            # Don't fail the entire checkpoint save if cleanup fails

    @staticmethod
    def _extract_step_from_checkpoint_path(checkpoint_path: str) -> int:
        """Extract step number from checkpoint path."""
        # Try to extract step from step-based checkpoints first
        step_match = re.search(r"-step-(\d+)", checkpoint_path)
        if step_match:
            return int(step_match.group(1))

        # Fall back to epoch extraction for epoch-based checkpoints
        epoch_match = re.search(r"-epoch-(\d+)", checkpoint_path)
        if epoch_match:
            # For epoch checkpoints, we don't have the exact step, return 0
            return 0

        # If no pattern matches, return 0
        return 0

    @staticmethod
    def create_save_model_hook_for_finetuning(
        accelerator: Accelerator,
        args: argparse.Namespace,
    ):
        """
        Create save hook for full fine-tuning (similar to LoRA approach).
        For full fine-tuning, we want to save the complete model state.
        """

        def save_model_hook(models, weights, output_dir):
            """Save hook for full fine-tuning - saves all model states."""
            if not accelerator.is_main_process:
                return

            logger.info(
                f"🔧 Full fine-tuning save: Saving complete model state to {output_dir}"
            )

            # For full fine-tuning, we keep all models and weights
            # This ensures optimizer, scheduler, and model states are all saved
            logger.info(f"📦 Saving {len(models)} model components")

            # Save fine-tuning metadata for proper loading
            finetune_metadata = {
                "training_type": "full_finetune",
                "model_type": "WanModel",
                "saved_models_count": len(models),
                "format_version": "2.0",
                "timestamp": time.time(),
            }

            finetune_metadata_path = os.path.join(output_dir, "finetune_metadata.json")
            with open(finetune_metadata_path, "w") as f:
                json.dump(finetune_metadata, f, indent=2)

            logger.info(f"💾 Saved fine-tuning metadata to: {finetune_metadata_path}")
            # No model filtering - keep all models for full fine-tuning

        return save_model_hook

    @staticmethod
    def create_load_model_hook_for_finetuning(
        accelerator: Accelerator,
        args: argparse.Namespace,
    ):
        """
        Create load hook for full fine-tuning (similar to LoRA approach).
        For full fine-tuning, we want to load the complete model state.
        """

        def load_model_hook(models, input_dir):
            """Load hook for full fine-tuning - loads all model states."""
            logger.info(
                f"🔄 Full fine-tuning load: Loading complete model state from {input_dir}"
            )
            logger.info(f"📦 Found {len(models)} model components to load")

            # Check for fine-tuning metadata
            finetune_metadata_path = os.path.join(input_dir, "finetune_metadata.json")
            finetune_metadata = None

            if os.path.exists(finetune_metadata_path):
                try:
                    with open(finetune_metadata_path, "r") as f:
                        finetune_metadata = json.load(f)
                    logger.info(
                        f"📋 Found fine-tuning metadata: {finetune_metadata_path}"
                    )
                    logger.info(
                        f"   Training type: {finetune_metadata.get('training_type', 'unknown')}"
                    )
                    logger.info(
                        f"   Saved models: {finetune_metadata.get('saved_models_count', 'unknown')}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load fine-tuning metadata: {e}")

            # For full fine-tuning, we keep all models
            # No model filtering - load everything for complete state restoration
            logger.info(
                f"✅ Loading all {len(models)} model components for full fine-tuning"
            )

            # Log model information
            for i, model in enumerate(models):
                model_type = type(model).__name__
                logger.info(f"   Model {i}: {model_type}")

        return load_model_hook

    @staticmethod
    def register_hooks_for_finetuning(
        accelerator: Accelerator,
        args: argparse.Namespace,
    ) -> None:
        """
        Register save and load hooks with the accelerator for full fine-tuning.
        This makes fine-tuning checkpoint handling identical to LoRA.
        """
        save_hook = CheckpointUtils.create_save_model_hook_for_finetuning(
            accelerator, args
        )
        load_hook = CheckpointUtils.create_load_model_hook_for_finetuning(
            accelerator, args
        )

        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)

        logger.info(
            "✅ Fine-tuning checkpoint hooks registered (matching LoRA approach)"
        )
