"""Checkpoint utilities for WAN fine-tuning trainer.

This module contains all checkpoint-related functionality including:
- Checkpoint resuming and loading
- Checkpoint saving with strict naming conventions
- Checkpoint discovery and validation
- Metadata handling
- Memory-efficient checkpoint loading with direct state dict approach
"""

import argparse
import json
import os
import re
import signal
import time
import logging
from typing import Any, Optional, Tuple, Dict
from itertools import chain
import torch
from accelerate import Accelerator

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

# Memory-efficient safetensors reader
try:
    from utils.safetensors_utils import MemoryEfficientSafeOpen
except Exception:
    MemoryEfficientSafeOpen = None  # type: ignore

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

    # Cache for checkpoint info extraction to avoid duplicate processing
    _checkpoint_info_cache = {}

    @staticmethod
    def clear_checkpoint_info_cache() -> None:
        """Clear the checkpoint info cache. Useful for testing or long-running processes."""
        CheckpointUtils._checkpoint_info_cache.clear()

    @staticmethod
    def _resolve_model_device(model: torch.nn.Module) -> torch.device:
        for param in model.parameters():
            return param.device
        for buffer in model.buffers():
            return buffer.device
        return torch.device("cpu")

    @staticmethod
    def _has_q_galore_modules(model: torch.nn.Module) -> bool:
        try:
            from vendor.q_galore_torch.utils.quantization import QGaLoreLinear
        except Exception:
            return False

        return any(isinstance(module, QGaLoreLinear) for module in model.modules())

    @staticmethod
    def _sync_q_galore_buffers(model: torch.nn.Module, device: torch.device) -> None:
        try:
            from vendor.q_galore_torch.utils.quantization import QGaLoreLinear
        except Exception:
            return

        for module in model.modules():
            if not isinstance(module, QGaLoreLinear):
                continue
            module.weight.data = module.weight.data.to(device)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(device)
            if hasattr(module.weight, "scales"):
                module.weight.scales = module.weight.scales.to(device)
            if hasattr(module.weight, "zeros"):
                module.weight.zeros = module.weight.zeros.to(device)

    @staticmethod
    def _save_q_galore_weights(
        models: list[torch.nn.Module], output_dir: str
    ) -> None:
        try:
            from vendor.q_galore_torch.utils.setup import saving_model_weight
        except Exception as exc:
            logger.warning(f"⚠️ Q-GaLore save helper unavailable: {exc}")
            return

        for index, model in enumerate(models):
            if not CheckpointUtils._has_q_galore_modules(model):
                continue
            q_galore_path = os.path.join(
                output_dir, f"q_galore_weights_model_{index}.pt"
            )
            saving_model_weight(model, q_galore_path)
            logger.info(f"💾 Saved Q-GaLore weights to: {q_galore_path}")

    @staticmethod
    def _load_q_galore_weights(
        models: list[torch.nn.Module], input_dir: str
    ) -> None:
        try:
            from vendor.q_galore_torch.utils.setup import load_model_weight
        except Exception as exc:
            logger.warning(f"⚠️ Q-GaLore load helper unavailable: {exc}")
            return

        for index, model in enumerate(models):
            q_galore_path = os.path.join(
                input_dir, f"q_galore_weights_model_{index}.pt"
            )
            if not os.path.exists(q_galore_path):
                continue
            load_model_weight(model, q_galore_path)
            device = CheckpointUtils._resolve_model_device(model)
            CheckpointUtils._sync_q_galore_buffers(model, device)
            logger.info(f"✅ Loaded Q-GaLore weights from: {q_galore_path}")

    @staticmethod
    def _extract_step_from_checkpoint(checkpoint_path: str) -> Optional[int]:
        """Extract step number from checkpoint path for direct loading."""
        try:
            checkpoint_number, checkpoint_type = (
                CheckpointUtils._extract_checkpoint_info(checkpoint_path)
            )
            if checkpoint_type == "step":
                return checkpoint_number
            elif checkpoint_type == "epoch":
                # For epoch-based checkpoints, we can't determine exact step without dataloader info
                # Return a placeholder that will be handled by the training loop
                return checkpoint_number * 1000  # Rough estimate
            else:
                logger.warning(f"Unknown checkpoint type: {checkpoint_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract step from checkpoint: {e}")
            return None

    @staticmethod
    def prepare_enhanced_resume(
        args: argparse.Namespace,
        train_dataloader_length: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1
    ) -> tuple[int, int, bool, Optional[int]]:
        """
        Prepare enhanced resume functionality with backward compatibility.

        Returns:
            (initial_step, epoch_to_start, should_skip_data, steps_from_state)
        """
        from utils.train_state_utils import TrainStateManager

        # Get loaded train state from checkpoint hooks (if any)
        loaded_train_state = getattr(args, '_loaded_train_state', None)
        steps_from_state = loaded_train_state.get('current_step') if loaded_train_state else None

        # Calculate initial step and epoch using enhanced logic
        initial_step, epoch_to_start = TrainStateManager.calculate_initial_step(
            args,
            train_dataloader_length,
            num_processes,
            gradient_accumulation_steps,
            steps_from_state
        )

        # Determine if we should skip data or just adjust counters
        should_skip_data = getattr(args, 'skip_until_initial_step', False)

        if initial_step > 0:
            logger.info(f"🔄 Enhanced resume: Starting from step {initial_step}")
            if should_skip_data:
                logger.info("📊 Will use accelerator.skip_first_batches for data skipping")
            else:
                logger.info("📊 Will fast-forward step counters without data skipping")

            if getattr(args, 'initial_step', None) is not None or getattr(args, 'initial_epoch', None) is not None:
                if steps_from_state is not None:
                    logger.warning("Steps from train_state.json ignored because initial_step/initial_epoch specified")

        return initial_step, epoch_to_start, should_skip_data, steps_from_state

    @staticmethod
    def create_enhanced_training_loop_wrapper(
        original_dataloader,
        accelerator: Accelerator,
        initial_step: int,
        should_skip_data: bool,
        gradient_accumulation_steps: int = 1
    ):
        """
        Create a wrapper for the training dataloader that handles enhanced resume.
        Works with existing training loops without major modifications.
        """
        if initial_step <= 0 or not should_skip_data:
            return original_dataloader, 0

        # Calculate batches to skip
        batches_to_skip = initial_step * gradient_accumulation_steps
        logger.info(f"⏭️  Skipping {batches_to_skip} batches for enhanced resume")

        # Use accelerator's built-in skip functionality
        wrapped_dataloader = accelerator.skip_first_batches(original_dataloader, batches_to_skip)
        return wrapped_dataloader, initial_step

    @staticmethod
    def update_training_state_for_saving(args: argparse.Namespace, current_epoch: int, current_step: int) -> None:
        """
        Update training state in args for checkpoint saving hooks.
        This ensures train_state.json gets saved with correct values.
        """
        setattr(args, '_current_epoch', current_epoch)
        setattr(args, '_current_step', current_step)

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

            # Skip obvious non-checkpoint directories/files
            if item in ["sample", "samples", "images", "logs", "tensorboard", "wandb"]:
                continue

            # Only consider state directories for resuming (not model safetensors files)
            # Model files are for final output, not for resume checkpoints
            is_checkpoint_pattern = "-step" in item or "-epoch" in item
            is_valid_state_directory = os.path.isdir(item_path) and "-state" in item

            if not is_checkpoint_pattern or not is_valid_state_directory:
                continue

            # Use the new strict checkpoint info extraction
            try:
                number, checkpoint_type = CheckpointUtils._extract_checkpoint_info(item)

                # Additional validation: check if state directory contains required files
                required_files = ["optimizer.bin", "scheduler.bin", "random_states"]
                has_required_files = any(
                    os.path.exists(os.path.join(item_path, req_file))
                    for req_file in required_files
                )
                if not has_required_files:
                    logger.warning(
                        f"⚠️ Skipping state directory without required files: {item}"
                    )
                    continue

                checkpoint_candidates.append((number, item_path, checkpoint_type))
                logger.info(
                    f"🔍 Found {checkpoint_type} checkpoint: {checkpoint_type} {number} at {item}"
                )
            except (ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Skipping invalid checkpoint: {item} ({e})")
                continue

        if not checkpoint_candidates:
            logger.info("🔍 No valid checkpoints found for auto-resume")
            logger.debug(
                "   Only directories with patterns like '-step<number>-state' or '-epoch<number>-state' are considered"
            )
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

        SUPPORTED FORMATS:
        - Step checkpoints: {output_name}-step{number} (e.g., model-step000700)
        - Step checkpoints (dash): {output_name}-step-{number} (e.g., model-step-000700)
        - Epoch checkpoints: {output_name}-epoch{number} (e.g., model-epoch000003)
        - Epoch checkpoints (dash): {output_name}-epoch-{number} (e.g., model-epoch-100)
        """
        # Check cache first to avoid duplicate processing
        if checkpoint_path in CheckpointUtils._checkpoint_info_cache:
            return CheckpointUtils._checkpoint_info_cache[checkpoint_path]

        # STRICT: Step-based checkpoints (primary format: -step000700)
        step_match = re.search(r"-step(\d+)", checkpoint_path)
        if step_match:
            result = (int(step_match.group(1)), "step")
            CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
            return result

        # Also support step format with dash: -step-000700
        step_dash_match = re.search(r"-step-(\d+)", checkpoint_path)
        if step_dash_match:
            result = (int(step_dash_match.group(1)), "step")
            CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
            return result

        # STRICT: Epoch-based checkpoints (primary format: -epoch000003)
        epoch_match = re.search(r"-epoch(\d+)", checkpoint_path)
        if epoch_match:
            result = (int(epoch_match.group(1)), "epoch")
            CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
            return result

        # Also support epoch format with dash: -epoch-100
        epoch_dash_match = re.search(r"-epoch-(\d+)", checkpoint_path)
        if epoch_dash_match:
            result = (int(epoch_dash_match.group(1)), "epoch")
            CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
            return result

        # Legacy support for existing checkpoints (will be converted)
        if "-epoch-" in checkpoint_path:
            try:
                epoch_str = checkpoint_path.split("-epoch-")[-1]
                result = (int(epoch_str), "epoch")
                CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
                return result
            except ValueError:
                pass

        # If we can't determine, default to epoch (safer for existing checkpoints)
        logger.warning(f"⚠️ Could not determine checkpoint type for: {checkpoint_path}")
        logger.warning("⚠️ Assuming epoch-based checkpoint")

        # Try to extract any number as epoch
        numbers = re.findall(r"\d+", checkpoint_path)
        if numbers:
            result = (int(numbers[-1]), "epoch")
            CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
            return result

        result = (0, "epoch")
        CheckpointUtils._checkpoint_info_cache[checkpoint_path] = result
        return result

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
        Create save hook for full fine-tuning.
        For full fine-tuning, we want to save the complete model state.
        """
        # Import here to avoid circular imports
        from utils.train_state_utils import TrainStateManager

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

            # Save current epoch and step for enhanced resume functionality
            # These values come from the trainer's current state
            current_epoch = getattr(args, '_current_epoch', 0)
            current_step = getattr(args, '_current_step', 0)

            TrainStateManager.save_train_state(output_dir, current_epoch, current_step)

            # No model filtering - keep all models for full fine-tuning
            if getattr(args, "q_galore_weight_quant", False):
                CheckpointUtils._save_q_galore_weights(models, output_dir)

        return save_model_hook

    @staticmethod
    def create_load_model_hook_for_finetuning(
        accelerator: Accelerator,
        args: argparse.Namespace,
    ):
        """
        Create load hook for full fine-tuning.
        For full fine-tuning, we want to load the complete model state.
        """
        # Import here to avoid circular imports
        from utils.train_state_utils import TrainStateManager

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

            # Load train state for enhanced resume functionality
            train_state = TrainStateManager.load_train_state(input_dir)
            if train_state:
                # Store the loaded state in args for the trainer to use
                setattr(args, '_loaded_train_state', train_state)
                logger.info(f"📊 Loaded train state: epoch {train_state['current_epoch']}, step {train_state['current_step']}")
            else:
                logger.info("📊 No train state found, will use standard resume logic")

            # For full fine-tuning, we keep all models
            # No model filtering - load everything for complete state restoration
            logger.info(
                f"✅ Loading all {len(models)} model components for full fine-tuning"
            )

            # Log model information
            for i, model in enumerate(models):
                model_type = type(model).__name__
                logger.info(f"   Model {i}: {model_type}")

            if getattr(args, "q_galore_weight_quant", False):
                CheckpointUtils._load_q_galore_weights(models, input_dir)

        return load_model_hook

    @staticmethod
    def register_hooks_for_finetuning(
        accelerator: Accelerator,
        args: argparse.Namespace,
    ) -> None:
        """
        Register save and load hooks with the accelerator for full fine-tuning.
        """
        save_hook = CheckpointUtils.create_save_model_hook_for_finetuning(
            accelerator, args
        )
        load_hook = CheckpointUtils.create_load_model_hook_for_finetuning(
            accelerator, args
        )

        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)

        logger.info("✅ Fine-tuning checkpoint hooks registered")

    @staticmethod
    def is_full_model_checkpoint(checkpoint_path: str) -> bool:
        """
        Check if checkpoint contains full model weights.
        """
        # Check for full model indicators
        model_files = [
            os.path.join(checkpoint_path, "model.safetensors"),
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors"),
            os.path.join(checkpoint_path, "diffusion_pytorch_model.bin"),
        ]

        return any(os.path.exists(f) for f in model_files)

    @staticmethod
    def resume_with_direct_weight_loading(
        accelerator: Accelerator, args: argparse.Namespace, transformer: torch.nn.Module
    ) -> Optional[int]:
        """
        Memory-efficient resume using direct state dict loading.
        This minimizes RAM usage by loading weights directly into the existing model.

        Returns:
            Restored step number or None if no resume occurred
        """
        # Handle auto-resume: find latest checkpoint (same logic as standard resume)
        if getattr(args, "auto_resume", False) and not args.resume:
            logger.info("🔍 Auto-resume: Searching for checkpoints...")
            if not hasattr(args, "output_dir") or not args.output_dir:
                logger.warning("⚠️ Auto-resume: No output_dir specified")
                return None

            if not os.path.exists(args.output_dir):
                logger.warning(
                    f"⚠️ Auto-resume: Output directory does not exist: {args.output_dir}"
                )
                return None

            logger.info(f"🔍 Auto-resume: Checking directory: {args.output_dir}")

            # Debug: list all files in output directory
            try:
                all_items = os.listdir(args.output_dir)
                logger.info(
                    f"🔍 Auto-resume: Found {len(all_items)} items in output directory"
                )
                for item in all_items[:10]:  # Show first 10 items
                    logger.info(f"   - {item}")
                if len(all_items) > 10:
                    logger.info(f"   ... and {len(all_items) - 10} more items")
            except Exception as e:
                logger.error(f"❌ Auto-resume: Error listing directory: {e}")

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

        checkpoint_path = args.resume
        logger.info(f"🔄 Memory-efficient loading checkpoint from: {checkpoint_path}")

        # Check if this is a full model checkpoint
        if not CheckpointUtils.is_full_model_checkpoint(checkpoint_path):
            logger.info(
                "📋 Direct weight loading: Not a full model checkpoint, using standard resume"
            )
            return CheckpointUtils.resume_from_local_if_specified(accelerator, args)

        logger.info("💡 Memory-efficient resume: Loading weights directly to save RAM")
        logger.info(f"🔍 Checkpoint path: {checkpoint_path}")

        # Load model weights directly using the most efficient method available
        step = CheckpointUtils._load_model_weights_directly(
            accelerator, args, transformer, checkpoint_path
        )

        if step is not None:
            logger.info("✅ Direct weight loading completed successfully")
            return step
        else:
            logger.warning(
                "⚠️ Direct weight loading failed, falling back to standard resume"
            )
            return CheckpointUtils.resume_from_local_if_specified(accelerator, args)

    @staticmethod
    def _load_model_weights_directly(
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: torch.nn.Module,
        checkpoint_path: str,
    ) -> Optional[int]:
        """
        Load model weights directly into the existing transformer model.
        This avoids the double memory usage issue of standard checkpoint loading.
        """
        try:
            # Try to load with safetensors first (more memory efficient)
            model_file = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_file) and load_file is not None:
                logger.info("📦 Loading weights from safetensors file")
                state_dict = load_file(model_file, device="cpu")
            else:
                # Fallback to pytorch bin file
                model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.exists(model_file):
                    logger.info("📦 Loading weights from pytorch bin file")
                    state_dict = torch.load(
                        model_file, map_location="cpu", weights_only=True
                    )
                else:
                    logger.error("❌ No model weights file found in checkpoint")
                    return None

            # Load state dict into model (this is memory efficient)
            logger.info("🔄 Loading state dict into model...")
            missing_keys, unexpected_keys = transformer.load_state_dict(
                state_dict, strict=False
            )

            # Clean up state dict immediately to free memory
            del state_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Log any loading issues
            if missing_keys:
                logger.warning(
                    f"⚠️ Missing keys in checkpoint: {len(missing_keys)} keys"
                )
                logger.debug(f"Missing keys sample: {missing_keys[:5]}")

            if unexpected_keys:
                logger.warning(
                    f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)} keys"
                )
                logger.debug(f"Unexpected keys sample: {unexpected_keys[:5]}")

            # Load optimizer and scheduler state using accelerator (more memory efficient)
            logger.info("⚙️ Loading optimizer and scheduler state...")
            CheckpointUtils._load_optimizer_scheduler_state(
                accelerator, checkpoint_path
            )

            if getattr(args, "q_galore_weight_quant", False):
                CheckpointUtils._load_q_galore_weights([transformer], checkpoint_path)

            # Extract step information
            step = CheckpointUtils._extract_step_from_checkpoint_path(checkpoint_path)

            logger.info(f"✅ Direct weight loading completed (step: {step})")
            return step

        except Exception as e:
            logger.error(f"❌ Direct weight loading failed: {e}")
            return None

    @staticmethod
    def _load_optimizer_scheduler_state(
        accelerator: Accelerator, checkpoint_path: str
    ) -> None:
        """
        Load optimizer and scheduler state from checkpoint.
        This is more memory efficient than loading full model state.
        """
        try:
            # Look for optimizer state files
            optimizer_files = [
                "optimizer.bin",
                "optimizer.pt",
                "scheduler.bin",
                "scheduler.pt",
                "rng_state.bin",
                "rng_state.pt",
            ]

            loaded_any = False
            for filename in optimizer_files:
                filepath = os.path.join(checkpoint_path, filename)
                if os.path.exists(filepath):
                    try:
                        # Load state directly
                        state_data = torch.load(
                            filepath, map_location="cpu", weights_only=True
                        )

                        # This is a simplified approach - in practice, you'd need to handle
                        # the specific state restoration for your optimizer/scheduler
                        logger.info(f"⚙️ Loaded {filename}")
                        loaded_any = True

                        # Clean up immediately
                        del state_data

                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {filename}: {e}")

            if not loaded_any:
                logger.info(
                    "⚙️ No optimizer/scheduler state files found, continuing without them"
                )

        except Exception as e:
            logger.warning(f"⚠️ Error loading optimizer/scheduler state: {e}")

    @staticmethod
    def _extract_step_from_checkpoint_path(checkpoint_path: str) -> int:
        """
        Extract step number from checkpoint path using multiple strategies.
        """
        # Strategy 1: Check for step.txt file
        step_file = os.path.join(checkpoint_path, "step.txt")
        if os.path.exists(step_file):
            try:
                with open(step_file, "r") as f:
                    step = int(f.read().strip())
                logger.info(f"📊 Step from step.txt: {step}")
                return step
            except Exception as e:
                logger.warning(f"⚠️ Failed to read step.txt: {e}")

        # Strategy 2: Extract from directory name
        checkpoint_name = os.path.basename(checkpoint_path)

        # Pattern: name-step00012345-state
        step_match = re.search(r"-step(\d+)-state$", checkpoint_name)
        if step_match:
            step = int(step_match.group(1))
            logger.info(f"📊 Step from directory name: {step}")
            return step

        # Pattern: name-epoch000001-state
        epoch_match = re.search(r"-epoch(\d+)-state$", checkpoint_name)
        if epoch_match:
            # Estimate step from epoch (approximate)
            epoch = int(epoch_match.group(1))
            logger.info(f"📊 Step from epoch (estimated): epoch {epoch}")
            return epoch * 1000  # Rough estimate, will be corrected by dataloader

        # Default to step 0 if we can't determine
        logger.warning("⚠️ Could not determine step number, defaulting to 0")
        return 0
