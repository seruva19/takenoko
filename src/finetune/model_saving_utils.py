"""Model saving utilities for WAN fine-tuning trainer.

This module contains model-related functionality including:
- Model saving as safetensors
- Step and epoch-based saving
- Model cleanup operations
- Validation and sampling checks
"""

import argparse
import os
import logging
from typing import Any
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


class ModelSavingUtils:
    """Utility class for model operations."""
    
    def __init__(self, mixed_precision_dtype, full_bf16, fused_backward_pass, mem_eff_save):
        """Initialize with trainer settings."""
        self.mixed_precision_dtype = mixed_precision_dtype
        self.full_bf16 = full_bf16
        self.fused_backward_pass = fused_backward_pass
        self.mem_eff_save = mem_eff_save
    
    def save_model_safetensors(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        model_path: str,
        step: int,
        final: bool = False,
    ):
        """Save model as safetensors using memory-efficient saving."""
        if accelerator.is_main_process:
            save_path = model_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Get transformer state dict DIRECTLY
            unwrapped_transformer = accelerator.unwrap_model(transformer)
            state_dict = unwrapped_transformer.state_dict()

            # Convert to target dtype
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(self.mixed_precision_dtype)

            # Create metadata
            metadata = {
                "step": str(step),
                "finetune_type": "wan_full_finetune",
                "full_bf16": str(self.full_bf16),
                "fused_backward_pass": str(self.fused_backward_pass),
                "mem_eff_save": str(self.mem_eff_save),
                "architecture": "WanFinetune",
            }

            # Save with memory-efficient method if enabled
            if self.mem_eff_save:
                logger.info(f"💾 Using memory-efficient save for {save_path}")
                try:
                    from utils.safetensors_utils import mem_eff_save_file

                    mem_eff_save_file(state_dict, save_path, metadata)
                except ImportError:
                    logger.warning(
                        "⚠️  Memory-efficient save not available, using standard save"
                    )
                    from safetensors.torch import save_file

                    save_file(state_dict, save_path, metadata)
            else:
                # Standard safetensors save
                from safetensors.torch import save_file

                save_file(state_dict, save_path, metadata)

            logger.info(f"Saved full fine-tuning checkpoint to {save_path}")

    def handle_step_saving(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        training_model: torch.nn.Module,
        global_step: int,
    ) -> None:
        """Handle step-based saving with both model and state saving."""
        save_every_n_steps = getattr(args, "save_every_n_steps", None)
        if (
            not save_every_n_steps
            or global_step % save_every_n_steps != 0
            or global_step == 0
        ):
            return

        output_dir = getattr(args, "output_dir", "output")
        output_name = getattr(args, "output_name", "wan_finetune")

        # Save model as safetensors
        model_path = os.path.join(
            output_dir, f"{output_name}-step{global_step:06d}.safetensors"
        )
        self.save_model_safetensors(
            args, accelerator, training_model, model_path, global_step
        )

        # Save state for resuming if enabled
        if getattr(args, "save_state", True):
            StateUtils.save_and_remove_state_stepwise(args, accelerator, global_step)

        # Cleanup old models if save_last_n_steps is set
        self.cleanup_old_step_models(args, global_step)

    def handle_epoch_end_saving(
        self,
        args: argparse.Namespace,
        epoch: int,
        accelerator: Accelerator,
        training_model: torch.nn.Module,
        global_step: int,
    ) -> None:
        """Handle epoch-based saving for full model fine-tuning."""
        save_every_n_epochs = getattr(args, "save_every_n_epochs", None)
        if not save_every_n_epochs or (epoch + 1) % save_every_n_epochs != 0:
            return

        # Import checkpoint utils for checkpoint saving
        from finetune.checkpoint_utils import CheckpointUtils

        # Save full model checkpoint (includes optimizer, scheduler, etc.)
        CheckpointUtils.save_full_model_checkpoint(accelerator, args, epoch + 1, global_step)

        # Also save model as safetensors for inference
        output_dir = getattr(args, "output_dir", "output")
        output_name = getattr(args, "output_name", "wan_finetune")

        model_path = os.path.join(
            output_dir, f"{output_name}-epoch{epoch+1:04d}.safetensors"
        )
        self.save_model_safetensors(
            args, accelerator, training_model, model_path, global_step
        )

        # Cleanup old models if save_last_n_epochs is set
        self.cleanup_old_epoch_models(args, epoch + 1)

        logger.info(f"💾 Saved epoch {epoch+1} checkpoint")

    def cleanup_old_step_models(
        self, args: argparse.Namespace, current_step: int
    ) -> None:
        """Cleanup old step-based model files."""
        save_last_n_steps = getattr(args, "save_last_n_steps", None)
        save_every_n_steps = getattr(args, "save_every_n_steps", None)

        if not save_last_n_steps or not save_every_n_steps:
            return

        from utils.train_utils import get_remove_step_no

        remove_step = get_remove_step_no(args, current_step)

        if remove_step and remove_step > 0:
            output_dir = getattr(args, "output_dir", "output")
            output_name = getattr(args, "output_name", "wan_finetune")
            old_model_path = os.path.join(
                output_dir, f"{output_name}-step{remove_step:06d}.safetensors"
            )

            if os.path.exists(old_model_path):
                os.remove(old_model_path)
                logger.info(f"🗑️ Removed old model: {old_model_path}")

    def cleanup_old_epoch_models(
        self, args: argparse.Namespace, current_epoch: int
    ) -> None:
        """Cleanup old epoch-based model files."""
        save_last_n_epochs = getattr(args, "save_last_n_epochs", None)
        save_every_n_epochs = getattr(args, "save_every_n_epochs", None)

        if not save_last_n_epochs or not save_every_n_epochs:
            return

        from utils.train_utils import get_remove_epoch_no

        remove_epoch = get_remove_epoch_no(args, current_epoch)

        if remove_epoch and remove_epoch > 0:
            output_dir = getattr(args, "output_dir", "output")
            output_name = getattr(args, "output_name", "wan_finetune")
            old_model_path = os.path.join(
                output_dir, f"{output_name}-epoch{remove_epoch:04d}.safetensors"
            )

            if os.path.exists(old_model_path):
                os.remove(old_model_path)
                logger.info(f"🗑️ Removed old epoch model: {old_model_path}")

    @staticmethod
    def should_sample_images(
        args: argparse.Namespace, global_step: int, epoch: int
    ) -> bool:
        """Check if we should sample images at this step/epoch."""
        # Step-based sampling
        if getattr(args, "sample_every_n_steps", None):
            if global_step % args.sample_every_n_steps == 0 and global_step > 0:
                return True

        # Epoch-based sampling
        if getattr(args, "sample_every_n_epochs", None):
            if epoch % args.sample_every_n_epochs == 0:
                return True

        # Sample at first step if configured
        if getattr(args, "sample_at_first", False) and global_step == 0:
            return True

        return False

    @staticmethod
    def should_validate(args: argparse.Namespace, global_step: int) -> bool:
        """Check if we should validate at this step."""
        if getattr(args, "validate_every_n_steps", None):
            if global_step % args.validate_every_n_steps == 0 and global_step > 0:
                return True
        return False

    @staticmethod
    def handle_validation(
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        val_dataset_group: Any,
        global_step: int,
        last_validated_step: int,
    ) -> int:
        """Handle validation during training."""
        if global_step == last_validated_step:
            return last_validated_step
        # TODO: Add validation logic
        return global_step


class StateUtils:
    """Utility class for state management operations."""
    
    @staticmethod
    def save_and_remove_state_stepwise(
        args: argparse.Namespace, accelerator: Accelerator, global_step: int
    ) -> None:
        """Save accelerator state for step-based checkpoints and cleanup old ones."""
        if not accelerator.is_main_process:
            return

        output_dir = getattr(args, "output_dir", "output")
        output_name = getattr(args, "output_name", "wan_finetune")

        # Save current state
        state_dir = os.path.join(output_dir, f"{output_name}-{global_step:06d}-state")
        logger.info(f"💾 Saving training state to: {state_dir}")
        accelerator.save_state(state_dir)

        # Cleanup old states
        StateUtils._cleanup_old_states_stepwise(args, global_step)

    @staticmethod
    def save_and_remove_state_on_epoch_end(
        args: argparse.Namespace, accelerator: Accelerator, epoch: int, global_step: int
    ) -> None:
        """Save accelerator state for epoch-based checkpoints and cleanup old ones."""
        if not accelerator.is_main_process:
            return

        output_dir = getattr(args, "output_dir", "output")
        output_name = getattr(args, "output_name", "wan_finetune")

        # Save current state
        state_dir = os.path.join(output_dir, f"{output_name}-epoch{epoch+1:04d}-state")
        logger.info(f"💾 Saving training state (epoch end) to: {state_dir}")
        accelerator.save_state(state_dir)

        # Cleanup old states
        StateUtils._cleanup_old_states_on_epoch_end(args, epoch + 1)

    @staticmethod
    def _cleanup_old_states_stepwise(
        args: argparse.Namespace, current_step: int
    ) -> None:
        """Remove old step-based state directories."""
        save_last_n_steps = getattr(args, "save_last_n_steps", None)
        save_every_n_steps = getattr(args, "save_every_n_steps", None)

        if not save_last_n_steps or not save_every_n_steps:
            return

        from utils.train_utils import get_remove_step_no

        remove_step = get_remove_step_no(args, current_step)

        if remove_step and remove_step > 0:
            output_dir = getattr(args, "output_dir", "output")
            output_name = getattr(args, "output_name", "wan_finetune")
            old_state_dir = os.path.join(
                output_dir, f"{output_name}-{remove_step:06d}-state"
            )

            if os.path.exists(old_state_dir):
                import shutil
                shutil.rmtree(old_state_dir)
                logger.info(f"🗑️ Removed old state: {old_state_dir}")

    @staticmethod
    def _cleanup_old_states_on_epoch_end(
        args: argparse.Namespace, current_epoch: int
    ) -> None:
        """Remove old epoch-based state directories."""
        save_last_n_epochs = getattr(args, "save_last_n_epochs", None)
        save_every_n_epochs = getattr(args, "save_every_n_epochs", None)

        if not save_last_n_epochs or not save_every_n_epochs:
            return

        from utils.train_utils import get_remove_epoch_no

        remove_epoch = get_remove_epoch_no(args, current_epoch)

        if remove_epoch and remove_epoch > 0:
            output_dir = getattr(args, "output_dir", "output")
            output_name = getattr(args, "output_name", "wan_finetune")
            old_state_dir = os.path.join(
                output_dir, f"{output_name}-epoch{remove_epoch:04d}-state"
            )

            if os.path.exists(old_state_dir):
                import shutil
                shutil.rmtree(old_state_dir)
                logger.info(f"🗑️ Removed old epoch state: {old_state_dir}")