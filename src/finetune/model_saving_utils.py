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

    def __init__(
        self, mixed_precision_dtype, full_bf16, fused_backward_pass, mem_eff_save
    ):
        """Initialize with trainer settings."""
        self.mixed_precision_dtype = mixed_precision_dtype
        self.full_bf16 = full_bf16
        self.fused_backward_pass = fused_backward_pass
        self.mem_eff_save = mem_eff_save

    @staticmethod
    def resolve_bf16_checkpoint(dit_path: str, accelerator: Accelerator) -> str:
        """
        Resolve BF16 checkpoint path, converting if necessary.
        Returns the path to use for loading.
        """
        # Extract filename from path (works for URLs and local paths)
        if dit_path.startswith("http"):
            filename = dit_path.split("/")[-1]
        else:
            filename = os.path.basename(dit_path)

        # If already BF16, use as-is
        if filename.startswith("bf16_"):
            if accelerator.is_main_process:
                logger.info(f"🔄 Using existing BF16 checkpoint: {dit_path}")
            return dit_path

        # Generate BF16 filename and local path
        bf16_filename = f"bf16_{filename}"
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        bf16_local_path = os.path.join(models_dir, bf16_filename)

        # If BF16 version already exists, use it
        if os.path.exists(bf16_local_path):
            if accelerator.is_main_process:
                logger.info(f"🔄 Using cached BF16 checkpoint: {bf16_local_path}")
            return bf16_local_path

        # Need to convert - download original first if it's a URL
        if dit_path.startswith("http"):
            if accelerator.is_main_process:
                logger.info(f"🔄 Downloading and converting {filename} to BF16...")
                # Use existing model loading mechanism to download
                original_local_path = os.path.join(models_dir, filename)
                ModelSavingUtils._download_and_convert_to_bf16(
                    dit_path, original_local_path, bf16_local_path
                )
            else:
                # Non-main processes wait for conversion to complete
                ModelSavingUtils._wait_for_bf16_conversion(bf16_local_path)
        else:
            # Local file - convert directly
            if accelerator.is_main_process:
                logger.info(f"🔄 Converting local checkpoint {filename} to BF16...")
                ModelSavingUtils._convert_checkpoint_to_bf16(dit_path, bf16_local_path)
            else:
                # Non-main processes wait for conversion
                ModelSavingUtils._wait_for_bf16_conversion(bf16_local_path)

        return bf16_local_path

    @staticmethod
    def _download_and_convert_to_bf16(
        url: str, original_path: str, bf16_path: str
    ) -> None:
        """Download checkpoint from URL and convert to BF16."""
        # Use existing model downloading mechanism
        try:
            from utils.model_utils import load_file_from_url

            load_file_from_url(url, original_path)
            ModelSavingUtils._convert_checkpoint_to_bf16(original_path, bf16_path)
            # Optionally remove original to save space
            if os.path.exists(original_path):
                os.remove(original_path)
                logger.info(f"🗑️ Removed original checkpoint: {original_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download and convert checkpoint: {e}")
            raise

    @staticmethod
    def _convert_checkpoint_to_bf16(input_path: str, output_path: str) -> None:
        """Convert checkpoint from FP16 to BF16."""
        try:
            from safetensors.torch import load_file, save_file
            import torch

            logger.info(f"🔄 Converting {input_path} to BF16...")

            # Load the checkpoint
            state_dict = load_file(input_path)

            # Convert all tensors to BF16
            bf16_state_dict = {}
            for key, tensor in state_dict.items():
                if tensor.dtype == torch.float16:
                    bf16_state_dict[key] = tensor.to(torch.bfloat16)
                else:
                    bf16_state_dict[key] = tensor

            # Save as BF16 checkpoint
            metadata = {"converted_to_bf16": "true", "source": input_path}
            save_file(bf16_state_dict, output_path, metadata=metadata)

            logger.info(f"✅ BF16 checkpoint saved: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to convert checkpoint to BF16: {e}")
            raise

    @staticmethod
    def _wait_for_bf16_conversion(bf16_path: str, timeout: int = 300) -> None:
        """Wait for BF16 conversion to complete (for non-main processes)."""
        import time

        start_time = time.time()

        while not os.path.exists(bf16_path):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"BF16 conversion timeout: {bf16_path}")
            time.sleep(5)  # Check every 5 seconds

        logger.info(f"✅ BF16 conversion completed: {bf16_path}")

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

        # Save accelerator state for checkpoint resume functionality
        if getattr(args, "save_state", True):
            state_dir = os.path.join(
                output_dir, f"{output_name}-step{global_step:06d}-state"
            )
            logger.info(f"💾 Saving checkpoint state to: {state_dir}")
            accelerator.save_state(state_dir)
            # Save step info for resume
            step_file = os.path.join(state_dir, "step.txt")
            with open(step_file, "w") as f:
                f.write(str(global_step))

        # Cleanup old models and states if save_last_n_steps is set
        self.cleanup_old_step_models(args, global_step)
        self.cleanup_old_step_states(args, global_step)

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

        # Save model as safetensors for inference
        output_dir = getattr(args, "output_dir", "output")
        output_name = getattr(args, "output_name", "wan_finetune")

        model_path = os.path.join(
            output_dir, f"{output_name}-epoch{epoch+1:04d}.safetensors"
        )
        self.save_model_safetensors(
            args, accelerator, training_model, model_path, global_step
        )

        # Save accelerator state for checkpoint resume functionality
        if getattr(args, "save_state", True):
            state_dir = os.path.join(
                output_dir, f"{output_name}-epoch{epoch+1:04d}-state"
            )
            logger.info(f"💾 Saving epoch checkpoint state to: {state_dir}")
            accelerator.save_state(state_dir)
            # Save step and epoch info for resume
            step_file = os.path.join(state_dir, "step.txt")
            with open(step_file, "w") as f:
                f.write(str(global_step))
            epoch_file = os.path.join(state_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch + 1))

        # Cleanup old models and states if save_last_n_epochs is set
        self.cleanup_old_epoch_models(args, epoch + 1)
        self.cleanup_old_epoch_states(args, epoch + 1)

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

    def cleanup_old_step_states(
        self, args: argparse.Namespace, current_step: int
    ) -> None:
        """Cleanup old step-based state directories."""
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
                output_dir, f"{output_name}-step{remove_step:06d}-state"
            )

            if os.path.exists(old_state_dir):
                import shutil

                shutil.rmtree(old_state_dir)
                logger.info(f"🗑️ Removed old step state: {old_state_dir}")

    def cleanup_old_epoch_states(
        self, args: argparse.Namespace, current_epoch: int
    ) -> None:
        """Cleanup old epoch-based state directories."""
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
