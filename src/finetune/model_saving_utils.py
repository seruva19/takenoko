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
from utils.safetensors_utils import MemoryEfficientSafeOpen

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

    @staticmethod
    def _prepare_state_dict_for_save(
        state_dict: dict[str, torch.Tensor],
        args: argparse.Namespace,
    ) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
        """Optionally convert to Comfy-style keys and merge with the source checkpoint."""
        save_comfy_format = bool(getattr(args, "save_comfy_format", False))
        save_merged_checkpoint = bool(getattr(args, "save_merged_checkpoint", False))
        if not save_comfy_format and not save_merged_checkpoint:
            return state_dict, None

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                new_key = key
            elif key.startswith("model."):
                new_key = "model.diffusion_model." + key[len("model.") :]
            else:
                new_key = "model.diffusion_model." + key
            renamed[new_key] = value

        extra_metadata: dict[str, str] | None = None
        if not save_merged_checkpoint:
            return renamed, extra_metadata

        checkpoint_path = getattr(args, "dit", None)
        if not checkpoint_path or str(checkpoint_path).startswith(("http://", "https://")):
            logger.warning(
                "save_merged_checkpoint requires a local dit checkpoint path; exporting Comfy-format weights without merge."
            )
            return renamed, extra_metadata

        logger.info(
            "Merging finetuned checkpoint with original Wan checkpoint: %s",
            checkpoint_path,
        )
        with MemoryEfficientSafeOpen(str(checkpoint_path)) as handle:
            all_keys = handle.keys()
            missing_keys: list[str] = []
            dtype_fixed = 0
            for key in all_keys:
                overlap_key = key if key in renamed else None
                if overlap_key is None:
                    if key.startswith("model."):
                        prefixed_key = "model.diffusion_model." + key[len("model.") :]
                    elif key.startswith("model.diffusion_model."):
                        prefixed_key = key
                    else:
                        prefixed_key = "model.diffusion_model." + key
                    if prefixed_key in renamed:
                        overlap_key = prefixed_key
                if overlap_key is None:
                    missing_keys.append(key)
                    continue
                original_dtype = handle.get_tensor(key).dtype
                if renamed[overlap_key].dtype != original_dtype:
                    renamed[overlap_key] = renamed[overlap_key].to(original_dtype)
                    dtype_fixed += 1
            if dtype_fixed:
                logger.info(
                    "Restored original dtype for %d overlapping checkpoint tensors",
                    dtype_fixed,
                )
            for key in missing_keys:
                renamed[key] = handle.get_tensor(key)
            original_metadata = handle.metadata()
            if original_metadata and "config" in original_metadata:
                extra_metadata = {"config": original_metadata["config"]}
        logger.info(
            "Merged checkpoint has %d keys (%d copied from original).",
            len(renamed),
            len(missing_keys),
        )
        return renamed, extra_metadata

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
            state_dict, extra_metadata = self._prepare_state_dict_for_save(
                state_dict,
                args,
            )

            # Create metadata
            metadata = {
                "step": str(step),
                "finetune_type": "wan_full_finetune",
                "full_bf16": str(self.full_bf16),
                "fused_backward_pass": str(self.fused_backward_pass),
                "mem_eff_save": str(self.mem_eff_save),
                "architecture": "WanFinetune",
                "ss_save_comfy_format": str(
                    bool(getattr(args, "save_comfy_format", False))
                ),
                "ss_save_merged_checkpoint": str(
                    bool(getattr(args, "save_merged_checkpoint", False))
                ),
                "ss_motion_preservation": str(
                    bool(getattr(args, "motion_preservation", False))
                ),
                "ss_motion_preservation_mode": str(
                    getattr(args, "motion_preservation_mode", "temporal")
                ),
                "ss_motion_preservation_multiplier": str(
                    getattr(args, "motion_preservation_multiplier", 0.0)
                ),
                "ss_motion_preservation_anchor_cache_size": str(
                    getattr(args, "motion_preservation_anchor_cache_size", 0)
                ),
                "ss_motion_preservation_anchor_cache_auto": str(
                    bool(getattr(args, "motion_preservation_anchor_cache_auto", False))
                ),
                "ss_motion_preservation_anchor_cache_path": str(
                    getattr(args, "motion_preservation_anchor_cache_path", None)
                ),
                "ss_motion_preservation_anchor_cache_rebuild": str(
                    bool(
                        getattr(
                            args, "motion_preservation_anchor_cache_rebuild", False
                        )
                    )
                ),
                "ss_motion_preservation_anchor_source": str(
                    getattr(args, "motion_preservation_anchor_source", "synthetic")
                ),
                "ss_motion_preservation_warmup_steps": str(
                    getattr(args, "motion_preservation_warmup_steps", 0)
                ),
                "ss_motion_preservation_interval": str(
                    getattr(args, "motion_preservation_interval", 1)
                ),
                "ss_motion_preservation_probability": str(
                    getattr(args, "motion_preservation_probability", None)
                ),
                "ss_motion_preservation_num_sigmas": str(
                    getattr(args, "motion_preservation_num_sigmas", 1)
                ),
                "ss_motion_preservation_sigma_values": str(
                    getattr(args, "motion_preservation_sigma_values", None)
                ),
                "ss_motion_preservation_sigma_min": str(
                    getattr(args, "motion_preservation_sigma_min", 0.2)
                ),
                "ss_motion_preservation_sigma_max": str(
                    getattr(args, "motion_preservation_sigma_max", 0.8)
                ),
                "ss_motion_preservation_sigma_sampling": str(
                    getattr(args, "motion_preservation_sigma_sampling", "uniform")
                ),
                "ss_motion_preservation_sigma_sampling_power": str(
                    getattr(args, "motion_preservation_sigma_sampling_power", 1.0)
                ),
                "ss_motion_preservation_second_order_weight": str(
                    getattr(args, "motion_preservation_second_order_weight", 0.0)
                ),
                "ss_motion_preservation_teacher_chunk_frames": str(
                    getattr(args, "motion_preservation_teacher_chunk_frames", 0)
                ),
                "ss_motion_preservation_separate_backward": str(
                    bool(
                        getattr(args, "motion_preservation_separate_backward", False)
                    )
                ),
                "ss_motion_preservation_fused_defer_step": str(
                    bool(
                        getattr(args, "motion_preservation_fused_defer_step", False)
                    )
                ),
                "ss_motion_prior_cache_only": str(
                    bool(getattr(args, "motion_prior_cache_only", False))
                ),
                "ss_motion_prior_require_temporal": str(
                    bool(getattr(args, "motion_prior_require_temporal", False))
                ),
                "ss_motion_attention_preservation": str(
                    bool(getattr(args, "motion_attention_preservation", False))
                ),
                "ss_motion_attention_preservation_weight": str(
                    getattr(args, "motion_attention_preservation_weight", 0.0)
                ),
                "ss_motion_attention_preservation_loss": str(
                    getattr(args, "motion_attention_preservation_loss", "kl")
                ),
                "ss_motion_attention_preservation_queries": str(
                    getattr(args, "motion_attention_preservation_queries", 32)
                ),
                "ss_motion_attention_preservation_keys": str(
                    getattr(args, "motion_attention_preservation_keys", 64)
                ),
                "ss_motion_attention_preservation_per_head": str(
                    bool(
                        getattr(args, "motion_attention_preservation_per_head", False)
                    )
                ),
                "ss_motion_attention_preservation_temperature": str(
                    getattr(args, "motion_attention_preservation_temperature", 1.0)
                ),
                "ss_motion_attention_preservation_symmetric_kl": str(
                    bool(
                        getattr(
                            args,
                            "motion_attention_preservation_symmetric_kl",
                            False,
                        )
                    )
                ),
                "ss_motion_attention_preservation_blocks": str(
                    getattr(args, "motion_attention_preservation_blocks", None)
                ),
                "ss_motion_attention_preservation_active": str(
                    bool(
                        getattr(args, "_motion_attention_preservation_active", False)
                    )
                ),
                "ss_motion_attention_preservation_module_count": str(
                    int(
                        getattr(
                            args,
                            "_motion_attention_preservation_module_count",
                            0,
                        )
                    )
                ),
                "ss_motion_preservation_runtime_anchor_cache_size": str(
                    getattr(args, "_motion_preservation_runtime_anchor_cache_size", 0)
                ),
                "ss_motion_preservation_runtime_temporal_anchor_count": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_temporal_anchor_count",
                        0,
                    )
                ),
                "ss_motion_preservation_runtime_synthetic_anchor_count": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_synthetic_anchor_count",
                        0,
                    )
                ),
                "ss_motion_preservation_runtime_dataset_anchor_count": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_dataset_anchor_count",
                        0,
                    )
                ),
                "ss_motion_preservation_runtime_cache_loaded": str(
                    bool(
                        getattr(
                            args,
                            "_motion_preservation_runtime_cache_loaded",
                            False,
                        )
                    )
                ),
                "ss_motion_preservation_runtime_cache_built": str(
                    bool(
                        getattr(
                            args,
                            "_motion_preservation_runtime_cache_built",
                            False,
                        )
                    )
                ),
                "ss_motion_preservation_runtime_invocations": str(
                    getattr(args, "_motion_preservation_runtime_invocations", 0)
                ),
                "ss_motion_preservation_runtime_applied": str(
                    getattr(args, "_motion_preservation_runtime_applied", 0)
                ),
                "ss_motion_preservation_runtime_apply_rate": str(
                    getattr(args, "_motion_preservation_runtime_apply_rate", 0.0)
                ),
                "ss_motion_preservation_runtime_schedule_skip_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_schedule_skip_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_zero_weight_skip_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_zero_weight_skip_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_no_anchor_skip_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_no_anchor_skip_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_invalid_anchor_skip_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_invalid_anchor_skip_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_temporal_fallback_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_temporal_fallback_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_attention_apply_rate": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_attention_apply_rate",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_temporal_anchor_ratio": str(
                    getattr(
                        args,
                        "_motion_preservation_runtime_temporal_anchor_ratio",
                        0.0,
                    )
                ),
                "ss_motion_preservation_runtime_last_error": str(
                    getattr(args, "_motion_preservation_runtime_last_error", "")
                ),
                "ss_ewc_lambda": str(getattr(args, "ewc_lambda", 0.0)),
                "ss_ewc_num_batches": str(getattr(args, "ewc_num_batches", 0)),
                "ss_ewc_target": str(
                    getattr(args, "ewc_target", "attn_norm_bias")
                ),
                "ss_ewc_max_param_tensors": str(
                    getattr(args, "ewc_max_param_tensors", 0)
                ),
                "ss_ewc_cache_path": str(getattr(args, "ewc_cache_path", None)),
                "ss_ewc_cache_rebuild": str(
                    bool(getattr(args, "ewc_cache_rebuild", False))
                ),
                "ss_freeze_early_blocks": str(
                    getattr(args, "freeze_early_blocks", 0)
                ),
                "ss_freeze_block_indices": str(
                    getattr(args, "freeze_block_indices", None)
                ),
                "ss_block_lr_scales": str(getattr(args, "block_lr_scales", None)),
                "ss_non_block_lr_scale": str(
                    getattr(args, "non_block_lr_scale", 1.0)
                ),
                "ss_attn_geometry_lr_scale": str(
                    getattr(args, "attn_geometry_lr_scale", 1.0)
                ),
                "ss_freeze_attn_geometry": str(
                    bool(getattr(args, "freeze_attn_geometry", False))
                ),
                "ss_full_ft_lr_group_scales": str(
                    getattr(args, "_full_ft_lr_group_scales", None)
                ),
                "ss_full_ft_frozen_blocks_applied": str(
                    getattr(args, "_full_ft_frozen_blocks_applied", None)
                ),
                "ss_full_ft_trainable_by_block": str(
                    getattr(args, "_full_ft_trainable_by_block", None)
                ),
                "ss_full_ft_trainable_attn_geometry_tensors": str(
                    getattr(args, "_full_ft_trainable_attn_geometry_tensors", 0)
                ),
                "ss_full_ft_frozen_attn_geometry_tensors": str(
                    getattr(args, "_full_ft_frozen_attn_geometry_tensors", 0)
                ),
            }
            if extra_metadata:
                metadata.update(extra_metadata)

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
