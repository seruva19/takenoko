"""Logging utilities for WAN fine-tuning trainer.

This module contains logging-related functionality including:
- Detailed network information logging
- Parameter analysis and categorization
- Training progress logging helpers
- Memory usage reporting
"""

import logging
from typing import Any, Dict, Optional
import argparse
import torch

try:
    from common.logger import get_logger
except ImportError:
    # Fallback for testing or different import contexts
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger


logger = get_logger(__name__, level=logging.INFO)


class NetworkLoggingUtils:
    """Utility class for network and parameter logging."""

    @staticmethod
    def log_detailed_network_info(transformer: Any, args: argparse.Namespace) -> None:
        """Log detailed information about the transformer and trainable parameters."""
        logger.info("Detailed Transformer Information:")
        logger.info("=" * 80)

        # Count different types of parameters
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        # Parameter type categories
        param_categories = {
            "attention": 0,
            "mlp": 0,
            "norm": 0,
            "embedding": 0,
            "modulation": 0,
            "other": 0,
        }

        # Log transformer blocks information
        if hasattr(transformer, "blocks"):
            logger.info(f"Transformer Blocks ({len(transformer.blocks)} blocks):")
            for i, block in enumerate(transformer.blocks):
                block_type = type(block).__name__
                trainable_count = sum(
                    p.numel() for p in block.parameters() if p.requires_grad
                )
                total_count = sum(p.numel() for p in block.parameters())
                logger.info(
                    f"  {i+1:3d}: {block_type:<30} trainable: {trainable_count:>10,} / {total_count:>10,}"
                )

        # Log detailed parameter information
        logger.info("\nTrainable Parameters:")
        for name, param in transformer.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

                # Categorize parameters by name patterns
                param_type = NetworkLoggingUtils._categorize_parameter(
                    name, param_categories, param_count
                )

                # Log top-level parameters (avoid too much noise)
                if name.count(".") <= 2:  # Only log top-level parameter groups
                    logger.info(
                        f"  {name:<60} {str(param.shape):<20} {param_count:>10,} [{param_type}]"
                    )
            else:
                frozen_params += param_count

        # Log parameter category summary
        logger.info("\nParameter Category Summary:")
        for category, count in param_categories.items():
            if count > 0:
                percentage = (
                    (count / trainable_params * 100) if trainable_params > 0 else 0
                )
                logger.info(
                    f"  {category.capitalize():<12}: {count:>12,} ({percentage:5.1f}%)"
                )

        # Log overall statistics
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total parameters:     {total_params:>15,}")
        logger.info(
            f"  Trainable parameters: {trainable_params:>15,} ({trainable_params/total_params*100:5.1f}%)"
        )
        logger.info(
            f"  Frozen parameters:   {frozen_params:>15,} ({frozen_params/total_params*100:5.1f}%)"
        )

        # Log memory usage estimate
        param_bytes = total_params * 4  # Assume float32 (4 bytes per parameter)
        logger.info(f"  Estimated memory:   {param_bytes/1024**3:>8.2f} GB (float32)")

    @staticmethod
    def _categorize_parameter(
        name: str, param_categories: Dict[str, int], param_count: int
    ) -> str:
        """Categorize parameter by name patterns and update counters."""
        name_lower = name.lower()

        if any(
            x in name_lower
            for x in [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ]
        ):
            param_categories["attention"] += param_count
            return "Attention"
        elif any(x in name_lower for x in ["mlp", "ffn", "linear", "gelu"]):
            param_categories["mlp"] += param_count
            return "MLP"
        elif any(x in name_lower for x in ["norm", "layer_norm", "group_norm"]):
            param_categories["norm"] += param_count
            return "Norm"
        elif any(x in name_lower for x in ["embed", "positional", "patch"]):
            param_categories["embedding"] += param_count
            return "Embedding"
        elif any(x in name_lower for x in ["modulation", "ada", "scale", "shift"]):
            param_categories["modulation"] += param_count
            return "Modulation"
        else:
            param_categories["other"] += param_count
            return "Other"

    @staticmethod
    def log_training_configuration(args: argparse.Namespace, trainer_instance) -> None:
        """Log comprehensive training configuration."""
        logger.info("WanFinetune Training Configuration:")
        logger.info("=" * 50)
        logger.info(f"  Full BF16: {trainer_instance.full_bf16}")
        logger.info(f"  Fused backward pass: {trainer_instance.fused_backward_pass}")
        logger.info(f"  Memory efficient save: {trainer_instance.mem_eff_save}")
        logger.info(f"  DiT dtype: {args.dit_dtype}")
        logger.info(f"  Mixed precision: {args.mixed_precision}")
        if hasattr(trainer_instance, "use_stochastic_rounding"):
            logger.info(
                f"  Stochastic rounding: {trainer_instance.use_stochastic_rounding}"
            )

    @staticmethod
    def log_checkpoint_debug_info(restored_step, starting_epoch, global_step, args):
        """Log comprehensive checkpoint resume debugging information (only in debug mode)."""
        # Only show detailed debug info if verbose logging is enabled
        if getattr(args, "verbose_logging", False) or getattr(args, "debug", False):
            logger.info(f"🔄 CHECKPOINT DEBUG:")
            logger.info(f"   - restored_step: {restored_step}")
            logger.info(f"   - starting_epoch: {starting_epoch}")
            logger.info(f"   - global_step: {global_step}")
            logger.info(
                f"   - max_train_steps: {getattr(args, 'max_train_steps', 'Not set')}"
            )
            logger.info(
                f"   - max_train_epochs: {getattr(args, 'max_train_epochs', 'Not set')}"
            )
            if hasattr(args, "max_train_steps") and args.max_train_steps:
                logger.info(
                    f"   - Will global_step >= max_train_steps? {global_step >= args.max_train_steps}"
                )

    @staticmethod
    def log_epoch_range_info(starting_epoch, max_epochs):
        """Log epoch range information concisely."""
        epochs_to_train = max(0, max_epochs - starting_epoch)
        if starting_epoch > 0:
            logger.info(
                f"🔄 Resuming training: epoch {starting_epoch+1}/{max_epochs} ({epochs_to_train} epochs remaining)"
            )
        else:
            logger.info(f"🔄 Starting training: {epochs_to_train} epochs planned")

    @staticmethod
    def log_training_loop_entry(starting_epoch, max_epochs):
        """Log information before entering training loops (only in debug mode)."""
        # Only show this debug info if really needed - it's quite verbose
        pass  # Remove this verbose logging - epoch info is already logged above

    @staticmethod
    def log_parameter_validation_info(
        dit_params_count, param_names, args, text_encoder=None
    ):
        """Log parameter validation and detection warnings."""
        # Expected parameter count for 14B model
        expected_14b_params = 14_000_000_000
        if dit_params_count > expected_14b_params * 0.8:  # Within 80%
            logger.info(f"✅ Training {dit_params_count:,} parameters (~14B model)")
        else:
            logger.warning(
                f"⚠️  Only {dit_params_count:,} parameters - this might not be full fine-tuning"
            )

        # Check for LoRA-like parameters
        lora_detected = False
        for name in param_names[0][:50]:  # Check first 50 params
            if any(
                lora_keyword in name.lower()
                for lora_keyword in ["lora_", "adapter", "rank"]
            ):
                lora_detected = True
                logger.warning(f"⚠️  LoRA-like parameter detected: {name}")
                break

        # Log T5 encoder info if present
        if getattr(args, "finetune_text_encoder", False) and text_encoder is not None:
            logger.info("T5 text encoder fine-tuning enabled")


class TrainingProgressLogger:
    """Helper class for training progress logging."""

    @staticmethod
    def log_step_info(global_step, loss_item, learning_rate, logger_obj=logger):
        """Log step information with loss and learning rate."""
        logger_obj.info(
            f"Step {global_step}: Loss = {loss_item:.6f}, LR = {learning_rate}"
        )

    @staticmethod
    def log_sampling_debug(
        global_step, should_sample, sampling_manager_exists, vae_exists
    ):
        """Log sampling condition debugging information."""
        logger.debug(f"🎨 Sampling conditions at step {global_step}:")
        logger.debug(f"   - should_sample: {should_sample}")
        logger.debug(f"   - sampling_manager exists: {sampling_manager_exists}")
        logger.debug(f"   - vae exists: {vae_exists}")

    @staticmethod
    def log_comprehensive_training_logs(logs: Dict[str, Any], global_step: int):
        """Log comprehensive training metrics."""
        if "train_loss" in logs:
            logger.info(
                f"Step {global_step}: Loss = {logs['train_loss']:.6f}, "
                f"LR = {logs.get('learning_rate', 'N/A')}"
            )

        # Log additional metrics if present
        if "gradient_norm" in logs:
            logger.debug(f"   - Gradient norm: {logs['gradient_norm']:.6f}")
        if "epoch" in logs:
            logger.debug(f"   - Epoch: {logs['epoch']}")


class TimestepDistributionLogger:
    """Utility class for timestep distribution logging (reuses existing methods)."""

    @staticmethod
    def initialize_and_log_timestep_distribution(
        args: argparse.Namespace,
        accelerator: Any,
        timestep_distribution: Any,
    ) -> None:
        """Initialize timestep distribution and log initial charts.

        This function reuses existing timestep distribution methods without duplication.

        Args:
            args: Training arguments
            accelerator: Accelerator instance
            timestep_distribution: TimestepDistribution instance
        """
        from scheduling.timestep_utils import initialize_timestep_distribution
        from scheduling.timestep_logging import (
            log_initial_timestep_distribution,
            log_show_timesteps_figure_unconditional,
        )

        # Initialize timestep distribution using existing function
        initialize_timestep_distribution(args, timestep_distribution)

        # Log initial timestep distribution if enabled (reuse existing method)
        if (
            accelerator.is_main_process
            and getattr(args, "log_timestep_distribution_init", True)
            and getattr(args, "log_with", "").lower() in ["tensorboard", "all"]
        ):
            try:
                log_initial_timestep_distribution(
                    accelerator, args, timestep_distribution
                )
                logger.debug("✅ Initial timestep distribution logged to TensorBoard")
            except Exception as e:
                logger.debug(f"Timestep distribution logging failed: {e}")

        # Log timestep figure if enabled (reuse existing method)
        try:
            if accelerator.is_main_process:
                # Create noise scheduler for logging (same as in TrainingCore)
                from modules.scheduling_flow_match_discrete import (
                    FlowMatchDiscreteScheduler,
                )

                noise_scheduler = FlowMatchDiscreteScheduler(
                    shift=getattr(args, "discrete_flow_shift", 3.0),
                    reverse=True,
                    solver="euler",
                )
                log_show_timesteps_figure_unconditional(
                    accelerator,
                    args,
                    timestep_distribution,
                    noise_scheduler,
                )
                logger.debug("✅ Timestep distribution figure logged")
        except Exception as e:
            logger.debug(f"Timestep figure logging failed: {e}")
