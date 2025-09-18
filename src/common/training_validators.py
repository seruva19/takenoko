"""
Training Configuration Validators

Contains validation functions for training configurations to catch common issues
and provide helpful warnings/recommendations to users.
"""

import argparse
import importlib.util
import logging
from typing import Dict, Any
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def validate_full_finetune_precision(args: argparse.Namespace) -> None:
    """
    Validate that full fine-tuning is using appropriate precision settings.
    
    WAN full fine-tuning requires BF16 precision for optimal memory usage and stability.
    This function warns users if they're using suboptimal precision settings.
    """
    # Only validate for WAN full fine-tuning
    if getattr(args, 'network_module', '') != "networks.wan_finetune":
        return
    
    mixed_precision = getattr(args, 'mixed_precision', 'no')
    
    if mixed_precision.lower() not in ["bf16", "bfloat16"]:
        logger.warning("=" * 80)
        logger.warning("⚠️  WARNING: WAN Full Fine-Tuning Precision Issue")
        logger.warning("=" * 80)
        logger.warning(f"You are using full fine-tuning (network_module = 'networks.wan_finetune')")
        logger.warning(f"but mixed_precision is set to '{mixed_precision}'.")
        logger.warning("")
        logger.warning("🔥 RECOMMENDATION: Use BF16 precision for optimal results!")
        logger.warning("")
        logger.warning("Benefits of BF16 for full fine-tuning:")
        logger.warning("  ✅ Better memory efficiency (2x less VRAM usage)")
        logger.warning("  ✅ Faster training on modern GPUs")
        logger.warning("  ✅ Better numerical stability than FP16")
        logger.warning("  ✅ Designed for large model training")
        logger.warning("")
        logger.warning("To fix this:")
        logger.warning("  1. Set mixed_precision = 'bf16' in your config")
        logger.warning("  2. If your model is FP16, convert it to BF16 using:")
        logger.warning(f"     python tools/convert_model_precision.py \\")
        logger.warning(f"       --input /path/to/your/model_fp16.safetensors \\")
        logger.warning(f"       --target bf16 --auto-name")
        logger.warning("")
        logger.warning("Example conversion:")
        logger.warning(f"     python tools/convert_model_precision.py \\")
        logger.warning(f"       --input models/wan2.2_t2v_low_noise_14B_fp16.safetensors \\")
        logger.warning(f"       --target bf16 --auto-name")
        logger.warning("     # Creates: models/wan2.2_t2v_low_noise_14B_bf16.safetensors")
        logger.warning("")
        logger.warning("=" * 80)


def validate_memory_settings(args: argparse.Namespace) -> None:
    """
    Validate memory-related settings for large model training.
    """
    # Only validate for full fine-tuning which has high memory requirements
    if getattr(args, 'network_module', '') != "networks.wan_finetune":
        return
    
    batch_size = getattr(args, 'batch_size', 1)
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Check for potentially problematic memory settings
    memory_warnings = []
    
    if batch_size > 4:
        memory_warnings.append(f"Large batch_size ({batch_size}) may cause OOM for 14B model training")
    
    if not gradient_checkpointing:
        memory_warnings.append("gradient_checkpointing=false may use excessive memory for large models")
    
    if gradient_accumulation_steps == 1 and batch_size > 2:
        memory_warnings.append("Consider using gradient_accumulation_steps > 1 for better memory efficiency")
    
    if memory_warnings:
        logger.warning("💾 Memory Usage Recommendations:")
        for warning in memory_warnings:
            logger.warning(f"  ⚠️  {warning}")
        logger.warning(f"  ℹ️  Effective batch size: {effective_batch_size}")


def validate_stochastic_rounding(args: argparse.Namespace) -> None:
    """
    Validate stochastic rounding configuration.
    
    Provides recommendations for optimal stochastic rounding usage.
    """
    # Only validate for WAN full fine-tuning
    if getattr(args, 'network_module', '') != "networks.wan_finetune":
        return
    
    use_stochastic_rounding = getattr(args, 'use_stochastic_rounding', False)
    mixed_precision = getattr(args, 'mixed_precision', 'no')
    
    if use_stochastic_rounding and mixed_precision.lower() != "bf16":
        logger.warning("=" * 80)
        logger.warning("⚠️  WARNING: Stochastic Rounding Configuration Issue")
        logger.warning("=" * 80)
        logger.warning("You have enabled stochastic rounding but are not using BF16 precision.")
        logger.warning("Stochastic rounding is specifically designed for BF16 training!")
        logger.warning("")
        logger.warning("RECOMMENDATION:")
        logger.warning("  Set mixed_precision = 'bf16' to benefit from stochastic rounding")
        logger.warning("=" * 80)
    elif mixed_precision.lower() == "bf16" and not use_stochastic_rounding:
        logger.info("💡 TIP: Consider enabling stochastic rounding for BF16 full fine-tuning")
        logger.info("   Add use_stochastic_rounding = true to your config for:")
        logger.info("   ✅ Better numerical stability")
        logger.info("   ✅ Reduced gradient underflow")
        logger.info("   ✅ Improved convergence for large models")


def validate_vae_loss_settings(args: argparse.Namespace) -> None:
    """Validate VAE loss mixing parameters and optional dependencies."""

    if getattr(args, "network_module", "") != "networks.vae_wan":
        return

    weights = {
        "vae_mse_weight": float(getattr(args, "vae_mse_weight", 1.0)),
        "vae_mae_weight": float(getattr(args, "vae_mae_weight", 0.0)),
        "vae_lpips_weight": float(getattr(args, "vae_lpips_weight", 0.0)),
        "vae_edge_weight": float(getattr(args, "vae_edge_weight", 0.0)),
        "vae_kl_weight": float(getattr(args, "vae_kl_weight", 1e-6)),
    }

    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"{key} must be non-negative (got {value})")

    window = int(getattr(args, "vae_loss_balancer_window", 0))
    if window < 0:
        raise ValueError(
            f"vae_loss_balancer_window must be >= 0 (got {window})"
        )

    percentile = float(getattr(args, "vae_loss_balancer_percentile", 95))
    if not (0 < percentile <= 100):
        raise ValueError(
            "vae_loss_balancer_percentile must be within (0, 100]"
        )

    if weights["vae_lpips_weight"] > 0.0:
        if importlib.util.find_spec("lpips") is None:
            logger.warning(
                "LPIPS weighting is enabled but the 'lpips' package could not be found. "
                "Install it with 'pip install lpips' or set vae_lpips_weight = 0."
            )


def validate_training_config(args: argparse.Namespace) -> None:
    """
    Run all training configuration validations.
    
    This is the main entry point that should be called during training initialization
    to validate the configuration and provide helpful warnings.
    """
    validate_full_finetune_precision(args)
    validate_memory_settings(args)
    validate_stochastic_rounding(args)
    validate_vae_loss_settings(args)


def validate_model_compatibility(args: argparse.Namespace) -> None:
    """
    Validate model and training configuration compatibility.
    """
    target_model = getattr(args, 'target_model', 'wan21')
    network_module = getattr(args, 'network_module', 'networks.lora_wan')
    
    # Check for known compatibility issues
    if target_model == 'wan22' and network_module != "networks.wan_finetune":
        logger.info("ℹ️  Note: WAN22 works best with full fine-tuning (network_module = 'networks.wan_finetune')")
    
    if network_module == "networks.wan_finetune":
        logger.info("🔥 Full Fine-tuning Mode: Training ALL model parameters directly")
        logger.info("   This requires more VRAM but provides the best quality results")
