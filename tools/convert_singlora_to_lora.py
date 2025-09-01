#!/usr/bin/env python3

"""
SingLoRA to Traditional LoRA Conversion Script

Converts SingLoRA weights to traditional LoRA format for compatibility
with existing inference frameworks like ComfyUI.

WARNING: This conversion loses SingLoRA-specific capabilities:
- Ramp-up training functionality
- Single matrix mathematical properties
- Enhanced non-square matrix optimization
- Ability to continue SingLoRA training

Use only for inference deployment to existing frameworks.
"""

import argparse
import os
import sys
import torch
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert_singlora_to_traditional_lora(
    singlora_weights: Dict[str, torch.Tensor], preserve_precision: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert SingLoRA weights to traditional LoRA format.

    Args:
        singlora_weights: Dictionary containing SingLoRA weights
        preserve_precision: Whether to preserve numerical precision

    Returns:
        Dictionary with traditional LoRA weight structure

    Note:
        This conversion approximates SingLoRA's A @ A.T with traditional
        LoRA's lora_up @ lora_down structure. Some precision may be lost.
    """
    traditional_weights = {}
    conversion_stats = {"modules_converted": 0, "total_parameters": 0, "warnings": []}

    # Group weights by module name
    modules = {}
    for key, value in singlora_weights.items():
        if "." not in key:
            continue

        parts = key.split(".")
        module_name = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
        param_name = parts[-1]

        if module_name not in modules:
            modules[module_name] = {}
        modules[module_name][param_name] = value

    logger.info(f"Found {len(modules)} SingLoRA modules to convert")

    for module_name, module_weights in modules.items():
        if "A" not in module_weights:
            logger.warning(f"Module {module_name} missing A matrix, skipping")
            conversion_stats["warnings"].append(f"Missing A matrix: {module_name}")
            continue

        A_matrix = module_weights["A"]  # Shape: [larger_dim, rank]
        larger_dim, rank = A_matrix.shape

        # Get alpha value
        alpha = 1.0
        if "config" in module_weights and hasattr(module_weights["config"], "item"):
            alpha = module_weights["config"].item()
        elif "alpha" in module_weights:
            alpha = (
                module_weights["alpha"].item()
                if hasattr(module_weights["alpha"], "item")
                else module_weights["alpha"]
            )

        logger.debug(f"Converting {module_name}: A={A_matrix.shape}, alpha={alpha}")

        # Convert A @ A.T to lora_up @ lora_down format
        # Strategy: Use SVD to decompose A @ A.T ≈ U @ S @ V.T
        # Then set lora_down = sqrt(S) @ V.T and lora_up = U @ sqrt(S)

        try:
            if preserve_precision:
                # High precision conversion using SVD
                AA_T = A_matrix @ A_matrix.T  # Reconstruct the full update matrix

                # SVD decomposition
                U, S, V = torch.svd(AA_T)

                # Take only the first 'rank' components
                S_trunc = S[:rank]
                U_trunc = U[:, :rank]  # [larger_dim, rank]
                V_trunc = V[:, :rank]  # [larger_dim, rank]

                # Create sqrt decomposition
                sqrt_S = torch.sqrt(
                    torch.clamp(S_trunc, min=1e-8)
                )  # Avoid numerical issues

                lora_down = (V_trunc * sqrt_S).T  # [rank, larger_dim]
                lora_up = U_trunc * sqrt_S  # [larger_dim, rank]

            else:
                # Simple conversion: treat A as lora_up, A.T as lora_down
                # This is less accurate but faster and simpler
                lora_up = A_matrix  # [larger_dim, rank]
                lora_down = A_matrix.T  # [rank, larger_dim]

            # Ensure correct shapes for traditional LoRA
            # Traditional LoRA expects:
            # - lora_down: [rank, in_features]
            # - lora_up: [out_features, rank]

            # We need to determine in_features and out_features from context
            # Since we don't have that info, we'll use the larger_dim for both
            # This works for square matrices and is an approximation for non-square

            traditional_weights[f"{module_name}.lora_down.weight"] = lora_down
            traditional_weights[f"{module_name}.lora_up.weight"] = lora_up
            traditional_weights[f"{module_name}.alpha"] = torch.tensor(
                alpha, dtype=torch.float32
            )

            conversion_stats["modules_converted"] += 1
            conversion_stats["total_parameters"] += lora_down.numel() + lora_up.numel()

            logger.debug(
                f"✓ Converted {module_name}: down={lora_down.shape}, up={lora_up.shape}"
            )

        except Exception as e:
            logger.error(f"Failed to convert {module_name}: {e}")
            conversion_stats["warnings"].append(
                f"Conversion failed: {module_name} - {str(e)}"
            )
            continue

    # Copy any non-SingLoRA weights (metadata, etc.)
    for key, value in singlora_weights.items():
        if not any(key.startswith(f"{mod}.") for mod in modules.keys()):
            traditional_weights[key] = value

    logger.info(
        f"Conversion complete: {conversion_stats['modules_converted']} modules, "
        f"{conversion_stats['total_parameters']:,} parameters"
    )

    if conversion_stats["warnings"]:
        logger.warning(
            f"{len(conversion_stats['warnings'])} warnings during conversion"
        )
        for warning in conversion_stats["warnings"]:
            logger.warning(f"  - {warning}")

    return traditional_weights


def verify_conversion(
    original_weights: Dict[str, torch.Tensor],
    converted_weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Verify the conversion by comparing mathematical results.

    Returns:
        Dictionary with verification metrics
    """
    verification_results = {
        "modules_checked": 0,
        "max_error": 0.0,
        "avg_error": 0.0,
        "passed_verification": True,
    }

    errors = []

    # Group original weights by module
    original_modules = {}
    for key, value in original_weights.items():
        if "." in key:
            module_name = ".".join(key.split(".")[:-1])
            if module_name not in original_modules:
                original_modules[module_name] = {}
            original_modules[module_name][key.split(".")[-1]] = value

    for module_name, module_weights in original_modules.items():
        if "A" not in module_weights:
            continue

        # Get original SingLoRA result
        A = module_weights["A"]
        singlora_result = A @ A.T

        # Get converted LoRA result
        lora_down_key = f"{module_name}.lora_down.weight"
        lora_up_key = f"{module_name}.lora_up.weight"

        if lora_down_key in converted_weights and lora_up_key in converted_weights:
            lora_down = converted_weights[lora_down_key]
            lora_up = converted_weights[lora_up_key]
            lora_result = lora_up @ lora_down

            # Calculate error
            error = torch.norm(singlora_result - lora_result).item()
            relative_error = error / torch.norm(singlora_result).item()
            errors.append(relative_error)

            verification_results["modules_checked"] += 1
            logger.debug(f"Module {module_name}: relative error = {relative_error:.6f}")

    if errors:
        verification_results["max_error"] = max(errors)
        verification_results["avg_error"] = sum(errors) / len(errors)
        verification_results["passed_verification"] = (
            verification_results["max_error"] < 1e-3
        )

    return verification_results


def main():
    parser = argparse.ArgumentParser(
        description="Convert SingLoRA weights to traditional LoRA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_singlora_to_lora.py model.safetensors model_traditional.safetensors
  python convert_singlora_to_lora.py --verify --precision high model.safetensors output.safetensors
  
Warning:
  This conversion loses SingLoRA-specific training capabilities.
  Use only for inference compatibility with existing frameworks.
        """,
    )

    parser.add_argument(
        "input_path", help="Path to SingLoRA weights file (.safetensors)"
    )

    parser.add_argument(
        "output_path", help="Path for traditional LoRA weights file (.safetensors)"
    )

    parser.add_argument(
        "--precision",
        choices=["high", "fast"],
        default="high",
        help="Conversion precision: 'high' uses SVD (slower, more accurate), 'fast' uses direct mapping",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion accuracy by comparing mathematical results",
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite output file if it exists"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    if not args.input_path.endswith(".safetensors"):
        logger.error("Input file must be in .safetensors format")
        sys.exit(1)

    if os.path.exists(args.output_path) and not args.force:
        logger.error(
            f"Output file exists: {args.output_path} (use --force to overwrite)"
        )
        sys.exit(1)

    try:
        # Import safetensors
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            logger.error(
                "safetensors package not found. Install with: pip install safetensors"
            )
            sys.exit(1)

        # Load SingLoRA weights
        logger.info(f"Loading SingLoRA weights from: {args.input_path}")
        singlora_weights = load_file(args.input_path)
        logger.info(f"Loaded {len(singlora_weights)} weight tensors")

        # Check if this actually contains SingLoRA weights
        has_singlora = any(key.endswith(".A") for key in singlora_weights.keys())
        if not has_singlora:
            logger.warning(
                "Input file does not appear to contain SingLoRA weights (no .A matrices found)"
            )

        # Perform conversion
        logger.info("Converting to traditional LoRA format...")
        preserve_precision = args.precision == "high"
        traditional_weights = convert_singlora_to_traditional_lora(
            singlora_weights, preserve_precision=preserve_precision
        )

        # Verify conversion if requested
        if args.verify:
            logger.info("Verifying conversion accuracy...")
            verification = verify_conversion(singlora_weights, traditional_weights)
            logger.info(
                f"Verification: {verification['modules_checked']} modules checked"
            )
            logger.info(f"Max error: {verification['max_error']:.6f}")
            logger.info(f"Avg error: {verification['avg_error']:.6f}")

            if not verification["passed_verification"]:
                logger.warning(
                    "Conversion verification failed - high numerical error detected"
                )
                if not args.force:
                    logger.error("Use --force to save despite verification failure")
                    sys.exit(1)

        # Save traditional LoRA weights
        logger.info(f"Saving traditional LoRA weights to: {args.output_path}")

        # Preserve original metadata with conversion info
        conversion_metadata = {
            "converted_from": "singlora",
            "conversion_precision": args.precision,
            "conversion_tool": "takenoko_singlora_converter",
        }

        save_file(traditional_weights, args.output_path, metadata=conversion_metadata)

        logger.info("✅ Conversion completed successfully!")
        logger.info(
            f"Traditional LoRA weights saved with {len(traditional_weights)} tensors"
        )

        # Final warning
        logger.warning(
            "⚠️  IMPORTANT: Converted weights lose SingLoRA-specific capabilities"
        )
        logger.warning("   - Cannot continue SingLoRA training")
        logger.warning("   - Lost ramp-up functionality")
        logger.warning("   - Lost single-matrix mathematical properties")
        logger.warning("   Use only for inference with existing frameworks!")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
