#!/usr/bin/env python3
"""
Model Precision Converter for Takenoko

Converts model checkpoints between different precision formats (FP32, FP16, BF16).
Essential for WAN full fine-tuning which requires BF16 precision.

Usage:
    python tools/convert_model_precision.py --input model.safetensors --output model_bf16.safetensors --target bf16
    python tools/convert_model_precision.py --input model.safetensors --target bf16 --auto-name
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def get_tensor_dtype_info(tensors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Get statistics about tensor dtypes in the model."""
    dtype_counts = {}
    for tensor in tensors.values():
        dtype_str = str(tensor.dtype).replace("torch.", "")
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
    return dtype_counts


def convert_tensor_dtype(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Convert tensor to target dtype safely."""
    if tensor.dtype == target_dtype:
        return tensor

    # Special handling for specific dtype conversions
    if target_dtype == torch.bfloat16:
        # BF16 conversion - safe for most tensors
        return tensor.to(torch.bfloat16)
    elif target_dtype == torch.float16:
        # FP16 conversion - check for potential overflow
        if tensor.dtype == torch.float32:
            # Check for values that might overflow in FP16
            max_val = tensor.abs().max().item()
            if max_val > 65504:  # FP16 max value
                print(
                    f"⚠️  Warning: Large values detected ({max_val:.2e}), may overflow in FP16"
                )
        return tensor.to(torch.float16)
    elif target_dtype == torch.float32:
        # FP32 conversion - always safe (upcast)
        return tensor.to(torch.float32)
    else:
        return tensor.to(target_dtype)


def get_auto_output_name(input_path: str, target_precision: str) -> str:
    """Generate automatic output filename based on target precision."""
    input_path = Path(input_path)
    stem = input_path.stem

    # Remove existing precision suffixes
    for precision in ["_fp32", "_fp16", "_bf16", "_float32", "_float16", "_bfloat16"]:
        if stem.endswith(precision):
            stem = stem[: -len(precision)]
            break

    # Add new precision suffix
    precision_suffix = f"_{target_precision}"
    new_stem = f"{stem}{precision_suffix}"

    return str(input_path.parent / f"{new_stem}{input_path.suffix}")


def convert_model_precision(
    input_path: str,
    output_path: str,
    target_precision: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Convert model precision.

    Args:
        input_path: Path to input model
        output_path: Path to output model
        target_precision: Target precision (fp32, fp16, bf16)
        dry_run: Only analyze, don't convert
        verbose: Verbose output

    Returns:
        Success status
    """
    # Map precision strings to torch dtypes
    precision_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    if target_precision.lower() not in precision_map:
        print(f"❌ Error: Unsupported precision '{target_precision}'")
        print(f"Supported: {list(precision_map.keys())}")
        return False

    target_dtype = precision_map[target_precision.lower()]

    try:
        if verbose:
            print(f"🔄 Loading model from: {input_path}")

        # Load the model
        tensors = load_file(input_path)

        if verbose:
            print(f"📊 Model loaded: {len(tensors)} tensors")

            # Show current dtype distribution
            dtype_info = get_tensor_dtype_info(tensors)
            print("Current precision distribution:")
            for dtype, count in dtype_info.items():
                print(f"  - {dtype}: {count} tensors")

        if dry_run:
            print(f"🔍 Dry run: Would convert to {target_precision}")
            return True

        # Convert tensors
        converted_tensors = {}

        if verbose:
            print(f"🔄 Converting to {target_precision}...")
            tensor_items = tqdm(tensors.items(), desc="Converting tensors")
        else:
            tensor_items = tensors.items()

        conversion_stats = {"converted": 0, "unchanged": 0}

        for key, tensor in tensor_items:
            if tensor.dtype != target_dtype:
                converted_tensors[key] = convert_tensor_dtype(tensor, target_dtype)
                conversion_stats["converted"] += 1
            else:
                converted_tensors[key] = tensor
                conversion_stats["unchanged"] += 1

        if verbose:
            print(f"📈 Conversion complete:")
            print(f"  - Converted: {conversion_stats['converted']} tensors")
            print(f"  - Unchanged: {conversion_stats['unchanged']} tensors")

        # Calculate file sizes
        input_size = os.path.getsize(input_path)

        if verbose:
            print(f"💾 Saving to: {output_path}")

        # Save converted model
        save_file(converted_tensors, output_path)

        output_size = os.path.getsize(output_path)

        if verbose:
            print(f"✅ Conversion successful!")
            print(f"📁 Input size:  {input_size / 1024**3:.2f} GB")
            print(f"📁 Output size: {output_size / 1024**3:.2f} GB")

            size_ratio = output_size / input_size
            if size_ratio < 0.9:
                print(f"💡 Size reduction: {(1-size_ratio)*100:.1f}%")
            elif size_ratio > 1.1:
                print(f"💡 Size increase: {(size_ratio-1)*100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert model checkpoints between precision formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert FP16 model to BF16 (recommended for WAN full fine-tuning)
  python tools/convert_model_precision.py --input model_fp16.safetensors --target bf16 --auto-name
  
  # Convert to FP32 with custom output name
  python tools/convert_model_precision.py --input model.safetensors --output model_fp32.safetensors --target fp32
  
  # Analyze model without converting (dry run)
  python tools/convert_model_precision.py --input model.safetensors --target bf16 --dry-run
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input model path (.safetensors)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output model path (.safetensors). If not specified, use --auto-name",
    )

    parser.add_argument(
        "--target",
        "-t",
        required=True,
        choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Target precision format",
    )

    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="Automatically generate output filename with precision suffix",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze model without converting"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.endswith(".safetensors"):
        print(f"❌ Error: Input file must be a .safetensors file")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.auto_name:
        output_path = get_auto_output_name(args.input, args.target)
    else:
        print("❌ Error: Must specify either --output or --auto-name")
        sys.exit(1)

    # Check output file existence
    if not args.dry_run and os.path.exists(output_path) and not args.overwrite:
        print(f"❌ Error: Output file exists: {output_path}")
        print("Use --overwrite to overwrite existing files")
        sys.exit(1)

    if not args.quiet:
        print("🔧 Takenoko Model Precision Converter")
        print("=" * 50)
        print(f"Input:  {args.input}")
        if not args.dry_run:
            print(f"Output: {output_path}")
        print(f"Target: {args.target.upper()}")
        print("=" * 50)

    # Perform conversion
    success = convert_model_precision(
        input_path=args.input,
        output_path=output_path,
        target_precision=args.target,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if not success:
        sys.exit(1)

    if not args.quiet and not args.dry_run:
        print(f"\n🎉 Model successfully converted to {args.target.upper()}!")

        # Special message for BF16 conversions
        if args.target.lower() in ["bf16", "bfloat16"]:
            print("\n💡 BF16 models are recommended for:")
            print("   - WAN full fine-tuning")
            print("   - Large model training with memory constraints")
            print("   - Training on modern GPUs (RTX 30/40 series, A100, H100)")


if __name__ == "__main__":
    main()
