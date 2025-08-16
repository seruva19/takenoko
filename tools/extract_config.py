#!/usr/bin/env python3
"""
Extract original training configuration from safetensors metadata.

Usage:
    python tools/extract_config.py <model_path> [output_path]

Examples:
    python tools/extract_config.py output/wan21_lora.safetensors
    python tools/extract_config.py output/wan21_lora.safetensors my_config.toml
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.safetensors_utils import (
    save_config_from_safetensors,
    load_metadata_from_safetensors_file,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training config from safetensors metadata"
    )
    parser.add_argument("model_path", help="Path to the safetensors model file")
    parser.add_argument(
        "output_path", nargs="?", help="Output path for the config file (optional)"
    )
    parser.add_argument(
        "--show-metadata", action="store_true", help="Show all metadata keys"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not args.model_path.endswith(".safetensors"):
        print(f"Warning: File doesn't have .safetensors extension: {args.model_path}")

    # Load metadata
    metadata = load_metadata_from_safetensors_file(args.model_path)

    if not metadata:
        print(f"Error: No metadata found in {args.model_path}")
        sys.exit(1)

    # Show metadata if requested
    if args.show_metadata:
        print("Available metadata keys:")
        for key in sorted(metadata.keys()):
            value = metadata[key]
            if len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")
        print()

    # Extract and save config
    success = save_config_from_safetensors(args.model_path, args.output_path)

    if success:
        print("✅ Config extraction completed successfully!")
    else:
        print("❌ Config extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
