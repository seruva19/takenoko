#!/usr/bin/env python3
"""
Test script to verify Takenoko installation
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test that all key dependencies can be imported"""
    print("Testing imports...")

    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import accelerate

        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        return False

    try:
        import transformers

        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import diffusers

        print(f"✓ Diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers import failed: {e}")
        return False

    try:
        import safetensors

        print(f"✓ Safetensors {safetensors.__version__}")  # type: ignore
    except ImportError as e:
        print(f"✗ Safetensors import failed: {e}")
        return False

    try:
        import toml

        print(f"✓ TOML {toml.__version__}")  # type: ignore
    except ImportError as e:
        print(f"✗ TOML import failed: {e}")
        return False

    try:
        import cv2

        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        from PIL import Image

        print("✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False

    try:
        import numpy as np

        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import bitsandbytes

        print(f"✓ BitsAndBytes {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"✗ BitsAndBytes import failed: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")  # type: ignore
            print(
                f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def test_project_structure():
    """Test that project structure is correct"""
    print("\nTesting project structure...")

    required_files = [
        "src/takenoko.py",
        "src/core/wan_network_trainer.py",
        "configs/examples/full_config_template.toml",
        "pyproject.toml",
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - missing!")
            return False

    return True


def test_config_loading():
    """Test that configuration files can be loaded"""
    print("\nTesting default configuration loading...")

    try:
        import toml

        config_path = "configs/examples/full_config_template.toml"

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)
            print(f"✓ Configuration loaded from {config_path}")
            return True
        else:
            print(f"✗ Configuration file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Takenoko Installation Test")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("CUDA Test", test_cuda),
        ("Project Structure", test_project_structure),
        ("Configuration Loading", test_config_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ❌ {test_name} failed!")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nYou can now run the trainer using:")
        print("  run_trainer.bat")
        print("\nOr manually with:")
        print("  python src/takenoko.py configs/examples/full_config_template.toml")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry running install.bat again or check the error messages above.")

    print("=" * 50)


if __name__ == "__main__":
    main()
