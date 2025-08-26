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

        print(f"🟢 PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"🚫 PyTorch import failed: {e}")
        return False

    try:
        import accelerate

        print(f"🟢 Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"🚫 Accelerate import failed: {e}")
        return False

    try:
        import transformers

        print(f"🟢 Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"🚫 Transformers import failed: {e}")
        return False

    try:
        import diffusers

        print(f"🟢 Diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"🚫 Diffusers import failed: {e}")
        return False

    try:
        import safetensors

        print(f"🟢 Safetensors {safetensors.__version__}")  # type: ignore
    except ImportError as e:
        print(f"🚫 Safetensors import failed: {e}")
        return False

    try:
        import toml

        print(f"🟢 TOML {toml.__version__}")  # type: ignore
    except ImportError as e:
        print(f"🚫 TOML import failed: {e}")
        return False

    try:
        import cv2

        print(f"🟢 OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        from PIL import Image

        print("🟢 Pillow")
    except ImportError as e:
        print(f"🚫 Pillow import failed: {e}")
        return False

    try:
        import numpy as np

        print(f"🟢 NumPy {np.__version__}")
    except ImportError as e:
        print(f"🚫 NumPy import failed: {e}")
        return False

    try:
        import bitsandbytes

        print(f"🟢 BitsAndBytes {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"🚫 BitsAndBytes import failed: {e}")
        return False

    return True


def test_flash_attention():
    """Test Flash Attention availability"""
    print("\nTesting Flash Attention...")

    try:
        import torch

        # Check if flash_attn package is installed
        try:
            import flash_attn

            print(f"🟢 Flash Attention package {flash_attn.__version__}")

            # Test basic flash attention functionality
            try:
                from flash_attn import flash_attn_func

                # insert emoji check below
                print("🟢 Flash Attention functions available")

                # Test if flash attention actually works
                try:
                    # Create test tensors for flash attention (using fp16 as required)
                    q = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=torch.float16,
                    )
                    k = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=torch.float16,
                    )
                    v = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=torch.float16,
                    )

                    # Test flash attention (this will use CPU fallback if CUDA not available)
                    output = flash_attn_func(q, k, v)
                    print("🟢 Flash Attention computation works")
                except Exception as e:
                    print(f"✗ Flash Attention computation failed: {e}")

            except ImportError:
                print("🚫 Flash Attention functions not available")

        except ImportError:
            print(
                "🚫 Flash Attention package not installed (install with: pip install flash-attn)"
            )

        # Check if xformers is available (alternative to flash_attn)
        try:
            import xformers  # type: ignore

            print(f"🟢 XFormers {xformers.__version__}")

            # Test xformers attention
            try:
                from xformers.ops import memory_efficient_attention  # type: ignore

                print("🟢 XFormers memory efficient attention available")
            except ImportError:
                print("🚫 XFormers memory efficient attention not available")

        except ImportError:
            print("🚫 XFormers not installed (install with: pip install xformers)")

        return True

    except Exception as e:
        print(f"🚫 Flash Attention test failed: {e}")
        return False


def test_sage_attention():
    """Test Sage Attention availability"""
    print("\nTesting Sage Attention...")

    try:
        # Check if sage_attn package is installed
        try:
            import sage_attn  # type: ignore

            print(f"🟢 Sage Attention package {sage_attn.__version__}")

            # Test basic sage attention functionality
            try:
                from sage_attn import sage_attention  # type: ignore

                print("🟢 Sage Attention functions available")

                # Test if sage attention actually works
                try:
                    import torch

                    # Create test tensors for sage attention
                    q = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    k = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    v = torch.randn(
                        2,
                        4,
                        8,
                        64,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    # Test sage attention
                    output = sage_attention(q, k, v)
                    print("🟢 Sage Attention computation works")
                except Exception as e:
                    print(f"🚫 Sage Attention computation failed: {e}")

            except ImportError:
                print("🚫 Sage Attention functions not available")

        except ImportError:
            print(
                "🚫 Sage Attention package not installed (install with: pip install sage-attn)"
            )

        return True

    except Exception as e:
        print(f"🚫 Sage Attention test failed: {e}")
        return False


def test_triton():
    """Test Triton availability"""
    print("\nTesting Triton...")

    try:
        # Check if triton is importable
        try:
            import triton

            print(f"🟢 Triton {triton.__version__}")

            # Test basic triton functionality
            try:
                import triton.compiler

                print("🟢 Triton compiler available")
            except ImportError:
                print("🚫 Triton compiler not available")

        except ImportError:
            print("🚫 Triton not installed (install with: pip install triton)")
            return True  # Not critical for basic functionality

        # Test if inductor backend works (requires C++ compiler)
        try:
            import torch

            test_fn = torch.compile(lambda x: x * 2, backend="inductor")
            test_result = test_fn(torch.tensor([1.0]))
            print("🟢 Triton inductor backend works")
        except Exception as e:
            # Extract just the main error message
            error_msg = str(e)
            if "Compiler: cl is not found" in error_msg:
                error_msg = "cl is not found"
            elif "backend='inductor' raised" in error_msg:
                error_msg = "inductor backend compilation failed"
            else:
                error_msg = f"{error_msg}"
            print(
                f"🚫 Triton inductor backend not working: {error_msg} (will fallback to eager backend)"
            )

        return True

    except Exception as e:
        print(f"🚫 Triton test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"🟢 CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"🟢 CUDA version: {torch.version.cuda}")  # type: ignore
            print(
                f"🟢 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            return True
        else:
            print("🚫 CUDA not available - will use CPU")
            return True
    except Exception as e:
        print(f"🚫 CUDA test failed: {e}")
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
            print(f"🟢 {file_path}")
        else:
            print(f"🚫 {file_path} - missing!")
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
            print(f"🟢 Configuration loaded from {config_path}")
            return True
        else:
            print(f"🚫 Configuration file not found: {config_path}")
            return False
    except Exception as e:
        print(f"🚫 Configuration loading failed: {e}")
        return False


def test_torch_compile():
    """Test torch.compile functionality"""
    print("\nTesting torch.compile...")

    try:
        import torch

        # Test basic compilation with eager backend
        def test_function(x):
            return torch.nn.functional.relu(x)

        try:
            compiled_fn = torch.compile(test_function, backend="eager")
            test_input = torch.randn(10, 10)
            result = compiled_fn(test_input)
            print("🟢 torch.compile with eager backend")
            return True
        except Exception as e:
            print(f"🚫 torch.compile test failed: {e}")
            return False

    except Exception as e:
        print(f"🚫 torch.compile test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Takenoko Installation Test")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("CUDA Test", test_cuda),
        ("Flash Attention Test", test_flash_attention),
        ("Sage Attention Test", test_sage_attention),
        ("Triton Test", test_triton),
        ("Project Structure", test_project_structure),
        ("Configuration Loading", test_config_loading),
        ("Torch.Compile Test", test_torch_compile),
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
