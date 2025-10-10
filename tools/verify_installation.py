#!/usr/bin/env python3

"""
Test script to verify Takenoko installation
"""

import sys
import os
import subprocess
from pathlib import Path


def ensure_msvc_env() -> bool:
    """Ensure MSVC/OpenMP environment variables are available on Windows."""
    if os.name != "nt":
        return True

    required_keys = ("INCLUDE", "LIB", "LIBPATH")
    if all(os.environ.get(key) for key in required_keys):
        return True

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        print(
            "⚠  torch.compile backend 'inductor' skipped: Visual Studio Build Tools not detected."
        )
        print(
            "    😌 INFO: Install VS Build Tools (Desktop C++ + OpenMP) to enable inductor."
        )
        return False

    try:
        install_dir = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.Component.MSBuild",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        print(f"⚠  torch.compile backend 'inductor' skipped: vswhere failed ({exc}).")
        print(
            "    😌 INFO: Reinstall VS Build Tools or run install from a VS developer prompt."
        )
        return False

    if not install_dir:
        print(
            "⚠  torch.compile backend 'inductor' skipped: VS Build Tools installation not found."
        )
        print(
            "    😌 INFO: Install VS Build Tools (Desktop C++ + OpenMP) to enable inductor."
        )
        return False

    vcvars = Path(install_dir) / "VC/Auxiliary/Build/vcvarsall.bat"
    if not vcvars.exists():
        print(f"⚠  torch.compile backend 'inductor' skipped: {vcvars} missing.")
        print("    😌 INFO: Repair or reinstall VS Build Tools.")
        return False

    try:
        env_dump = subprocess.check_output(
            f'"{vcvars}" x64 & set',
            shell=True,
            encoding="mbcs",
            errors="ignore",
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"⚠  torch.compile backend 'inductor' skipped: vcvarsall.bat failed ({exc})."
        )
        print(
            "    😌 INFO: Run install from an MSVC developer prompt after fixing VS setup."
        )
        return False

    for line in env_dump.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key_upper = key.upper()
        if key_upper in {"INCLUDE", "LIB", "LIBPATH", "PATH"}:
            if key_upper == "PATH":
                existing = os.environ.get(key_upper, "")
                os.environ[key_upper] = value + (
                    os.pathsep + existing if existing else ""
                )
            else:
                os.environ[key_upper] = value

    return True


def summarize_exception(exc: Exception) -> str:
    """Return a compact, single-line summary for an exception."""
    exc_text = str(exc)
    if "UnicodeDecodeError" in exc_text:
        return "UnicodeDecodeError reading MSVC output (missing OpenMP headers?)"
    message_lines = [line.strip() for line in exc_text.splitlines() if line.strip()]
    if not message_lines:
        return repr(exc)
    return message_lines[0]


def emit_backend_warning(backend: str, exc: Exception) -> None:
    """Emit warning and guidance for torch.compile backend failures."""
    summary = summarize_exception(exc)
    print(f"⚠  torch.compile backend '{backend}' failed: {summary}")

    tips: list[str] = []
    text = str(exc)
    if "UnicodeDecodeError" in text or "omp.h" in text:
        tips.append(
            "Install VS Build Tools with OpenMP and run install from the Native Tools prompt."
        )
    elif "cl is not found" in text:
        tips.append(
            "Run install from the Native Tools command prompt so cl.exe is on PATH."
        )
    else:
        tips.append("Set TORCH_LOGS=+dynamo for details or continue with eager mode.")

    for tip in tips:
        print(f"    😌 INFO: {tip}")
    if backend == "eager":
        print(
            "    😌 INFO: Investigate the eager backend failure; training depends on it."
        )
    else:
        print(
            "    😌 INFO: Training will continue using eager mode; this is not fatal."
        )


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
                    print(
                        "    😌 INFO: Flash Attention is optional; training can proceed without it."
                    )

            except ImportError:
                print("🚫 Flash Attention functions not available")
                print(
                    "    😌 INFO: Flash Attention is optional; training can proceed without it."
                )

        except ImportError:
            print(
                "🚫 Flash Attention package not installed (install with: pip install flash-attn)"
            )
            print(
                "    😌 INFO: Flash Attention is optional; training can proceed without it."
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
                print(
                    "    😌 INFO: XFormers attention is optional; training can proceed without it."
                )

        except ImportError:
            print("🚫 XFormers not installed (install with: pip install xformers)")
            print("    😌 INFO: XFormers is optional; training can proceed without it.")

        return True

    except Exception as e:
        print(f"🚫 Flash Attention test failed: {e}")
        print(
            "    😌 INFO: Flash Attention is optional; training can proceed without it."
        )
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
                    print(
                        "    😌 INFO: Sage Attention is optional; training can proceed without it."
                    )

            except ImportError:
                print("🚫 Sage Attention functions not available")
                print(
                    "    😌 INFO: Sage Attention is optional; training can proceed without it."
                )

        except ImportError:
            print(
                "🚫 Sage Attention package not installed (install with: pip install sage-attn)"
            )
            print(
                "    😌 INFO: Sage Attention is optional; training can proceed without it."
            )

        return True

    except Exception as e:
        print(f"🚫 Sage Attention test failed: {e}")
        print(
            "    😌 INFO: Sage Attention is optional; training can proceed without it."
        )
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
                print(
                    "    😌 INFO: Triton is optional; training can proceed without it."
                )

        except ImportError:
            print("🚫 Triton not installed (install with: pip install triton)")
            print("    😌 INFO: Triton is optional; training can proceed without it.")
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
            print(
                "    😌 INFO: Training will continue in eager mode; this is not fatal."
            )

        return True

    except Exception as e:
        print(f"🚫 Triton test failed: {e}")
        print("    😌 INFO: Triton is optional; training can proceed without it.")
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
            print("    😌 INFO: Training will run on CPU; expect slower performance.")
            return True
    except Exception as e:
        print(f"🚫 CUDA test failed: {e}")
        print(
            "    😌 INFO: Training can still run on CPU; investigate CUDA setup if GPU is desired."
        )
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
    except Exception as e:
        print(f"🚫 torch.compile import failed: {e}")
        print("    😌 INFO: Install PyTorch to run training.")
        return False

    def test_function(x):
        return torch.nn.functional.relu(x)

    success = True
    for backend in ("eager", "inductor"):
        if backend == "inductor" and os.name == "nt" and not ensure_msvc_env():
            print(
                "⚠  torch.compile backend 'inductor' skipped: MSVC/OpenMP toolchain not detected."
            )
            print(
                "    😌 INFO: Install VS Build Tools (Desktop C++ + OpenMP) and rerun from the Native Tools prompt."
            )
            print(
                "    😌 INFO: Training will continue using eager mode; this is not fatal."
            )
            continue

        try:
            compiled_fn = torch.compile(test_function, backend=backend)
            test_input = torch.randn(10, 10)
            _ = compiled_fn(test_input)
            print(f"🟢 torch.compile backend '{backend}' passed")
        except Exception as exc:
            emit_backend_warning(backend, exc)
            if backend == "eager":
                success = False

    return success


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
