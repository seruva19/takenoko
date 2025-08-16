"""
Centralized optional dependency management to prevent duplicate warnings.

This module provides a global warning system that ensures each missing optional
dependency is only warned about once across all modules.
"""

import logging
from typing import Any, Dict, Set


def get_simple_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Simple logger without custom formatting to avoid circular imports."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_simple_logger(__name__, level=logging.INFO)

# Global tracking of warnings to prevent duplicates across all modules
_GLOBAL_WARNINGS_SHOWN: Set[str] = set()


def try_import_with_warning(
    module_name: str,
    import_items: list[str] | None = None,
    success_message: str | None = None,
    warning_message: str | None = None,
) -> tuple[bool, Dict[str, Any]]:
    """
    Try to import a module and its items with centralized warning tracking.

    Args:
        module_name: Name of the module to import (e.g., 'flash_attn')
        import_items: List of items to import from the module (e.g., ['flash_attn_func'])
        success_message: Custom success message (defaults to "✅️ {module_name}: available")
        warning_message: Custom warning message (defaults to "⚠️ {module_name}: not available")

    Returns:
        tuple: (success: bool, imports: Dict[str, Any])
               success is True if import succeeded, False otherwise
               imports contains the imported items (None values if import failed)
    """
    if success_message is None:
        success_message = f"✅️ {module_name}: available"
    if warning_message is None:
        warning_message = f"⚠️ {module_name}: not available"

    imports = {}

    try:
        from importlib import import_module

        # Import the main module
        module = import_module(module_name)

        # Import specific items if requested
        if import_items:
            for item in import_items:
                if (
                    "." in item
                ):  # Handle nested imports like 'flash_attn.flash_attn_interface.flash_attn_func'
                    full_module_path, attr_name = item.rsplit(".", 1)
                    submodule = import_module(full_module_path)
                    imports[attr_name] = getattr(submodule, attr_name)
                else:
                    imports[item] = getattr(module, item)
        else:
            # For modules without specific import items, use the last part of the module name as key
            key = module_name.split(".")[-1]
            imports[key] = module

        # Only log success message once per module
        if module_name not in _GLOBAL_WARNINGS_SHOWN:
            if success_message != "":
                logger.info(success_message)
            _GLOBAL_WARNINGS_SHOWN.add(module_name)

        return True, imports

    except ImportError:
        # Only show warning once globally using an environment variable
        import os

        env_var = f"{module_name.upper().replace('.', '_')}_WARNING_SHOWN"
        if not os.environ.get(env_var, False):
            logger.warning(warning_message)
            os.environ[env_var] = "1"
        # Set all imports to None
        if import_items:
            for item in import_items:
                if "." in item:
                    imports[item.split(".")[-1]] = None
                else:
                    imports[item] = None
        else:
            key = module_name.split(".")[-1]
            imports[key] = None

        return False, imports


def setup_flash_attention():
    """Setup FlashAttention imports with centralized warning management."""
    # Import the main module first
    success, main_imports = try_import_with_warning(
        "flash_attn",
        success_message="✅️ FlashAttention: available",
        warning_message="⚠️ FlashAttention: not available",
    )

    if not success:
        return None, None, None, None

    # Import specific functions from submodules
    success, func_imports = try_import_with_warning(
        "flash_attn.flash_attn_interface",
        ["_flash_attn_forward", "flash_attn_varlen_func", "flash_attn_func"],
    )

    return (
        main_imports["flash_attn"],
        func_imports["_flash_attn_forward"],
        func_imports["flash_attn_varlen_func"],
        func_imports["flash_attn_func"],
    )


def setup_sageattention():
    """Setup SageAttention imports with centralized warning management."""
    success, imports = try_import_with_warning(
        "sageattention",
        ["sageattn_varlen", "sageattn"],
        "✅️ SageAttention: available",
        "⚠️ SageAttention: not available",
    )

    return imports["sageattn_varlen"], imports["sageattn"]


def setup_xformers():
    """Setup Xformers imports with centralized warning management."""
    success, imports = try_import_with_warning(
        "xformers.ops",
        success_message="✅️ Xformers: available",
        warning_message="⚠️ Xformers: not available",
    )

    return imports.get("ops")


def setup_pillow_extensions():
    """Setup Pillow extension imports with centralized warning management."""
    results = {}

    # Try pillow_avif
    success, imports = try_import_with_warning(
        "pillow_avif",
        success_message="",
        warning_message="📄 pillow_avif: not available",
    )
    results["pillow_avif"] = imports.get("pillow_avif")

    # Try jxlpy
    success, imports = try_import_with_warning(
        "jxlpy",
        ["JXLImagePlugin"],
        success_message="",
        warning_message="📄 jxlpy: not available",
    )
    results["jxlpy"] = imports.get("JXLImagePlugin")

    # Try pillow_jxl
    success, imports = try_import_with_warning(
        "pillow_jxl", success_message="", warning_message="📄 pillow_jxl: not available"
    )
    results["pillow_jxl"] = imports.get("pillow_jxl")

    return results
