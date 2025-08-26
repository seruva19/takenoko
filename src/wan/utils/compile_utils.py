import torch
from typing import Any
import logging
import platform
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def check_triton_availability() -> bool:
    """Check if Triton is available and working"""
    try:
        import triton  # type: ignore

        # Test basic Triton functionality
        import triton.compiler  # type: ignore

        # Test if inductor backend actually works (not just if Triton is importable)
        try:
            import torch

            # Test a simple compilation with inductor
            test_fn = torch.compile(lambda x: x * 2, backend="inductor")
            test_result = test_fn(torch.tensor([1.0]))
            return True
        except Exception as e:
            logger.error(f"❌ Triton inductor backend not working: {e}")
            # Inductor failed, likely due to missing C++ compiler
            return False

    except (ImportError, AttributeError):
        return False


def select_backend(requested_backend: str) -> str:
    """Select the appropriate backend based on availability and requirements"""

    # If eager is requested, use it directly
    if requested_backend == "eager":
        logger.info("✅ Using 'eager' backend as requested")
        return "eager"

    # If inductor is requested, check Triton availability
    if requested_backend == "inductor":
        if check_triton_availability():
            logger.info("✅ Triton available - using 'inductor' backend")
            return "inductor"
        else:
            logger.warning(
                "⚠️ Triton not available - cancelling optimized torch compile"
            )
            return "CANCEL"

    # For other backends, try them as-is
    logger.info(f"✅ Using '{requested_backend}' backend")
    return requested_backend


def compile_optimize(model: Any) -> None:
    """One-shot torch.compile of hot paths on a WanModel-like object.

    Expects attributes:
      - optimized_torch_compile: bool
      - compile_args: list | tuple | None -> [backend, mode, dynamic, fullgraph]
      - text_embedding, head, blocks
      - rope_func (optional: "comfy")

    The function is safe to call multiple times; it disables model.optimized_torch_compile
    after first successful setup. All failures are swallowed to avoid breaking runs.
    """

    logger.info("🔥 Compiling model with optimized_torch_compile")
    try:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None or not bool(
            getattr(model, "optimized_torch_compile", False)
        ):
            return

        # one-time setup
        setattr(model, "optimized_torch_compile", False)

        backend, mode, dynamic, fullgraph = getattr(model, "compile_args", None) or [
            "eager",
            "default",
            None,
            "False",
        ]

        # Select appropriate backend based on availability
        selected_backend = select_backend(backend)

        # If backend selection was cancelled, exit early
        if selected_backend == "CANCEL":
            logger.warning(
                "🚫 Optimized torch compile cancelled - proceeding without compilation"
            )
            return

        dyn_val = (
            None
            if dynamic is None or str(dynamic).lower() == "auto"
            else (str(dynamic).lower() == "true")
        )
        fg_val = str(fullgraph).lower() == "true"

        # Compile core modules
        try:
            model.text_embedding = compile_fn(  # type: ignore[attr-defined]
                model.text_embedding,
                backend=selected_backend,
                mode=mode,
                dynamic=dyn_val,
                fullgraph=fg_val,
            )
            logger.info(
                f"✅ Successfully compiled text_embedding with backend: {selected_backend}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to compile text_embedding: {e}")
            # Try fallback to "eager" backend if current backend fails
            if selected_backend != "eager":
                try:
                    logger.info(
                        "🔄 Trying fallback to 'eager' backend for text_embedding"
                    )
                    model.text_embedding = compile_fn(  # type: ignore[attr-defined]
                        model.text_embedding,
                        backend="eager",
                        mode=mode,
                        dynamic=dyn_val,
                        fullgraph=fg_val,
                    )
                    logger.info(
                        "✅ Successfully compiled text_embedding with eager backend"
                    )
                except Exception as e2:
                    logger.error(
                        f"❌ Failed to compile text_embedding with eager backend: {e2}"
                    )
            pass
        try:
            model.head = compile_fn(  # type: ignore[attr-defined]
                model.head,
                backend=selected_backend,
                mode=mode,
                dynamic=dyn_val,
                fullgraph=fg_val,
            )
            logger.info(
                f"✅ Successfully compiled head with backend: {selected_backend}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to compile head: {e}")
            # Try fallback to "eager" backend if current backend fails
            if selected_backend != "eager":
                try:
                    logger.info("🔄 Trying fallback to 'eager' backend for head")
                    model.head = compile_fn(  # type: ignore[attr-defined]
                        model.head,
                        backend="eager",
                        mode=mode,
                        dynamic=dyn_val,
                        fullgraph=fg_val,
                    )
                    logger.info("✅ Successfully compiled head with eager backend")
                except Exception as e2:
                    logger.error(f"❌ Failed to compile head with eager backend: {e2}")
            pass
        try:
            for i, _blk in enumerate(model.blocks):  # type: ignore[attr-defined]
                _blk._forward = compile_fn(  # type: ignore[attr-defined]
                    _blk._forward,
                    backend=selected_backend,
                    mode=mode,
                    dynamic=dyn_val,
                    fullgraph=fg_val,
                )
            logger.info(
                f"✅ Successfully compiled {len(model.blocks)} blocks with backend: {selected_backend}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to compile blocks: {e}")
            # Try fallback to "eager" backend if current backend fails
            if selected_backend != "eager":
                try:
                    logger.info("🔄 Trying fallback to 'eager' backend for blocks")
                    for i, _blk in enumerate(model.blocks):  # type: ignore[attr-defined]
                        _blk._forward = compile_fn(  # type: ignore[attr-defined]
                            _blk._forward,
                            backend="eager",
                            mode=mode,
                            dynamic=dyn_val,
                            fullgraph=fg_val,
                        )
                    logger.info("✅ Successfully compiled blocks with eager backend")
                except Exception as e2:
                    logger.error(
                        f"❌ Failed to compile blocks with eager backend: {e2}"
                    )
            pass

        # Optional: comfy rope path
        if str(getattr(model, "rope_func", "default")) == "comfy":
            try:
                for _blk in model.blocks:  # type: ignore[attr-defined]
                    if hasattr(_blk.self_attn, "comfyrope"):
                        _blk.self_attn.comfyrope = compile_fn(  # type: ignore[attr-defined]
                            _blk.self_attn.comfyrope,
                            backend=selected_backend,
                            mode=mode,
                            dynamic=dyn_val,
                            fullgraph=fg_val,
                        )
                logger.info(
                    f"✅ Successfully compiled comfyrope with backend: {selected_backend}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to compile comfyrope: {e}")
                # Try fallback to "eager" backend if current backend fails
                if selected_backend != "eager":
                    try:
                        logger.info(
                            "🔄 Trying fallback to 'eager' backend for comfyrope"
                        )
                        for _blk in model.blocks:  # type: ignore[attr-defined]
                            if hasattr(_blk.self_attn, "comfyrope"):
                                _blk.self_attn.comfyrope = compile_fn(  # type: ignore[attr-defined]
                                    _blk.self_attn.comfyrope,
                                    backend="eager",
                                    mode=mode,
                                    dynamic=dyn_val,
                                    fullgraph=fg_val,
                                )
                        logger.info(
                            "✅ Successfully compiled comfyrope with eager backend"
                        )
                    except Exception as e2:
                        logger.error(
                            f"❌ Failed to compile comfyrope with eager backend: {e2}"
                        )
                pass
    except Exception as e:
        logger.error(f"❌ Failed to compile model: {e}")
        # Best-effort only; never break the main path
        return
