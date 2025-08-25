import torch
from typing import Any


def compile_optimize(model: Any) -> None:
    """One-shot torch.compile of hot paths on a WanModel-like object.

    Expects attributes:
      - optimized_compile: bool
      - compile_args: list | tuple | None -> [backend, mode, dynamic, fullgraph]
      - text_embedding, head, blocks
      - rope_func (optional: "comfy")

    The function is safe to call multiple times; it disables model.optimized_compile
    after first successful setup. All failures are swallowed to avoid breaking runs.
    """
    try:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None or not bool(getattr(model, "optimized_compile", False)):
            return

        # one-time setup
        setattr(model, "optimized_compile", False)

        backend, mode, dynamic, fullgraph = getattr(model, "compile_args", None) or [
            "inductor",
            "default",
            None,
            "False",
        ]
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
                backend=backend,
                mode=mode,
                dynamic=dyn_val,
                fullgraph=fg_val,
            )
        except Exception:
            pass
        try:
            model.head = compile_fn(  # type: ignore[attr-defined]
                model.head,
                backend=backend,
                mode=mode,
                dynamic=dyn_val,
                fullgraph=fg_val,
            )
        except Exception:
            pass
        try:
            for _blk in model.blocks:  # type: ignore[attr-defined]
                _blk._forward = compile_fn(  # type: ignore[attr-defined]
                    _blk._forward,
                    backend=backend,
                    mode=mode,
                    dynamic=dyn_val,
                    fullgraph=fg_val,
                )
        except Exception:
            pass

        # Optional: comfy rope path
        if str(getattr(model, "rope_func", "default")) == "comfy":
            try:
                for _blk in model.blocks:  # type: ignore[attr-defined]
                    if hasattr(_blk.self_attn, "comfyrope"):
                        _blk.self_attn.comfyrope = compile_fn(  # type: ignore[attr-defined]
                            _blk.self_attn.comfyrope,
                            backend=backend,
                            mode=mode,
                            dynamic=dyn_val,
                            fullgraph=fg_val,
                        )
            except Exception:
                pass
    except Exception:
        # Best-effort only; never break the main path
        return
