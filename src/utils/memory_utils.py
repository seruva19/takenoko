from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional


def configure_cuda_allocator_from_config(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Configure PyTorch CUDA caching allocator via environment variable.

    Reads top-level keys (recommended) usually placed under the OPTIMIZATION section
    of the TOML. Legacy [cuda_allocator] table is no longer supported.

    Supported keys (top-level):
      - cuda_allocator_enable: bool
      - cuda_allocator_max_split_size_mb: int > 0
      - cuda_allocator_expandable_segments: bool

    If PYTORCH_CUDA_ALLOC_CONF is already set in the environment, this function
    will not override it.
    """

    log = logger or logging.getLogger(__name__)

    # Respect existing environment
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        log.info("Using existing PYTORCH_CUDA_ALLOC_CONF from environment")
        return

    # Preferred: top-level keys
    enable = bool(config.get("cuda_allocator_enable", False))
    max_split_raw = config.get("cuda_allocator_max_split_size_mb")
    expandable_raw = config.get("cuda_allocator_expandable_segments")

    if not enable:
        return

    options: list[str] = []

    # expandable_segments
    if expandable_raw is not None:
        try:
            expandable_val = bool(expandable_raw)
            options.append(
                f"expandable_segments:{'True' if expandable_val else 'False'}"
            )
        except Exception:
            # ignore invalid value
            pass

    # max_split_size_mb
    if max_split_raw is not None:
        try:
            msz = int(max_split_raw)  # type: ignore[arg-type]
            if msz > 0:
                options.append(f"max_split_size_mb:{msz}")
        except Exception:
            # ignore invalid value
            pass

    if not options:
        return

    env_value = ",".join(options)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_value
    log.info(f"Set PYTORCH_CUDA_ALLOC_CONF='{env_value}' from config")
