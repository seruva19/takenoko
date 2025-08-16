from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int
        Base random seed to set across libraries. Expected range: [0, 2**31-1].
    deterministic : bool, default True
        If True, configures cuDNN for deterministic behavior. May reduce performance.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN / backend settings
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        # When deterministic is True, benchmark should be False for determinism
        torch.backends.cudnn.benchmark = not bool(deterministic)
    except Exception:
        # Some backends may not be available depending on the build
        pass
