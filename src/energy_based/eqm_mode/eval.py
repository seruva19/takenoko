from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_npz_from_samples(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str = "eqm_samples",
    limit: Optional[int] = None,
) -> Path:
    """Persist generated samples to an NPZ file for downstream evaluation.

    Args:
        samples: Tensor shaped [N, C, ...] in the range [0, 1] or [-1, 1].
        output_dir: Directory where the NPZ should be written.
        prefix: Filename prefix.
        limit: Optional cap on the number of samples written.

    Returns:
        Path to the saved NPZ file.
    """

    samples = samples.detach().cpu()
    if limit is not None and limit > 0:
        samples = samples[:limit]

    if samples.dtype.is_floating_point:
        samples = samples.clamp(0, 1)
        samples = (samples * 255).to(torch.uint8)

    np_array = samples.numpy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / f"{prefix}.npz"
    np.savez(npz_path, arr_0=np_array)
    return npz_path
