"""Utility helpers for working with EqM NPZ exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def load_npz_array(path: Path) -> np.ndarray:
    """Load the `arr_0` entry from an NPZ file."""
    data = np.load(path)
    if "arr_0" not in data:
        raise KeyError(f"File {path} does not contain 'arr_0'.")
    return data["arr_0"]


def merge_npz_files(
    files: Sequence[Path],
    *,
    limit: Optional[int] = None,
) -> np.ndarray:
    """Concatenate multiple NPZ payloads along the batch axis."""
    if not files:
        raise ValueError("At least one NPZ file is required to merge.")

    arrays = [load_npz_array(path) for path in files]
    reference_shape = arrays[0].shape[1:]
    for path, array in zip(files, arrays):
        if array.shape[1:] != reference_shape:
            raise ValueError(
                f"Shape mismatch for {path} (expected {reference_shape}, got {array.shape[1:]})"
            )

    merged = np.concatenate(arrays, axis=0)
    if limit is not None and limit > 0:
        merged = merged[:limit]
    return merged


def merge_npz_directory(
    source_dir: Path,
    *,
    pattern: str = "*.npz",
    limit: Optional[int] = None,
) -> np.ndarray:
    """Convenience wrapper that merges every NPZ file in ``source_dir``."""
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files matching {pattern} found in {source_dir}")
    return merge_npz_files(files, limit=limit)


def compute_npz_summary(array: np.ndarray) -> dict:
    """Return a dictionary of simple statistics for an NPZ payload."""
    summary = {
        "count": int(array.shape[0]) if array.ndim >= 1 else 1,
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
    if array.ndim >= 2:
        summary["channels"] = int(array.shape[1])
    if array.ndim >= 3:
        summary["height"] = int(array.shape[-2])
        summary["width"] = int(array.shape[-1])
    return summary


def aggregate_npz_directory(
    source_dir: Path,
    *,
    pattern: str = "*.npz",
    limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> dict:
    """Load NPZ samples from a directory, optionally save them, and return stats."""
    merged = merge_npz_directory(source_dir, pattern=pattern, limit=limit)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, arr_0=merged)

    summary = compute_npz_summary(merged)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            _format_summary(summary),
            encoding="utf-8",
        )

    return summary


def _format_summary(summary: dict) -> str:
    """Return a deterministic JSON-ish string for persisted summaries."""
    # Avoid importing json just to serialise a small dict.
    lines = ["{"]
    items = list(summary.items())
    for idx, (key, value) in enumerate(items):
        comma = "," if idx + 1 < len(items) else ""
        if isinstance(value, str):
            lines.append(f'  "{key}": "{value}"{comma}')
        else:
            lines.append(f'  "{key}": {value}{comma}')
    lines.append("}")
    return "\n".join(lines)
