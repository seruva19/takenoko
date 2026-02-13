"""
DeT-style trajectory supervision dataset precheck helpers.

This module verifies:
1. Each video has a matching trajectory file.
2. Each trajectory file maps to a known video (no orphans).
3. Trajectory tensors are readable and shaped as [T, N, C>=2].
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

WINDOW_SUFFIX_RE = re.compile(r"_(\d+)-(\d+)$")


@dataclass(frozen=True)
class TrajectoryPrecheckOptions:
    dataset_root: Path
    videos_subdir: str = "videos"
    trajectories_subdir: str = "trajectories"
    video_exts_csv: str = ".mp4,.avi,.mov,.mkv,.webm"
    trajectory_ext: str = ".pth"
    recursive: bool = False
    expect_visibility: bool = True
    require_finite: bool = True
    skip_tensor_check: bool = False
    max_report: int = 20


@dataclass(frozen=True)
class TrajectoryPrecheckResult:
    dataset_root: Path
    dataset_roots_detected: int
    total_video_files: int
    total_trajectory_files: int
    missing: Tuple[str, ...]
    orphan: Tuple[str, ...]
    duplicate_videos: Tuple[str, ...]
    duplicate_trajs: Tuple[str, ...]
    invalid_traj: Tuple[str, ...]
    warnings: Tuple[str, ...]
    error_message: Optional[str] = None

    @property
    def has_failure(self) -> bool:
        return bool(
            self.error_message
            or self.missing
            or self.orphan
            or self.duplicate_videos
            or self.duplicate_trajs
            or self.invalid_traj
        )

    @property
    def status(self) -> str:
        if self.has_failure:
            return "FAILED"
        if self.warnings:
            return "PASSED_WITH_WARNINGS"
        return "PASSED"


def parse_exts(csv_text: str) -> List[str]:
    exts: List[str] = []
    for raw in csv_text.split(","):
        ext = raw.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    return exts


def canonical_stem(path: Path) -> str:
    return WINDOW_SUFFIX_RE.sub("", path.stem)


def collect_files(root: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    if not root.exists():
        return []
    ext_set = {ext.lower() for ext in exts}
    pattern = "**/*" if recursive else "*"
    files: List[Path] = []
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in ext_set:
            files.append(path)
    files.sort()
    return files


def find_dataset_roots(dataset_root: Path, videos_subdir: str) -> List[Path]:
    direct = dataset_root / videos_subdir
    if direct.exists() and direct.is_dir():
        return [dataset_root]

    roots: List[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / videos_subdir).exists():
            roots.append(child)
    return roots


def first_tensor(payload: object) -> Optional[torch.Tensor]:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if torch.is_tensor(value):
                return value
        return None
    if isinstance(payload, (list, tuple)):
        for value in payload:
            if torch.is_tensor(value):
                return value
        return None
    return None


def validate_trajectory_tensor(
    path: Path,
    *,
    expect_visibility: bool,
    require_finite: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return ([f"{path}: failed to load ({exc})"], warnings)

    tensor = first_tensor(payload)
    if tensor is None:
        return ([f"{path}: no tensor found in payload"], warnings)

    try:
        tensor = tensor.detach().to(dtype=torch.float32, device="cpu")
    except Exception as exc:
        return ([f"{path}: failed to normalize tensor ({exc})"], warnings)

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 3:
        errors.append(f"{path}: expected 3D tensor [T,N,C], got shape={tuple(tensor.shape)}")
        return (errors, warnings)

    if tensor.shape[-1] < 2:
        errors.append(f"{path}: expected last dim >= 2, got shape={tuple(tensor.shape)}")
        return (errors, warnings)

    if require_finite and not bool(torch.isfinite(tensor).all().item()):
        errors.append(f"{path}: tensor contains non-finite values")

    if expect_visibility and tensor.shape[-1] < 3:
        warnings.append(f"{path}: no visibility channel (C={tensor.shape[-1]})")

    coords = tensor[..., :2]
    abs_max = float(coords.abs().max().item()) if coords.numel() > 0 else 0.0
    if abs_max > 2.0:
        warnings.append(
            f"{path}: coordinate magnitude looks unnormalized (max_abs={abs_max:.3f})"
        )

    # Heuristic warning for likely [N,T,C] storage.
    if tensor.shape[0] > tensor.shape[1]:
        warnings.append(
            f"{path}: first dim > second dim ({tensor.shape[0]}>{tensor.shape[1]}), verify [T,N,C] ordering"
        )

    return (errors, warnings)


def print_examples(label: str, lines: Sequence[str], max_examples: int) -> List[str]:
    lines_out: List[str] = []
    if not lines:
        return lines_out
    lines_out.append(f"{label}: {len(lines)}")
    for line in lines[:max_examples]:
        lines_out.append(f"  - {line}")
    remaining = len(lines) - max_examples
    if remaining > 0:
        lines_out.append(f"  - ... and {remaining} more")
    return lines_out


def run_trajectory_precheck(
    options: TrajectoryPrecheckOptions,
) -> TrajectoryPrecheckResult:
    dataset_root = Path(options.dataset_root)
    if not dataset_root.exists():
        return TrajectoryPrecheckResult(
            dataset_root=dataset_root,
            dataset_roots_detected=0,
            total_video_files=0,
            total_trajectory_files=0,
            missing=(),
            orphan=(),
            duplicate_videos=(),
            duplicate_trajs=(),
            invalid_traj=(),
            warnings=(),
            error_message=f"dataset root does not exist: {dataset_root}",
        )

    video_exts = parse_exts(options.video_exts_csv)
    traj_ext = (
        options.trajectory_ext
        if options.trajectory_ext.startswith(".")
        else f".{options.trajectory_ext}"
    )
    traj_ext = traj_ext.lower()

    dataset_roots = find_dataset_roots(dataset_root, options.videos_subdir)
    if not dataset_roots:
        return TrajectoryPrecheckResult(
            dataset_root=dataset_root,
            dataset_roots_detected=0,
            total_video_files=0,
            total_trajectory_files=0,
            missing=(),
            orphan=(),
            duplicate_videos=(),
            duplicate_trajs=(),
            invalid_traj=(),
            warnings=(),
            error_message=(
                "no dataset roots found. Expected either:\n"
                f"  - {dataset_root / options.videos_subdir}\n"
                f"  - {dataset_root}/*/{options.videos_subdir}"
            ),
        )

    missing: List[str] = []
    orphan: List[str] = []
    duplicate_videos: List[str] = []
    duplicate_trajs: List[str] = []
    invalid_traj: List[str] = []
    warnings: List[str] = []

    total_video_files = 0
    total_trajectory_files = 0

    for root in dataset_roots:
        rel_root = "." if root == dataset_root else os.path.relpath(root, dataset_root)
        videos_dir = root / options.videos_subdir
        trajectories_dir = root / options.trajectories_subdir

        video_files = collect_files(videos_dir, video_exts, options.recursive)
        traj_files = collect_files(trajectories_dir, [traj_ext], options.recursive)
        total_video_files += len(video_files)
        total_trajectory_files += len(traj_files)

        video_map: Dict[str, List[Path]] = {}
        for video_path in video_files:
            stem = canonical_stem(video_path)
            video_map.setdefault(stem, []).append(video_path)

        traj_map: Dict[str, List[Path]] = {}
        for traj_path in traj_files:
            stem = canonical_stem(traj_path)
            traj_map.setdefault(stem, []).append(traj_path)

        for stem, items in video_map.items():
            if len(items) > 1:
                duplicate_videos.append(f"{rel_root}: stem='{stem}' ({len(items)} files)")
        for stem, items in traj_map.items():
            if len(items) > 1:
                duplicate_trajs.append(f"{rel_root}: stem='{stem}' ({len(items)} files)")

        for stem in sorted(video_map.keys()):
            if stem not in traj_map:
                expected = trajectories_dir / f"{stem}{traj_ext}"
                missing.append(f"{rel_root}: missing trajectory for '{stem}' (expected {expected})")

        for stem in sorted(traj_map.keys()):
            if stem not in video_map:
                orphan.append(f"{rel_root}: orphan trajectory '{stem}'")

        if not options.skip_tensor_check:
            for stem, paths in traj_map.items():
                del stem
                for path in paths:
                    errs, warns = validate_trajectory_tensor(
                        path,
                        expect_visibility=options.expect_visibility,
                        require_finite=options.require_finite,
                    )
                    invalid_traj.extend(errs)
                    warnings.extend(warns)

    return TrajectoryPrecheckResult(
        dataset_root=dataset_root,
        dataset_roots_detected=len(dataset_roots),
        total_video_files=total_video_files,
        total_trajectory_files=total_trajectory_files,
        missing=tuple(missing),
        orphan=tuple(orphan),
        duplicate_videos=tuple(duplicate_videos),
        duplicate_trajs=tuple(duplicate_trajs),
        invalid_traj=tuple(invalid_traj),
        warnings=tuple(warnings),
        error_message=None,
    )


def format_trajectory_precheck_report(
    result: TrajectoryPrecheckResult,
    max_report: int = 20,
) -> str:
    lines = [
        "DeT Trajectory Precheck",
        f"- dataset_root: {result.dataset_root}",
        f"- dataset_roots_detected: {result.dataset_roots_detected}",
        f"- video_files: {result.total_video_files}",
        f"- trajectory_files: {result.total_trajectory_files}",
    ]

    if result.error_message:
        lines.append(f"ERROR: {result.error_message}")

    lines.extend(print_examples("Missing trajectories", result.missing, max_report))
    lines.extend(print_examples("Orphan trajectories", result.orphan, max_report))
    lines.extend(print_examples("Duplicate video stems", result.duplicate_videos, max_report))
    lines.extend(print_examples("Duplicate trajectory stems", result.duplicate_trajs, max_report))
    lines.extend(print_examples("Invalid trajectories", result.invalid_traj, max_report))
    lines.extend(print_examples("Warnings", result.warnings, max_report))
    lines.append(f"Result: {result.status}")
    return "\n".join(lines)
