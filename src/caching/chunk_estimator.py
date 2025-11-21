from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional

import av

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.datasource_utils import glob_images, glob_videos, load_video
from dataset.frame_extraction import generate_crop_positions


def _safe_count_video_frames(video_path: str) -> int:
    """Return total frame count for a video path.

    Tries fast metadata first via PyAV. Falls back to decoding or image dir listing.
    """
    try:
        if os.path.isfile(video_path):
            container: Any = av.open(video_path)
            try:
                stream = container.streams.video[0]
                frames_meta: int = int(getattr(stream, "frames", 0) or 0)
                frame_count: int = frames_meta if frames_meta > 0 else 0
                if frame_count <= 0:
                    try:
                        frame_count = sum(1 for _ in container.decode(video=0))
                    except Exception:
                        frame_count = 0
            finally:
                try:
                    container.close()
                except Exception:
                    pass
            return frame_count
        else:
            # Directory of frames
            return len(glob_images(video_path))
    except Exception:
        # Fallback: load fully if cheap count fails
        try:
            return len(load_video(video_path))
        except Exception:
            return 0


def _build_dataset_group(dataset_config_path: str, args: argparse.Namespace):
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(dataset_config_path)
    blueprint = blueprint_generator.generate(user_config, args)

    all_dataset_blueprints = list(blueprint.train_dataset_group.datasets)
    if len(blueprint.val_dataset_group.datasets) > 0:
        all_dataset_blueprints.extend(blueprint.val_dataset_group.datasets)

    combined = config_utils.DatasetGroupBlueprint(all_dataset_blueprints)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(
        combined,
        training=False,
        prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
    )
    return dataset_group


def _collect_dataset_stats(dataset_group) -> List[dict[str, Any]]:
    stats: List[dict[str, Any]] = []
    for ds in dataset_group.datasets:
        vdir: Optional[str] = getattr(ds, "video_directory", None)
        if not vdir:
            continue

        video_paths: List[str] = glob_videos(vdir)
        mode: str = getattr(ds, "frame_extraction", "head")
        target_frames = list(getattr(ds, "target_frames", [1]) or [1])
        frame_stride: int = int(getattr(ds, "frame_stride", 1) or 1)
        frame_sample: int = int(getattr(ds, "frame_sample", 1) or 1)
        max_frames = getattr(ds, "max_frames", None)
        caption_ext = getattr(ds, "caption_extension", ".txt")
        cache_dir = getattr(ds, "latents_cache_dir", None)

        ds_cache_chunks = 0
        ds_effective_chunks = 0
        ds_cycle_lengths: List[int] = []
        ds_video_count = 0

        for vp in video_paths:
            frame_count = _safe_count_video_frames(vp)
            if frame_count <= 0:
                continue
            pairs = generate_crop_positions(
                frame_count=frame_count,
                target_frames=target_frames,
                mode=mode,
                frame_stride=frame_stride,
                frame_sample=frame_sample,
                max_frames=max_frames,
            )
            chunk_count = len(pairs)
            if chunk_count == 0:
                continue

            ds_cache_chunks += chunk_count
            if mode == "epoch_slide":
                ds_effective_chunks += 1
                ds_cycle_lengths.append(chunk_count)
            else:
                ds_effective_chunks += chunk_count
            ds_video_count += 1

        if ds_cache_chunks == 0:
            continue

        entry: dict[str, Any] = {
            "video_directory": vdir,
            "chunks": ds_cache_chunks,
            "effective_chunks": ds_effective_chunks,
            "caption_extension": caption_ext,
            "latents_cache_dir": cache_dir,
            "mode": mode,
            "videos": ds_video_count,
        }

        if mode == "epoch_slide":
            entry["epoch_slide"] = True
            entry["epoch_cycle_min"] = min(ds_cycle_lengths) if ds_cycle_lengths else 0
            entry["epoch_cycle_max"] = max(ds_cycle_lengths) if ds_cycle_lengths else 0
            entry["epoch_cycle_avg"] = (
                sum(ds_cycle_lengths) / len(ds_cycle_lengths)
                if ds_cycle_lengths
                else 0.0
            )
        else:
            entry["epoch_slide"] = False

        stats.append(entry)

    return stats


def estimate_latent_cache_chunks(
    dataset_config_path: str, args: argparse.Namespace
) -> int:
    """Estimate total number of latent cache chunks across all video datasets."""
    dataset_group = _build_dataset_group(dataset_config_path, args)
    stats = _collect_dataset_stats(dataset_group)
    return sum(entry["chunks"] for entry in stats)


def estimate_latent_cache_chunks_per_dataset(
    dataset_config_path: str, args: argparse.Namespace
) -> List[dict[str, Any]]:
    """Estimate chunk counts per dataset for latent caching.

    Returns a list of dicts with keys: 'video_directory', 'chunks', 'caption_extension',
    'latents_cache_dir'. Datasets without a video_directory are skipped.
    """
    dataset_group = _build_dataset_group(dataset_config_path, args)
    return _collect_dataset_stats(dataset_group)
