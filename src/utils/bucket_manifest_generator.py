"""
Bucket Manifest Generator

Generates detailed text reports of bucket distribution and dataset statistics
at training initialization. Saved to the output directory for reference.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def generate_bucket_manifest(
    train_dataset_group,
    val_dataset_group: Optional[Any],
    output_dir: str,
    output_name: str,
    global_step: int = 0,
    args: Optional[Any] = None,
) -> None:
    """
    Generate a detailed bucket manifest text file at training start.

    Creates a human-readable report of all buckets, their resolutions,
    frame counts (for video), and item distribution.

    Args:
        train_dataset_group: Training dataset group
        val_dataset_group: Optional validation dataset group
        output_dir: Output directory path
        output_name: Output name for the training run
        global_step: Current global step (skip if > 0 to avoid logging on resume)
        args: Optional argparse.Namespace with additional config
    """
    # Only generate on first run, not on resume
    if global_step > 0:
        logger.debug("Skipping bucket manifest generation (resuming from checkpoint)")
        return

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate manifest filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = os.path.join(
            output_dir, f"{output_name}_bucket_manifest_{timestamp}.txt"
        )

        logger.info(f"📋 Generating bucket manifest: {manifest_path}")

        with open(manifest_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("TAKENOKO TRAINING BUCKET MANIFEST\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Name: {output_name}\n")
            f.write(f"Output Directory: {output_dir}\n")
            if args:
                f.write(f"Task: {getattr(args, 'task', 'N/A')}\n")
                f.write(f"Network Type: {getattr(args, 'network_module', 'N/A')}\n")
                f.write(f"Learning Rate: {getattr(args, 'learning_rate', 'N/A')}\n")
                f.write(
                    f"Batch Size (base): {getattr(args, 'train_batch_size', 'N/A')}\n"
                )
            f.write("=" * 80 + "\n\n")

            # Write training dataset manifest
            _write_dataset_manifest(f, train_dataset_group, "TRAINING DATASET", args)

            # Write validation dataset manifest if available
            if val_dataset_group is not None:
                f.write("\n\n")
                _write_dataset_manifest(
                    f, val_dataset_group, "VALIDATION DATASET", args
                )

            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF MANIFEST\n")
            f.write("=" * 80 + "\n")

        logger.info(f"✅ Bucket manifest saved: {manifest_path}")

    except Exception as e:
        logger.warning(f"Failed to generate bucket manifest: {e}")


def _write_dataset_manifest(
    f,
    dataset_group,
    title: str,
    args: Optional[Any] = None,
) -> None:
    """
    Write detailed manifest for a dataset group.

    Args:
        f: File handle to write to
        dataset_group: Dataset group to analyze
        title: Title for this section
        args: Optional argparse.Namespace
    """
    f.write(f"{title}\n")
    f.write("-" * 80 + "\n\n")

    # Overall statistics
    total_items = getattr(dataset_group, "num_train_items", 0)
    num_datasets = (
        len(dataset_group.datasets) if hasattr(dataset_group, "datasets") else 0
    )

    f.write(f"Total Items: {total_items:,}\n")
    f.write(f"Number of Datasets: {num_datasets}\n\n")

    # Collect bucket information
    bucket_info = _collect_bucket_info(dataset_group)

    if not bucket_info:
        f.write("(No bucket information available)\n")
        return

    # Write summary table
    f.write("BUCKET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(
        f"{'Bucket Resolution':<25} {'Frames':<10} {'Items':<12} {'%':<8} {'Aspect Ratio':<12}\n"
    )
    f.write("-" * 80 + "\n")

    # Sort buckets by resolution (width * height), then by frame count
    sorted_buckets = sorted(
        bucket_info.items(),
        key=lambda x: (x[0][0] * x[0][1], x[0][2] if len(x[0]) > 2 else 0),
    )

    total_bucket_items = sum(info["count"] for info in bucket_info.values())

    for bucket_key, info in sorted_buckets:
        if len(bucket_key) == 3:
            # Video bucket: (width, height, frames)
            width, height, frames = bucket_key
            resolution_str = f"{width}x{height}x{frames}"
            frames_str = str(frames)
        else:
            # Image bucket: (width, height)
            width, height = bucket_key
            resolution_str = f"{width}x{height}"
            frames_str = "1 (image)"

        count = info["count"]
        percentage = (count / total_bucket_items * 100) if total_bucket_items > 0 else 0
        aspect_ratio = width / height if height > 0 else 0.0

        f.write(
            f"{resolution_str:<25} {frames_str:<10} {count:<12,} "
            f"{percentage:>6.2f}% {aspect_ratio:>11.3f}\n"
        )

    f.write("-" * 80 + "\n")
    f.write(
        f"Total: {total_bucket_items:,} items across {len(bucket_info)} buckets\n\n"
    )

    # Write detailed per-bucket information
    f.write("\nDETAILED BUCKET INFORMATION\n")
    f.write("=" * 80 + "\n\n")

    for idx, (bucket_key, info) in enumerate(sorted_buckets, 1):
        if len(bucket_key) == 3:
            width, height, frames = bucket_key
            resolution_str = f"{width}x{height}x{frames}"
            bucket_type = "Video"
        else:
            width, height = bucket_key
            resolution_str = f"{width}x{height}"
            frames = 1
            bucket_type = "Image"

        f.write(f"Bucket #{idx}: {resolution_str}\n")
        f.write(f"  Type: {bucket_type}\n")
        f.write(f"  Resolution: {width}x{height} ({width * height:,} pixels)\n")
        if bucket_type == "Video":
            f.write(f"  Frame Count: {frames}\n")
        f.write(f"  Aspect Ratio: {width / height if height > 0 else 0:.3f}\n")
        f.write(f"  Items: {info['count']:,}\n")
        f.write(f"  Batch Size: {info.get('batch_size', 'N/A')}\n")
        f.write(f"  Num Repeats: {info.get('num_repeats', 'N/A')}\n")

        # Dataset indices using this bucket
        dataset_indices = info.get("dataset_indices", [])
        if dataset_indices:
            f.write(
                f"  Dataset Indices: {', '.join(map(str, sorted(set(dataset_indices))))}\n"
            )

        f.write("\n")

    # Write resolution distribution summary
    _write_resolution_summary(f, bucket_info)

    # Write temporal statistics if available (for video datasets)
    _write_temporal_summary(f, bucket_info)


def _collect_bucket_info(dataset_group) -> Dict[Tuple, Dict]:
    """
    Collect bucket information from dataset group by reading the actual
    bucketed data from each dataset's batch_manager.

    This ensures the manifest shows the ACTUAL bucket resolutions that will
    be used during training, not just the configured base resolutions.

    Returns:
        Dictionary mapping bucket keys to info dicts.
        Bucket keys are:
        - (width, height) for images
        - (width, height, frames) for videos
    """
    bucket_info = defaultdict(
        lambda: {
            "count": 0,
            "batch_size": None,
            "num_repeats": None,
            "dataset_indices": [],
        }
    )

    if not hasattr(dataset_group, "datasets"):
        return {}

    for dataset_idx, dataset in enumerate(dataset_group.datasets):
        # Get batch size and repeats from dataset
        batch_size = getattr(dataset, "batch_size", None)
        num_repeats = getattr(dataset, "num_repeats", 1)

        # Read from dataset's batch_manager which has the actual bucketed items
        batch_manager = getattr(dataset, "batch_manager", None)

        if batch_manager and hasattr(batch_manager, "buckets"):
            # Iterate through actual buckets from the batch manager
            for bucket_reso, items in batch_manager.buckets.items():
                # bucket_reso is already the actual bucket resolution tuple
                # For images: (width, height)
                # For videos: (width, height, frames)
                bucket_key = bucket_reso

                bucket_info[bucket_key]["count"] += len(items)
                bucket_info[bucket_key]["batch_size"] = batch_size
                bucket_info[bucket_key]["num_repeats"] = num_repeats
                bucket_info[bucket_key]["dataset_indices"].append(dataset_idx)
        else:
            # Fallback: dataset doesn't have batch_manager yet
            # This shouldn't happen in normal training flow, but handle gracefully
            logger.warning(
                f"Dataset {dataset_idx} has no batch_manager. "
                f"Bucket info may be incomplete."
            )

            # Try to get from num_train_items at least
            num_items = getattr(dataset, "num_train_items", 0)
            if num_items > 0:
                # Use configured resolution as fallback
                resolution = getattr(dataset, "resolution", None)
                target_frames = getattr(dataset, "target_frames", None)

                if resolution:
                    if target_frames and len(target_frames) > 0:
                        # Video dataset - add frame count
                        for frames in target_frames:
                            bucket_key = (resolution[0], resolution[1], frames)
                            # Estimate items per frame count
                            bucket_info[bucket_key]["count"] += num_items // len(
                                target_frames
                            )
                            bucket_info[bucket_key]["batch_size"] = batch_size
                            bucket_info[bucket_key]["num_repeats"] = num_repeats
                            bucket_info[bucket_key]["dataset_indices"].append(
                                dataset_idx
                            )
                    else:
                        # Image dataset
                        bucket_key = (resolution[0], resolution[1])
                        bucket_info[bucket_key]["count"] += num_items
                        bucket_info[bucket_key]["batch_size"] = batch_size
                        bucket_info[bucket_key]["num_repeats"] = num_repeats
                        bucket_info[bucket_key]["dataset_indices"].append(dataset_idx)

    return dict(bucket_info)


def _write_resolution_summary(f, bucket_info: Dict) -> None:
    """Write resolution distribution summary."""
    if not bucket_info:
        return

    # Group by base resolution (ignoring frame count)
    resolution_groups = defaultdict(int)
    for bucket_key, info in bucket_info.items():
        if len(bucket_key) == 3:
            width, height, _ = bucket_key
        else:
            width, height = bucket_key
        resolution_groups[(width, height)] += info["count"]

    f.write("\nRESOLUTION DISTRIBUTION\n")
    f.write("-" * 80 + "\n")

    total = sum(resolution_groups.values())
    sorted_resolutions = sorted(
        resolution_groups.items(), key=lambda x: x[1], reverse=True
    )

    for (width, height), count in sorted_resolutions:
        percentage = (count / total * 100) if total > 0 else 0
        aspect_ratio = width / height if height > 0 else 0
        f.write(
            f"  {width}x{height}: {count:,} items ({percentage:.2f}%), "
            f"AR={aspect_ratio:.3f}\n"
        )

    f.write(
        f"\nTotal: {total:,} items across {len(resolution_groups)} unique resolutions\n"
    )


def _write_temporal_summary(f, bucket_info: Dict) -> None:
    """Write temporal (video frame count) summary if applicable."""
    # Check if we have any video buckets
    video_buckets = [k for k in bucket_info.keys() if len(k) == 3]

    if not video_buckets:
        return

    f.write("\n\nTEMPORAL DISTRIBUTION (VIDEO FRAMES)\n")
    f.write("-" * 80 + "\n")

    # Group by frame count
    frame_groups = defaultdict(int)
    for bucket_key in video_buckets:
        _, _, frames = bucket_key
        frame_groups[frames] += bucket_info[bucket_key]["count"]

    total_video_items = sum(frame_groups.values())
    sorted_frames = sorted(frame_groups.items())

    for frames, count in sorted_frames:
        percentage = (count / total_video_items * 100) if total_video_items > 0 else 0
        f.write(f"  {frames} frames: {count:,} items ({percentage:.2f}%)\n")

    f.write(f"\nTotal video items: {total_video_items:,}\n")

    # Statistics
    if frame_groups:
        frame_counts = []
        for frames, count in frame_groups.items():
            frame_counts.extend([frames] * count)

        import statistics

        f.write(f"  Min frames: {min(frame_counts)}\n")
        f.write(f"  Max frames: {max(frame_counts)}\n")
        f.write(f"  Mean frames: {statistics.mean(frame_counts):.2f}\n")
        f.write(f"  Median frames: {statistics.median(frame_counts):.2f}\n")
