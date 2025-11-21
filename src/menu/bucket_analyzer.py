"""
Bucket Analysis Tool for Takenoko Menu

Analyzes dataset configurations and predicts bucket distributions.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image
import av

from dataset.config_utils import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from dataset.buckets import BucketSelector
from dataset.datasource_utils import glob_images, glob_videos
from dataset.frame_extraction import generate_crop_positions
import numpy as np
from collections import Counter


class BucketAnalyzer:
    """Analyzes how datasets will be bucketed."""

    def __init__(self, dataset_params, dataset_type: str):
        """
        Args:
            dataset_params: Dataset parameters from blueprint
            dataset_type: "image" or "video"
        """
        self.params = dataset_params
        self.dataset_type = dataset_type

        # Create bucket selector
        self.bucket_selector = BucketSelector(
            resolution=self.params.resolution,
            enable_bucket=self.params.enable_bucket,
            no_upscale=self.params.bucket_no_upscale,
            constraint_type=self.params.bucket_constraint_type,
            constrained_dimension=self.params._constrained_dimension,
        )

        self.bucket_stats: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

    def analyze_image(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Analyze a single image and return its bucket resolution."""
        try:
            with Image.open(image_path) as img:
                original_size = img.size  # (width, height)
                bucket_reso = self.bucket_selector.get_bucket_resolution(original_size)

                self.bucket_stats[bucket_reso].append(
                    {
                        "path": image_path,
                        "original_size": original_size,
                        "type": "image",
                    }
                )

                return bucket_reso
        except Exception as e:
            print(f"   ⚠️  Failed to analyze {Path(image_path).name}: {e}")
            return None

    def analyze_video(self, video_path: str) -> Optional[List[Tuple[int, int]]]:
        """Analyze a single video and return bucket resolutions."""
        try:
            target_frames = getattr(self.params, "target_frames", [1])
            frame_extraction_mode = getattr(self.params, "frame_extraction", "head")
            frame_sample = getattr(self.params, "frame_sample", 1)
            frame_stride = getattr(self.params, "frame_stride", 1)
            max_frames = getattr(self.params, "max_frames", None)

            if os.path.isfile(video_path):
                container = av.open(video_path)
                video_stream = container.streams.video[0]
                try:
                    frame_count = video_stream.frames
                    if frame_count == 0:
                        raise ValueError("Stream reports 0 frames")
                except Exception:
                    frame_count = 0
                    for _ in container.decode(video=0):
                        frame_count += 1

                container.seek(0)
                first_frame = next(container.decode(video=0)).to_image()
                container.close()

                if first_frame and frame_count > 0:
                    original_size = first_frame.size
                    return self._record_video_segments(
                        video_path=video_path,
                        original_size=original_size,
                        frame_count=frame_count,
                        entry_type="video",
                        mode=frame_extraction_mode,
                        target_frames=target_frames,
                        frame_sample=frame_sample,
                        frame_stride=frame_stride,
                        max_frames=max_frames,
                    )
            else:
                image_files = glob_images(video_path)
                if image_files:
                    with Image.open(image_files[0]) as img:
                        original_size = img.size
                    frame_count = len(image_files)
                    return self._record_video_segments(
                        video_path=video_path,
                        original_size=original_size,
                        frame_count=frame_count,
                        entry_type="video_dir",
                        mode=frame_extraction_mode,
                        target_frames=target_frames,
                        frame_sample=frame_sample,
                        frame_stride=frame_stride,
                        max_frames=max_frames,
                    )
        except Exception as e:
            print(f"   ⚠️  Failed to analyze {Path(video_path).name}: {e}")
            return None

    def _record_video_segments(
        self,
        video_path: str,
        original_size: Tuple[int, int],
        frame_count: int,
        entry_type: str,
        mode: str,
        target_frames: List[int],
        frame_sample: int,
        frame_stride: int,
        max_frames: Optional[int],
    ) -> List[Tuple[int, int]]:
        bucket_reso = self.bucket_selector.get_bucket_resolution(original_size)
        segments = generate_crop_positions(
            frame_count=frame_count,
            target_frames=target_frames,
            mode=mode,
            frame_stride=frame_stride,
            frame_sample=frame_sample,
            max_frames=max_frames,
        )

        if not segments:
            return []

        buckets: List[Tuple[int, int]] = []
        total_segments = len(segments)

        if mode == "epoch_slide":
            # Only one effective sample per epoch; store metadata for cycle analysis
            seg_length = segments[0][1]
            self.bucket_stats[bucket_reso].append(
                {
                    "path": video_path,
                    "original_size": original_size,
                    "type": entry_type,
                    "frames": seg_length,
                    "total_frames": frame_count,
                    "segment_idx": 0,
                    "total_segments": total_segments,
                    "effective_segments": 1,
                    "epoch_slide": True,
                }
            )
            buckets.append(bucket_reso)
        else:
            for seg_idx, (_, seg_length) in enumerate(segments):
                self.bucket_stats[bucket_reso].append(
                    {
                        "path": video_path,
                        "original_size": original_size,
                        "type": entry_type,
                        "frames": seg_length,
                        "total_frames": frame_count,
                        "segment_idx": seg_idx,
                        "total_segments": total_segments,
                        "effective_segments": 1,
                        "epoch_slide": False,
                    }
                )
                buckets.append(bucket_reso)

        return buckets

    def scan_dataset(self) -> int:
        """Scan all items in the dataset. Returns number of items scanned."""
        scanned = 0

        if self.dataset_type == "image":
            image_dir = self.params.image_directory
            if not os.path.exists(image_dir):
                print(f"   ⚠️  Image directory not found: {image_dir}")
                return 0

            image_files = glob_images(image_dir)
            for img_path in image_files:
                if self.analyze_image(img_path):
                    scanned += 1

        elif self.dataset_type == "video":
            video_dir = self.params.video_directory
            if not os.path.exists(video_dir):
                print(f"   ⚠️  Video directory not found: {video_dir}")
                return 0

            video_files = glob_videos(video_dir)

            # Also check for video directories (frame sequences)
            try:
                subdirs = [
                    d
                    for d in Path(video_dir).iterdir()
                    if d.is_dir() and glob_images(str(d))
                ]
                all_videos = video_files + [str(d) for d in subdirs]
            except:
                all_videos = video_files

            for vid_path in all_videos:
                result = self.analyze_video(vid_path)
                if result:
                    scanned += len(result)  # Count each target_frame as separate item

        return scanned

    def analyze_temporal_statistics(self) -> Optional[Dict]:
        """
        Analyze temporal characteristics of video dataset.

        Returns statistics about:
        - Frame count distribution
        - Segment distribution per video
        - Temporal coverage analysis
        - Motion/scene change estimates (if available)
        """
        if self.dataset_type != "video":
            return None

        temporal_stats = {
            "frame_counts": [],
            "segment_counts": Counter(),
            "effective_segments": 0,
            "physical_segments": 0,
            "videos_analyzed": 0,
            "short_videos": 0,  # < min(target_frames)
            "long_videos": 0,  # > max(target_frames)
            "perfect_fit": 0,  # == target_frames
            "epoch_slide_videos": 0,
            "epoch_slide_cycles": Counter(),
        }

        target_frames = getattr(self.params, "target_frames", [1])
        min_target = min(target_frames)
        max_target = max(target_frames)

        for bucket_reso, items in self.bucket_stats.items():
            for item in items:
                if item["type"] in ["video", "video_dir"]:
                    total_frames = item.get("total_frames", 0)
                    total_segments = item.get("total_segments", 1)

                    # Only count unique videos (segment_idx == 0 or not present)
                    if item.get("segment_idx", 0) == 0:
                        temporal_stats["frame_counts"].append(total_frames)
                        temporal_stats["segment_counts"][total_segments] += 1
                        temporal_stats["videos_analyzed"] += 1
                        temporal_stats["physical_segments"] += total_segments

                        if total_frames < min_target:
                            temporal_stats["short_videos"] += 1
                        elif total_frames > max_target:
                            temporal_stats["long_videos"] += 1
                        elif total_frames in target_frames:
                            temporal_stats["perfect_fit"] += 1

                        if item.get("epoch_slide"):
                            temporal_stats["epoch_slide_videos"] += 1
                            temporal_stats["epoch_slide_cycles"][total_segments] += 1

                    temporal_stats["effective_segments"] += item.get(
                        "effective_segments", 1
                    )

        if not temporal_stats["frame_counts"]:
            return None

        # Calculate statistics
        frame_counts = np.array(temporal_stats["frame_counts"])
        temporal_stats["stats"] = {
            "min_frames": int(np.min(frame_counts)),
            "max_frames": int(np.max(frame_counts)),
            "mean_frames": float(np.mean(frame_counts)),
            "median_frames": float(np.median(frame_counts)),
            "std_frames": float(np.std(frame_counts)),
            "p25_frames": float(np.percentile(frame_counts, 25)),
            "p75_frames": float(np.percentile(frame_counts, 75)),
        }

        # Calculate segment efficiency
        temporal_stats["avg_segments_per_video"] = (
            temporal_stats["physical_segments"] / temporal_stats["videos_analyzed"]
            if temporal_stats["videos_analyzed"] > 0
            else 0
        )

        return temporal_stats

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_items = sum(len(items) for items in self.bucket_stats.values())
        num_buckets = len(self.bucket_stats)

        bucket_summaries = []
        for bucket_reso, items in sorted(self.bucket_stats.items()):
            width, height = bucket_reso
            area = width * height
            aspect_ratio = width / height

            # Calculate average original size
            avg_width = sum(item["original_size"][0] for item in items) / len(items)
            avg_height = sum(item["original_size"][1] for item in items) / len(items)

            bucket_summaries.append(
                {
                    "resolution": bucket_reso,
                    "count": len(items),
                    "percentage": (
                        (len(items) / total_items * 100) if total_items > 0 else 0
                    ),
                    "aspect_ratio": aspect_ratio,
                    "area": area,
                    "avg_original_size": (avg_width, avg_height),
                    "items": items,
                }
            )

        return {
            "total_items": total_items,
            "num_buckets": num_buckets,
            "buckets": bucket_summaries,
            "batch_size": self.params.batch_size,
            "num_repeats": self.params.num_repeats,
            "constraint_type": self.params.bucket_constraint_type,
            "constrained_dimension": self.params._constrained_dimension,
        }


def print_dataset_summary(
    dataset_idx: int,
    dataset_type: str,
    summary: Dict,
    dataset_params,
    temporal_stats: Optional[Dict] = None,
):
    """Print bucket analysis summary for a single dataset."""
    print(f"\n{'='*80}")
    print(f"DATASET {dataset_idx + 1}: {dataset_type.upper()}")
    print(f"{'='*80}")

    # Show directory
    if dataset_type == "image":
        print(f"Directory: {dataset_params.image_directory}")
    else:
        print(f"Directory: {dataset_params.video_directory}")

    print(f"\nConfiguration:")
    print(f"  Resolution: {dataset_params.resolution}")
    print(f"  Batch size: {summary['batch_size']}")
    print(f"  Num repeats: {summary['num_repeats']}")
    print(f"  Bucketing: {'Enabled' if dataset_params.enable_bucket else 'Disabled'}")
    print(f"  Constraint type: {summary['constraint_type']}")
    if summary["constrained_dimension"]:
        print(f"  Constrained dimension: {summary['constrained_dimension']}")
    if dataset_params.bucket_no_upscale:
        print(f"  No upscale: Yes")

    print(f"\nOverview:")
    print(f"  Total items found: {summary['total_items']}")
    print(f"  Total buckets: {summary['num_buckets']}")
    print(f"  Items per epoch: {summary['total_items'] * summary['num_repeats']}")
    batches_per_epoch = (
        summary["total_items"] * summary["num_repeats"] + summary["batch_size"] - 1
    ) // summary["batch_size"]
    print(f"  Batches per epoch: {batches_per_epoch}")

    if summary["num_buckets"] == 0:
        print(f"\n⚠️  No items found to analyze!")
        return

    print(f"\n{'='*80}")
    print(f"BUCKET DISTRIBUTION")
    print(f"{'='*80}")
    print(
        f"{'Resolution':<15} {'Count':<8} {'%':<8} {'AR':<8} {'Area':<12} {'Avg Original':<20}"
    )
    print("-" * 80)

    for bucket in summary["buckets"]:
        w, h = bucket["resolution"]
        avg_w, avg_h = bucket["avg_original_size"]
        print(
            f"{w}x{h:<11} {bucket['count']:<8} {bucket['percentage']:>6.1f}% "
            f"{bucket['aspect_ratio']:>6.2f}   {bucket['area']:<12,} "
            f"{int(avg_w)}x{int(avg_h)}"
        )

    # Show temporal statistics for video datasets
    if dataset_type == "video" and temporal_stats:
        print(f"\n{'='*80}")
        print(f"TEMPORAL STATISTICS")
        print(f"{'='*80}")

        stats = temporal_stats["stats"]
        print(f"\nFrame Count Distribution:")
        print(f"  Videos analyzed: {temporal_stats['videos_analyzed']}")
        print(
            f"  Effective per-epoch segments: {temporal_stats['effective_segments']}"
        )
        print(
            f"  Physical segments (cached windows): {temporal_stats['physical_segments']}"
        )
        print(f"  Avg windows/video: {temporal_stats['avg_segments_per_video']:.2f}")
        print(f"\n  Min frames: {stats['min_frames']}")
        print(f"  Max frames: {stats['max_frames']}")
        print(f"  Mean frames: {stats['mean_frames']:.1f} ± {stats['std_frames']:.1f}")
        print(f"  Median frames: {stats['median_frames']:.1f}")
        print(f"  25th percentile: {stats['p25_frames']:.1f}")
        print(f"  75th percentile: {stats['p75_frames']:.1f}")

        # Show target_frames compatibility
        target_frames = getattr(dataset_params, "target_frames", [1])
        print(f"\nTarget Frames Compatibility (target={target_frames}):")
        print(
            f"  Short videos (< min target): {temporal_stats['short_videos']} ({temporal_stats['short_videos']/temporal_stats['videos_analyzed']*100:.1f}%)"
        )
        print(
            f"  Perfect fit (== target): {temporal_stats['perfect_fit']} ({temporal_stats['perfect_fit']/temporal_stats['videos_analyzed']*100:.1f}%)"
        )
        print(
            f"  Long videos (> max target): {temporal_stats['long_videos']} ({temporal_stats['long_videos']/temporal_stats['videos_analyzed']*100:.1f}%)"
        )

        # Show segment distribution
        print(f"\nSegment Distribution:")
        sorted_segments = sorted(temporal_stats["segment_counts"].items())
        for num_segments, count in sorted_segments[:10]:  # Show top 10
            pct = count / temporal_stats["videos_analyzed"] * 100
            print(f"  {num_segments} segment(s): {count} videos ({pct:.1f}%)")
        if len(sorted_segments) > 10:
            print(f"  ... and {len(sorted_segments) - 10} more segment counts")

        if temporal_stats["epoch_slide_videos"] > 0:
            print(f"\nEpoch Slide Coverage:")
            total_epoch_videos = temporal_stats["epoch_slide_videos"]
            print(f"  Videos using epoch_slide: {total_epoch_videos}")
            for cycle_len, count in sorted(
                temporal_stats["epoch_slide_cycles"].items()
            ):
                pct = count / total_epoch_videos * 100
                print(
                    f"  Cycle length {cycle_len} windows: {count} videos ({pct:.1f}%)"
                )

    # Show most common buckets with examples
    if summary["buckets"]:
        print(f"\n{'='*80}")
        print(f"TOP 3 BUCKETS (by item count)")
        print(f"{'='*80}")

        top_buckets = sorted(
            summary["buckets"], key=lambda x: x["count"], reverse=True
        )[:3]

        for i, bucket in enumerate(top_buckets, 1):
            w, h = bucket["resolution"]
            print(
                f"\n{i}. Bucket {w}x{h} - {bucket['count']} items ({bucket['percentage']:.1f}%)"
            )

            # Show first 3 items as examples
            for j, item in enumerate(bucket["items"][:3], 1):
                orig_w, orig_h = item["original_size"]
                path = Path(item["path"]).name
                if len(path) > 50:
                    path = "..." + path[-47:]

                if item["type"] == "image":
                    print(f"   [{j}] {path:<50} {orig_w}x{orig_h}")
                else:
                    frames_info = f"({item.get('frames', 1)}f)"
                    # Show segment info if available
                    if "total_segments" in item and item["total_segments"] > 1:
                        seg_info = (
                            f"seg {item['segment_idx']+1}/{item['total_segments']}"
                        )
                        print(
                            f"   [{j}] {path:<50} {orig_w}x{orig_h} {frames_info} {seg_info}"
                        )
                    else:
                        print(f"   [{j}] {path:<50} {orig_w}x{orig_h} {frames_info}")

            if len(bucket["items"]) > 3:
                print(f"   ... and {len(bucket['items']) - 3} more")


def print_unified_summary_tables(
    train_summaries: List[Dict], val_summaries: List[Dict]
):
    """Print comprehensive unified tables for all bucket distributions."""

    def print_unified_table(summaries: List[Dict], title: str):
        """Print a unified table for a set of summaries."""
        if not summaries:
            return

        print(f"\n{'='*120}")
        print(f"{title}")
        print(f"{'='*120}")

        # Collect all unique buckets across all datasets
        all_buckets = defaultdict(
            lambda: {
                "datasets": [],
                "total_count": 0,
                "total_percentage": 0.0,
                "aspect_ratio": 0.0,
                "area": 0,
            }
        )

        # Aggregate bucket data
        for ds_info in summaries:
            ds_idx = ds_info["index"]
            ds_type = ds_info["type"]
            summary = ds_info["summary"]

            for bucket in summary["buckets"]:
                resolution = bucket["resolution"]
                w, h = resolution
                key = (w, h)

                all_buckets[key]["datasets"].append(
                    {
                        "index": ds_idx,
                        "type": ds_type,
                        "count": bucket["count"],
                        "percentage": bucket["percentage"],
                    }
                )
                all_buckets[key]["total_count"] += bucket["count"]
                all_buckets[key]["aspect_ratio"] = bucket["aspect_ratio"]
                all_buckets[key]["area"] = bucket["area"]

        # Calculate total items across all datasets
        total_items = sum(
            ds_info["summary"]["total_items"] * ds_info["summary"]["num_repeats"]
            for ds_info in summaries
        )

        # Sort buckets by total count
        sorted_buckets = sorted(
            all_buckets.items(), key=lambda x: x[1]["total_count"], reverse=True
        )

        # Print header
        header_parts = ["Resolution", "Total", "%", "AR", "Area"]
        for ds_info in summaries:
            ds_idx = ds_info["index"]
            ds_type = ds_info["type"][:3].upper()  # IMG or VID
            header_parts.append(f"D{ds_idx}({ds_type})")

        # Build format string
        col_widths = [12, 8, 7, 6, 12] + [10] * len(summaries)
        header = "".join(
            f"{part:<{width}}" for part, width in zip(header_parts, col_widths)
        )
        print(header)
        print("-" * 120)

        # Print each bucket
        for (w, h), bucket_data in sorted_buckets:
            # Calculate overall percentage
            overall_pct = (
                (bucket_data["total_count"] / total_items * 100)
                if total_items > 0
                else 0
            )

            row = [
                f"{w}x{h}",
                f'{bucket_data["total_count"]}',
                f"{overall_pct:.1f}%",
                f'{bucket_data["aspect_ratio"]:.2f}',
                f'{bucket_data["area"]:,}',
            ]

            # Add per-dataset counts
            for ds_info in summaries:
                ds_idx = ds_info["index"]
                # Find if this dataset has this bucket
                ds_count = "-"
                for ds_data in bucket_data["datasets"]:
                    if ds_data["index"] == ds_idx:
                        ds_count = f'{ds_data["count"]} ({ds_data["percentage"]:.1f}%)'
                        break
                row.append(ds_count)

            # Print formatted row
            print("".join(f"{val:<{width}}" for val, width in zip(row, col_widths)))

        # Print summary stats
        print("-" * 120)
        summary_row = ["TOTAL", str(total_items), "100.0%", "-", "-"]
        for ds_info in summaries:
            ds_total = (
                ds_info["summary"]["total_items"] * ds_info["summary"]["num_repeats"]
            )
            summary_row.append(str(ds_total))
        print("".join(f"{val:<{width}}" for val, width in zip(summary_row, col_widths)))

        # Print dataset info
        print(f"\n{'Legend:'}")
        for ds_info in summaries:
            ds_idx = ds_info["index"]
            ds_type = ds_info["type"]
            params = ds_info["params"]
            summary = ds_info["summary"]

            if ds_type == "image":
                directory = params.image_directory
            else:
                directory = params.video_directory

            # Build distinguishing info
            info_parts = []

            # Caption extension (if not default)
            caption_ext = getattr(params, "caption_extension", ".txt")
            if caption_ext and caption_ext != ".txt":
                info_parts.append(f"captions={caption_ext}")

            # Cache directory (if specified)
            cache_dir = getattr(params, "latents_cache_dir", None)
            if cache_dir:
                cache_name = Path(cache_dir).name
                info_parts.append(f"cache={cache_name}")

            # Target frames (if not default)
            target_frames = getattr(params, "target_frames", None)
            if target_frames and target_frames != [1]:
                info_parts.append(f"frames={target_frames}")

            # Build info string
            distinguishing_info = f" ({', '.join(info_parts)})" if info_parts else ""

            print(
                f"  D{ds_idx} ({ds_type}): {Path(directory).name}{distinguishing_info} "
                f"- {summary['total_items']} items × {summary['num_repeats']} repeats "
                f"= {summary['total_items'] * summary['num_repeats']} total "
                f"(batch_size={summary['batch_size']})"
            )

        # Add unified temporal statistics summary for video datasets
        video_datasets = [
            ds for ds in summaries if ds["type"] == "video" and ds.get("temporal_stats")
        ]
        if video_datasets:
            print(f"\n{'='*120}")
            print(f"UNIFIED TEMPORAL STATISTICS (Video Datasets Only)")
            print(f"{'='*120}")

            # Aggregate temporal stats
            total_videos = sum(
                ds["temporal_stats"]["videos_analyzed"] for ds in video_datasets
            )
            total_effective = sum(
                ds["temporal_stats"]["effective_segments"] for ds in video_datasets
            )
            total_physical = sum(
                ds["temporal_stats"]["physical_segments"] for ds in video_datasets
            )
            all_frame_counts = []
            all_short = sum(
                ds["temporal_stats"]["short_videos"] for ds in video_datasets
            )
            all_long = sum(ds["temporal_stats"]["long_videos"] for ds in video_datasets)
            all_perfect = sum(
                ds["temporal_stats"]["perfect_fit"] for ds in video_datasets
            )

            for ds in video_datasets:
                all_frame_counts.extend(ds["temporal_stats"]["frame_counts"])

            if all_frame_counts:
                # Calculate aggregate statistics
                frame_counts = np.array(all_frame_counts)
                print(f"\nAggregate Frame Distribution:")
                print(f"  Total videos: {total_videos}")
                print(f"  Effective per-epoch segments: {total_effective}")
                print(f"  Physical segments (cached windows): {total_physical}")
                print(
                    f"  Avg windows/video: {total_physical/total_videos:.2f}"
                )
                print(f"  Min frames: {int(np.min(frame_counts))}")
                print(f"  Max frames: {int(np.max(frame_counts))}")
                print(
                    f"  Mean frames: {np.mean(frame_counts):.1f} ± {np.std(frame_counts):.1f}"
                )
                print(f"  Median frames: {np.median(frame_counts):.1f}")

                print(f"\nTarget Compatibility (across all video datasets):")
                print(
                    f"  Short videos: {all_short} ({all_short/total_videos*100:.1f}%)"
                )
                print(
                    f"  Perfect fit: {all_perfect} ({all_perfect/total_videos*100:.1f}%)"
                )
                print(f"  Long videos: {all_long} ({all_long/total_videos*100:.1f}%)")

                # Per-dataset comparison table
                print(f"\nPer-Dataset Temporal Comparison:")
                print(
                    f"{'Dataset':<15} {'Videos':<10} {'Phys Seg':<10} {'Eff Seg':<10} {'Avg Win/Vid':<12} {'Mean±Std Frames':<25} {'Short%':<10} {'Long%':<10}"
                )
                print("-" * 120)

                for ds in video_datasets:
                    ts = ds["temporal_stats"]
                    ds_idx = ds["index"]
                    ds_type_short = ds["type"][:3].upper()
                    frames = np.array(ts["frame_counts"])

                    print(
                        f"D{ds_idx}({ds_type_short}){'':9} "
                        f"{ts['videos_analyzed']:<10} "
                        f"{ts['physical_segments']:<10} "
                        f"{ts['effective_segments']:<10} "
                        f"{ts['avg_segments_per_video']:<12.2f} "
                        f"{np.mean(frames):.1f}±{np.std(frames):.1f}{'':13} "
                        f"{ts['short_videos']/ts['videos_analyzed']*100:>6.1f}%    "
                        f"{ts['long_videos']/ts['videos_analyzed']*100:>6.1f}%"
                    )

    # Print training table
    if train_summaries:
        print_unified_table(train_summaries, "UNIFIED TRAINING BUCKET DISTRIBUTION")

    # Print validation table
    if val_summaries:
        print_unified_table(val_summaries, "UNIFIED VALIDATION BUCKET DISTRIBUTION")


def analyze_dataset_buckets(config, args):
    """Main function to analyze all datasets in config."""

    # Create blueprint from config
    sanitizer = ConfigSanitizer()
    generator = BlueprintGenerator(sanitizer)
    blueprint = generator.generate(config, args)

    # Track all summaries for unified table
    train_summaries = []
    val_summaries = []

    # Analyze training datasets
    if len(blueprint.train_dataset_group.datasets) > 0:
        print(f"\n{'='*80}")
        print(
            f"TRAINING DATASETS ({len(blueprint.train_dataset_group.datasets)} total)"
        )
        print(f"{'='*80}")

        for i, dataset_blueprint in enumerate(blueprint.train_dataset_group.datasets):
            dataset_type = "image" if dataset_blueprint.is_image_dataset else "video"
            params = dataset_blueprint.params

            print(f"\nScanning {dataset_type} dataset {i+1}...")
            analyzer = BucketAnalyzer(params, dataset_type)
            num_scanned = analyzer.scan_dataset()
            print(f"   Scanned {num_scanned} items")

            # Analyze temporal statistics for video datasets
            temporal_stats = None
            if dataset_type == "video":
                print(f"   Analyzing temporal characteristics...")
                temporal_stats = analyzer.analyze_temporal_statistics()

            summary = analyzer.get_summary()
            print_dataset_summary(i, dataset_type, summary, params, temporal_stats)

            # Store for unified table
            train_summaries.append(
                {
                    "index": i + 1,
                    "type": dataset_type,
                    "summary": summary,
                    "params": params,
                    "temporal_stats": temporal_stats,  # Add temporal stats
                }
            )

    # Analyze validation datasets if any
    if len(blueprint.val_dataset_group.datasets) > 0:
        print(f"\n{'='*80}")
        print(
            f"VALIDATION DATASETS ({len(blueprint.val_dataset_group.datasets)} total)"
        )
        print(f"{'='*80}")

        for i, dataset_blueprint in enumerate(blueprint.val_dataset_group.datasets):
            dataset_type = "image" if dataset_blueprint.is_image_dataset else "video"
            params = dataset_blueprint.params

            print(f"\nScanning {dataset_type} validation dataset {i+1}...")
            analyzer = BucketAnalyzer(params, dataset_type)
            num_scanned = analyzer.scan_dataset()
            print(f"   Scanned {num_scanned} items")

            # Analyze temporal statistics for video datasets
            temporal_stats = None
            if dataset_type == "video":
                print(f"   Analyzing temporal characteristics...")
                temporal_stats = analyzer.analyze_temporal_statistics()

            summary = analyzer.get_summary()
            print_dataset_summary(i, dataset_type, summary, params, temporal_stats)

            # Store for unified table
            val_summaries.append(
                {
                    "index": i + 1,
                    "type": dataset_type,
                    "summary": summary,
                    "params": params,
                    "temporal_stats": temporal_stats,  # Add temporal stats
                }
            )

    # Print unified summary tables
    print_unified_summary_tables(train_summaries, val_summaries)

    print(f"\n{'='*80}")
    print(f"BUCKET ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
