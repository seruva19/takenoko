"""
Dataset Statistics Logger for TensorBoard

Logs comprehensive dataset statistics at training start (only on first run, not on resume).
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def log_dataset_stats_to_tensorboard(
    accelerator,
    train_dataset_group,
    val_dataset_group: Optional[Any] = None,
    global_step: int = 0,
) -> None:
    """
    Log dataset statistics to TensorBoard at training start.
    
    Only logs on first run (global_step == 0), not on resume.
    Only runs on main process.
    
    Args:
        accelerator: Accelerator instance with trackers
        train_dataset_group: Training dataset group
        val_dataset_group: Optional validation dataset group
        global_step: Current global step (skip if > 0 to avoid logging on resume)
    """
    # Only log on first run, not on resume
    if global_step > 0:
        logger.debug("Skipping dataset stats logging (resuming from checkpoint)")
        return
    
    # Only on main process
    if not accelerator.is_main_process:
        return
    
    # Only if trackers are available
    if not hasattr(accelerator, 'trackers') or not accelerator.trackers:
        logger.debug("No trackers available for dataset stats logging")
        return
    
    try:
        logger.info("📊 Logging dataset statistics to TensorBoard...")
        
        # Get TensorBoard writer
        tb_writer = None
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tb_writer = tracker.writer
                break
        
        if tb_writer is None:
            logger.debug("TensorBoard tracker not found")
            return
        
        # Log training dataset stats
        _log_dataset_group_stats(
            tb_writer,
            train_dataset_group,
            prefix="dataset/train",
            step=0
        )
        
        # Log validation dataset stats if available
        if val_dataset_group is not None:
            _log_dataset_group_stats(
                tb_writer,
                val_dataset_group,
                prefix="dataset/val",
                step=0
            )
        
        # Flush to ensure stats are written
        tb_writer.flush()
        
        # Log bucket distribution for training data
        log_bucket_distribution_to_tensorboard(
            accelerator,
            train_dataset_group,
            prefix="dataset/train/buckets",
            step=0
        )
        
        # Log bucket distribution for validation data if available
        if val_dataset_group is not None:
            log_bucket_distribution_to_tensorboard(
                accelerator,
                val_dataset_group,
                prefix="dataset/val/buckets",
                step=0
            )
        
        logger.info("✅ Dataset statistics logged to TensorBoard")
        
    except Exception as e:
        logger.warning(f"Failed to log dataset stats to TensorBoard: {e}")


def _log_dataset_group_stats(
    tb_writer,
    dataset_group,
    prefix: str = "dataset",
    step: int = 0,
) -> None:
    """
    Log statistics for a dataset group.
    
    Args:
        tb_writer: TensorBoard SummaryWriter
        dataset_group: Dataset group to analyze
        prefix: Prefix for TensorBoard tags (e.g., "dataset/train")
        step: Step number for logging
    """
    # Overall dataset statistics
    total_items = dataset_group.num_train_items
    num_datasets = len(dataset_group.datasets)
    
    tb_writer.add_scalar(f"{prefix}/total_items", total_items, step)
    tb_writer.add_scalar(f"{prefix}/num_datasets", num_datasets, step)
    
    # Per-dataset statistics
    all_batch_sizes = []
    all_repeats = []
    resolution_counts = {}
    total_segments = 0
    
    # For histograms
    resolution_areas = []
    aspect_ratios = []
    item_counts = []
    
    for idx, dataset in enumerate(dataset_group.datasets):
        batch_size = getattr(dataset, 'batch_size', 1)
        num_repeats = getattr(dataset, 'num_repeats', 1)
        
        all_batch_sizes.append(batch_size)
        all_repeats.append(num_repeats)
        
        # Get resolution if available
        resolution = getattr(dataset, 'resolution', None)
        if resolution:
            res_key = f"{resolution[0]}x{resolution[1]}"
            resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1
            
            # Calculate area and aspect ratio for histograms
            width, height = resolution[0], resolution[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 1.0
            
            resolution_areas.append(area)
            aspect_ratios.append(aspect_ratio)
        
        # Log per-dataset metrics
        tb_writer.add_scalar(f"{prefix}/dataset_{idx+1}/batch_size", batch_size, step)
        tb_writer.add_scalar(f"{prefix}/dataset_{idx+1}/num_repeats", num_repeats, step)
        
        # Try to get item count for this dataset
        try:
            dataset_items = len(dataset)
            total_segments += dataset_items
            item_counts.append(dataset_items)
            tb_writer.add_scalar(f"{prefix}/dataset_{idx+1}/items", dataset_items, step)
        except:
            pass
    
    # Aggregate statistics (scalars)
    if all_batch_sizes:
        tb_writer.add_scalar(f"{prefix}/batch_size_mean", np.mean(all_batch_sizes), step)
        tb_writer.add_scalar(f"{prefix}/batch_size_max", np.max(all_batch_sizes), step)
        tb_writer.add_scalar(f"{prefix}/batch_size_min", np.min(all_batch_sizes), step)
    
    if all_repeats:
        tb_writer.add_scalar(f"{prefix}/repeats_mean", np.mean(all_repeats), step)
        tb_writer.add_scalar(f"{prefix}/repeats_max", np.max(all_repeats), step)
    
    # Histograms for distributions
    if len(all_batch_sizes) > 0:
        tb_writer.add_histogram(f"{prefix}/distribution/batch_sizes", np.array(all_batch_sizes), step)
    
    if len(all_repeats) > 0:
        tb_writer.add_histogram(f"{prefix}/distribution/num_repeats", np.array(all_repeats), step)
    
    if len(item_counts) > 0:
        tb_writer.add_histogram(f"{prefix}/distribution/items_per_dataset", np.array(item_counts), step)
    
    if len(resolution_areas) > 0:
        tb_writer.add_histogram(f"{prefix}/distribution/resolution_areas", np.array(resolution_areas), step)
    
    if len(aspect_ratios) > 0:
        tb_writer.add_histogram(f"{prefix}/distribution/aspect_ratios", np.array(aspect_ratios), step)
    
    # Resolution distribution (scalars for counts)
    for res_key, count in resolution_counts.items():
        # Replace 'x' with underscore for TensorBoard compatibility
        safe_res_key = res_key.replace('x', '_')
        tb_writer.add_scalar(f"{prefix}/resolution/{safe_res_key}", count, step)
    
    # Total training segments/items
    if total_segments > 0:
        tb_writer.add_scalar(f"{prefix}/total_segments", total_segments, step)
        
        # Effective training items (segments × repeats)
        effective_items = sum(
            len(ds) * getattr(ds, 'num_repeats', 1)
            for ds in dataset_group.datasets
            if hasattr(ds, '__len__')
        )
        if effective_items > 0:
            tb_writer.add_scalar(f"{prefix}/effective_items_per_epoch", effective_items, step)


def log_temporal_stats_to_tensorboard(
    accelerator,
    temporal_stats_dict: Dict[str, Any],
    prefix: str = "dataset/temporal",
    step: int = 0,
) -> None:
    """
    Log temporal statistics (from bucket analyzer) to TensorBoard.
    
    Args:
        accelerator: Accelerator instance
        temporal_stats_dict: Dictionary with temporal statistics
        prefix: Prefix for TensorBoard tags
        step: Step number for logging
    """
    if not accelerator.is_main_process:
        return
    
    if not hasattr(accelerator, 'trackers') or not accelerator.trackers:
        return
    
    try:
        tb_writer = None
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tb_writer = tracker.writer
                break
        
        if tb_writer is None:
            return
        
        stats = temporal_stats_dict.get('stats', {})
        
        # Frame distribution statistics (scalars)
        if 'min_frames' in stats:
            tb_writer.add_scalar(f"{prefix}/min_frames", stats['min_frames'], step)
        if 'max_frames' in stats:
            tb_writer.add_scalar(f"{prefix}/max_frames", stats['max_frames'], step)
        if 'mean_frames' in stats:
            tb_writer.add_scalar(f"{prefix}/mean_frames", stats['mean_frames'], step)
        if 'median_frames' in stats:
            tb_writer.add_scalar(f"{prefix}/median_frames", stats['median_frames'], step)
        if 'std_frames' in stats:
            tb_writer.add_scalar(f"{prefix}/std_frames", stats['std_frames'], step)
        
        # Frame count histogram
        frame_counts = temporal_stats_dict.get('frame_counts', [])
        if len(frame_counts) > 0:
            tb_writer.add_histogram(
                f"{prefix}/distribution/frame_counts",
                np.array(frame_counts),
                step
            )
        
        # Segment distribution histogram
        segment_counts = temporal_stats_dict.get('segment_counts', {})
        if segment_counts:
            # Convert Counter to arrays for histogram
            segments_per_video = []
            for num_segments, count in segment_counts.items():
                segments_per_video.extend([num_segments] * count)
            
            if len(segments_per_video) > 0:
                tb_writer.add_histogram(
                    f"{prefix}/distribution/segments_per_video",
                    np.array(segments_per_video),
                    step
                )
        
        # Video categorization (scalars)
        if 'videos_analyzed' in temporal_stats_dict:
            total_videos = temporal_stats_dict['videos_analyzed']
            tb_writer.add_scalar(f"{prefix}/total_videos", total_videos, step)
            
            if total_videos > 0:
                short_pct = temporal_stats_dict.get('short_videos', 0) / total_videos * 100
                perfect_pct = temporal_stats_dict.get('perfect_fit', 0) / total_videos * 100
                long_pct = temporal_stats_dict.get('long_videos', 0) / total_videos * 100
                
                tb_writer.add_scalar(f"{prefix}/short_videos_pct", short_pct, step)
                tb_writer.add_scalar(f"{prefix}/perfect_fit_pct", perfect_pct, step)
                tb_writer.add_scalar(f"{prefix}/long_videos_pct", long_pct, step)
                
                # Create categorical distribution histogram
                # 0 = short, 1 = perfect, 2 = long
                category_distribution = (
                    [0] * temporal_stats_dict.get('short_videos', 0) +
                    [1] * temporal_stats_dict.get('perfect_fit', 0) +
                    [2] * temporal_stats_dict.get('long_videos', 0)
                )
                if len(category_distribution) > 0:
                    tb_writer.add_histogram(
                        f"{prefix}/distribution/video_categories",
                        np.array(category_distribution),
                        step
                    )
        
        # Segmentation statistics (scalars)
        if 'total_segments' in temporal_stats_dict:
            tb_writer.add_scalar(f"{prefix}/total_segments", temporal_stats_dict['total_segments'], step)
        if 'avg_segments_per_video' in temporal_stats_dict:
            tb_writer.add_scalar(f"{prefix}/avg_segments_per_video", temporal_stats_dict['avg_segments_per_video'], step)
        
        tb_writer.flush()
        
    except Exception as e:
        logger.debug(f"Failed to log temporal stats: {e}")


def log_bucket_distribution_to_tensorboard(
    accelerator,
    train_dataset_group,
    prefix: str = "dataset/buckets",
    step: int = 0,
) -> None:
    """
    Log bucket distribution statistics to TensorBoard.
    
    Analyzes how items are distributed across resolution buckets.
    
    Args:
        accelerator: Accelerator instance
        train_dataset_group: Training dataset group
        prefix: Prefix for TensorBoard tags
        step: Step number for logging
    """
    if not accelerator.is_main_process:
        return
    
    if not hasattr(accelerator, 'trackers') or not accelerator.trackers:
        return
    
    try:
        tb_writer = None
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tb_writer = tracker.writer
                break
        
        if tb_writer is None:
            return
        
        # Collect bucket statistics
        bucket_sizes = []
        bucket_areas = []
        bucket_aspect_ratios = []
        bucket_item_counts = []
        
        for dataset in train_dataset_group.datasets:
            resolution = getattr(dataset, 'resolution', None)
            if resolution:
                width, height = resolution[0], resolution[1]
                area = width * height
                aspect_ratio = width / height if height > 0 else 1.0
                
                bucket_areas.append(area)
                bucket_aspect_ratios.append(aspect_ratio)
                bucket_sizes.append(width * height)  # Total pixels
                
                # Get item count for this bucket
                try:
                    item_count = len(dataset)
                    bucket_item_counts.append(item_count)
                except:
                    pass
        
        # Log bucket distribution histograms
        if len(bucket_areas) > 0:
            tb_writer.add_histogram(
                f"{prefix}/distribution/bucket_areas",
                np.array(bucket_areas),
                step
            )
        
        if len(bucket_aspect_ratios) > 0:
            tb_writer.add_histogram(
                f"{prefix}/distribution/bucket_aspect_ratios",
                np.array(bucket_aspect_ratios),
                step
            )
        
        if len(bucket_item_counts) > 0:
            tb_writer.add_histogram(
                f"{prefix}/distribution/items_per_bucket",
                np.array(bucket_item_counts),
                step
            )
            
            # Log aggregate statistics
            tb_writer.add_scalar(f"{prefix}/total_buckets", len(bucket_item_counts), step)
            tb_writer.add_scalar(f"{prefix}/mean_items_per_bucket", np.mean(bucket_item_counts), step)
            tb_writer.add_scalar(f"{prefix}/max_items_per_bucket", np.max(bucket_item_counts), step)
            tb_writer.add_scalar(f"{prefix}/min_items_per_bucket", np.min(bucket_item_counts), step)
            tb_writer.add_scalar(f"{prefix}/std_items_per_bucket", np.std(bucket_item_counts), step)
        
        tb_writer.flush()
        
    except Exception as e:
        logger.debug(f"Failed to log bucket distribution: {e}")
