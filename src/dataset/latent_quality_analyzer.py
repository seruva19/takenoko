"""Comprehensive latent quality analysis for Takenoko training.

This module provides complete latent quality analysis for both image and video datasets:
- Basic latent statistics (mean, std deviation)
- Video-specific temporal analysis (consistency, scene transitions, motion)
- TensorBoard integration and visualization
- Dataset type-aware analysis (ImageDataset vs VideoDataset)
- Orchestration for trainer integration
"""

import logging
import math
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch

try:
    from common.logger import get_logger
except ImportError:
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

logger = get_logger(__name__, level=logging.INFO)


class LatentQualityResult:
    """Results from basic latent quality analysis."""

    def __init__(self, latent_name: str, mean: float, std: float):
        self.latent_name = latent_name
        self.mean = mean
        self.std = std
        self.warning_score = 0.0

    @property
    def is_problematic(self) -> bool:
        """Check if latent has quality issues."""
        return self.warning_score > 0.0


class VideoLatentQualityResult:
    """Results from video-specific latent quality analysis."""

    def __init__(self, video_name: str, frame_count: int):
        self.video_name = video_name
        self.frame_count = frame_count

        # Per-frame statistics
        self.frame_means: List[float] = []
        self.frame_stds: List[float] = []

        # Temporal consistency metrics
        self.temporal_consistency_score: float = 0.0
        self.mean_drift: float = 0.0
        self.std_drift: float = 0.0

        # Scene transition detection
        self.scene_transitions: List[int] = []
        self.transition_score: float = 0.0

        # Motion analysis
        self.motion_intensity: float = 0.0
        self.motion_smoothness: float = 0.0

        # Overall quality
        self.video_quality_score: float = 0.0

    @property
    def is_problematic(self) -> bool:
        """Check if video has quality issues."""
        return (self.temporal_consistency_score > 2.0 or
                self.transition_score > 3.0 or
                self.motion_smoothness < 0.3)


class LatentQualityAnalyzer:
    """Comprehensive latent quality analyzer for both image and video data."""

    def __init__(self, mean_threshold: float = 0.16, std_threshold: float = 1.35):
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.std_threshold_min = 1.0 / std_threshold

    def analyze_latent_file(self, latent_path: str, latent_name: str = None) -> Optional[LatentQualityResult]:
        """Analyze a single latent file for basic quality metrics."""
        if latent_name is None:
            latent_name = os.path.basename(latent_path)

        try:
            latent_data = None

            if latent_path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(latent_path, framework="pt") as f:
                        for key in f.keys():
                            latent = f.get_tensor(key)
                            if hasattr(latent, 'numpy'):
                                if latent.dtype == torch.bfloat16:
                                    latent = latent.float()
                                latent_data = latent.numpy().astype(np.float32)
                            elif hasattr(latent, 'cpu'):
                                if latent.dtype == torch.bfloat16:
                                    latent = latent.float()
                                latent_data = latent.cpu().numpy().astype(np.float32)
                            else:
                                latent_data = np.array(latent, dtype=np.float32)
                            break
                except ImportError:
                    logger.warning(f"safetensors not available, cannot analyze {latent_path}")
                    return None
            else:
                with np.load(latent_path) as latents:
                    for key in latents.keys():
                        latent_data = latents[key].astype(np.float32)
                        break

            if latent_data is None:
                logger.warning(f"No latent data found in {latent_path}")
                return None

            # Calculate statistics
            mean = float(np.mean(latent_data))
            std = float(np.std(latent_data))

            result = LatentQualityResult(latent_name, mean, std)
            result.warning_score = self._calculate_warning_score(mean, std)
            return result

        except Exception as e:
            logger.warning(f"Failed to analyze latent {latent_path}: {e}")
            return None

    def analyze_video_latent(self, latent_path: str, video_name: str = None) -> Optional[VideoLatentQualityResult]:
        """Analyze a video latent file with temporal awareness."""
        if video_name is None:
            video_name = os.path.basename(latent_path)

        try:
            video_latent = None

            if latent_path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(latent_path, framework="pt") as f:
                        for key in f.keys():
                            data = f.get_tensor(key)
                            if hasattr(data, 'numpy'):
                                data = data.numpy()
                            elif hasattr(data, 'cpu'):
                                data = data.cpu().numpy()
                            else:
                                data = np.array(data)
                            if data.ndim >= 4:  # Likely video: (frames, channels, height, width)
                                video_latent = data
                                break
                except ImportError:
                    logger.warning(f"safetensors not available, cannot analyze {latent_path}")
                    return None
            else:
                with np.load(latent_path) as latents:
                    for key in latents.keys():
                        data = latents[key]
                        if data.ndim >= 4:  # Likely video: (frames, channels, height, width)
                            video_latent = data
                            break

            if video_latent is None:
                logger.warning(f"No video latent data found in {latent_path}")
                return None

            return self._analyze_video_tensor(video_latent, video_name)

        except Exception as e:
            logger.warning(f"Failed to analyze video latent {latent_path}: {e}")
            return None

    def _analyze_video_tensor(self, video_latent: np.ndarray, video_name: str) -> VideoLatentQualityResult:
        """Analyze a video latent tensor."""
        if video_latent.ndim != 4:
            logger.warning(f"Unexpected video latent shape: {video_latent.shape}")
            return VideoLatentQualityResult(video_name, 0)

        frame_count = video_latent.shape[0]
        result = VideoLatentQualityResult(video_name, frame_count)

        # Analyze each frame
        for frame_idx in range(frame_count):
            frame = video_latent[frame_idx]
            frame_mean = float(np.mean(frame))
            frame_std = float(np.std(frame))

            result.frame_means.append(frame_mean)
            result.frame_stds.append(frame_std)

        # Calculate temporal consistency metrics
        result.temporal_consistency_score = self._calculate_temporal_consistency(result)
        result.mean_drift, result.std_drift = self._calculate_drift(result)

        # Detect scene transitions
        result.scene_transitions = self._detect_scene_transitions(video_latent)
        result.transition_score = len(result.scene_transitions) / max(frame_count - 1, 1)

        # Analyze motion patterns
        result.motion_intensity, result.motion_smoothness = self._analyze_motion(video_latent)

        # Calculate overall video quality score
        result.video_quality_score = self._calculate_video_quality_score(result)

        return result

    def _calculate_warning_score(self, mean: float, std: float) -> float:
        """Calculate warning score for basic latent quality."""
        score = 0.0
        if abs(mean) > self.mean_threshold:
            score += abs(mean) * 5
        if std > self.std_threshold or std < self.std_threshold_min:
            score += max(abs(std - 1.0), abs(std - 1.0)) * 2
        return score

    def _calculate_temporal_consistency(self, result: VideoLatentQualityResult) -> float:
        """Calculate how consistent statistics are across frames."""
        if len(result.frame_means) < 2:
            return 0.0

        mean_variance = np.var(result.frame_means)
        std_variance = np.var(result.frame_stds)

        consistency_score = mean_variance * 10 + std_variance * 5
        return float(consistency_score)

    def _calculate_drift(self, result: VideoLatentQualityResult) -> Tuple[float, float]:
        """Calculate statistical drift across the video."""
        if len(result.frame_means) < 2:
            return 0.0, 0.0

        frames = np.arange(len(result.frame_means))
        mean_drift = abs(np.polyfit(frames, result.frame_means, 1)[0])
        std_drift = abs(np.polyfit(frames, result.frame_stds, 1)[0])

        return float(mean_drift), float(std_drift)

    def _detect_scene_transitions(self, video_latent: np.ndarray) -> List[int]:
        """Detect scene transitions by analyzing frame-to-frame changes."""
        transitions = []
        if video_latent.shape[0] < 2:
            return transitions

        for i in range(1, video_latent.shape[0]):
            frame_diff = np.mean(np.abs(video_latent[i] - video_latent[i-1]))
            transition_threshold = 0.5
            if frame_diff > transition_threshold:
                transitions.append(i)

        return transitions

    def _analyze_motion(self, video_latent: np.ndarray) -> Tuple[float, float]:
        """Analyze motion patterns in the video."""
        if video_latent.shape[0] < 3:
            return 0.0, 1.0

        frame_diffs = []
        for i in range(1, video_latent.shape[0]):
            diff = np.mean(np.abs(video_latent[i] - video_latent[i-1]))
            frame_diffs.append(diff)

        motion_intensity = float(np.mean(frame_diffs))
        motion_variance = np.var(frame_diffs)
        motion_smoothness = 1.0 / (1.0 + motion_variance * 10)

        return motion_intensity, float(motion_smoothness)

    def _calculate_video_quality_score(self, result: VideoLatentQualityResult) -> float:
        """Calculate overall video quality score (higher = more problematic)."""
        score = 0.0
        score += result.temporal_consistency_score * 0.3
        score += (result.mean_drift + result.std_drift) * 100

        if result.transition_score > 0.2:
            score += result.transition_score * 2

        if result.motion_smoothness < 0.5:
            score += (0.5 - result.motion_smoothness) * 3

        return score

    def create_visualization(self, results: List[LatentQualityResult]) -> Optional[torch.Tensor]:
        """Create quality visualization plot."""
        try:
            import matplotlib.pyplot as plt

            if not results:
                return None

            means = [r.mean for r in results]
            stds = [r.std for r in results]
            scores = [r.warning_score for r in results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Scatter plot
            scatter = ax1.scatter(means, stds, c=scores, cmap='viridis', alpha=0.7)
            ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal Std=1.0')
            ax1.axhline(y=self.std_threshold, color='red', linestyle=':', alpha=0.7)
            ax1.axhline(y=self.std_threshold_min, color='red', linestyle=':', alpha=0.7)
            ax1.axvline(x=0.0, color='green', linestyle='--', alpha=0.7, label='Ideal Mean=0.0')
            ax1.axvline(x=self.mean_threshold, color='red', linestyle=':', alpha=0.7)
            ax1.axvline(x=-self.mean_threshold, color='red', linestyle=':', alpha=0.7)
            ax1.set_xlabel('Mean')
            ax1.set_ylabel('Standard Deviation')
            ax1.set_title('Latent Statistics Overview')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Warning Score')

            # Histogram
            ax2.hist(means, bins=30, alpha=0.6, label='Means', color='blue')
            ax2.axvline(x=0.0, color='green', linestyle='--', alpha=0.7, label='Ideal Mean=0.0')
            ax2.axvline(x=self.mean_threshold, color='red', linestyle=':', alpha=0.7)
            ax2.axvline(x=-self.mean_threshold, color='red', linestyle=':', alpha=0.7)
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')
            ax2.set_title('Mean Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            fig.canvas.draw()
            # Use buffer_rgba() instead of deprecated tostring_rgb()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            buf = buf[:, :, :3]
            plt.close(fig)

            return torch.from_numpy(buf.transpose(2, 0, 1))

        except Exception as e:
            logger.warning(f"Failed to create quality visualization: {e}")
            return None

    def log_to_tensorboard(self, results: List[LatentQualityResult], writer: Any, global_step: int, tag_prefix: str = "latent_quality") -> None:
        """Log basic latent quality metrics to TensorBoard."""
        if not results:
            return

        means = [r.mean for r in results]
        stds = [r.std for r in results]
        scores = [r.warning_score for r in results]

        writer.add_scalar(f'{tag_prefix}/mean_avg', np.mean(means), global_step)
        writer.add_scalar(f'{tag_prefix}/mean_std', np.std(means), global_step)
        writer.add_scalar(f'{tag_prefix}/std_avg', np.mean(stds), global_step)
        writer.add_scalar(f'{tag_prefix}/std_std', np.std(stds), global_step)
        writer.add_scalar(f'{tag_prefix}/warning_score_avg', np.mean(scores), global_step)
        writer.add_scalar(f'{tag_prefix}/problematic_count', sum(1 for r in results if r.is_problematic), global_step)

        writer.add_histogram(f'{tag_prefix}/means_distribution', np.array(means), global_step)
        writer.add_histogram(f'{tag_prefix}/stds_distribution', np.array(stds), global_step)

        # Add visualization
        visualization = self.create_visualization(results)
        if visualization is not None:
            writer.add_image(f'{tag_prefix}/quality_overview', visualization, global_step)

        # Add ALL problematic files list to TensorBoard with full details
        problematic = [r for r in results if r.is_problematic]
        if problematic:
            # Sort by warning score (most problematic first)
            sorted_problematic = sorted(problematic, key=lambda x: x.warning_score, reverse=True)

            problematic_text = f"All Problematic Image Latents ({len(problematic)}/{len(results)}) - Sorted by Severity:\n\n"
            for i, result in enumerate(sorted_problematic):
                recommendation = self._get_image_recommendation(result.mean, result.std)
                severity = "Critical" if result.warning_score > 2.0 else "Warning"
                problematic_text += (f"{i+1}. {severity} - {result.latent_name}\n"
                                   f"   Score: {result.warning_score:.3f} | "
                                   f"Mean: {result.mean:.4f} | Std: {result.std:.4f}\n"
                                   f"   Recommendation: {recommendation}\n\n")

            # Use step 1 if global_step is 0 (TensorBoard sometimes ignores step 0)
            step_to_use = max(1, global_step)
            writer.add_text(f'{tag_prefix}/all_problematic_files', problematic_text, step_to_use)
            writer.add_scalar(f'{tag_prefix}/problematic_files_count', len(problematic), step_to_use)

    def log_video_quality_to_tensorboard(self, results: List[VideoLatentQualityResult], writer: Any, global_step: int, tag_prefix: str = "video_quality") -> None:
        """Log video-specific quality metrics to TensorBoard."""
        if not results:
            return

        consistency_scores = [r.temporal_consistency_score for r in results]
        motion_intensities = [r.motion_intensity for r in results]
        motion_smoothness_scores = [r.motion_smoothness for r in results]
        transition_scores = [r.transition_score for r in results]

        writer.add_scalar(f'{tag_prefix}/temporal_consistency_mean', np.mean(consistency_scores), global_step)
        writer.add_scalar(f'{tag_prefix}/temporal_consistency_max', np.max(consistency_scores), global_step)
        writer.add_scalar(f'{tag_prefix}/motion_intensity_mean', np.mean(motion_intensities), global_step)
        writer.add_scalar(f'{tag_prefix}/motion_smoothness_mean', np.mean(motion_smoothness_scores), global_step)
        writer.add_scalar(f'{tag_prefix}/scene_transition_rate', np.mean(transition_scores), global_step)
        writer.add_scalar(f'{tag_prefix}/problematic_videos_count', sum(1 for r in results if r.is_problematic), global_step)
        writer.add_scalar(f'{tag_prefix}/problematic_videos_ratio', sum(1 for r in results if r.is_problematic) / len(results), global_step)

        writer.add_histogram(f'{tag_prefix}/temporal_consistency_distribution', np.array(consistency_scores), global_step)
        writer.add_histogram(f'{tag_prefix}/motion_smoothness_distribution', np.array(motion_smoothness_scores), global_step)

        # Text summary of ALL problematic videos with full details
        problematic_videos = [r for r in results if r.is_problematic]
        if problematic_videos:
            # Sort by video quality score (most problematic first)
            sorted_problematic = sorted(problematic_videos, key=lambda x: x.video_quality_score, reverse=True)

            detailed_text = f"All Problematic Videos ({len(problematic_videos)}/{len(results)}) - Sorted by Severity:\n\n"
            for i, video in enumerate(sorted_problematic):
                severity = "🔴 Critical" if video.video_quality_score > 3.0 else "🟡 Warning"
                recommendation = self._get_video_recommendation(video)

                detailed_text += (f"{i+1}. {severity} - {video.video_name} ({video.frame_count} frames)\n"
                                f"   Overall Score: {video.video_quality_score:.3f}\n"
                                f"   Temporal Consistency: {video.temporal_consistency_score:.3f} | "
                                f"Motion Smoothness: {video.motion_smoothness:.3f}\n"
                                f"   Scene Transitions: {video.transition_score:.3f} | "
                                f"Motion Intensity: {video.motion_intensity:.3f}\n"
                                f"   Mean Drift: {video.mean_drift:.4f} | Std Drift: {video.std_drift:.4f}\n"
                                f"   Recommendation: {recommendation}\n\n")
            writer.add_text(f'{tag_prefix}/all_problematic_videos', detailed_text, global_step)

    def _report_image_results(self, results: List[LatentQualityResult]) -> None:
        """Report basic latent quality results."""
        if not results:
            return

        problematic = [r for r in results if r.is_problematic]
        total = len(results)

        if not problematic:
            logger.info(f"✅ All {total} image latents passed quality checks")
        else:
            logger.info(f"📊 Image latents: {len(problematic)}/{total} problematic (details in TensorBoard)")

    def _report_video_results(self, results: List[VideoLatentQualityResult]) -> None:
        """Report video-specific quality results."""
        if not results:
            return

        problematic = [r for r in results if r.is_problematic]
        total = len(results)

        if not problematic:
            logger.info(f"✅ All {total} video latents have good temporal consistency")
        else:
            logger.info(f"🎬 Video latents: {len(problematic)}/{total} problematic (details in TensorBoard)")

    def _get_image_recommendation(self, mean: float, std: float) -> str:
        """Get one-line recommendation for problematic image latent."""
        if abs(mean) > self.mean_threshold and (std > self.std_threshold or std < self.std_threshold_min):
            return f"Adjust VAE scaling and normalization (mean≈0, std≈1)"
        elif abs(mean) > self.mean_threshold:
            return f"Center latent distribution (current mean {mean:.3f})"
        elif std > self.std_threshold:
            return f"Reduce latent variance (std {std:.3f} too high)"
        elif std < self.std_threshold_min:
            return f"Increase latent variance (std {std:.3f} too low)"
        return "Check VAE encoder configuration"

    def _get_video_recommendation(self, video: VideoLatentQualityResult) -> str:
        """Get one-line recommendation for problematic video."""
        if video.temporal_consistency_score > 2.0:
            return f"Poor temporal consistency ({video.temporal_consistency_score:.2f}) - check frame transitions"
        elif video.transition_score > 3.0:
            return f"Too many scene transitions ({video.transition_score:.2f}) - reduce rapid cuts"
        elif video.motion_smoothness < 0.3:
            return f"Jerky motion ({video.motion_smoothness:.2f}) - smooth motion patterns needed"
        return f"Multiple temporal issues (score: {video.video_quality_score:.2f})"


# Orchestrator functionality
class LatentQualityOrchestrator:
    """Orchestrates latent quality analysis for training sessions."""

    @staticmethod
    def should_run_analysis(args: Any) -> bool:
        """Check if latent quality analysis should be run."""
        return getattr(args, "latent_quality_analysis", False)

    @staticmethod
    def setup_and_validate_config(args: Any) -> bool:
        """Setup latent quality analysis with validation and configuration logging."""
        if not LatentQualityOrchestrator.should_run_analysis(args):
            return False

        if not LatentQualityOrchestrator._validate_configuration(args):
            return False

        LatentQualityOrchestrator._log_configuration(args)
        return True

    @staticmethod
    def analyze_dataset_on_load(args: Any, train_dataset_group: Any) -> None:
        """Run initial dataset quality analysis after loading."""
        if not LatentQualityOrchestrator.should_run_analysis(args):
            return

        try:
            analyze_training_latents_by_type(
                dataset_groups=[train_dataset_group],
                mean_threshold=getattr(args, "latent_mean_threshold", 0.16),
                std_threshold=getattr(args, "latent_std_threshold", 1.35),
                visualize_worst=getattr(args, "latent_quality_visualizer", False),
                tensorboard_writer=None,
                global_step=0
            )
        except Exception as e:
            logger.warning(f"Dataset latent quality analysis failed: {e}")

    @staticmethod
    def analyze_with_tensorboard(args: Any, train_dataset_group: Any, accelerator: Any, global_step: int = 0) -> None:
        """Run latent quality analysis with TensorBoard logging."""
        if not LatentQualityOrchestrator.should_run_analysis(args):
            return

        if not getattr(args, "latent_quality_tensorboard", True):
            logger.debug("TensorBoard logging disabled for latent quality")
            return

        if not (accelerator.is_main_process and len(accelerator.trackers) > 0):
            logger.warning("TensorBoard not available: either not main process or no trackers configured")
            return

        try:
            tensorboard_writer = LatentQualityOrchestrator._get_tensorboard_writer(accelerator)

            if tensorboard_writer:
                logger.info("🔍 Running latent quality analysis with TensorBoard logging...")
                analyze_training_latents_by_type(
                    dataset_groups=[train_dataset_group],
                    mean_threshold=getattr(args, "latent_mean_threshold", 0.16),
                    std_threshold=getattr(args, "latent_std_threshold", 1.35),
                    visualize_worst=False,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step
                )
                logger.info("✅ Latent quality TensorBoard integration completed")
            else:
                logger.debug("TensorBoard writer not found for latent quality logging")

        except Exception as e:
            logger.warning(f"❌ Latent quality TensorBoard integration failed: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")

    @staticmethod
    def _get_tensorboard_writer(accelerator: Any) -> Optional[Any]:
        """Extract TensorBoard writer from accelerator trackers."""
        try:
            for tracker in accelerator.trackers:
                if hasattr(tracker, 'writer'):
                    return tracker.writer
                elif hasattr(tracker, 'run') and hasattr(tracker.run, 'summary'):
                    return tracker
            return None
        except Exception as e:
            logger.debug(f"Failed to extract TensorBoard writer: {e}")
            return None

    @staticmethod
    def _log_configuration(args: Any) -> None:
        """Log latent quality analysis configuration."""
        logger.info("📊 Latent Quality Analysis Configuration:")
        logger.info(f"   Mean threshold: ±{getattr(args, 'latent_mean_threshold', 0.16)}")
        logger.info(f"   Std threshold: {getattr(args, 'latent_std_threshold', 1.35)} (min: {1.0/getattr(args, 'latent_std_threshold', 1.35):.3f})")
        logger.info(f"   Visualizer: {getattr(args, 'latent_quality_visualizer', False)}")
        logger.info(f"   TensorBoard logging: {getattr(args, 'latent_quality_tensorboard', True)}")

    @staticmethod
    def _validate_configuration(args: Any) -> bool:
        """Validate latent quality analysis configuration."""
        mean_threshold = getattr(args, "latent_mean_threshold", 0.16)
        std_threshold = getattr(args, "latent_std_threshold", 1.35)

        if mean_threshold <= 0:
            logger.error("latent_mean_threshold must be positive")
            return False

        if std_threshold < 1.0:
            logger.error("latent_std_threshold must be >= 1.0")
            return False

        # Validate feature dependencies
        if getattr(args, "latent_quality_visualizer", False):
            try:
                import cv2
            except ImportError:
                logger.warning("latent_quality_visualizer requires opencv-python, disabling visualization")
                args.latent_quality_visualizer = False

        if getattr(args, "latent_quality_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                logger.warning("TensorBoard not available, disabling TensorBoard logging")
                args.latent_quality_tensorboard = False

        return True


def analyze_training_latents_by_type(
    dataset_groups,
    mean_threshold: float = 0.16,
    std_threshold: float = 1.35,
    visualize_worst: bool = False,
    tensorboard_writer: Any = None,
    global_step: int = 0
) -> Tuple[List[LatentQualityResult], List[VideoLatentQualityResult]]:
    """
    Analyze training latents separated by dataset type (image vs video).

    Returns:
        Tuple of (image_results, video_results)
    """
    analyzer = LatentQualityAnalyzer(mean_threshold, std_threshold)

    # Separate files by dataset type
    image_files = []
    video_files = []

    for dataset_group in dataset_groups:
        group_name = getattr(dataset_group, 'name', f'group_{id(dataset_group)}')
        for dataset in dataset_group.datasets:
            # Determine if this is a video dataset
            is_video_dataset = hasattr(dataset, '__class__') and 'Video' in dataset.__class__.__name__

            # Collect latent files
            cache_dirs = []
            if hasattr(dataset, 'latents_cache') and dataset.latents_cache:
                cache_dirs.append(Path(dataset.latents_cache))
            if hasattr(dataset, 'cache_directory') and dataset.cache_directory:
                cache_dirs.append(Path(dataset.cache_directory))

            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    all_files = list(cache_dir.glob('*.npz')) + list(cache_dir.glob('*.safetensors'))
                    # Skip text encoder files
                    latent_files = [f for f in all_files if not ('_te.safetensors' in f.name or '_text_encoder' in f.name)]

                    for latent_file in latent_files:
                        file_name = f"{group_name}/{latent_file.name}"
                        if is_video_dataset:
                            video_files.append((str(latent_file), file_name))
                        else:
                            image_files.append((str(latent_file), file_name))

    logger.info(f"🔍 Analyzing {len(image_files)} image latents and {len(video_files)} video latents...")

    # Analyze image latents
    image_results = []
    if image_files:
        for latent_path, latent_name in image_files:
            result = analyzer.analyze_latent_file(latent_path, latent_name)
            if result:
                image_results.append(result)

        analyzer._report_image_results(image_results)

        if tensorboard_writer:
            analyzer.log_to_tensorboard(image_results, tensorboard_writer, global_step, "image_quality")

    # Analyze video latents
    video_results = []
    if video_files:
        for latent_path, video_name in video_files:
            result = analyzer.analyze_video_latent(latent_path, video_name)
            if result and result.frame_count > 0:
                video_results.append(result)

        analyzer._report_video_results(video_results)

        if tensorboard_writer:
            analyzer.log_video_quality_to_tensorboard(video_results, tensorboard_writer, global_step, "video_quality")

    return image_results, video_results


# Convenience functions for trainer integration
def setup_latent_quality_for_trainer(args: Any) -> bool:
    """Setup latent quality analysis for any trainer type."""
    return LatentQualityOrchestrator.setup_and_validate_config(args)


def run_dataset_analysis_for_trainer(args: Any, train_dataset_group: Any) -> None:
    """Run dataset analysis for any trainer type."""
    LatentQualityOrchestrator.analyze_dataset_on_load(args, train_dataset_group)


def run_tensorboard_analysis_for_trainer(
    args: Any,
    train_dataset_group: Any,
    accelerator: Any,
    global_step: int = 0
) -> None:
    """Run TensorBoard analysis for any trainer type."""
    LatentQualityOrchestrator.analyze_with_tensorboard(
        args, train_dataset_group, accelerator, global_step
    )