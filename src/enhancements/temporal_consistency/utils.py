import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time

import logging
from common.logger import get_logger
from enhancements.temporal_consistency.config import TemporalConsistencyConfig

logger = get_logger(__name__, level=logging.INFO)


def validate_video_tensor(
    tensor: torch.Tensor, min_frames: int = 2, max_frames: int = 32
) -> bool:
    """Validate that tensor is a proper video tensor for processing.

    Args:
        tensor: Input tensor to validate
        min_frames: Minimum number of frames required
        max_frames: Maximum frames to avoid OOM issues

    Returns:
        True if tensor is valid, False otherwise
    """
    # Check basic shape requirements
    if len(tensor.shape) not in [4, 5]:  # [F,C,H,W] or [B,F,C,H,W]
        return False

    if len(tensor.shape) == 4:
        num_frames = tensor.shape[0]
        h, w = tensor.shape[-2:]
    else:
        num_frames = tensor.shape[1]
        h, w = tensor.shape[-2:]

    # Check minimum frames
    if num_frames < min_frames:
        return False

    # Check maximum frames for memory safety
    if num_frames > max_frames:
        logger.warning(f"Too many frames ({num_frames} > {max_frames}), may cause OOM")
        # Don't return False, just warn

    # Check minimum spatial dimensions for FFT
    if h < 8 or w < 8:
        return False

    # Check maximum spatial dimensions for memory safety
    if h > 1024 or w > 1024:
        logger.warning(
            f"Large spatial dimensions ({h}x{w}), FFT operations may be slow"
        )

    # Check for reasonable aspect ratio
    if max(h, w) / min(h, w) > 8:
        logger.warning(f"Extreme aspect ratio detected: {h}x{w}")

    # Check for reasonable tensor values (not NaN/Inf)
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning("NaN or Inf values detected in video tensor")
        return False

    # Check tensor dtype
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        logger.warning(
            f"Unusual tensor dtype: {tensor.dtype}, converting to float32 for FFT"
        )

    return True


def compute_video_statistics(video_tensor: torch.Tensor) -> Dict[str, Any]:
    """Compute comprehensive statistics for video tensor analysis.

    Args:
        video_tensor: Video tensor to analyze

    Returns:
        Dictionary with video statistics
    """
    if not validate_video_tensor(video_tensor):
        return {"error": "Invalid video tensor"}

    # Handle batch dimension
    if len(video_tensor.shape) == 5:
        batch_size, num_frames, channels, height, width = video_tensor.shape
        # Flatten batch for analysis
        video_flat = video_tensor.view(-1, channels, height, width)  # [B*F, C, H, W]
    else:
        batch_size = 1
        num_frames, channels, height, width = video_tensor.shape
        video_flat = video_tensor

    # Basic statistics
    mean_val = torch.mean(video_tensor).item()
    std_val = torch.std(video_tensor).item()
    min_val = torch.min(video_tensor).item()
    max_val = torch.max(video_tensor).item()

    # Temporal variation (frame-to-frame differences)
    temporal_diffs = []
    if len(video_tensor.shape) == 5:
        for b in range(batch_size):
            for f in range(num_frames - 1):
                diff = torch.mean(
                    torch.abs(video_tensor[b, f + 1] - video_tensor[b, f])
                ).item()
                temporal_diffs.append(diff)
    else:
        for f in range(num_frames - 1):
            diff = torch.mean(torch.abs(video_tensor[f + 1] - video_tensor[f])).item()
            temporal_diffs.append(diff)

    # Spatial variation (within-frame complexity)
    spatial_vars = []
    for i in range(min(video_flat.shape[0], 10)):  # Sample up to 10 frames
        frame_var = torch.var(video_flat[i]).item()
        spatial_vars.append(frame_var)

    return {
        "shape": {
            "batch_size": batch_size,
            "num_frames": num_frames,
            "channels": channels,
            "height": height,
            "width": width,
            "total_pixels": batch_size * num_frames * channels * height * width,
        },
        "value_stats": {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val,
        },
        "temporal_stats": {
            "avg_frame_difference": np.mean(temporal_diffs) if temporal_diffs else 0.0,
            "max_frame_difference": max(temporal_diffs) if temporal_diffs else 0.0,
            "min_frame_difference": min(temporal_diffs) if temporal_diffs else 0.0,
            "temporal_stability": (
                1.0 - (np.mean(temporal_diffs) / max(std_val, 1e-8))
                if temporal_diffs
                else 1.0
            ),
        },
        "spatial_stats": {
            "avg_spatial_variance": np.mean(spatial_vars) if spatial_vars else 0.0,
            "max_spatial_variance": max(spatial_vars) if spatial_vars else 0.0,
            "spatial_complexity": (
                np.mean(spatial_vars) / max(mean_val**2, 1e-8) if spatial_vars else 0.0
            ),
        },
    }


def adaptive_threshold_selection(
    video_tensor: torch.Tensor,
    target_preservation_ratio: float = 0.75,
    threshold_range: Tuple[float, float] = (0.1, 0.4),
) -> float:
    """Adaptively select frequency threshold based on video content analysis.

    Args:
        video_tensor: Video tensor to analyze
        target_preservation_ratio: Target ratio of energy to preserve in low frequencies
        threshold_range: Valid range for threshold values

    Returns:
        Optimal threshold value
    """
    if not validate_video_tensor(video_tensor):
        return 0.25  # Default fallback

    # Sample a few frames for analysis to avoid computational overhead
    if len(video_tensor.shape) == 5:
        sample_frames = video_tensor[0, :3]  # First 3 frames of first batch
    else:
        sample_frames = video_tensor[:3]  # First 3 frames

    min_thresh, max_thresh = threshold_range

    try:
        # Binary search for optimal threshold
        low, high = min_thresh, max_thresh

        for _ in range(8):  # Max 8 iterations for efficiency
            mid_threshold = (low + high) / 2

            # Calculate energy preservation at this threshold
            total_energy = 0.0
            preserved_energy = 0.0

            for frame in sample_frames:
                # Convert to frequency domain
                fft_frame = torch.fft.fft2(frame.float())
                magnitude = torch.abs(fft_frame) ** 2
                total_energy += torch.sum(magnitude).item()

                # Create low-pass mask
                h, w = frame.shape[-2:]
                freq_h = torch.fft.fftfreq(h, device=frame.device)
                freq_w = torch.fft.fftfreq(w, device=frame.device)
                freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
                freq_magnitude = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)
                max_freq = torch.sqrt(
                    torch.tensor(0.5**2 + 0.5**2, device=frame.device)
                )
                freq_magnitude = freq_magnitude / max_freq

                low_pass_mask = (freq_magnitude <= mid_threshold).float()
                preserved_energy += torch.sum(magnitude * low_pass_mask).item()

            preservation_ratio = (
                preserved_energy / total_energy if total_energy > 0 else 0
            )

            if preservation_ratio < target_preservation_ratio:
                low = mid_threshold
            else:
                high = mid_threshold

            # Early stopping if close enough
            if abs(preservation_ratio - target_preservation_ratio) < 0.05:
                break

        optimal_threshold = (low + high) / 2
        return float(np.clip(optimal_threshold, min_thresh, max_thresh))

    except Exception as e:
        logger.warning(f"Failed to compute adaptive threshold: {e}")
        return 0.25  # Default fallback


def estimate_computational_cost(
    video_shape: Tuple[int, ...], config: "TemporalConsistencyConfig"
) -> Dict[str, Any]:
    """Estimate computational cost of temporal consistency enhancement.

    Args:
        video_shape: Shape of video tensor
        config: Temporal consistency configuration

    Returns:
        Dictionary with cost estimates
    """
    if len(video_shape) == 5:
        batch_size, num_frames, channels, height, width = video_shape
    else:
        batch_size = 1
        num_frames, channels, height, width = video_shape

    # Estimate FFT operations
    fft_ops_per_frame = height * width * channels * np.log2(height * width)
    total_fft_ops = fft_ops_per_frame * num_frames * batch_size

    # Estimate frame pair comparisons
    max_distance = min(config.freq_temporal_max_distance, num_frames - 1)
    frame_pairs = batch_size * max_distance

    # Estimate memory usage (rough approximation)
    base_memory_mb = (batch_size * num_frames * channels * height * width * 4) / (
        1024**2
    )  # 4 bytes per float32
    fft_memory_overhead = base_memory_mb * 2  # Complex numbers double memory
    cache_memory_mb = (
        (config.freq_temporal_cache_size * height * width * 4) / (1024**2)
        if config.freq_temporal_enable_caching
        else 0
    )

    estimated_memory_mb = base_memory_mb + fft_memory_overhead + cache_memory_mb

    # Estimate processing time (very rough)
    estimated_time_ms = (total_fft_ops / 1e6) + (frame_pairs * 0.1)  # Rough estimates

    return {
        "video_shape": video_shape,
        "estimated_fft_operations": int(total_fft_ops),
        "frame_pairs_to_process": frame_pairs,
        "estimated_memory_usage_mb": estimated_memory_mb,
        "estimated_processing_time_ms": estimated_time_ms,
        "computational_complexity": "O(N*F*H*W*log(H*W))",
        "memory_complexity": "O(N*F*H*W)",
        "recommendations": {
            "reduce_max_temporal_distance": max_distance > 4,
            "enable_caching": not config.freq_temporal_enable_caching
            and estimated_memory_mb < 1000,
            "reduce_batch_size": estimated_memory_mb > 2000,
        },
    }


class TemporalConsistencyMonitor:
    """Monitor and analyze temporal consistency enhancement performance."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.enhancement_history = []
        self.performance_history = []

    def log_enhancement(self, step: int, enhancement_info: Dict[str, Any]):
        """Log enhancement information."""
        log_entry = {"step": step, "timestamp": time.time(), **enhancement_info}

        self.enhancement_history.append(log_entry)

        # Maintain history size
        if len(self.enhancement_history) > self.history_size:
            self.enhancement_history.pop(0)

    def log_performance(self, step: int, performance_stats: Dict[str, Any]):
        """Log performance statistics."""
        log_entry = {"step": step, "timestamp": time.time(), **performance_stats}

        self.performance_history.append(log_entry)

        # Maintain history size
        if len(self.performance_history) > self.history_size:
            self.performance_history.pop(0)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        if not self.enhancement_history:
            return {"error": "No enhancement history available"}

        recent_enhancements = self.enhancement_history[-100:]  # Last 100

        # Calculate success rate
        successful_enhancements = [
            e
            for e in recent_enhancements
            if e.get("temporal_enhancement_applied", False)
        ]
        success_rate = (
            len(successful_enhancements) / len(recent_enhancements)
            if recent_enhancements
            else 0.0
        )

        # Calculate average enhancement impact
        enhancement_ratios = [
            e.get("enhancement_ratio", 0.0) for e in successful_enhancements
        ]
        avg_enhancement_ratio = (
            np.mean(enhancement_ratios) if enhancement_ratios else 0.0
        )

        # Calculate processing times
        processing_times = [
            e.get("processing_time_ms", 0.0) for e in successful_enhancements
        ]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0

        report = {
            "summary_period": f"{len(recent_enhancements)} recent enhancements",
            "success_rate": success_rate,
            "average_enhancement_ratio": avg_enhancement_ratio,
            "average_processing_time_ms": avg_processing_time,
            "total_enhancements_logged": len(self.enhancement_history),
            "recent_steps": (
                [e["step"] for e in recent_enhancements[-5:]]
                if recent_enhancements
                else []
            ),
        }

        if enhancement_ratios:
            report["enhancement_ratio_stats"] = {
                "min": min(enhancement_ratios),
                "max": max(enhancement_ratios),
                "std": np.std(enhancement_ratios),
            }

        if processing_times:
            report["processing_time_stats"] = {
                "min": min(processing_times),
                "max": max(processing_times),
                "std": np.std(processing_times),
            }

        return report

    def clear_history(self):
        """Clear all logged history."""
        self.enhancement_history.clear()
        self.performance_history.clear()
