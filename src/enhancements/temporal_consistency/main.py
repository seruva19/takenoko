"""
Temporal Consistency Enhancement for Takenoko.
Applies frequency-domain temporal consistency analysis to video training samples,
encouraging the model to learn temporally coherent representations.
"""

import torch
from .config import TemporalConsistencyConfig
from .temporal_enhancer import TemporalConsistencyEnhancer
from .frequency_analyzer import FrequencyAnalyzer
from .utils import (
    validate_video_tensor,
    TemporalConsistencyMonitor,
)
from .training_integration import (
    TemporalConsistencyTrainingIntegration,
    enhance_loss_with_temporal_consistency,
)


def create_enhancer_from_args(args, device=None) -> TemporalConsistencyEnhancer:
    """Create TemporalConsistencyEnhancer from command line arguments.

    Args:
        args: Command line arguments or config namespace
        device: Device to run operations on

    Returns:
        Configured TemporalConsistencyEnhancer instance
    """
    config = TemporalConsistencyConfig.from_args(args)
    return TemporalConsistencyEnhancer(config, device)


def initialize_enhancer_from_args(args, device=None):
    """Initialize enhancer using the encapsulated class method.

    This is preferred over create_enhancer_from_args as it handles all logic internally.
    """
    return TemporalConsistencyEnhancer.initialize_from_args(args, device)


def is_temporal_consistency_available() -> bool:
    """Check if temporal consistency features are available."""
    try:
        import torch
        import torch.fft

        return hasattr(torch.fft, "fft2") and hasattr(torch.fft, "ifft2")
    except ImportError:
        return False


def quick_video_analysis(video_tensor: torch.Tensor, threshold: float = 0.25) -> dict:
    """Quick temporal consistency analysis for debugging.

    Args:
        video_tensor: Video tensor to analyze
        threshold: Low-frequency threshold

    Returns:
        Dictionary with quick analysis results
    """
    if not is_temporal_consistency_available():
        return {"error": "torch.fft not available"}

    if not validate_video_tensor(video_tensor):
        return {"error": "Invalid video tensor"}

    try:
        analyzer = FrequencyAnalyzer()
        analysis = analyzer.analyze_temporal_consistency(video_tensor)

        # Add threshold-specific analysis
        if len(video_tensor.shape) == 5:
            sample_frame = video_tensor[0, 0]
        else:
            sample_frame = video_tensor[0]

        low_freq, mid_freq, high_freq = analyzer.decompose_frequency_components(
            sample_frame, low_threshold=threshold
        )

        import torch

        analysis["frequency_decomposition"] = {
            "low_freq_energy": torch.sum(low_freq**2).item(),
            "mid_freq_energy": torch.sum(mid_freq**2).item(),
            "high_freq_energy": torch.sum(high_freq**2).item(),
            "threshold_used": threshold,
        }

        return analysis

    except Exception as e:
        return {"error": str(e)}


def create_temporal_consistency_integration(args, device=None):
    """Create temporal consistency integration for training core.

    This function creates and returns the training integration instance
    that training_core.py expects.

    Args:
        args: Command line arguments or config namespace
        device: Device to run operations on

    Returns:
        TemporalConsistencyTrainingIntegration instance if enabled, None otherwise
    """
    try:
        config = TemporalConsistencyConfig.from_args(args)
        if config.is_enabled():
            return TemporalConsistencyTrainingIntegration.initialize_and_create(
                args, device
            )
        else:
            return None
    except Exception as e:
        import logging
        from common.logger import get_logger

        logger = get_logger(__name__)
        logger.warning(f"Failed to create temporal consistency integration: {e}")
        return None
