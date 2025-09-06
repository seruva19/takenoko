import torch
import torch.fft
from typing import Tuple, Optional, Dict, Any
from functools import lru_cache

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class FrequencyAnalyzer:
    """FFT-based frequency domain operations for temporal consistency analysis."""

    def __init__(self, device: torch.device = None, enable_caching: bool = True):
        self.device = device or torch.device("cpu")
        self.enable_caching = enable_caching
        self.mask_cache = {}

    def _create_frequency_mask(
        self,
        shape: Tuple[int, int],
        threshold: float,
        is_low_pass: bool = True,
        preserve_dc: bool = True,
    ) -> torch.Tensor:
        """Create frequency domain mask with caching."""
        cache_key = (
            (shape, threshold, is_low_pass, preserve_dc, self.device)
            if self.enable_caching
            else None
        )

        if cache_key and cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        h, w = shape

        # Create frequency coordinates centered at DC
        freq_h = torch.fft.fftfreq(h, device=self.device)
        freq_w = torch.fft.fftfreq(w, device=self.device)

        # Create 2D frequency grid
        freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")

        # Calculate magnitude (distance from DC)
        freq_magnitude = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)

        # Normalize frequency magnitude to [0, 1]
        max_freq = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=self.device))
        freq_magnitude = freq_magnitude / max_freq

        # Create mask based on threshold
        if is_low_pass:
            mask = (freq_magnitude <= threshold).float()
        else:
            mask = (freq_magnitude > threshold).float()

        # Preserve DC component only for low-pass filters (DC is lowest frequency)
        if preserve_dc and is_low_pass:
            mask[0, 0] = 1.0

        # Cache the mask
        if cache_key:
            self.mask_cache[cache_key] = mask

        return mask

    def decompose_frequency_components(
        self,
        tensor: torch.Tensor,
        low_threshold: float = 0.25,
        high_threshold: float = 0.7,
        preserve_dc: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose tensor into low, mid, and high frequency components.

        Args:
            tensor: Input tensor [..., H, W]
            low_threshold: Threshold for low-pass filter
            high_threshold: Threshold for high-pass filter
            preserve_dc: Whether to preserve DC component

        Returns:
            Tuple of (low_freq, mid_freq, high_freq) components
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # Convert to float32 for FFT precision
        tensor_f32 = tensor.float()

        # Apply 2D FFT to last two dimensions
        fft_tensor = torch.fft.fft2(tensor_f32, dim=(-2, -1))

        # Create masks for different frequency bands
        spatial_shape = original_shape[-2:]

        low_mask = self._create_frequency_mask(
            spatial_shape, low_threshold, is_low_pass=True, preserve_dc=preserve_dc
        ).to(tensor.device)

        high_mask = self._create_frequency_mask(
            spatial_shape, high_threshold, is_low_pass=False, preserve_dc=preserve_dc
        ).to(tensor.device)

        # Mid frequencies are between low and high thresholds
        # Ensure non-negative values by proper threshold ordering
        if high_threshold <= low_threshold:
            logger.warning(
                f"High threshold ({high_threshold}) <= low threshold ({low_threshold}), using mid_mask=0"
            )
            mid_mask = torch.zeros_like(low_mask)
        else:
            mid_mask = 1.0 - low_mask - high_mask
            mid_mask = torch.clamp(mid_mask, min=0.0)  # Ensure non-negative

        # Apply masks and inverse FFT
        low_fft = fft_tensor * low_mask
        mid_fft = fft_tensor * mid_mask
        high_fft = fft_tensor * high_mask

        low_freq = torch.fft.ifft2(low_fft, dim=(-2, -1)).real.to(original_dtype)
        mid_freq = torch.fft.ifft2(mid_fft, dim=(-2, -1)).real.to(original_dtype)
        high_freq = torch.fft.ifft2(high_fft, dim=(-2, -1)).real.to(original_dtype)

        return low_freq, mid_freq, high_freq

    def extract_structural_component(
        self, tensor: torch.Tensor, threshold: float = 0.25
    ) -> torch.Tensor:
        """Extract low-frequency structural component (inspired by Ouroboros-Diffusion).

        This is the core operation from the paper adapted for training.
        """
        low_freq, _, _ = self.decompose_frequency_components(
            tensor, low_threshold=threshold, preserve_dc=True
        )
        return low_freq

    def extract_motion_component(
        self, tensor: torch.Tensor, threshold: float = 0.7
    ) -> torch.Tensor:
        """Extract high-frequency motion/detail component."""
        _, _, high_freq = self.decompose_frequency_components(
            tensor, high_threshold=threshold, preserve_dc=True
        )
        return high_freq

    def compute_temporal_frequency_difference(
        self, frame1: torch.Tensor, frame2: torch.Tensor, component: str = "low"
    ) -> torch.Tensor:
        """Compute frequency-domain difference between frames.

        Args:
            frame1: First frame tensor
            frame2: Second frame tensor
            component: "low", "mid", "high" or "all"

        Returns:
            Frequency component difference
        """
        if component == "low":
            comp1 = self.extract_structural_component(frame1)
            comp2 = self.extract_structural_component(frame2)
        elif component == "high":
            comp1 = self.extract_motion_component(frame1)
            comp2 = self.extract_motion_component(frame2)
        elif component == "all":
            comp1, comp2 = frame1, frame2
        else:  # mid
            _, comp1, _ = self.decompose_frequency_components(frame1)
            _, comp2, _ = self.decompose_frequency_components(frame2)

        return comp2 - comp1

    def analyze_temporal_consistency(
        self, video_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze temporal consistency of a video tensor for debugging.

        Args:
            video_tensor: Video tensor [B, F, C, H, W] or [F, C, H, W]

        Returns:
            Dictionary with temporal consistency metrics
        """
        if len(video_tensor.shape) == 4:
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension

        batch_size, num_frames, channels, height, width = video_tensor.shape

        if num_frames < 2:
            return {"error": "Need at least 2 frames for temporal analysis"}

        # Compute frame-to-frame consistency in different frequency bands
        low_consistency = []
        mid_consistency = []
        high_consistency = []

        for b in range(batch_size):
            for f in range(num_frames - 1):
                frame1 = video_tensor[b, f]
                frame2 = video_tensor[b, f + 1]

                # Decompose into frequency components
                low1, mid1, high1 = self.decompose_frequency_components(frame1)
                low2, mid2, high2 = self.decompose_frequency_components(frame2)

                # Compute similarities (higher = more consistent)
                low_sim = 1.0 - torch.mean((low1 - low2) ** 2).item()
                mid_sim = 1.0 - torch.mean((mid1 - mid2) ** 2).item()
                high_sim = 1.0 - torch.mean((high1 - high2) ** 2).item()

                low_consistency.append(low_sim)
                mid_consistency.append(mid_sim)
                high_consistency.append(high_sim)

        return {
            "avg_low_freq_consistency": sum(low_consistency) / len(low_consistency),
            "avg_mid_freq_consistency": sum(mid_consistency) / len(mid_consistency),
            "avg_high_freq_consistency": sum(high_consistency) / len(high_consistency),
            "num_frame_pairs": len(low_consistency),
            "temporal_smoothness_score": sum(low_consistency)
            / len(low_consistency),  # Low-freq is most important for smoothness
        }

    def clear_cache(self):
        """Clear frequency mask cache."""
        self.mask_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "cache_enabled": self.enable_caching,
            "cached_masks": len(self.mask_cache),
            "cache_keys": (
                list(self.mask_cache.keys())
                if len(self.mask_cache) < 10
                else "too_many_to_display"
            ),
        }
