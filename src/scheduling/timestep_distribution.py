## Based on: https://github.com/tdrussell/diffusion-pipe/commit/53f4fe7569eedce4ae3a877bbfcff7e784de1d53 (MIT)

"""Timestep distribution utilities for diffusion model training.

This module provides pre-computed timestep distribution functionality for more consistent
and reproducible training compared to on-the-fly random sampling.
"""

import argparse
import logging
from typing import Optional

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TimestepDistribution:
    """Handles pre-computed timestep distribution for consistent training.

    This class provides better timestep coverage and reproducibility compared to
    random on-the-fly sampling by pre-computing a quantized distribution.

    Benefits:
    - Guaranteed uniform timestep coverage (no random gaps)
    - Reproducible training (same distribution every run)
    - Better convergence due to consistent denoising coverage
    - Performance optimization (no repeated distribution calculations)
    - Support for curriculum learning via timestep range slicing
    """

    def __init__(self):
        self.distribution: Optional[torch.Tensor] = None
        self.is_initialized: bool = False
        # One-time usage log guard so we don't spam every step
        self.usage_logged: bool = False

    def initialize(self, args: argparse.Namespace) -> None:
        """Initialize the pre-computed timestep distribution.

        Args:
            args: Training arguments containing timestep sampling configuration
        """
        if self.is_initialized:
            return

        logger.info("Initializing pre-computed timestep distribution...")

        # Log configuration details
        self._log_configuration(args)

        # Create the base distribution
        self.distribution = self._create_distribution(args)
        initial_buckets = len(self.distribution)

        # Apply min/max timestep constraints if specified
        self._apply_timestep_constraints(args)

        self.is_initialized = True

        # Log final results with statistics
        self._log_initialization_results(args, initial_buckets)

    def _create_distribution(self, args: argparse.Namespace) -> torch.Tensor:
        """Create the base timestep distribution based on sampling method.

        Args:
            args: Training arguments containing timestep sampling configuration

        Returns:
            Pre-computed timestep distribution tensor
        """
        n_buckets = getattr(args, "precomputed_timestep_buckets", 10000)
        delta = 1 / n_buckets
        min_quantile = delta
        max_quantile = 1 - delta
        quantiles = torch.linspace(min_quantile, max_quantile, n_buckets)

        # Apply distribution transformation based on sampling method
        if args.timestep_sampling == "sigmoid":
            dist = torch.distributions.normal.Normal(0, 1)
            t = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t = torch.sigmoid(t * sigmoid_scale)
        elif args.timestep_sampling == "shift":
            dist = torch.distributions.normal.Normal(0, 1)
            t = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t = torch.sigmoid(t * sigmoid_scale)
            shift = getattr(args, "discrete_flow_shift", 1.0)
            t = (t * shift) / (1 + (shift - 1) * t)
        elif args.timestep_sampling == "logit_normal":
            dist = torch.distributions.normal.Normal(0, 1)
            t = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)
        elif args.timestep_sampling == "content":
            # Cubic sampling favoring earlier timesteps for content/structure training
            t = quantiles ** (1 / 3)  # Inverse of cubic transformation
        elif args.timestep_sampling == "style":
            # Cubic sampling favoring later timesteps for style training
            t = 1 - (1 - quantiles) ** (
                1 / 3
            )  # Inverse of inverse cubic transformation
        elif args.timestep_sampling == "content_style_blend":
            # Blend between content and style sampling
            blend_ratio = getattr(args, "content_style_blend_ratio", 0.5)
            content_t = quantiles ** (1 / 3)
            style_t = 1 - (1 - quantiles) ** (1 / 3)
            t = blend_ratio * content_t + (1 - blend_ratio) * style_t
        elif args.timestep_sampling == "flux_shift":
            # Precompute using a fixed base area; can be overridden via args.precomputed_midshift_area
            dist = torch.distributions.normal.Normal(0, 1)
            z = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t = torch.sigmoid(z * sigmoid_scale)
            base_area = getattr(args, "precomputed_midshift_area", None)
            if base_area is None:
                # Try to infer from height/width if available
                h = getattr(args, "latent_height", None)
                w = getattr(args, "latent_width", None)
                if h is None or w is None:
                    h = getattr(args, "height", None)
                    w = getattr(args, "width", None)
                    if h is not None and w is not None:
                        # Heuristic: latents are often 1/8 or 1/16 of pixel dims; use /16 then /2 in code path => /32
                        h = max(int(h) // 32, 1)
                        w = max(int(w) // 32, 1)
                if h is None or w is None:
                    base_area = 1024.0
                else:
                    base_area = float((h // 2) * (w // 2))
            # Flux mapping
            # y1=0.5 at x=256, y2=1.15 at x=4096
            m = (1.15 - 0.5) / (4096 - 256)
            b = 0.5 - m * 256
            mu = m * float(base_area) + b
            shift = torch.exp(torch.tensor(mu))
            t = (t * shift) / (1 + (shift - 1) * t)
        elif args.timestep_sampling == "qwen_shift":
            # Qwen variant with different mu mapping
            dist = torch.distributions.normal.Normal(0, 1)
            z = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t = torch.sigmoid(z * sigmoid_scale)
            base_area = getattr(args, "precomputed_midshift_area", None)
            if base_area is None:
                h = getattr(args, "latent_height", None)
                w = getattr(args, "latent_width", None)
                if h is None or w is None:
                    h = getattr(args, "height", None)
                    w = getattr(args, "width", None)
                    if h is not None and w is not None:
                        h = max(int(h) // 32, 1)
                        w = max(int(w) // 32, 1)
                if h is None or w is None:
                    base_area = 1024.0
                else:
                    base_area = float((h // 2) * (w // 2))
            # Qwen mapping: (256, 0.5) -> (8192, 0.9)
            m = (0.9 - 0.5) / (8192 - 256)
            b = 0.5 - m * 256
            mu = m * float(base_area) + b
            shift = torch.exp(torch.tensor(mu))
            t = (t * shift) / (1 + (shift - 1) * t)
        elif args.timestep_sampling == "logsnr":
            # Use Normal icdf with provided mean/std, then transform
            mean = getattr(args, "logit_mean", 0.0)
            std = getattr(args, "logit_std", 1.0)
            dist = torch.distributions.normal.Normal(mean, std)
            logsnr = dist.icdf(quantiles)
            t = torch.sigmoid(-logsnr / 2)
        elif args.timestep_sampling in ("qinglong_flux", "qinglong_qwen"):
            # Mixture: 80% mid_shift, 7.5% logsnr, 12.5% logsnr2
            n_buckets = getattr(args, "precomputed_timestep_buckets", 10000)
            mid_count = int(n_buckets * 0.80)
            l1_count = int(n_buckets * 0.075)
            l2_count = n_buckets - mid_count - l1_count

            # Build quantiles per component
            def linspace01(n: int) -> torch.Tensor:
                if n <= 0:
                    return torch.empty(0)
                delta = 1 / max(n, 1)
                return torch.linspace(delta, 1 - delta, n)

            # mid_shift component
            dist0 = torch.distributions.normal.Normal(0, 1)
            z0 = dist0.icdf(linspace01(mid_count))
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            t0 = torch.sigmoid(z0 * sigmoid_scale)
            base_area = getattr(args, "precomputed_midshift_area", None)
            if base_area is None:
                h = getattr(args, "latent_height", None)
                w = getattr(args, "latent_width", None)
                if h is None or w is None:
                    h = getattr(args, "height", None)
                    w = getattr(args, "width", None)
                    if h is not None and w is not None:
                        h = max(int(h) // 32, 1)
                        w = max(int(w) // 32, 1)
                if h is None or w is None:
                    base_area = 1024.0
                else:
                    base_area = float((h // 2) * (w // 2))

            if args.timestep_sampling == "qinglong_qwen":
                m = (0.9 - 0.5) / (8192 - 256)
                b = 0.5 - m * 256
            else:
                m = (1.15 - 0.5) / (4096 - 256)
                b = 0.5 - m * 256
            mu = m * float(base_area) + b
            shift = torch.exp(torch.tensor(mu))
            t0 = (t0 * shift) / (1 + (shift - 1) * t0)

            # logsnr component
            mean1 = getattr(args, "logit_mean", 0.0)
            std1 = getattr(args, "logit_std", 1.0)
            dist1 = torch.distributions.normal.Normal(mean1, std1)
            z1 = dist1.icdf(linspace01(l1_count))
            t1 = torch.sigmoid(-z1 / 2)

            # logsnr2 component (fixed mean=5.36, std=1.0)
            dist2 = torch.distributions.normal.Normal(5.36, 1.0)
            z2 = dist2.icdf(linspace01(l2_count))
            t2 = torch.sigmoid(-z2 / 2)

            # Combine and sort
            t = torch.cat([t0, t1, t2])
            t, _ = torch.sort(t)
        elif args.timestep_sampling == "bell_shaped":
            # Bell-Shaped: build density on x∈[0,1], compute CDF, then invert to get quantiles
            x = torch.linspace(0.0, 1.0, n_buckets)
            bell_std = float(getattr(args, "bell_std", 0.2))
            bell_center = float(getattr(args, "bell_center", 0.5))
            # Unnormalized density
            y = torch.exp(-0.5 * ((x - bell_center) / max(bell_std, 1e-6)) ** 2)
            # Avoid degenerate cases
            y = torch.clamp(y, min=1e-12)
            # CDF in [0,1]
            cdf = torch.cumsum(y, dim=0)
            cdf = cdf / cdf[-1]
            # Invert CDF at uniform quantiles (reuse 'quantiles' spacing from above scope)
            q = torch.linspace(1 / n_buckets, 1 - 1 / n_buckets, n_buckets)
            idx = torch.searchsorted(cdf, q, right=False).clamp(
                min=1, max=n_buckets - 1
            )
            cdf_lo = cdf[idx - 1]
            cdf_hi = cdf[idx]
            x_lo = x[idx - 1]
            x_hi = x[idx]
            # Linear interpolation between (cdf_lo,x_lo) and (cdf_hi,x_hi)
            denom = torch.clamp(cdf_hi - cdf_lo, min=1e-12)
            w = (q - cdf_lo) / denom
            t = x_lo + w * (x_hi - x_lo)
        elif args.timestep_sampling == "half_bell":
            # Half Bell-Shaped: density is bell on first half, flat on second; invert CDF
            x = torch.linspace(0.0, 1.0, n_buckets)
            bell_std = float(getattr(args, "bell_std", 0.2))
            y = torch.exp(-0.5 * ((x - 0.5) / max(bell_std, 1e-6)) ** 2)
            # Flatten second half to its max value
            mid_point = n_buckets // 2
            if mid_point < n_buckets:
                y[mid_point:] = y[:mid_point].max()
            # Normalize and build CDF
            y = torch.clamp(y, min=1e-12)
            cdf = torch.cumsum(y, dim=0)
            cdf = cdf / cdf[-1]
            # Invert at uniform quantiles
            q = torch.linspace(1 / n_buckets, 1 - 1 / n_buckets, n_buckets)
            idx = torch.searchsorted(cdf, q, right=False).clamp(
                min=1, max=n_buckets - 1
            )
            cdf_lo = cdf[idx - 1]
            cdf_hi = cdf[idx]
            x_lo = x[idx - 1]
            x_hi = x[idx]
            denom = torch.clamp(cdf_hi - cdf_lo, min=1e-12)
            w = (q - cdf_lo) / denom
            t = x_lo + w * (x_hi - x_lo)
        elif args.timestep_sampling == "lognorm_blend":
            # LogNormal Blend - combines lognormal distribution with linear sampling
            alpha = getattr(args, "lognorm_blend_alpha", 0.75)

            # LogNormal distribution for first portion
            lognormal = torch.distributions.LogNormal(loc=0, scale=0.333)
            t1_size = int(n_buckets * alpha)
            t1 = lognormal.icdf(quantiles[:t1_size])
            t1_max = t1.max()  # type: ignore
            if t1_max > 0:
                t1 = 1 - t1 / t1_max  # type: ignore # Scale to [0, 1]
            else:
                t1 = torch.zeros_like(t1)  # type: ignore

            # Linear distribution for remaining portion
            t2_size = n_buckets - t1_size
            t2 = torch.linspace(0, 1, t2_size)

            # Combine and sort
            t = torch.cat([t1, t2])
            t, _ = torch.sort(t)
        elif args.timestep_sampling == "lognorm_continuous_blend":
            # LogNormal Continuous Blend - every sample is a weighted combination of both distributions
            alpha = getattr(args, "lognorm_blend_alpha", 0.75)

            # LogNormal component
            dist = torch.distributions.LogNormal(0, 1)
            lognorm_samples = dist.icdf(quantiles)
            lognorm_samples = torch.clamp(lognorm_samples, 0, 1)  # type: ignore

            # Linear component
            linear_samples = quantiles

            # Blend every sample with weights alpha and (1-alpha)
            t = alpha * lognorm_samples + (1 - alpha) * linear_samples
        elif args.timestep_sampling == "enhanced_sigmoid":
            # Enhanced sigmoid with additional parameters
            dist = torch.distributions.normal.Normal(0, 1)
            t = dist.icdf(quantiles)
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            sigmoid_bias = getattr(args, "sigmoid_bias", 0.0)
            t = torch.sigmoid((t + sigmoid_bias) * sigmoid_scale)
        else:
            # Uniform distribution for "uniform" and other methods
            t = quantiles

        return t

    def _log_configuration(self, args: argparse.Namespace) -> None:
        """Log detailed configuration information."""
        logger.info("Timestep Distribution Configuration:")
        logger.info(f"   Sampling Method: '{args.timestep_sampling}'")
        logger.info(
            f"   Bucket Count: {getattr(args, 'precomputed_timestep_buckets', 10000):,}"
        )

        # Log method-specific parameters
        if args.timestep_sampling == "sigmoid":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
        elif args.timestep_sampling == "shift":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            shift = getattr(args, "discrete_flow_shift", 1.0)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(f"   Flow Shift: {shift}")
        elif args.timestep_sampling == "logit_normal":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(f"   Distribution: Logit-Normal (Normal(0,1) -> sigmoid)")
        elif args.timestep_sampling == "bell_shaped":
            bell_std = getattr(args, "bell_std", 0.2)
            bell_center = getattr(args, "bell_center", 0.5)
            logger.info(f"   Bell Standard Deviation: {bell_std}")
            logger.info(f"   Bell Center: {bell_center}")
            logger.info(
                f"   Distribution: Bell-Shaped Mean-Normalized (centered at {bell_center})"
            )
        elif args.timestep_sampling == "half_bell":
            bell_std = getattr(args, "bell_std", 0.2)
            logger.info(f"   Bell Standard Deviation: {bell_std}")
            logger.info(f"   Distribution: Half Bell-Shaped (bell curve + flat tail)")
        elif args.timestep_sampling == "lognorm_blend":
            alpha = getattr(args, "lognorm_blend_alpha", 0.75)
            logger.info(f"   LogNormal Blend Alpha: {alpha}")
            logger.info(f"   Distribution: LogNormal({alpha}) + Linear({1-alpha})")
        elif args.timestep_sampling == "enhanced_sigmoid":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            sigmoid_bias = getattr(args, "sigmoid_bias", 0.0)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(f"   Sigmoid Bias: {sigmoid_bias}")
            logger.info(f"   Distribution: Enhanced Sigmoid (with bias)")
        elif args.timestep_sampling == "flux_shift":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            base_area = getattr(args, "precomputed_midshift_area", None)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(
                f"   MidShift Base Area: {base_area if base_area is not None else 'auto'}"
            )
            logger.info("   Distribution: Flux Mid-Shift (precomputed)")
        elif args.timestep_sampling == "qwen_shift":
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            base_area = getattr(args, "precomputed_midshift_area", None)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(
                f"   MidShift Base Area: {base_area if base_area is not None else 'auto'}"
            )
            logger.info("   Distribution: Qwen Mid-Shift (precomputed)")
        elif args.timestep_sampling == "logsnr":
            mean = getattr(args, "logit_mean", 0.0)
            std = getattr(args, "logit_std", 1.0)
            logger.info(f"   LogSNR Mean: {mean}")
            logger.info(f"   LogSNR Std: {std}")
            logger.info("   Distribution: LogSNR")
        elif args.timestep_sampling in ("qinglong_flux", "qinglong_qwen"):
            sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
            base_area = getattr(args, "precomputed_midshift_area", None)
            mean = getattr(args, "logit_mean", 0.0)
            std = getattr(args, "logit_std", 1.0)
            logger.info(f"   Sigmoid Scale: {sigmoid_scale}")
            logger.info(
                f"   MidShift Base Area: {base_area if base_area is not None else 'auto'}"
            )
            logger.info(f"   LogSNR Mean: {mean}")
            logger.info(f"   LogSNR Std: {std}")
            variant = "Qwen" if args.timestep_sampling == "qinglong_qwen" else "Flux"
            logger.info(
                f"   Distribution: Qinglong ({variant}) Mixture 0.80/0.075/0.125"
            )

        # Log timestep constraints
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        if min_timestep is not None or max_timestep is not None:
            min_val = min_timestep if min_timestep is not None else 0
            max_val = max_timestep if max_timestep is not None else 1000
            logger.info(f"   Timestep Range: [{min_val}, {max_val}] (out of [0, 1000])")
            logger.info(
                f"   Normalized Range: [{min_val/1000:.3f}, {max_val/1000:.3f}]"
            )
        else:
            logger.info(f"   Timestep Range: [0, 1000] (full range)")

    def _log_initialization_results(
        self, args: argparse.Namespace, initial_buckets: int
    ) -> None:
        """Log initialization results with statistics."""
        final_buckets = len(self.distribution)  # type: ignore
        stats = self.get_stats()

        logger.info("Pre-computed timestep distribution ready.")
        logger.info(
            f"   Final Buckets: {final_buckets:,} (from {initial_buckets:,} initial)"
        )

        if final_buckets < initial_buckets:
            reduction_pct = (1 - final_buckets / initial_buckets) * 100
            logger.info(
                f"   Constraint Reduction: {reduction_pct:.1f}% of buckets removed"
            )

        logger.info(
            f"   Value Range: [{stats['min_value']:.6f}, {stats['max_value']:.6f}]"
        )
        logger.info(
            f"   Distribution stats: mean={stats['mean_value']:.6f}, std={stats['std_value']:.6f}"
        )

        # Memory usage estimation
        memory_mb = (final_buckets * 4) / (1024 * 1024)  # 4 bytes per float32
        logger.info(f"   Memory Usage: ~{memory_mb:.2f} MB")

    def _apply_timestep_constraints(self, args: argparse.Namespace) -> None:
        """Apply min/max timestep constraints to the distribution.

        Args:
            args: Training arguments containing min/max timestep settings
        """
        t_min = (
            getattr(args, "min_timestep", 0) / 1000.0
            if hasattr(args, "min_timestep") and args.min_timestep is not None
            else 0.0
        )
        t_max = (
            getattr(args, "max_timestep", 1000) / 1000.0
            if hasattr(args, "max_timestep") and args.max_timestep is not None
            else 1.0
        )

        if (t_min > 0.0 or t_max < 1.0) and self.distribution is not None:
            original_size = len(self.distribution)
            self.distribution = self._slice_distribution(
                self.distribution, t_min, t_max
            )
            logger.info(
                f"   Applied constraints: {original_size:,} -> {len(self.distribution):,} buckets"
            )

    def _slice_distribution(
        self, distribution: torch.Tensor, min_t: float = 0.0, max_t: float = 1.0
    ) -> torch.Tensor:
        """Slice timestep distribution to specific range for curriculum learning.

        Args:
            distribution: Full timestep distribution
            min_t: Minimum timestep value (0.0 to 1.0)
            max_t: Maximum timestep value (0.0 to 1.0)

        Returns:
            Sliced distribution within the specified range
        """
        start = torch.searchsorted(distribution, min_t).item()
        end = torch.searchsorted(distribution, max_t).item()
        return distribution[start:end]

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        quantile: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample timesteps from the pre-computed distribution.

        Args:
            batch_size: Number of timesteps to sample
            device: Target device for the sampled timesteps
            quantile: Optional specific quantile for deterministic sampling

        Returns:
            Sampled timesteps tensor of shape [batch_size]
        """
        if not self.is_initialized or self.distribution is None:
            raise RuntimeError(
                "TimestepDistribution not initialized. Call initialize() first."
            )

        if quantile is not None:
            # Deterministic sampling at specific quantile
            # Validate quantile is a number before conversion
            if not isinstance(quantile, (int, float)):
                raise ValueError(f"quantile must be a number, got {type(quantile)}")
            
            q = float(max(0.0, min(quantile, 1.0 - 1e-9)))
            i = (torch.full((batch_size,), q) * len(self.distribution)).to(torch.int64)
            i = torch.clamp(i, 0, len(self.distribution) - 1)
        else:
            # Random sampling from distribution
            i = torch.randint(
                0, len(self.distribution), size=(batch_size,), dtype=torch.int64
            )

        return self.distribution[i].to(device)

    def get_stats(self) -> dict:
        """Get statistics about the current distribution.

        Returns:
            Dictionary containing distribution statistics
        """
        if not self.is_initialized or self.distribution is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "num_buckets": len(self.distribution),
            "min_value": self.distribution.min().item(),
            "max_value": self.distribution.max().item(),
            "mean_value": self.distribution.mean().item(),
            "std_value": self.distribution.std().item(),
        }


def create_timestep_distribution() -> TimestepDistribution:
    """Factory function to create a new TimestepDistribution instance.

    Returns:
        New TimestepDistribution instance
    """
    return TimestepDistribution()


def should_use_precomputed_timesteps(args: argparse.Namespace) -> bool:
    """Check if pre-computed timestep distribution should be used.

    Args:
        args: Training arguments

    Returns:
        True if pre-computed timesteps should be used, False otherwise
    """
    return getattr(args, "use_precomputed_timesteps", False)
