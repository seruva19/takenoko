## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/hv_train_network.py (Apache)

"""Timestep utilities for diffusion model training.

This module consolidates all timestep-related functionality including:
- Time shift transformations
- Linear function generation for spatial scaling
- Noisy model input generation with various sampling strategies
- Integration with pre-computed timestep distributions
"""

import argparse
import math
from typing import Callable, Optional, Tuple, Any

import torch
import numpy as np

from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from scheduling.timestep_distribution import (
    TimestepDistribution,
    should_use_precomputed_timesteps,
)
from utils.train_utils import get_sigmas, compute_density_for_timestep_sampling
from scheduling.fopp import (
    FoPPScheduler,
    get_alpha_bar_schedule,
    apply_asynchronous_noise,
)
from common.logger import get_logger

logger = get_logger(__name__)

# Guard flags to avoid spamming logs
_warned_double_constraint: bool = False


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Apply a time shift transformation to the timestep tensor.

    Args:
        mu (float): Shift parameter (usually >0).
        sigma (float): Exponent parameter.
        t (torch.Tensor): Timesteps in [0, 1], shape (B,).

    Returns:
        torch.Tensor: Shifted timesteps, shape (B,).
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    """Return a linear function f(x) = m*x + b passing through (x1, y1) and (x2, y2).

    Args:
        x1 (float): First x-coordinate
        y1 (float): First y-coordinate
        x2 (float): Second x-coordinate
        y2 (float): Second y-coordinate

    Returns:
        Callable[[float], float]: Linear function f(x) = m*x + b
    """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def compute_sampled_timesteps_and_weighting(
    args: Any,
    timestep_distribution: Any,
    noise_scheduler: Any,
    num_samples: int = 100000,
    batch_size: int = 1000,
) -> tuple[list[int], list[float]]:
    """Compute sampled timestep counts (0..999) and per-index loss weighting.

    Returns:
    - sampled_timesteps: length-1000 list of counts
    - sampled_weighting: length-1000 list of weights
    """
    import torch

    # Ensure distribution is initialized once
    try:
        from scheduling.timestep_utils import initialize_timestep_distribution

        initialize_timestep_distribution(args, timestep_distribution)
    except Exception:
        pass

    BATCH_SIZE = max(1, int(batch_size))
    N_TRY = max(BATCH_SIZE, int(num_samples))

    latents = torch.zeros(BATCH_SIZE, 1, 1, 1, 1, dtype=torch.float16)
    noise = torch.ones_like(latents)

    sampled_timesteps = [0] * 1000

    try:
        from scheduling.timestep_utils import get_noisy_model_input_and_timesteps

        for _ in range(N_TRY // BATCH_SIZE):
            actual_timesteps, _, _ = get_noisy_model_input_and_timesteps(
                args,
                noise,
                latents,
                noise_scheduler,
                torch.device("cpu"),
                torch.float16,
                timestep_distribution,
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                ti = int(t.item())
                if 0 <= ti < 1000:
                    sampled_timesteps[ti] += 1
    except Exception:
        # Fallback: uniform sampling if anything fails
        for i in range(1000):
            sampled_timesteps[i] = N_TRY // 1000

    # Compute per-index loss weighting
    sampled_weighting: list[float] = [1.0] * 1000
    try:
        from utils.train_utils import compute_loss_weighting_for_sd3

        for i in range(1000):
            ts = torch.tensor([i + 1], device="cpu")
            w = compute_loss_weighting_for_sd3(
                getattr(args, "weighting_scheme", "none"),
                noise_scheduler,
                ts,
                "cpu",
                torch.float16,
            )
            if w is None or (torch.isinf(w).any() or torch.isnan(w).any()):
                sampled_weighting[i] = 1.0
            else:
                sampled_weighting[i] = float(w.item())
    except Exception:
        pass

    return sampled_timesteps, sampled_weighting


def _sample_fopp_timesteps(
    args: argparse.Namespace,
    latents: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Sample FoPP AR-Diffusion timesteps for video training.

    Args:
        args: Training arguments
        latents: Input latents tensor
        device: Target device

    Returns:
        Tuple of (timesteps, alpha_bar_schedule)
    """
    batch_size = latents.shape[0]
    num_frames = latents.shape[1]
    num_timesteps = getattr(args, "fopp_num_timesteps", 1000)
    schedule_type = getattr(args, "fopp_schedule_type", "linear")
    beta_start = getattr(args, "fopp_beta_start", 0.0001)
    beta_end = getattr(args, "fopp_beta_end", 0.002)
    seed = getattr(args, "fopp_seed", None)

    # Sample per-frame timesteps for each video in the batch
    fopp_sched = FoPPScheduler(
        num_frames=num_frames,
        num_timesteps=num_timesteps,
        device=None if latents.device.type == "cpu" else latents.device,
        seed=seed,
    )
    timesteps_np = fopp_sched.sample_batch(batch_size)  # (B, F), np.int
    timesteps = torch.from_numpy(timesteps_np).to(device=device, dtype=torch.long)

    # Get alpha_bar schedule (configurable)
    alpha_bar = get_alpha_bar_schedule(
        num_timesteps,
        schedule_type=schedule_type,  # type: ignore
        beta_start=beta_start,
        beta_end=beta_end,
    )

    return timesteps, alpha_bar


def _normal_ppf(u: torch.Tensor) -> torch.Tensor:
    """Inverse CDF (percent point function) for standard normal using erfinv."""
    eps = 1e-7
    u = torch.clamp(u, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def map_uniform_to_sampling(
    args: argparse.Namespace, t_uniform: torch.Tensor, latents: torch.Tensor
) -> torch.Tensor:
    """Map uniform-in-[0,1] samples to the current timestep_sampling distribution.

    This mirrors transformations in _generate_timesteps_from_distribution but uses
    inverse-CDF style mappings from provided uniform samples instead of RNG.
    """
    method = str(getattr(args, "timestep_sampling", "uniform")).lower()
    device = t_uniform.device
    t_uniform = t_uniform.clamp(0.0, 1.0)

    if method == "uniform":
        return t_uniform

    if method == "sigmoid":
        z = _normal_ppf(t_uniform)
        return torch.sigmoid(getattr(args, "sigmoid_scale", 1.0) * z)

    if method == "shift":
        z = _normal_ppf(t_uniform)
        t = torch.sigmoid(getattr(args, "sigmoid_scale", 1.0) * z)
        shift = float(getattr(args, "discrete_flow_shift", 1.0))
        return (t * shift) / (1 + (shift - 1) * t)

    if method in ("flux_shift", "qwen_shift"):
        z = _normal_ppf(t_uniform)
        t = torch.sigmoid(getattr(args, "sigmoid_scale", 1.0) * z)
        # compute base area
        if latents is not None and latents.ndim >= 4:
            h, w = latents.shape[-2:]
            base_area = float((h // 2) * (w // 2))
        else:
            base_area = 1024.0
        if method == "qwen_shift":
            m = (0.9 - 0.5) / (8192 - 256)
            b = 0.5 - m * 256
        else:
            m = (1.15 - 0.5) / (4096 - 256)
            b = 0.5 - m * 256
        mu = m * base_area + b
        shift = math.exp(mu)
        return (t * shift) / (1 + (shift - 1) * t)

    if method == "logit_normal":
        z = _normal_ppf(t_uniform)
        z = z * float(getattr(args, "sigmoid_scale", 1.0))
        return torch.sigmoid(z)

    if method == "bell_shaped":
        # Build bell density over x∈[0,1], invert its CDF at given uniform quantiles
        n = max(int(1e4), 2048)
        x = torch.linspace(0.0, 1.0, n, device=device)
        bell_std = float(getattr(args, "bell_std", 0.2))
        bell_center = float(getattr(args, "bell_center", 0.5))
        y = torch.exp(-0.5 * ((x - bell_center) / max(bell_std, 1e-6)) ** 2)
        y = torch.clamp(y, min=1e-12)
        cdf = torch.cumsum(y, dim=0)
        cdf = cdf / cdf[-1]
        # Invert CDF by interpolation
        idx = torch.searchsorted(cdf, t_uniform.clamp(0.0, 1.0), right=False)
        idx = idx.clamp(min=1, max=n - 1)
        cdf_lo = cdf[idx - 1]
        cdf_hi = cdf[idx]
        x_lo = x[idx - 1]
        x_hi = x[idx]
        denom = torch.clamp(cdf_hi - cdf_lo, min=1e-12)
        w = (t_uniform - cdf_lo) / denom
        return x_lo + w * (x_hi - x_lo)

    if method == "half_bell":
        n = max(int(1e4), 2048)
        x = torch.linspace(0.0, 1.0, n, device=device)
        bell_std = float(getattr(args, "bell_std", 0.2))
        y = torch.exp(-0.5 * ((x - 0.5) / max(bell_std, 1e-6)) ** 2)
        mid = n // 2
        y[mid:] = y[:mid].max()
        y = torch.clamp(y, min=1e-12)
        cdf = torch.cumsum(y, dim=0)
        cdf = cdf / cdf[-1]
        idx = torch.searchsorted(cdf, t_uniform.clamp(0.0, 1.0), right=False)
        idx = idx.clamp(min=1, max=n - 1)
        cdf_lo = cdf[idx - 1]
        cdf_hi = cdf[idx]
        x_lo = x[idx - 1]
        x_hi = x[idx]
        denom = torch.clamp(cdf_hi - cdf_lo, min=1e-12)
        w = (t_uniform - cdf_lo) / denom
        return x_lo + w * (x_hi - x_lo)

    if method == "logsnr":
        z = _normal_ppf(t_uniform)
        mean = float(getattr(args, "logit_mean", 0.0))
        std = float(getattr(args, "logit_std", 1.0))
        logsnr = mean + std * z
        return torch.sigmoid(-logsnr / 2.0)

    if method in ("qinglong_flux", "qinglong_qwen"):
        # Use mixture with thresholds on uniform sample
        # 0..0.80 -> mid_shift, 0.80..0.875 -> logsnr(mean,std), 0.875..1.0 -> logsnr2(5.36,1.0)
        u = t_uniform
        out = torch.zeros_like(u, device=device)
        # mid_shift
        mask0 = u < 0.80
        if mask0.any():
            # rescale to [0,1] within this segment to preserve shape
            u0 = (u[mask0] - 0.0) / 0.80
            z0 = _normal_ppf(u0)
            t0 = torch.sigmoid(getattr(args, "sigmoid_scale", 1.0) * z0)
            # area and mu
            if latents is not None and latents.ndim >= 4:
                h, w = latents.shape[-2:]
                base_area = float((h // 2) * (w // 2))
            else:
                base_area = 1024.0
            if method == "qinglong_qwen":
                m = (0.9 - 0.5) / (8192 - 256)
                b = 0.5 - m * 256
            else:
                m = (1.15 - 0.5) / (4096 - 256)
                b = 0.5 - m * 256
            mu = m * base_area + b
            shift = math.exp(mu)
            out[mask0] = (t0 * shift) / (1 + (shift - 1) * t0)
        # logsnr
        mask1 = (u >= 0.80) & (u < 0.875)
        if mask1.any():
            u1 = (u[mask1] - 0.80) / (0.075)
            z1 = _normal_ppf(u1)
            mean = float(getattr(args, "logit_mean", 0.0))
            std = float(getattr(args, "logit_std", 1.0))
            logsnr = mean + std * z1
            out[mask1] = torch.sigmoid(-logsnr / 2.0)
        # logsnr2 fixed mean
        mask2 = u >= 0.875
        if mask2.any():
            u2 = (u[mask2] - 0.875) / (0.125)
            z2 = _normal_ppf(u2)
            logsnr2 = 5.36 + 1.0 * z2
            out[mask2] = torch.sigmoid(-logsnr2 / 2.0)
        return out

    # Default fallback: return uniform
    return t_uniform


def _generate_timesteps_from_distribution(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate timesteps using the specified distribution method.

    This is a shared method used by both _sample_standard_timesteps and
    _apply_timestep_constraints to avoid code duplication.

    Args:
        args: Training arguments
        batch_size: Number of samples to generate
        device: Target device
        latents: Optional latents tensor for flux_shift spatial calculations

    Returns:
        Timesteps tensor in [0, 1] range
    """
    if args.timestep_sampling == "uniform":
        t = torch.rand((batch_size,), device=device)

    elif args.timestep_sampling == "sigmoid":
        t = torch.sigmoid(
            args.sigmoid_scale * torch.randn((batch_size,), device=device)
        )

    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        logits_norm = torch.randn(batch_size, device=device)
        logits_norm = (
            logits_norm * args.sigmoid_scale
        )  # larger scale for more uniform sampling
        t = logits_norm.sigmoid()
        t = (t * shift) / (1 + (shift - 1) * t)

    elif args.timestep_sampling == "flux_shift":
        # https://github.com/kohya-ss/sd-scripts/pull/1541
        logits_norm = torch.randn(batch_size, device=device)
        logits_norm = logits_norm * args.sigmoid_scale
        t = logits_norm.sigmoid()

        # Compute mu as a function of spatial size (matching upstream implementation)
        if latents is not None:
            h, w = latents.shape[-2:] if latents.ndim >= 4 else (1, 1)
            # we are pre-packed so must adjust for packed size
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            # def time_shift(mu: float, sigma: float, t: torch.Tensor):
            #     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma) # sigma=1.0
            shift = math.exp(mu)
            t = (t * shift) / (1 + (shift - 1) * t)
        else:
            # Fall back to simple sigmoid if latents not available
            logger.warning("flux_shift without latents, using simple sigmoid")

    elif args.timestep_sampling == "qwen_shift":
        # Qwen shift uses a different linear mapping for mu than flux
        logits_norm = torch.randn(batch_size, device=device)
        logits_norm = logits_norm * args.sigmoid_scale
        t = logits_norm.sigmoid()

        if latents is not None:
            h, w = latents.shape[-2:] if latents.ndim >= 4 else (1, 1)
            # Use upstream qwen mapping: (x1=256, y1=0.5) -> (x2=8192, y2=0.9)
            mu = get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
            shift = math.exp(mu)
            t = (t * shift) / (1 + (shift - 1) * t)
        else:
            logger.warning("qwen_shift without latents, using simple sigmoid")

    elif args.timestep_sampling == "logit_normal":
        # Use logit-normal distribution
        dist = torch.distributions.normal.Normal(0, 1)
        t = dist.sample((batch_size,)).to(device)

        # Apply sigmoid scaling
        sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
        t = t * sigmoid_scale
        t = torch.sigmoid(t)

    elif args.timestep_sampling == "bell_shaped":
        # Bell-shaped distribution centered at `bell_center` with spread `bell_std`.
        # This focuses sampling around a specific region (e.g., near 0.95 → timestep ~950).
        x = torch.rand(batch_size, device=device)
        bell_std = float(getattr(args, "bell_std", 0.2))
        bell_center = float(getattr(args, "bell_center", 0.5))
        # Gaussian-shaped bump around center
        y = torch.exp(-0.5 * ((x - bell_center) / max(bell_std, 1e-6)) ** 2)
        # Normalize to [0, 1]
        y_shifted = y - y.min()
        t = y_shifted / y_shifted.max()

    elif args.timestep_sampling == "half_bell":
        # Half Bell-Shaped (HBSMNTW) - bell curve for first half, flat for second half
        x = torch.rand(batch_size, device=device)
        bell_std = getattr(args, "bell_std", 0.2)
        y = torch.exp(-0.5 * ((x - 0.5) / bell_std) ** 2)
        y_shifted = y - y.min()

        # Flatten second half to max value
        mid_point = batch_size // 2
        y_shifted[mid_point:] = y_shifted[:mid_point].max()

        t = y_shifted / y_shifted.max()

    elif args.timestep_sampling == "lognorm_blend":
        # LogNormal Blend - combines lognormal distribution with linear sampling
        alpha = getattr(args, "lognorm_blend_alpha", 0.75)

        # Determine how many samples to use for each distribution
        t1_size = int(batch_size * alpha)
        t2_size = batch_size - t1_size

        if t1_size > 0:
            # LogNormal distribution for first portion
            lognormal = torch.distributions.LogNormal(loc=0, scale=0.333)
            t1 = lognormal.sample((t1_size,)).to(device)  # type: ignore
            t1_max = t1.max()
            if t1_max > 0:
                t1 = 1 - t1 / t1_max  # Scale to [0, 1]
            else:
                t1 = torch.zeros_like(t1)
        else:
            t1 = torch.empty(0, device=device)

        if t2_size > 0:
            # Linear distribution for remaining portion
            t2 = torch.rand(t2_size, device=device)
        else:
            t2 = torch.empty(0, device=device)

        # Combine and sort
        t = torch.cat([t1, t2])
        if len(t) > 0:
            t, _ = torch.sort(t)

    elif args.timestep_sampling == "enhanced_sigmoid":
        # Enhanced sigmoid with additional parameters
        sigmoid_scale = getattr(args, "sigmoid_scale", 1.0)
        sigmoid_bias = getattr(args, "sigmoid_bias", 0.0)
        t = torch.randn(batch_size, device=device)
        t = torch.sigmoid((t + sigmoid_bias) * sigmoid_scale)

    # https://github.com/kohya-ss/musubi-tuner/pull/407
    elif args.timestep_sampling == "logsnr":
        # https://arxiv.org/abs/2411.14793v3
        logsnr = torch.normal(
            mean=args.logit_mean, std=args.logit_std, size=(batch_size,), device=device
        )
        t = torch.sigmoid(-logsnr / 2)

    elif args.timestep_sampling in ("qinglong_flux", "qinglong_qwen"):
        # Qinglong triple hybrid sampling: mid_shift:logsnr:logsnr2 = .80:.075:.125
        # First decide which method to use for each sample independently
        decision_t = torch.rand((batch_size,), device=device)

        # Create masks based on decision_t: .80 for mid_shift, 0.075 for logsnr, and 0.125 for logsnr2
        flux_mask = decision_t < 0.80  # 80% for mid_shift (flux or qwen variant)
        logsnr_mask = (decision_t >= 0.80) & (decision_t < 0.875)  # 7.5% for logsnr
        logsnr_mask2 = decision_t >= 0.875  # 12.5% for logsnr with -logit_mean

        # Initialize output tensor
        t = torch.zeros((batch_size,), device=device)

        # Generate mid_shift samples for selected indices (80%)
        if flux_mask.any():
            flux_count = int(flux_mask.sum().item())
            h, w = latents.shape[-2:] if latents is not None else (1, 1)
            # Choose mu mapping: flux variant uses flux mapping; qwen variant uses alternate mapping
            if args.timestep_sampling == "qinglong_qwen":
                mu = get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)(
                    (h // 2) * (w // 2)
                )
            else:
                # "qinglong_flux" uses flux mapping
                mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            shift = math.exp(mu)

            logits_norm_flux = torch.randn(flux_count, device=device)
            logits_norm_flux = logits_norm_flux * args.sigmoid_scale
            t_flux = logits_norm_flux.sigmoid()
            t_flux = (t_flux * shift) / (1 + (shift - 1) * t_flux)

            t[flux_mask] = t_flux

        # Generate logsnr samples for selected indices (7.5%)
        if logsnr_mask.any():
            logsnr_count = int(logsnr_mask.sum().item())
            logsnr = torch.normal(
                mean=args.logit_mean,
                std=args.logit_std,
                size=(logsnr_count,),
                device=device,
            )
            t_logsnr = torch.sigmoid(-logsnr / 2)

            t[logsnr_mask] = t_logsnr

        # Generate logsnr2 samples with -logit_mean for selected indices (12.5%)
        if logsnr_mask2.any():
            logsnr2_count = int(logsnr_mask2.sum().item())
            logsnr2 = torch.normal(
                mean=5.36, std=1.0, size=(logsnr2_count,), device=device
            )
            t_logsnr2 = torch.sigmoid(-logsnr2 / 2)

            t[logsnr_mask2] = t_logsnr2

    else:
        raise ValueError(f"Unknown timestep sampling method: {args.timestep_sampling}")

    return t


def _sample_standard_timesteps(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    latents: torch.Tensor,
    timestep_distribution: Optional[TimestepDistribution] = None,
) -> torch.Tensor:
    """Sample standard timesteps using various sampling strategies.

    Args:
        args: Training arguments
        batch_size: Number of samples to generate
        device: Target device
        timestep_distribution: Optional pre-computed distribution

    Returns:
        Sampled timesteps tensor
    """
    # Check if we should use pre-computed timestep distribution
    if (
        should_use_precomputed_timesteps(args)
        and timestep_distribution is not None
        and timestep_distribution.is_initialized
    ):
        # Use pre-computed distribution for supported methods
        if args.timestep_sampling in [
            "uniform",
            "sigmoid",
            "shift",
            "flux_shift",
            "qwen_shift",
            "logit_normal",
            "bell_shaped",
            "half_bell",
            "lognorm_blend",
            "enhanced_sigmoid",
            "logsnr",
            "qinglong_flux",
            "qinglong_qwen",
        ]:
            t = timestep_distribution.sample(batch_size, device)
        else:
            # Fall back to original sampling for unsupported methods
            logger.debug(
                f"⚠️  {args.timestep_sampling} does not support precomputed timesteps, using original sampling"
            )
            t = _generate_timesteps_from_distribution(args, batch_size, device, latents)
    else:
        # Use original sampling method
        if should_use_precomputed_timesteps(args):
            logger.debug(
                "⚠️  Fallback to original sampling: precomputed distribution not initialized"
            )
        t = _generate_timesteps_from_distribution(args, batch_size, device, latents)

    return t


def _apply_timestep_constraints(
    t: torch.Tensor,
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    latents: Optional[torch.Tensor] = None,
    presampled_uniform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply min/max timestep constraints to the timestep tensor.

    Args:
        t: Timestep tensor in [0, 1]
        args: Training arguments containing min/max timestep settings
        batch_size: Number of samples to generate
        device: Target device
        latents: Optional latents tensor for flux_shift spatial calculations

    Returns:
        Constrained timestep tensor
    """
    t_min = args.min_timestep if args.min_timestep is not None else 0
    t_max = args.max_timestep if args.max_timestep is not None else 1000.0
    t_min /= 1000.0
    t_max /= 1000.0

    # Ensure t is a tensor before calling .view()
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)

    # Optional: if timesteps already lie within [t_min, t_max], and user requested
    # to skip extra constraining, return as-is to avoid double scaling that
    # compresses the distribution toward the upper bound.
    if bool(getattr(args, "skip_extra_timestep_constraint", False)):
        eps = float(getattr(args, "timestep_constraint_epsilon", 1e-6))
        if torch.all(t >= (t_min - eps)) and torch.all(t <= (t_max + eps)):
            return t

    # Check if we should preserve the distribution shape
    if not getattr(args, "preserve_distribution_shape", False):
        # Simple scaling approach (original behavior)
        return t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
    else:
        # Rejection sampling to preserve distribution shape
        fast = bool(getattr(args, "fast_rejection_sampling", False))
        if fast:
            try:
                overdraw = float(getattr(args, "rejection_overdraw_factor", 4.0))
                max_iters = int(getattr(args, "rejection_max_iters", 10))
                available: list[torch.Tensor] = []

                # Use presampled uniforms first if available
                needed = int(batch_size)
                if presampled_uniform is not None and needed > 0:
                    mapped = map_uniform_to_sampling(
                        args,
                        presampled_uniform.to(device=device),
                        latents if latents is not None else torch.empty(0),
                    )
                    mask = (mapped >= t_min) & (mapped <= t_max)
                    selected = mapped[mask]
                    take = int(min(needed, selected.numel()))
                    if take > 0:
                        available.append(selected[:take])
                        needed -= take

                for _ in range(max_iters):
                    if needed <= 0:
                        break
                    n = max(int(math.ceil(needed * overdraw)), needed)
                    u = torch.rand((n,), device=device)
                    mapped = map_uniform_to_sampling(
                        args, u, latents if latents is not None else torch.empty(0)
                    )
                    mask = (mapped >= t_min) & (mapped <= t_max)
                    selected = mapped[mask]
                    if selected.numel() > 0:
                        take = int(min(needed, selected.numel()))
                        available.append(selected[:take])
                        needed -= take

                if needed > 0:
                    logger.warning(
                        "Preserve-shape(fast): insufficient in-range samples; falling back to linear scaling"
                    )
                    return t * (t_max - t_min) + t_min

                return torch.cat(available, dim=0)
            except Exception as e:
                logger.warning(
                    f"Fast rejection sampling failed ({e}); falling back to simple scaling"
                )
                return t * (t_max - t_min) + t_min
        else:
            max_loops = 1000
            available_t = []

            for i in range(max_loops):
                # Generate candidates:
                # - On first loop, if presampled_uniform provided, map those to the distribution
                # - Otherwise, generate fresh uniform and map
                try:
                    if i == 0 and presampled_uniform is not None:
                        new_t = map_uniform_to_sampling(
                            args,
                            presampled_uniform.to(device=device),
                            latents if latents is not None else torch.empty(0),
                        )
                    else:
                        u = torch.rand((batch_size,), device=device)
                        new_t = map_uniform_to_sampling(
                            args, u, latents if latents is not None else torch.empty(0)
                        )
                except Exception as e:
                    logger.warning(
                        f"Error mapping timesteps: {e}, falling back to simple scaling"
                    )
                    return t * (t_max - t_min) + t_min

                # Check which timesteps fall within bounds
                for t_i in new_t:
                    if t_min <= (t_i.item()) <= t_max:
                        available_t.append(t_i)
                    if len(available_t) == batch_size:
                        break
                if len(available_t) == batch_size:
                    break

            if len(available_t) < batch_size:
                logger.warning(
                    f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
                )
                # Fall back to original timesteps with simple scaling
                return t * (t_max - t_min) + t_min
            else:
                return torch.stack(available_t, dim=0)  # [batch_size, ]


def _sample_sigma_timesteps(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    noise_scheduler: FlowMatchDiscreteScheduler,
) -> torch.Tensor:
    """Sample timesteps for sigma-based weighting schemes.

    Args:
        args: Training arguments
        batch_size: Number of samples to generate
        device: Target device
        noise_scheduler: Noise scheduler for sigma computation

    Returns:
        Tuple of (timesteps, sigmas)
    """
    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=batch_size,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )

    t_min = args.min_timestep if args.min_timestep is not None else 0
    t_max = args.max_timestep if args.max_timestep is not None else 1000
    indices = (u * (t_max - t_min) + t_min).long()

    timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

    # Ensure timesteps is a tensor
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, device=device)

    return timesteps


def get_noisy_model_input_and_timesteps(
    args: argparse.Namespace,
    noise: torch.Tensor,
    latents: torch.Tensor,
    noise_scheduler: FlowMatchDiscreteScheduler,
    device: torch.device,
    dtype: torch.dtype,
    timestep_distribution: Optional[TimestepDistribution] = None,
    presampled_uniform: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Generate noisy model input and timesteps for training.

    This function handles various timestep sampling strategies including:
    - FoPP AR-Diffusion for video training
    - Standard sampling methods (uniform, sigmoid, shift, flux_shift)
    - Sigma-based weighting schemes
    - Pre-computed timestep distributions

    Args:
        args: Training arguments
        noise: Noise tensor
        latents: Input latents tensor
        noise_scheduler: Noise scheduler for sigma computation
        device: Target device
        dtype: Data type for computations
        timestep_distribution: Optional pre-computed timestep distribution

    Returns:
        Tuple of (noisy_model_input, timesteps, sigmas)
    """
    batch_size = noise.shape[0]

    # FoPP AR-Diffusion branch
    if getattr(args, "timestep_sampling", None) == "fopp":
        # Log if precomputed was requested but this method doesn't support it
        if should_use_precomputed_timesteps(args):
            logger.debug(
                "⚠️  FoPP AR-Diffusion does not support precomputed timesteps (complex AR logic)"
            )

        # --- FoPP AR-Diffusion: asynchronous, non-decreasing per-frame timesteps ---
        assert latents.ndim >= 3, "Latents must be at least (B, F, ...) for FoPP."

        timesteps, alpha_bar = _sample_fopp_timesteps(args, latents, device)
        noisy_model_input = apply_asynchronous_noise(
            latents, timesteps, noise, alpha_bar
        )
        return noisy_model_input, timesteps, None

    else:
        sigmas = None

        if (
            args.timestep_sampling == "uniform"
            or args.timestep_sampling == "sigmoid"
            or args.timestep_sampling == "shift"
            or args.timestep_sampling == "flux_shift"
            or args.timestep_sampling == "qwen_shift"
            or args.timestep_sampling == "logit_normal"
            or args.timestep_sampling == "logsnr"
            or args.timestep_sampling == "qinglong_flux"
            or args.timestep_sampling == "qinglong_qwen"
        ):
            # Sample timesteps using standard methods
            if presampled_uniform is not None and not should_use_precomputed_timesteps(
                args
            ):
                # Map uniform [0,1] to selected sampling distribution
                t = map_uniform_to_sampling(
                    args, presampled_uniform.to(device=device), latents
                )
            else:
                t = _sample_standard_timesteps(
                    args, batch_size, device, latents, timestep_distribution
                )

            # Apply timestep constraints
            # Warn once if configuration may cause unintended compression when using precomputed timesteps
            try:
                if (
                    should_use_precomputed_timesteps(args)
                    and not bool(getattr(args, "preserve_distribution_shape", False))
                    and not bool(getattr(args, "skip_extra_timestep_constraint", False))
                ):
                    t_min = (getattr(args, "min_timestep", 0) or 0) / 1000.0
                    t_max = (getattr(args, "max_timestep", 1000) or 1000) / 1000.0
                    if t.numel() > 0 and t.min() >= t_min and t.max() <= t_max:
                        global _warned_double_constraint
                        if not _warned_double_constraint:
                            logger.warning(
                                "⚠️ Timesteps: precomputed distribution + linear constraint without skip flag will compress the range. Consider skip_extra_timestep_constraint=true or preserve_distribution_shape=true (or disable precomputed)."
                            )
                            _warned_double_constraint = True
            except Exception:
                pass

            t = _apply_timestep_constraints(
                t, args, batch_size, device, latents, presampled_uniform
            )

            # Convert to timestep indices and create noisy input
            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps = timesteps + 1  # 1 to 1000
            # Optional: round training timesteps to nearest integer grid
            if getattr(args, "round_training_timesteps", False):
                try:
                    max_ts = int(
                        getattr(
                            getattr(noise_scheduler, "config", object()),
                            "num_train_timesteps",
                            1000,
                        )
                    )
                except Exception:
                    max_ts = 1000
                timesteps = timesteps.round().clamp_(1, max_ts)
            # Ensure timesteps is a tensor
            if not isinstance(timesteps, torch.Tensor):
                timesteps = torch.tensor(timesteps, device=device)

        else:  # sigma
            # Sample timesteps for sigma-based weighting schemes
            timesteps = _sample_sigma_timesteps(
                args, batch_size, device, noise_scheduler
            )

            # Add noise according to flow matching.
            sigmas = get_sigmas(
                noise_scheduler,
                timesteps,
                device,
                n_dim=latents.ndim,
                dtype=dtype,
                source="training/sigma-path",
            )
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    # Ensure return types are correct
    assert isinstance(
        noisy_model_input, torch.Tensor
    ), "noisy_model_input must be a torch.Tensor"
    assert isinstance(timesteps, torch.Tensor), "timesteps must be a torch.Tensor"

    return noisy_model_input, timesteps, sigmas


def initialize_timestep_distribution(
    args: argparse.Namespace,
    timestep_distribution: TimestepDistribution,
) -> None:
    """Initialize pre-computed timestep distribution if enabled.

    Args:
        args: Training arguments
        timestep_distribution: TimestepDistribution instance to initialize
    """
    if (
        should_use_precomputed_timesteps(args)
        and not timestep_distribution.is_initialized
    ):
        timestep_distribution.initialize(args)
        # One-time usage message after initialization
        if (
            hasattr(timestep_distribution, "usage_logged")
            and not timestep_distribution.usage_logged
        ):
            stats = timestep_distribution.get_stats()
            logger.debug(
                f"🚀 Using pre-computed timestep distribution: {stats['num_buckets']:,} buckets"
            )
            timestep_distribution.usage_logged = True
    elif should_use_precomputed_timesteps(args):
        # Only log once before the first step
        if (
            hasattr(timestep_distribution, "usage_logged")
            and not timestep_distribution.usage_logged
        ):
            stats = timestep_distribution.get_stats()
            logger.debug(
                f"🚀 Using pre-computed timestep distribution: {stats['num_buckets']:,} buckets"
            )
            timestep_distribution.usage_logged = True
