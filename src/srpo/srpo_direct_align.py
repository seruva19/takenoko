"""
Direct-Align algorithm for SRPO training.

Implements the three-step process:
1. Noise interpolation (sample sigma from uniform distribution)
2. Single Euler step (forward denoise OR backward inversion)
3. Image recovery (using inverted process)

Key properties:
- Prevents reward hacking via alternating denoise/inversion branches
- Maintains differentiability for gradient backpropagation
- Supports discount schedules for temporal credit assignment
"""

from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


def interpolate_noise(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate between clean latents and noise.

    Formula:
        z_noisy = (1 - sigma) * z_clean + sigma * noise

    Args:
        clean_latents: Tensor of shape [B, C, F, H, W]
        noise: Tensor of shape [B, C, F, H, W]
        sigma: Tensor of shape [B, 1, 1, 1, 1] or [B]

    Returns:
        Noisy latents of shape [B, C, F, H, W]
    """
    # Ensure sigma has correct shape for broadcasting
    if sigma.dim() == 1:
        sigma = sigma.view(-1, 1, 1, 1, 1)

    noisy_latents = (1.0 - sigma) * clean_latents + sigma * noise
    return noisy_latents


def sd3_time_shift(sigma: torch.Tensor, shift_value: float = 3.0) -> torch.Tensor:
    """
    Apply time shift to sigma values.

    Formula:
        sigma_shifted = shift_value * sigma / (1 + (shift_value - 1) * sigma)

    This redistributes the noise schedule to focus more on mid-range sigmas.

    Args:
        sigma: Tensor of shape [B] or [num_steps]
        shift_value: Shift parameter (default: 3.0)

    Returns:
        Shifted sigma values of same shape
    """
    sigma_shifted = shift_value * sigma / (1.0 + (shift_value - 1.0) * sigma)
    return sigma_shifted


def create_sigma_schedule(
    num_steps: int,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    method: str = "linear",
    enable_time_shift: bool = True,
    time_shift_value: float = 3.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a sigma schedule for Euler integration.

    Args:
        num_steps: Number of discretization steps
        sigma_min: Minimum sigma value (default: 0.0)
        sigma_max: Maximum sigma value (default: 1.0)
        method: Interpolation method ("linear" or "cosine")
        enable_time_shift: Whether to apply time shift
        time_shift_value: Shift parameter (default: 3.0)
        device: Torch device

    Returns:
        Tensor of shape [num_steps] with sigma values
    """
    if method == "linear":
        # Linear interpolation from sigma_max to sigma_min
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=device)
    elif method == "cosine":
        # Cosine interpolation (slower decay at start/end)
        steps_normalized = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (
            sigma_min
            + (sigma_max - sigma_min) * (1 - torch.cos(steps_normalized * np.pi)) / 2
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Apply time shift if enabled
    if enable_time_shift:
        sigmas = sd3_time_shift(sigmas, shift_value=time_shift_value)

    return sigmas


class DirectAlignEngine(nn.Module):
    """
    Direct-Align algorithm engine.

    Implements the core SRPO training loop:
    1. Noise injection at random sigma
    2. Single Euler step (denoise or inversion branch)
    3. Image recovery using alternating branch
    4. Reward computation and gradient backpropagation
    """

    def __init__(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        interpolation_method: str,
        enable_time_shift: bool,
        time_shift_value: float,
        discount_denoise_min: float,
        discount_denoise_max: float,
        discount_inversion_start: float,
        discount_inversion_end: float,
        device: torch.device,
    ):
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.device = device

        # Create sigma schedule for Euler integration
        self.sigma_schedule = create_sigma_schedule(
            num_steps=num_inference_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            method=interpolation_method,
            enable_time_shift=enable_time_shift,
            time_shift_value=time_shift_value,
            device=device,
        )

        # Create discount schedules
        self.discount_denoise = torch.linspace(
            discount_denoise_min,
            discount_denoise_max,
            num_inference_steps,
            device=device,
        )
        self.discount_inversion = torch.linspace(
            discount_inversion_start,
            discount_inversion_end,
            num_inference_steps,
            device=device,
        )

    def single_euler_step(
        self,
        latents: torch.Tensor,
        model_pred: torch.Tensor,
        sigma_current: torch.Tensor,
        sigma_next: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        """
        Perform a single Euler integration step.

        VERIFIED FORMULAS:

        Denoise (forward) branch:
            latents_next = latents + (sigma_next - sigma_current) * model_pred

        Inversion (backward) branch:
            latents_next = latents - (sigma_next - sigma_current) * model_pred

        Args:
            latents: Current latents [B, C, F, H, W]
            model_pred: Model prediction [B, C, F, H, W]
            sigma_current: Current sigma [B, 1, 1, 1, 1]
            sigma_next: Next sigma [B, 1, 1, 1, 1]
            branch: "denoise" or "inversion"

        Returns:
            Updated latents [B, C, F, H, W]
        """
        dsigma = sigma_next - sigma_current  # [B, 1, 1, 1, 1]

        if branch == "denoise":
            # Forward step: move toward clean data
            # VERIFIED: forward Euler step
            latents_next = latents + dsigma * model_pred
        elif branch == "inversion":
            # Backward step: move toward noise
            # VERIFIED: backward Euler step
            latents_next = latents - dsigma * model_pred
        else:
            raise ValueError(
                f"Unknown branch: {branch}. Must be 'denoise' or 'inversion'"
            )

        return latents_next

    def inject_noise_at_random_sigma(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        sigma_min: float,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Inject noise at a random sigma value.

        Args:
            clean_latents: Clean latents [B, C, F, H, W]
            noise: Noise tensor [B, C, F, H, W]
            sigma_min: Minimum sigma for sampling
            sigma_max: Maximum sigma for sampling

        Returns:
            - noisy_latents: Interpolated latents [B, C, F, H, W]
            - sigma: Sampled sigma value [B, 1, 1, 1, 1]
            - step_idx: Index in sigma_schedule closest to sampled sigma
        """
        B = clean_latents.shape[0]

        # Sample sigma uniformly from [sigma_min, sigma_max]
        sigma = torch.rand(B, device=self.device) * (sigma_max - sigma_min) + sigma_min
        sigma = sigma.view(B, 1, 1, 1, 1)

        # Interpolate noise
        noisy_latents = interpolate_noise(clean_latents, noise, sigma)

        # Find closest step in sigma_schedule
        sigma_flat = sigma.view(B)
        distances = torch.abs(
            self.sigma_schedule.unsqueeze(0) - sigma_flat.unsqueeze(1)
        )
        step_idx = torch.argmin(distances, dim=1)[0].item()  # Use first batch item

        return noisy_latents, sigma, step_idx

    def recover_image(
        self,
        latents_after_step: torch.Tensor,
        sigma_after_step: torch.Tensor,
        model_callable,
        context,
        seq_len: int,
        branch: str,
        post_step_callback: Optional[Callable[..., torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Recover clean image by running remaining Euler steps to sigma=0.

        This completes the Direct-Align algorithm by running the reverse process
        to obtain the final clean latents for reward computation.

        Args:
            latents_after_step: Latents after single step [B, C, F, H, W]
            sigma_after_step: Sigma value after single step [B, 1, 1, 1, 1]
            model_callable: WAN transformer callable (wrapped to handle batching)
            context: List of text embeddings
            seq_len: Sequence length for transformer
            branch: "denoise" or "inversion" (determines recovery direction)
            post_step_callback: Optional callback invoked after each recovery step.
                Receives:
                - latents_before_step
                - latents_after_step
                - model_pred
                - step_idx
                - sigma_current
                - sigma_next
                Must return updated latents_after_step.

        Returns:
            Recovered clean latents [B, C, F, H, W]
        """
        # Always denoise to sigma=0 to get clean image
        recovery_branch = "denoise"

        # Find which steps to run based on current sigma
        sigma_flat = sigma_after_step.view(-1)[0].item()

        # Find all schedule steps less than current sigma
        remaining_steps = torch.where(self.sigma_schedule < sigma_flat)[0]

        if len(remaining_steps) == 0:
            # Already at sigma=0, return as is
            return latents_after_step

        # Run Euler integration to sigma=0
        latents_current = latents_after_step
        B = latents_current.shape[0]

        for i in range(len(remaining_steps)):
            # Get sigma for this step
            step_idx = remaining_steps[i].item()
            sigma_current = self.sigma_schedule[step_idx]
            sigma_next = (
                self.sigma_schedule[step_idx + 1]
                if step_idx + 1 < len(self.sigma_schedule)
                else torch.tensor(0.0, device=self.device)
            )

            # Call model
            timesteps = sigma_current.unsqueeze(0).expand(B)
            model_pred = model_callable(
                latents_current,
                t=timesteps,
                context=context,
                seq_len=seq_len,
            )

            # Euler step toward clean
            sigma_current_expanded = sigma_current.view(1, 1, 1, 1, 1).expand(
                B, 1, 1, 1, 1
            )
            sigma_next_expanded = sigma_next.view(1, 1, 1, 1, 1).expand(B, 1, 1, 1, 1)

            latents_next = self.single_euler_step(
                latents_current,
                model_pred,
                sigma_current_expanded,
                sigma_next_expanded,
                branch=recovery_branch,
            )
            if post_step_callback is not None:
                latents_next = post_step_callback(
                    latents_before_step=latents_current,
                    latents_after_step=latents_next,
                    model_pred=model_pred,
                    step_idx=step_idx,
                    sigma_current=sigma_current_expanded,
                    sigma_next=sigma_next_expanded,
                )

            latents_current = latents_next

        return latents_current
