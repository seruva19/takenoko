## Based on: https://arxiv.org/abs/2503.07418

"""
FoPP (Frame-oriented Probability Propagation) scheduler and asynchronous noise application for AR-Diffusion training.

Implements the algorithm from "AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion".

Author: [Your Name]
"""

import numpy as np
import torch
from typing import Tuple, Optional, Literal


class FoPPScheduler:
    """
    Implements the Frame-oriented Probability Propagation (FoPP) timestep scheduler
    for AR-Diffusion, supporting batch sampling and precomputed DP matrices.

    Args:
        num_frames (int): Number of frames per video (F)
        num_timesteps (int): Number of diffusion timesteps (T)
        device (torch.device, optional): Device for output tensors
    """

    def __init__(
        self,
        num_frames: int,
        num_timesteps: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        if not (num_frames > 0 and num_timesteps > 0):
            raise ValueError("Number of frames and timesteps must be positive.")
        self.num_frames = num_frames
        self.num_timesteps = num_timesteps
        self.device = device
        # Create a deterministic RNG if seed provided; otherwise use a fresh generator
        self._rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        self.ds, self.de = self._precompute_matrices()

    def _precompute_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precomputes the d^s and d^e matrices using dynamic programming.
        """
        F, T = self.num_frames, self.num_timesteps
        ds = np.zeros((F, T), dtype=np.float64)
        de = np.zeros((F, T), dtype=np.float64)
        ds[F - 1, :] = 1
        ds[:, T - 1] = 1
        for i in range(F - 2, -1, -1):
            for j in range(T - 2, -1, -1):
                ds[i, j] = ds[i, j + 1] + ds[i + 1, j]
        de[0, :] = 1
        de[:, 0] = 1
        for i in range(1, F):
            for j in range(1, T):
                de[i, j] = de[i, j - 1] + de[i - 1, j]
        return ds, de

    def sample(self) -> np.ndarray:
        """
        Samples a single non-decreasing timestep composition (t_1, ..., t_F).
        Returns:
            np.ndarray: (F,) array of sampled timesteps (0-based)
        """
        F, T = self.num_frames, self.num_timesteps
        timesteps = np.zeros(F, dtype=int)
        f = self._rng.integers(0, F)
        tf = self._rng.integers(0, T)
        timesteps[f] = tf
        for i in range(f - 1, -1, -1):
            t_next = timesteps[i + 1]
            possible_range_end = t_next + 1
            weights = self.de[i, :possible_range_end]
            probs = weights / (np.sum(weights) + 1e-8)
            sampled_t = self._rng.choice(possible_range_end, p=probs)
            timesteps[i] = sampled_t
        for i in range(f + 1, F):
            t_prev = timesteps[i - 1]
            possible_timesteps = np.arange(t_prev, T)
            if len(possible_timesteps) == 0:
                timesteps[i] = t_prev
                continue
            weights = self.ds[i, t_prev:]
            probs = weights / (np.sum(weights) + 1e-8)
            sampled_t = self._rng.choice(possible_timesteps, p=probs)
            timesteps[i] = sampled_t
        return timesteps

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """
        Samples a batch of non-decreasing timestep compositions.
        Args:
            batch_size (int): Number of samples to generate
        Returns:
            np.ndarray: (batch_size, F) array of sampled timesteps
        """
        return np.stack([self.sample() for _ in range(batch_size)], axis=0)


def get_linear_beta_schedule(
    num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.002
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a linear beta schedule and computes alphas and alpha_bar (cumulative product).
    Args:
        num_timesteps (int): Number of diffusion steps (T)
        beta_start (float): Linear start value for beta
        beta_end (float): Linear end value for beta
    Returns:
        betas (np.ndarray): (T,) array
        alphas (np.ndarray): (T,) array
        alpha_bar (np.ndarray): (T,) array
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def get_cosine_beta_schedule(
    num_timesteps: int, s: float = 0.008
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a cosine beta schedule (for experimentation).
    Args:
        num_timesteps (int): Number of diffusion steps (T)
        s (float): Small offset for stability
    Returns:
        betas (np.ndarray): (T,) array
        alphas (np.ndarray): (T,) array
        alpha_bar (np.ndarray): (T,) array
    """
    steps = np.arange(num_timesteps + 1, dtype=np.float64)
    t = steps / num_timesteps
    alphas_cumprod = np.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def get_alpha_bar_schedule(
    num_timesteps: int,
    schedule_type: Literal["linear", "cosine"] = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.002,
) -> np.ndarray:
    """
    Returns the alpha_bar schedule for the given type.
    Args:
        num_timesteps (int): Number of diffusion steps (T)
        schedule_type (str): 'linear' or 'cosine'
        beta_start (float): For linear schedule
        beta_end (float): For linear schedule
    Returns:
        alpha_bar (np.ndarray): (T,) array
    """
    if schedule_type == "linear":
        _, _, alpha_bar = get_linear_beta_schedule(num_timesteps, beta_start, beta_end)
    elif schedule_type == "cosine":
        _, _, alpha_bar = get_cosine_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    return alpha_bar


def apply_asynchronous_noise(
    z0: torch.Tensor,
    timesteps_composition: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: np.ndarray,
) -> torch.Tensor:
    """
    Applies noise to each frame in the latent batch according to its own timestep.
    Args:
        z0 (torch.Tensor): Clean latents, shape (B, F, ...)
        timesteps_composition (torch.Tensor): Timesteps per frame, shape (B, F), int64
        noise (torch.Tensor): Noise tensor, same shape as z0
        alpha_bar (np.ndarray): Precomputed alpha_bar schedule, shape (T,)
    Returns:
        torch.Tensor: Noisy latents, same shape as z0
    """
    # Gather alpha_bar for each frame
    # timesteps_composition is 0-based, alpha_bar[0] is valid
    device = z0.device
    B, F = timesteps_composition.shape
    alpha_bar_t = torch.from_numpy(alpha_bar).to(device=device, dtype=z0.dtype)[
        timesteps_composition
    ]
    # Reshape for broadcasting
    while alpha_bar_t.dim() < z0.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    noisy_z = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise
    return noisy_z
