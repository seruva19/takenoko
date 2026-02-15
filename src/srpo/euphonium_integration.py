from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch


def resolve_guidance_scale(
    guidance_scale: float,
    guidance_kl_beta: float,
    guidance_eta: float,
) -> float:
    """
    Resolve effective guidance scale.

    - guidance_scale >= 0: use as-is.
    - guidance_scale < 0: use eta^2 / (2 * kl_beta), matching Euphonium-style mode.
    """
    if guidance_scale >= 0.0:
        return guidance_scale

    if guidance_kl_beta <= 0.0:
        raise ValueError(
            "guidance_kl_beta must be > 0 when guidance_scale < 0 "
            f"(got {guidance_kl_beta})."
        )
    if guidance_eta < 0.0:
        raise ValueError(
            "guidance_eta must be >= 0 when guidance_scale < 0 "
            f"(got {guidance_eta})."
        )
    return (guidance_eta * guidance_eta) / (2.0 * guidance_kl_beta)


def should_apply_process_reward_step(
    step_idx: int,
    total_steps: int,
    start_step: int,
    end_step: int,
    interval: int,
) -> bool:
    """Return True when process reward guidance should run at this denoising step."""
    if interval <= 0:
        return False
    if step_idx < start_step:
        return False

    effective_end = total_steps if end_step < 0 else min(end_step, total_steps)
    if step_idx >= effective_end:
        return False

    return ((step_idx - start_step) % interval) == 0


def estimate_spsa_reward_gradient(
    *,
    noisy_latents: torch.Tensor,
    reward_score_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
    num_samples: int,
) -> torch.Tensor:
    """
    Estimate reward gradient using SPSA finite differences in latent space.

    For each sample, draw random perturbation u and estimate:
        grad ~= ((r(x + sigma*u) - r(x - sigma*u)) / (2*sigma)) * u
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0 for SPSA gradient, got {sigma}.")
    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be > 0 for SPSA gradient, got {num_samples}."
        )

    latents = noisy_latents.detach()
    grad_accum = torch.zeros_like(latents, dtype=torch.float32)
    sigma_value = torch.as_tensor(sigma, device=latents.device, dtype=latents.dtype)

    for _ in range(num_samples):
        perturb = torch.randn_like(latents)
        latents_plus = latents + sigma_value * perturb
        latents_minus = latents - sigma_value * perturb

        scores_plus = reward_score_fn(latents_plus).to(torch.float32)
        scores_minus = reward_score_fn(latents_minus).to(torch.float32)
        if scores_plus.ndim != 1 or scores_minus.ndim != 1:
            raise ValueError(
                "SPSA reward_score_fn must return 1D [B] reward tensors."
            )

        reward_delta = (scores_plus - scores_minus).to(
            device=latents.device, dtype=torch.float32
        )
        while reward_delta.ndim < latents.ndim:
            reward_delta = reward_delta.unsqueeze(-1)

        grad_sample = (reward_delta / (2.0 * float(sigma))) * perturb.to(torch.float32)
        grad_accum += grad_sample

    return (grad_accum / float(num_samples)).to(dtype=latents.dtype)


def apply_process_reward_guidance(
    latents: torch.Tensor,
    reward_gradient: Optional[torch.Tensor],
    guidance_scale: float,
    guidance_kl_beta: float,
    guidance_eta: float,
    normalize_gradient: bool,
    use_delta_t_for_guidance: bool,
    delta_t: Optional[torch.Tensor],
) -> torch.Tensor:
    """Inject process-reward guidance into a latent state update."""
    if reward_gradient is None:
        return latents

    resolved_scale = resolve_guidance_scale(
        guidance_scale=guidance_scale,
        guidance_kl_beta=guidance_kl_beta,
        guidance_eta=guidance_eta,
    )
    if resolved_scale == 0.0:
        return latents

    guided_gradient = reward_gradient.to(device=latents.device, dtype=latents.dtype)
    if normalize_gradient:
        reduce_dims = tuple(range(1, guided_gradient.ndim))
        grad_norm = torch.linalg.vector_norm(guided_gradient, dim=reduce_dims, keepdim=True)
        guided_gradient = guided_gradient / (grad_norm + 1e-8)

    scale = torch.as_tensor(resolved_scale, device=latents.device, dtype=latents.dtype)
    if use_delta_t_for_guidance and delta_t is not None:
        scale = scale * delta_t.to(device=latents.device, dtype=latents.dtype).abs()

    return latents + scale * guided_gradient


def compute_process_rewards(
    predicted_clean_latents: torch.Tensor,
    clean_latent_target: torch.Tensor,
    detach_target: bool,
) -> torch.Tensor:
    """
    Compute a latent process reward.

    Reward = negative MSE between one-step predicted clean latent and clean latent target.
    Higher reward means lower reconstruction error at the sampled step.
    """
    target = clean_latent_target.detach() if detach_target else clean_latent_target
    diff = predicted_clean_latents - target
    reduce_dims = tuple(range(1, diff.ndim))
    mse = diff.pow(2).mean(dim=reduce_dims)
    return -mse


def _group_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.numel() < 2:
        return values

    mean = values.mean()
    std = values.std(unbiased=False)
    if not torch.isfinite(std):
        return values - mean
    if std.abs().item() < eps:
        return values - mean
    return (values - mean) / (std + eps)


def combine_dual_reward_signal(
    outcome_rewards: torch.Tensor,
    process_rewards: Optional[torch.Tensor],
    mode: str,
    process_coef: float,
    outcome_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combine outcome and process rewards using Euphonium-style dual reward modes.

    Modes:
    - none: keep baseline SRPO reward behavior.
    - only: use process-reward advantages.
    - both: weighted sum of normalized outcome + process advantages.
    """
    mode = mode.lower()
    if mode not in {"none", "only", "both"}:
        raise ValueError(f"Unsupported dual reward mode: {mode!r}")

    outcome_adv = _group_normalize(outcome_rewards)
    process_adv = _group_normalize(process_rewards) if process_rewards is not None else None
    fallback_used = 0.0

    if mode == "none":
        reward_signal = outcome_rewards
    elif mode == "only":
        if process_adv is None:
            reward_signal = outcome_rewards
            fallback_used = 1.0
        else:
            reward_signal = process_coef * process_adv
    else:
        if process_adv is None:
            reward_signal = outcome_coef * outcome_adv
            fallback_used = 1.0
        else:
            reward_signal = (
                outcome_coef * outcome_adv + process_coef * process_adv
            )

    metrics: Dict[str, float] = {
        "euphonium_outcome_reward_mean": float(outcome_rewards.mean().detach().item()),
        "euphonium_outcome_reward_std": float(outcome_rewards.std(unbiased=False).detach().item())
        if outcome_rewards.numel() > 1
        else 0.0,
        "euphonium_dual_mode_fallback": fallback_used,
    }
    if process_rewards is not None:
        metrics["euphonium_process_reward_mean"] = float(
            process_rewards.mean().detach().item()
        )
        metrics["euphonium_process_reward_std"] = float(
            process_rewards.std(unbiased=False).detach().item()
        ) if process_rewards.numel() > 1 else 0.0
    else:
        metrics["euphonium_process_reward_mean"] = 0.0
        metrics["euphonium_process_reward_std"] = 0.0

    metrics["euphonium_reward_signal_mean"] = float(reward_signal.mean().detach().item())
    metrics["euphonium_reward_signal_std"] = float(
        reward_signal.std(unbiased=False).detach().item()
    ) if reward_signal.numel() > 1 else 0.0
    return reward_signal, metrics
