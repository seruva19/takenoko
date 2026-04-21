from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch


def _expand_like(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    expanded = values
    while expanded.dim() < reference.dim():
        expanded = expanded.unsqueeze(-1)
    return expanded.to(device=reference.device, dtype=reference.dtype)


def timesteps_to_normalized_t(
    timesteps: torch.Tensor,
    noise_scheduler: Any,
) -> torch.Tensor:
    """Map scheduler timesteps to normalized SOAR t in [0, 1]."""

    scheduler_timesteps = noise_scheduler.timesteps.to(
        device=timesteps.device,
        dtype=torch.float32,
    )
    flat_timesteps = timesteps.reshape(-1).to(torch.float32)
    nearest_idx = torch.argmin(
        (scheduler_timesteps.unsqueeze(0) - flat_timesteps.unsqueeze(1)).abs(),
        dim=1,
    )
    denom = max(int(scheduler_timesteps.numel()) - 1, 1)
    normalized = 1.0 - nearest_idx.to(torch.float32) / float(denom)
    return normalized.reshape(timesteps.shape).clamp(0.0, 1.0)


def normalized_t_to_scheduler_sigma(
    normalized_t: torch.Tensor,
    noise_scheduler: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map normalized t in [0, 1] back to scheduler sigma and discrete timestep."""

    scheduler_sigmas = noise_scheduler.sigmas.to(
        device=normalized_t.device,
        dtype=torch.float32,
    )
    scheduler_timesteps = noise_scheduler.timesteps.to(device=normalized_t.device)
    max_index = max(int(scheduler_timesteps.numel()) - 1, 0)
    if max_index == 0:
        indices = torch.zeros_like(normalized_t, dtype=torch.long)
    else:
        indices = torch.round(
            (1.0 - normalized_t.to(torch.float32)) * float(max_index)
        ).long()
        indices = indices.clamp(0, max_index)
    sigmas = scheduler_sigmas.index_select(0, indices.reshape(-1)).reshape(
        normalized_t.shape
    )
    timesteps = scheduler_timesteps.index_select(0, indices.reshape(-1)).reshape(
        normalized_t.shape
    )
    return sigmas, timesteps


def scheduler_sigma_to_normalized_t(
    sigmas: torch.Tensor,
    noise_scheduler: Any,
) -> torch.Tensor:
    """Map scheduler sigmas back to normalized SOAR t in [0, 1]."""

    scheduler_sigmas = noise_scheduler.sigmas.to(
        device=sigmas.device,
        dtype=torch.float32,
    )
    flat_sigmas = sigmas.reshape(-1).to(torch.float32)
    nearest_idx = torch.argmin(
        (scheduler_sigmas.unsqueeze(0) - flat_sigmas.unsqueeze(1)).abs(),
        dim=1,
    )
    denom = max(int(scheduler_sigmas.numel()) - 1, 1)
    normalized = 1.0 - nearest_idx.to(torch.float32) / float(denom)
    return normalized.reshape(sigmas.shape).clamp(0.0, 1.0)


def stochastic_rollout_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma_curr: torch.Tensor,
    sigma_next: torch.Tensor,
    sde_rollout_type: str,
    sde_noise_scale: float,
    sigma_max_value: float,
) -> torch.Tensor:
    """Apply one stochastic SOAR rollout step."""

    sample_f32 = sample.to(torch.float32)
    velocity_f32 = velocity.to(torch.float32)
    sigma_curr_nd = _expand_like(sigma_curr.to(torch.float32), sample_f32)
    sigma_next_nd = _expand_like(sigma_next.to(torch.float32), sample_f32)
    dt = sigma_next_nd - sigma_curr_nd

    if sde_rollout_type == "simple":
        next_sample = sample_f32 + velocity_f32 * dt
        next_sample = next_sample + float(sde_noise_scale) * torch.sqrt(
            dt.abs().clamp_min(0.0)
        ) * torch.randn_like(sample_f32)
    elif sde_rollout_type == "sde":
        pred_original_sample = sample_f32 - sigma_curr_nd * velocity_f32
        next_sample = (1.0 - sigma_next_nd) * pred_original_sample
        next_sample = next_sample + sigma_next_nd * torch.randn_like(sample_f32)
    elif sde_rollout_type == "flow_sde":
        sigma_curr_safe = torch.where(
            sigma_curr_nd == 1.0,
            torch.full_like(sigma_curr_nd, float(sigma_max_value)),
            sigma_curr_nd,
        )
        std_dev_t = (
            torch.sqrt(
                sigma_curr_nd.clamp_min(1e-8)
                / (1.0 - sigma_curr_safe).clamp_min(1e-8)
            )
            * float(sde_noise_scale)
        )
        prev_sample_mean = sample_f32 * (
            1.0 + (std_dev_t.square() / (2.0 * sigma_curr_nd.clamp_min(1e-8))) * dt
        )
        prev_sample_mean = prev_sample_mean + velocity_f32 * (
            1.0
            + (
                std_dev_t.square()
                * (1.0 - sigma_curr_nd)
                / (2.0 * sigma_curr_nd.clamp_min(1e-8))
            )
        ) * dt
        next_sample = prev_sample_mean + std_dev_t * torch.sqrt(
            (-dt).clamp_min(0.0)
        ) * torch.randn_like(sample_f32)
    elif sde_rollout_type == "cps":
        std_dev_t = sigma_next_nd * math.sin(float(sde_noise_scale) * math.pi / 2.0)
        pred_original_sample = sample_f32 - sigma_curr_nd * velocity_f32
        noise_estimate = sample_f32 + velocity_f32 * (1.0 - sigma_curr_nd)
        prev_sample_mean = pred_original_sample * (1.0 - sigma_next_nd)
        prev_sample_mean = prev_sample_mean + noise_estimate * torch.sqrt(
            (sigma_next_nd.square() - std_dev_t.square()).clamp_min(0.0)
        )
        next_sample = prev_sample_mean + std_dev_t * torch.randn_like(sample_f32)
    else:
        raise ValueError(f"Unsupported sde_rollout_type: {sde_rollout_type}")

    return next_sample.to(dtype=sample.dtype)


def compute_soar_loss_delta(
    *,
    base_loss_mean: torch.Tensor,
    base_count: int,
    aux_loss_sum: torch.Tensor,
    aux_count: int,
    lambda_aux: float,
    world_size: int,
) -> torch.Tensor:
    """Convert paper-faithful SOAR normalization into an additive delta term."""

    if base_count < 1 or aux_count < 1 or lambda_aux <= 0.0:
        return base_loss_mean.new_zeros(())

    local_main_count = base_loss_mean.new_tensor(float(base_count), dtype=torch.float32)
    local_aux_count = base_loss_mean.new_tensor(float(aux_count), dtype=torch.float32)

    global_main_count = local_main_count.detach().clone()
    global_aux_count = local_aux_count.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_main_count, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(global_aux_count, op=torch.distributed.ReduceOp.SUM)

    total_count = torch.clamp(
        global_main_count + float(lambda_aux) * global_aux_count,
        min=1.0,
    )
    local_main_sum = base_loss_mean.to(torch.float32) * float(base_count)
    combined_objective = (
        local_main_sum + float(lambda_aux) * aux_loss_sum.to(torch.float32)
    ) / total_count
    combined_objective = combined_objective * float(max(int(world_size), 1))
    return combined_objective.to(dtype=base_loss_mean.dtype) - base_loss_mean


@torch.no_grad()
def build_soar_auxiliary_points(
    *,
    z_t0: torch.Tensor,
    timesteps: torch.Tensor,
    v_cfg: torch.Tensor,
    z1: torch.Tensor,
    noise_scheduler: Any,
    num_rollout_paths: int,
    points_per_path: int,
    rollout_step_count: int,
    use_same_noise: bool = True,
    enable_sde_branch: bool = False,
    sde_rollout_type: str = "flow_sde",
    sde_noise_scale: float = 0.5,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Construct detached SOAR auxiliary points from one-step rollout branches."""

    if num_rollout_paths < 1 or points_per_path < 1:
        empty_stats = {
            "boundary_hit_ratio": z_t0.new_zeros((), dtype=torch.float32),
            "sigma_t0_mean": z_t0.new_zeros((), dtype=torch.float32),
            "sigma_t1_mean": z_t0.new_zeros((), dtype=torch.float32),
            "sigma_tprime_mean": z_t0.new_zeros((), dtype=torch.float32),
            "paths_used": z_t0.new_zeros((), dtype=torch.float32),
            "aux_point_count": z_t0.new_zeros((), dtype=torch.float32),
        }
        return [], empty_stats

    batch_size = int(z_t0.shape[0])
    device = z_t0.device
    dtype = z_t0.dtype

    t0 = timesteps_to_normalized_t(timesteps, noise_scheduler)
    sigma_t0, _ = normalized_t_to_scheduler_sigma(t0, noise_scheduler)
    t1 = (t0.detach() - 1.0 / float(max(int(rollout_step_count), 1))).clamp_min(0.0)
    sigma_t1, timestep_t1 = normalized_t_to_scheduler_sigma(t1, noise_scheduler)
    hit_clean_boundary = t1 <= 0.0

    scheduler_sigmas = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
    sigma_max_value = float(
        scheduler_sigmas[1].item() if scheduler_sigmas.numel() > 1 else scheduler_sigmas[0].item()
    )
    full_indices = torch.arange(batch_size, device=device, dtype=torch.long)
    sigma_tprime_values: List[torch.Tensor] = []
    aux_points: List[Dict[str, torch.Tensor]] = []

    def _append_interpolated_points(
        *,
        path_indices: torch.Tensor,
        z_t1_prime: torch.Tensor,
        sigma_t1_sub: torch.Tensor,
        sigma_upper_sub: torch.Tensor,
        path_index: int,
    ) -> None:
        if path_indices.numel() == 0:
            return

        rand_fracs = torch.rand(
            points_per_path,
            path_indices.shape[0],
            device=device,
            dtype=sigma_t1_sub.dtype,
        )
        sigma_targets = sigma_t1_sub.unsqueeze(0) + rand_fracs * (
            sigma_upper_sub.unsqueeze(0) - sigma_t1_sub.unsqueeze(0)
        )

        for point_index in range(points_per_path):
            sigma_t_prime = sigma_targets[point_index].detach()
            interp_denom = (1.0 - sigma_t1_sub).clamp_min(1e-8)
            alpha = ((sigma_t_prime - sigma_t1_sub) / interp_denom).clamp(0.0, 1.0)
            alpha_nd = _expand_like(alpha, z_t1_prime)

            if use_same_noise:
                z1_sub = z1.index_select(0, path_indices)
            else:
                z1_sub = torch.randn_like(z_t1_prime)

            z_interp = ((1.0 - alpha_nd) * z_t1_prime + alpha_nd * z1_sub).detach()
            t_prime = scheduler_sigma_to_normalized_t(sigma_t_prime, noise_scheduler)
            _, aux_timesteps = normalized_t_to_scheduler_sigma(t_prime, noise_scheduler)
            sigma_tprime_values.append(sigma_t_prime)
            aux_points.append(
                {
                    "sample_indices": path_indices.detach(),
                    "latents": z_interp,
                    "sigmas": sigma_t_prime,
                    "timesteps": aux_timesteps.detach(),
                    "path_index": torch.full(
                        (path_indices.shape[0],),
                        int(path_index),
                        device=device,
                        dtype=torch.long,
                    ),
                }
            )

    sigma_t0_nd = _expand_like(sigma_t0, z_t0)
    sigma_t1_nd = _expand_like(sigma_t1, z_t0)
    z_t1_prime_ode = (z_t0 + v_cfg * (sigma_t1_nd - sigma_t0_nd)).detach()
    sigma_upper = torch.ones_like(sigma_t1)
    _append_interpolated_points(
        path_indices=full_indices,
        z_t1_prime=z_t1_prime_ode,
        sigma_t1_sub=sigma_t1.detach(),
        sigma_upper_sub=sigma_upper.detach(),
        path_index=0,
    )

    active_sde_indices = (~hit_clean_boundary).nonzero(as_tuple=False).reshape(-1)
    use_sde = enable_sde_branch and num_rollout_paths > 1 and active_sde_indices.numel() > 0
    if use_sde:
        z_t0_sde = z_t0.index_select(0, active_sde_indices)
        v_cfg_sde = v_cfg.index_select(0, active_sde_indices)
        sigma_t0_sde = sigma_t0.index_select(0, active_sde_indices).detach()
        sigma_t1_sde = sigma_t1.index_select(0, active_sde_indices).detach()
        sigma_upper_sde = sigma_upper.index_select(0, active_sde_indices).detach()

        for path_index in range(1, int(num_rollout_paths)):
            z_t1_prime_sde = stochastic_rollout_step(
                sample=z_t0_sde,
                velocity=v_cfg_sde,
                sigma_curr=sigma_t0_sde,
                sigma_next=sigma_t1_sde,
                sde_rollout_type=sde_rollout_type,
                sde_noise_scale=sde_noise_scale,
                sigma_max_value=sigma_max_value,
            ).detach()
            _append_interpolated_points(
                path_indices=active_sde_indices,
                z_t1_prime=z_t1_prime_sde,
                sigma_t1_sub=sigma_t1_sde,
                sigma_upper_sub=sigma_upper_sde,
                path_index=path_index,
            )

    sigma_tprime_mean = (
        torch.cat(sigma_tprime_values, dim=0).mean()
        if sigma_tprime_values
        else z_t0.new_zeros((), dtype=torch.float32)
    )
    path_count = 1
    if use_sde:
        path_count = int(num_rollout_paths)

    stats = {
        "boundary_hit_ratio": hit_clean_boundary.to(torch.float32).mean(),
        "sigma_t0_mean": sigma_t0.detach().to(torch.float32).mean(),
        "sigma_t1_mean": sigma_t1.detach().to(torch.float32).mean(),
        "sigma_tprime_mean": sigma_tprime_mean.detach().to(torch.float32),
        "paths_used": z_t0.new_tensor(float(path_count), dtype=torch.float32),
        "aux_point_count": z_t0.new_tensor(float(len(aux_points)), dtype=torch.float32),
        "timestep_t1": timestep_t1.detach().to(torch.float32).mean(),
    }
    return aux_points, stats
