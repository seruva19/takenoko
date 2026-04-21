from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.logger import get_logger
from criteria.loss_factory import conditional_loss_with_pseudo_huber
from enhancements.soar.algorithm import (
    build_soar_auxiliary_points,
    compute_soar_loss_delta,
)
from utils.train_utils import compute_loss_weighting_for_sd3

logger = get_logger(__name__)


def _loss_kwargs_from_args(args: Any) -> Dict[str, Any]:
    return {
        "fourier_mode": getattr(args, "fourier_mode", "weighted"),
        "fourier_dims": getattr(args, "fourier_dims", (-2, -1)),
        "fourier_eps": getattr(args, "fourier_eps", 1e-8),
        "fourier_multiscale_factors": getattr(
            args,
            "fourier_multiscale_factors",
            [1, 2, 4],
        ),
        "fourier_adaptive_threshold": getattr(args, "fourier_adaptive_threshold", 0.1),
        "fourier_adaptive_alpha": getattr(args, "fourier_adaptive_alpha", 0.5),
        "fourier_high_freq_weight": getattr(args, "fourier_high_freq_weight", 2.0),
        "wavelet_type": getattr(args, "wavelet_type", "haar"),
        "wavelet_levels": getattr(args, "wavelet_levels", 1),
        "wavelet_mode": getattr(args, "wavelet_mode", "zero"),
        "clustered_mse_num_clusters": getattr(args, "clustered_mse_num_clusters", 8),
        "clustered_mse_cluster_weight": getattr(
            args,
            "clustered_mse_cluster_weight",
            1.0,
        ),
        "huber_delta": getattr(args, "huber_delta", 1.0),
        "ew_boundary_shift": getattr(args, "ew_boundary_shift", 0.0),
        "stepped_step_size": getattr(args, "stepped_step_size", 50),
        "stepped_multiplier": getattr(args, "stepped_multiplier", 10.0),
    }


class SoarHelper(nn.Module):
    """Train-time HY-SOAR auxiliary objective helper."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model
        self.args = args

    def setup_hooks(self) -> None:
        return None

    def remove_hooks(self) -> None:
        return None

    def _compute_seq_len(self, latents: torch.Tensor) -> int:
        if latents.dim() == 5:
            lat_f, lat_h, lat_w = latents.shape[2:5]
        elif latents.dim() == 4:
            lat_f = 1
            lat_h, lat_w = latents.shape[2:4]
        else:
            raise ValueError(f"Unsupported latent rank for HY-SOAR: {latents.dim()}")

        patch_size = getattr(self.diffusion_model, "patch_size", (1, 2, 2))
        if len(patch_size) == 3:
            pt, ph, pw = patch_size
        elif len(patch_size) == 2:
            pt = 1
            ph, pw = patch_size
        else:
            raise ValueError(f"Unsupported patch_size for HY-SOAR: {patch_size}")

        return max((lat_f * lat_h * lat_w) // max(int(pt * ph * pw), 1), 1)

    def _extract_context(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[List[torch.Tensor]]:
        context_source = batch.get("t5")
        if context_source is None:
            context_source = batch.get("text_encoder_hidden_states")
        if context_source is None:
            context_source = batch.get("encoder_hidden_states")
        if context_source is None:
            context_source = batch.get("context")
        if context_source is None:
            return None

        if torch.is_tensor(context_source):
            context_source = [context_source]
        elif not isinstance(context_source, list):
            context_source = list(context_source)

        return [tensor.to(device=device, dtype=dtype) for tensor in context_source]

    def _build_unconditional_context(
        self,
        cond_context: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        mode = str(getattr(self.args, "soar_uncond_context_mode", "zero")).lower()
        if mode != "zero":
            raise ValueError(f"Unsupported HY-SOAR unconditional context mode: {mode}")
        return [torch.zeros_like(tensor) for tensor in cond_context]

    def _duplicate_y(self, y_value: Any) -> Any:
        if torch.is_tensor(y_value):
            return torch.cat([y_value, y_value], dim=0)
        return y_value

    def _forward_model(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        y_value: Any,
    ) -> torch.Tensor:
        model_pred_list = self.diffusion_model(
            latents,
            t=timesteps,
            context=context,
            seq_len=seq_len,
            y=y_value,
        )
        return torch.stack(model_pred_list, dim=0)

    def compute_loss(
        self,
        *,
        args: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        batch: Dict[str, Any],
        base_loss: torch.Tensor,
        validation_mode: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = base_loss.new_zeros(())
        zero_metrics = {
            "soar_aux_loss": zero,
            "soar_aux_raw_loss": zero,
            "soar_aux_points": zero,
            "soar_aux_points_per_sample": zero,
            "soar_paths_used": zero,
            "soar_boundary_hit_ratio": zero,
            "soar_sigma_t0_mean": zero,
            "soar_sigma_t1_mean": zero,
            "soar_sigma_tprime_mean": zero,
        }
        if (
            not getattr(args, "enable_soar", False)
            or validation_mode
            or float(getattr(args, "soar_lambda_aux", 0.0)) <= 0.0
            or int(getattr(args, "soar_aux_points_per_path", 0)) <= 0
        ):
            return zero, zero_metrics

        cond_context = self._extract_context(
            batch,
            device=latents.device,
            dtype=network_dtype,
        )
        if not cond_context:
            logger.warning("HY-SOAR skipped: no cached text context found in batch.")
            return zero, zero_metrics

        uncond_context = self._build_unconditional_context(cond_context)
        seq_len = self._compute_seq_len(noisy_model_input)
        y_value = batch.get("y")

        with torch.no_grad():
            rollout_context = [
                torch.cat([uncond_tensor, cond_tensor], dim=0)
                for uncond_tensor, cond_tensor in zip(uncond_context, cond_context)
            ]
            rollout_latents = torch.cat(
                [noisy_model_input.detach(), noisy_model_input.detach()],
                dim=0,
            ).to(device=latents.device, dtype=network_dtype)
            rollout_timesteps = torch.cat([timesteps, timesteps], dim=0).to(
                device=latents.device
            )
            rollout_pred = self._forward_model(
                latents=rollout_latents,
                timesteps=rollout_timesteps,
                context=rollout_context,
                seq_len=seq_len,
                y_value=self._duplicate_y(y_value),
            )
            v_uncond, v_cond = rollout_pred.chunk(2, dim=0)
            rollout_velocity = (
                v_uncond
                + float(getattr(args, "soar_rollout_cfg_scale", 4.5))
                * (v_cond - v_uncond)
            ).detach()

            aux_points, aux_stats = build_soar_auxiliary_points(
                z_t0=noisy_model_input.detach(),
                timesteps=timesteps.detach(),
                v_cfg=rollout_velocity,
                z1=noise.detach().to(device=latents.device, dtype=network_dtype),
                noise_scheduler=noise_scheduler,
                num_rollout_paths=int(getattr(args, "soar_num_rollout_paths", 1)),
                points_per_path=int(getattr(args, "soar_aux_points_per_path", 6)),
                rollout_step_count=int(getattr(args, "soar_rollout_step_count", 40)),
                use_same_noise=bool(getattr(args, "soar_use_same_noise", True)),
                enable_sde_branch=bool(getattr(args, "soar_enable_sde_branch", False)),
                sde_rollout_type=str(
                    getattr(args, "soar_sde_rollout_type", "flow_sde")
                ).lower(),
                sde_noise_scale=float(getattr(args, "soar_sde_noise_scale", 0.5)),
            )

        if not aux_points:
            zero_metrics.update(
                {
                    "soar_paths_used": aux_stats["paths_used"].to(
                        device=base_loss.device,
                        dtype=base_loss.dtype,
                    ),
                    "soar_boundary_hit_ratio": aux_stats["boundary_hit_ratio"].to(
                        device=base_loss.device,
                        dtype=base_loss.dtype,
                    ),
                    "soar_sigma_t0_mean": aux_stats["sigma_t0_mean"].to(
                        device=base_loss.device,
                        dtype=base_loss.dtype,
                    ),
                    "soar_sigma_t1_mean": aux_stats["sigma_t1_mean"].to(
                        device=base_loss.device,
                        dtype=base_loss.dtype,
                    ),
                    "soar_sigma_tprime_mean": aux_stats["sigma_tprime_mean"].to(
                        device=base_loss.device,
                        dtype=base_loss.dtype,
                    ),
                }
            )
            return zero, zero_metrics

        aux_loss_sum = torch.zeros((), device=latents.device, dtype=torch.float32)
        aux_point_count = 0
        world_size = int(getattr(accelerator, "num_processes", 1))

        for point in aux_points:
            sample_indices = point["sample_indices"]
            z_t_prime = point["latents"].to(device=latents.device, dtype=network_dtype)
            sigma_t_prime = point["sigmas"].to(device=latents.device, dtype=torch.float32)
            timesteps_t_prime = point["timesteps"].to(device=latents.device)

            z0_sub = latents.index_select(0, sample_indices).to(
                device=latents.device,
                dtype=network_dtype,
            )
            noise_sub = noise.index_select(0, sample_indices).to(
                device=latents.device,
                dtype=network_dtype,
            )
            cond_sub = [
                context_tensor.index_select(0, sample_indices)
                for context_tensor in cond_context
            ]
            sigma_t_prime_nd = sigma_t_prime
            while sigma_t_prime_nd.dim() < z_t_prime.dim():
                sigma_t_prime_nd = sigma_t_prime_nd.unsqueeze(-1)
            sigma_t_prime_nd = sigma_t_prime_nd.to(
                device=latents.device,
                dtype=network_dtype,
            )
            corrected_target = (z_t_prime - z0_sub) / sigma_t_prime_nd.clamp_min(1e-8)

            point_pred = self._forward_model(
                latents=z_t_prime,
                timesteps=timesteps_t_prime,
                context=cond_sub,
                seq_len=self._compute_seq_len(z_t_prime),
                y_value=(
                    y_value.index_select(0, sample_indices)
                    if torch.is_tensor(y_value)
                    else y_value
                ),
            )
            per_element_loss = conditional_loss_with_pseudo_huber(
                point_pred.to(network_dtype),
                corrected_target.to(network_dtype),
                loss_type=args.loss_type,
                huber_c=args.pseudo_huber_c,
                current_step=getattr(args, "current_step", None),
                total_steps=getattr(args, "total_steps", None),
                schedule_type=args.pseudo_huber_schedule_type,
                c_min=args.pseudo_huber_c_min,
                c_max=args.pseudo_huber_c_max,
                reduction="none",
                timesteps=timesteps_t_prime,
                noise=noise_sub,
                noisy_latents=z_t_prime,
                clean_latents=z0_sub,
                noise_scheduler=noise_scheduler,
                **_loss_kwargs_from_args(args),
            )

            point_weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=str(getattr(args, "weighting_scheme", "none")),
                noise_scheduler=noise_scheduler,
                timesteps=timesteps_t_prime,
                device=latents.device,
                dtype=torch.float32,
            )
            if point_weighting is not None:
                point_weighting = point_weighting.to(
                    device=per_element_loss.device,
                    dtype=per_element_loss.dtype,
                )
                if point_weighting.dim() == per_element_loss.dim() + 1 and point_weighting.shape[2] == 1:
                    point_weighting = point_weighting.squeeze(2)
                while point_weighting.dim() < per_element_loss.dim():
                    point_weighting = point_weighting.unsqueeze(-1)
                per_element_loss = per_element_loss * point_weighting

            sample_weights = batch.get("weight")
            if sample_weights is not None:
                sample_weights = sample_weights.index_select(0, sample_indices).to(
                    device=per_element_loss.device,
                    dtype=per_element_loss.dtype,
                )
                while sample_weights.dim() < per_element_loss.dim():
                    sample_weights = sample_weights.unsqueeze(-1)
                per_element_loss = per_element_loss * sample_weights

            per_sample_loss = per_element_loss.reshape(per_element_loss.shape[0], -1).mean(dim=1)
            aux_loss_sum = aux_loss_sum + per_sample_loss.to(torch.float32).sum()
            aux_point_count += int(per_sample_loss.shape[0])

        if aux_point_count < 1:
            return zero, zero_metrics

        lambda_aux = float(getattr(args, "soar_lambda_aux", 1.0))
        aux_delta = compute_soar_loss_delta(
            base_loss_mean=base_loss,
            base_count=int(latents.shape[0]),
            aux_loss_sum=aux_loss_sum,
            aux_count=aux_point_count,
            lambda_aux=lambda_aux,
            world_size=world_size,
        )
        aux_loss_mean = aux_loss_sum / float(max(aux_point_count, 1))
        metrics = {
            "soar_aux_loss": (aux_loss_mean * lambda_aux).to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_aux_raw_loss": aux_loss_mean.to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_aux_points": base_loss.new_tensor(float(aux_point_count)),
            "soar_aux_points_per_sample": base_loss.new_tensor(
                float(aux_point_count) / float(max(int(latents.shape[0]), 1))
            ),
            "soar_paths_used": aux_stats["paths_used"].to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_boundary_hit_ratio": aux_stats["boundary_hit_ratio"].to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_sigma_t0_mean": aux_stats["sigma_t0_mean"].to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_sigma_t1_mean": aux_stats["sigma_t1_mean"].to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
            "soar_sigma_tprime_mean": aux_stats["sigma_tprime_mean"].to(
                device=base_loss.device,
                dtype=base_loss.dtype,
            ),
        }
        return aux_delta, metrics
