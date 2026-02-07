"""Temporal-Conditional LoRA network module for WAN training.

This module extends base LoRA with optional FiLM-style conditioning on the
LoRA down-projected activations. The default configuration keeps the feature
disabled to preserve baseline behavior.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from common.logger import get_logger
from networks.lora_wan import (
    LoRAModule,
    LoRANetwork,
    WAN_TARGET_REPLACE_MODULES,
    create_arch_network_from_weights as create_base_arch_network_from_weights,
)

logger = get_logger(__name__)


class TCLoRAModule(LoRAModule):
    """LoRA module with optional temporal conditioning modulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tc_enabled: bool = False
        self.tc_use_timestep: bool = True
        self.tc_condition_dim: int = 64
        self.tc_hidden_dim: int = 128
        self.tc_modulation_scale: float = 1.0
        self.tc_allow_sequence_condition: bool = True
        self.tc_timestep_max: int = 1000
        self._tc_condition: Optional[torch.Tensor] = None
        self._tc_warned_condition_dim_mismatch: bool = False
        self.tc_gamma: Optional[nn.Sequential] = None
        self.tc_beta: Optional[nn.Sequential] = None

    def configure_tc(
        self,
        enabled: bool,
        use_timestep: bool,
        condition_dim: int,
        hidden_dim: int,
        modulation_scale: float,
        allow_sequence_condition: bool,
        timestep_max: int,
    ) -> None:
        self.tc_enabled = bool(enabled)
        self.tc_use_timestep = bool(use_timestep)
        self.tc_condition_dim = max(1, int(condition_dim))
        self.tc_hidden_dim = max(1, int(hidden_dim))
        self.tc_modulation_scale = max(0.0, float(modulation_scale))
        self.tc_allow_sequence_condition = bool(allow_sequence_condition)
        self.tc_timestep_max = max(1, int(timestep_max))

        if not self.tc_enabled:
            self.clear_tc_condition()
            return

        if self.tc_gamma is not None and self.tc_beta is not None:
            return

        self.tc_gamma = nn.Sequential(
            nn.Linear(self.tc_condition_dim, self.tc_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.tc_hidden_dim, self.lora_dim, bias=True),
        )
        self.tc_beta = nn.Sequential(
            nn.Linear(self.tc_condition_dim, self.tc_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.tc_hidden_dim, self.lora_dim, bias=True),
        )
        # Start from identity-like behavior.
        nn.init.zeros_(self.tc_gamma[-1].weight)
        nn.init.zeros_(self.tc_gamma[-1].bias)
        nn.init.zeros_(self.tc_beta[-1].weight)
        nn.init.zeros_(self.tc_beta[-1].bias)

    @staticmethod
    def _sinusoidal_timestep_embedding(
        timesteps: torch.Tensor, dim: int
    ) -> torch.Tensor:
        if dim <= 0:
            raise ValueError("embedding dimension must be > 0")

        timesteps = timesteps.to(dtype=torch.float32)
        half_dim = dim // 2
        if half_dim == 0:
            return timesteps[:, None]

        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * exponent
        )
        angles = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros((emb.shape[0], 1), device=emb.device)], dim=-1
            )
        return emb

    def clear_tc_condition(self) -> None:
        self._tc_condition = None

    def set_external_condition(self, condition: Optional[torch.Tensor]) -> None:
        if not self.tc_enabled:
            self._tc_condition = None
            return
        if condition is None:
            self._tc_condition = None
            return
        if condition.dim() not in (2, 3):
            self._tc_condition = None
            return
        if condition.shape[-1] != self.tc_condition_dim:
            if not self._tc_warned_condition_dim_mismatch:
                logger.warning(
                    "TC-LoRA condition dim mismatch for %s: expected %s, got %s. Skipping conditioning.",
                    self.lora_name,
                    self.tc_condition_dim,
                    condition.shape[-1],
                )
                self._tc_warned_condition_dim_mismatch = True
            self._tc_condition = None
            return
        self._tc_condition = condition

    def set_timestep_condition(
        self, timesteps: Optional[torch.Tensor], max_timestep: int
    ) -> None:
        if not self.tc_enabled or not self.tc_use_timestep or timesteps is None:
            self._tc_condition = None
            return

        t = timesteps.detach()
        if t.dim() == 0:
            t = t.reshape(1)
        else:
            t = t.reshape(-1)

        t = t.to(dtype=torch.float32)
        denom = float(max(1, int(max_timestep)))
        t = torch.clamp(t / denom, 0.0, 1.0)
        self._tc_condition = self._sinusoidal_timestep_embedding(
            t, self.tc_condition_dim
        )

    def _apply_temporal_modulation(self, lx: torch.Tensor) -> torch.Tensor:
        if (
            not self.tc_enabled
            or self.tc_modulation_scale <= 0.0
            or self._tc_condition is None
            or self.tc_gamma is None
            or self.tc_beta is None
        ):
            return lx

        cond = self._tc_condition.to(device=lx.device, dtype=lx.dtype)

        if cond.dim() == 2:
            if cond.shape[0] == 1 and lx.shape[0] > 1:
                cond = cond.expand(lx.shape[0], -1)
            if cond.shape[0] != lx.shape[0]:
                return lx
            gamma = self.tc_gamma(cond) + 1.0
            beta = self.tc_beta(cond)
            gamma = 1.0 + (gamma - 1.0) * self.tc_modulation_scale
            beta = beta * self.tc_modulation_scale
            if lx.dim() == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            elif lx.dim() == 4:
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
            return lx * gamma + beta

        if cond.dim() == 3 and lx.dim() == 3 and self.tc_allow_sequence_condition:
            if cond.shape[0] != lx.shape[0]:
                return lx
            num_tokens = lx.shape[1]
            temporal_steps = cond.shape[1]
            if temporal_steps <= 0:
                return lx
            if temporal_steps == 1:
                cond_global = cond[:, 0, :]
                gamma = self.tc_gamma(cond_global) + 1.0
                beta = self.tc_beta(cond_global)
                gamma = 1.0 + (gamma - 1.0) * self.tc_modulation_scale
                beta = beta * self.tc_modulation_scale
                return lx * gamma.unsqueeze(1) + beta.unsqueeze(1)

            token_idx = torch.arange(num_tokens, device=lx.device)
            temporal_idx = torch.clamp(
                (token_idx * temporal_steps) // max(num_tokens, 1),
                min=0,
                max=temporal_steps - 1,
            )
            cond_tokens = cond[:, temporal_idx, :]
            gamma = self.tc_gamma(cond_tokens) + 1.0
            beta = self.tc_beta(cond_tokens)
            gamma = 1.0 + (gamma - 1.0) * self.tc_modulation_scale
            beta = beta * self.tc_modulation_scale
            return lx * gamma + beta

        # Fallback to global modulation for unsupported layout combinations.
        cond_global = cond.mean(dim=1)
        gamma = self.tc_gamma(cond_global) + 1.0
        beta = self.tc_beta(cond_global)
        gamma = 1.0 + (gamma - 1.0) * self.tc_modulation_scale
        beta = beta * self.tc_modulation_scale
        if lx.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        elif lx.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return lx * gamma + beta

    def forward(self, x):  # type: ignore[override]
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            if self.rank_dropout is not None and self.training:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                )
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                lx = lx * mask
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            else:
                scale = self.scale

            lx = self._apply_temporal_modulation(lx)
            lx = self.lora_up(lx)

            if (
                self.training
                and self._ggpo_enabled
                and self._org_module_shape is not None
                and self._perturbation_norm_factor is not None
                and self.combined_weight_norms is not None
                and self.grad_norms is not None
            ):
                try:
                    with torch.no_grad():
                        sigma = float(self.ggpo_sigma)  # type: ignore[arg-type]
                        beta = float(self.ggpo_beta)  # type: ignore[arg-type]
                        perturbation_scale = (
                            sigma * self.combined_weight_norms + beta * self.grad_norms
                        ).to(device=self.device, dtype=torch.float32)
                        perturbation_scale = (
                            perturbation_scale * float(self._perturbation_norm_factor)
                        )
                        pert = torch.randn(
                            self._org_module_shape,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        pert = pert * perturbation_scale.to(self.dtype)
                    perturbation_out = torch.nn.functional.linear(x, pert)
                    return org_forwarded + lx * self.multiplier * scale + perturbation_out
                except Exception:
                    return org_forwarded + lx * self.multiplier * scale

            return org_forwarded + lx * self.multiplier * scale

        lxs = [lora_down(x) for lora_down in self.lora_down]  # type: ignore

        if self.dropout is not None and self.training:
            lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

        if self.rank_dropout is not None and self.training:
            masks = [
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
                for lx in lxs
            ]
            for i in range(len(lxs)):
                if len(lxs[i].size()) == 3:
                    masks[i] = masks[i].unsqueeze(1)
                elif len(lxs[i].size()) == 4:
                    masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                lxs[i] = lxs[i] * masks[i]
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lxs = [self._apply_temporal_modulation(lx) for lx in lxs]
        lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore
        return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class TCLoRANetwork(LoRANetwork):
    """LoRA network that supports optional temporal conditioning modulation."""

    def __init__(
        self,
        *args,
        tc_lora_enabled: bool = False,
        tc_lora_use_timestep: bool = True,
        tc_lora_condition_dim: int = 64,
        tc_lora_hidden_dim: int = 128,
        tc_lora_modulation_scale: float = 1.0,
        tc_lora_allow_sequence_condition: bool = True,
        tc_lora_use_motion_condition: bool = False,
        tc_lora_motion_include_timestep: bool = True,
        tc_lora_motion_frame_dim: int = 2,
        tc_lora_timestep_max: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(*args, module_class=TCLoRAModule, **kwargs)
        self.tc_lora_enabled = bool(tc_lora_enabled)
        self.tc_lora_use_timestep = bool(tc_lora_use_timestep)
        self.tc_lora_condition_dim = int(tc_lora_condition_dim)
        self.tc_lora_hidden_dim = int(tc_lora_hidden_dim)
        self.tc_lora_modulation_scale = float(tc_lora_modulation_scale)
        self.tc_lora_allow_sequence_condition = bool(tc_lora_allow_sequence_condition)
        self.tc_lora_use_motion_condition = bool(tc_lora_use_motion_condition)
        self.tc_lora_motion_include_timestep = bool(tc_lora_motion_include_timestep)
        self.tc_lora_motion_frame_dim = max(0, int(tc_lora_motion_frame_dim))
        self.tc_lora_timestep_max = int(tc_lora_timestep_max)
        self._configure_modules()

    def _configure_modules(self) -> None:
        for lora in getattr(self, "unet_loras", []):
            if isinstance(lora, TCLoRAModule):
                lora.configure_tc(
                    enabled=self.tc_lora_enabled,
                    use_timestep=self.tc_lora_use_timestep,
                    condition_dim=self.tc_lora_condition_dim,
                    hidden_dim=self.tc_lora_hidden_dim,
                    modulation_scale=self.tc_lora_modulation_scale,
                    allow_sequence_condition=self.tc_lora_allow_sequence_condition,
                    timestep_max=self.tc_lora_timestep_max,
                )

    @torch.no_grad()
    def set_tc_lora_timestep(
        self, timesteps: torch.Tensor, max_timestep: Optional[int] = None
    ) -> None:
        max_t = int(max_timestep or self.tc_lora_timestep_max)
        for lora in getattr(self, "unet_loras", []):
            if isinstance(lora, TCLoRAModule):
                lora.set_timestep_condition(timesteps, max_t)

    @torch.no_grad()
    def set_tc_lora_condition(self, condition: Optional[torch.Tensor]) -> None:
        for lora in getattr(self, "unet_loras", []):
            if isinstance(lora, TCLoRAModule):
                lora.set_external_condition(condition)

    @torch.no_grad()
    def clear_tc_lora_condition(self) -> None:
        for lora in getattr(self, "unet_loras", []):
            if isinstance(lora, TCLoRAModule):
                lora.clear_tc_condition()

    @staticmethod
    def _scalar_to_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
        if dim <= 0:
            shape = tuple(values.shape) + (0,)
            return torch.zeros(shape, device=values.device, dtype=torch.float32)
        flat = values.reshape(-1)
        emb = TCLoRAModule._sinusoidal_timestep_embedding(flat, dim)
        return emb.reshape(*values.shape, dim)

    def _build_motion_condition(
        self,
        latents: torch.Tensor,
        timesteps: Optional[torch.Tensor],
        max_timestep: int,
    ) -> Optional[torch.Tensor]:
        if latents.dim() < 4:
            return None

        if latents.dim() == 4:
            latents = latents.unsqueeze(0)

        if latents.dim() < 5:
            return None

        frame_dim = min(max(1, self.tc_lora_motion_frame_dim), latents.dim() - 1)
        if latents.shape[frame_dim] <= 1 and latents.dim() > 2 and latents.shape[2] > 1:
            frame_dim = 2

        latent_seq = torch.movedim(latents.to(torch.float32), frame_dim, 1)
        if latent_seq.shape[1] <= 1:
            return None

        deltas = latent_seq[:, 1:, ...] - latent_seq[:, :-1, ...]
        motion_mag = deltas.abs().flatten(2).mean(dim=-1)
        motion_signed = deltas.flatten(2).mean(dim=-1)
        motion_mag = torch.log1p(motion_mag.clamp_min(0.0))
        motion_signed = torch.tanh(motion_signed)

        cond_dim = max(1, int(self.tc_lora_condition_dim))
        mag_dim = max(1, cond_dim // 2)
        signed_dim = cond_dim - mag_dim

        mag_emb = self._scalar_to_embedding(motion_mag, mag_dim)
        if signed_dim > 0:
            signed_emb = self._scalar_to_embedding(motion_signed, signed_dim)
            motion_cond = torch.cat([mag_emb, signed_emb], dim=-1)
        else:
            motion_cond = mag_emb

        motion_global = self._scalar_to_embedding(motion_mag.mean(dim=1), cond_dim)
        motion_cond = motion_cond + 0.5 * motion_global.unsqueeze(1)

        if self.tc_lora_motion_include_timestep and timesteps is not None:
            t = timesteps.detach().reshape(-1).to(
                device=motion_cond.device, dtype=torch.float32
            )
            batch_size = motion_cond.shape[0]
            if t.numel() == 1 and batch_size > 1:
                t = t.expand(batch_size)
            if t.numel() == batch_size:
                t = torch.clamp(t / float(max(1, int(max_timestep))), 0.0, 1.0)
                t_emb = TCLoRAModule._sinusoidal_timestep_embedding(t, cond_dim)
                motion_cond = motion_cond + t_emb.unsqueeze(1)

        return motion_cond

    @torch.no_grad()
    def set_tc_lora_runtime_condition(
        self,
        latents: Optional[torch.Tensor],
        timesteps: Optional[torch.Tensor],
        max_timestep: Optional[int] = None,
    ) -> None:
        max_t = int(max_timestep or self.tc_lora_timestep_max)
        if not self.tc_lora_enabled:
            self.clear_tc_lora_condition()
            return

        motion_condition: Optional[torch.Tensor] = None
        if self.tc_lora_use_motion_condition and latents is not None:
            try:
                motion_condition = self._build_motion_condition(
                    latents=latents,
                    timesteps=timesteps,
                    max_timestep=max_t,
                )
            except Exception as exc:
                logger.debug("TC-LoRA motion condition build failed: %s", exc)
                motion_condition = None

        if motion_condition is not None:
            self.set_tc_lora_condition(motion_condition)
            return

        if self.tc_lora_use_timestep and timesteps is not None:
            self.set_tc_lora_timestep(timesteps, max_timestep=max_t)
            return

        self.clear_tc_lora_condition()


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _parse_tc_lora_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["tc_lora_enabled"] = _parse_bool(kwargs.get("tc_lora_enabled", False), False)
    out["tc_lora_use_timestep"] = _parse_bool(
        kwargs.get("tc_lora_use_timestep", True), True
    )
    try:
        out["tc_lora_condition_dim"] = int(kwargs.get("tc_lora_condition_dim", 64))
    except Exception:
        out["tc_lora_condition_dim"] = 64
    try:
        out["tc_lora_hidden_dim"] = int(kwargs.get("tc_lora_hidden_dim", 128))
    except Exception:
        out["tc_lora_hidden_dim"] = 128
    try:
        out["tc_lora_modulation_scale"] = float(
            kwargs.get("tc_lora_modulation_scale", 1.0)
        )
    except Exception:
        out["tc_lora_modulation_scale"] = 1.0
    out["tc_lora_allow_sequence_condition"] = _parse_bool(
        kwargs.get("tc_lora_allow_sequence_condition", True), True
    )
    out["tc_lora_use_motion_condition"] = _parse_bool(
        kwargs.get("tc_lora_use_motion_condition", False), False
    )
    out["tc_lora_motion_include_timestep"] = _parse_bool(
        kwargs.get("tc_lora_motion_include_timestep", True), True
    )
    try:
        out["tc_lora_motion_frame_dim"] = int(kwargs.get("tc_lora_motion_frame_dim", 2))
    except Exception:
        out["tc_lora_motion_frame_dim"] = 2
    try:
        out["tc_lora_timestep_max"] = int(kwargs.get("tc_lora_timestep_max", 1000))
    except Exception:
        out["tc_lora_timestep_max"] = 1000
    return out


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    tc_kwargs = _parse_tc_lora_kwargs(dict(kwargs))

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        try:
            exclude_patterns = ast.literal_eval(exclude_patterns)
        except Exception:
            pass
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        try:
            include_patterns = ast.literal_eval(include_patterns)
        except Exception:
            pass

    network = TCLoRANetwork(
        WAN_TARGET_REPLACE_MODULES,
        "tc_lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim if network_dim is not None else 4,
        alpha=network_alpha if network_alpha is not None else 1.0,
        dropout=neuron_dropout,
        rank_dropout=(
            float(kwargs.get("rank_dropout", 0.0))
            if kwargs.get("rank_dropout", None) is not None
            else None
        ),
        module_dropout=(
            float(kwargs.get("module_dropout", 0.0))
            if kwargs.get("module_dropout", None) is not None
            else None
        ),
        conv_lora_dim=(
            int(kwargs.get("conv_dim", 0))
            if kwargs.get("conv_dim", None) is not None
            else None
        ),
        conv_alpha=(
            float(kwargs.get("conv_alpha", 0.0))
            if kwargs.get("conv_alpha", None) is not None
            else None
        ),
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=bool(kwargs.get("verbose", False)),
        **tc_kwargs,
    )
    logger.info(
        "TC-LoRA network created (enabled=%s, use_timestep=%s, use_motion=%s, cond_dim=%s, hidden_dim=%s, scale=%s).",
        network.tc_lora_enabled,
        network.tc_lora_use_timestep,
        network.tc_lora_use_motion_condition,
        network.tc_lora_condition_dim,
        network.tc_lora_hidden_dim,
        network.tc_lora_modulation_scale,
    )
    return network


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    **kwargs,
):
    return create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    # For inference-time merge paths, reuse stable base behavior.
    if for_inference:
        return create_base_arch_network_from_weights(
            multiplier,
            weights_sd,
            text_encoders=text_encoders,
            unet=unet,
            for_inference=for_inference,
            **kwargs,
        )

    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, torch.Tensor] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = int(value.shape[0])

    tc_kwargs = _parse_tc_lora_kwargs(dict(kwargs))
    network = TCLoRANetwork(
        WAN_TARGET_REPLACE_MODULES,
        "tc_lora_unet",
        text_encoders,  # type: ignore[arg-type]
        unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        lora_dim=1,
        alpha=1.0,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        **tc_kwargs,
    )
    return network
