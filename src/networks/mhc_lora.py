"""
mHC-LoRA: Manifold-Constrained Hyper-Connections style LoRA for Takenoko.

Implements multi-path LoRA with a doubly-stochastic mixing matrix (Sinkhorn)
to preserve an identity-like residual while allowing cross-path mixing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from common.logger import get_logger
from networks.lora_wan import LoRANetwork, WAN_TARGET_REPLACE_MODULES
from configs.mhc_config import parse_mhc_config

logger = get_logger(__name__)



class MhcLoRAModule(nn.Module):
    """LoRA module with multiple paths mixed via a doubly stochastic matrix."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        split_dims: Optional[List[int]] = None,
        # mHC-specific parameters
        mhc_num_paths: int = 2,
        mhc_sinkhorn_iters: int = 20,
        mhc_mixing_init: str = "identity",
        mhc_mixing_strength: float = 1.0,
        mhc_mixing_strength_end: Optional[float] = None,
        mhc_mixing_temperature: float = 1.0,
        mhc_mixing_temperature_end: Optional[float] = None,
        mhc_mixing_schedule_steps: int = 0,
        mhc_output_stream: int = 0,
        mhc_output_mode: str = "stream",
        mhc_nonneg_mixing: bool = True,
        mhc_dynamic_mixing: bool = False,
        mhc_dynamic_hidden_dim: int = 0,
        mhc_dynamic_scale: float = 1.0,
        mhc_dynamic_share: str = "none",
        mhc_dynamic_layer: Optional[nn.Module] = None,
        mhc_timestep_mixing: bool = False,
        mhc_timestep_max: int = 1000,
        mhc_timestep_gamma: float = 1.0,
        mhc_timestep_strength_min: float = 0.0,
        mhc_path_scale_init: float = 1.0,
        mhc_path_scale_trainable: bool = True,
        mhc_path_dropout: Optional[float] = None,
        mhc_freeze_mixing_steps: int = 0,
        mhc_identity_clamp_steps: int = 0,
        mhc_identity_clamp_max_offdiag: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = getattr(org_module, "in_features", None)
            out_dim = getattr(org_module, "out_features", None)
            if in_dim is None or out_dim is None:
                raise RuntimeError(
                    "mHC-LoRA: Unsupported module type for linear-like replacement"
                )

        self.lora_dim = int(lora_dim)
        self.split_dims = split_dims

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.lora_dim)
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = float(multiplier)
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.mhc_num_paths = max(1, int(mhc_num_paths))
        self.mhc_sinkhorn_iters = max(1, int(mhc_sinkhorn_iters))
        self.mhc_mixing_init = str(mhc_mixing_init)
        self.mhc_mixing_strength = float(mhc_mixing_strength)
        self.mhc_mixing_strength_end = (
            float(mhc_mixing_strength_end)
            if mhc_mixing_strength_end is not None
            else None
        )
        self.mhc_mixing_temperature = float(mhc_mixing_temperature)
        self.mhc_mixing_temperature_end = (
            float(mhc_mixing_temperature_end)
            if mhc_mixing_temperature_end is not None
            else None
        )
        self.mhc_mixing_schedule_steps = max(0, int(mhc_mixing_schedule_steps))
        self.mhc_output_stream = int(mhc_output_stream)
        self.mhc_output_mode = str(mhc_output_mode)
        self.mhc_nonneg_mixing = bool(mhc_nonneg_mixing)
        self.mhc_dynamic_mixing = bool(mhc_dynamic_mixing)
        self.mhc_dynamic_hidden_dim = int(mhc_dynamic_hidden_dim)
        self.mhc_dynamic_scale = float(mhc_dynamic_scale)
        self.mhc_dynamic_share = str(mhc_dynamic_share)
        self.mhc_dynamic_layer = mhc_dynamic_layer
        self.mhc_timestep_mixing = bool(mhc_timestep_mixing)
        self.mhc_timestep_max = max(1, int(mhc_timestep_max))
        self.mhc_timestep_gamma = float(mhc_timestep_gamma)
        self.mhc_timestep_strength_min = float(mhc_timestep_strength_min)
        self.mhc_path_scale_init = float(mhc_path_scale_init)
        self.mhc_path_scale_trainable = bool(mhc_path_scale_trainable)
        self.mhc_path_dropout = (
            float(mhc_path_dropout) if mhc_path_dropout is not None else None
        )
        self.mhc_freeze_mixing_steps = max(0, int(mhc_freeze_mixing_steps))
        self.mhc_identity_clamp_steps = max(0, int(mhc_identity_clamp_steps))
        self.mhc_identity_clamp_max_offdiag = float(mhc_identity_clamp_max_offdiag)

        if self.mhc_output_stream < 0 or self.mhc_output_stream >= self.mhc_num_paths:
            self.mhc_output_stream = 0

        if self.mhc_output_mode not in {"stream", "sum", "mean"}:
            self.mhc_output_mode = "stream"

        self._init_paths(in_dim, out_dim, org_module)
        self._init_mixing()
        self._init_dynamic_mixing(in_dim)
        self._init_path_scales()

        self.org_module_ref = [org_module]
        self._is_conv = org_module.__class__.__name__ == "Conv2d"
        self._last_mixing: Optional[torch.Tensor] = None
        self._mix_step = 0
        self._use_external_step = False
        self._timestep_value: Optional[float] = None
        self._timestep_tensor: Optional[torch.Tensor] = None
        self._mixing_frozen = False
        self._last_strength_mean: Optional[float] = None

    def _init_paths(self, in_dim: int, out_dim: int, org_module: nn.Module) -> None:
        if self.split_dims is not None:
            assert sum(self.split_dims) == out_dim, "sum of split_dims must equal out_dim"
            if org_module.__class__.__name__ == "Conv2d":
                raise RuntimeError("mHC-LoRA does not support split_dims for Conv2d")
            self.lora_down = nn.ModuleList()
            self.lora_up = nn.ModuleList()
            for _ in range(self.mhc_num_paths):
                down_list = nn.ModuleList(
                    [nn.Linear(in_dim, self.lora_dim, bias=False) for _ in self.split_dims]
                )
                up_list = nn.ModuleList(
                    [nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in self.split_dims]
                )
                for lora_down in down_list:
                    nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
                for lora_up in up_list:
                    nn.init.zeros_(lora_up.weight)
                self.lora_down.append(down_list)
                self.lora_up.append(up_list)
            return

        self.lora_down = nn.ModuleList()
        self.lora_up = nn.ModuleList()
        for _ in range(self.mhc_num_paths):
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                lora_down = nn.Conv2d(
                    in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
                )
                lora_up = nn.Conv2d(
                    self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
                )
            else:
                lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
                lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

            nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_up.weight)
            self.lora_down.append(lora_down)
            self.lora_up.append(lora_up)

    def _init_mixing(self) -> None:
        if self.mhc_num_paths <= 1:
            self.mhc_mixing_logits = None
            return
        n = self.mhc_num_paths
        if self.mhc_mixing_init == "uniform":
            base = torch.full((n, n), fill_value=1.0 / n)
        elif self.mhc_mixing_init == "random":
            base = torch.rand(n, n)
        else:
            base = torch.full((n, n), fill_value=1e-3)
            base.fill_diagonal_(1.0)
        if self.mhc_nonneg_mixing:
            init = torch.log(base)
        else:
            init = base
        self.mhc_mixing_logits = nn.Parameter(init)

    def _init_dynamic_mixing(self, in_dim: int) -> None:
        if not self.mhc_dynamic_mixing or self.mhc_num_paths <= 1:
            self.mhc_dynamic_layer = None
            return
        if self.mhc_dynamic_layer is not None:
            return
        out_dim = self.mhc_num_paths * self.mhc_num_paths
        hidden = max(0, int(self.mhc_dynamic_hidden_dim))
        if hidden > 0:
            self.mhc_dynamic_layer = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, out_dim, bias=True),
            )
        else:
            self.mhc_dynamic_layer = nn.Linear(in_dim, out_dim, bias=True)

    def set_mix_step(self, step: int) -> None:
        self._mix_step = max(int(step), 0)
        self._use_external_step = True
        self._update_mixing_freeze()

    def set_timestep(self, timesteps: torch.Tensor) -> None:
        try:
            t = timesteps.detach()
            if t.numel() > 1:
                t_val = float(t.mean().item())
            else:
                t_val = float(t.item())
            self._timestep_value = t_val
            self._timestep_tensor = t.reshape(-1).to(device=t.device)
        except Exception:
            self._timestep_value = None
            self._timestep_tensor = None

    def _update_mixing_freeze(self) -> None:
        if self.mhc_num_paths <= 1:
            return
        should_freeze = self.mhc_freeze_mixing_steps > 0 and self._mix_step < self.mhc_freeze_mixing_steps
        if should_freeze == self._mixing_frozen:
            return
        self._mixing_frozen = should_freeze
        requires_grad = not should_freeze
        if isinstance(self.mhc_mixing_logits, nn.Parameter):
            self.mhc_mixing_logits.requires_grad_(requires_grad)
        if self.mhc_dynamic_layer is not None:
            for param in self.mhc_dynamic_layer.parameters():
                param.requires_grad_(requires_grad)

    def _init_path_scales(self) -> None:
        if self.mhc_num_paths <= 1:
            self.mhc_path_scale = None
            return
        init = torch.full((self.mhc_num_paths,), self.mhc_path_scale_init)
        if self.mhc_path_scale_trainable:
            self.mhc_path_scale = nn.Parameter(init)
        else:
            self.register_buffer("mhc_path_scale", init, persistent=False)

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _sinkhorn(self, matrix: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        out = matrix
        for _ in range(self.mhc_sinkhorn_iters):
            out = out / (out.sum(dim=1, keepdim=True) + eps)
            out = out / (out.sum(dim=0, keepdim=True) + eps)
        return out

    def _pool_dynamic_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_conv:
            if x.dim() == 4:
                return x.mean(dim=(0, 2, 3))
            if x.dim() == 5:
                return x.mean(dim=(0, 2, 3, 4))
        if x.dim() == 2:
            return x.mean(dim=0)
        if x.dim() == 3:
            return x.mean(dim=(0, 1))
        return x.mean(dim=tuple(range(x.dim() - 1)))

    def _compute_dynamic_logits(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.mhc_dynamic_mixing or self.mhc_dynamic_layer is None:
            return None
        pooled = self._pool_dynamic_features(x)
        return self.mhc_dynamic_layer(pooled).view(self.mhc_num_paths, self.mhc_num_paths)

    def _scheduled_mixing_values(self) -> Tuple[float, float]:
        strength = self.mhc_mixing_strength
        temperature = self.mhc_mixing_temperature
        if self.mhc_mixing_schedule_steps > 0:
            if self.mhc_mixing_strength_end is not None:
                progress = min(
                    float(max(self._mix_step, 0)) / self.mhc_mixing_schedule_steps,
                    1.0,
                )
                strength = (
                    (1.0 - progress) * self.mhc_mixing_strength
                    + progress * self.mhc_mixing_strength_end
                )
            if self.mhc_mixing_temperature_end is not None:
                progress = min(
                    float(max(self._mix_step, 0)) / self.mhc_mixing_schedule_steps,
                    1.0,
                )
                temperature = (
                    (1.0 - progress) * self.mhc_mixing_temperature
                    + progress * self.mhc_mixing_temperature_end
                )
        return strength, temperature

    def _timestep_strengths(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if not self.mhc_timestep_mixing or self._timestep_tensor is None:
            return None
        t = self._timestep_tensor
        if t.numel() == 1:
            t_norm = min(max(float(t.item()) / self.mhc_timestep_max, 0.0), 1.0)
            scale = (1.0 - t_norm) ** max(self.mhc_timestep_gamma, 0.0)
            min_strength = min(max(self.mhc_timestep_strength_min, 0.0), 1.0)
            s = min_strength + (self._scheduled_mixing_values()[0] - min_strength) * scale
            return torch.full((batch_size,), s, device=device, dtype=dtype)
        if t.numel() != batch_size:
            t = t[:batch_size]
        t_norm = torch.clamp(t.to(device=device, dtype=dtype) / self.mhc_timestep_max, 0.0, 1.0)
        scale = (1.0 - t_norm) ** max(self.mhc_timestep_gamma, 0.0)
        min_strength = min(max(self.mhc_timestep_strength_min, 0.0), 1.0)
        base_strength = self._scheduled_mixing_values()[0]
        return min_strength + (base_strength - min_strength) * scale

    def _compute_mixing_matrix(
        self, x: Optional[torch.Tensor] = None, apply_strength: bool = True
    ) -> torch.Tensor:
        if self.mhc_num_paths <= 1 or self.mhc_mixing_logits is None:
            return torch.eye(1, device=self.device, dtype=self.dtype)
        logits = self.mhc_mixing_logits
        dyn = self._compute_dynamic_logits(x) if x is not None else None
        if dyn is not None:
            logits = logits + dyn * self.mhc_dynamic_scale
        strength, temperature = self._scheduled_mixing_values()
        temp = max(temperature, 1e-6)
        logits = logits / temp
        logits = logits - logits.max()
        if self.mhc_nonneg_mixing:
            matrix = torch.exp(logits)
        else:
            matrix = torch.abs(logits)
        matrix = self._sinkhorn(matrix)
        if apply_strength:
            strength = min(max(strength, 0.0), 1.0)
            if strength < 1.0:
                eye = torch.eye(self.mhc_num_paths, device=matrix.device, dtype=matrix.dtype)
                matrix = (1.0 - strength) * eye + strength * matrix
        if (
            self.mhc_identity_clamp_steps > 0
            and self._mix_step < self.mhc_identity_clamp_steps
            and self.mhc_identity_clamp_max_offdiag > 0.0
        ):
            max_offdiag = float(self.mhc_identity_clamp_max_offdiag)
            if max_offdiag < 1.0:
                eye = torch.eye(self.mhc_num_paths, device=matrix.device, dtype=matrix.dtype)
                offdiag = matrix * (1.0 - eye)
                offdiag = torch.minimum(offdiag, torch.full_like(offdiag, max_offdiag))
                matrix = offdiag + (matrix * eye)
                matrix = self._sinkhorn(matrix)
        return matrix

    def _apply_rank_dropout(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        scale = self.scale
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((tensor.size(0), self.lora_dim), device=tensor.device)
                > self.rank_dropout
            )
            if tensor.dim() == 3:
                mask = mask.unsqueeze(1)
            elif tensor.dim() == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            tensor = tensor * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        return tensor, scale

    def _compute_path_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        path_scales = None
        if self.mhc_num_paths > 1 and self.mhc_path_scale is not None:
            path_scales = self.mhc_path_scale.to(device=x.device, dtype=x.dtype)
        path_dropout = self.mhc_path_dropout
        for idx, (down, up) in enumerate(zip(self.lora_down, self.lora_up)):
            if self.split_dims is not None:
                lxs = [lora_down(x) for lora_down in down]
                if self.dropout is not None and self.training:
                    lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]
                if self.rank_dropout is not None and self.training:
                    masks = [
                        torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                        > self.rank_dropout
                        for lx in lxs
                    ]
                    for i in range(len(lxs)):
                        if lxs[i].dim() == 3:
                            masks[i] = masks[i].unsqueeze(1)
                        elif lxs[i].dim() == 4:
                            masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                        lxs[i] = lxs[i] * masks[i]
                    scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
                else:
                    scale = self.scale
                lxs = [lora_up(lx) for lora_up, lx in zip(up, lxs)]
                path_out = torch.cat(lxs, dim=-1) * scale
                if path_scales is not None:
                    path_out = path_out * path_scales[idx]
                if path_dropout is not None and self.training:
                    keep = torch.rand(1, device=path_out.device) >= path_dropout
                    if not keep:
                        path_out = torch.zeros_like(path_out)
                    else:
                        path_out = path_out / max(1.0 - path_dropout, 1e-6)
                outputs.append(path_out)
                continue

            lx = down(x)
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)
            lx, scale = self._apply_rank_dropout(lx)
            lx = up(lx)
            path_out = lx * scale
            if path_scales is not None:
                path_out = path_out * path_scales[idx]
            if path_dropout is not None and self.training:
                keep = torch.rand(1, device=path_out.device) >= path_dropout
                if not keep:
                    path_out = torch.zeros_like(path_out)
                else:
                    path_out = path_out / max(1.0 - path_dropout, 1e-6)
            outputs.append(path_out)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded
        if self.training and not self._use_external_step:
            self._mix_step += 1

        if self.mhc_num_paths <= 1:
            path_outputs = self._compute_path_outputs(x)
            return org_forwarded + path_outputs[0] * self.multiplier

        path_outputs = self._compute_path_outputs(x)
        stack = torch.stack(path_outputs, dim=0)
        strengths = self._timestep_strengths(stack.shape[1], stack.device, stack.dtype)
        mixing = self._compute_mixing_matrix(
            x, apply_strength=strengths is None
        ).to(device=stack.device, dtype=stack.dtype)
        self._last_mixing = mixing
        mixed = torch.einsum("ij,j...->i...", mixing, stack)
        if self.mhc_output_mode == "sum":
            mixed_out = mixed.sum(dim=0)
            identity_out = stack.sum(dim=0)
        elif self.mhc_output_mode == "mean":
            mixed_out = mixed.mean(dim=0)
            identity_out = stack.mean(dim=0)
        else:
            mixed_out = mixed[self.mhc_output_stream]
            identity_out = stack[self.mhc_output_stream]
        if strengths is None:
            out = mixed_out
            self._last_strength_mean = None
        else:
            view_shape = [strengths.shape[0]] + [1] * (mixed_out.dim() - 1)
            s = strengths.view(view_shape)
            out = identity_out + (mixed_out - identity_out) * s
            try:
                self._last_strength_mean = float(strengths.mean().item())
            except Exception:
                self._last_strength_mean = None
        return org_forwarded + out * self.multiplier

    def get_weight(self) -> torch.Tensor:
        if self.split_dims is not None:
            raise RuntimeError("mHC-LoRA get_weight does not support split_dims")
        mixing = self._compute_mixing_matrix()
        if self.mhc_output_mode == "sum":
            mix_row = mixing.sum(dim=0)
        elif self.mhc_output_mode == "mean":
            mix_row = mixing.mean(dim=0)
        else:
            mix_row = mixing[self.mhc_output_stream]
        if self.mhc_num_paths > 1 and self.mhc_path_scale is not None:
            mix_row = mix_row * self.mhc_path_scale.to(device=mix_row.device, dtype=mix_row.dtype)
        weight = None
        for idx, (down, up) in enumerate(zip(self.lora_down, self.lora_up)):
            if down.weight.dim() == 2:
                path_weight = (up.weight @ down.weight) * self.scale
            elif down.weight.size()[2:4] == (1, 1):
                path_weight = (
                    (up.weight.squeeze(3).squeeze(2) @ down.weight.squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down.weight.permute(1, 0, 2, 3), up.weight
                ).permute(1, 0, 2, 3)
                path_weight = conved * self.scale
            path_weight = path_weight * mix_row[idx].to(device=path_weight.device)
            weight = path_weight if weight is None else weight + path_weight
        return weight  # type: ignore[return-value]

    def identity_regularization(self) -> torch.Tensor:
        mixing = self._last_mixing
        if mixing is None:
            mixing = self._compute_mixing_matrix()
        eye = torch.eye(self.mhc_num_paths, device=mixing.device, dtype=mixing.dtype)
        return torch.mean((mixing - eye) ** 2)

    def mixing_entropy(self) -> torch.Tensor:
        mixing = self._last_mixing
        if mixing is None:
            mixing = self._compute_mixing_matrix()
        eps = 1e-8
        probs = mixing / (mixing.sum(dim=1, keepdim=True) + eps)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
        return entropy.mean()

    def merge_to(self, sd: Dict[str, torch.Tensor], dtype, device, non_blocking=False) -> None:
        org_module = self.org_module_ref[0]
        org_sd = org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        mix = self._compute_mixing_matrix()
        if self.mhc_output_mode == "sum":
            mix_row = mix.sum(dim=0)
        elif self.mhc_output_mode == "mean":
            mix_row = mix.mean(dim=0)
        else:
            mix_row = mix[self.mhc_output_stream]
        mix_row = mix_row.to(device=weight.device, dtype=weight.dtype)
        if self.mhc_num_paths > 1 and self.mhc_path_scale is not None:
            mix_row = mix_row * self.mhc_path_scale.to(device=mix_row.device, dtype=mix_row.dtype)

        merged = torch.zeros_like(weight)
        for idx in range(self.mhc_num_paths):
            down_weight = sd[f"lora_down.{idx}.weight"].to(
                device, dtype=torch.float, non_blocking=non_blocking
            )
            up_weight = sd[f"lora_up.{idx}.weight"].to(
                device, dtype=torch.float, non_blocking=non_blocking
            )
            if len(weight.size()) == 2:
                path = (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                path = (
                    up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scale
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                path = conved * self.scale
            merged = merged + path * mix_row[idx].to(device=merged.device, dtype=merged.dtype)

        org_sd["weight"] = (weight + self.multiplier * merged).to(
            org_device, dtype=dtype
        )
        org_module.load_state_dict(org_sd)


class MhcLoRAInfModule(MhcLoRAModule):
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            **kwargs,
        )
        self.enabled = True
        self.network = None

    def set_network(self, network) -> None:
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.org_forward(x)
        return super().forward(x)


class MhcLoRANetwork(LoRANetwork):
    """LoRA network that instantiates MhcLoRAModule with multi-path mixing."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        mhc_num_paths: int = 2,
        mhc_sinkhorn_iters: int = 20,
        mhc_mixing_init: str = "identity",
        mhc_mixing_strength: float = 1.0,
        mhc_mixing_strength_end: Optional[float] = None,
        mhc_mixing_temperature: float = 1.0,
        mhc_mixing_temperature_end: Optional[float] = None,
        mhc_mixing_schedule_steps: int = 0,
        mhc_output_stream: int = 0,
        mhc_output_mode: str = "stream",
        mhc_nonneg_mixing: bool = True,
        mhc_dynamic_mixing: bool = False,
        mhc_dynamic_hidden_dim: int = 0,
        mhc_dynamic_scale: float = 1.0,
        mhc_dynamic_share: str = "none",
        mhc_timestep_mixing: bool = False,
        mhc_timestep_max: int = 1000,
        mhc_timestep_gamma: float = 1.0,
        mhc_timestep_strength_min: float = 0.0,
        mhc_path_scale_init: float = 1.0,
        mhc_path_scale_trainable: bool = True,
        mhc_path_dropout: Optional[float] = None,
        mhc_freeze_mixing_steps: int = 0,
        mhc_identity_clamp_steps: int = 0,
        mhc_identity_clamp_max_offdiag: float = 0.0,
        module_class: Optional[type] = None,
        **kwargs,
    ) -> None:
        self.mhc_num_paths = mhc_num_paths
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_mixing_init = mhc_mixing_init
        self.mhc_mixing_strength = mhc_mixing_strength
        self.mhc_mixing_strength_end = mhc_mixing_strength_end
        self.mhc_mixing_temperature = mhc_mixing_temperature
        self.mhc_mixing_temperature_end = mhc_mixing_temperature_end
        self.mhc_mixing_schedule_steps = mhc_mixing_schedule_steps
        self.mhc_output_stream = mhc_output_stream
        self.mhc_output_mode = mhc_output_mode
        self.mhc_nonneg_mixing = mhc_nonneg_mixing
        self.mhc_dynamic_mixing = mhc_dynamic_mixing
        self.mhc_dynamic_hidden_dim = mhc_dynamic_hidden_dim
        self.mhc_dynamic_scale = mhc_dynamic_scale
        self.mhc_dynamic_share = mhc_dynamic_share
        self.mhc_timestep_mixing = mhc_timestep_mixing
        self.mhc_timestep_max = mhc_timestep_max
        self.mhc_timestep_gamma = mhc_timestep_gamma
        self.mhc_timestep_strength_min = mhc_timestep_strength_min
        self.mhc_path_scale_init = mhc_path_scale_init
        self.mhc_path_scale_trainable = mhc_path_scale_trainable
        self.mhc_path_dropout = mhc_path_dropout
        self.mhc_freeze_mixing_steps = mhc_freeze_mixing_steps
        self.mhc_identity_clamp_steps = mhc_identity_clamp_steps
        self.mhc_identity_clamp_max_offdiag = mhc_identity_clamp_max_offdiag

        super().__init__(
            target_replace_modules=target_replace_modules,
            prefix=prefix,
            text_encoders=text_encoders,
            unet=unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            module_class=module_class or self._create_mhc_lora_module,
            **kwargs,
        )

    def _create_mhc_lora_module(
        self,
        lora_name,
        org_module,
        multiplier,
        lora_dim,
        alpha,
        **kwargs,
    ):
        dynamic_layer = None
        if self.mhc_dynamic_mixing and str(self.mhc_dynamic_share) == "layer":
            try:
                import re as _re

                match = _re.search(r"_blocks_(\d+)_", str(lora_name))
                layer_idx = int(match.group(1)) if match else -1
                if not hasattr(self, "_mhc_dynamic_layers"):
                    self._mhc_dynamic_layers = {}
                if layer_idx not in self._mhc_dynamic_layers:
                    self._mhc_dynamic_layers[layer_idx] = None
                dynamic_layer = self._mhc_dynamic_layers[layer_idx]
                if dynamic_layer is None:
                    if org_module.__class__.__name__ == "Conv2d":
                        in_dim = org_module.in_channels
                    else:
                        in_dim = getattr(org_module, "in_features", None)
                        if in_dim is None:
                            in_dim = 0
                    if in_dim <= 0:
                        dynamic_layer = None
                        self._mhc_dynamic_layers[layer_idx] = dynamic_layer
                        raise RuntimeError("Unable to infer in_dim for dynamic layer share")
                    out_dim = self.mhc_num_paths * self.mhc_num_paths
                    hidden = max(0, int(self.mhc_dynamic_hidden_dim))
                    if hidden > 0:
                        dynamic_layer = nn.Sequential(
                            nn.Linear(in_dim, hidden, bias=True),
                            nn.SiLU(),
                            nn.Linear(hidden, out_dim, bias=True),
                        )
                    else:
                        dynamic_layer = nn.Linear(in_dim, out_dim, bias=True)
                    self._mhc_dynamic_layers[layer_idx] = dynamic_layer
            except Exception:
                dynamic_layer = None
        return MhcLoRAModule(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=kwargs.get("dropout"),
            rank_dropout=kwargs.get("rank_dropout"),
            module_dropout=kwargs.get("module_dropout"),
            mhc_num_paths=self.mhc_num_paths,
            mhc_sinkhorn_iters=self.mhc_sinkhorn_iters,
            mhc_mixing_init=self.mhc_mixing_init,
            mhc_mixing_strength=self.mhc_mixing_strength,
            mhc_mixing_strength_end=self.mhc_mixing_strength_end,
            mhc_mixing_temperature=self.mhc_mixing_temperature,
            mhc_mixing_temperature_end=self.mhc_mixing_temperature_end,
            mhc_mixing_schedule_steps=self.mhc_mixing_schedule_steps,
            mhc_output_stream=self.mhc_output_stream,
            mhc_output_mode=self.mhc_output_mode,
            mhc_nonneg_mixing=self.mhc_nonneg_mixing,
            mhc_dynamic_mixing=self.mhc_dynamic_mixing,
            mhc_dynamic_hidden_dim=self.mhc_dynamic_hidden_dim,
            mhc_dynamic_scale=self.mhc_dynamic_scale,
            mhc_dynamic_share=self.mhc_dynamic_share,
            mhc_dynamic_layer=dynamic_layer,
            mhc_timestep_mixing=self.mhc_timestep_mixing,
            mhc_timestep_max=self.mhc_timestep_max,
            mhc_timestep_gamma=self.mhc_timestep_gamma,
            mhc_timestep_strength_min=self.mhc_timestep_strength_min,
            mhc_path_scale_init=self.mhc_path_scale_init,
            mhc_path_scale_trainable=self.mhc_path_scale_trainable,
            mhc_path_dropout=self.mhc_path_dropout,
            mhc_freeze_mixing_steps=self.mhc_freeze_mixing_steps,
            mhc_identity_clamp_steps=self.mhc_identity_clamp_steps,
            mhc_identity_clamp_max_offdiag=self.mhc_identity_clamp_max_offdiag,
        )

    def set_current_step(self, step: int) -> None:
        for lora in getattr(self, "unet_loras", []):
            if hasattr(lora, "set_mix_step"):
                lora.set_mix_step(step)

    def set_mhc_timestep(self, timesteps: torch.Tensor) -> None:
        for lora in getattr(self, "unet_loras", []):
            if hasattr(lora, "set_timestep"):
                lora.set_timestep(timesteps)

    def get_mhc_mixing_stats(self) -> Optional[Dict[str, float]]:
        entropies = []
        offdiag_means = []
        identity_devs = []
        weights = []
        for lora in getattr(self, "unet_loras", []):
            mixing = getattr(lora, "_last_mixing", None)
            if mixing is None:
                continue
            with torch.no_grad():
                m = mixing.detach().float()
                n = m.shape[0]
                eye = torch.eye(n, device=m.device, dtype=m.dtype)
                offdiag = m * (1.0 - eye)
                offdiag_means.append(offdiag.mean().item())
                identity_devs.append(((m - eye) ** 2).mean().item())
                eps = 1e-8
                probs = m / (m.sum(dim=1, keepdim=True) + eps)
                entropy = -(probs * torch.log(probs + eps)).sum(dim=1).mean()
                entropies.append(entropy.item())
            strength_mean = getattr(lora, "_last_strength_mean", None)
            if strength_mean is None:
                weights.append(1.0)
            else:
                weights.append(max(float(strength_mean), 0.0))
        if not entropies:
            return None
        total_weight = sum(weights) if weights else 0.0
        if total_weight <= 0.0:
            total_weight = float(len(entropies))
            weights = [1.0] * len(entropies)
        def _weighted_avg(values: List[float]) -> float:
            return float(sum(v * w for v, w in zip(values, weights)) / total_weight)
        return {
            "mhc/mixing_entropy": _weighted_avg(entropies),
            "mhc/mixing_offdiag_mean": _weighted_avg(offdiag_means),
            "mhc/mixing_identity_deviation": _weighted_avg(identity_devs),
        }

    def get_mhc_mixing_histogram(self) -> Optional[torch.Tensor]:
        values = []
        for lora in getattr(self, "unet_loras", []):
            mixing = getattr(lora, "_last_mixing", None)
            if mixing is None:
                continue
            values.append(mixing.detach().flatten().float())
        if not values:
            return None
        return torch.cat(values, dim=0)


def _detect_mhc_num_paths(weights_sd: Dict[str, torch.Tensor]) -> int:
    max_index = -1
    for key in weights_sd.keys():
        if ".lora_down." in key and key.endswith(".weight"):
            parts = key.split(".lora_down.")
            if len(parts) < 2:
                continue
            idx_part = parts[1].split(".", 1)[0]
            if idx_part.isdigit():
                max_index = max(max_index, int(idx_part))
    return max_index + 1 if max_index >= 0 else 1


def _extract_module_dims(
    weights_sd: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, torch.Tensor] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down." in key and key.endswith(".weight"):
            dim = int(value.shape[0])
            modules_dim[lora_name] = dim
    return modules_dim, modules_alpha


def create_mhc_lora_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> MhcLoRANetwork:
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    mhc_num_paths = int(kwargs.pop("mhc_num_paths", 2))
    mhc_sinkhorn_iters = int(kwargs.pop("mhc_sinkhorn_iters", 20))
    mhc_mixing_init = str(kwargs.pop("mhc_mixing_init", "identity"))
    mhc_mixing_strength = float(kwargs.pop("mhc_mixing_strength", 1.0))
    def _pop_optional_float(key: str, default=None) -> Optional[float]:
        val = kwargs.pop(key, default)
        if val is None:
            return None
        if isinstance(val, str) and val.strip().lower() in {"none", "null", ""}:
            return None
        try:
            return float(val)
        except Exception:
            return None

    mhc_mixing_strength_end = _pop_optional_float("mhc_mixing_strength_end")
    mhc_mixing_temperature = float(kwargs.pop("mhc_mixing_temperature", 1.0))
    mhc_mixing_temperature_end = _pop_optional_float("mhc_mixing_temperature_end")
    mhc_mixing_schedule_steps = int(kwargs.pop("mhc_mixing_schedule_steps", 0))
    mhc_output_stream = int(kwargs.pop("mhc_output_stream", 0))
    mhc_output_mode = str(kwargs.pop("mhc_output_mode", "stream"))
    mhc_nonneg_mixing = (
        str(kwargs.pop("mhc_nonneg_mixing", "true")).lower() == "true"
    )
    mhc_dynamic_mixing = (
        str(kwargs.pop("mhc_dynamic_mixing", "false")).lower() == "true"
    )
    mhc_dynamic_hidden_dim = int(kwargs.pop("mhc_dynamic_hidden_dim", 0))
    mhc_dynamic_scale = float(kwargs.pop("mhc_dynamic_scale", 1.0))
    mhc_dynamic_share = str(kwargs.pop("mhc_dynamic_share", "none"))
    mhc_timestep_mixing = (
        str(kwargs.pop("mhc_timestep_mixing", "false")).lower() == "true"
    )
    mhc_timestep_max = int(kwargs.pop("mhc_timestep_max", 1000))
    mhc_timestep_gamma = float(kwargs.pop("mhc_timestep_gamma", 1.0))
    mhc_timestep_strength_min = float(kwargs.pop("mhc_timestep_strength_min", 0.0))
    mhc_path_scale_init = float(kwargs.pop("mhc_path_scale_init", 1.0))
    mhc_path_scale_trainable = (
        str(kwargs.pop("mhc_path_scale_trainable", "true")).lower() == "true"
    )
    mhc_path_dropout = kwargs.pop("mhc_path_dropout", None)
    if mhc_path_dropout is not None:
        mhc_path_dropout = float(mhc_path_dropout)
    mhc_freeze_mixing_steps = int(kwargs.pop("mhc_freeze_mixing_steps", 0))
    mhc_identity_clamp_steps = int(kwargs.pop("mhc_identity_clamp_steps", 0))
    mhc_identity_clamp_max_offdiag = float(
        kwargs.pop("mhc_identity_clamp_max_offdiag", 0.0)
    )
    mhc_identity_reg_lambda = float(kwargs.pop("mhc_identity_reg_lambda", 0.0))
    mhc_identity_reg_warmup_steps = int(
        kwargs.pop("mhc_identity_reg_warmup_steps", 0)
    )
    mhc_identity_reg_power = float(
        kwargs.pop("mhc_identity_reg_power", 1.0)
    )
    mhc_entropy_reg_lambda = float(kwargs.pop("mhc_entropy_reg_lambda", 0.0))
    mhc_entropy_reg_target = _pop_optional_float("mhc_entropy_reg_target")

    # Parse standard LoRA kwargs
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        conv_alpha = float(conv_alpha) if conv_alpha is not None else 1.0

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        import ast as _ast

        exclude_patterns = _ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        import ast as _ast

        include_patterns = _ast.literal_eval(include_patterns)
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if str(verbose) == "True" else bool(verbose)

    network = MhcLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="mhc_lora_unet",
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        mhc_num_paths=mhc_num_paths,
        mhc_sinkhorn_iters=mhc_sinkhorn_iters,
        mhc_mixing_init=mhc_mixing_init,
        mhc_mixing_strength=mhc_mixing_strength,
        mhc_mixing_strength_end=mhc_mixing_strength_end,
        mhc_mixing_temperature=mhc_mixing_temperature,
        mhc_mixing_temperature_end=mhc_mixing_temperature_end,
        mhc_mixing_schedule_steps=mhc_mixing_schedule_steps,
        mhc_output_stream=mhc_output_stream,
        mhc_output_mode=mhc_output_mode,
        mhc_nonneg_mixing=mhc_nonneg_mixing,
        mhc_dynamic_mixing=mhc_dynamic_mixing,
        mhc_dynamic_hidden_dim=mhc_dynamic_hidden_dim,
        mhc_dynamic_scale=mhc_dynamic_scale,
        mhc_dynamic_share=mhc_dynamic_share,
        mhc_timestep_mixing=mhc_timestep_mixing,
        mhc_timestep_max=mhc_timestep_max,
        mhc_timestep_gamma=mhc_timestep_gamma,
        mhc_timestep_strength_min=mhc_timestep_strength_min,
        mhc_path_scale_init=mhc_path_scale_init,
        mhc_path_scale_trainable=mhc_path_scale_trainable,
        mhc_path_dropout=mhc_path_dropout,
        mhc_freeze_mixing_steps=mhc_freeze_mixing_steps,
        mhc_identity_clamp_steps=mhc_identity_clamp_steps,
        mhc_identity_clamp_max_offdiag=mhc_identity_clamp_max_offdiag,
    )

    network.mhc_num_paths = mhc_num_paths
    network.mhc_identity_reg_lambda = mhc_identity_reg_lambda
    network.mhc_identity_reg_warmup_steps = mhc_identity_reg_warmup_steps
    network.mhc_identity_reg_power = mhc_identity_reg_power
    network.mhc_entropy_reg_lambda = mhc_entropy_reg_lambda
    network.mhc_entropy_reg_target = mhc_entropy_reg_target

    logger.info(
        "mHC-LoRA initialized: paths=%s, sinkhorn_iters=%s, mix_init=%s, mix_strength=%s",
        mhc_num_paths,
        mhc_sinkhorn_iters,
        mhc_mixing_init,
        mhc_mixing_strength,
    )
    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)
    return network


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> MhcLoRANetwork:
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        import ast as _ast

        exclude_patterns = _ast.literal_eval(exclude_patterns)
    exclude_patterns.append(
        r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"
    )
    kwargs["exclude_patterns"] = exclude_patterns

    return create_mhc_lora_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> MhcLoRANetwork:
    mhc_num_paths = _detect_mhc_num_paths(weights_sd)
    modules_dim, modules_alpha = _extract_module_dims(weights_sd)
    def _get_optional_float(key: str) -> Optional[float]:
        val = kwargs.get(key, None)
        if val is None:
            return None
        if isinstance(val, str) and val.strip().lower() in {"none", "null", ""}:
            return None
        try:
            return float(val)
        except Exception:
            return None
    mhc_output_mode = str(kwargs.get("mhc_output_mode", "stream"))
    mhc_output_stream = int(kwargs.get("mhc_output_stream", 0))
    mhc_nonneg_mixing = bool(kwargs.get("mhc_nonneg_mixing", True))
    mhc_sinkhorn_iters = int(kwargs.get("mhc_sinkhorn_iters", 20))
    mhc_mixing_init = str(kwargs.get("mhc_mixing_init", "identity"))
    mhc_mixing_strength = float(kwargs.get("mhc_mixing_strength", 1.0))
    mhc_mixing_strength_end = _get_optional_float("mhc_mixing_strength_end")
    mhc_mixing_temperature = float(kwargs.get("mhc_mixing_temperature", 1.0))
    mhc_mixing_temperature_end = _get_optional_float("mhc_mixing_temperature_end")
    mhc_mixing_schedule_steps = int(kwargs.get("mhc_mixing_schedule_steps", 0))
    mhc_dynamic_mixing = bool(kwargs.get("mhc_dynamic_mixing", False))
    mhc_dynamic_hidden_dim = int(kwargs.get("mhc_dynamic_hidden_dim", 0))
    mhc_dynamic_scale = float(kwargs.get("mhc_dynamic_scale", 1.0))
    mhc_dynamic_share = str(kwargs.get("mhc_dynamic_share", "none"))
    mhc_timestep_mixing = bool(kwargs.get("mhc_timestep_mixing", False))
    mhc_timestep_max = int(kwargs.get("mhc_timestep_max", 1000))
    mhc_timestep_gamma = float(kwargs.get("mhc_timestep_gamma", 1.0))
    mhc_timestep_strength_min = float(kwargs.get("mhc_timestep_strength_min", 0.0))
    mhc_path_scale_init = float(kwargs.get("mhc_path_scale_init", 1.0))
    mhc_path_scale_trainable = bool(
        kwargs.get("mhc_path_scale_trainable", True)
    )
    mhc_path_dropout = kwargs.get("mhc_path_dropout", None)
    if mhc_path_dropout is not None:
        mhc_path_dropout = float(mhc_path_dropout)
    mhc_freeze_mixing_steps = int(kwargs.get("mhc_freeze_mixing_steps", 0))
    mhc_identity_clamp_steps = int(kwargs.get("mhc_identity_clamp_steps", 0))
    mhc_identity_clamp_max_offdiag = float(
        kwargs.get("mhc_identity_clamp_max_offdiag", 0.0)
    )
    mhc_identity_reg_lambda = float(kwargs.get("mhc_identity_reg_lambda", 0.0))
    mhc_identity_reg_warmup_steps = int(
        kwargs.get("mhc_identity_reg_warmup_steps", 0)
    )
    mhc_identity_reg_power = float(kwargs.get("mhc_identity_reg_power", 1.0))
    mhc_entropy_reg_lambda = float(kwargs.get("mhc_entropy_reg_lambda", 0.0))
    mhc_entropy_reg_target = _get_optional_float("mhc_entropy_reg_target")

    def _module_factory(lora_name, org_module, multiplier, lora_dim, alpha, **kw):
        module_cls = MhcLoRAInfModule if for_inference else MhcLoRAModule
        return module_cls(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            mhc_num_paths=mhc_num_paths,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_mixing_init=mhc_mixing_init,
            mhc_mixing_strength=mhc_mixing_strength,
            mhc_mixing_strength_end=mhc_mixing_strength_end,
            mhc_mixing_temperature=mhc_mixing_temperature,
            mhc_mixing_temperature_end=mhc_mixing_temperature_end,
            mhc_mixing_schedule_steps=mhc_mixing_schedule_steps,
            mhc_output_stream=mhc_output_stream,
            mhc_output_mode=mhc_output_mode,
            mhc_nonneg_mixing=mhc_nonneg_mixing,
            mhc_dynamic_mixing=mhc_dynamic_mixing,
            mhc_dynamic_hidden_dim=mhc_dynamic_hidden_dim,
            mhc_dynamic_scale=mhc_dynamic_scale,
            mhc_dynamic_share=mhc_dynamic_share,
            mhc_timestep_mixing=mhc_timestep_mixing,
            mhc_timestep_max=mhc_timestep_max,
            mhc_timestep_gamma=mhc_timestep_gamma,
            mhc_timestep_strength_min=mhc_timestep_strength_min,
            mhc_path_scale_init=mhc_path_scale_init,
            mhc_path_scale_trainable=mhc_path_scale_trainable,
            mhc_path_dropout=mhc_path_dropout,
            mhc_freeze_mixing_steps=mhc_freeze_mixing_steps,
            mhc_identity_clamp_steps=mhc_identity_clamp_steps,
            mhc_identity_clamp_max_offdiag=mhc_identity_clamp_max_offdiag,
        )

    network = MhcLoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="mhc_lora_unet",
        text_encoders=text_encoders,  # type: ignore
        unet=unet,  # type: ignore
        multiplier=multiplier,
        lora_dim=1,
        alpha=1.0,
        module_class=_module_factory,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        mhc_num_paths=mhc_num_paths,
        mhc_sinkhorn_iters=mhc_sinkhorn_iters,
        mhc_mixing_init=mhc_mixing_init,
        mhc_mixing_strength=mhc_mixing_strength,
        mhc_mixing_strength_end=mhc_mixing_strength_end,
        mhc_mixing_temperature=mhc_mixing_temperature,
        mhc_mixing_temperature_end=mhc_mixing_temperature_end,
        mhc_mixing_schedule_steps=mhc_mixing_schedule_steps,
        mhc_output_stream=mhc_output_stream,
        mhc_output_mode=mhc_output_mode,
        mhc_nonneg_mixing=mhc_nonneg_mixing,
        mhc_dynamic_mixing=mhc_dynamic_mixing,
        mhc_dynamic_hidden_dim=mhc_dynamic_hidden_dim,
        mhc_dynamic_scale=mhc_dynamic_scale,
        mhc_dynamic_share=mhc_dynamic_share,
        mhc_timestep_mixing=mhc_timestep_mixing,
        mhc_timestep_max=mhc_timestep_max,
        mhc_timestep_gamma=mhc_timestep_gamma,
        mhc_timestep_strength_min=mhc_timestep_strength_min,
        mhc_path_scale_init=mhc_path_scale_init,
        mhc_path_scale_trainable=mhc_path_scale_trainable,
        mhc_path_dropout=mhc_path_dropout,
        mhc_freeze_mixing_steps=mhc_freeze_mixing_steps,
        mhc_identity_clamp_steps=mhc_identity_clamp_steps,
        mhc_identity_clamp_max_offdiag=mhc_identity_clamp_max_offdiag,
    )
    network.mhc_num_paths = mhc_num_paths
    network.mhc_identity_reg_lambda = mhc_identity_reg_lambda
    network.mhc_identity_reg_warmup_steps = mhc_identity_reg_warmup_steps
    network.mhc_identity_reg_power = mhc_identity_reg_power
    network.mhc_entropy_reg_lambda = mhc_entropy_reg_lambda
    network.mhc_entropy_reg_target = mhc_entropy_reg_target
    return network
