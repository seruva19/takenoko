"""Ortho-Hydra adapter network module for WAN DiT models."""

from __future__ import annotations

import ast
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.logger import get_logger
from modules.ramtorch_linear_factory import is_linear_like
from networks.lora_wan import LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw)


def _parse_int(raw: Any, default: int, minimum: int = 0) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, value)


def _parse_float(raw: Any, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        value = default
    return max(minimum, value)


def _parse_patterns(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return None


class OrthoHydraLoRAModule(nn.Module):
    """Cayley-rotated HydraLoRA with disjoint per-expert output bases."""

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
        ortho_hydra_num_experts: int = 4,
        ortho_hydra_svd_niter: int = 2,
        ortho_hydra_router_init_std: float = 0.01,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if split_dims is not None:
            raise ValueError("Ortho-Hydra does not support split_dims modules.")
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("Ortho-Hydra supports linear-like modules only.")
        if not (is_linear_like(org_module) or org_module.__class__.__name__ == "Linear"):
            raise RuntimeError(
                f"Ortho-Hydra: unsupported module type {type(org_module).__name__}."
            )

        in_dim = getattr(org_module, "in_features", None)
        out_dim = getattr(org_module, "out_features", None)
        if in_dim is None or out_dim is None:
            raise RuntimeError("Ortho-Hydra requires in_features/out_features.")

        self.lora_name = lora_name
        self.lora_dim = int(lora_dim)
        self.num_experts = max(2, int(ortho_hydra_num_experts))
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.multiplier = float(multiplier)
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.enabled = True
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self._last_gate: Optional[Tensor] = None

        if isinstance(alpha, Tensor):
            alpha_value = float(alpha.detach().float().item())
        else:
            alpha_value = float(alpha if alpha is not None else self.lora_dim)
        if alpha_value == 0.0:
            alpha_value = float(self.lora_dim)
        self.scale = alpha_value / float(self.lora_dim)
        self.register_buffer("alpha", torch.tensor(alpha_value, dtype=torch.float32))

        p_bases, q_basis, disjoint = self._build_svd_bases(
            org_module.weight.detach(),  # type: ignore[attr-defined]
            self.num_experts,
            self.lora_dim,
            int(ortho_hydra_svd_niter),
        )
        self.register_buffer("P_bases", p_bases)
        self.register_buffer("Q_basis", q_basis)
        self.register_buffer(
            "ortho_hydra_disjoint_basis",
            torch.tensor(1 if disjoint else 0, dtype=torch.int64),
        )

        self.S_q = nn.Parameter(torch.zeros(self.lora_dim, self.lora_dim))
        self.S_p = nn.Parameter(
            torch.zeros(self.num_experts, self.lora_dim, self.lora_dim)
        )
        self.lambda_layer = nn.Parameter(torch.zeros(1, self.lora_dim))

        self.router = nn.Linear(self.lora_dim, self.num_experts, bias=True)
        with torch.no_grad():
            self.router.weight.zero_()
            init_std = float(ortho_hydra_router_init_std)
            if init_std > 0.0:
                nn.init.normal_(self.router.weight, mean=0.0, std=init_std)
            self.router.bias.zero_()

        self.register_buffer(
            "_eye_r",
            torch.eye(self.lora_dim, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _build_svd_bases(
        weight: Tensor,
        num_experts: int,
        rank: int,
        svd_niter: int,
    ) -> Tuple[Tensor, Tensor, bool]:
        rows = int(weight.shape[0])
        cols = int(weight.reshape(weight.shape[0], -1).shape[1])
        max_rank = min(rows, cols)
        if rank > max_rank:
            raise ValueError(
                f"Ortho-Hydra rank {rank} exceeds min(out_dim={rows}, in_dim={cols})={max_rank}."
            )

        target_cols = int(num_experts * rank)
        disjoint = target_cols <= max_rank
        keep_cols = target_cols if disjoint else rank
        q = min(keep_cols + 6, max_rank)
        work = weight.reshape(rows, cols).to(dtype=torch.float32)

        try:
            u, _s, v = torch.svd_lowrank(work, q=q, niter=max(0, int(svd_niter)))
        except Exception:
            logger.warning(
                "Ortho-Hydra randomized SVD failed; falling back to exact SVD.",
                exc_info=True,
            )
            u, _s, vh = torch.linalg.svd(work, full_matrices=False)
            v = vh.transpose(0, 1)

        q_basis = v[:, :rank].transpose(0, 1).contiguous()
        if disjoint:
            p_stack = u[:, :target_cols].reshape(rows, num_experts, rank)
            p_bases = p_stack.permute(1, 0, 2).contiguous()
        else:
            logger.warning(
                "Ortho-Hydra falling back to shared P basis because num_experts * rank (%s) exceeds min(out_dim, in_dim) (%s).",
                target_cols,
                max_rank,
            )
            p_shared = u[:, :rank].contiguous()
            p_bases = p_shared.unsqueeze(0).expand(num_experts, -1, -1).contiguous()

        return p_bases.cpu(), q_basis.cpu(), disjoint

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

    @staticmethod
    def _cayley(skew_source: Tensor, eye: Tensor) -> Tensor:
        a = skew_source - skew_source.transpose(-2, -1)
        if a.dim() == 2:
            eye_in = eye
        else:
            eye_in = eye.unsqueeze(0).expand_as(a)
        return torch.linalg.solve(eye_in + a, eye_in - a)

    def _compute_gate(self, lx: Tensor) -> Tensor:
        if lx.dim() >= 3:
            batch = lx.shape[0]
            pooled = lx.reshape(batch, -1, lx.shape[-1]).float().pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx.float()
        pooled = pooled.to(device=self.router.weight.device, dtype=self.router.weight.dtype)
        return torch.softmax(self.router(pooled), dim=-1)

    def _apply_rank_dropout(self, lx: Tensor) -> Tuple[Tensor, float]:
        if self.rank_dropout is None or not self.training:
            return lx, self.scale
        dropout = float(self.rank_dropout)
        if dropout <= 0.0:
            return lx, self.scale
        keep = 1.0 - dropout
        if keep <= 0.0:
            return torch.zeros_like(lx), self.scale
        mask_shape = [int(lx.shape[0])] + [1] * max(0, lx.dim() - 2) + [self.lora_dim]
        mask = (torch.rand(mask_shape, device=lx.device) < keep).to(dtype=lx.dtype)
        return lx * mask, self.scale / keep

    def forward(self, x: Tensor) -> Tensor:
        org_forwarded = self.org_forward(x)
        if not self.enabled:
            return org_forwarded
        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < float(self.module_dropout):
                return org_forwarded

        eye = self._eye_r.to(device=self.S_q.device, dtype=torch.float32)
        skew = torch.cat([self.S_q.unsqueeze(0), self.S_p], dim=0)
        rotations = self._cayley(skew.float(), eye)
        work_dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        q_eff = rotations[0].to(device=x.device, dtype=work_dtype) @ self.Q_basis.to(
            device=x.device, dtype=work_dtype
        )
        p_eff = self.P_bases.to(device=x.device, dtype=work_dtype) @ rotations[1:].to(
            device=x.device, dtype=work_dtype
        )

        lx = F.linear(x.to(work_dtype), q_eff)
        gate = self._compute_gate(lx)
        if self.training:
            self._last_gate = gate

        lx = lx * self.lambda_layer.to(device=lx.device, dtype=lx.dtype)
        if self.dropout is not None and self.training:
            lx = F.dropout(lx, p=float(self.dropout))
        lx, scale = self._apply_rank_dropout(lx)

        p_combined = torch.einsum("be,eor->bor", gate.to(work_dtype), p_eff)
        original_shape = lx.shape
        batch = int(original_shape[0])
        lx_3d = lx.reshape(batch, -1, original_shape[-1])
        out = torch.bmm(lx_3d, p_combined.transpose(1, 2))
        out = out.reshape(*original_shape[:-1], -1)
        return org_forwarded + (out * self.multiplier * scale).to(org_forwarded.dtype)

    def merge_to(
        self,
        sd: Dict[str, Tensor],
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        non_blocking: bool = False,
    ) -> None:
        del sd, dtype, device, non_blocking
        raise ValueError(
            "Ortho-Hydra is a routed adapter and cannot be statically merged without losing routing."
        )


class OrthoHydraLoRANetwork(LoRANetwork):
    """LoRA network that instantiates Ortho-Hydra modules."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Any,
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        ortho_hydra_num_experts: int = 4,
        ortho_hydra_balance_loss_weight: float = 0.001,
        ortho_hydra_balance_warmup_steps: int = 0,
        ortho_hydra_svd_niter: int = 2,
        ortho_hydra_router_init_std: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.ortho_hydra_num_experts = max(2, int(ortho_hydra_num_experts))
        self.ortho_hydra_balance_loss_weight = float(
            max(0.0, ortho_hydra_balance_loss_weight)
        )
        self.ortho_hydra_current_balance_loss_weight = (
            0.0
            if int(ortho_hydra_balance_warmup_steps) > 0
            else self.ortho_hydra_balance_loss_weight
        )
        self.ortho_hydra_balance_warmup_steps = max(
            0, int(ortho_hydra_balance_warmup_steps)
        )
        self.ortho_hydra_svd_niter = max(0, int(ortho_hydra_svd_niter))
        self.ortho_hydra_router_init_std = float(
            max(0.0, ortho_hydra_router_init_std)
        )
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
            module_class=self._create_ortho_hydra_module,
            **kwargs,
        )
        logger.info(
            "Ortho-Hydra network created (experts=%s, rank=%s, balance_weight=%s, warmup_steps=%s).",
            self.ortho_hydra_num_experts,
            lora_dim,
            self.ortho_hydra_balance_loss_weight,
            self.ortho_hydra_balance_warmup_steps,
        )

    def _create_ortho_hydra_module(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float,
        lora_dim: int,
        alpha: float,
        **kwargs: Any,
    ) -> OrthoHydraLoRAModule:
        return OrthoHydraLoRAModule(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=kwargs.get("dropout"),
            rank_dropout=kwargs.get("rank_dropout"),
            module_dropout=kwargs.get("module_dropout"),
            split_dims=kwargs.get("split_dims"),
            ortho_hydra_num_experts=self.ortho_hydra_num_experts,
            ortho_hydra_svd_niter=self.ortho_hydra_svd_niter,
            ortho_hydra_router_init_std=self.ortho_hydra_router_init_std,
        )

    def is_mergeable(self) -> bool:
        return False

    def step_ortho_hydra_balance_warmup(self, global_step: Optional[int]) -> None:
        if self.ortho_hydra_balance_warmup_steps <= 0:
            self.ortho_hydra_current_balance_loss_weight = (
                self.ortho_hydra_balance_loss_weight
            )
            return
        if global_step is None:
            return
        self.ortho_hydra_current_balance_loss_weight = (
            0.0
            if int(global_step) < self.ortho_hydra_balance_warmup_steps
            else self.ortho_hydra_balance_loss_weight
        )

    @staticmethod
    def _switch_balance(gate: Tensor) -> Tensor:
        num_experts = int(gate.shape[-1])
        expert_idx = gate.argmax(dim=-1)
        frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype))
        frac = frac / max(1, int(gate.shape[0]))
        gate_mean = gate.mean(dim=0)
        return num_experts * (frac * gate_mean).sum()

    def get_ortho_hydra_balance_loss(self) -> Tensor:
        total: Optional[Tensor] = None
        count = 0
        for lora in getattr(self, "unet_loras", []) + getattr(
            self, "text_encoder_loras", []
        ):
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            term = self._switch_balance(gate)
            total = term if total is None else total + term
            count += 1
        if total is None or count == 0:
            return next(self.parameters()).new_tensor(0.0)
        return total / float(count)

    def get_ortho_hydra_router_stats(self) -> Dict[str, float]:
        gates: List[Tensor] = []
        for lora in getattr(self, "unet_loras", []) + getattr(
            self, "text_encoder_loras", []
        ):
            gate = getattr(lora, "_last_gate", None)
            if gate is not None:
                gates.append(gate.detach().float())
        if not gates:
            return {}

        entropies = []
        dead_fractions = []
        top1_max_fractions = []
        eps = 1e-8
        for gate in gates:
            entropy = -(gate * torch.log(gate + eps)).sum(dim=-1).mean()
            entropy = entropy / math.log(max(int(gate.shape[-1]), 2))
            entropies.append(float(entropy.item()))
            top1 = gate.argmax(dim=-1)
            counts = torch.bincount(top1, minlength=int(gate.shape[-1])).float()
            fractions = counts / max(1.0, float(counts.sum().item()))
            dead_fractions.append(float((counts == 0).float().mean().item()))
            top1_max_fractions.append(float(fractions.max().item()))

        return {
            "ortho_hydra/router_entropy": float(sum(entropies) / len(entropies)),
            "ortho_hydra/dead_expert_fraction": float(
                sum(dead_fractions) / len(dead_fractions)
            ),
            "ortho_hydra/top1_max_fraction": float(
                sum(top1_max_fractions) / len(top1_max_fractions)
            ),
        }


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs: Any,
) -> OrthoHydraLoRANetwork:
    include_time_modules = _parse_bool(kwargs.get("include_time_modules"), False)

    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns"))
    if exclude_patterns is None:
        exclude_patterns = []

    excluded_parts = ["patch_embedding", "text_embedding", "norm", "head"]
    if not include_time_modules:
        excluded_parts.extend(["time_embedding", "time_projection"])
    exclude_patterns.append(r".*(" + "|".join(excluded_parts) + r").*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        WAN_TARGET_REPLACE_MODULES,
        "ortho_hydra_lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs: Any,
) -> OrthoHydraLoRANetwork:
    del vae
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    include_patterns = _parse_patterns(kwargs.get("include_patterns"))
    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns"))
    extra_include_patterns = _parse_patterns(kwargs.get("extra_include_patterns"))
    extra_exclude_patterns = _parse_patterns(kwargs.get("extra_exclude_patterns"))
    verbose = _parse_bool(kwargs.get("verbose"), False)

    network = OrthoHydraLoRANetwork(
        target_replace_modules=target_replace_modules,
        prefix=prefix,
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=float(network_alpha),
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=None,
        conv_alpha=None,
        modules_dim=kwargs.get("modules_dim"),
        modules_alpha=kwargs.get("modules_alpha"),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        verbose=verbose,
        ortho_hydra_num_experts=_parse_int(
            kwargs.get("ortho_hydra_num_experts"), 4, 2
        ),
        ortho_hydra_balance_loss_weight=_parse_float(
            kwargs.get("ortho_hydra_balance_loss_weight"), 0.001, 0.0
        ),
        ortho_hydra_balance_warmup_steps=_parse_int(
            kwargs.get("ortho_hydra_balance_warmup_steps"), 0, 0
        ),
        ortho_hydra_svd_niter=_parse_int(kwargs.get("ortho_hydra_svd_niter"), 2, 0),
        ortho_hydra_router_init_std=_parse_float(
            kwargs.get("ortho_hydra_router_init_std"), 0.01, 0.0
        ),
    )
    return network


def _detect_modules_from_weights(
    weights_sd: Dict[str, Tensor],
) -> Tuple[Dict[str, int], Dict[str, Tensor]]:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif key.endswith(".Q_basis"):
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".lambda_layer"):
            modules_dim[lora_name] = int(value.shape[-1])
    return modules_dim, modules_alpha


def _detect_num_experts(weights_sd: Dict[str, Tensor], fallback: int = 4) -> int:
    for key, value in weights_sd.items():
        if key.endswith(".P_bases") or key.endswith(".S_p"):
            return max(2, int(value.shape[0]))
    return fallback


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> OrthoHydraLoRANetwork:
    del for_inference
    return create_network_from_weights(
        WAN_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        **kwargs,
    )


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    **kwargs: Any,
) -> OrthoHydraLoRANetwork:
    if unet is None:
        raise ValueError("unet is required to create Ortho-Hydra from weights.")
    modules_dim, modules_alpha = _detect_modules_from_weights(weights_sd)
    kwargs["modules_dim"] = modules_dim
    kwargs["modules_alpha"] = modules_alpha
    kwargs["ortho_hydra_num_experts"] = _detect_num_experts(
        weights_sd,
        _parse_int(kwargs.get("ortho_hydra_num_experts"), 4, 2),
    )
    network = create_network(
        target_replace_modules,
        "ortho_hydra_lora_unet",
        multiplier,
        None,
        None,
        None,  # type: ignore[arg-type]
        text_encoders or [],
        unet,
        **kwargs,
    )
    return network
