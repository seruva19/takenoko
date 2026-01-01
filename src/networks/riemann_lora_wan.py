"""Riemannian-style LoRA network module (pair-aware optimizer params + LOI hook)."""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork

logger = get_logger(__name__)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rsvd_on_products(
    grad_w_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    grad_wt_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    out_dim: int,
    in_dim: int,
    rank: int,
    oversample: int,
    n_iter: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    k = min(rank + oversample, min(out_dim, in_dim))
    omega = torch.randn((in_dim, k), device=device, dtype=dtype)
    y = grad_w_fn(omega)
    if y is None:
        return None
    q, _ = torch.linalg.qr(y, mode="reduced")
    for _ in range(max(0, n_iter)):
        z = grad_wt_fn(q)
        if z is None:
            return None
        q, _ = torch.linalg.qr(z, mode="reduced")
        y = grad_w_fn(q)
        if y is None:
            return None
        q, _ = torch.linalg.qr(y, mode="reduced")
    b = grad_wt_fn(q)
    if b is None:
        return None
    b = b.T
    u_hat, s, v_h = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    v = v_h.T
    return u, s, v


class RiemannLoRAModule(LoRAModule):
    """LoRA module that supports temporary adapter overrides for LOI."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loi_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_grad_output: Optional[torch.Tensor] = None

    def set_loi_override(
        self, z1: Optional[torch.Tensor], z2: Optional[torch.Tensor]
    ) -> None:
        if z1 is None or z2 is None:
            self._loi_override = None
        else:
            self._loi_override = (z1, z2)

    def forward(self, x):
        if self.training:
            self._last_input = x.detach()
        if self._loi_override is not None and self.training:
            if self.split_dims is not None:
                out = super().forward(x)
            else:
                z1, z2 = self._loi_override
                org_forwarded = self.org_forward(x)
                lx = torch.nn.functional.linear(x, z2.T)
                lx = torch.nn.functional.linear(lx, z1)
                out = org_forwarded + lx
        else:
            out = super().forward(x)
        if self.training and out.requires_grad:
            def _hook(grad: torch.Tensor) -> None:
                self._last_grad_output = grad.detach()
            out.register_hook(_hook)
        return out


class RiemannLoRANetwork(LoRANetwork):
    """LoRA network that emits per-pair optimizer groups and supports LOI."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: List[CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
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
            conv_lora_dim=conv_lora_dim,
            conv_alpha=conv_alpha,
            module_class=RiemannLoRAModule,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            verbose=verbose,
            ggpo_sigma=ggpo_sigma,
            ggpo_beta=ggpo_beta,
        )
        self._loi_enabled = _coerce_bool(kwargs.get("loi_enabled"), False)
        self._loi_alpha = _coerce_float(kwargs.get("loi_alpha"), 0.1)
        self._loi_oversample = _coerce_int(kwargs.get("loi_oversample"), 2)
        self._loi_power_iters = _coerce_int(kwargs.get("loi_power_iters"), 1)
        self._loi_max_elements = _coerce_int(kwargs.get("loi_max_elements"), 2000000)
        self._loi_applied = False
        self.loi_requires_backprop = self._loi_enabled
        if self.loraplus_lr_ratio is not None:
            logger.warning(
                "RiemannLoRA ignores loraplus_lr_ratio; consider using base LoRA "
                "for LoRA+ experiments."
            )

    def prepare_optimizer_params(
        self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.requires_grad_(True)
        params: List[Dict[str, Any]] = []
        lr_descriptions: List[str] = []

        for lora in self.unet_loras:
            if not hasattr(lora, "lora_down") or not hasattr(lora, "lora_up"):
                continue
            lora_down = lora.lora_down.weight
            lora_up = lora.lora_up.weight
            group = {
                "params": [lora_down, lora_up],
                "lora_down": lora_down,
                "lora_up": lora_up,
                "lora_module": lora,
                "pair_name": getattr(lora, "lora_name", "unknown"),
                "rank": getattr(lora, "lora_dim", None),
            }
            if unet_lr is not None and unet_lr != 0:
                group["lr"] = unet_lr
            params.append(group)
            lr_descriptions.append("unet")

        return params, lr_descriptions

    def _set_loi_override_for(
        self,
        lora: RiemannLoRAModule,
        z1: Optional[torch.Tensor],
        z2: Optional[torch.Tensor],
    ) -> None:
        lora.set_loi_override(z1, z2)

    def maybe_apply_loi_init(
        self,
        global_step: int,
        grads_ready: bool,
        extra_backward_fn: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> bool:
        if self._loi_applied or not self._loi_enabled:
            return False
        if not grads_ready or global_step != 0:
            return False
        if extra_backward_fn is None:
            return False

        applied = 0
        for lora in self.unet_loras:
            if not hasattr(lora, "lora_down") or not hasattr(lora, "lora_up"):
                continue
            lora_down = lora.lora_down.weight
            lora_up = lora.lora_up.weight
            if lora_up.ndim != 2 or lora_down.ndim != 2:
                continue
            out_dim, in_dim = lora_up.shape[0], lora_down.shape[1]
            if out_dim * in_dim > self._loi_max_elements:
                continue
            r = lora_down.shape[0]
            target_rank = min(2 * r, min(out_dim, in_dim))

            def grad_w_fn(mat: torch.Tensor) -> Optional[torch.Tensor]:
                k = mat.shape[1]
                z2 = mat.detach().clone().requires_grad_(True)
                z1 = torch.zeros(
                    (out_dim, k), device=lora_up.device, dtype=lora_up.dtype
                ).requires_grad_(True)
                def _backward():
                    self._set_loi_override_for(lora, z1, z2)
                    return (z1, z2)
                adapter_state = extra_backward_fn(_backward)
                grad_w = None
                if adapter_state is not None:
                    grad_w = adapter_state[0].grad
                self._set_loi_override_for(lora, None, None)
                if grad_w is None:
                    return None
                return grad_w.to(dtype=torch.float32)

            def grad_wt_fn(mat: torch.Tensor) -> Optional[torch.Tensor]:
                k = mat.shape[1]
                z1 = mat.detach().clone().requires_grad_(True)
                z2 = torch.zeros(
                    (in_dim, k), device=lora_down.device, dtype=lora_down.dtype
                ).requires_grad_(True)
                def _backward():
                    self._set_loi_override_for(lora, z1, z2)
                    return (z1, z2)
                adapter_state = extra_backward_fn(_backward)
                grad_w = None
                if adapter_state is not None:
                    grad_w = adapter_state[1].grad
                self._set_loi_override_for(lora, None, None)
                if grad_w is None:
                    return None
                return grad_w.to(dtype=torch.float32)

            u_s_v = _rsvd_on_products(
                grad_w_fn,
                grad_wt_fn,
                out_dim,
                in_dim,
                rank=target_rank,
                oversample=self._loi_oversample,
                n_iter=self._loi_power_iters,
                device=lora_up.device,
                dtype=torch.float32,
            )
            if u_s_v is None:
                continue
            u, _, v = u_s_v
            if u.shape[1] < r or v.shape[1] < 2 * r:
                continue
            u1 = u[:, :r]
            v2 = v[:, r : 2 * r]
            alpha = float(self._loi_alpha)
            scale = math.sqrt(abs(alpha)) if alpha != 0 else 0.0
            sign = 1.0 if alpha >= 0 else -1.0
            new_up = (u1 * (scale * sign)).to(lora_up.dtype)
            new_down = (scale * v2.T).to(lora_down.dtype)
            lora_up.copy_(new_up)
            lora_down.copy_(new_down)
            applied += 1

        if applied > 0:
            self._loi_applied = True
            logger.info("RiemannLoRA LOI applied to %d LoRA pairs.", applied)
            return True
        return False


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
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    ggpo_sigma = kwargs.get("ggpo_sigma", None)
    ggpo_beta = kwargs.get("ggpo_beta", None)
    try:
        ggpo_sigma = float(ggpo_sigma) if ggpo_sigma is not None else None
    except Exception:
        ggpo_sigma = None
    try:
        ggpo_beta = float(ggpo_beta) if ggpo_beta is not None else None
    except Exception:
        ggpo_beta = None

    network = RiemannLoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,  # type: ignore
        unet,
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
        ggpo_sigma=ggpo_sigma,
        ggpo_beta=ggpo_beta,
        loi_enabled=kwargs.get("loi_enabled"),
        loi_alpha=kwargs.get("loi_alpha"),
        loi_oversample=kwargs.get("loi_oversample"),
        loi_power_iters=kwargs.get("loi_power_iters"),
        loi_max_elements=kwargs.get("loi_max_elements"),
    )

    return network
