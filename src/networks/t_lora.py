import math
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn

# Reuse the existing LoRA implementation and API surface
from networks.lora_wan import LoRAModule, LoRANetwork
from common.logger import get_logger


logger = get_logger(__name__)


class TLoraModule(LoRAModule):
    """
    LoRA module with optional per-step rank mask (T-LoRA style).

    The mask selects the first r ranks to be active. When mask is None, behavior
    is identical to the base LoRAModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1D mask of length lora_dim; None disables masking
        self.register_buffer("_rank_mask", None, persistent=False)  # type: ignore
        # Optional orthogonal parametrization members (populated by TLoraNetwork)
        self._ortho_enabled: bool = False
        self.q_layer: Optional[nn.Linear] = None
        self.p_layer: Optional[nn.Linear] = None
        self.base_q: Optional[nn.Linear] = None
        self.base_p: Optional[nn.Linear] = None
        self.lambda_layer: Optional[torch.Tensor] = None
        self.base_lambda: Optional[torch.Tensor] = None

    @property
    def rank_mask(self) -> Optional[torch.Tensor]:
        return getattr(self, "_rank_mask", None)

    @rank_mask.setter
    def rank_mask(self, mask: Optional[torch.Tensor]) -> None:  # type: ignore
        if mask is None:
            self._rank_mask = None  # type: ignore
            return
        # Expect 1D [rank] or [1,rank]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != self.lora_dim:
            raise ValueError(
                f"Rank mask length {mask.shape[-1]} does not match lora_dim {self.lora_dim}"
            )
        # Keep mask as float to allow device/dtype alignment during forward
        self._rank_mask = mask.detach()  # type: ignore

    def _apply_rank_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the rank mask to the LoRA down-projected tensor if present."""
        mask = self.rank_mask
        if mask is None:
            return tensor
        m = mask.to(device=tensor.device, dtype=tensor.dtype)
        if tensor.dim() == 3:
            # [B, N, rank]
            m = m.unsqueeze(1)
        elif tensor.dim() == 4:
            # [B, rank, H, W]
            m = m.unsqueeze(-1).unsqueeze(-1)
        return tensor * m

    def forward(self, x):  # type: ignore[override]
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        # Orthogonal parameterization branch (enabled by TLoraNetwork)
        if getattr(self, "_ortho_enabled", False) and self.split_dims is None:
            # q_layer/p_layer/lambda_layer must be prepared by TLoraNetwork
            q_layer = cast(nn.Linear, self.q_layer)
            p_layer = cast(nn.Linear, self.p_layer)
            base_q_layer = cast(nn.Linear, self.base_q)
            base_p_layer = cast(nn.Linear, self.base_p)
            lambda_layer = cast(torch.Tensor, self.lambda_layer)
            base_lambda = cast(torch.Tensor, self.base_lambda)

            q = q_layer(x)
            # normal dropout on down-projection
            if self.dropout is not None and self.training:
                q = torch.nn.functional.dropout(q, p=self.dropout)

            # rank dropout (stochastic)
            scale = self.scale
            if self.rank_dropout is not None and self.training:
                mask = (
                    torch.rand((q.size(0), self.lora_dim), device=q.device)
                    > self.rank_dropout
                )
                if len(q.size()) == 3:
                    mask = mask.unsqueeze(1)
                elif len(q.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                q = q * mask
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))

            # Apply deterministic T-LoRA rank mask and per-rank lambda
            q = q * lambda_layer.to(device=q.device, dtype=q.dtype)
            q = self._apply_rank_mask(q)
            up = p_layer(q)

            # Subtract base branch (frozen copy)
            with torch.no_grad():
                bq = base_q_layer(x)
                bq = bq * base_lambda.to(device=bq.device, dtype=bq.dtype)
                bq = self._apply_rank_mask(bq)
                base_up = base_p_layer(bq)
            return org_forwarded + (up - base_up) * self.multiplier * scale

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout (stochastic); applied before deterministic mask
            if self.rank_dropout is not None and self.training:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                )
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            else:
                scale = self.scale

            # Deterministic rank mask (T-LoRA)
            lx = self._apply_rank_mask(lx)

            lx = self.lora_up(lx)

            # GGPO branch retained as in base module
            if (
                self.training
                and getattr(self, "_ggpo_enabled", False)
                and getattr(self, "_org_module_shape", None) is not None
                and getattr(self, "_perturbation_norm_factor", None) is not None
                and getattr(self, "combined_weight_norms", None) is not None
                and getattr(self, "grad_norms", None) is not None
            ):
                try:
                    with torch.no_grad():
                        sigma = float(self.ggpo_sigma)  # type: ignore[arg-type]
                        beta = float(self.ggpo_beta)  # type: ignore[arg-type]
                        combined = cast(torch.Tensor, self.combined_weight_norms)
                        grad = cast(torch.Tensor, self.grad_norms)
                        perturbation_scale: torch.Tensor = (
                            sigma * combined + beta * grad
                        ).to(device=self.device, dtype=torch.float32)
                        norm_factor = float(self._perturbation_norm_factor)  # type: ignore[arg-type]
                        perturbation_scale = perturbation_scale * norm_factor
                        shape = cast(torch.Size, self._org_module_shape)
                        pert = torch.randn(shape, device=self.device, dtype=self.dtype)  # type: ignore[arg-type]
                        pert = pert * perturbation_scale.to(self.dtype)
                    perturbation_out = torch.nn.functional.linear(x, pert)
                    return (
                        org_forwarded + lx * self.multiplier * scale + perturbation_out
                    )
                except Exception:
                    return org_forwarded + lx * self.multiplier * scale
            else:
                return org_forwarded + lx * self.multiplier * scale
        else:
            # Split-dims path: apply mask per-chunk after down and before up
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

            # Deterministic rank mask
            lxs = [self._apply_rank_mask(lx) for lx in lxs]

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore
            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class TLoraNetwork(LoRANetwork):
    """LoRA network that instantiates TLoraModule and exposes a per-step mask API."""

    def __init__(
        self,
        *args,
        tlora_min_rank: int = 1,
        tlora_alpha: float = 1.0,
        tlora_boundary_timestep: int = 875,
        tlora_trainer_type: str = "lora",
        sig_type: str = "last",
        ortho_from_layer: bool = False,
        ortho_reg_lambda: Optional[float] = None,
        ortho_reg_lambda_p: Optional[float] = None,
        ortho_reg_lambda_q: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Force TLoraModule usage
        super().__init__(*args, module_class=TLoraModule, **kwargs)
        self.tlora_min_rank = max(0, int(tlora_min_rank))
        self.tlora_alpha = float(tlora_alpha)
        self.tlora_boundary_timestep = max(0, int(tlora_boundary_timestep))
        self.tlora_trainer_type = str(tlora_trainer_type)
        self.sig_type = str(sig_type)
        self.ortho_from_layer = bool(ortho_from_layer)
        # Regularization weights
        lam_all = float(ortho_reg_lambda) if ortho_reg_lambda is not None else 0.0
        lam_p = float(ortho_reg_lambda_p) if ortho_reg_lambda_p is not None else None
        lam_q = float(ortho_reg_lambda_q) if ortho_reg_lambda_q is not None else None
        self.ortho_reg_lambda_p: float = float(lam_p if lam_p is not None else lam_all)
        self.ortho_reg_lambda_q: float = float(lam_q if lam_q is not None else lam_all)

        # If orthogonal LoRA is requested, retrofit modules with orthogonal parameterization
        if self.tlora_trainer_type.lower() == "ortho_lora":
            self._enable_orthogonal_parameterization()

    def _enable_orthogonal_parameterization(self) -> None:
        for lora in getattr(self, "unet_loras", []):
            # Only support Linear-based LoRA for orthogonal parameterization
            try:
                is_linear = hasattr(lora, "lora_down") and lora.lora_down.__class__.__name__ == "Linear"  # type: ignore[attr-defined]
                if not is_linear:
                    # Skip Conv2d LoRAs silently
                    continue

                # Freeze standard LoRA weights (they will not be used)
                lora.lora_down.weight.requires_grad_(False)  # type: ignore[attr-defined]
                lora.lora_up.weight.requires_grad_(False)  # type: ignore[attr-defined]

                # Infer dimensions from existing modules
                rank = int(lora.lora_down.weight.shape[0])  # type: ignore[attr-defined]
                in_features = int(lora.lora_down.weight.shape[1])  # type: ignore[attr-defined]
                out_features = int(lora.lora_up.weight.shape[0])  # type: ignore[attr-defined]

                device = lora.lora_down.weight.device  # type: ignore[attr-defined]
                dtype = lora.lora_down.weight.dtype  # type: ignore[attr-defined]

                # Create orthogonal parametrization layers
                q_layer = torch.nn.Linear(
                    in_features, rank, bias=False, device=device, dtype=dtype
                )
                p_layer = torch.nn.Linear(
                    rank, out_features, bias=False, device=device, dtype=dtype
                )
                lambda_layer = torch.nn.Parameter(
                    torch.ones(1, rank, device=device, dtype=dtype)
                )

                # Initialize via SVD
                if self.ortho_from_layer and hasattr(lora, "org_module") and hasattr(lora.org_module, "weight"):  # type: ignore[attr-defined]
                    # Use original layer weights
                    with torch.no_grad():
                        W = lora.org_module.weight.detach().to(dtype=torch.float32, device=device)  # type: ignore[attr-defined]
                        try:
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        except Exception:
                            U, S, Vh = torch.svd(W)
                        if self.sig_type == "principal":
                            q_layer.weight.copy_(Vh[:rank, :].to(dtype))
                            p_layer.weight.copy_(U[:, :rank].to(dtype))
                            lambda_layer.data.copy_(S[:rank].unsqueeze(0).to(dtype))
                        elif self.sig_type == "middle":
                            start_u = max(0, (U.shape[1] - rank) // 2)
                            start_v = max(0, (Vh.shape[0] - rank) // 2)
                            start_s = max(0, (S.shape[0] - rank) // 2)
                            q_layer.weight.copy_(
                                Vh[start_v : start_v + rank, :].to(dtype)
                            )
                            p_layer.weight.copy_(
                                U[:, start_u : start_u + rank].to(dtype)
                            )
                            lambda_layer.data.copy_(
                                S[start_s : start_s + rank].unsqueeze(0).to(dtype)
                            )
                        else:  # last
                            q_layer.weight.copy_(Vh[-rank:, :].to(dtype))
                            p_layer.weight.copy_(U[:, -rank:].to(dtype))
                            lambda_layer.data.copy_(S[-rank:].unsqueeze(0).to(dtype))
                else:
                    # Random base matrix SVD
                    with torch.no_grad():
                        base_m = torch.normal(
                            mean=0.0,
                            std=(1.0 / max(1, rank)),
                            size=(in_features, out_features),
                            device=device,
                            dtype=torch.float32,
                        )
                        U, S, Vh = torch.linalg.svd(base_m, full_matrices=False)
                        if self.sig_type == "principal":
                            q_layer.weight.copy_(U[:rank, :].to(dtype))
                            p_layer.weight.copy_(Vh[:, :rank].to(dtype))
                            lambda_layer.data.copy_(S[:rank].unsqueeze(0).to(dtype))
                        elif self.sig_type == "middle":
                            start_u = max(0, (U.shape[0] - rank) // 2)
                            start_v = max(0, (Vh.shape[1] - rank) // 2)
                            start_s = max(0, (S.shape[0] - rank) // 2)
                            q_layer.weight.copy_(
                                U[start_u : start_u + rank, :].to(dtype)
                            )
                            p_layer.weight.copy_(
                                Vh[:, start_v : start_v + rank].to(dtype)
                            )
                            lambda_layer.data.copy_(
                                S[start_s : start_s + rank].unsqueeze(0).to(dtype)
                            )
                        else:  # last
                            q_layer.weight.copy_(U[-rank:, :].to(dtype))
                            p_layer.weight.copy_(Vh[:, -rank:].to(dtype))
                            lambda_layer.data.copy_(S[-rank:].unsqueeze(0).to(dtype))

                # Frozen base copies
                base_p = torch.nn.Linear(
                    rank, out_features, bias=False, device=device, dtype=dtype
                )
                base_q = torch.nn.Linear(
                    in_features, rank, bias=False, device=device, dtype=dtype
                )
                with torch.no_grad():
                    base_p.weight.copy_(p_layer.weight.detach())
                    base_q.weight.copy_(q_layer.weight.detach())
                base_p.requires_grad_(False)
                base_q.requires_grad_(False)
                base_lambda = torch.nn.Parameter(lambda_layer.detach().clone())
                base_lambda.requires_grad = False

                # Attach to module
                lora.q_layer = q_layer  # type: ignore[attr-defined]
                lora.p_layer = p_layer  # type: ignore[attr-defined]
                lora.lambda_layer = lambda_layer  # type: ignore[attr-defined]
                lora.base_p = base_p  # type: ignore[attr-defined]
                lora.base_q = base_q  # type: ignore[attr-defined]
                lora.base_lambda = base_lambda  # type: ignore[attr-defined]
                lora._ortho_enabled = True  # type: ignore[attr-defined]

                # Optional: expose regularization() for orthogonality penalty
                def _regularization(self_lora):  # type: ignore
                    pW = self_lora.p_layer.weight  # type: ignore[attr-defined]
                    qW = self_lora.q_layer.weight  # type: ignore[attr-defined]
                    r = pW.shape[1]
                    I_p = torch.eye(r, device=pW.device, dtype=pW.dtype)
                    I_q = torch.eye(r, device=qW.device, dtype=qW.dtype)
                    p_reg = torch.sum((pW.T @ pW - I_p) ** 2)
                    q_reg = torch.sum((qW @ qW.T - I_q) ** 2)
                    return p_reg, q_reg

                if not hasattr(lora, "regularization"):
                    lora.regularization = _regularization.__get__(lora, lora.__class__)  # type: ignore[attr-defined]
            except Exception:
                # Fail-closed: leave as standard T-LoRA
                continue

    @torch.no_grad()
    def clear_rank_mask(self) -> None:
        for lora in getattr(self, "unet_loras", []):
            if isinstance(lora, TLoraModule):
                lora.rank_mask = None

    @torch.no_grad()
    def update_rank_mask(
        self, active_rank: int, device: Optional[torch.device] = None
    ) -> None:
        """Set the first active_rank channels to 1 for every TLoraModule."""
        for lora in getattr(self, "unet_loras", []):
            if not isinstance(lora, TLoraModule):
                continue
            r = max(0, min(active_rank, lora.lora_dim))
            if r <= 0:
                mask = torch.zeros(
                    (1, lora.lora_dim), device=device or next(lora.parameters()).device
                )
            elif r >= lora.lora_dim:
                mask = torch.ones(
                    (1, lora.lora_dim), device=device or next(lora.parameters()).device
                )
            else:
                mask = torch.cat(
                    [
                        torch.ones(
                            (1, r), device=device or next(lora.parameters()).device
                        ),
                        torch.zeros(
                            (1, lora.lora_dim - r),
                            device=device or next(lora.parameters()).device,
                        ),
                    ],
                    dim=-1,
                )
            lora.rank_mask = mask

    @torch.no_grad()
    def update_rank_mask_from_timesteps(
        self,
        timesteps: torch.Tensor,
        max_timestep: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        try:
            # Use the first sample's timestep to compute a global rank for the step
            t = timesteps[0]
            if t.numel() > 1:
                # Reduce to scalar if needed
                t = t.reshape(-1)[0]
            t_val = float(t.item())
        except Exception:
            t_val = 0.0

        # Apply explicit boundary for WAN-Video style expert partitioning
        if t_val > self.tlora_boundary_timestep:
            # High-noise region (boundary-max_timestep): Apply rank gating
            T_min, T_max = self.tlora_boundary_timestep, max_timestep
            T_range = max(1, T_max - T_min)
            t_normalized = (t_val - T_min) / T_range
            frac = max(0.0, min(1.0, 1.0 - t_normalized))  # Higher t → lower frac
            frac = frac**self.tlora_alpha
        else:
            # Low-noise region (0-boundary): Use full rank
            frac = 1.0

        # Update each module with computed rank fraction
        for lora in getattr(self, "unet_loras", []):
            if not isinstance(lora, TLoraModule):
                continue
            max_rank = int(lora.lora_dim)
            r_float = frac * (max_rank - self.tlora_min_rank) + self.tlora_min_rank
            r = int(max(0, min(max_rank, math.floor(r_float))))
            mask = torch.ones(
                (1, max_rank), device=device or next(lora.parameters()).device
            )
            if r < max_rank:
                mask[:, r:] = 0
            lora.rank_mask = mask


def _parse_tlora_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Accept strings from network_args and convert types
    def _pop_cast(key: str, cast_fn, default=None):
        val = kwargs.pop(key, default)
        if val is None:
            return default
        try:
            return cast_fn(val)
        except Exception:
            return default

    out["tlora_min_rank"] = _pop_cast("tlora_min_rank", int, 1)
    out["tlora_alpha"] = _pop_cast("tlora_alpha", float, 1.0)
    out["tlora_boundary_timestep"] = _pop_cast("tlora_boundary_timestep", int, 875)
    # Orthogonal LoRA options
    trainer_type = kwargs.pop("tlora_trainer_type", None)
    if isinstance(trainer_type, str):
        out["tlora_trainer_type"] = trainer_type
    sig_type = kwargs.pop("sig_type", None)
    if isinstance(sig_type, str):
        out["sig_type"] = sig_type
    ortho_from_layer = kwargs.pop("ortho_from_layer", None)
    if isinstance(ortho_from_layer, str):
        ortho_from_layer = ortho_from_layer.lower() in ["1", "true", "yes", "on"]
    if isinstance(ortho_from_layer, bool):
        out["ortho_from_layer"] = ortho_from_layer

    # Orthogonal regularization lambdas (optional)
    def _pop_float(key: str) -> Optional[float]:
        val = kwargs.pop(key, None)
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    lam = _pop_float("ortho_reg_lambda")
    lam_p = _pop_float("ortho_reg_lambda_p")
    lam_q = _pop_float("ortho_reg_lambda_q")
    if lam is not None:
        out["ortho_reg_lambda"] = lam
    if lam_p is not None:
        out["ortho_reg_lambda_p"] = lam_p
    if lam_q is not None:
        out["ortho_reg_lambda_q"] = lam_q
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
    tlora_kwargs = _parse_tlora_kwargs(dict(kwargs))
    # Normalize include/exclude patterns if passed as string literals
    import ast as _ast

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        try:
            exclude_patterns = _ast.literal_eval(exclude_patterns)
        except Exception:
            pass
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        try:
            include_patterns = _ast.literal_eval(include_patterns)
        except Exception:
            pass
    network = TLoraNetwork(
        ["WanAttentionBlock"],
        "t_lora_unet",
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
        **tlora_kwargs,
    )
    logger.info(
        f"TLoraNetwork created (min_rank={network.tlora_min_rank}, alpha={network.tlora_alpha})"
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
    # Alias to create_arch_network for compatibility
    return create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        **kwargs,
    )
