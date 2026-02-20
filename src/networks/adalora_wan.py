import ast
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import (
    LoRAModule,
    LoRANetwork,
    WAN_TARGET_REPLACE_MODULES,
    create_arch_network_from_weights as create_base_arch_network_from_weights,
)


logger = get_logger(__name__)


class AdaLoRAModule(LoRAModule):
    """AdaLoRA module using rank-wise scaling vector E and dynamic rank masking."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        initialize: Optional[str] = None,
        pissa_niter: Optional[int] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
    ):
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=split_dims,
            initialize=initialize,
            pissa_niter=pissa_niter,
            ggpo_sigma=ggpo_sigma,
            ggpo_beta=ggpo_beta,
        )

        self.lora_e = nn.Parameter(torch.ones(self.lora_dim, dtype=torch.float32))
        self.register_buffer(
            "rank_pattern",
            torch.ones(self.lora_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ranknum",
            torch.tensor(float(self.lora_dim), dtype=torch.float32),
            persistent=False,
        )

        self._ortho_enabled = False

    @property
    def adaptive_scale(self) -> Tensor:
        alpha = self.alpha.to(dtype=torch.float32)
        return alpha / (self.ranknum.clamp_min(1e-6))

    @torch.no_grad()
    def set_rank_pattern(self, pattern: Tensor) -> None:
        if not isinstance(pattern, Tensor):
            raise TypeError("rank pattern must be a torch.Tensor")
        if pattern.numel() != self.lora_dim:
            raise ValueError(
                f"rank pattern size mismatch for {self.lora_name}: expected {self.lora_dim}, got {pattern.numel()}"
            )

        rank_pattern = pattern.detach().to(device=self.device, dtype=torch.float32).reshape(self.lora_dim)
        self.rank_pattern = rank_pattern
        self.ranknum = rank_pattern.sum().detach().to(device=self.device, dtype=torch.float32)

        # Keep deactivated singular values at zero for stability.
        self.lora_e.data.mul_(rank_pattern.to(device=self.lora_e.device, dtype=self.lora_e.dtype))

    def _apply_rank_scaling(self, lx: Tensor) -> Tensor:
        e = self.lora_e.to(device=lx.device, dtype=lx.dtype)
        p = self.rank_pattern.to(device=lx.device, dtype=lx.dtype)
        scale = e * p
        if lx.ndim == 2:
            return lx * scale.view(1, -1)
        if lx.ndim == 3:
            return lx * scale.view(1, 1, -1)
        if lx.ndim == 4:
            return lx * scale.view(1, -1, 1, 1)
        return lx

    def regularization(self):
        """Orthogonality regularization used by AdaLoRA paper/repo-style training."""
        if not getattr(self, "_ortho_enabled", False):
            return None, None

        def _to_rank_first(weight: Tensor, rank_dim: int) -> Tensor:
            return weight.to(torch.float32).movedim(rank_dim, 0).reshape(weight.shape[rank_dim], -1)

        try:
            if self.split_dims is None:
                down_weight = cast(Tensor, self.lora_down.weight)
                up_weight = cast(Tensor, self.lora_up.weight)

                a_mat = _to_rank_first(down_weight, 0)
                b_mat = _to_rank_first(up_weight, 1)

                eye_a = torch.eye(a_mat.shape[0], device=a_mat.device, dtype=a_mat.dtype)
                eye_b = torch.eye(b_mat.shape[0], device=b_mat.device, dtype=b_mat.dtype)

                reg_a = torch.norm(a_mat @ a_mat.t() - eye_a, p="fro")
                reg_b = torch.norm(b_mat @ b_mat.t() - eye_b, p="fro")
                return reg_a, reg_b

            reg_a_sum: Optional[Tensor] = None
            reg_b_sum: Optional[Tensor] = None
            for down_module, up_module in zip(self.lora_down, self.lora_up):  # type: ignore[arg-type]
                a_mat = _to_rank_first(cast(Tensor, down_module.weight), 0)
                b_mat = _to_rank_first(cast(Tensor, up_module.weight), 1)
                eye_a = torch.eye(a_mat.shape[0], device=a_mat.device, dtype=a_mat.dtype)
                eye_b = torch.eye(b_mat.shape[0], device=b_mat.device, dtype=b_mat.dtype)
                reg_a = torch.norm(a_mat @ a_mat.t() - eye_a, p="fro")
                reg_b = torch.norm(b_mat @ b_mat.t() - eye_b, p="fro")
                reg_a_sum = reg_a if reg_a_sum is None else reg_a_sum + reg_a
                reg_b_sum = reg_b if reg_b_sum is None else reg_b_sum + reg_b
            return reg_a_sum, reg_b_sum
        except Exception:
            return None, None

    def forward(self, x):
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
                dropout_scale = 1.0 / (1.0 - self.rank_dropout)
            else:
                dropout_scale = 1.0

            lx = self._apply_rank_scaling(lx)
            lx = self.lora_up(lx)
            total_scale = self.adaptive_scale.to(device=lx.device, dtype=lx.dtype) * dropout_scale
            return org_forwarded + lx * self.multiplier * total_scale

        lxs = [lora_down(x) for lora_down in self.lora_down]  # type: ignore[arg-type]

        if self.dropout is not None and self.training:
            lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

        if self.rank_dropout is not None and self.training:
            masks = [
                torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                for lx in lxs
            ]
            for idx in range(len(lxs)):
                if len(lxs[idx].size()) == 3:
                    masks[idx] = masks[idx].unsqueeze(1)
                elif len(lxs[idx].size()) == 4:
                    masks[idx] = masks[idx].unsqueeze(-1).unsqueeze(-1)
                lxs[idx] = lxs[idx] * masks[idx]
            dropout_scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            dropout_scale = 1.0

        lxs = [self._apply_rank_scaling(lx) for lx in lxs]
        lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore[arg-type]

        total_scale = self.adaptive_scale.to(device=lxs[0].device, dtype=lxs[0].dtype) * dropout_scale
        return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * total_scale


class AdaLoRANetwork(LoRANetwork):
    """AdaLoRA network with rank-allocation schedule from paper/repo concepts."""

    def __init__(
        self,
        *args,
        adalora_enabled: bool = False,
        adalora_init_rank: Optional[int] = None,
        adalora_target_rank: Optional[int] = None,
        adalora_tinit: int = 0,
        adalora_tfinal: int = 0,
        adalora_delta_t: int = 100,
        adalora_beta1: float = 0.85,
        adalora_beta2: float = 0.85,
        adalora_orth_reg_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, module_class=AdaLoRAModule, **kwargs)

        self.adalora_enabled = bool(adalora_enabled)
        self.adalora_step = 0
        self.adalora_total_steps = 0
        self.adalora_tinit = max(0, int(adalora_tinit))
        self.adalora_tfinal = max(0, int(adalora_tfinal))
        self.adalora_delta_t = max(1, int(adalora_delta_t))
        self.adalora_beta1 = float(min(0.999, max(0.0, adalora_beta1)))
        self.adalora_beta2 = float(min(0.999, max(0.0, adalora_beta2)))

        self._adalora_modules: List[AdaLoRAModule] = [
            cast(AdaLoRAModule, lora)
            for lora in (self.text_encoder_loras + self.unet_loras)
            if isinstance(lora, AdaLoRAModule) and getattr(lora, "split_dims", None) is None
        ]
        skipped_modules = sum(
            1
            for lora in (self.text_encoder_loras + self.unet_loras)
            if isinstance(lora, AdaLoRAModule) and getattr(lora, "split_dims", None) is not None
        )
        if skipped_modules > 0:
            logger.warning(
                "AdaLoRA rank allocation currently skips %s split-dims modules.",
                skipped_modules,
            )

        self._ipt: Dict[str, Tensor] = {}
        self._exp_avg_ipt: Dict[str, Tensor] = {}
        self._exp_avg_unc: Dict[str, Tensor] = {}

        init_rank_default = int(self.lora_dim)
        init_rank = int(adalora_init_rank) if adalora_init_rank is not None else init_rank_default
        target_rank = (
            int(adalora_target_rank)
            if adalora_target_rank is not None
            else max(1, init_rank_default // 2)
        )
        self.adalora_init_rank = max(1, min(init_rank, init_rank_default))
        self.adalora_target_rank = max(1, min(target_rank, self.adalora_init_rank))

        self.init_bgt = self.adalora_init_rank * len(self._adalora_modules)
        self.target_bgt = self.adalora_target_rank * len(self._adalora_modules)

        self.ortho_reg_lambda_p = float(max(0.0, adalora_orth_reg_weight))
        self.ortho_reg_lambda_q = float(max(0.0, adalora_orth_reg_weight))
        for module in self._adalora_modules:
            module._ortho_enabled = self.ortho_reg_lambda_p > 0.0 or self.ortho_reg_lambda_q > 0.0

        if self.adalora_enabled and not self._adalora_modules:
            logger.warning("AdaLoRA enabled but no compatible modules were found. Disabling AdaLoRA allocation.")
            self.adalora_enabled = False

        if self.adalora_enabled:
            self._apply_initial_rank_pattern()
        elif self._adalora_modules:
            for module in self._adalora_modules:
                module.set_rank_pattern(torch.ones(module.lora_dim, device=module.device, dtype=torch.float32))

    def _apply_initial_rank_pattern(self) -> None:
        for module in self._adalora_modules:
            pattern = torch.zeros(module.lora_dim, device=module.device, dtype=torch.float32)
            keep = max(1, min(module.lora_dim, self.adalora_init_rank))
            pattern[:keep] = 1.0
            module.set_rank_pattern(pattern)

    def prepare_network(self, args) -> None:
        if not self.adalora_enabled:
            return

        try:
            self.adalora_total_steps = max(0, int(getattr(args, "max_train_steps", 0) or 0))
        except Exception:
            self.adalora_total_steps = 0

        if self.adalora_total_steps > 0 and self.adalora_tinit >= self.adalora_total_steps:
            logger.warning(
                "AdaLoRA tinit=%s exceeds max_train_steps=%s; clamping tinit.",
                self.adalora_tinit,
                self.adalora_total_steps,
            )
            self.adalora_tinit = max(0, self.adalora_total_steps - 1)

        logger.info(
            "AdaLoRA enabled (modules=%s, init_rank=%s, target_rank=%s, tinit=%s, tfinal=%s, delta_t=%s).",
            len(self._adalora_modules),
            self.adalora_init_rank,
            self.adalora_target_rank,
            self.adalora_tinit,
            self.adalora_tfinal,
            self.adalora_delta_t,
        )

    def _param_key(self, module: AdaLoRAModule, suffix: str) -> str:
        return f"{module.lora_name}.{suffix}"

    @torch.no_grad()
    def _update_ipt(self, key: str, param: Tensor) -> None:
        if param.grad is None:
            return

        ipt = (param * param.grad).abs().detach().to(torch.float32)
        prev_avg = self._exp_avg_ipt.get(key)
        prev_unc = self._exp_avg_unc.get(key)

        if prev_avg is None:
            exp_avg = ipt
        else:
            exp_avg = self.adalora_beta1 * prev_avg + (1.0 - self.adalora_beta1) * ipt

        cur_unc = (ipt - exp_avg).abs()
        if prev_unc is None:
            exp_unc = cur_unc
        else:
            exp_unc = self.adalora_beta2 * prev_unc + (1.0 - self.adalora_beta2) * cur_unc

        self._ipt[key] = ipt
        self._exp_avg_ipt[key] = exp_avg
        self._exp_avg_unc[key] = exp_unc

    def _reduce_rank_vector(self, tensor: Tensor, rank_dim: int) -> Tensor:
        moved = tensor.movedim(rank_dim, 0)
        return moved.reshape(moved.shape[0], -1).mean(dim=1)

    def _rank_importance(self, module: AdaLoRAModule) -> Tensor:
        down_key = self._param_key(module, "lora_down.weight")
        up_key = self._param_key(module, "lora_up.weight")
        e_key = self._param_key(module, "lora_e")

        down_avg = self._exp_avg_ipt.get(down_key)
        up_avg = self._exp_avg_ipt.get(up_key)
        e_avg = self._exp_avg_ipt.get(e_key)
        down_unc = self._exp_avg_unc.get(down_key)
        up_unc = self._exp_avg_unc.get(up_key)
        e_unc = self._exp_avg_unc.get(e_key)

        if (
            down_avg is None
            or up_avg is None
            or e_avg is None
            or down_unc is None
            or up_unc is None
            or e_unc is None
        ):
            return module.rank_pattern.detach().to(torch.float32)

        down_score = self._reduce_rank_vector((down_avg * down_unc).abs(), rank_dim=0)
        up_score = self._reduce_rank_vector((up_avg * up_unc).abs(), rank_dim=1)
        e_score = (e_avg * e_unc).abs().reshape(module.lora_dim)
        return (down_score + up_score + e_score).to(torch.float32)

    def _budget_schedule(self, step: int) -> Tuple[int, bool]:
        if step <= self.adalora_tinit:
            return self.init_bgt, False

        if self.adalora_total_steps <= 0:
            if step % self.adalora_delta_t == 0:
                return self.target_bgt, True
            return self.target_bgt, False

        final_start = max(self.adalora_tinit + 1, self.adalora_total_steps - self.adalora_tfinal)

        if step > final_start:
            return self.target_bgt, True

        if final_start <= self.adalora_tinit:
            current_bgt = self.target_bgt
        else:
            mul_coeff = 1.0 - float(step - self.adalora_tinit) / float(final_start - self.adalora_tinit)
            current_bgt = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)

        do_mask = (step % self.adalora_delta_t) == 0
        return max(self.target_bgt, min(self.init_bgt, current_bgt)), do_mask

    @torch.no_grad()
    def _mask_to_budget(self, budget: int) -> None:
        if not self._adalora_modules:
            return

        all_scores: List[Tensor] = []
        all_index: List[Tuple[AdaLoRAModule, int]] = []
        for module in self._adalora_modules:
            importance = self._rank_importance(module)
            for ridx in range(module.lora_dim):
                all_scores.append(importance[ridx].reshape(()))
                all_index.append((module, ridx))

        if not all_scores:
            return

        score_tensor = torch.stack(all_scores)
        total_rank = int(score_tensor.numel())
        target = max(0, min(total_rank, int(budget)))

        masks: Dict[str, Tensor] = {
            module.lora_name: torch.zeros(module.lora_dim, device=module.device, dtype=torch.float32)
            for module in self._adalora_modules
        }

        if target > 0:
            _, keep_ids = torch.topk(score_tensor, k=target, largest=True, sorted=False)
            for idx in keep_ids.tolist():
                module, ridx = all_index[int(idx)]
                masks[module.lora_name][ridx] = 1.0

        for module in self._adalora_modules:
            module.set_rank_pattern(masks[module.lora_name])

        logger.info("AdaLoRA rank mask update at step %s: budget=%s/%s", self.adalora_step, target, total_rank)

    def on_step_start(self):
        super().on_step_start()
        if not self.adalora_enabled:
            return

        self.adalora_step += 1
        budget, do_mask = self._budget_schedule(self.adalora_step)
        if do_mask:
            self._mask_to_budget(budget)

    @torch.no_grad()
    def update_grad_norms(self) -> None:
        # Preserve existing LoRA GGPO behavior.
        super().update_grad_norms()

        if not self.adalora_enabled:
            return

        # In final stage, freeze allocation and skip importance refresh.
        if self.adalora_total_steps > 0 and self.adalora_step > (self.adalora_total_steps - self.adalora_tfinal):
            return

        for module in self._adalora_modules:
            self._update_ipt(self._param_key(module, "lora_down.weight"), module.lora_down.weight)
            self._update_ipt(self._param_key(module, "lora_up.weight"), module.lora_up.weight)
            self._update_ipt(self._param_key(module, "lora_e"), module.lora_e)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    return create_base_arch_network_from_weights(
        multiplier=multiplier,
        weights_sd=weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )


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
    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = include_time_modules.lower() in (
            "true",
            "1",
            "yes",
            "y",
            "on",
        )
    include_time_modules = bool(include_time_modules)

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    excluded_parts = ["patch_embedding", "text_embedding", "norm", "head"]
    if not include_time_modules:
        excluded_parts.extend(["time_embedding", "time_projection"])
    exclude_patterns.append(r".*(" + "|".join(excluded_parts) + r").*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        WAN_TARGET_REPLACE_MODULES,
        "adalora_unet",
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
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        conv_alpha = 1.0 if conv_alpha is None else float(conv_alpha)

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
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)
    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = include_time_modules.lower() in (
            "true",
            "1",
            "yes",
            "y",
            "on",
        )
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

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

    initialize = kwargs.get("initialize", "kaiming")
    if initialize is None:
        initialize = "kaiming"
    initialize = str(initialize).strip().lower()
    pissa_niter = kwargs.get("pissa_niter", None)
    if pissa_niter is not None:
        try:
            pissa_niter = int(pissa_niter)
            if pissa_niter <= 0:
                pissa_niter = None
        except Exception:
            pissa_niter = None
    if initialize.startswith("pissa_niter_"):
        if pissa_niter is None:
            try:
                pissa_niter = int(initialize.rsplit("_", 1)[-1])
            except Exception:
                pissa_niter = None
        initialize = "pissa"
    elif initialize == "pissa_niter":
        initialize = "pissa"
    elif initialize in {"", "default"}:
        initialize = "kaiming"
    elif initialize != "pissa":
        initialize = "kaiming"

    adalora_enabled = str(kwargs.get("adalora_enabled", "false")).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    adalora_init_rank = kwargs.get("adalora_init_rank", network_dim)
    adalora_target_rank = kwargs.get("adalora_target_rank", max(1, int(network_dim) // 2))
    adalora_tinit = kwargs.get("adalora_tinit", 0)
    adalora_tfinal = kwargs.get("adalora_tfinal", 0)
    adalora_delta_t = kwargs.get("adalora_delta_t", 100)
    adalora_beta1 = kwargs.get("adalora_beta1", 0.85)
    adalora_beta2 = kwargs.get("adalora_beta2", 0.85)
    adalora_orth_reg_weight = kwargs.get("adalora_orth_reg_weight", 0.0)

    network = AdaLoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,  # type: ignore[arg-type]
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
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        verbose=verbose,
        ggpo_sigma=cast(Optional[float], ggpo_sigma),
        ggpo_beta=cast(Optional[float], ggpo_beta),
        initialize=initialize,
        pissa_niter=cast(Optional[int], pissa_niter),
        adalora_enabled=adalora_enabled,
        adalora_init_rank=int(adalora_init_rank),
        adalora_target_rank=int(adalora_target_rank),
        adalora_tinit=int(adalora_tinit),
        adalora_tfinal=int(adalora_tfinal),
        adalora_delta_t=int(adalora_delta_t),
        adalora_beta1=float(adalora_beta1),
        adalora_beta2=float(adalora_beta2),
        adalora_orth_reg_weight=float(adalora_orth_reg_weight),
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network

