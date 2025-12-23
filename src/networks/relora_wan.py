import ast
import math
import weakref
from typing import List, Optional, cast

import torch
import torch.nn as nn

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)


class ReLoRAModule(LoRAModule):
    """LoRA module that supports merge-and-reinit for ReLoRA training."""

    def apply_to(self) -> None:
        self._org_module_ref = weakref.ref(self.org_module)
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _get_org_module(self) -> Optional[nn.Module]:
        ref = getattr(self, "_org_module_ref", None)
        if ref is None:
            return None
        return ref()

    @torch.no_grad()
    def merge_and_reinit(self) -> None:
        org_module = self._get_org_module()
        if org_module is None or not hasattr(org_module, "weight"):
            logger.warning("ReLoRA merge skipped: missing original module reference")
            return

        org_weight = org_module.weight
        if isinstance(org_weight, torch.Tensor):
            if not torch.is_floating_point(org_weight):
                logger.warning(
                    "ReLoRA merge skipped: non-floating weight dtype %s",
                    org_weight.dtype,
                )
                return
            fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
            fp8_e5m2 = getattr(torch, "float8_e5m2", None)
            if org_weight.dtype in {fp8_e4m3, fp8_e5m2}:
                logger.warning(
                    "ReLoRA merge skipped: fp8 weight dtype %s (quantized merge not implemented)",
                    org_weight.dtype,
                )
                return

        if self.split_dims is None:
            down_weight = self.lora_down.weight.to(torch.float32)
            up_weight = self.lora_up.weight.to(torch.float32)
            delta = up_weight @ down_weight
            delta = delta * float(self.scale)

            org_weight = org_module.weight.data.to(torch.float32)
            if org_weight.ndim == 2:
                org_module.weight.data.copy_(
                    (org_weight + delta).to(org_module.weight.dtype)
                )
            elif down_weight.ndim == 4:
                if down_weight.shape[2:4] == (1, 1):
                    merged = (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    ).unsqueeze(2).unsqueeze(3)
                else:
                    merged = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3), up_weight
                    ).permute(1, 0, 2, 3)
                merged = merged * float(self.scale)
                org_module.weight.data.copy_(
                    (org_weight + merged).to(org_module.weight.dtype)
                )
            else:
                logger.warning(
                    "ReLoRA merge skipped: unsupported weight shape %s",
                    tuple(org_weight.shape),
                )
        else:
            total_dims = sum(self.split_dims)
            org_weight = org_module.weight.data.to(torch.float32)
            for i in range(len(self.split_dims)):
                down_weight = self.lora_down[i].weight.to(torch.float32)
                up_weight = self.lora_up[i].weight.to(torch.float32)
                padded_up_weight = torch.zeros(
                    (total_dims, up_weight.size(0)),
                    device=up_weight.device,
                    dtype=torch.float32,
                )
                start = sum(self.split_dims[:i])
                end = sum(self.split_dims[: i + 1])
                padded_up_weight[start:end] = up_weight
                org_weight = org_weight + padded_up_weight @ down_weight * float(
                    self.scale
                )
            org_module.weight.data.copy_(org_weight.to(org_module.weight.dtype))

        # Reinitialize LoRA weights after merge
        if self.split_dims is None:
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)


class ReLoRANetwork(LoRANetwork):
    """LoRA network with merge/reset hooks for ReLoRA training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, module_class=ReLoRAModule, **kwargs)

    @torch.no_grad()
    def merge_and_reinit(self) -> None:
        for lora in getattr(self, "text_encoder_loras", []) + getattr(
            self, "unet_loras", []
        ):
            if hasattr(lora, "merge_and_reinit"):
                lora.merge_and_reinit()

    def get_relora_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for lora in getattr(self, "text_encoder_loras", []) + getattr(
            self, "unet_loras", []
        ):
            if hasattr(lora, "lora_down"):
                params.extend(list(lora.lora_down.parameters()))
            if hasattr(lora, "lora_up"):
                params.extend(list(lora.lora_up.parameters()))
        return params


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
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.append(
        r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"
    )

    kwargs["exclude_patterns"] = exclude_patterns

    network_dim = network_dim if network_dim is not None else 4
    network_alpha = network_alpha if network_alpha is not None else 1.0

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

    network = ReLoRANetwork(
        WAN_TARGET_REPLACE_MODULES,
        "relora_unet",
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
        ggpo_sigma=cast(Optional[float], ggpo_sigma),
        ggpo_beta=cast(Optional[float], ggpo_beta),
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    logger.info(
        "ReLoRANetwork created (dim=%s, alpha=%s)",
        network_dim,
        network_alpha,
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
