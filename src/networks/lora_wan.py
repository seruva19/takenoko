## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora.py (Apache)

# LoRA network module: currently conv2d is not fully supported
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import ast
import math
import os
import re
from typing import Dict, List, Optional, Type, Union, cast
from torch import Tensor
from transformers import CLIPTextModel
import numpy as np
import torch
import torch.nn as nn

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


WAN_TARGET_REPLACE_MODULES: list[str] = ["WanAttentionBlock"]


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

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
        # LoRA-GGPO parameters (optional)
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of multi-head attention.
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(
                    in_dim, self.lora_dim, kernel_size, stride, padding, bias=False  # type: ignore
                )
                self.lora_up = torch.nn.Conv2d(
                    self.lora_dim, out_dim, (1, 1), (1, 1), bias=False  # type: ignore
                )
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)  # type: ignore
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)  # type: ignore

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported
            assert (
                sum(split_dims) == out_dim
            ), "sum of split_dims must be equal to out_dim"
            assert (
                org_module.__class__.__name__ == "Linear"
            ), "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_dim, self.lora_dim, bias=False)  # type: ignore
                    for _ in range(len(split_dims))
                ]
            )
            self.lora_up = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.lora_dim, split_dim, bias=False)
                    for split_dim in split_dims
                ]
            )
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))  # type: ignore
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)  # type: ignore

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # for save/load

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # === GGPO setup ===
        self.ggpo_sigma: Optional[float] = (
            float(ggpo_sigma) if ggpo_sigma is not None else None
        )
        self.ggpo_beta: Optional[float] = (
            float(ggpo_beta) if ggpo_beta is not None else None
        )
        # Only enable GGPO for Linear layers where shapes are well-defined
        self._ggpo_enabled: bool = (
            self.ggpo_sigma is not None
            and self.ggpo_beta is not None
            and org_module.__class__.__name__ == "Linear"
        )
        self._org_module_shape: Optional[torch.Size] = None
        self._perturbation_norm_factor: Optional[float] = None
        self._org_weight_row_norm_estimate: Optional[Tensor] = None
        self.combined_weight_norms: Optional[Tensor] = None
        self.grad_norms: Optional[Tensor] = None

        if self._ggpo_enabled:
            try:
                self._org_module_shape = org_module.weight.shape  # type: ignore[attr-defined]
                # Normalization by sqrt(num_rows)
                if self._org_module_shape is None:
                    raise RuntimeError(
                        "Missing org module shape for GGPO initialization"
                    )
                rows0 = int(self._org_module_shape[0])
                self._perturbation_norm_factor = 1.0 / math.sqrt(float(rows0))
                self._initialize_org_weight_norm_estimate(org_module.weight)  # type: ignore[arg-type]
            except Exception:
                # Fail-closed if any unexpected module structure
                self._ggpo_enabled = False

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def _initialize_org_weight_norm_estimate(self, org_weight: Tensor) -> None:
        """Estimate average per-row L2 norm of the frozen base weight (Linear only).

        Uses a random subset of rows to avoid scanning large matrices every step.
        """
        try:
            num_rows: int = int(org_weight.shape[0])
            sample_size: int = min(1024, num_rows)
            if sample_size <= 0:
                return
            row_indices = torch.randperm(num_rows, device=org_weight.device)[
                :sample_size
            ]
            sampled_rows = org_weight.index_select(0, row_indices).to(
                dtype=torch.float32
            )
            # Flatten across input dimension for per-row L2 norm
            sampled_norms = torch.linalg.vector_norm(
                sampled_rows.flatten(1), ord=2, dim=1
            )
            estimate = sampled_norms.mean()
            # Keep as 1-element tensor on module device for broadcasting
            self._org_weight_row_norm_estimate = estimate.detach().to(
                device=self.device, dtype=torch.float32
            )
        except Exception:
            # Non-fatal: disable GGPO if estimation fails
            self._ggpo_enabled = False

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
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

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            # GGPO perturbation (Linear only), applied in training
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
                        # per-row scale: sigma*||W'|| + beta*||grad||
                        sigma = cast(float, self.ggpo_sigma)
                        beta = cast(float, self.ggpo_beta)
                        perturbation_scale: Tensor = (
                            sigma * self.combined_weight_norms + beta * self.grad_norms
                        ).to(device=self.device, dtype=torch.float32)
                        # normalize by sqrt(num_rows)
                        norm_factor = cast(float, self._perturbation_norm_factor)
                        perturbation_scale = perturbation_scale * norm_factor
                        # Random Gaussian perturbation with per-row scaling
                        shape = cast(torch.Size, self._org_module_shape)
                        pert = torch.randn(
                            shape,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        # Broadcast per-row factor to full weight shape (out, in)
                        pert = pert * perturbation_scale.to(self.dtype)
                    # Apply linear with perturbation as weight
                    perturbation_out = torch.nn.functional.linear(x, pert)
                    return (
                        org_forwarded + lx * self.multiplier * scale + perturbation_out
                    )
                except Exception:
                    # On any failure, fall back to standard forward
                    return org_forwarded + lx * self.multiplier * scale
            else:
                return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]  # type: ignore

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                    for lx in lxs
                ]
                for i in range(len(lxs)):
                    if len(lx.size()) == 3:  # type: ignore
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lx.size()) == 4:  # type: ignore
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # for reference
        self.enabled = True
        self.network: LoRANetwork = None  # type: ignore

    def set_network(self, network):
        self.network = network

    # merge weight to org_module
    # def merge_to(self, sd, dtype, device, non_blocking=False):
    #     if torch.cuda.is_available():
    #         stream = torch.cuda.Stream(device=device)
    #         with torch.cuda.stream(stream):
    #             print(f"merge_to {self.lora_name}")
    #             self._merge_to(sd, dtype, device, non_blocking)
    #             torch.cuda.synchronize(device=device)
    #             print(f"merge_to {self.lora_name} done")
    #         torch.cuda.empty_cache()
    #     else:
    #         self._merge_to(sd, dtype, device, non_blocking)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(
            device, dtype=torch.float, non_blocking=non_blocking
        )  # for calculation

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(
                device, dtype=torch.float, non_blocking=non_blocking
            )
            up_weight = sd["lora_up.weight"].to(
                device, dtype=torch.float, non_blocking=non_blocking
            )

            # merge weight
            if len(weight.size()) == 2:
                # linear
                weight = (
                    weight + self.multiplier * (up_weight @ down_weight) * self.scale
                )
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + self.multiplier * conved * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(
                org_device, dtype=dtype
            )  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(
                    device, torch.float, non_blocking=non_blocking
                )  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(
                    device, torch.float, non_blocking=non_blocking
                )  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros(
                    (total_dims, up_weight.size(0)), device=device, dtype=torch.float
                )
                padded_up_weight[
                    sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])
                ] = up_weight

                # merge weight
                weight = (
                    weight + self.multiplier * (up_weight @ down_weight) * self.scale
                )

            # set weight to org_module
            org_sd["weight"] = weight.to(
                org_device, dtype
            )  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)

    # return weight for merge
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:  # type: ignore
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale  # type: ignore
        elif down_weight.size()[2:4] == (1, 1):  # type: ignore
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))  # type: ignore
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight  # type: ignore
            ).permute(
                1, 0, 2, 3
            )  # type: ignore
            weight = self.multiplier * conved * self.scale

        return weight

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]  # type: ignore
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore
            return (
                self.org_forward(x)
                + torch.cat(lxs, dim=-1) * self.multiplier * self.scale
            )

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'img_mod', 'txt_mod' or 'modulation' in the name
    exclude_patterns.append(
        r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"
    )

    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        WAN_TARGET_REPLACE_MODULES,
        "lora_unet",
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
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # TODO generic rank/dim setting with regular expression

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # regular expression for module selection: exclude and include
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    # Parse GGPO parameters (may arrive as strings via network_args)
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

    network = LoRANetwork(
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
        ggpo_sigma=cast(Optional[float], ggpo_sigma),
        ggpo_beta=cast(Optional[float], ggpo_beta),
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    # loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    # loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    # loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    # loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if (
        loraplus_lr_ratio is not None
    ):  # or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(
            loraplus_lr_ratio
        )  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


class LoRANetwork(torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        # LoRA-GGPO parameters
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix

        # Store GGPO params to pass into modules
        self.ggpo_sigma: Optional[float] = (
            float(ggpo_sigma) if ggpo_sigma is not None else None
        )
        self.ggpo_beta: Optional[float] = (
            float(ggpo_beta) if ggpo_beta is not None else None
        )

        self.loraplus_lr_ratio = None
        # self.loraplus_unet_lr_ratio = None
        # self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            # if self.conv_lora_dim is not None:
            #     logger.info(
            #         f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
            #     )
        # if train_t5xxl:
        #     logger.info(f"train T5XXL as well")

        # compile regular expression if specified
        exclude_re_patterns = []
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")
                    continue
                exclude_re_patterns.append(re_pattern)

        include_re_patterns = []
        if include_patterns is not None:
            for pattern in include_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")
                    continue
                include_re_patterns.append(re_pattern)

        # create module instances
        def create_modules(
            is_unet: bool,
            pfx: str,
            root_module: torch.nn.Module,
            target_replace_mods: Optional[List[str]] = None,
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if (
                    target_replace_mods is None
                    or module.__class__.__name__ in target_replace_mods
                ):
                    if target_replace_mods is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{pfx}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.match(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.match(original_name):
                                    included = True
                                    break
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # filter by name (not used in the current implementation)
                            if filter is not None and not filter in lora_name:
                                continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # Module specification exists
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]  # type: ignore
                            else:
                                # Normally, target all
                                if is_linear or is_conv2d_1x1:
                                    dim = (
                                        default_dim
                                        if default_dim is not None
                                        else self.lora_dim
                                    )
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # Output skipped information
                                if (
                                    is_linear
                                    or is_conv2d_1x1
                                    or (self.conv_lora_dim is not None)
                                ):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,  # type: ignore
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                ggpo_sigma=self.ggpo_sigma,
                                ggpo_beta=self.ggpo_beta,
                            )
                            loras.append(lora)

                if target_replace_mods is None:
                    break  # all modules are searched
            return loras, skipped  # type: ignore

        # # create LoRA for text encoder
        # # it is redundant to create LoRA modules even if they are not used

        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        # skipped_te = []
        # for i, text_encoder in enumerate(text_encoders):
        #     index = i
        #     if not train_t5xxl and index > 0:  # 0: CLIP, 1: T5XXL, so we skip T5XXL if train_t5xxl is False
        #         break
        #     logger.info(f"create LoRA for Text Encoder {index+1}:")
        #     text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        #     logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
        #     self.text_encoder_loras.extend(text_encoder_loras)
        #     skipped_te += skipped

        # create LoRA for U-Net
        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(  # type: ignore
            True, prefix, unet, target_replace_modules
        )

        logger.info(f"create LoRA for U-Net/DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
        if verbose and len(skipped) > 0:  # type: ignore
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped"  # type: ignore
            )
            for name in skipped:  # type: ignore
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def prepare_network(self, args):
        """
        called after the network is created
        """
        pass

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(
        self,
        text_encoders: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(
        self,
        text_encoders,
        unet,
        weights_sd,
        dtype=None,
        device=None,
        non_blocking=False,
    ):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers is enough
            futures = []
            for lora in self.text_encoder_loras + self.unet_loras:
                sd_for_lora = {}
                for key in weights_sd.keys():
                    if key.startswith(lora.lora_name):
                        sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
                if len(sd_for_lora) == 0:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue

                # lora.merge_to(sd_for_lora, dtype, device)
                futures.append(
                    executor.submit(
                        lora.merge_to, sd_for_lora, dtype, device, non_blocking  # type: ignore
                    )
                )

        for future in futures:
            future.result()

        logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio
    ):  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_lr_ratio}")
        # logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(
        self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs
    ):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio, input_lr_scale):
            param_groups = {"lora": {}, "plus": {}, "patch_embedding": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if "patch_embedding" in name:
                        param_groups["patch_embedding"][
                            f"{lora.lora_name}.{name}"
                        ] = param
                    elif loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    elif key == "patch_embedding":
                        param_data["lr"] = lr * input_lr_scale
                    else:
                        param_data["lr"] = lr

                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append(key if key != "lora" else "")

            return params, descriptions

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras, unet_lr, self.loraplus_lr_ratio, input_lr_scale
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["unet" + (" " + d if d else "") for d in descriptions]
            )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    # ===== GGPO Orchestration =====
    @torch.no_grad()
    def update_norms(self) -> None:
        for lora in getattr(self, "text_encoder_loras", []) + getattr(self, "unet_loras", []):  # type: ignore
            if hasattr(lora, "update_norms"):
                lora.update_norms()

    @torch.no_grad()
    def update_grad_norms(self) -> None:
        for lora in getattr(self, "text_encoder_loras", []) + getattr(self, "unet_loras", []):  # type: ignore
            if hasattr(lora, "update_grad_norms"):
                lora.update_grad_norms()

    @torch.no_grad()
    def grad_norms(self) -> Optional[Tensor]:
        values: List[Tensor] = []
        for lora in getattr(self, "text_encoder_loras", []) + getattr(self, "unet_loras", []):  # type: ignore
            val = getattr(lora, "grad_norms", None)
            if isinstance(val, Tensor):
                values.append(val)
        if not values:
            return None
        return torch.cat(values, dim=0).mean()

    @torch.no_grad()
    def combined_weight_norms(self) -> Optional[Tensor]:
        values: List[Tensor] = []
        for lora in getattr(self, "text_encoder_loras", []) + getattr(self, "unet_loras", []):  # type: ignore
            val = getattr(lora, "combined_weight_norms", None)
            if isinstance(val, Tensor):
                values.append(val)
        if not values:
            return None
        return torch.cat(values, dim=0).mean()

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that handles missing keys gracefully.
        This is needed because the network structure might change between
        saving and loading, especially when resuming training.
        """
        # Get the current state dict keys
        current_keys = set(self.state_dict().keys())
        saved_keys = set(state_dict.keys())

        # Analyze the key differences
        missing_from_saved = current_keys - saved_keys
        missing_from_current = saved_keys - current_keys
        matching_keys = current_keys & saved_keys

        # Log detailed information about the key differences
        logger.info(f"🔍 LoRA state dict analysis:")
        logger.info(f"   Current network has {len(current_keys)} keys")
        logger.info(f"   Saved state has {len(saved_keys)} keys")
        logger.info(f"   Matching keys: {len(matching_keys)}")
        logger.info(f"   Missing from saved: {len(missing_from_saved)}")
        logger.info(f"   Missing from current: {len(missing_from_current)}")

        # Filter the input state dict to only include keys that exist in the current network
        filtered_state_dict = {}
        ignored_keys = []

        for key, value in state_dict.items():
            if key in current_keys:
                filtered_state_dict[key] = value
            else:
                ignored_keys.append(key)

        # Log information about the filtering
        if ignored_keys:
            logger.warning(
                f"LoRANetwork: {len(ignored_keys)} keys from saved state "
                f"are not present in current network structure and will be ignored."
            )

            # Show first few ignored keys for debugging
            sample_ignored = ignored_keys[:10]
            logger.info(f"   Sample ignored keys: {sample_ignored}")

        # Check if we have a reasonable number of matching keys
        if len(filtered_state_dict) == 0:
            logger.error(
                "❌ No matching keys found between saved state and current network!"
            )
            logger.error("This indicates a fundamental mismatch in network structure.")
            return super().load_state_dict({}, strict=False)

        match_ratio = len(filtered_state_dict) / len(current_keys)
        logger.info(
            f"✅ Loading {len(filtered_state_dict)} keys (match ratio: {match_ratio:.2%})"
        )

        if match_ratio < 0.5:
            logger.warning(
                f"⚠️  Low match ratio ({match_ratio:.2%}). This may indicate significant "
                f"network structure changes between save and load."
            )

        # Call parent's load_state_dict with filtered state dict
        result = super().load_state_dict(filtered_state_dict, strict=False)

        # Log the result
        if result.missing_keys:
            logger.info(
                f"📋 Final result: {len(result.missing_keys)} keys still missing from current network"
            )

        if result.unexpected_keys:
            logger.info(
                f"📋 Final result: {len(result.unexpected_keys)} unexpected keys"
            )

        # Final success/failure assessment
        total_expected = len(current_keys)
        total_loaded = len(filtered_state_dict) - len(result.missing_keys)
        load_success_ratio = total_loaded / total_expected if total_expected > 0 else 0

        logger.info(
            f"🎯 LoRA load success: {total_loaded}/{total_expected} keys ({load_success_ratio:.2%})"
        )

        if load_success_ratio >= 0.8:
            logger.info("✅ LoRA state loading successful!")
        elif load_success_ratio >= 0.5:
            logger.warning("⚠️  Partial LoRA state loading - some keys missing")
        else:
            logger.error("❌ LoRA state loading largely failed - most keys missing")

        return result

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from utils import model_utils

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True  # type: ignore

    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True  # type: ignore

    def pre_calculation(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(
                org_weight.device, dtype=org_weight.dtype
            )
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False  # type: ignore
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    return create_network_from_weights(
        WAN_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,  # type: ignore
        unet,  # type: ignore
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network
