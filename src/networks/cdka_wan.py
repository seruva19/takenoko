"""CDKA WAN network module.

Implements a dedicated Kronecker-structured adapter network as a standalone
`network_module` entrypoint (`networks.cdka_wan`). This keeps CDKA behavior
separate from regular LoRA while reusing the existing WAN LoRA network shell.
"""

from __future__ import annotations

import ast
import math
from typing import Dict, List, Optional, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)


_DEFAULT_CDKA_R1 = 1
_DEFAULT_CDKA_R2 = 1
_DEFAULT_MERGE_CHUNK = 64


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_cdka_marker(initialize: Optional[str]) -> Tuple[int, int, bool]:
    marker = str(initialize or "").strip().lower()
    if not marker.startswith("cdka_"):
        return _DEFAULT_CDKA_R1, _DEFAULT_CDKA_R2, True

    tokens = marker.split("_")
    r1 = _DEFAULT_CDKA_R1
    r2 = _DEFAULT_CDKA_R2
    allow_padding = True
    idx = 0
    while idx < len(tokens):
        key = tokens[idx]
        if key == "r1" and idx + 1 < len(tokens):
            try:
                r1 = max(1, int(tokens[idx + 1]))
            except Exception:
                r1 = _DEFAULT_CDKA_R1
            idx += 2
            continue
        if key == "r2" and idx + 1 < len(tokens):
            try:
                r2 = max(1, int(tokens[idx + 1]))
            except Exception:
                r2 = _DEFAULT_CDKA_R2
            idx += 2
            continue
        if key == "pad" and idx + 1 < len(tokens):
            allow_padding = tokens[idx + 1] in {"1", "true", "yes", "on"}
            idx += 2
            continue
        idx += 1

    return r1, r2, allow_padding


class CDKAModule(LoRAModule):
    """Kronecker-structured CDKA adapter for linear-like modules."""

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
    ) -> None:
        if split_dims is not None:
            raise ValueError("CDKA does not support split_dims modules.")
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("CDKA currently supports linear-like modules only.")

        in_features = getattr(org_module, "in_features", None)
        out_features = getattr(org_module, "out_features", None)
        if in_features is None or out_features is None:
            raise RuntimeError("CDKA requires a linear-like module with in/out features.")

        cdka_r1, cdka_r2, allow_padding = _parse_cdka_marker(initialize)
        cdka_rank = max(1, int(lora_dim))
        merge_chunk_size = _DEFAULT_MERGE_CHUNK
        if pissa_niter is not None:
            try:
                merge_chunk_size = max(1, int(pissa_niter))
            except Exception:
                merge_chunk_size = _DEFAULT_MERGE_CHUNK

        # Initialize base fields, then replace projection layers with CDKA shapes.
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=cdka_rank,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=None,
            initialize="kaiming",
            pissa_niter=None,
            ggpo_sigma=None,
            ggpo_beta=None,
        )

        self._ggpo_enabled = False
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cdka_r1_value = int(cdka_r1)
        self.cdka_r2_value = int(cdka_r2)
        self.cdka_rank_value = int(cdka_rank)
        self.cdka_allow_padding = bool(allow_padding)
        self.cdka_merge_chunk_size = int(merge_chunk_size)

        if self.cdka_allow_padding:
            self.cdka_in_block = max(1, math.ceil(self.in_features / self.cdka_r2_value))
            self.cdka_out_block = max(1, math.ceil(self.out_features / self.cdka_r1_value))
        else:
            if self.in_features % self.cdka_r2_value != 0:
                raise ValueError(
                    f"CDKA requires in_features divisible by cdka_r2 when cdka_allow_padding=false "
                    f"(got in_features={self.in_features}, cdka_r2={self.cdka_r2_value})."
                )
            if self.out_features % self.cdka_r1_value != 0:
                raise ValueError(
                    f"CDKA requires out_features divisible by cdka_r1 when cdka_allow_padding=false "
                    f"(got out_features={self.out_features}, cdka_r1={self.cdka_r1_value})."
                )
            self.cdka_in_block = self.in_features // self.cdka_r2_value
            self.cdka_out_block = self.out_features // self.cdka_r1_value

        self.in_padded = self.cdka_in_block * self.cdka_r2_value
        self.out_padded = self.cdka_out_block * self.cdka_r1_value

        self.lora_down = nn.Linear(
            self.cdka_in_block,
            self.cdka_r1_value * self.cdka_rank_value,
            bias=False,
        )
        self.lora_up = nn.Linear(
            self.cdka_r2_value * self.cdka_rank_value,
            self.cdka_out_block,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.register_buffer("cdka_r1", torch.tensor(self.cdka_r1_value, dtype=torch.int64))
        self.register_buffer("cdka_r2", torch.tensor(self.cdka_r2_value, dtype=torch.int64))
        self.register_buffer(
            "cdka_rank",
            torch.tensor(self.cdka_rank_value, dtype=torch.int64),
        )
        self.register_buffer(
            "cdka_in_block_size",
            torch.tensor(self.cdka_in_block, dtype=torch.int64),
        )
        self.register_buffer(
            "cdka_out_block_size",
            torch.tensor(self.cdka_out_block, dtype=torch.int64),
        )
        self.register_buffer(
            "cdka_allow_padding_flag",
            torch.tensor(1 if self.cdka_allow_padding else 0, dtype=torch.int64),
        )

    def _pad_last_dim(self, x: Tensor, padded_dim: int) -> Tensor:
        current = int(x.shape[-1])
        if current == padded_dim:
            return x
        if current > padded_dim:
            return x[..., :padded_dim]
        return F.pad(x, (0, padded_dim - current))

    def _forward_cdka(
        self,
        x: Tensor,
        weight_a: Tensor,
        weight_b: Tensor,
        apply_rank_dropout: bool,
    ) -> Tuple[Tensor, float]:
        x_padded = self._pad_last_dim(x, self.in_padded)
        prefix_shape = x_padded.shape[:-1]
        prefix_ndim = len(prefix_shape)

        x_blocks = x_padded.reshape(*prefix_shape, self.cdka_r2_value, self.cdka_in_block)
        out_a = F.linear(x_blocks, weight_a)
        out_a = out_a.reshape(
            *prefix_shape,
            self.cdka_r2_value,
            self.cdka_r1_value,
            self.cdka_rank_value,
        )

        rank_dropout_scale = 1.0
        if apply_rank_dropout and self.rank_dropout is not None and self.training:
            keep_prob = 1.0 - float(self.rank_dropout)
            if keep_prob <= 0.0:
                return torch.zeros(
                    *prefix_shape,
                    self.out_features,
                    device=x.device,
                    dtype=x.dtype,
                ), 1.0
            mask = (
                torch.rand(
                    out_a.shape,
                    device=out_a.device,
                    dtype=torch.float32,
                )
                < keep_prob
            ).to(dtype=out_a.dtype)
            out_a = out_a * mask
            rank_dropout_scale = 1.0 / keep_prob

        permute_order: List[int] = list(range(prefix_ndim))
        permute_order.extend([prefix_ndim + 1, prefix_ndim, prefix_ndim + 2])
        out_a = out_a.permute(*permute_order).reshape(
            *prefix_shape,
            self.cdka_r1_value,
            self.cdka_r2_value * self.cdka_rank_value,
        )
        out_b = F.linear(out_a, weight_b)

        permute_back: List[int] = list(range(prefix_ndim))
        permute_back.extend([prefix_ndim + 1, prefix_ndim])
        out_b = out_b.permute(*permute_back).reshape(*prefix_shape, self.out_padded)
        out_b = out_b[..., : self.out_features]
        return out_b, rank_dropout_scale

    @torch.no_grad()
    def _delta_weight_from_factors(
        self,
        weight_a: Tensor,
        weight_b: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        in_features = int(self.in_features)
        out_features = int(self.out_features)
        chunk = max(1, int(self.cdka_merge_chunk_size))

        weight_a = weight_a.to(device=device, dtype=torch.float32)
        weight_b = weight_b.to(device=device, dtype=torch.float32)

        delta = torch.zeros((out_features, in_features), device=device, dtype=torch.float32)
        eye = torch.eye(in_features, device=device, dtype=torch.float32)
        for start in range(0, in_features, chunk):
            end = min(in_features, start + chunk)
            basis = eye[start:end, :]
            chunk_out, _ = self._forward_cdka(
                basis,
                weight_a,
                weight_b,
                apply_rank_dropout=False,
            )
            delta[:, start:end] = chunk_out.transpose(0, 1)

        return delta.to(dtype=dtype)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        delta = self._delta_weight_from_factors(
            self.lora_down.weight,
            self.lora_up.weight,
            device=self.device,
            dtype=torch.float32,
        )
        return delta * float(multiplier) * float(self.scale)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        del non_blocking  # unused

        org_sd = self.org_module.state_dict()
        org_weight = org_sd["weight"]
        org_dtype = org_weight.dtype
        org_device = org_weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        weight_a = (
            sd["lora_down.weight"]
            if isinstance(sd, dict) and "lora_down.weight" in sd
            else self.lora_down.weight
        )
        weight_b = (
            sd["lora_up.weight"]
            if isinstance(sd, dict) and "lora_up.weight" in sd
            else self.lora_up.weight
        )

        delta = self._delta_weight_from_factors(
            cast(Tensor, weight_a),
            cast(Tensor, weight_b),
            device=cast(torch.device, device),
            dtype=torch.float32,
        )
        merged = org_weight.to(device=device, dtype=torch.float32) + (
            float(self.multiplier) * float(self.scale) * delta
        )
        org_sd["weight"] = merged.to(device=org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=org_forwarded.device) < float(self.module_dropout):
                return org_forwarded

        cdka_input = x
        if self.dropout is not None and self.training:
            cdka_input = F.dropout(cdka_input, p=float(self.dropout))
        cdka_input = cdka_input.to(self.lora_down.weight.dtype)

        delta, rank_dropout_scale = self._forward_cdka(
            cdka_input,
            self.lora_down.weight,
            self.lora_up.weight,
            apply_rank_dropout=True,
        )
        delta = delta.to(dtype=org_forwarded.dtype)
        return org_forwarded + delta * self.multiplier * self.scale * rank_dropout_scale


class CDKAInfModule(CDKAModule):
    """Inference variant with adapter on/off toggle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional[CDKANetwork] = None

    def set_network(self, network) -> None:
        self.network = network

    def default_forward(self, x):
        org_forwarded = self.org_forward(x)
        cdka_input = x.to(self.lora_down.weight.dtype)
        delta, _ = self._forward_cdka(
            cdka_input,
            self.lora_down.weight,
            self.lora_up.weight,
            apply_rank_dropout=False,
        )
        return org_forwarded + delta.to(org_forwarded.dtype) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class CDKANetwork(LoRANetwork):
    def __init__(
        self,
        *args,
        module_class: Type[object] = CDKAModule,
        cdka_r1: int = _DEFAULT_CDKA_R1,
        cdka_r2: int = _DEFAULT_CDKA_R2,
        cdka_allow_padding: bool = True,
        cdka_merge_chunk_size: int = _DEFAULT_MERGE_CHUNK,
        **kwargs,
    ) -> None:
        super().__init__(*args, module_class=module_class, **kwargs)
        self.cdka_r1 = int(cdka_r1)
        self.cdka_r2 = int(cdka_r2)
        self.cdka_allow_padding = bool(cdka_allow_padding)
        self.cdka_merge_chunk_size = max(1, int(cdka_merge_chunk_size))

    def prepare_network(self, args) -> None:
        logger.info(
            "CDKA enabled (r1=%s, r2=%s, allow_padding=%s, merge_chunk_size=%s, modules=%s).",
            self.cdka_r1,
            self.cdka_r2,
            self.cdka_allow_padding,
            self.cdka_merge_chunk_size,
            len(self.text_encoder_loras) + len(self.unet_loras),
        )

    def apply_max_norm_regularization(self, max_norm_value, device):
        norms: List[float] = []
        keys_scaled = 0
        for lora in self.text_encoder_loras + self.unet_loras:
            if not isinstance(lora, CDKAModule):
                continue
            delta = lora.get_weight(multiplier=1.0).to(device=device, dtype=torch.float32)
            norm = delta.norm().clamp(min=float(max_norm_value) / 2.0)
            desired = torch.clamp(norm, max=float(max_norm_value))
            ratio = float((desired / norm).item())
            if ratio != 1.0:
                sqrt_ratio = ratio ** 0.5
                lora.lora_up.weight.data.mul_(sqrt_ratio)
                lora.lora_down.weight.data.mul_(sqrt_ratio)
                keys_scaled += 1
                delta = delta * ratio
            norms.append(float(delta.norm().item()))

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    return create_network_from_weights(
        WAN_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
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
        include_time_modules = _parse_bool(include_time_modules)
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
        "cdka_unet",
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
    del vae  # API compatibility

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
        include_time_modules = _parse_bool(include_time_modules)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    cdka_rank_raw = kwargs.get("cdka_rank", network_dim)
    cdka_r1_raw = kwargs.get("cdka_r1", _DEFAULT_CDKA_R1)
    cdka_r2_raw = kwargs.get("cdka_r2", _DEFAULT_CDKA_R2)
    cdka_allow_padding_raw = kwargs.get("cdka_allow_padding", True)
    cdka_merge_chunk_raw = kwargs.get("cdka_merge_chunk_size", _DEFAULT_MERGE_CHUNK)
    try:
        cdka_rank = max(1, int(cdka_rank_raw))
    except Exception as exc:
        raise ValueError(f"cdka_rank must be integer >= 1, got {cdka_rank_raw!r}") from exc
    try:
        cdka_r1 = max(1, int(cdka_r1_raw))
    except Exception as exc:
        raise ValueError(f"cdka_r1 must be integer >= 1, got {cdka_r1_raw!r}") from exc
    try:
        cdka_r2 = max(1, int(cdka_r2_raw))
    except Exception as exc:
        raise ValueError(f"cdka_r2 must be integer >= 1, got {cdka_r2_raw!r}") from exc
    cdka_allow_padding = _parse_bool(cdka_allow_padding_raw)
    try:
        cdka_merge_chunk_size = max(1, int(cdka_merge_chunk_raw))
    except Exception as exc:
        raise ValueError(
            f"cdka_merge_chunk_size must be integer >= 1, got {cdka_merge_chunk_raw!r}"
        ) from exc

    initialize_marker = (
        f"cdka_r1_{cdka_r1}_r2_{cdka_r2}_pad_{1 if cdka_allow_padding else 0}"
    )

    network = CDKANetwork(
        target_replace_modules,
        prefix,
        text_encoders,  # type: ignore[arg-type]
        unet,
        multiplier=multiplier,
        lora_dim=cdka_rank,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=None,
        conv_alpha=None,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        verbose=verbose,
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize_marker,
        pissa_niter=cdka_merge_chunk_size,
        module_class=CDKAModule,
        cdka_r1=cdka_r1,
        cdka_r2=cdka_r2,
        cdka_allow_padding=cdka_allow_padding,
        cdka_merge_chunk_size=cdka_merge_chunk_size,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


def _infer_cdka_layout(
    weights_sd: Dict[str, torch.Tensor],
    default_r1: int = _DEFAULT_CDKA_R1,
    default_r2: int = _DEFAULT_CDKA_R2,
) -> Tuple[int, int, bool]:
    inferred_r1 = int(default_r1)
    inferred_r2 = int(default_r2)
    allow_padding = True
    for key, value in weights_sd.items():
        if key.endswith(".cdka_r1"):
            try:
                inferred_r1 = max(1, int(value.item()))
            except Exception:
                inferred_r1 = default_r1
        elif key.endswith(".cdka_r2"):
            try:
                inferred_r2 = max(1, int(value.item()))
            except Exception:
                inferred_r2 = default_r2
        elif key.endswith(".cdka_allow_padding_flag"):
            try:
                allow_padding = bool(int(value.item()) != 0)
            except Exception:
                allow_padding = True
    return inferred_r1, inferred_r2, allow_padding


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    default_r1 = kwargs.get("cdka_r1", _DEFAULT_CDKA_R1)
    default_r2 = kwargs.get("cdka_r2", _DEFAULT_CDKA_R2)
    try:
        default_r1 = max(1, int(default_r1))
    except Exception:
        default_r1 = _DEFAULT_CDKA_R1
    try:
        default_r2 = max(1, int(default_r2))
    except Exception:
        default_r2 = _DEFAULT_CDKA_R2

    inferred_r1, inferred_r2, allow_padding = _infer_cdka_layout(
        weights_sd,
        default_r1=default_r1,
        default_r2=default_r2,
    )

    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".", 1)[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif "lora_down.weight" in key:
            raw_dim = int(value.shape[0])
            if inferred_r1 > 1 and raw_dim % inferred_r1 == 0:
                modules_dim[lora_name] = raw_dim // inferred_r1
            else:
                if inferred_r1 > 1 and raw_dim % inferred_r1 != 0:
                    logger.warning(
                        "CDKA weight %s has lora_down rows=%s not divisible by inferred cdka_r1=%s; "
                        "falling back to rank=%s and cdka_r1=1 for load compatibility.",
                        lora_name,
                        raw_dim,
                        inferred_r1,
                        raw_dim,
                    )
                    inferred_r1 = 1
                modules_dim[lora_name] = raw_dim

    module_class: Type[object] = CDKAInfModule if for_inference else CDKAModule
    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules)
    include_time_modules = bool(include_time_modules)

    if not extra_include_patterns:
        extra_include_patterns = []
    if include_time_modules:
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)
    else:
        for lora_name in modules_dim.keys():
            if "time_embedding" in lora_name or "time_projection" in lora_name:
                for pattern in ("^time_embedding\\.", "^time_projection\\."):
                    if pattern not in extra_include_patterns:
                        extra_include_patterns.append(pattern)
                break

    cdka_merge_chunk_raw = kwargs.get("cdka_merge_chunk_size", _DEFAULT_MERGE_CHUNK)
    try:
        cdka_merge_chunk_size = max(1, int(cdka_merge_chunk_raw))
    except Exception:
        cdka_merge_chunk_size = _DEFAULT_MERGE_CHUNK

    initialize_marker = (
        f"cdka_r1_{inferred_r1}_r2_{inferred_r2}_pad_{1 if allow_padding else 0}"
    )
    network = CDKANetwork(
        target_replace_modules,
        "cdka_unet",
        text_encoders,  # type: ignore[arg-type]
        unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        extra_exclude_patterns=cast(Optional[Sequence[str]], extra_exclude_patterns),
        extra_include_patterns=cast(Optional[Sequence[str]], extra_include_patterns),
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize_marker,
        pissa_niter=cdka_merge_chunk_size,
        cdka_r1=inferred_r1,
        cdka_r2=inferred_r2,
        cdka_allow_padding=allow_padding,
        cdka_merge_chunk_size=cdka_merge_chunk_size,
    )
    return network
