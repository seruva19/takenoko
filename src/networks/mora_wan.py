import ast
import math
from typing import Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)

_SUPPORTED_MORA_TYPES = {1, 2, 3, 4, 6}
_DEFAULT_MORA_TYPE = 1
_DEFAULT_MORA_MERGE_CHUNK = 64


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_mora_marker(initialize: Optional[str]) -> Tuple[bool, int]:
    marker = str(initialize or "").strip().lower()
    from_weights = False
    mora_type = _DEFAULT_MORA_TYPE

    if marker.startswith("mora_from_weights_type_"):
        from_weights = True
        suffix = marker.rsplit("_", 1)[-1]
        try:
            mora_type = int(suffix)
        except Exception:
            mora_type = _DEFAULT_MORA_TYPE
    elif marker.startswith("mora_train_type_"):
        suffix = marker.rsplit("_", 1)[-1]
        try:
            mora_type = int(suffix)
        except Exception:
            mora_type = _DEFAULT_MORA_TYPE

    if mora_type not in _SUPPORTED_MORA_TYPES:
        logger.warning(
            "Unsupported mora_type parsed from marker %r, falling back to %s.",
            marker,
            _DEFAULT_MORA_TYPE,
        )
        mora_type = _DEFAULT_MORA_TYPE

    return from_weights, mora_type


class MoRAModule(LoRAModule):
    """MoRA module with high-rank square update and deterministic in/out remapping."""

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
            raise ValueError(
                "MoRA does not currently support split_dims modules. "
                "Use non-split linear targets."
            )

        from_weights_mode, mora_type = _parse_mora_marker(initialize)
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("MoRA currently supports linear-like modules only.")

        in_features = getattr(org_module, "in_features", None)
        out_features = getattr(org_module, "out_features", None)
        if in_features is None or out_features is None:
            raise RuntimeError("MoRA requires linear-like module with in/out features.")

        base_rank = max(1, int(lora_dim))
        if from_weights_mode:
            effective_rank = base_rank
        else:
            effective_rank = self._compute_mora_rank(
                in_features=int(in_features),
                out_features=int(out_features),
                base_rank=base_rank,
                mora_type=mora_type,
            )

        merge_chunk_size = _DEFAULT_MORA_MERGE_CHUNK
        if pissa_niter is not None:
            try:
                merge_chunk_size = max(1, int(pissa_niter))
            except Exception:
                merge_chunk_size = _DEFAULT_MORA_MERGE_CHUNK

        # Initialize base fields, then replace LoRA down/up by MoRA square update matrix.
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=effective_rank,
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

        self.lora_down = nn.Linear(effective_rank, effective_rank, bias=False)
        self.lora_up = self.lora_down
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        self.scale = 1.0
        self._ggpo_enabled = False
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mora_rank = int(effective_rank)
        self.mora_type_value = int(mora_type)
        self.mora_merge_chunk_size = int(merge_chunk_size)

        self.register_buffer(
            "mora_type",
            torch.tensor(self.mora_type_value, dtype=torch.int64),
        )
        self.register_buffer(
            "mora_base_rank",
            torch.tensor(base_rank, dtype=torch.int64),
        )

        self._rope_cache: Dict[Tuple[int, int, str, str], Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _compute_mora_rank(
        in_features: int,
        out_features: int,
        base_rank: int,
        mora_type: int,
    ) -> int:
        rank = int(round(math.sqrt((in_features + out_features) * base_rank)))
        rank = max(1, rank)
        if mora_type == 6 and (rank % 2) == 1:
            rank += 1
        return rank

    def _pad_with_prefix(self, x: Tensor, pad_size: int) -> Tensor:
        if pad_size <= 0:
            return x
        return torch.cat([x, x[..., :pad_size]], dim=-1)

    def _rope(self, x: Tensor) -> Tensor:
        # x shape: [..., seq, dim]
        seq_len = int(x.shape[-2])
        head_dim = int(x.shape[-1])
        key = (seq_len, head_dim, str(x.device), str(x.dtype))
        cached = self._rope_cache.get(key, None)

        if cached is None:
            half = max(1, head_dim // 2)
            inv_freq = 1.0 / (
                10000.0
                ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half))
            )
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            if emb.shape[-1] < head_dim:
                emb = torch.cat([emb, torch.zeros(seq_len, 1, device=x.device)], dim=-1)
            emb = emb[:, :head_dim]
            cos = emb.cos().to(dtype=x.dtype)
            sin = emb.sin().to(dtype=x.dtype)
            cached = (cos, sin)
            self._rope_cache[key] = cached

        cos, sin = cached
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        half = head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half : half * 2]
        rotated = torch.cat((-x2, x1), dim=-1)
        if head_dim > (half * 2):
            rotated = torch.cat((rotated, x[..., half * 2 :]), dim=-1)
        return (x * cos) + (rotated * sin)

    def _project_to_rank(self, x: Tensor) -> Tensor:
        in_features = int(x.shape[-1])
        rank = int(self.mora_rank)
        mora_type = int(self.mora_type.item())

        if mora_type in {1, 4}:
            sum_inter = in_features // rank
            if in_features % rank != 0:
                pad_size = rank - (in_features % rank)
                x = self._pad_with_prefix(x, pad_size)
                sum_inter += 1
            return x.view(*x.shape[:-1], sum_inter, rank).sum(dim=-2)

        if mora_type in {2, 3}:
            reduce_cols = in_features // rank
            if in_features % rank != 0:
                pad_size = rank - (in_features % rank)
                x = self._pad_with_prefix(x, pad_size)
                reduce_cols += 1
            return x.view(*x.shape[:-1], rank, reduce_cols).sum(dim=-1)

        if mora_type == 6:
            if in_features % 2 == 1:
                x = self._pad_with_prefix(x, 1)
                in_features += 1
            sum_inter = in_features // rank
            if in_features % rank != 0:
                pad_size = rank - (in_features % rank)
                x = self._pad_with_prefix(x, pad_size)
                sum_inter += 1
            chunks = x.view(*x.shape[:-1], sum_inter, rank)
            rope_chunks = self._rope(chunks).view(*x.shape[:-1], sum_inter, rank)
            return torch.cat((rope_chunks, chunks), dim=-2).sum(dim=-2)

        # Fallback for unknown types (should not happen due parser validation).
        return x[..., :rank]

    def _expand_from_rank(self, x: Tensor, out_features: int) -> Tensor:
        rank = int(self.mora_rank)
        mora_type = int(self.mora_type.item())

        if mora_type in {1, 3}:
            repeat_time = (out_features + rank - 1) // rank
            return x.repeat_interleave(repeat_time, dim=-1)[..., :out_features]

        if mora_type in {2, 4}:
            repeat_time = (out_features + rank - 1) // rank
            reps = [1] * x.dim()
            reps[-1] = repeat_time
            return x.repeat(*reps)[..., :out_features]

        if mora_type == 6:
            repeat_time = (out_features + rank - 1) // rank
            out = x.repeat_interleave(repeat_time, dim=-1)
            if out.shape[-1] < out_features:
                repeat_time2 = (out_features + int(out.shape[-1]) - 1) // int(out.shape[-1])
                out = out.repeat_interleave(repeat_time2, dim=-1)
            return out[..., :out_features]

        return x[..., :out_features]

    def _apply_mora(self, x: Tensor, weight: Tensor) -> Tuple[Tensor, float]:
        rank_x = self._project_to_rank(x)

        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand(rank_x.shape, device=rank_x.device, dtype=torch.float32)
                > float(self.rank_dropout)
            ).to(dtype=rank_x.dtype)
            rank_x = rank_x * mask
            dropout_scale = 1.0 / (1.0 - float(self.rank_dropout))
        else:
            dropout_scale = 1.0

        rank_out = torch.nn.functional.linear(
            rank_x,
            weight.to(device=rank_x.device, dtype=rank_x.dtype),
        )
        return self._expand_from_rank(rank_out, self.out_features), dropout_scale

    @torch.no_grad()
    def _delta_weight_from_matrix(
        self,
        matrix: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        in_features = int(self.in_features)
        out_features = int(self.out_features)
        chunk = max(1, int(self.mora_merge_chunk_size))

        weight = matrix.to(device=device, dtype=torch.float32)
        delta = torch.zeros((out_features, in_features), device=device, dtype=torch.float32)
        eye = torch.eye(in_features, device=device, dtype=torch.float32)

        was_training = self.training
        self.eval()
        try:
            for start in range(0, in_features, chunk):
                end = min(in_features, start + chunk)
                basis = eye[start:end, :]
                rank_basis = self._project_to_rank(basis)
                rank_delta = torch.nn.functional.linear(rank_basis, weight)
                expanded = self._expand_from_rank(rank_delta, out_features)
                delta[:, start:end] = expanded.transpose(0, 1)
        finally:
            if was_training:
                self.train()

        return delta.to(dtype=dtype)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        delta = self._delta_weight_from_matrix(
            self.lora_down.weight,
            device=self.device,
            dtype=torch.float32,
        )
        return delta * float(multiplier) * float(self.scale)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        del non_blocking  # not used

        org_sd = self.org_module.state_dict()
        org_weight = org_sd["weight"]
        org_dtype = org_weight.dtype
        org_device = org_weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        weight_matrix = None
        if isinstance(sd, dict):
            if "lora_down.weight" in sd:
                weight_matrix = sd["lora_down.weight"]
            elif "lora_up.weight" in sd:
                weight_matrix = sd["lora_up.weight"]
        if weight_matrix is None:
            weight_matrix = self.lora_down.weight

        delta = self._delta_weight_from_matrix(
            cast(Tensor, weight_matrix),
            device=cast(torch.device, device),
            dtype=torch.float32,
        )
        merged = org_weight.to(device=device, dtype=torch.float32) + (
            float(self.multiplier) * delta * float(self.scale)
        )
        org_sd["weight"] = merged.to(device=org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        mora_input = x
        if self.dropout is not None and self.training:
            mora_input = torch.nn.functional.dropout(mora_input, p=self.dropout)

        delta, dropout_scale = self._apply_mora(mora_input, self.lora_down.weight)
        return org_forwarded + delta * self.multiplier * self.scale * dropout_scale


class MoRAInfModule(MoRAModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional[MoRANetwork] = None

    def set_network(self, network) -> None:
        self.network = network

    def default_forward(self, x):
        delta, _ = self._apply_mora(x, self.lora_down.weight)
        return self.org_forward(x) + delta * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class MoRANetwork(LoRANetwork):
    def __init__(
        self,
        *args,
        module_class: Type[object] = MoRAModule,
        mora_type: int = _DEFAULT_MORA_TYPE,
        mora_merge_chunk_size: int = _DEFAULT_MORA_MERGE_CHUNK,
        **kwargs,
    ) -> None:
        super().__init__(*args, module_class=module_class, **kwargs)
        self.mora_type = int(mora_type)
        self.mora_merge_chunk_size = max(1, int(mora_merge_chunk_size))

    def prepare_network(self, args) -> None:
        logger.info(
            "MoRA enabled (mora_type=%s, merge_chunk_size=%s, modules=%s).",
            self.mora_type,
            self.mora_merge_chunk_size,
            len(self.text_encoder_loras) + len(self.unet_loras),
        )


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
        "mora_unet",
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
    del vae  # kept for API compatibility

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
        include_time_modules = _parse_bool(include_time_modules)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    mora_type_raw = kwargs.get("mora_type", _DEFAULT_MORA_TYPE)
    try:
        mora_type = int(mora_type_raw)
    except Exception as exc:
        raise ValueError(f"mora_type must be integer, got {mora_type_raw!r}") from exc
    if mora_type not in _SUPPORTED_MORA_TYPES:
        raise ValueError(
            f"mora_type must be one of {sorted(_SUPPORTED_MORA_TYPES)}, got {mora_type}"
        )

    mora_merge_chunk_raw = kwargs.get("mora_merge_chunk_size", _DEFAULT_MORA_MERGE_CHUNK)
    try:
        mora_merge_chunk_size = max(1, int(mora_merge_chunk_raw))
    except Exception as exc:
        raise ValueError(
            f"mora_merge_chunk_size must be integer >= 1, got {mora_merge_chunk_raw!r}"
        ) from exc

    # LoRANetwork forwards `initialize` and `pissa_niter` into module construction.
    # We encode MoRA mode/type and merge chunk through those fields.
    initialize_marker = f"mora_train_type_{mora_type}"

    network = MoRANetwork(
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
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize_marker,
        pissa_niter=mora_merge_chunk_size,
        module_class=MoRAModule,
        mora_type=mora_type,
        mora_merge_chunk_size=mora_merge_chunk_size,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    inferred_mora_type = _DEFAULT_MORA_TYPE

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif "lora_down.weight" in key:
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".mora_type"):
            try:
                inferred_mora_type = int(value.item())
            except Exception:
                inferred_mora_type = _DEFAULT_MORA_TYPE

    if inferred_mora_type not in _SUPPORTED_MORA_TYPES:
        inferred_mora_type = _DEFAULT_MORA_TYPE

    module_class: Type[object] = MoRAInfModule if for_inference else MoRAModule

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

    mora_merge_chunk_raw = kwargs.get("mora_merge_chunk_size", _DEFAULT_MORA_MERGE_CHUNK)
    try:
        mora_merge_chunk_size = max(1, int(mora_merge_chunk_raw))
    except Exception:
        mora_merge_chunk_size = _DEFAULT_MORA_MERGE_CHUNK

    initialize_marker = f"mora_from_weights_type_{inferred_mora_type}"

    network = MoRANetwork(
        target_replace_modules,
        "mora_unet",
        text_encoders,  # type: ignore[arg-type]
        unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize_marker,
        pissa_niter=mora_merge_chunk_size,
        mora_type=inferred_mora_type,
        mora_merge_chunk_size=mora_merge_chunk_size,
    )
    return network
