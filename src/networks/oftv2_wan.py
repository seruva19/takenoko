"""OFTv2 adapter network module for WAN models."""

from __future__ import annotations

import ast
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.logger import get_logger
from modules.ramtorch_linear_factory import is_linear_like
from networks.lora_wan import LoRANetwork, WAN_TARGET_REPLACE_MODULES
from networks.oftv2_utils import OFTRotationModule


logger = get_logger(__name__)

_DEFAULT_OFTV2_BLOCK_SIZE = 32
_DEFAULT_OFTV2_EPS = 1e-4
_DEFAULT_OFTV2_NUM_NEUMANN_TERMS = 5


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


def _parse_optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


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


def _quadratic_block_size_from_n_elements(n_elements: int) -> int:
    discriminant = 1 + (8 * int(n_elements))
    root = int(math.isqrt(discriminant))
    if root * root != discriminant:
        raise ValueError(
            f"Cannot infer OFTv2 block size from n_elements={n_elements}."
        )
    block_size = (1 + root) // 2
    if block_size * (block_size - 1) // 2 != int(n_elements):
        raise ValueError(
            f"Invalid OFTv2 n_elements={n_elements}; no integer block size matches."
        )
    return block_size


class OFTv2Module(nn.Module):
    """Applies a block-diagonal orthogonal rotation to frozen base weights."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = _DEFAULT_OFTV2_BLOCK_SIZE,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        split_dims: Optional[List[int]] = None,
        initialize: Optional[str] = None,
        pissa_niter: Optional[int] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
        oftv2_block_size: Optional[int] = None,
        oftv2_coft: bool = False,
        oftv2_eps: float = _DEFAULT_OFTV2_EPS,
        oftv2_block_share: bool = False,
        oftv2_use_cayley_neumann: bool = True,
        oftv2_num_cayley_neumann_terms: int = _DEFAULT_OFTV2_NUM_NEUMANN_TERMS,
        oftv2_module_dropout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        del alpha, initialize, pissa_niter, ggpo_sigma, ggpo_beta, kwargs
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = float(multiplier)

        if split_dims is not None:
            raise ValueError("OFTv2 does not support split_dims modules.")

        self._is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self._is_linear_like = bool(
            is_linear_like(org_module) or org_module.__class__.__name__ == "Linear"
        )

        if not self._is_conv2d and not self._is_linear_like:
            raise RuntimeError(
                f"OFTv2: unsupported module type {type(org_module).__name__}."
            )

        if self._is_conv2d:
            weight = org_module.weight
            kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])
            self.in_features = int(weight.shape[1] * kernel_h * kernel_w)
            self.out_features = int(weight.shape[0])
        else:
            in_features = getattr(org_module, "in_features", None)
            out_features = getattr(org_module, "out_features", None)
            if in_features is None or out_features is None:
                raise RuntimeError(
                    "OFTv2 requires Linear-like modules with in_features/out_features."
                )
            self.in_features = int(in_features)
            self.out_features = int(out_features)

        requested_block_size = int(
            oftv2_block_size
            if oftv2_block_size is not None
            else int(lora_dim if lora_dim is not None else _DEFAULT_OFTV2_BLOCK_SIZE)
        )
        if requested_block_size <= 0:
            raise ValueError("oftv2_block_size must be >= 1.")

        actual_block_size, adjustment = self._resolve_block_size(
            self.in_features,
            requested_block_size,
        )
        self.adjustment_info = adjustment
        self.rank = int(self.in_features // actual_block_size)
        self.oftv2_block_size = int(actual_block_size)
        self.oftv2_coft = bool(oftv2_coft)
        self.oftv2_eps = float(oftv2_eps)
        self.oftv2_block_share = bool(oftv2_block_share)
        self.oftv2_use_cayley_neumann = bool(oftv2_use_cayley_neumann)
        self.oftv2_num_cayley_neumann_terms = int(
            max(1, oftv2_num_cayley_neumann_terms)
        )
        self.oftv2_module_dropout = float(
            oftv2_module_dropout
            if oftv2_module_dropout is not None
            else (module_dropout if module_dropout is not None else 0.0)
        )

        self.register_buffer(
            "oft_block_size_buffer",
            torch.tensor(self.oftv2_block_size, dtype=torch.int64),
        )
        self.register_buffer(
            "oft_rank_buffer",
            torch.tensor(self.rank, dtype=torch.int64),
        )
        self.register_buffer(
            "oft_coft_buffer",
            torch.tensor(1 if self.oftv2_coft else 0, dtype=torch.int64),
        )
        self.register_buffer(
            "oft_coft_eps_buffer",
            torch.tensor(self.oftv2_eps, dtype=torch.float32),
        )
        self.register_buffer(
            "oft_block_share_buffer",
            torch.tensor(1 if self.oftv2_block_share else 0, dtype=torch.int64),
        )
        self.register_buffer(
            "oft_use_cayley_neumann_buffer",
            torch.tensor(
                1 if self.oftv2_use_cayley_neumann else 0,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "oft_num_cayley_neumann_terms_buffer",
            torch.tensor(self.oftv2_num_cayley_neumann_terms, dtype=torch.int64),
        )

        if dropout is not None:
            logger.warning(
                "OFTv2 ignores network_dropout/neuron dropout for %s; use oftv2_module_dropout instead.",
                self.lora_name,
            )
        if rank_dropout is not None:
            logger.warning(
                "OFTv2 ignores rank_dropout for %s.",
                self.lora_name,
            )

        n_elements = self.oftv2_block_size * (self.oftv2_block_size - 1) // 2
        rotation_rank = 1 if self.oftv2_block_share else self.rank
        self.oft_rotation = OFTRotationModule(
            r=rotation_rank,
            n_elements=n_elements,
            block_size=self.oftv2_block_size,
            in_features=self.in_features,
            coft=self.oftv2_coft,
            coft_eps=self.oftv2_eps,
            block_share=self.oftv2_block_share,
            use_cayley_neumann=self.oftv2_use_cayley_neumann,
            num_cayley_neumann_terms=self.oftv2_num_cayley_neumann_terms,
            dropout_probability=self.oftv2_module_dropout,
        )
        nn.init.zeros_(self.oft_rotation.weight)

        self.org_module = org_module
        self.org_module_ref = [org_module]

    @staticmethod
    def _resolve_block_size(
        in_features: int,
        requested_block_size: int,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        if requested_block_size <= in_features and in_features % requested_block_size == 0:
            return requested_block_size, None

        lower = min(requested_block_size, in_features)
        while lower > 1 and (in_features % lower) != 0:
            lower -= 1

        higher = max(1, requested_block_size)
        while higher <= in_features and (in_features % higher) != 0:
            higher += 1
        if higher > in_features:
            higher = in_features

        if (requested_block_size - lower) <= (higher - requested_block_size):
            adjusted = lower
        else:
            adjusted = higher

        adjusted = max(1, int(adjusted))
        return adjusted, (requested_block_size, adjusted)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _resolve_org_module_for_merge(self) -> nn.Module:
        if hasattr(self, "org_module"):
            return self.org_module  # type: ignore[attr-defined]
        if hasattr(self, "org_module_ref") and self.org_module_ref:
            return self.org_module_ref[0]  # type: ignore[attr-defined]
        raise RuntimeError(f"OFTv2 module {self.lora_name} lost its base module reference.")

    def _build_rotation_from_sd(
        self,
        sd: Optional[Dict[str, Tensor]] = None,
    ) -> torch.Tensor:
        weight = self.oft_rotation.weight
        block_share = self.oftv2_block_share
        use_cayley_neumann = self.oftv2_use_cayley_neumann
        num_terms = self.oftv2_num_cayley_neumann_terms
        coft = self.oftv2_coft
        coft_eps = self.oftv2_eps

        if isinstance(sd, dict):
            if "oft_rotation.weight" in sd:
                weight = sd["oft_rotation.weight"].to(
                    device=self.oft_rotation.weight.device,
                    dtype=self.oft_rotation.weight.dtype,
                )
            elif "oft_R.weight" in sd:
                weight = sd["oft_R.weight"].to(
                    device=self.oft_rotation.weight.device,
                    dtype=self.oft_rotation.weight.dtype,
                )

            if "oft_block_share_buffer" in sd:
                block_share = bool(int(sd["oft_block_share_buffer"].item()))
            if "oft_use_cayley_neumann_buffer" in sd:
                use_cayley_neumann = bool(
                    int(sd["oft_use_cayley_neumann_buffer"].item())
                )
            if "oft_num_cayley_neumann_terms_buffer" in sd:
                num_terms = int(sd["oft_num_cayley_neumann_terms_buffer"].item())
            if "oft_coft_buffer" in sd:
                coft = bool(int(sd["oft_coft_buffer"].item()))
            if "oft_coft_eps_buffer" in sd:
                coft_eps = float(sd["oft_coft_eps_buffer"].item())

        return self.oft_rotation.build_rotation(
            weight_override=weight,
            block_share_override=block_share,
            use_cayley_neumann_override=use_cayley_neumann,
            num_cayley_neumann_terms_override=num_terms,
            coft_override=coft,
            coft_eps_override=coft_eps,
        )

    def _materialize_rotated_weight(
        self,
        base_weight: Tensor,
        sd: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        rotation = self._build_rotation_from_sd(sd).to(
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        reshaped = base_weight.reshape(
            base_weight.shape[0],
            self.rank,
            self.oftv2_block_size,
        )
        rotated = torch.einsum("ork,rkc->orc", reshaped, rotation)
        return rotated.reshape_as(base_weight)

    def get_weight(self, multiplier: Optional[float] = None) -> Tensor:
        effective_multiplier = (
            float(self.multiplier) if multiplier is None else float(multiplier)
        )
        org_module = self._resolve_org_module_for_merge()
        base_weight = org_module.weight.detach().to(torch.float32)
        rotated_weight = self._materialize_rotated_weight(base_weight)
        return (rotated_weight - base_weight) * effective_multiplier

    def forward(self, x: Tensor) -> Tensor:
        if self._is_linear_like:
            rotated_x = self.oft_rotation(x)
            return self.org_forward(rotated_x)

        org_module = self._resolve_org_module_for_merge()
        rotated_weight = self._materialize_rotated_weight(
            org_module.weight.to(dtype=x.dtype, device=x.device)
        )
        return F.conv2d(
            x,
            rotated_weight,
            org_module.bias,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )

    def merge_to(
        self,
        sd: Optional[Dict[str, Tensor]],
        dtype,
        device,
        non_blocking: bool = False,
    ) -> None:
        del non_blocking
        org_module = self._resolve_org_module_for_merge()
        org_sd = org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        work_device = org_device if device is None else device
        work_dtype = org_dtype if dtype is None else dtype

        rotated_weight = self._materialize_rotated_weight(
            weight.to(device=work_device, dtype=torch.float32),
            sd=sd,
        )
        merged = rotated_weight.to(device=org_device, dtype=work_dtype)
        org_sd["weight"] = merged
        org_module.load_state_dict(org_sd)


class OFTv2InfModule(OFTv2Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional["OFTv2Network"] = None

    def set_network(self, network) -> None:
        self.network = network

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return self.org_forward(x)
        return super().forward(x)


class OFTv2Network(LoRANetwork):
    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = _DEFAULT_OFTV2_BLOCK_SIZE,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Optional[Type[object]] = None,
        oftv2_block_size: int = _DEFAULT_OFTV2_BLOCK_SIZE,
        oftv2_coft: bool = False,
        oftv2_eps: float = _DEFAULT_OFTV2_EPS,
        oftv2_block_share: bool = False,
        oftv2_use_cayley_neumann: bool = True,
        oftv2_num_cayley_neumann_terms: int = _DEFAULT_OFTV2_NUM_NEUMANN_TERMS,
        oftv2_module_dropout: float = 0.0,
        oftv2_use_module_dim_for_block_size: bool = False,
        **kwargs,
    ) -> None:
        self.oftv2_block_size = int(max(1, int(oftv2_block_size)))
        self.oftv2_coft = bool(oftv2_coft)
        self.oftv2_eps = float(oftv2_eps)
        self.oftv2_block_share = bool(oftv2_block_share)
        self.oftv2_use_cayley_neumann = bool(oftv2_use_cayley_neumann)
        self.oftv2_num_cayley_neumann_terms = int(
            max(1, int(oftv2_num_cayley_neumann_terms))
        )
        self.oftv2_module_dropout = float(oftv2_module_dropout)
        self.oftv2_use_module_dim_for_block_size = bool(
            oftv2_use_module_dim_for_block_size
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
            conv_lora_dim=conv_lora_dim,
            conv_alpha=conv_alpha,
            module_class=module_class or self._create_oftv2_module,
            **kwargs,
        )

    def _create_oftv2_module(
        self,
        lora_name,
        org_module,
        multiplier,
        lora_dim,
        alpha,
        **kwargs,
    ):
        return OFTv2Module(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            oftv2_block_size=(
                int(lora_dim)
                if self.oftv2_use_module_dim_for_block_size
                else self.oftv2_block_size
            ),
            oftv2_coft=self.oftv2_coft,
            oftv2_eps=self.oftv2_eps,
            oftv2_block_share=self.oftv2_block_share,
            oftv2_use_cayley_neumann=self.oftv2_use_cayley_neumann,
            oftv2_num_cayley_neumann_terms=self.oftv2_num_cayley_neumann_terms,
            oftv2_module_dropout=self.oftv2_module_dropout,
            **kwargs,
        )

    def prepare_network(self, args) -> None:
        adjusted_modules = [
            (module.lora_name, module.adjustment_info)
            for module in (self.text_encoder_loras + self.unet_loras)
            if isinstance(module, OFTv2Module) and module.adjustment_info is not None
        ]
        logger.info(
            "OFTv2 enabled (block_size=%s, coft=%s, eps=%s, block_share=%s, cayley_neumann=%s, neumann_terms=%s, module_dropout=%s, modules=%s).",
            getattr(args, "oftv2_block_size", _DEFAULT_OFTV2_BLOCK_SIZE),
            bool(getattr(args, "oftv2_coft", False)),
            float(getattr(args, "oftv2_eps", _DEFAULT_OFTV2_EPS)),
            bool(getattr(args, "oftv2_block_share", False)),
            bool(getattr(args, "oftv2_use_cayley_neumann", True)),
            int(
                getattr(
                    args,
                    "oftv2_num_cayley_neumann_terms",
                    _DEFAULT_OFTV2_NUM_NEUMANN_TERMS,
                )
            ),
            float(getattr(args, "oftv2_module_dropout", 0.0)),
            len(self.unet_loras),
        )
        if adjusted_modules:
            preview = ", ".join(
                f"{name} ({old}->{new})"
                for name, (old, new) in adjusted_modules[:8]
            )
            logger.info(
                "OFTv2 adjusted block_size for %s module(s) to match layer widths: %s",
                len(adjusted_modules),
                preview,
            )

    def apply_max_norm_regularization(self, max_norm_value, device):
        del max_norm_value, device
        logger.warning(
            "scale_weight_norms is LoRA-specific and is ignored for OFTv2 adapters."
        )
        return 0, None, None

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                value = state_dict[key]
                value = value.detach().clone().to("cpu")
                if torch.is_floating_point(value):
                    value = value.to(dtype)
                state_dict[key] = value

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from utils import model_utils

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(
                state_dict,
                metadata,
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
            return

        torch.save(state_dict, file)


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
    del vae
    return create_network(
        WAN_TARGET_REPLACE_MODULES,
        "oftv2_unet",
        multiplier,
        network_dim,
        network_alpha,
        None,
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
    vae: Optional[nn.Module],
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae, network_alpha

    if network_dim is None:
        network_dim = _DEFAULT_OFTV2_BLOCK_SIZE

    include_patterns = _parse_patterns(kwargs.get("include_patterns", None))
    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns", None))
    extra_include_patterns = _parse_patterns(kwargs.get("extra_include_patterns", None))
    extra_exclude_patterns = _parse_patterns(kwargs.get("extra_exclude_patterns", None))

    include_time_modules = _parse_bool(kwargs.get("include_time_modules", False), False)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    oftv2_block_size = int(kwargs.get("oftv2_block_size", network_dim))
    oftv2_coft = _parse_bool(kwargs.get("oftv2_coft", False), False)
    oftv2_eps = float(kwargs.get("oftv2_eps", _DEFAULT_OFTV2_EPS))
    oftv2_block_share = _parse_bool(kwargs.get("oftv2_block_share", False), False)
    oftv2_use_cayley_neumann = _parse_bool(
        kwargs.get("oftv2_use_cayley_neumann", True),
        True,
    )
    oftv2_num_terms = int(
        kwargs.get(
            "oftv2_num_cayley_neumann_terms",
            _DEFAULT_OFTV2_NUM_NEUMANN_TERMS,
        )
    )
    oftv2_module_dropout = float(
        kwargs.get(
            "oftv2_module_dropout",
            kwargs.get("module_dropout", 0.0),
        )
    )
    if not (0.0 <= oftv2_module_dropout < 1.0):
        raise ValueError("oftv2_module_dropout must be in [0, 1).")

    rank_dropout = _parse_optional_float(kwargs.get("rank_dropout", None))
    if rank_dropout is not None:
        logger.warning("OFTv2 ignores rank_dropout=%s.", rank_dropout)

    if neuron_dropout is not None:
        logger.warning(
            "OFTv2 ignores network_dropout=%s; use oftv2_module_dropout instead.",
            neuron_dropout,
        )

    network = OFTv2Network(
        target_replace_modules=target_replace_modules,
        prefix=prefix,
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=1.0,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=oftv2_module_dropout,
        conv_lora_dim=None,
        conv_alpha=None,
        module_class=OFTv2Module,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        verbose=_parse_bool(kwargs.get("verbose", False), False),
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize="kaiming",
        pissa_niter=None,
        oftv2_block_size=oftv2_block_size,
        oftv2_coft=oftv2_coft,
        oftv2_eps=oftv2_eps,
        oftv2_block_share=oftv2_block_share,
        oftv2_use_cayley_neumann=oftv2_use_cayley_neumann,
        oftv2_num_cayley_neumann_terms=oftv2_num_terms,
        oftv2_module_dropout=oftv2_module_dropout,
    )
    return network


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


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    inferred_block_size = int(kwargs.get("oftv2_block_size", _DEFAULT_OFTV2_BLOCK_SIZE))
    inferred_block_share = _parse_bool(kwargs.get("oftv2_block_share", False), False)
    inferred_coft = _parse_bool(kwargs.get("oftv2_coft", False), False)
    inferred_eps = float(kwargs.get("oftv2_eps", _DEFAULT_OFTV2_EPS))
    inferred_use_cayley_neumann = _parse_bool(
        kwargs.get("oftv2_use_cayley_neumann", True),
        True,
    )
    inferred_num_terms = int(
        kwargs.get(
            "oftv2_num_cayley_neumann_terms",
            _DEFAULT_OFTV2_NUM_NEUMANN_TERMS,
        )
    )
    inferred_module_dropout = float(
        kwargs.get(
            "oftv2_module_dropout",
            kwargs.get("module_dropout", 0.0),
        )
    )

    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if key.endswith(".oft_block_size_buffer"):
            inferred_block_size = int(value.item())
            modules_dim[lora_name] = inferred_block_size
            modules_alpha[lora_name] = torch.tensor(1.0)
        elif key.endswith(".oft_rotation.weight"):
            block_size = _quadratic_block_size_from_n_elements(int(value.shape[1]))
            modules_dim[lora_name] = block_size
            modules_alpha[lora_name] = torch.tensor(1.0)
        elif key.endswith(".oft_block_share_buffer"):
            inferred_block_share = bool(int(value.item()))
        elif key.endswith(".oft_coft_buffer"):
            inferred_coft = bool(int(value.item()))
        elif key.endswith(".oft_coft_eps_buffer"):
            inferred_eps = float(value.item())
        elif key.endswith(".oft_use_cayley_neumann_buffer"):
            inferred_use_cayley_neumann = bool(int(value.item()))
        elif key.endswith(".oft_num_cayley_neumann_terms_buffer"):
            inferred_num_terms = int(value.item())

    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules, default=False)
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

    module_class: Type[object] = OFTv2InfModule if for_inference else OFTv2Module
    network = OFTv2Network(
        target_replace_modules=target_replace_modules,
        prefix="oftv2_unet",
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize="kaiming",
        pissa_niter=None,
        oftv2_block_size=inferred_block_size,
        oftv2_coft=inferred_coft,
        oftv2_eps=inferred_eps,
        oftv2_block_share=inferred_block_share,
        oftv2_use_cayley_neumann=inferred_use_cayley_neumann,
        oftv2_num_cayley_neumann_terms=inferred_num_terms,
        oftv2_module_dropout=inferred_module_dropout,
        oftv2_use_module_dim_for_block_size=True,
    )
    return network
