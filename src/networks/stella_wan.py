"""StelLA LoRA network module with Stiefel-manifold optimizer hooks.

This implementation is fully opt-in via:
- `network_module = "networks.stella_wan"`
- `network_args = ["stella_enabled=true", ...]`

Inference behavior remains unchanged because StelLA updates are merged into
the same base-weight path as regular LoRA adapters.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)

_VALID_STELLA_RETRACTIONS = {"polar", "exp_map"}
_VALID_STELLA_INITS = {"orthonormal", "svd_major", "svd_minor", "kaiming"}
_DEFAULT_STELLA_RETRACTION = "polar"
_DEFAULT_STELLA_INIT = "orthonormal"

_STELLA_INIT_ALIASES = {
    "orthonormal": "orthonormal",
    "orthogonal": "orthonormal",
    "rando": "orthonormal",
    "random_qr": "orthonormal",
    "random": "orthonormal",
    "kaiming": "kaiming",
    "default": "kaiming",
    "svd_major": "svd_major",
    "svd_minor": "svd_minor",
}

_STELLA_UP_PATTERNS = (
    re.compile(r"(^|\.)lora_up\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_u\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_u\.[^.]+\.weight$", re.IGNORECASE),
)
_STELLA_DOWN_PATTERNS = (
    re.compile(r"(^|\.)lora_down\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_vt\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_vt\.[^.]+\.weight$", re.IGNORECASE),
)
_STELLA_S_PATTERNS = (
    re.compile(r"(^|\.)stella_s\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_s\.[^.]+\.weight$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_s$", re.IGNORECASE),
    re.compile(r"(^|\.)stella_s\.[^.]+$", re.IGNORECASE),
)
_STELLA_ALPHA_PATTERNS = (
    re.compile(r"(^|\.)alpha$", re.IGNORECASE),
    re.compile(r"(^|\.)lora_alpha$", re.IGNORECASE),
    re.compile(r"(^|\.)alpha\.[^.]+$", re.IGNORECASE),
    re.compile(r"(^|\.)lora_alpha\.[^.]+$", re.IGNORECASE),
)


def _lookup_sd_tensor(
    sd: Dict[str, Tensor],
    direct_keys: Tuple[str, ...],
    key_patterns: Tuple[re.Pattern[str], ...],
) -> Optional[Tensor]:
    if not isinstance(sd, dict) or not sd:
        return None

    lower_to_key: Dict[str, str] = {}
    for raw_key in sd.keys():
        if isinstance(raw_key, str):
            lower_to_key[raw_key.lower()] = raw_key

    for key in direct_keys:
        matched_key = lower_to_key.get(str(key).lower(), None)
        if matched_key is not None:
            value = sd.get(matched_key, None)
            if isinstance(value, Tensor):
                return value

    for raw_key, value in sd.items():
        if not isinstance(raw_key, str) or not isinstance(value, Tensor):
            continue
        if any(pattern.search(raw_key) for pattern in key_patterns):
            return value

    return None


def _matches_any_pattern(text: str, patterns: Tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _split_lora_key(key: str) -> Tuple[str, str]:
    if not isinstance(key, str) or "." not in key:
        return "", ""

    lowered = key.lower()
    markers = (
        ".lora_down",
        ".lora_up",
        ".stella_vt",
        ".stella_u",
        ".stella_s",
        ".alpha",
        ".lora_alpha",
    )
    marker_idx: Optional[int] = None
    for marker in markers:
        idx = lowered.find(marker)
        if idx <= 0:
            continue
        if marker_idx is None or idx < marker_idx:
            marker_idx = idx

    if marker_idx is not None:
        return key[:marker_idx], key[marker_idx + 1 :]

    lora_name, suffix = key.split(".", 1)
    return lora_name, suffix


def _canonicalize_stella_state_key(key: str) -> str:
    if not isinstance(key, str) or not key:
        return key

    canonical = key.replace("stella_U", "stella_u")
    canonical = canonical.replace("stella_Vt", "stella_vt")
    canonical = canonical.replace("stella_S", "stella_s")

    canonical = re.sub(
        r"(^|\.)lora_alpha(\.[^.]+)?$",
        lambda match: f"{match.group(1)}alpha",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = re.sub(
        r"(^|\.)stella_u(\.[^.]+)?\.weight$",
        lambda match: f"{match.group(1)}lora_up.weight",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = re.sub(
        r"(^|\.)stella_vt(\.[^.]+)?\.weight$",
        lambda match: f"{match.group(1)}lora_down.weight",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = re.sub(
        r"(^|\.)stella_s(\.[^.]+)?\.weight$",
        lambda match: f"{match.group(1)}stella_s.weight",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = re.sub(
        r"(^|\.)stella_s\.(?!weight$)[^.]+$",
        lambda match: f"{match.group(1)}stella_s",
        canonical,
        flags=re.IGNORECASE,
    )
    return canonical


def _canonicalize_stella_state_dict_keys(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    normalized: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        canonical_key = _canonicalize_stella_state_key(key)
        if canonical_key not in normalized or canonical_key == key:
            normalized[canonical_key] = value
    return normalized


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_choice(raw: Any, key: str, allowed: set[str], default: str) -> str:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _normalize_stella_init(raw: Any) -> str:
    if raw is None:
        return _DEFAULT_STELLA_INIT
    if isinstance(raw, bool):
        return "orthonormal" if raw else "kaiming"
    lowered = str(raw).strip().lower()
    if lowered == "":
        return _DEFAULT_STELLA_INIT
    if lowered in {"1", "true", "yes", "y", "on"}:
        return "orthonormal"
    if lowered in {"0", "false", "no", "n", "off"}:
        return "kaiming"
    normalized = _STELLA_INIT_ALIASES.get(lowered, lowered)
    if normalized not in _VALID_STELLA_INITS:
        raise ValueError(
            f"stella_init must be one of {sorted(_VALID_STELLA_INITS)} "
            f"(aliases: {sorted(_STELLA_INIT_ALIASES.keys())}), got {raw!r}"
        )
    return normalized


def _parse_positive_float(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be a float > 0, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def _parse_non_negative_int(raw: Any, key: str, default: int = 0) -> int:
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer >= 0, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _parse_grad_scaling(raw: Any) -> bool | float:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value <= 0.0:
            raise ValueError(
                f"stella_grad_scaling must be bool or float > 0, got {value}"
            )
        return value
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        value = _parse_positive_float(lowered, "stella_grad_scaling")
        return value
    raise ValueError(f"stella_grad_scaling must be bool or float > 0, got {raw!r}")


def _symm(x: Tensor) -> Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _euclidean_to_riemannian(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (grad.transpose(-1, -2) @ x)


def _tangent_project(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ _symm(x.transpose(-1, -2) @ grad)


def _polar_uf(matrix: Tensor) -> Tensor:
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _polar_retraction(x: Tensor, grad: Tensor) -> Tensor:
    return _polar_uf(x + grad)


def _exp_map_retraction(x: Tensor, grad: Tensor) -> Tensor:
    x_t_grad = x.transpose(-1, -2) @ grad
    q, r = torch.linalg.qr(grad - x @ x_t_grad, mode="reduced")
    zeros = torch.zeros_like(r)
    ident = torch.eye(r.shape[-2], device=r.device, dtype=r.dtype)
    top_row = torch.cat([x_t_grad, -r.transpose(-1, -2)], dim=-1)
    bottom_row = torch.cat([r, zeros], dim=-1)
    block = torch.cat([top_row, bottom_row], dim=-2)
    exp_block = torch.linalg.matrix_exp(block)
    iz = torch.cat([ident, zeros], dim=-2)
    mn = exp_block @ iz
    xq = torch.cat([x, q], dim=-1)
    return xq @ mn


def _oriented_matrix(weight: Tensor) -> Tuple[Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(weight.shape)}")
    transpose = bool(weight.shape[0] < weight.shape[1])
    if transpose:
        return weight.transpose(0, 1), True
    return weight, False


def _restore_oriented_matrix(matrix: Tensor, transpose: bool) -> Tensor:
    return matrix.transpose(0, 1) if transpose else matrix


class _DiagonalLinear(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(features), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x * self.weight.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return x * self.weight.to(device=x.device, dtype=x.dtype)


@dataclass
class _StellaCacheEntry:
    param: nn.Parameter
    pre_value: Tensor
    transposed: bool
    row_dim: int
    grad_reference_dim: Optional[float]
    shape_key: Tuple[int, int]


class StellaModule(LoRAModule):
    """LoRA module with an additional S factor and StelLA-compatible init."""

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
        use_rslora: bool = False,
        stella_diag_s: bool = False,
        stella_init: str = _DEFAULT_STELLA_INIT,
    ) -> None:
        del pissa_niter, ggpo_sigma, ggpo_beta  # StelLA does not use these paths.

        if split_dims is not None:
            raise ValueError(
                "StelLA does not support split_dims modules in this integration."
            )

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=None,
            initialize="kaiming",
            pissa_niter=None,
            ggpo_sigma=None,
            ggpo_beta=None,
            use_rslora=use_rslora,
        )

        self.stella_diag_s = bool(stella_diag_s)
        self.stella_init = _parse_choice(
            _normalize_stella_init(stella_init),
            "stella_init",
            _VALID_STELLA_INITS,
            _DEFAULT_STELLA_INIT,
        )
        if self.stella_diag_s:
            self.stella_s = _DiagonalLinear(int(self.lora_dim))
        else:
            self.stella_s = nn.Linear(int(self.lora_dim), int(self.lora_dim), bias=False)
        self._ggpo_enabled = False

        self._initialize_stella_factors(org_module.weight)

    @torch.no_grad()
    def _initialize_stella_factors(self, org_weight: Tensor) -> None:
        if self.stella_init == "kaiming":
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
            self._reset_stella_s_to_identity()
            return

        if self.stella_init.startswith("svd_"):
            mode = self.stella_init.split("_", 1)[1]
            if self._try_svd_init(org_weight, mode=mode):
                self._reset_stella_s_to_identity()
                return
            logger.warning(
                "StelLA SVD init fallback to orthonormal for module %s.", self.lora_name
            )

        torch.nn.init.orthogonal_(self.lora_up.weight)
        torch.nn.init.orthogonal_(self.lora_down.weight)
        self._reset_stella_s_to_identity()

    @torch.no_grad()
    def _try_svd_init(self, org_weight: Tensor, mode: str) -> bool:
        if not isinstance(self.lora_down, nn.Linear) or not isinstance(
            self.lora_up, nn.Linear
        ):
            return False
        if not isinstance(org_weight, Tensor) or org_weight.ndim != 2:
            return False
        try:
            weight = org_weight.detach().to(dtype=torch.float32, device="cpu")
            u, _, vh = torch.linalg.svd(weight, full_matrices=False)
            rank = int(self.lora_dim)
            if mode == "major":
                up = u[:, :rank]
                down = vh[:rank, :]
            elif mode == "minor":
                up = u[:, -rank:]
                down = vh[-rank:, :]
            else:
                return False
            self.lora_up.weight.data.copy_(
                up.to(device=self.lora_up.weight.device, dtype=self.lora_up.weight.dtype)
            )
            self.lora_down.weight.data.copy_(
                down.to(
                    device=self.lora_down.weight.device,
                    dtype=self.lora_down.weight.dtype,
                )
            )
            return True
        except Exception:
            return False

    @torch.no_grad()
    def _reset_stella_s_to_identity(self) -> None:
        if self.stella_diag_s:
            cast(_DiagonalLinear, self.stella_s).weight.data.fill_(1.0)
            return
        weight = cast(nn.Linear, self.stella_s).weight
        weight.data.zero_()
        eye_rank = min(weight.shape[0], weight.shape[1])
        weight.data[:eye_rank, :eye_rank] = torch.eye(
            eye_rank,
            device=weight.device,
            dtype=weight.dtype,
        )

    def _apply_stella_s(self, x: Tensor) -> Tensor:
        if self.stella_diag_s:
            return cast(_DiagonalLinear, self.stella_s)(x)

        s_linear = cast(nn.Linear, self.stella_s)
        if x.ndim == 4:
            x_perm = x.permute(0, 2, 3, 1)
            x_proj = torch.nn.functional.linear(
                x_perm,
                s_linear.weight.to(device=x.device, dtype=x.dtype),
            )
            return x_proj.permute(0, 3, 1, 2)
        return torch.nn.functional.linear(
            x,
            s_linear.weight.to(device=x.device, dtype=x.dtype),
        )

    def _stella_s_matrix(
        self,
        device: torch.device,
        dtype: torch.dtype,
        sd: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        if self.stella_diag_s:
            diag_tensor = cast(_DiagonalLinear, self.stella_s).weight
            if sd is not None:
                resolved_diag = _lookup_sd_tensor(
                    sd,
                    direct_keys=(
                        "stella_s.weight",
                        "stella_s",
                        "stella_S.weight",
                        "stella_S",
                    ),
                    key_patterns=_STELLA_S_PATTERNS,
                )
                if resolved_diag is not None:
                    diag_tensor = cast(Tensor, resolved_diag)
            diag_tensor = diag_tensor.to(device=device, dtype=dtype).reshape(-1)
            return torch.diag(diag_tensor)

        s_tensor = cast(nn.Linear, self.stella_s).weight
        if sd is not None:
            resolved_s = _lookup_sd_tensor(
                sd,
                direct_keys=(
                    "stella_s.weight",
                    "stella_s",
                    "stella_S.weight",
                    "stella_S",
                ),
                key_patterns=_STELLA_S_PATTERNS,
            )
            if resolved_s is not None:
                s_tensor = cast(Tensor, resolved_s)
        return s_tensor.to(device=device, dtype=dtype)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            if lx.ndim == 4:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                ).unsqueeze(-1).unsqueeze(-1)
            elif lx.ndim == 3:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                ).unsqueeze(1)
            else:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                )
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self._apply_stella_s(lx)
        lx = self.lora_up(lx)
        return org_forwarded + lx * self.multiplier * scale

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        if isinstance(self.lora_down, nn.Linear) and isinstance(self.lora_up, nn.Linear):
            up = self.lora_up.weight.to(torch.float32)
            down = self.lora_down.weight.to(torch.float32)
            s_mat = self._stella_s_matrix(
                device=up.device,
                dtype=up.dtype,
                sd=None,
            )
            delta = up @ s_mat @ down
            return delta * float(multiplier) * float(self.scale)

        if isinstance(self.lora_down, nn.Conv2d) and isinstance(self.lora_up, nn.Conv2d):
            up_2d = self.lora_up.weight.to(torch.float32).squeeze(3).squeeze(2)
            down = self.lora_down.weight.to(torch.float32)
            s_mat = self._stella_s_matrix(
                device=up_2d.device,
                dtype=up_2d.dtype,
                sd=None,
            )
            up_s = (up_2d @ s_mat).unsqueeze(2).unsqueeze(3)
            delta = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3),
                up_s,
            ).permute(1, 0, 2, 3)
            return delta * float(multiplier) * float(self.scale)

        raise ValueError(
            f"Unsupported StelLA module type for get_weight: {type(self.lora_down)}"
        )

    def merge_to(self, sd, dtype, device, non_blocking=False):
        del non_blocking  # not used

        org_sd = self.org_module.state_dict()
        org_weight = org_sd["weight"]
        org_dtype = org_weight.dtype
        org_device = org_weight.device
        work_device = org_device if device is None else device
        work_dtype = org_dtype if dtype is None else dtype

        if isinstance(sd, dict):
            down = _lookup_sd_tensor(
                sd,
                direct_keys=("lora_down.weight", "stella_vt.weight", "stella_Vt.weight"),
                key_patterns=_STELLA_DOWN_PATTERNS,
            )
            if down is None:
                down = cast(Tensor, self.lora_down.weight)

            up = _lookup_sd_tensor(
                sd,
                direct_keys=("lora_up.weight", "stella_u.weight", "stella_U.weight"),
                key_patterns=_STELLA_UP_PATTERNS,
            )
            if up is None:
                up = cast(Tensor, self.lora_up.weight)

            alpha = _lookup_sd_tensor(
                sd,
                direct_keys=("alpha", "lora_alpha"),
                key_patterns=_STELLA_ALPHA_PATTERNS,
            )
            if alpha is None:
                scale = float(self.scale)
            else:
                rank = int(down.shape[0]) if down.ndim >= 1 else int(self.lora_dim)
                rank = max(1, rank)
                scale = float(alpha.to(torch.float32).item()) / (
                    math.sqrt(float(rank))
                    if self.use_rslora
                    else float(rank)
                )
        else:
            down = cast(Tensor, self.lora_down.weight)
            up = cast(Tensor, self.lora_up.weight)
            scale = float(self.scale)

        s_mat = self._stella_s_matrix(
            device=work_device,
            dtype=torch.float32,
            sd=sd if isinstance(sd, dict) else None,
        )

        if isinstance(self.lora_down, nn.Linear) and isinstance(self.lora_up, nn.Linear):
            delta = (
                up.to(device=work_device, dtype=torch.float32)
                @ s_mat
                @ down.to(device=work_device, dtype=torch.float32)
            )
        elif isinstance(self.lora_down, nn.Conv2d) and isinstance(self.lora_up, nn.Conv2d):
            up_2d = up.to(device=work_device, dtype=torch.float32).squeeze(3).squeeze(2)
            up_s = (up_2d @ s_mat).unsqueeze(2).unsqueeze(3)
            down_weight = down.to(device=work_device, dtype=torch.float32)
            delta = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3),
                up_s,
            ).permute(1, 0, 2, 3)
        else:
            raise ValueError(
                f"Unsupported StelLA module type for merge: {type(self.lora_down)}"
            )

        merged = org_weight.to(device=work_device, dtype=torch.float32) + (
            float(self.multiplier) * delta * float(scale)
        )
        org_sd["weight"] = merged.to(device=org_device, dtype=work_dtype)
        self.org_module.load_state_dict(org_sd)


class StellaInfModule(StellaModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional["StellaNetwork"] = None

    def set_network(self, network) -> None:
        self.network = network

    def default_forward(self, x):
        lx = self.lora_down(x)
        lx = self._apply_stella_s(lx)
        lx = self.lora_up(lx)
        return self.org_forward(x) + lx * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class StellaNetwork(LoRANetwork):
    def __init__(
        self,
        *args,
        module_class: Type[object] = StellaModule,
        stella_retraction: str = _DEFAULT_STELLA_RETRACTION,
        stella_diag_s: bool = False,
        stella_grad_scaling: bool | float = True,
        stella_grad_reference_dim: int = 0,
        stella_init: str = _DEFAULT_STELLA_INIT,
        **kwargs,
    ) -> None:
        initialize_marker = kwargs.pop("initialize", stella_init)
        super().__init__(
            *args,
            module_class=module_class,
            initialize=initialize_marker,
            **kwargs,
        )
        self.stella_retraction = _parse_choice(
            stella_retraction,
            "stella_retraction",
            _VALID_STELLA_RETRACTIONS,
            _DEFAULT_STELLA_RETRACTION,
        )
        self.stella_diag_s = bool(stella_diag_s)
        self.stella_grad_scaling = _parse_grad_scaling(stella_grad_scaling)
        self.stella_grad_reference_dim = int(max(0, stella_grad_reference_dim))
        self.stella_init = _parse_choice(
            _normalize_stella_init(stella_init),
            "stella_init",
            _VALID_STELLA_INITS,
            _DEFAULT_STELLA_INIT,
        )
        self._stella_cache: List[_StellaCacheEntry] = []
        self._auto_grad_reference_dim: Optional[float] = None

        if self.loraplus_lr_ratio is not None:
            logger.warning(
                "StelLA ignores loraplus_lr_ratio and uses dedicated parameter groups."
            )

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict):
            state_dict = _canonicalize_stella_state_dict_keys(
                cast(Dict[str, Tensor], state_dict)
            )
        return super().load_state_dict(state_dict, strict)

    def _iter_stella_modules(self) -> List[StellaModule]:
        return [
            cast(StellaModule, module)
            for module in (self.text_encoder_loras + self.unet_loras)
            if isinstance(module, StellaModule)
        ]

    def prepare_network(self, args) -> None:
        if getattr(args, "stella_grad_reference_dim", None) is not None:
            try:
                self.stella_grad_reference_dim = max(
                    0, int(getattr(args, "stella_grad_reference_dim", 0))
                )
            except Exception:
                self.stella_grad_reference_dim = 0
        self._auto_grad_reference_dim = None
        if (
            isinstance(self.stella_grad_scaling, bool)
            and self.stella_grad_scaling
            and self.stella_grad_reference_dim <= 0
        ):
            dims = [
                int(getattr(module, "in_features", 0) or 0)
                for module in self._iter_stella_modules()
                if int(getattr(module, "in_features", 0) or 0) > 0
            ]
            if dims:
                mode_dim = Counter(dims).most_common(1)[0][0]
                self._auto_grad_reference_dim = float(mode_dim)
        logger.info(
            "StelLA enabled (retraction=%s, diag_s=%s, grad_scaling=%s, grad_reference_dim=%s, auto_grad_reference_dim=%s, init=%s, modules=%s).",
            self.stella_retraction,
            self.stella_diag_s,
            self.stella_grad_scaling,
            self.stella_grad_reference_dim,
            self._auto_grad_reference_dim,
            self.stella_init,
            len(self._iter_stella_modules()),
        )

    def _resolve_grad_reference_dim(self, module: StellaModule) -> Optional[float]:
        if isinstance(self.stella_grad_scaling, bool):
            if not self.stella_grad_scaling:
                return None
            if self.stella_grad_reference_dim > 0:
                return float(self.stella_grad_reference_dim)
            if self._auto_grad_reference_dim is not None:
                return float(self._auto_grad_reference_dim)
            in_dim = int(getattr(module, "in_features", 0) or 0)
            out_dim = int(getattr(module, "out_features", 0) or 0)
            if in_dim > 0 and out_dim > 0:
                return float(min(in_dim, out_dim))
            return None
        return float(self.stella_grad_scaling)

    @torch.no_grad()
    def pre_optimizer_step(self) -> None:
        self._stella_cache = []
        modules = self._iter_stella_modules()
        if not modules:
            return

        grouped: Dict[Tuple[int, int], List[Tuple[nn.Parameter, Tensor, Tensor, bool, Optional[float]]]] = defaultdict(list)
        for module in modules:
            grad_ref_dim = self._resolve_grad_reference_dim(module)
            for param in (module.lora_up.weight, module.lora_down.weight):
                if param.grad is None or param.ndim != 2:
                    continue
                mat, transposed = _oriented_matrix(param.data)
                grad_mat, _ = _oriented_matrix(param.grad.data)
                shape_key = (int(mat.shape[0]), int(mat.shape[1]))
                grouped[shape_key].append(
                    (
                        param,
                        mat.detach().to(dtype=torch.float32),
                        grad_mat.detach().to(dtype=torch.float32),
                        transposed,
                        grad_ref_dim,
                    )
                )

        for shape_key, items in grouped.items():
            mats = torch.stack([it[1] for it in items], dim=0)
            grads = torch.stack([it[2] for it in items], dim=0)
            riemann_grads = _euclidean_to_riemannian(mats, grads)
            for idx, item in enumerate(items):
                param, _, _, transposed, grad_ref_dim = item
                grad_target = _restore_oriented_matrix(
                    riemann_grads[idx], transposed
                ).to(
                    device=param.grad.device,  # type: ignore[union-attr]
                    dtype=param.grad.dtype,  # type: ignore[union-attr]
                )
                param.grad.copy_(grad_target)  # type: ignore[union-attr]
                self._stella_cache.append(
                    _StellaCacheEntry(
                        param=param,
                        pre_value=mats[idx],
                        transposed=transposed,
                        row_dim=int(mats[idx].shape[0]),
                        grad_reference_dim=grad_ref_dim,
                        shape_key=shape_key,
                    )
                )

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        if not self._stella_cache:
            return

        if self.stella_retraction == "polar":
            retraction = _polar_retraction
        elif self.stella_retraction == "exp_map":
            retraction = _exp_map_retraction
        else:
            raise ValueError(f"Unknown StelLA retraction: {self.stella_retraction}")

        grouped_entries: Dict[Tuple[Tuple[int, int], Optional[float]], List[_StellaCacheEntry]] = defaultdict(list)
        for entry in self._stella_cache:
            grouped_entries[(entry.shape_key, entry.grad_reference_dim)].append(entry)

        for (_, ref_dim), entries in grouped_entries.items():
            pre_stack = torch.stack([entry.pre_value for entry in entries], dim=0)
            current_stack = torch.stack(
                [
                    _oriented_matrix(entry.param.data)[0].detach().to(dtype=torch.float32)
                    for entry in entries
                ],
                dim=0,
            )
            delta = current_stack - pre_stack

            row_dim = int(pre_stack.shape[1])
            if (
                ref_dim is not None
                and float(ref_dim) > 0.0
                and row_dim > 0
                and float(row_dim) != float(ref_dim)
            ):
                delta = delta / math.sqrt(float(row_dim) / float(ref_dim))

            tangent_delta = _tangent_project(pre_stack, delta)
            retracted_stack = retraction(pre_stack, tangent_delta)

            for idx, entry in enumerate(entries):
                target = _restore_oriented_matrix(
                    retracted_stack[idx], entry.transposed
                ).to(
                    device=entry.param.device,
                    dtype=entry.param.dtype,
                )
                entry.param.copy_(target)

        self._stella_cache = []

    def prepare_optimizer_params(
        self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs
    ):
        del kwargs
        self.requires_grad_(True)

        groups: Dict[str, Dict[str, nn.Parameter]] = {
            "stiefel": {},
            "middle": {},
            "patch_embedding": {},
        }

        for lora in self.unet_loras:
            for name, param in lora.named_parameters():
                full_name = f"{lora.lora_name}.{name}"
                if "patch_embedding" in name:
                    groups["patch_embedding"][full_name] = param
                elif name.endswith("lora_up.weight") or name.endswith("lora_down.weight"):
                    groups["stiefel"][full_name] = param
                else:
                    groups["middle"][full_name] = param

        params: List[Dict[str, Any]] = []
        lr_descriptions: List[str] = []

        if groups["stiefel"] and unet_lr not in (None, 0):
            params.append(
                {
                    "params": groups["stiefel"].values(),
                    "lr": unet_lr,
                    "weight_decay": 0.0,
                }
            )
            lr_descriptions.append("unet stiefel")

        if groups["middle"] and unet_lr not in (None, 0):
            params.append(
                {
                    "params": groups["middle"].values(),
                    "lr": unet_lr,
                }
            )
            lr_descriptions.append("unet")

        if groups["patch_embedding"] and unet_lr not in (None, 0):
            params.append(
                {
                    "params": groups["patch_embedding"].values(),
                    "lr": unet_lr * input_lr_scale,
                }
            )
            lr_descriptions.append("unet patch_embedding")

        return params, lr_descriptions


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
        "stella_unet",
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
    del vae  # API compatibility.

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

    stella_retraction = _parse_choice(
        kwargs.get("stella_retraction", _DEFAULT_STELLA_RETRACTION),
        "stella_retraction",
        _VALID_STELLA_RETRACTIONS,
        _DEFAULT_STELLA_RETRACTION,
    )
    stella_diag_s = _parse_bool(kwargs.get("stella_diag_s", False))
    stella_grad_scaling = _parse_grad_scaling(kwargs.get("stella_grad_scaling", True))
    stella_grad_reference_dim = _parse_non_negative_int(
        kwargs.get("stella_grad_reference_dim", 0),
        "stella_grad_reference_dim",
        default=0,
    )
    stella_init = _parse_choice(
        _normalize_stella_init(kwargs.get("stella_init", _DEFAULT_STELLA_INIT)),
        "stella_init",
        _VALID_STELLA_INITS,
        _DEFAULT_STELLA_INIT,
    )

    use_rslora = kwargs.get("use_rslora", False)
    if isinstance(use_rslora, str):
        use_rslora = _parse_bool(use_rslora)
    use_rslora = bool(use_rslora)

    network = StellaNetwork(
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
        initialize=stella_init,
        pissa_niter=None,
        use_rslora=use_rslora,
        module_class=StellaModule,
        stella_retraction=stella_retraction,
        stella_diag_s=stella_diag_s,
        stella_grad_scaling=stella_grad_scaling,
        stella_grad_reference_dim=stella_grad_reference_dim,
        stella_init=stella_init,
    )

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
    for key, value in weights_sd.items():
        if not isinstance(key, str) or "." not in key:
            continue

        lora_name, suffix = _split_lora_key(key)
        if not lora_name or not suffix:
            continue

        suffix_lower = suffix.lower()
        if _matches_any_pattern(suffix_lower, _STELLA_ALPHA_PATTERNS):
            modules_alpha[lora_name] = value
        elif _matches_any_pattern(suffix_lower, _STELLA_DOWN_PATTERNS):
            modules_dim[lora_name] = int(value.shape[0])

    module_class: Type[object] = StellaInfModule if for_inference else StellaModule

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

    stella_retraction = _parse_choice(
        kwargs.get("stella_retraction", _DEFAULT_STELLA_RETRACTION),
        "stella_retraction",
        _VALID_STELLA_RETRACTIONS,
        _DEFAULT_STELLA_RETRACTION,
    )
    stella_diag_s = _parse_bool(kwargs.get("stella_diag_s", False))
    stella_grad_scaling = _parse_grad_scaling(kwargs.get("stella_grad_scaling", True))
    stella_grad_reference_dim = _parse_non_negative_int(
        kwargs.get("stella_grad_reference_dim", 0),
        "stella_grad_reference_dim",
        default=0,
    )
    stella_init = _parse_choice(
        _normalize_stella_init(kwargs.get("stella_init", _DEFAULT_STELLA_INIT)),
        "stella_init",
        _VALID_STELLA_INITS,
        _DEFAULT_STELLA_INIT,
    )

    use_rslora = kwargs.get("use_rslora", False)
    if isinstance(use_rslora, str):
        use_rslora = _parse_bool(use_rslora)
    use_rslora = bool(use_rslora)

    network = StellaNetwork(
        target_replace_modules,
        "stella_unet",
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
        initialize=stella_init,
        pissa_niter=None,
        use_rslora=use_rslora,
        stella_retraction=stella_retraction,
        stella_diag_s=stella_diag_s,
        stella_grad_scaling=stella_grad_scaling,
        stella_grad_reference_dim=stella_grad_reference_dim,
        stella_init=stella_init,
    )
    return network
