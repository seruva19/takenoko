"""Mano optimizer integration.

Mano applies manifold-normalized updates to matrix parameters and uses AdamW for
auxiliary (typically <2D) parameters.

Usage example:

```
optimizer_type = "Mano"
optimizer_args = [
    "mano_lr=1e-3",
    "mano_momentum=0.95",
    "mano_nesterov=false",
    "mano_weight_decay=0.001",
    "mano_scale_factor=0.2",
    "mano_adam_lr=2e-5",
    "mano_betas=(0.9,0.95)",
    "mano_adam_eps=1e-8",
    "mano_exclude_embeddings=true",
    "mano_exclude_lm_head=true",
]
```
"""

import ast
import math
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from optimizers.optimizer_utils import (
    adam_update,
    apply_weight_decay,
    track_gradient_consistency,
    track_update_ratio,
)

logger = get_logger(__name__)

__all__ = [
    "mano_update",
    "SingleDeviceManoWithAuxAdam",
    "apply_mano_config_overrides",
]

# Mano configuration knobs:
# - mano_lr: learning rate for matrix parameters (defaults to muon_lr or 1e-3)
# - mano_momentum: momentum coefficient for tangent-space update (default 0.95)
# - mano_eps: numerical stability epsilon for normalization operations (default 1e-8)
# - mano_nesterov: enable Nesterov momentum for tangent projection (default False)
# - mano_scale_factor: RMS scaling constant used for Mano update magnitude (default 0.2)
# - mano_weight_decay: weight decay for Mano parameter group (default weight_decay)
# - mano_adam_lr: learning rate for auxiliary Adam parameters (default learning_rate)
# - mano_betas: Adam betas for auxiliary parameters (default [0.9, 0.95])
# - mano_adam_eps: Adam epsilon for auxiliary parameters (default 1e-8)
# - mano_adam_weight_decay: optional weight decay override for auxiliary Adam group
# - mano_exclude_embeddings: route embedding-layer params to auxiliary Adam (default False)
# - mano_exclude_lm_head: route lm_head params to auxiliary Adam (default False)
# - mano_exclude_name_patterns: fnmatch patterns for params to route to auxiliary Adam

_MANO_FLOAT_KEYS = {
    "mano_lr",
    "mano_momentum",
    "mano_eps",
    "mano_scale_factor",
    "mano_weight_decay",
    "mano_adam_lr",
    "mano_adam_eps",
    "mano_adam_weight_decay",
}
_MANO_PAIR_KEYS = {"mano_betas"}
_MANO_BOOL_KEYS = {
    "mano_nesterov",
    "mano_exclude_embeddings",
    "mano_exclude_lm_head",
}
_MANO_LIST_KEYS = {"mano_exclude_name_patterns"}


def _fetch_override_source(args: Any) -> Dict[str, Any]:
    raw_config = getattr(args, "raw_config", None)
    return raw_config if isinstance(raw_config, dict) else {}


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as err:  # pragma: no cover - defensive
            raise ValueError(f"{name} must be a float-compatible value.") from err
    raise TypeError(f"{name} must be float-like, received {type(value).__name__}.")


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean-compatible value.")


def _coerce_pair(name: str, value: Any) -> Tuple[float, float]:
    parsed: Optional[Any] = None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parts = [item.strip() for item in value.split(",") if item.strip()]
            if len(parts) == 2:
                parsed = [float(parts[0]), float(parts[1])]
    else:
        parsed = value

    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        try:
            return float(parsed[0]), float(parsed[1])
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            raise ValueError(f"{name} elements must be numeric.") from err

    raise ValueError(f"{name} must contain exactly two numeric values.")


def _coerce_string_list(name: str, value: Any) -> list[str]:
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{name} must be a string or a list/tuple of strings.")

    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain only strings.")
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return result


def apply_mano_config_overrides(args: Any, optimizer_kwargs: Dict[str, Any]) -> None:
    """Populate optimizer kwargs with Mano config overrides when provided."""

    config_source = _fetch_override_source(args)

    def _maybe_get(key: str) -> Any:
        attr_value = getattr(args, key, None)
        if attr_value is not None:
            return attr_value
        return config_source.get(key)

    for key in _MANO_FLOAT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_float(key, raw_value)

    for key in _MANO_PAIR_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_pair(key, raw_value)

    for key in _MANO_BOOL_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_bool(key, raw_value)

    for key in _MANO_LIST_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_string_list(key, raw_value)


def _flatten_parameter(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if tensor.ndim < 2:
        raise ValueError(
            "Mano expects tensors with at least 2 dimensions. "
            f"Received shape={tuple(tensor.shape)}"
        )
    if tensor.ndim == 2:
        return tensor, tensor.shape
    return tensor.reshape(tensor.shape[0], -1), tensor.shape


def mano_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    param: torch.Tensor,
    *,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = False,
    dim: int = 0,
) -> torch.Tensor:
    """Compute Mano tangent-space direction for a single matrix-shaped view."""
    momentum.mul_(beta).add_(grad)
    candidate = grad.add(momentum, alpha=beta) if nesterov else momentum

    param_unit = param / torch.clamp(
        torch.norm(param, p=2, dim=dim, keepdim=True),
        min=eps,
    )
    tangent = candidate - (torch.sum(candidate * param_unit, dim=dim, keepdim=True) * param_unit)
    return tangent / torch.clamp(torch.norm(tangent, p=2, dim=dim, keepdim=True), min=eps)


class SingleDeviceManoWithAuxAdam(torch.optim.Optimizer):
    """Apply Mano to matrix params and AdamW-style updates to auxiliary params."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_mano" in group
            if group["use_mano"]:
                group["params"] = sorted(
                    group["params"],
                    key=lambda x: x.size(),
                    reverse=True,
                )
                group["lr"] = group.get("lr", 0.001)
                group["momentum"] = group.get("momentum", 0.95)
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["nesterov"] = group.get("nesterov", False)
                group["scale_factor"] = group.get("scale_factor", 0.2)
                group["steps"] = int(group.get("steps", 0))
                group["log_muon_metrics"] = group.get("log_muon_metrics", False)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "eps",
                    "weight_decay",
                    "use_mano",
                    "nesterov",
                    "scale_factor",
                    "steps",
                    "initial_lr",
                    "weight_decay_type",
                    "log_muon_metrics",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = tuple(group.get("betas", (0.9, 0.95)))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_mano",
                    "initial_lr",
                    "weight_decay_type",
                }
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_mano"]:
                dim = int(group["steps"] % 2)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim < 2:
                        raise ValueError(
                            "Mano received a parameter with fewer than 2 dimensions. "
                            f"Parameter shape: {tuple(p.shape)}"
                        )

                    grad_view, original_shape = _flatten_parameter(p.grad)
                    param_view, _ = _flatten_parameter(p)

                    state = self.state[p]
                    if (
                        len(state) == 0
                        or "momentum_buffer" not in state
                        or state["momentum_buffer"].shape != grad_view.shape
                    ):
                        state["momentum_buffer"] = torch.zeros_like(grad_view)

                    momentum_buffer = state["momentum_buffer"]

                    if group.get("log_muon_metrics", False):
                        track_gradient_consistency(state, grad_view, momentum_buffer)

                    direction = mano_update(
                        grad_view,
                        momentum_buffer,
                        param_view,
                        beta=group["momentum"],
                        eps=group["eps"],
                        nesterov=bool(group["nesterov"]),
                        dim=dim,
                    )
                    update = direction.reshape(original_shape).to(dtype=p.dtype)

                    if group.get("log_muon_metrics", False):
                        track_update_ratio(state, p, update, group["lr"])

                    apply_weight_decay(
                        p,
                        update,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group.get("lr")),
                    )

                    adjusted_lr = (
                        group["lr"]
                        * group["scale_factor"]
                        * math.sqrt(max(1, direction.shape[dim]))
                    )
                    p.add_(update, alpha=-adjusted_lr)

                group["steps"] += 1
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    apply_weight_decay(
                        p,
                        update,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group.get("lr")),
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss
