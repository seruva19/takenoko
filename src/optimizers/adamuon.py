"""AdaMuon optimizer integration.

The optimizer applies Muon-style orthogonalization with element-wise adaptivity.

Usage example:

```
optimizer_type = "AdaMuon"
optimizer_args = [
    "adamuon_lr=1e-3",
    "adamuon_momentum=0.95",
    "adamuon_beta2=0.95",
    "adamuon_ns_steps=5",
    "adamuon_scale_factor=0.2",
    "adamuon_nesterov=true",
    "adamuon_sign_stabilization=true",
    "adamuon_adam_lr=2e-5",
    "adamuon_betas=(0.9,0.95)",
    "adamuon_weight_decay=0.001",
]
```
"""

import ast
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from optimizers.muon import adam_update, zeropower_via_newtonschulz5

logger = get_logger(__name__)

__all__ = [
    "adamuon_update",
    "SingleDeviceAdaMuonWithAuxAdam",
    "apply_adamuon_config_overrides",
]

# AdaMuon configuration knobs:
# - adamuon_lr: spectral-norm learning rate for matrix params (defaults to muon_lr or 1e-3)
# - adamuon_momentum: first-order momentum coefficient (beta1, default 0.95)
# - adamuon_beta2: element-wise variance decay (default 0.95)
# - adamuon_eps: numerical stability term applied to v-buffer and RMS scaling
# - adamuon_ns_steps: Newton-Schulz iterations (default 5 for accurate orthogonalization)
# - adamuon_weight_decay: weight decay for AdaMuon parameter group (default weight_decay)
# - adamuon_adam_lr: learning rate for auxiliary AdamW parameters (default learning_rate)
# - adamuon_betas: AdamW betas for auxiliary parameters (default [0.9, 0.95])
# - adamuon_scale_factor: RMS alignment constant (default 0.2, matches Muon guidance)
# - adamuon_nesterov: enable Nesterov update on first-order momentum (default True)
# - adamuon_sign_stabilization: orthogonalize sign(momentum) instead of raw momentum (default True)


_ADAMUON_FLOAT_KEYS = {
    "adamuon_lr",
    "adamuon_momentum",
    "adamuon_beta2",
    "adamuon_eps",
    "adamuon_weight_decay",
    "adamuon_adam_lr",
    "adamuon_scale_factor",
}
_ADAMUON_INT_KEYS = {"adamuon_ns_steps"}
_ADAMUON_PAIR_KEYS = {"adamuon_betas"}
_ADAMUON_BOOL_KEYS = {"adamuon_nesterov", "adamuon_sign_stabilization"}


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


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, received {value}.")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as err:  # pragma: no cover - defensive
            raise ValueError(f"{name} must be an integer value.") from err
    raise TypeError(f"{name} must be int-like, received {type(value).__name__}.")


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


def apply_adamuon_config_overrides(args: Any, optimizer_kwargs: Dict[str, Any]) -> None:
    """Populate optimizer kwargs with AdaMuon config overrides when provided."""

    config_source = _fetch_override_source(args)

    def _maybe_get(key: str) -> Any:
        attr_value = getattr(args, key, None)
        if attr_value is not None:
            return attr_value
        return config_source.get(key)

    for key in _ADAMUON_FLOAT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_float(key, raw_value)

    for key in _ADAMUON_INT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_int(key, raw_value)

    for key in _ADAMUON_PAIR_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_pair(key, raw_value)

    for key in _ADAMUON_BOOL_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_bool(key, raw_value)


def _flatten_parameter(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Return a 2D view of the tensor (batch, features) alongside the original shape."""
    if tensor.ndim < 2:
        raise ValueError(
            "AdaMuon expects tensors with at least 2 dimensions. Received "
            f"shape={tuple(tensor.shape)}"
        )
    if tensor.ndim == 2:
        return tensor, tensor.shape
    return tensor.reshape(tensor.shape[0], -1), tensor.shape


def adamuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    v_buffer: torch.Tensor,
    *,
    beta1: float = 0.95,
    beta2: float = 0.95,
    eps: float = 1e-8,
    ns_steps: int = 5,
    nesterov: bool = True,
    scale_factor: float = 0.2,
    sign_stabilization: bool = True,
) -> torch.Tensor:
    """
    Compute the AdaMuon update for a single parameter tensor.

    The update:
        1. Updates a first-order momentum buffer.
        2. Applies optional Nesterov acceleration.
        3. Orthogonalizes sign(momentum) via Newton-Schulz for stability.
        4. Tracks element-wise variance statistics and normalizes rows.
        5. Scales the result to match Adam's RMS magnitude.
    """
    if grad.ndim < 2:
        raise ValueError(
            "AdaMuon received a parameter with fewer than 2 dimensions. "
            f"Parameter shape: {tuple(grad.shape)}"
        )

    # First-order momentum update
    momentum.mul_(beta1).add_(grad)
    update_tensor = grad.add(momentum, alpha=beta1) if nesterov else momentum

    update_view, original_shape = _flatten_parameter(update_tensor)
    ortho_input = update_view.sign() if sign_stabilization else update_view
    orth_update = zeropower_via_newtonschulz5(ortho_input, steps=ns_steps)

    rows, cols = orth_update.shape
    if v_buffer.shape[0] != rows:
        raise ValueError(
            f"AdaMuon v_buffer row mismatch: expected {rows}, found {v_buffer.shape[0]}"
        )

    per_neuron_sq = orth_update.pow(2).mean(dim=-1)
    v_buffer.lerp_(per_neuron_sq.to(v_buffer.dtype), 1 - beta2)

    normalization = (v_buffer + eps).unsqueeze(-1).rsqrt()
    orth_update.mul_(normalization)

    denom = orth_update.norm().add_(eps)
    scale = scale_factor * (rows * cols) ** 0.5
    orth_update.mul_(scale / denom)

    return orth_update.reshape(original_shape).to(grad.dtype)


class SingleDeviceAdaMuonWithAuxAdam(torch.optim.Optimizer):
    """
    AdaMuon variant that applies AdaMuon to matrix parameters and AdamW to the rest.

    Designed for single-device (or Accelerator-managed data parallel) usage by default.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group["lr"] = group.get("lr", 0.001)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                group["scale_factor"] = group.get("scale_factor", 0.2)
                group["sign_stabilization"] = group.get("sign_stabilization", True)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "beta2",
                    "eps",
                    "weight_decay",
                    "use_muon",
                    "ns_steps",
                    "nesterov",
                    "scale_factor",
                    "sign_stabilization",
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
                    "use_muon",
                }
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim < 2:
                        raise ValueError(
                            "AdaMuon received a parameter with fewer than 2 dimensions. "
                            f"Parameter shape: {tuple(p.shape)}"
                        )

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["v_buffer"] = torch.zeros(
                            p.shape[0],
                            device=p.device,
                            dtype=torch.float32,
                        )

                    update = adamuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["v_buffer"],
                        beta1=group["momentum"],
                        beta2=group["beta2"],
                        eps=group["eps"],
                        ns_steps=group["ns_steps"],
                        nesterov=bool(group["nesterov"]),
                        scale_factor=group["scale_factor"],
                        sign_stabilization=bool(group["sign_stabilization"]),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
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
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
