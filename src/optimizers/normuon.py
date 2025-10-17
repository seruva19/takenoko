import ast
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from optimizers.muon import adam_update, zeropower_via_newtonschulz5

logger = get_logger(__name__)

__all__ = [
    "normuon_update",
    "SingleDeviceNorMuon",
    "SingleDeviceNorMuonWithAuxAdam",
]

# NorMuon configuration knobs:
# - normuon_lr: spectral-norm learning rate for matrix params (defaults to muon_lr or 1e-3)
# - normuon_momentum: first-order momentum coefficient (beta1, default 0.9)
# - normuon_beta2: per-neuron variance decay (default 0.95)
# - normuon_eps: numerical stability term during variance normalization
# - normuon_ns_steps: Newton-Schulz iterations (default 3 for fast orthogonalization)
# - normuon_weight_decay: weight decay for NorMuon parameter group (default weight_decay)
# - normuon_adam_lr: learning rate for auxiliary AdamW parameters (default learning_rate)
# - normuon_betas: AdamW betas for auxiliary parameters (default [0.9, 0.95])


_NORMUON_FLOAT_KEYS = {
    "normuon_lr",
    "normuon_momentum",
    "normuon_beta2",
    "normuon_eps",
    "normuon_weight_decay",
    "normuon_adam_lr",
}
_NORMUON_INT_KEYS = {"normuon_ns_steps"}
_NORMUON_PAIR_KEYS = {"normuon_betas"}


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


def apply_normuon_config_overrides(
    args: Any, optimizer_kwargs: Dict[str, Any]
) -> None:
    """Populate optimizer kwargs with NorMuon config overrides when provided."""

    config_source = _fetch_override_source(args)

    def _maybe_get(key: str) -> Any:
        attr_value = getattr(args, key, None)
        if attr_value is not None:
            return attr_value
        return config_source.get(key)

    for key in _NORMUON_FLOAT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_float(key, raw_value)

    for key in _NORMUON_INT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_int(key, raw_value)

    for key in _NORMUON_PAIR_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_pair(key, raw_value)


def _flatten_parameter(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Return a 2D view of the tensor (batch, features) alongside the original shape."""
    if tensor.ndim < 2:
        raise ValueError(
            "NorMuon expects tensors with at least 2 dimensions. Received "
            f"shape={tuple(tensor.shape)}"
        )
    if tensor.ndim == 2:
        return tensor, tensor.shape
    return tensor.reshape(tensor.shape[0], -1), tensor.shape


def normuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    v_buffer: torch.Tensor,
    *,
    beta1: float = 0.95,
    beta2: float = 0.95,
    eps: float = 1e-10,
    ns_steps: int = 5,
) -> torch.Tensor:
    """
    Compute the NorMuon update for a single parameter tensor.

    The update follows the "Normalized Muon" algorithm, which orthogonalizes the
    first-order momentum using a Newton-Schulz iteration and then normalizes the
    per-neuron updates with a second-order buffer.
    """
    grad_view, original_shape = _flatten_parameter(grad)
    momentum_view, _ = _flatten_parameter(momentum)

    # First-order momentum update (identical to Muon)
    momentum_view.lerp_(grad_view, 1 - beta1)

    # Orthogonalize via Newton-Schulz
    orth_update = zeropower_via_newtonschulz5(momentum_view, steps=ns_steps)
    m, n = orth_update.shape

    # Second-order (per neuron) variance tracking
    per_neuron_sq = orth_update.norm(dim=-1).pow(2).div(n)
    v_buffer.lerp_(per_neuron_sq.to(v_buffer.dtype), 1 - beta2)

    # Normalize by second-order statistics
    orth_update.mul_((v_buffer + eps).unsqueeze(-1).rsqrt())

    # Global scaling factor recommended by the NorMuon paper
    scale = 0.2 * (m * n) ** 0.5 / (orth_update.norm() + 1e-7)
    orth_update.mul_(scale)

    update = orth_update.reshape(original_shape)
    return update.to(grad.dtype)


class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    NorMuon optimizer for single-device training.

    Applies NorMuon updates to 2D (or flattenable) tensors.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        beta2: float = 0.95,
        eps: float = 1e-10,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            eps=eps,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim < 2:
                    raise ValueError(
                        "NorMuon received a parameter with fewer than 2 dimensions. "
                        f"Parameter shape: {tuple(p.shape)}"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["v_buffer"] = torch.zeros(
                        p.shape[0], device=p.device, dtype=p.dtype
                    )

                update = normuon_update(
                    p.grad,
                    state["momentum_buffer"],
                    state["v_buffer"],
                    beta1=group["momentum"],
                    beta2=group["beta2"],
                    eps=group["eps"],
                    ns_steps=group["ns_steps"],
                )

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    NorMuon variant that applies NorMuon to matrix parameters and AdamW to the rest.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "beta2",
                    "eps",
                    "weight_decay",
                    "use_muon",
                    "ns_steps",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
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
                            "NorMuon received a parameter with fewer than 2 dimensions. "
                            f"Parameter shape: {tuple(p.shape)}"
                        )

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["v_buffer"] = torch.zeros(
                            p.shape[0], device=p.device, dtype=p.dtype
                        )

                    update = normuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["v_buffer"],
                        beta1=group["momentum"],
                        beta2=group["beta2"],
                        eps=group["eps"],
                        ns_steps=group["ns_steps"],
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
