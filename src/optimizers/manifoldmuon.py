"""ManifoldMuon optimizer integration.

ManifoldMuon constrains weight matrices to the Stiefel manifold (all singular values = 1)
while using spectral norm-based distance measurements. This provides tighter control over
weight conditioning compared to standard Muon.

Key differences from standard Muon:
- Constrains both updates AND weight matrices to the manifold
- Uses dual ascent to enforce manifold constraints
- Provides better numerical stability and controlled scaling

Usage example:

```
optimizer_type = "ManifoldMuon"
optimizer_args = [
    "manifoldmuon_lr=1e-3",
    "manifoldmuon_eta=0.1",
    "manifoldmuon_alpha=0.01",
    "manifoldmuon_dual_steps=100",
    "manifoldmuon_tolerance=1e-6",
    "manifoldmuon_adam_lr=2e-5",
    "manifoldmuon_betas=(0.9,0.95)",
    "manifoldmuon_weight_decay=0.001",
]
```
"""

import ast
from typing import Any, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from optimizers.muon import adam_update
from optimizers.optimizer_utils import apply_weight_decay

logger = get_logger(__name__)

__all__ = [
    "msign",
    "manifold_muon_update",
    "SingleDeviceManifoldMuonWithAuxAdam",
    "apply_manifoldmuon_config_overrides",
]

# ManifoldMuon configuration knobs:
# - manifoldmuon_lr: learning rate for matrix params (default 1e-3)
# - manifoldmuon_eta: step size for descent (default 0.1)
# - manifoldmuon_alpha: dual variable update rate (default 0.01)
# - manifoldmuon_dual_steps: maximum iterations for dual ascent (default 100)
# - manifoldmuon_tolerance: convergence tolerance (default 1e-6)
# - manifoldmuon_weight_decay: weight decay for ManifoldMuon parameter group
# - manifoldmuon_adam_lr: learning rate for auxiliary AdamW parameters
# - manifoldmuon_betas: AdamW betas for auxiliary parameters (default [0.9, 0.95])

_MANIFOLDMUON_FLOAT_KEYS = {
    "manifoldmuon_lr",
    "manifoldmuon_eta",
    "manifoldmuon_alpha",
    "manifoldmuon_tolerance",
    "manifoldmuon_weight_decay",
    "manifoldmuon_adam_lr",
}
_MANIFOLDMUON_INT_KEYS = {"manifoldmuon_dual_steps"}
_MANIFOLDMUON_PAIR_KEYS = {"manifoldmuon_betas"}


def _fetch_override_source(args: Any) -> Dict[str, Any]:
    raw_config = getattr(args, "raw_config", None)
    return raw_config if isinstance(raw_config, dict) else {}


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as err:
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
        except ValueError as err:
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
        except (TypeError, ValueError) as err:
            raise ValueError(f"{name} elements must be numeric.") from err

    raise ValueError(f"{name} must contain exactly two numeric values.")


def apply_manifoldmuon_config_overrides(
    args: Any, optimizer_kwargs: Dict[str, Any]
) -> None:
    """Populate optimizer kwargs with ManifoldMuon config overrides when provided."""

    config_source = _fetch_override_source(args)

    def _maybe_get(key: str) -> Any:
        attr_value = getattr(args, key, None)
        if attr_value is not None:
            return attr_value
        return config_source.get(key)

    for key in _MANIFOLDMUON_FLOAT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_float(key, raw_value)

    for key in _MANIFOLDMUON_INT_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_int(key, raw_value)

    for key in _MANIFOLDMUON_PAIR_KEYS:
        if key in optimizer_kwargs:
            continue
        raw_value = _maybe_get(key)
        if raw_value is None:
            continue
        optimizer_kwargs[key] = _coerce_pair(key, raw_value)


def msign(W: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute matrix sign function using Newton-Schulz iteration.

    The matrix sign function is used for retraction back to the Stiefel manifold.
    For a matrix with SVD W = U S V^T, sign(W) = U V^T (i.e., sets all singular values to 1).

    Args:
        W: Input matrix (will be transposed if wider than tall)
        steps: Number of Newton-Schulz iterations (default 5)

    Returns:
        Matrix sign of W
    """
    # Handle shape - work with tall matrices
    transposed = False
    if W.size(-2) < W.size(-1):
        W = W.mT
        transposed = True

    # Newton-Schulz iteration for matrix sign
    # X_{k+1} = 0.5 * X_k * (3*I - X_k^T * X_k)
    X = W / (W.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        # Compute X^T @ X
        XTX = X.mT @ X
        # Compute 3*I - X^T @ X
        I = torch.eye(XTX.size(-1), device=X.device, dtype=X.dtype)
        coeff = 3.0 * I - XTX
        # Update: X = 0.5 * X @ (3*I - X^T @ X)
        X = 0.5 * X @ coeff

    if transposed:
        X = X.mT

    return X


def manifold_muon_update(
    W: torch.Tensor,
    G: torch.Tensor,
    eta: float = 0.1,
    alpha: float = 0.01,
    dual_steps: int = 100,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """
    Perform ManifoldMuon update with dual ascent.

    Solves the constrained optimization problem:
        min_A <A, G>  subject to W^T A + A^T W = 0

    Where W is the current weight and G is the gradient, using dual ascent
    to enforce the tangent space constraint.

    Args:
        W: Current weight matrix
        G: Gradient matrix
        eta: Step size for descent (default 0.1)
        alpha: Dual variable update rate (default 0.01)
        dual_steps: Maximum iterations for dual ascent (default 100)
        tolerance: Convergence tolerance (default 1e-6)

    Returns:
        Updated weight matrix projected to Stiefel manifold
    """
    # Handle shape - work with tall matrices
    transposed = False
    if W.size(-2) < W.size(-1):
        W = W.mT
        G = G.mT
        transposed = True

    # Initialize dual variable (Lagrange multiplier)
    # Lambda = -0.25 * (W^T @ G + G^T @ W)
    Lambda = -0.25 * (W.mT @ G + G.mT @ W)

    # Dual ascent loop
    for step in range(dual_steps):
        # Compute candidate direction: A = msign(G + 2 * W @ Lambda)
        A = msign(G + 2.0 * W @ Lambda, steps=5)

        # Measure tangency deviation: H = W^T @ A + A^T @ W
        H = W.mT @ A + A.mT @ W

        # Check convergence
        H_norm = H.norm()
        denominator = max(W.norm() * A.norm(), 1e-12)
        relative_error = H_norm / denominator

        if relative_error < tolerance:
            break

        # Update dual variable with decaying step
        decay = 1.0 - step / dual_steps
        Lambda = Lambda - alpha * decay * H

    # Compute final update direction
    A = msign(G + 2.0 * W @ Lambda, steps=5)

    # Apply gradient descent
    new_W = W - eta * A

    # Project back to manifold using sign function
    new_W = msign(new_W, steps=5)

    if transposed:
        new_W = new_W.mT

    return new_W


class SingleDeviceManifoldMuonWithAuxAdam(torch.optim.Optimizer):
    """
    ManifoldMuon optimizer with auxiliary AdamW for non-matrix parameters.

    This optimizer applies ManifoldMuon to ≥2D matrix parameters and AdamW
    to <2D scalar parameters (biases, gains, etc.).

    ManifoldMuon constrains weight matrices to the Stiefel manifold, ensuring
    all singular values equal 1. This provides:
    - Better numerical conditioning
    - Controlled adaptation without extreme scaling
    - Implicit regularization through manifold constraints

    Example usage:
        ```python
        # Separate parameters by dimensionality
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
            dict(
                params=matrix_params,
                use_muon=True,
                lr=1e-3,
                eta=0.1,
                alpha=0.01,
                dual_steps=100,
                tolerance=1e-6,
                weight_decay=0.001,
                initial_lr=1e-3,
                weight_decay_type="default",
            ),
            dict(
                params=scalar_params,
                use_muon=False,
                lr=2e-5,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=0.001,
                initial_lr=2e-5,
                weight_decay_type="default",
            ),
        ]

        optimizer = SingleDeviceManifoldMuonWithAuxAdam(param_groups)
        ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # ManifoldMuon defaults
                group["lr"] = group.get("lr", 1e-3)
                group["eta"] = group.get("eta", 0.1)
                group["alpha"] = group.get("alpha", 0.01)
                group["dual_steps"] = group.get("dual_steps", 100)
                group["tolerance"] = group.get("tolerance", 1e-6)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "eta",
                        "alpha",
                        "dual_steps",
                        "tolerance",
                        "weight_decay",
                        "use_muon",
                        "initial_lr",
                        "weight_decay_type",
                    ]
                )
            else:
                # AdamW defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "use_muon",
                        "initial_lr",
                        "weight_decay_type",
                    ]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Apply ManifoldMuon to matrix parameters
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    # ManifoldMuon requires current weights and gradients
                    # Note: Unlike standard Muon, we don't use momentum buffer
                    # The dual ascent handles the optimization dynamics

                    # Perform manifold update
                    update = manifold_muon_update(
                        W=p.data,
                        G=p.grad,
                        eta=group["eta"],
                        alpha=group["alpha"],
                        dual_steps=group["dual_steps"],
                        tolerance=group["tolerance"],
                    )

                    # The update is the new weight matrix on the manifold
                    # Compute the actual update direction
                    delta = update - p.data

                    # Apply weight decay
                    apply_weight_decay(
                        p,
                        delta,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group.get("lr")),
                    )

                    # Update parameters
                    # Note: We use the manifold-constrained update directly
                    p.data.copy_(update)

            else:
                # Apply AdamW to scalar parameters
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
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
