"""Riemannion optimizer (LoRA fixed-rank manifold update)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from common.logger import get_logger
from optimizers.optimizer_utils import apply_weight_decay

logger = get_logger(__name__)

optimizer_type = "Riemannion"
optimizer_args = [
    "riemannion_lr=1e-3",
    "riemannion_momentum=0.9",
    "riemannion_ns_steps=3",
    "riemannion_nesterov=True",
    "riemannion_weight_decay=0.001",
    "riemannion_max_elements=2000000",
    "riemannion_adam_lr=2e-5",
    "riemannion_betas=(0.9,0.95)",
]

_RIEMANNION_FLOAT_KEYS = {
    "riemannion_lr",
    "riemannion_momentum",
    "riemannion_weight_decay",
    "riemannion_adam_lr",
}
_RIEMANNION_INT_KEYS = {"riemannion_ns_steps", "riemannion_max_elements"}
_RIEMANNION_BOOL_KEYS = {"riemannion_nesterov"}
_RIEMANNION_PAIR_KEYS = {"riemannion_betas"}


def _coerce_float(key: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a float (got {value!r})") from exc


def _coerce_int(key: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an int (got {value!r})") from exc


def _coerce_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes", "y"}:
            return True
        if val in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{key} must be a bool (got {value!r})")


def _coerce_pair(key: str, value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"{key} must be a length-2 sequence (got {value!r})")


def apply_riemannion_config_overrides(
    args: Any, optimizer_kwargs: Dict[str, Any]
) -> None:
    """Populate optimizer kwargs with Riemannion config overrides when provided."""
    for key in _RIEMANNION_FLOAT_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_float(key, optimizer_kwargs[key])
    for key in _RIEMANNION_INT_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_int(key, optimizer_kwargs[key])
    for key in _RIEMANNION_BOOL_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_bool(key, optimizer_kwargs[key])
    for key in _RIEMANNION_PAIR_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_pair(key, optimizer_kwargs[key])


def _qr_decompose(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.linalg.qr(mat, mode="reduced")


def _low_rank_svd(
    a: torch.Tensor, b: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qa, ra = _qr_decompose(a)
    qb, rb = _qr_decompose(b)
    core = ra @ rb.T
    u_hat, s, v_hat = torch.linalg.svd(core, full_matrices=False)
    u_hat = u_hat[:, :rank]
    s = s[:rank]
    v_hat = v_hat[:rank, :]
    u = qa @ u_hat
    v = qb @ v_hat.T
    return u, s, v


def _ortho_lr(
    a_l: torch.Tensor,
    b_r: torch.Tensor,
    dot_a: torch.Tensor,
    dot_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ql, tl = _qr_decompose(torch.cat([a_l, dot_a], dim=1))
    b_stack = torch.cat([dot_b, b_r], dim=1)
    qr_t, tr = _qr_decompose(b_stack.T)
    qr = qr_t.T
    u_hat, _, v_hat = torch.linalg.svd(tl @ tr.T, full_matrices=False)
    a = ql @ u_hat
    b = qr @ v_hat.T
    return a, b


def _project_lr(
    a: torch.Tensor,
    b: torch.Tensor,
    a_l: torch.Tensor,
    b_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a_l_t_a = a_l.T @ a
    b_t_b_r = b.T @ b_r
    dot_a = (a - a_l @ a_l_t_a) @ b_t_b_r
    dot_b = b @ a_l_t_a
    return dot_a, dot_b


def _compute_grad_w(
    module: Any,
    lora_up: torch.Tensor,
    lora_down: torch.Tensor,
) -> Optional[torch.Tensor]:
    if module is None:
        return None
    x = getattr(module, "_last_input", None)
    grad_out = getattr(module, "_last_grad_output", None)
    if x is None or grad_out is None:
        return None
    if lora_up.ndim != 2 or lora_down.ndim != 2:
        return None
    in_dim = lora_down.shape[1]
    out_dim = lora_up.shape[0]
    x_mat = x.reshape(-1, in_dim).to(dtype=torch.float32)
    g_mat = grad_out.reshape(-1, out_dim).to(dtype=torch.float32)
    return g_mat.T @ x_mat


def _riemann_grad_components(
    grad_w: torch.Tensor, a_l: torch.Tensor, b_r: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    dot_a = grad_w @ b_r
    dot_a = dot_a - a_l @ (a_l.T @ dot_a)
    dot_b = grad_w.T @ a_l
    return dot_a, dot_b


class SingleDeviceRiemannionWithAuxAdam(torch.optim.Optimizer):
    """Riemannion optimizer with auxiliary AdamW for non-paired params."""

    def __init__(self, param_groups: Iterable[Dict[str, Any]]):
        normalized_groups: List[Dict[str, Any]] = []
        for group in param_groups:
            if "use_riemannion" not in group:
                raise ValueError("Riemannion param groups must set use_riemannion.")
            if group["use_riemannion"]:
                group = dict(group)
                group["lr"] = group.get("lr", 1e-3)
                group["momentum"] = group.get("momentum", 0.9)
                group["ns_steps"] = group.get("ns_steps", 3)
                group["nesterov"] = group.get("nesterov", True)
                group["weight_decay"] = group.get("weight_decay", 0.001)
                group["max_elements"] = group.get("max_elements", 2000000)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                group["weight_decay_type"] = group.get("weight_decay_type", "default")
                normalized_groups.append(group)
            else:
                group = dict(group)
                group["lr"] = group.get("lr", 2e-5)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                group["weight_decay_type"] = group.get("weight_decay_type", "default")
                normalized_groups.append(group)
        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("use_riemannion", False):
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step = state["step"]
                    betas = group["betas"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(p.grad, 1 - betas[0])
                    exp_avg_sq.lerp_(p.grad.square(), 1 - betas[1])
                    exp_avg_c = exp_avg / (1 - betas[0] ** step)
                    exp_avg_sq_c = exp_avg_sq / (1 - betas[1] ** step)
                    update = exp_avg_c / (exp_avg_sq_c.sqrt() + group["eps"])
                    apply_weight_decay(
                        p,
                        update,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group["lr"]),
                    )
                    p.add_(update, alpha=-group["lr"])
                continue

            lora_down = group.get("lora_down")
            lora_up = group.get("lora_up")
            lora_module = group.get("lora_module")
            if lora_down is None or lora_up is None:
                continue
            if lora_up.ndim != 2 or lora_down.ndim != 2:
                continue

            out_dim, in_dim = lora_up.shape[0], lora_down.shape[1]
            if out_dim * in_dim > group["max_elements"]:
                continue

            grad_w = _compute_grad_w(lora_module, lora_up, lora_down)
            if grad_w is None:
                continue

            a = lora_up.data
            b = lora_down.data.T
            a_l, r_a = _qr_decompose(a)
            b_r, r_b = _qr_decompose(b)

            dot_a, dot_b = _riemann_grad_components(grad_w, a_l, b_r)

            state = self.state[lora_up]
            a_hb = state.get("a_hb")
            b_hb = state.get("b_hb")
            if a_hb is None or b_hb is None:
                dot_a_prev = torch.zeros_like(dot_a)
                dot_b_prev = torch.zeros_like(dot_b)
            else:
                dot_a_prev, dot_b_prev = _project_lr(a_hb, b_hb, a_l, b_r)

            beta = group["momentum"]
            dot_a = beta * dot_a_prev + dot_a
            dot_b = beta * dot_b_prev + dot_b

            ortho_a, ortho_b = _ortho_lr(a_l, b_r, dot_a, dot_b)
            dot_a, dot_b = _project_lr(ortho_a, ortho_b, a_l, b_r)

            gamma = group["weight_decay"]
            eta = group["lr"]
            a_concat = torch.cat([-eta * dot_a, a_l], dim=1)
            b_concat = torch.cat([b_r, -eta * (dot_b + gamma * b_r)], dim=1)

            u, s, v = _low_rank_svd(a_concat, b_concat, rank=lora_down.shape[0])
            s_root = torch.sqrt(s.clamp(min=1e-12))
            new_up = (u * s_root).to(lora_up.dtype)
            new_down = (s_root[:, None] * v.T).to(lora_down.dtype)

            lora_up.copy_(new_up)
            lora_down.copy_(new_down)

            state["a_hb"] = torch.cat([dot_a, a_l], dim=1).detach()
            state["b_hb"] = torch.cat([b_r, dot_b], dim=1).detach()

            if lora_module is not None:
                lora_module._last_input = None
                lora_module._last_grad_output = None

        return loss
