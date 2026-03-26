"""Takenoko-native DION2-style optimizer.

This implementation is inspired by Microsoft DION2 but adapted to Takenoko's
LoRA-oriented training flow:
- 2D matrix parameters use a submatrix-selected orthonormalized update.
- Non-matrix parameters fall back to auxiliary AdamW inside the same optimizer.
- The optimizer is single-device / DDP-friendly and does not depend on DTensor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from common.logger import get_logger
from optimizers.optimizer_utils import (
    apply_weight_decay,
    track_gradient_consistency,
    track_update_ratio,
    zeropower_via_newtonschulz5,
)

logger = get_logger(__name__)

optimizer_type = "Dion2"
optimizer_args = [
    "dion2_rank_fraction=0.25",
    "dion2_momentum=0.95",
    "dion2_ns_steps=5",
    "dion2_lr_scale=1.0",
    "dion2_aux_lr_scale=1.0",
    "dion2_aux_optimizer='lion'",
    "dion2_weight_decay=0.001",
    "dion2_aux_betas=(0.9,0.999)",
    "dion2_aux_eps=1e-8",
    "dion2_selection='norm'",
    "dion2_subspace_cap=128",
    "dion2_exclude_embeddings=True",
    "dion2_exclude_lm_head=True",
    "dion2_exclude_name_patterns=[]",
    "dion2_lm_head_lr_scale=None",
    "dion2_embedding_weight_decay=None",
    "dion2_lm_head_weight_decay=None",
    "dion2_error_feedback=True",
    "dion2_error_decay=1.0",
    "dion2_distribute_work=False",
    "dion2_log_metrics=False",
]

_DION2_FLOAT_KEYS = {
    "dion2_rank_fraction",
    "dion2_momentum",
    "dion2_lr_scale",
    "dion2_aux_lr_scale",
    "dion2_weight_decay",
    "dion2_aux_eps",
    "dion2_error_decay",
}
_DION2_INT_KEYS = {"dion2_ns_steps", "dion2_subspace_cap"}
_DION2_BOOL_KEYS = {
    "dion2_log_metrics",
    "dion2_exclude_embeddings",
    "dion2_exclude_lm_head",
    "dion2_error_feedback",
    "dion2_distribute_work",
}
_DION2_PAIR_KEYS = {"dion2_aux_betas"}
_DION2_STRING_KEYS = {"dion2_aux_optimizer"}
_DION2_LIST_KEYS = {"dion2_exclude_name_patterns"}
_DION2_OPTIONAL_FLOAT_KEYS = {
    "dion2_lm_head_lr_scale",
    "dion2_embedding_weight_decay",
    "dion2_lm_head_weight_decay",
}
_VALID_SELECTIONS = {"norm", "stride"}
_VALID_AUX_OPTIMIZERS = {"adamw", "lion"}


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
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} must be a bool (got {value!r})")


def _coerce_pair(key: str, value: Any) -> Tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"{key} must be a length-2 sequence (got {value!r})")


def _coerce_string(key: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{key} must be a string (got None)")
    return str(value).strip().lower()


def _coerce_string_list(key: str, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"{key} must be a string or sequence of strings (got {value!r})")


def _coerce_optional_float(key: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return _coerce_float(key, value)


def apply_dion2_config_overrides(optimizer_kwargs: Dict[str, Any]) -> None:
    """Normalize DION2-specific kwargs parsed from optimizer_args."""

    for key in _DION2_FLOAT_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_float(key, optimizer_kwargs[key])
    for key in _DION2_INT_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_int(key, optimizer_kwargs[key])
    for key in _DION2_BOOL_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_bool(key, optimizer_kwargs[key])
    for key in _DION2_PAIR_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_pair(key, optimizer_kwargs[key])
    for key in _DION2_STRING_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_string(key, optimizer_kwargs[key])
    for key in _DION2_LIST_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_string_list(key, optimizer_kwargs[key])
    for key in _DION2_OPTIONAL_FLOAT_KEYS:
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = _coerce_optional_float(key, optimizer_kwargs[key])

    selection = optimizer_kwargs.get(
        "dion2_selection",
        "norm",
    )
    selection = str(selection).strip().lower()
    if selection not in _VALID_SELECTIONS:
        raise ValueError(
            f"dion2_selection must be one of {sorted(_VALID_SELECTIONS)}, got {selection!r}"
        )
    optimizer_kwargs["dion2_selection"] = selection

    aux_optimizer = optimizer_kwargs.get("dion2_aux_optimizer", "lion")
    aux_optimizer = str(aux_optimizer).strip().lower()
    if aux_optimizer not in _VALID_AUX_OPTIMIZERS:
        raise ValueError(
            f"dion2_aux_optimizer must be one of {sorted(_VALID_AUX_OPTIMIZERS)}, "
            f"got {aux_optimizer!r}"
        )
    optimizer_kwargs["dion2_aux_optimizer"] = aux_optimizer


def _select_indices(
    matrix: torch.Tensor,
    axis: int,
    fraction: float,
    subspace_cap: int,
    selection: str,
) -> torch.Tensor:
    size = int(matrix.shape[axis])
    if size <= 1:
        return torch.zeros(1, device=matrix.device, dtype=torch.long)

    count = max(1, min(size, subspace_cap, int(math.ceil(size * fraction))))
    if count >= size:
        return torch.arange(size, device=matrix.device, dtype=torch.long)

    if selection == "stride":
        indices = torch.linspace(
            0,
            size - 1,
            steps=count,
            device=matrix.device,
        ).round().to(torch.long)
        return torch.unique(indices, sorted=True)

    if axis == 0:
        scores = matrix.square().mean(dim=1)
    else:
        scores = matrix.square().mean(dim=0)
    return torch.topk(scores, k=count, largest=True, sorted=False).indices.sort().values


def _qr_basis(sample: torch.Tensor) -> torch.Tensor:
    if sample.numel() == 0:
        raise ValueError("sample must be non-empty")
    q, _ = torch.linalg.qr(sample, mode="reduced")
    return q.to(dtype=torch.float32)


def dion2_update(
    momentum_buffer: torch.Tensor,
    rank_fraction: float,
    ns_steps: int,
    selection: str,
    subspace_cap: int,
) -> torch.Tensor:
    """Build a DION2-style orthonormalized update from a reduced subspace."""

    if momentum_buffer.ndim != 2:
        raise ValueError(
            f"DION2 expects 2D matrix parameters, got ndim={momentum_buffer.ndim}"
        )

    work = momentum_buffer.to(dtype=torch.float32)
    if not torch.isfinite(work).all():
        work = torch.nan_to_num(work)
    if work.abs().max().item() == 0.0:
        return torch.zeros_like(work)

    row_idx = _select_indices(work, 0, rank_fraction, subspace_cap, selection)
    col_idx = _select_indices(work, 1, rank_fraction, subspace_cap, selection)

    left_basis = _qr_basis(work[:, col_idx])
    right_basis = _qr_basis(work[row_idx, :].T)

    core = left_basis.T @ work @ right_basis
    ortho_core = zeropower_via_newtonschulz5(core, steps=ns_steps).to(torch.float32)
    update = left_basis @ ortho_core @ right_basis.T
    update *= max(1.0, work.shape[0] / max(1, work.shape[1])) ** 0.5
    return update


class SingleDeviceDion2WithAuxAdam(torch.optim.Optimizer):
    """Single optimizer that applies DION2 to matrices and AdamW to aux params."""

    def __init__(
        self,
        param_groups: Iterable[Dict[str, Any]],
        process_group: Any = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        normalized_groups: List[Dict[str, Any]] = []

        for group in param_groups:
            if "use_dion2" not in group:
                raise ValueError("DION2 param groups must set use_dion2.")

            group = dict(group)
            if group["use_dion2"]:
                group["lr"] = group.get("lr", 1e-3)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.001)
                group["rank_fraction"] = group.get("rank_fraction", 0.25)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["selection"] = group.get("selection", "norm")
                group["subspace_cap"] = group.get("subspace_cap", 128)
                group["error_feedback"] = group.get("error_feedback", True)
                group["error_decay"] = group.get("error_decay", 1.0)
                group["distribute_work"] = group.get("distribute_work", False)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                group["weight_decay_type"] = group.get("weight_decay_type", "default")
                group["log_dion2_metrics"] = group.get("log_dion2_metrics", False)
                normalized_groups.append(group)
            else:
                group["lr"] = group.get("lr", 2e-5)
                group["algorithm"] = str(group.get("algorithm", "lion")).strip().lower()
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                group["weight_decay_type"] = group.get("weight_decay_type", "default")
                normalized_groups.append(group)

        super().__init__(normalized_groups, {})
        self.process_group = process_group
        self.world_size = int(world_size)
        self.rank = int(rank)
        self._distributed_work_warning_logged = False

    def _use_distributed_work(self, group: Dict[str, Any]) -> bool:
        if not bool(group.get("distribute_work", False)):
            return False
        if self.process_group is None or self.world_size <= 1:
            if not self._distributed_work_warning_logged:
                logger.warning(
                    "DION2 distributed work was requested but no usable process_group "
                    "or world_size > 1 is available; falling back to local updates."
                )
                self._distributed_work_warning_logged = True
            return False
        if not dist.is_available() or not dist.is_initialized():
            if not self._distributed_work_warning_logged:
                logger.warning(
                    "DION2 distributed work was requested but torch.distributed is not "
                    "initialized; falling back to local updates."
                )
                self._distributed_work_warning_logged = True
            return False
        return True

    def _initialize_matrix_state(
        self,
        p: torch.nn.Parameter,
        group: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            if group.get("error_feedback", True):
                state["error_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        elif group.get("error_feedback", True) and "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        return state

    def _broadcast_matrix_state(
        self,
        owner_rank: int,
        p: torch.nn.Parameter,
        state: Dict[str, Any],
        error_feedback: bool,
    ) -> None:
        dist.broadcast(p.data, src=owner_rank, group=self.process_group)
        dist.broadcast(
            state["momentum_buffer"],
            src=owner_rank,
            group=self.process_group,
        )
        if error_feedback:
            dist.broadcast(
                state["error_buffer"],
                src=owner_rank,
                group=self.process_group,
            )

    def _checkpoint_sync_enabled(self) -> bool:
        return any(
            group.get("use_dion2", False) and bool(group.get("distribute_work", False))
            for group in self.param_groups
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        matrix_param_index = 0
        for group in self.param_groups:
            if not group.get("use_dion2", False):
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad.detach().to(dtype=torch.float32)
                    state = self.state[p]
                    algorithm = group.get("algorithm", "lion")

                    if algorithm == "lion":
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] += 1
                        exp_avg = state["exp_avg"]
                        beta1, beta2 = group["betas"]
                        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign()
                        exp_avg.lerp_(grad, 1 - beta2)
                    else:
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                            state["exp_avg_sq"] = torch.zeros_like(
                                p, dtype=torch.float32
                            )

                        state["step"] += 1
                        step = state["step"]
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        beta1, beta2 = group["betas"]

                        exp_avg.lerp_(grad, 1 - beta1)
                        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                        exp_avg_c = exp_avg / (1 - beta1**step)
                        exp_avg_sq_c = exp_avg_sq / (1 - beta2**step)
                        update = exp_avg_c / (exp_avg_sq_c.sqrt() + group["eps"])

                    update_for_param = update.to(dtype=p.dtype)
                    apply_weight_decay(
                        p,
                        update_for_param,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group["lr"]),
                    )
                    p.add_(update_for_param, alpha=-group["lr"])
                continue

            for p in group["params"]:
                if p.ndim < 2:
                    continue

                use_distributed_work = self._use_distributed_work(group)
                owner_rank = (
                    matrix_param_index % self.world_size
                    if use_distributed_work
                    else self.rank
                )
                matrix_param_index += 1

                if p.grad is None:
                    continue

                state = self._initialize_matrix_state(p, group)
                state["step"] += 1

                if use_distributed_work and self.rank != owner_rank:
                    self._broadcast_matrix_state(
                        owner_rank,
                        p,
                        state,
                        bool(group.get("error_feedback", True)),
                    )
                    continue

                grad = p.grad.detach().to(dtype=torch.float32)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                update_input = momentum_buffer
                if group.get("error_feedback", True):
                    error_buffer = state["error_buffer"]
                    update_input = momentum_buffer + error_buffer

                if group.get("log_dion2_metrics", False):
                    track_gradient_consistency(state, grad, update_input)

                update = dion2_update(
                    update_input,
                    rank_fraction=group["rank_fraction"],
                    ns_steps=group["ns_steps"],
                    selection=group["selection"],
                    subspace_cap=group["subspace_cap"],
                )

                if group.get("error_feedback", True):
                    residual = update_input - update
                    error_decay = float(group.get("error_decay", 1.0))
                    if error_decay != 1.0:
                        residual.mul_(error_decay)
                    state["error_buffer"].copy_(residual)

                if group.get("log_dion2_metrics", False):
                    track_update_ratio(
                        state,
                        p.detach().to(dtype=torch.float32),
                        update,
                        group["lr"],
                    )

                update_for_param = update.to(dtype=p.dtype)
                apply_weight_decay(
                    p,
                    update_for_param,
                    group["lr"],
                    group["weight_decay"],
                    group.get("weight_decay_type", "default"),
                    group.get("initial_lr", group["lr"]),
                )
                p.add_(update_for_param, alpha=-group["lr"])
                if use_distributed_work:
                    self._broadcast_matrix_state(
                        owner_rank,
                        p,
                        state,
                        bool(group.get("error_feedback", True)),
                    )

        return loss

    @torch.no_grad()
    def synchronize_for_checkpoint(self) -> None:
        """Compatibility shim for APIs that expect optimizer-state sync before save."""
        if not self._checkpoint_sync_enabled():
            return None
        if self.process_group is None or self.world_size <= 1:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None

        with torch.no_grad():
            for group in self.param_groups:
                if not group.get("use_dion2", False) or not bool(
                    group.get("distribute_work", False)
                ):
                    continue
                for p in group["params"]:
                    state = self.state.get(p, {})
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.is_floating_point():
                            dist.all_reduce(
                                value,
                                op=dist.ReduceOp.SUM,
                                group=self.process_group,
                            )
                            value.div_(float(self.world_size))
        return None
