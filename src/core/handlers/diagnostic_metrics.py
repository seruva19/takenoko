from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F


def _gather_metric_tensor(accelerator: Any, values: torch.Tensor) -> torch.Tensor:
    values = values.detach().to(torch.float32)
    if values.numel() == 0:
        return values

    gather_for_metrics = getattr(accelerator, "gather_for_metrics", None)
    if callable(gather_for_metrics):
        try:
            return gather_for_metrics(values)
        except Exception:
            pass

    gather = getattr(accelerator, "gather", None)
    if callable(gather):
        try:
            return gather(values)
        except Exception:
            pass

    return values


def _reduce_scalar(
    accelerator: Any,
    value: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    reduce_fn = getattr(accelerator, "reduce", None)
    value = value.detach().to(torch.float32)
    if callable(reduce_fn):
        try:
            return reduce_fn(value, reduction=reduction)
        except TypeError:
            try:
                reduced = reduce_fn(value)
                if reduction == "mean":
                    world_size = max(int(getattr(accelerator, "num_processes", 1)), 1)
                    return reduced / float(world_size)
                return reduced
            except Exception:
                pass
        except Exception:
            pass

    return value


def _compute_per_sample_loss_stats(
    accelerator: Any,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    hard_example_fraction: float,
) -> Dict[str, float]:
    per_sample_loss = F.mse_loss(
        model_pred.detach().to(torch.float32),
        target.detach().to(torch.float32),
        reduction="none",
    )
    per_sample_loss = per_sample_loss.reshape(per_sample_loss.shape[0], -1).mean(dim=1)
    gathered = _gather_metric_tensor(accelerator, per_sample_loss)
    if gathered.numel() == 0:
        return {}

    loss_std = gathered.std(correction=0).item() if gathered.numel() > 1 else 0.0
    loss_var = gathered.var(correction=0).item() if gathered.numel() > 1 else 0.0
    topk_count = max(1, int(math.ceil(gathered.numel() * hard_example_fraction)))
    hard_topk_mean = gathered.topk(topk_count).values.mean().item()

    return {
        "train/loss_std_in_batch": float(loss_std),
        "train/loss_var_in_batch": float(loss_var),
        "train/loss_hard_topk_mean": float(hard_topk_mean),
        "train/loss_max_in_batch": float(gathered.max().item()),
    }


def _compute_boundary_frame_loss_stats(
    accelerator: Any,
    model_pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    if model_pred.dim() != 5 or target.dim() != 5:
        return {}
    if model_pred.shape[2] < 2 or target.shape[2] < 2:
        return {}

    first_loss = F.mse_loss(
        model_pred[:, :, 0].detach().to(torch.float32),
        target[:, :, 0].detach().to(torch.float32),
        reduction="none",
    )
    last_loss = F.mse_loss(
        model_pred[:, :, -1].detach().to(torch.float32),
        target[:, :, -1].detach().to(torch.float32),
        reduction="none",
    )
    first_loss = first_loss.reshape(first_loss.shape[0], -1).mean(dim=1)
    last_loss = last_loss.reshape(last_loss.shape[0], -1).mean(dim=1)

    gathered_first = _gather_metric_tensor(accelerator, first_loss)
    gathered_last = _gather_metric_tensor(accelerator, last_loss)
    if gathered_first.numel() == 0 or gathered_last.numel() == 0:
        return {}

    first_mean = float(gathered_first.mean().item())
    last_mean = float(gathered_last.mean().item())
    return {
        "video/loss_first_frame": first_mean,
        "video/loss_last_frame": last_mean,
        "video/loss_delta": float(last_mean - first_mean),
    }


def collect_diagnostic_metrics_logs(
    args: Any,
    accelerator: Any,
    global_step: int,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_components: Any,
) -> Dict[str, float]:
    """Collect opt-in diagnostic metrics that are absent from the default logging surface."""
    if not bool(getattr(args, "enable_diagnostic_metrics", False)):
        return {}

    logs: Dict[str, float] = {}

    if bool(
        getattr(args, "diagnostic_metrics_enable_effective_batch_size", True)
    ) and bool(getattr(accelerator, "sync_gradients", False)):
        local_batch = torch.tensor(
            float(model_pred.shape[0]),
            device=model_pred.device,
            dtype=torch.float32,
        )
        global_batch = _reduce_scalar(accelerator, local_batch, reduction="sum")
        grad_accum = max(int(getattr(args, "gradient_accumulation_steps", 1)), 1)
        logs["train/effective_batch_size"] = float(global_batch.item()) * float(
            grad_accum
        )

    interval = max(int(getattr(args, "diagnostic_metrics_interval", 50)), 1)
    if (int(global_step) % interval) != 0:
        return logs if bool(getattr(accelerator, "is_main_process", True)) else {}

    if bool(getattr(args, "diagnostic_metrics_enable_per_sample_loss_stats", True)):
        logs.update(
            _compute_per_sample_loss_stats(
                accelerator=accelerator,
                model_pred=model_pred,
                target=target,
                hard_example_fraction=float(
                    getattr(args, "diagnostic_metrics_hard_example_fraction", 0.1)
                ),
            )
        )

    if bool(getattr(args, "diagnostic_metrics_enable_boundary_frame_loss", True)):
        logs.update(
            _compute_boundary_frame_loss_stats(
                accelerator=accelerator,
                model_pred=model_pred,
                target=target,
            )
        )

    if bool(getattr(args, "diagnostic_metrics_enable_optical_flow_error", True)):
        optical_flow_loss = getattr(loss_components, "optical_flow_loss", None)
        if optical_flow_loss is not None:
            reduced_flow = _reduce_scalar(
                accelerator,
                optical_flow_loss.detach().to(torch.float32).mean(),
                reduction="mean",
            )
            logs["train/optical_flow_error"] = float(reduced_flow.item())

    return logs if bool(getattr(accelerator, "is_main_process", True)) else {}
