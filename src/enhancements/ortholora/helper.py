"""Training-only OrthoLORA gradient projection helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from common.logger import get_logger
from criteria.loss_factory import conditional_loss_with_pseudo_huber
from criteria.training_loss import _get_loss_kwargs


logger = get_logger(__name__)


@dataclass
class OrthoLoRAParamEntry:
    """Track a trainable LoRA parameter and its matrix role."""

    name: str
    role: str
    param: torch.nn.Parameter


@dataclass
class OrthoLoRABackwardPlan:
    """Projected replacement gradient for the current microstep."""

    entries: List[OrthoLoRAParamEntry]
    preexisting_grads: List[Optional[torch.Tensor]]
    aggregate_surrogate_grads: List[Optional[torch.Tensor]]
    projected_surrogate_grads: List[Optional[torch.Tensor]]
    metrics: Dict[str, float]


def _clone_optional_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().clone()


def _add_optional_tensors(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if left is None:
        return _clone_optional_tensor(right)
    if right is None:
        return _clone_optional_tensor(left)
    return left + right.to(device=left.device, dtype=left.dtype)


def _sub_optional_tensors(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if left is None and right is None:
        return None
    if left is None:
        assert right is not None
        return -right.detach().clone()
    if right is None:
        return left.detach().clone()
    return left - right.to(device=left.device, dtype=left.dtype)


def _mean_optional_tensors(
    tensors: Sequence[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    non_none = [tensor for tensor in tensors if tensor is not None]
    if not non_none:
        return None
    total = non_none[0].detach().clone()
    for tensor in non_none[1:]:
        total = total + tensor.to(device=total.device, dtype=total.dtype)
    return total / float(len(tensors))


def _gradient_dot(
    left: Sequence[Optional[torch.Tensor]], right: Sequence[Optional[torch.Tensor]]
) -> float:
    total = 0.0
    for left_tensor, right_tensor in zip(left, right):
        if left_tensor is None or right_tensor is None:
            continue
        total += float(
            torch.sum(
                left_tensor.detach().to(torch.float32)
                * right_tensor.detach().to(torch.float32)
            ).item()
        )
    return total


def _gradient_norm_sq(grads: Sequence[Optional[torch.Tensor]]) -> float:
    return max(_gradient_dot(grads, grads), 0.0)


def project_conflicting_gradients(
    group_grads: Sequence[Sequence[Optional[torch.Tensor]]],
    shuffle_order: bool = True,
) -> tuple[List[List[Optional[torch.Tensor]]], int]:
    """Project conflicting task gradients onto each other's normal plane."""

    projected: List[List[Optional[torch.Tensor]]] = [
        [_clone_optional_tensor(tensor) for tensor in grads] for grads in group_grads
    ]
    if len(projected) <= 1:
        return projected, 0

    order = list(range(len(projected)))
    if shuffle_order:
        permutation = torch.randperm(len(order)).tolist()
        order = [order[idx] for idx in permutation]

    conflict_count = 0
    for i in order:
        for j in order:
            if i == j:
                continue
            dot = _gradient_dot(projected[i], group_grads[j])
            if dot >= 0.0:
                continue
            norm_sq = _gradient_norm_sq(group_grads[j])
            if norm_sq <= 1e-12:
                continue
            coeff = dot / norm_sq
            updated: List[Optional[torch.Tensor]] = []
            for current, reference in zip(projected[i], group_grads[j]):
                if current is None:
                    updated.append(None)
                    continue
                if reference is None:
                    updated.append(current)
                    continue
                updated.append(
                    current
                    - reference.to(device=current.device, dtype=current.dtype) * coeff
                )
            projected[i] = updated
            conflict_count += 1

    return projected, conflict_count


class OrthoLoRAHelper:
    """Apply OrthoLORA projection to trainable LoRA gradients only."""

    def __init__(self, args: Any):
        self.enabled = bool(getattr(args, "enable_ortholora", False))
        self.group_by = str(getattr(args, "ortholora_group_by", "dataset_index"))
        self.min_active_groups = int(
            getattr(args, "ortholora_min_active_groups", 2)
        )
        self.min_group_samples = int(
            getattr(args, "ortholora_min_group_samples", 1)
        )
        self.log_metrics = bool(getattr(args, "ortholora_log_metrics", True))
        self._cached_network_id: Optional[int] = None
        self._cached_entries: List[OrthoLoRAParamEntry] = []
        self._warned_missing_groups = False
        self._warned_missing_params = False

    def setup_hooks(self) -> None:
        """No-op hook API for parity with other training helpers."""

    def remove_hooks(self) -> None:
        """No-op hook API for parity with other training helpers."""

    def _discover_lora_entries(self, network: Any) -> List[OrthoLoRAParamEntry]:
        network_id = id(network)
        if self._cached_network_id == network_id:
            return self._cached_entries

        entries: List[OrthoLoRAParamEntry] = []
        seen_param_ids: set[int] = set()
        for name, param in network.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            lowered = name.lower()
            role: Optional[str] = None
            if "lora_down" in lowered and (
                lowered.endswith(".weight") or lowered.endswith("lora_down")
            ):
                role = "lora_down"
            elif "lora_up" in lowered and (
                lowered.endswith(".weight") or lowered.endswith("lora_up")
            ):
                role = "lora_up"
            if role is None:
                continue
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            entries.append(OrthoLoRAParamEntry(name=name, role=role, param=param))

        self._cached_network_id = network_id
        self._cached_entries = entries
        return entries

    def _extract_group_labels(
        self, batch: Dict[str, Any], model_pred: torch.Tensor
    ) -> List[str]:
        raw_values: Optional[Iterable[Any]] = None

        if self.group_by in batch:
            raw_values = batch[self.group_by]
        elif self.group_by == "media_type":
            item_info = batch.get("item_info")
            if item_info is not None:
                labels: List[str] = []
                for item in item_info:
                    frame_count = getattr(item, "frame_count", None)
                    labels.append(
                        "video"
                        if frame_count is not None and int(frame_count) > 1
                        else "image"
                    )
                return labels
            if model_pred.dim() == 5 and model_pred.shape[2] > 1:
                return ["video"] * int(model_pred.shape[0])
            return ["image"] * int(model_pred.shape[0])

        if raw_values is None:
            return []
        if torch.is_tensor(raw_values):
            raw_values = raw_values.detach().cpu().tolist()

        return [str(value) for value in raw_values]

    def _compute_per_sample_surrogate_loss(
        self,
        args: Any,
        accelerator: Any,
        batch: Dict[str, Any],
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        latents: torch.Tensor,
        noise_scheduler: Optional[Any],
        network_dtype: torch.dtype,
    ) -> torch.Tensor:
        loss = conditional_loss_with_pseudo_huber(
            model_pred.to(network_dtype),
            target.to(network_dtype),
            loss_type=args.loss_type,
            huber_c=args.pseudo_huber_c,
            current_step=getattr(args, "current_step", None),
            total_steps=getattr(args, "total_steps", None),
            schedule_type=args.pseudo_huber_schedule_type,
            c_min=args.pseudo_huber_c_min,
            c_max=args.pseudo_huber_c_max,
            reduction="none",
            timesteps=timesteps,
            noise=noise,
            noisy_latents=noisy_model_input,
            clean_latents=latents,
            noise_scheduler=noise_scheduler,
            **_get_loss_kwargs(args),
        )

        sample_weights = batch.get("weight")
        if sample_weights is not None:
            sample_weights = sample_weights.to(
                device=accelerator.device,
                dtype=network_dtype,
            )
            while sample_weights.dim() < loss.dim():
                sample_weights = sample_weights.unsqueeze(-1)
            loss = loss * sample_weights

        mask = batch.get("mask_signal")
        if (
            mask is not None
            and not bool(getattr(args, "use_masked_training_with_prior", False))
        ):
            mask = mask.to(device=accelerator.device, dtype=network_dtype)
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask

        if weighting is not None:
            loss = loss * weighting.to(device=loss.device, dtype=loss.dtype)

        return loss.view(loss.size(0), -1).mean(dim=1)

    def _build_group_losses(
        self, per_sample_loss: torch.Tensor, group_labels: Sequence[str]
    ) -> List[torch.Tensor]:
        group_indices: Dict[str, List[int]] = {}
        for index, label in enumerate(group_labels):
            group_indices.setdefault(label, []).append(index)

        losses: List[torch.Tensor] = []
        for indices in group_indices.values():
            if len(indices) < self.min_group_samples:
                continue
            losses.append(per_sample_loss[indices].mean())
        return losses

    def prepare_backward_plan(
        self,
        args: Any,
        accelerator: Any,
        network: Any,
        batch: Dict[str, Any],
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        latents: torch.Tensor,
        noise_scheduler: Optional[Any],
        network_dtype: torch.dtype,
    ) -> Optional[OrthoLoRABackwardPlan]:
        if not self.enabled:
            return None

        entries = self._discover_lora_entries(network)
        if not entries:
            if not self._warned_missing_params:
                logger.warning(
                    "OrthoLORA enabled but no trainable lora_down/lora_up parameters were discovered; projection is disabled."
                )
                self._warned_missing_params = True
            return None

        group_labels = self._extract_group_labels(batch, model_pred)
        if len(group_labels) != int(model_pred.shape[0]):
            if not self._warned_missing_groups:
                logger.warning(
                    "OrthoLORA could not resolve '%s' labels for the current batch; projection is skipped.",
                    self.group_by,
                )
                self._warned_missing_groups = True
            return None

        per_sample_loss = self._compute_per_sample_surrogate_loss(
            args=args,
            accelerator=accelerator,
            batch=batch,
            model_pred=model_pred,
            target=target,
            weighting=weighting,
            timesteps=timesteps,
            noise=noise,
            noisy_model_input=noisy_model_input,
            latents=latents,
            noise_scheduler=noise_scheduler,
            network_dtype=network_dtype,
        )
        group_losses = self._build_group_losses(per_sample_loss, group_labels)
        if len(group_losses) < self.min_active_groups:
            return None

        params = [entry.param for entry in entries]
        group_grads: List[List[Optional[torch.Tensor]]] = []
        for loss in group_losses:
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            group_grads.append([_clone_optional_tensor(tensor) for tensor in grads])

        aggregate_surrogate_grads = [
            _mean_optional_tensors([group[idx] for group in group_grads])
            for idx in range(len(entries))
        ]
        projected_surrogate_grads: List[Optional[torch.Tensor]] = [
            _clone_optional_tensor(tensor) for tensor in aggregate_surrogate_grads
        ]

        conflict_pairs_down = 0
        conflict_pairs_up = 0
        for role in ("lora_down", "lora_up"):
            role_indices = [
                idx for idx, entry in enumerate(entries) if entry.role == role
            ]
            if not role_indices:
                continue
            role_group_grads = [
                [group[idx] for idx in role_indices] for group in group_grads
            ]
            projected_role_grads, conflict_count = project_conflicting_gradients(
                role_group_grads,
                shuffle_order=True,
            )
            role_means = [
                _mean_optional_tensors([group[idx] for group in projected_role_grads])
                for idx in range(len(role_indices))
            ]
            for local_idx, param_idx in enumerate(role_indices):
                projected_surrogate_grads[param_idx] = role_means[local_idx]
            if role == "lora_down":
                conflict_pairs_down = conflict_count
            else:
                conflict_pairs_up = conflict_count

        preexisting_grads = [_clone_optional_tensor(entry.param.grad) for entry in entries]
        metrics = {
            "ortholora/active_groups": float(len(group_losses)),
            "ortholora/conflict_pairs_down": float(conflict_pairs_down),
            "ortholora/conflict_pairs_up": float(conflict_pairs_up),
            "ortholora/eligible_params": float(len(entries)),
        }
        return OrthoLoRABackwardPlan(
            entries=entries,
            preexisting_grads=preexisting_grads,
            aggregate_surrogate_grads=aggregate_surrogate_grads,
            projected_surrogate_grads=projected_surrogate_grads,
            metrics=metrics,
        )

    def apply_backward_plan(self, plan: Optional[OrthoLoRABackwardPlan]) -> None:
        if plan is None:
            return

        with torch.no_grad():
            for idx, entry in enumerate(plan.entries):
                current_grad = _clone_optional_tensor(entry.param.grad)
                preexisting_grad = plan.preexisting_grads[idx]
                current_microstep_grad = _sub_optional_tensors(
                    current_grad,
                    preexisting_grad,
                )
                auxiliary_grad = _sub_optional_tensors(
                    current_microstep_grad,
                    plan.aggregate_surrogate_grads[idx],
                )
                replacement_microstep_grad = _add_optional_tensors(
                    auxiliary_grad,
                    plan.projected_surrogate_grads[idx],
                )
                final_grad = _add_optional_tensors(
                    preexisting_grad,
                    replacement_microstep_grad,
                )

                if final_grad is None:
                    entry.param.grad = None
                    continue

                final_grad = final_grad.to(
                    device=entry.param.device,
                    dtype=entry.param.dtype,
                )
                if entry.param.grad is None:
                    entry.param.grad = final_grad
                else:
                    entry.param.grad.copy_(final_grad)
