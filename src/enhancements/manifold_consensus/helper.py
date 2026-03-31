"""Manifold-consensus helper for WAN LoRA training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ManifoldConsensusConfig:
    enabled: bool
    weight: float
    num_views: int
    layer_start: int
    layer_end: int
    min_timestep: float
    max_timestep: float
    start_step: int
    interval_steps: int
    apply_prob: float
    loss_type: str
    include_student_in_target: bool
    normalize_features: bool

    @classmethod
    def from_args(cls, args: Any) -> "ManifoldConsensusConfig":
        return cls(
            enabled=bool(getattr(args, "enable_manifold_consensus", False)),
            weight=float(getattr(args, "manifold_consensus_weight", 0.1)),
            num_views=int(getattr(args, "manifold_consensus_num_views", 3)),
            layer_start=int(getattr(args, "manifold_consensus_layer_start", 20)),
            layer_end=int(getattr(args, "manifold_consensus_layer_end", 29)),
            min_timestep=float(
                getattr(args, "manifold_consensus_min_timestep", 970.0)
            ),
            max_timestep=float(
                getattr(args, "manifold_consensus_max_timestep", 1000.0)
            ),
            start_step=int(getattr(args, "manifold_consensus_start_step", 0)),
            interval_steps=int(getattr(args, "manifold_consensus_interval_steps", 1)),
            apply_prob=float(getattr(args, "manifold_consensus_apply_prob", 1.0)),
            loss_type=str(
                getattr(args, "manifold_consensus_loss_type", "mse")
            ).lower(),
            include_student_in_target=bool(
                getattr(args, "manifold_consensus_include_student_in_target", False)
            ),
            normalize_features=bool(
                getattr(args, "manifold_consensus_normalize_features", True)
            ),
        )


@dataclass
class ViewCapture:
    pooled_feature: torch.Tensor
    layer_states: Dict[int, torch.Tensor]


class ManifoldConsensusHelper(nn.Module):
    """Capture layer states, build a merged state, and score a continuation pass."""

    def __init__(self, transformer: nn.Module, args: Any) -> None:
        super().__init__()
        self.config = ManifoldConsensusConfig.from_args(args)
        self.enabled = self.config.enabled
        self.student = self._unwrap_model(transformer)

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._capture_slot: Optional[str] = None
        self._pending_pooled: Dict[int, torch.Tensor] = {}
        self._pending_states: Dict[int, torch.Tensor] = {}
        self._selected_layer_indices: List[int] = []

        self._step_active = False
        self._sample_mask: Optional[torch.Tensor] = None
        self._sample_timestep_values: Optional[torch.Tensor] = None
        self._student_capture: Optional[ViewCapture] = None
        self._consensus_pooled_sum: Optional[torch.Tensor] = None
        self._consensus_layer_state_sums: Dict[int, torch.Tensor] = {}
        self._consensus_count: int = 0
        self._pooled_view_features: List[torch.Tensor] = []

        self.last_loss: Optional[torch.Tensor] = None
        self.last_feature_cosine_similarity: Optional[torch.Tensor] = None
        self.last_prediction_mse: Optional[torch.Tensor] = None
        self.last_view_variance: Optional[torch.Tensor] = None
        self.last_active_ratio: Optional[torch.Tensor] = None
        self.last_timestep_mean: Optional[torch.Tensor] = None
        self.last_view_count: Optional[torch.Tensor] = None

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        tensor = output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            tensor = output[0]
        if torch.is_tensor(tensor):
            return tensor
        return None

    @staticmethod
    def _replace_output(original_output: Any, new_tensor: torch.Tensor) -> Any:
        if torch.is_tensor(original_output):
            return new_tensor
        if isinstance(original_output, tuple) and len(original_output) > 0:
            return (new_tensor,) + tuple(original_output[1:])
        if isinstance(original_output, list) and len(original_output) > 0:
            result = list(original_output)
            result[0] = new_tensor
            return result
        return new_tensor

    @staticmethod
    def _pool_hidden_state(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor.mean(dim=1)
        if tensor.ndim == 5:
            return tensor.mean(dim=(2, 3, 4))
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1).mean(dim=1, keepdim=True)

    def _get_blocks(self) -> List[nn.Module]:
        blocks = getattr(self.student, "blocks", None)
        if blocks is None and hasattr(self.student, "module"):
            blocks = getattr(self.student.module, "blocks", None)
        if blocks is None:
            raise ValueError("Manifold consensus requires transformer.blocks to exist")
        return list(blocks)

    def _make_capture_hook(self, layer_idx: int):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if self._capture_slot is None:
                return
            tensor = self._extract_tensor(output)
            if tensor is None:
                return
            self._pending_pooled[layer_idx] = self._pool_hidden_state(tensor)
            self._pending_states[layer_idx] = tensor.detach()

        return _hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        if not self.enabled:
            return

        blocks = self._get_blocks()
        depth = len(blocks)
        start = self.config.layer_start
        end = self.config.layer_end
        if start < 0 or end >= depth:
            raise ValueError(
                f"manifold consensus layer range [{start}, {end}] is out of range for depth={depth}"
            )

        self._selected_layer_indices = list(range(start, end + 1))
        for idx in self._selected_layer_indices:
            self._hooks.append(blocks[idx].register_forward_hook(self._make_capture_hook(idx)))

        logger.info(
            "Manifold consensus hooks ready: layers=%d-%d views=%d timestep_band=%.1f-%.1f",
            start,
            end,
            self.config.num_views,
            self.config.min_timestep,
            self.config.max_timestep,
        )

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks = []
        self.reset_step_state(clear_metrics=True)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return []

    def _resolve_sample_timestep_values(self, timesteps: torch.Tensor) -> torch.Tensor:
        values = timesteps.detach().float()
        if values.ndim <= 1:
            return values.view(-1)
        return values.reshape(values.shape[0], -1).mean(dim=1)

    def prepare_step(self, global_step: int, timesteps: torch.Tensor) -> bool:
        self.reset_step_state(clear_metrics=False)
        if not self.enabled or self.config.weight <= 0.0:
            return False
        if global_step < self.config.start_step:
            return False
        if ((global_step - self.config.start_step) % self.config.interval_steps) != 0:
            return False
        if self.config.apply_prob < 1.0 and random.random() > self.config.apply_prob:
            return False

        sample_values = self._resolve_sample_timestep_values(timesteps)
        sample_mask = (sample_values >= self.config.min_timestep) & (
            sample_values <= self.config.max_timestep
        )
        if not bool(sample_mask.any().item()):
            return False

        self._step_active = True
        self._sample_timestep_values = sample_values
        self._sample_mask = sample_mask
        self.last_timestep_mean = sample_values.mean().detach()
        self.last_active_ratio = sample_mask.float().mean().detach()
        return True

    def _begin_capture(self, slot: str) -> None:
        if not self._step_active:
            return
        self._capture_slot = slot
        self._pending_pooled = {}
        self._pending_states = {}

    def _finish_capture(self) -> Optional[ViewCapture]:
        slot = self._capture_slot
        self._capture_slot = None
        if slot is None or not self._pending_pooled:
            self._pending_pooled = {}
            self._pending_states = {}
            return None

        pooled_feature = torch.stack(
            [self._pending_pooled[idx] for idx in self._selected_layer_indices if idx in self._pending_pooled],
            dim=0,
        ).mean(dim=0)
        layer_states = {
            idx: self._pending_states[idx]
            for idx in self._selected_layer_indices
            if idx in self._pending_states
        }
        self._pending_pooled = {}
        self._pending_states = {}
        return ViewCapture(pooled_feature=pooled_feature, layer_states=layer_states)

    def begin_student_capture(self) -> None:
        self._student_capture = None
        self._consensus_pooled_sum = None
        self._consensus_layer_state_sums = {}
        self._consensus_count = 0
        self._pooled_view_features = []
        self._begin_capture("student")

    def finish_student_capture(self) -> Optional[ViewCapture]:
        capture = self._finish_capture()
        self._student_capture = capture
        if capture is not None and self.config.include_student_in_target:
            self._accumulate_capture(capture)
        return capture

    def begin_extra_view_capture(self) -> None:
        self._begin_capture("extra")

    def finish_extra_view_capture(self) -> Optional[ViewCapture]:
        capture = self._finish_capture()
        if capture is not None:
            self._accumulate_capture(capture)
        return capture

    def has_active_student_feature(self) -> bool:
        return self._step_active and self._student_capture is not None

    def should_collect_extra_views(self) -> bool:
        return self.has_active_student_feature() and self.config.num_views > 1

    def _accumulate_capture(self, capture: ViewCapture) -> None:
        pooled = capture.pooled_feature.detach()
        self._pooled_view_features.append(pooled)
        if self._consensus_pooled_sum is None:
            self._consensus_pooled_sum = pooled.clone()
        else:
            self._consensus_pooled_sum = self._consensus_pooled_sum + pooled

        for idx, state in capture.layer_states.items():
            detached_state = state.detach()
            if idx not in self._consensus_layer_state_sums:
                self._consensus_layer_state_sums[idx] = detached_state.clone()
            else:
                self._consensus_layer_state_sums[idx] = (
                    self._consensus_layer_state_sums[idx] + detached_state
                )
        self._consensus_count += 1

    def build_consensus_states(self) -> Optional[Dict[int, torch.Tensor]]:
        if self._consensus_count <= 0:
            return None

        consensus_states: Dict[int, torch.Tensor] = {}
        for idx in self._selected_layer_indices:
            if idx not in self._consensus_layer_state_sums:
                continue
            consensus_states[idx] = (
                self._consensus_layer_state_sums[idx] / float(self._consensus_count)
            ).detach()
        return consensus_states or None

    def build_consensus_feature(self) -> Optional[torch.Tensor]:
        if self._consensus_count <= 0 or self._consensus_pooled_sum is None:
            return None
        return (self._consensus_pooled_sum / float(self._consensus_count)).detach()

    def install_consensus_injection_hooks(
        self, consensus_states: Dict[int, torch.Tensor]
    ) -> List[torch.utils.hooks.RemovableHandle]:
        handles: List[torch.utils.hooks.RemovableHandle] = []
        blocks = self._get_blocks()

        for idx, merged_state in consensus_states.items():
            def _make_inject_hook(
                layer_idx: int, target_state: torch.Tensor
            ):
                def _hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
                    tensor = self._extract_tensor(output)
                    if tensor is None:
                        return output
                    replacement = target_state.to(device=tensor.device, dtype=tensor.dtype)
                    return self._replace_output(output, replacement)

                return _hook

            handles.append(blocks[idx].register_forward_hook(_make_inject_hook(idx, merged_state)))

        return handles

    def _weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor],
    ) -> torch.Tensor:
        err = (pred - target).pow(2)
        per_sample = err.view(pred.size(0), -1).mean(dim=1)
        if weighting is not None:
            w = weighting
            if w.ndim > 1:
                w = w.view(w.size(0), -1).mean(dim=1)
            if w.shape[0] != per_sample.shape[0]:
                w = w.expand(per_sample.shape[0])
            per_sample = per_sample * w.to(device=per_sample.device, dtype=per_sample.dtype)
        return per_sample.mean()

    def compute_loss(
        self,
        *,
        student_pred: torch.Tensor,
        consensus_teacher_pred: Optional[torch.Tensor],
        weighting: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        metrics: Dict[str, float] = {}
        self.last_loss = None
        self.last_feature_cosine_similarity = None
        self.last_prediction_mse = None
        self.last_view_variance = None
        self.last_view_count = None

        if not self.enabled or not self._step_active:
            return None, metrics
        if consensus_teacher_pred is None or self._student_capture is None:
            return None, metrics

        teacher_pred = consensus_teacher_pred.detach()
        if self.config.loss_type == "one_minus_cosine":
            student_flat = student_pred.view(student_pred.size(0), -1)
            teacher_flat = teacher_pred.view(teacher_pred.size(0), -1)
            per_sample = 1.0 - F.cosine_similarity(
                F.normalize(student_flat, dim=-1),
                F.normalize(teacher_flat, dim=-1),
                dim=-1,
            )
            if weighting is not None:
                w = weighting
                if w.ndim > 1:
                    w = w.view(w.size(0), -1).mean(dim=1)
                per_sample = per_sample * w.to(device=per_sample.device, dtype=per_sample.dtype)
            pred_loss = per_sample.mean()
        else:
            pred_loss = self._weighted_mse(student_pred, teacher_pred, weighting)

        loss = pred_loss * self.config.weight
        self.last_loss = loss.detach()
        self.last_prediction_mse = self._weighted_mse(student_pred.detach(), teacher_pred, weighting).detach()

        consensus_feature = self.build_consensus_feature()
        if consensus_feature is not None:
            student_feature = self._student_capture.pooled_feature
            if self.config.normalize_features:
                student_feature = F.normalize(student_feature, dim=-1)
                consensus_feature = F.normalize(consensus_feature, dim=-1)
            cosine = F.cosine_similarity(student_feature, consensus_feature, dim=-1).mean()
            self.last_feature_cosine_similarity = cosine.detach()

        if self._pooled_view_features:
            feature_stack = torch.stack(self._pooled_view_features, dim=0)
            self.last_view_variance = feature_stack.var(dim=0, unbiased=False).mean().detach()
            self.last_view_count = torch.tensor(
                float(len(self._pooled_view_features)),
                device=student_pred.device,
                dtype=student_pred.dtype,
            )

        if self.last_feature_cosine_similarity is not None:
            metrics["manifold_consensus/feature_cosine_similarity"] = float(
                self.last_feature_cosine_similarity.item()
            )
        if self.last_prediction_mse is not None:
            metrics["manifold_consensus/prediction_mse"] = float(
                self.last_prediction_mse.item()
            )
        if self.last_view_variance is not None:
            metrics["manifold_consensus/view_variance"] = float(
                self.last_view_variance.item()
            )
        if self.last_active_ratio is not None:
            metrics["manifold_consensus/active_ratio"] = float(
                self.last_active_ratio.item()
            )
        if self.last_timestep_mean is not None:
            metrics["manifold_consensus/timestep_mean"] = float(
                self.last_timestep_mean.item()
            )
        if self.last_view_count is not None:
            metrics["manifold_consensus/view_count"] = float(
                self.last_view_count.item()
            )
        return loss, metrics

    def reset_step_state(self, *, clear_metrics: bool = False) -> None:
        self._capture_slot = None
        self._pending_pooled = {}
        self._pending_states = {}
        self._step_active = False
        self._sample_mask = None
        self._sample_timestep_values = None
        self._student_capture = None
        self._consensus_pooled_sum = None
        self._consensus_layer_state_sums = {}
        self._consensus_count = 0
        self._pooled_view_features = []
        if clear_metrics:
            self.last_loss = None
            self.last_feature_cosine_similarity = None
            self.last_prediction_mse = None
            self.last_view_variance = None
            self.last_active_ratio = None
            self.last_timestep_mean = None
            self.last_view_count = None
