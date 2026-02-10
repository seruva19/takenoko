"""Training-time drifting loss helper (train-only, inference-neutral)."""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _QueueEntry:
    feature: torch.Tensor
    label: Optional[int]
    base_weight: float


class DriftingLossHelper:
    """Compute mean-shift style drifting losses with optional queue + multi-scale support."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.enabled = bool(getattr(args, "enable_drifting", False))
        self.temperature = float(getattr(args, "drifting_temperature", 0.05))
        self.max_feature_dim = int(getattr(args, "drifting_max_feature_dim", 1024))
        self.min_batch_size = int(getattr(args, "drifting_min_batch_size", 2))
        self.use_dual_axis_normalization = bool(
            getattr(args, "drifting_use_dual_axis_normalization", True)
        )
        self.multi_scale_enabled = bool(
            getattr(args, "drifting_multi_scale_enabled", False)
        )
        self.multi_scale_factors = list(
            getattr(args, "drifting_multi_scale_factors", [1, 2, 4])
        )
        self.multi_scale_reduce = str(
            getattr(args, "drifting_multi_scale_reduce", "mean")
        ).lower()

        self.queue_enabled = bool(getattr(args, "drifting_queue_enabled", False))
        self.queue_size_per_label = int(
            getattr(args, "drifting_queue_size_per_label", 128)
        )
        self.queue_size_global = int(getattr(args, "drifting_queue_size_global", 1000))
        self.queue_neg_size_global = int(
            getattr(args, "drifting_queue_neg_size_global", 1000)
        )
        self.queue_sample_per_step = int(
            getattr(args, "drifting_queue_sample_per_step", 64)
        )
        self.queue_warmup_steps = int(getattr(args, "drifting_queue_warmup_steps", 0))

        self.cfg_weighting_enabled = bool(
            getattr(args, "drifting_cfg_weighting_enabled", False)
        )
        self.cfg_null_label = int(getattr(args, "drifting_cfg_null_label", -1))
        self.cfg_conditional_weight = float(
            getattr(args, "drifting_cfg_conditional_weight", 1.0)
        )
        self.cfg_unconditional_weight = float(
            getattr(args, "drifting_cfg_unconditional_weight", 1.0)
        )
        self.cfg_apply_to_positive = bool(
            getattr(args, "drifting_cfg_apply_to_positive", True)
        )
        self.cfg_apply_to_negative = bool(
            getattr(args, "drifting_cfg_apply_to_negative", True)
        )
        self.cfg_use_batch_weight = bool(
            getattr(args, "drifting_cfg_use_batch_weight", False)
        )
        self.feature_encoder_enabled = bool(
            getattr(args, "drifting_feature_encoder_enabled", False)
        )
        self.feature_encoder_path = getattr(args, "drifting_feature_encoder_path", None)
        self.feature_encoder_input_size = int(
            getattr(args, "drifting_feature_encoder_input_size", 224)
        )
        self.feature_encoder_frame_reduce = str(
            getattr(args, "drifting_feature_encoder_frame_reduce", "mean")
        ).lower()
        self.feature_encoder_channel_mode = str(
            getattr(args, "drifting_feature_encoder_channel_mode", "first3")
        ).lower()
        self.feature_encoder_use_fp16 = bool(
            getattr(args, "drifting_feature_encoder_use_fp16", True)
        )
        self.feature_encoder_imagenet_norm = bool(
            getattr(args, "drifting_feature_encoder_imagenet_norm", False)
        )
        self.feature_encoder_strict = bool(
            getattr(args, "drifting_feature_encoder_strict", False)
        )

        self._pos_queue_global: Deque[_QueueEntry] = deque(maxlen=self.queue_size_global)
        self._neg_queue_global: Deque[_QueueEntry] = deque(
            maxlen=self.queue_neg_size_global
        )
        self._pos_queue_by_label: Dict[int, Deque[_QueueEntry]] = {}
        self._neg_queue_by_label: Dict[int, Deque[_QueueEntry]] = {}
        self._feature_encoder: Optional[torch.nn.Module] = None
        self._feature_encoder_failed = False
        self._feature_encoder_dtype: Optional[torch.dtype] = None
        self._last_metrics: Dict[str, torch.Tensor] = {}

    def setup_hooks(self) -> None:
        """Hook placeholder for interface consistency with other enhancements."""

    def remove_hooks(self) -> None:
        """Hook cleanup placeholder for interface consistency."""

    def _flatten_batch_features(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError("Drifting expects batch-first tensors with ndim >= 2")
        batch = int(tensor.shape[0])
        flattened = tensor.reshape(batch, -1).to(dtype=torch.float32)
        if flattened.shape[1] <= self.max_feature_dim:
            return flattened

        pooled = F.adaptive_avg_pool1d(
            flattened.unsqueeze(1),
            self.max_feature_dim,
        ).squeeze(1)
        return pooled

    def _normalize_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        if self.use_dual_axis_normalization:
            row_sum = kernel.sum(dim=-1, keepdim=True)
            col_sum = kernel.sum(dim=0, keepdim=True)
            normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
        else:
            normalizer = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return kernel / normalizer

    def _compute_drift_from_banks(
        self,
        *,
        gen: torch.Tensor,
        pos_bank: torch.Tensor,
        neg_bank: torch.Tensor,
        pos_weights: Optional[torch.Tensor],
        neg_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dist_pos = torch.cdist(gen, pos_bank, p=2)
        dist_neg = torch.cdist(gen, neg_bank, p=2)
        kernel_pos = torch.exp(-dist_pos / max(self.temperature, 1e-12))
        kernel_neg = torch.exp(-dist_neg / max(self.temperature, 1e-12))

        if pos_weights is not None and self.cfg_apply_to_positive:
            kernel_pos = kernel_pos * pos_weights.unsqueeze(0)
        if neg_weights is not None and self.cfg_apply_to_negative:
            kernel_neg = kernel_neg * neg_weights.unsqueeze(0)

        normalized_pos = self._normalize_kernel(kernel_pos)
        normalized_neg = self._normalize_kernel(kernel_neg)

        pos_coeff = normalized_pos * normalized_neg.sum(dim=-1, keepdim=True)
        neg_coeff = normalized_neg * normalized_pos.sum(dim=-1, keepdim=True)

        pos_v = pos_coeff @ pos_bank
        neg_v = neg_coeff @ neg_bank
        return pos_v - neg_v

    def _scale_tensor(self, tensor: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return tensor
        if tensor.ndim == 5:
            out_size = (
                max(1, int(tensor.shape[2]) // factor),
                max(1, int(tensor.shape[3]) // factor),
                max(1, int(tensor.shape[4]) // factor),
            )
            return F.adaptive_avg_pool3d(tensor.to(torch.float32), out_size)
        if tensor.ndim == 4:
            out_size = (
                max(1, int(tensor.shape[2]) // factor),
                max(1, int(tensor.shape[3]) // factor),
            )
            return F.adaptive_avg_pool2d(tensor.to(torch.float32), out_size)
        return tensor

    def _select_encoder_output(self, output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                selected = self._select_encoder_output(item)
                if selected is not None:
                    return selected
            return None
        if isinstance(output, dict):
            preferred_keys = (
                "pooler_output",
                "last_hidden_state",
                "features",
                "feature",
                "embedding",
                "embeddings",
                "output",
                "logits",
            )
            for key in preferred_keys:
                if key in output:
                    selected = self._select_encoder_output(output[key])
                    if selected is not None:
                        return selected
            for value in output.values():
                selected = self._select_encoder_output(value)
                if selected is not None:
                    return selected
            return None
        return None

    def _ensure_feature_encoder(
        self,
        *,
        device: torch.device,
    ) -> Optional[torch.nn.Module]:
        if not self.feature_encoder_enabled:
            return None
        if self._feature_encoder_failed:
            return None
        if self._feature_encoder is None:
            path = str(self.feature_encoder_path or "")
            if not path:
                self._feature_encoder_failed = True
                msg = (
                    "Drifting feature encoder is enabled but no path is configured."
                )
                if self.feature_encoder_strict:
                    raise RuntimeError(msg)
                logger.warning(msg)
                return None
            if not os.path.exists(path):
                self._feature_encoder_failed = True
                msg = f"Drifting feature encoder path does not exist: {path}"
                if self.feature_encoder_strict:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                return None

            module: Optional[torch.nn.Module] = None
            load_error: Optional[Exception] = None
            try:
                loaded = torch.jit.load(path, map_location="cpu")
                if isinstance(loaded, torch.nn.Module):
                    module = loaded
            except Exception as exc:
                load_error = exc

            if module is None:
                try:
                    loaded = torch.load(path, map_location="cpu")
                    if isinstance(loaded, torch.nn.Module):
                        module = loaded
                    elif isinstance(loaded, dict):
                        candidate = loaded.get("model")
                        if isinstance(candidate, torch.nn.Module):
                            module = candidate
                except Exception as exc:
                    load_error = exc

            if module is None:
                self._feature_encoder_failed = True
                msg = (
                    "Failed to load drifting feature encoder as torch module from "
                    f"{path}. Last error: {load_error}"
                )
                if self.feature_encoder_strict:
                    raise RuntimeError(msg)
                logger.warning(msg)
                return None

            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
            self._feature_encoder = module
            logger.info(
                "Loaded drifting feature encoder from %s.",
                path,
            )

        encoder = self._feature_encoder
        if encoder is None:
            return None

        target_dtype = torch.float32
        if self.feature_encoder_use_fp16 and device.type == "cuda":
            target_dtype = torch.float16
        if self._feature_encoder_dtype != target_dtype:
            encoder.to(device=device, dtype=target_dtype)
            self._feature_encoder_dtype = target_dtype
        else:
            encoder.to(device=device)
        return encoder

    def _prepare_encoder_input(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        x = tensor
        if x.ndim == 5:
            if self.feature_encoder_frame_reduce == "max":
                x = x.max(dim=2).values
            elif self.feature_encoder_frame_reduce == "middle":
                x = x[:, :, x.shape[2] // 2]
            else:
                x = x.mean(dim=2)

        if x.ndim != 4:
            return None

        channels = int(x.shape[1])
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif channels == 2:
            x = torch.cat([x, x[:, :1]], dim=1)
        elif channels > 3:
            if self.feature_encoder_channel_mode == "mean_to_rgb":
                mean_channel = x.mean(dim=1, keepdim=True)
                x = mean_channel.repeat(1, 3, 1, 1)
            else:
                x = x[:, :3]

        size = int(max(16, self.feature_encoder_input_size))
        if x.shape[-2] != size or x.shape[-1] != size:
            x = F.interpolate(
                x,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )

        if self.feature_encoder_imagenet_norm:
            x = (x + 1.0) * 0.5
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
        return x

    def _extract_features_with_encoder(
        self,
        *,
        tensor: torch.Tensor,
        requires_grad: bool,
    ) -> Optional[torch.Tensor]:
        encoder = self._ensure_feature_encoder(device=tensor.device)
        if encoder is None:
            return None

        encoder_input = self._prepare_encoder_input(tensor.to(torch.float32))
        if encoder_input is None:
            return None

        target_dtype = self._feature_encoder_dtype or torch.float32
        encoder_input = encoder_input.to(dtype=target_dtype)
        if requires_grad:
            output = encoder(encoder_input)
        else:
            with torch.no_grad():
                output = encoder(encoder_input)
        feature = self._select_encoder_output(output)
        if feature is None:
            return None
        return self._flatten_batch_features(feature)

    def _extract_features(
        self,
        *,
        tensor: torch.Tensor,
        requires_grad: bool,
    ) -> Tuple[torch.Tensor, bool]:
        if self.feature_encoder_enabled:
            encoded = self._extract_features_with_encoder(
                tensor=tensor,
                requires_grad=requires_grad,
            )
            if encoded is not None:
                return encoded, True
        if requires_grad:
            return self._flatten_batch_features(tensor), False
        with torch.no_grad():
            return self._flatten_batch_features(tensor), False

    def _resolve_scale_factors(self, tensor: torch.Tensor) -> List[int]:
        if not self.multi_scale_enabled:
            return [1]
        if tensor.ndim < 4:
            return [1]
        factors = sorted({int(max(1, scale)) for scale in self.multi_scale_factors})
        return factors or [1]

    def _extract_class_labels(
        self,
        *,
        batch: Optional[Dict[str, Any]],
        batch_size: int,
    ) -> Optional[List[Optional[int]]]:
        if not isinstance(batch, dict):
            return None
        keys = (
            "class_labels",
            "labels",
            "class_ids",
            "categories",
            "y",
            "class",
            "target",
            "caption_ids",
        )
        for key in keys:
            if key not in batch or batch[key] is None:
                continue
            value = batch[key]
            if isinstance(value, torch.Tensor):
                labels = value.detach().view(-1)
                if labels.numel() < batch_size:
                    continue
                return [int(v.item()) for v in labels[:batch_size]]
            if isinstance(value, (list, tuple)) and len(value) >= batch_size:
                out: List[Optional[int]] = []
                for item in value[:batch_size]:
                    try:
                        out.append(int(item))
                    except Exception:
                        out.append(None)
                return out
        return None

    def _extract_sample_weights(
        self,
        *,
        batch: Optional[Dict[str, Any]],
        batch_size: int,
    ) -> Optional[List[float]]:
        if not isinstance(batch, dict):
            return None
        if "weight" not in batch or batch["weight"] is None:
            return None
        value = batch["weight"]
        if isinstance(value, torch.Tensor):
            weights = value.detach().view(-1)
            if weights.numel() < batch_size:
                return None
            return [float(v.item()) for v in weights[:batch_size]]
        if isinstance(value, (list, tuple)) and len(value) >= batch_size:
            out: List[float] = []
            for item in value[:batch_size]:
                try:
                    out.append(float(item))
                except Exception:
                    out.append(1.0)
            return out
        return None

    def _compute_entry_weight(self, label: Optional[int], base_weight: float) -> float:
        weight = 1.0
        if self.cfg_weighting_enabled:
            if label is not None and int(label) == self.cfg_null_label:
                weight *= max(self.cfg_unconditional_weight, 0.0)
            else:
                weight *= max(self.cfg_conditional_weight, 0.0)
        if self.cfg_use_batch_weight:
            weight *= max(float(base_weight), 0.0)
        return max(weight, 0.0)

    def _entries_from_features(
        self,
        *,
        features: torch.Tensor,
        labels: Optional[List[Optional[int]]],
        sample_weights: Optional[List[float]],
    ) -> List[_QueueEntry]:
        batch_size = int(features.shape[0])
        if labels is None or len(labels) < batch_size:
            labels = [None] * batch_size
        if sample_weights is None or len(sample_weights) < batch_size:
            sample_weights = [1.0] * batch_size

        entries: List[_QueueEntry] = []
        for i in range(batch_size):
            entries.append(
                _QueueEntry(
                    feature=features[i].detach().to(device="cpu", dtype=torch.float32),
                    label=labels[i],
                    base_weight=float(sample_weights[i]),
                )
            )
        return entries

    def _get_label_queue(
        self,
        *,
        storage: Dict[int, Deque[_QueueEntry]],
        label: int,
    ) -> Deque[_QueueEntry]:
        queue = storage.get(label)
        if queue is None:
            queue = deque(maxlen=self.queue_size_per_label)
            storage[label] = queue
        return queue

    def _sample_from_queue(
        self,
        queue: Deque[_QueueEntry],
        count: int,
    ) -> List[_QueueEntry]:
        if count <= 0 or len(queue) == 0:
            return []
        items = list(queue)
        if len(items) <= count:
            return items
        indices = random.sample(range(len(items)), count)
        return [items[i] for i in indices]

    def _sample_label_queues(
        self,
        *,
        labels: Optional[List[Optional[int]]],
        storage: Dict[int, Deque[_QueueEntry]],
        budget: int,
    ) -> List[_QueueEntry]:
        if budget <= 0 or not labels:
            return []
        selected_labels = sorted(
            {
                int(label)
                for label in labels
                if label is not None and int(label) in storage
            }
        )
        if not selected_labels:
            return []
        per_label = max(1, budget // len(selected_labels))
        sampled: List[_QueueEntry] = []
        for label in selected_labels:
            sampled.extend(self._sample_from_queue(storage[label], per_label))
        return sampled

    def _select_bank_entries(
        self,
        *,
        current_entries: List[_QueueEntry],
        labels: Optional[List[Optional[int]]],
        global_queue: Deque[_QueueEntry],
        label_queues: Dict[int, Deque[_QueueEntry]],
        global_step: Optional[int],
    ) -> List[_QueueEntry]:
        selected = list(current_entries)
        if not self.queue_enabled:
            return selected
        if self.queue_sample_per_step <= 0:
            return selected
        if global_step is not None and global_step < self.queue_warmup_steps:
            return selected

        selected.extend(
            self._sample_from_queue(global_queue, self.queue_sample_per_step)
        )
        selected.extend(
            self._sample_label_queues(
                labels=labels,
                storage=label_queues,
                budget=self.queue_sample_per_step,
            )
        )
        return selected

    def _build_bank_tensors(
        self,
        *,
        entries: Sequence[_QueueEntry],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bank = torch.stack(
            [entry.feature.to(device=device, dtype=torch.float32) for entry in entries],
            dim=0,
        )
        weights = torch.tensor(
            [self._compute_entry_weight(entry.label, entry.base_weight) for entry in entries],
            device=device,
            dtype=torch.float32,
        )
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        return bank, weights

    def _update_queues(
        self,
        *,
        pos_entries: List[_QueueEntry],
        neg_entries: List[_QueueEntry],
    ) -> None:
        if not self.queue_enabled:
            return
        if self.queue_size_global > 0:
            for entry in pos_entries:
                self._pos_queue_global.append(entry)
        if self.queue_neg_size_global > 0:
            for entry in neg_entries:
                self._neg_queue_global.append(entry)
        if self.queue_size_per_label > 0:
            for entry in pos_entries:
                if entry.label is None:
                    continue
                self._get_label_queue(
                    storage=self._pos_queue_by_label,
                    label=int(entry.label),
                ).append(entry)
            for entry in neg_entries:
                if entry.label is None:
                    continue
                self._get_label_queue(
                    storage=self._neg_queue_by_label,
                    label=int(entry.label),
                ).append(entry)

    def compute_loss(
        self,
        *,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        if model_pred.shape[0] < self.min_batch_size or target.shape[0] < self.min_batch_size:
            return None
        if model_pred.shape[0] != target.shape[0]:
            return None

        batch_size = int(model_pred.shape[0])
        labels = self._extract_class_labels(batch=batch, batch_size=batch_size)
        sample_weights = self._extract_sample_weights(batch=batch, batch_size=batch_size)

        scale_factors = self._resolve_scale_factors(model_pred)
        losses: List[torch.Tensor] = []
        drift_norms: List[torch.Tensor] = []
        feature_dims: List[int] = []
        encoder_used_any = False
        queue_updated = False
        for scale in scale_factors:
            scaled_pred = self._scale_tensor(model_pred, scale)
            scaled_target = self._scale_tensor(target, scale)
            gen_features, encoder_used = self._extract_features(
                tensor=scaled_pred,
                requires_grad=True,
            )
            pos_features, _ = self._extract_features(
                tensor=scaled_target,
                requires_grad=False,
            )
            encoder_used_any = encoder_used_any or encoder_used
            if gen_features.shape[1] != pos_features.shape[1]:
                feature_dim = min(gen_features.shape[1], pos_features.shape[1])
                gen_features = gen_features[:, :feature_dim]
                pos_features = pos_features[:, :feature_dim]
            feature_dims.append(int(gen_features.shape[1]))

            current_pos_entries = self._entries_from_features(
                features=pos_features,
                labels=labels,
                sample_weights=sample_weights,
            )
            current_neg_entries = self._entries_from_features(
                features=gen_features,
                labels=labels,
                sample_weights=sample_weights,
            )

            pos_entries = self._select_bank_entries(
                current_entries=current_pos_entries,
                labels=labels,
                global_queue=self._pos_queue_global,
                label_queues=self._pos_queue_by_label,
                global_step=global_step,
            )
            neg_entries = self._select_bank_entries(
                current_entries=current_neg_entries,
                labels=labels,
                global_queue=self._neg_queue_global,
                label_queues=self._neg_queue_by_label,
                global_step=global_step,
            )

            if not pos_entries or not neg_entries:
                continue

            pos_bank, pos_weights = self._build_bank_tensors(
                entries=pos_entries,
                device=gen_features.device,
            )
            neg_bank, neg_weights = self._build_bank_tensors(
                entries=neg_entries,
                device=gen_features.device,
            )

            with torch.no_grad():
                drift = self._compute_drift_from_banks(
                    gen=gen_features.detach(),
                    pos_bank=pos_bank.detach(),
                    neg_bank=neg_bank.detach(),
                    pos_weights=pos_weights.detach(),
                    neg_weights=neg_weights.detach(),
                )
                drift_target = (gen_features.detach() + drift).detach()

            losses.append(F.mse_loss(gen_features, drift_target))
            drift_norms.append(torch.norm(drift, dim=-1).mean())

            if not queue_updated:
                self._update_queues(
                    pos_entries=current_pos_entries,
                    neg_entries=current_neg_entries,
                )
                queue_updated = True

        if not losses:
            return None

        if self.multi_scale_reduce == "sum":
            loss = torch.stack(losses, dim=0).sum()
        else:
            loss = torch.stack(losses, dim=0).mean()

        drift_norm = torch.stack(drift_norms, dim=0).mean()
        mean_feature_dim = (
            float(sum(feature_dims)) / max(1, len(feature_dims))
            if feature_dims
            else float(self.max_feature_dim)
        )
        self._last_metrics = {
            "drifting/drift_norm_mean": drift_norm.detach(),
            "drifting/feature_dim": drift_norm.new_tensor(mean_feature_dim),
            "drifting/effective_batch": drift_norm.new_tensor(float(batch_size)),
            "drifting/active_scales": drift_norm.new_tensor(float(len(scale_factors))),
            "drifting/queue_pos_size": drift_norm.new_tensor(
                float(len(self._pos_queue_global))
            ),
            "drifting/queue_neg_size": drift_norm.new_tensor(
                float(len(self._neg_queue_global))
            ),
            "drifting/encoder_active": drift_norm.new_tensor(
                1.0 if encoder_used_any else 0.0
            ),
            "drifting/encoder_feature_dim": drift_norm.new_tensor(mean_feature_dim),
        }
        return loss

    def get_metrics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_metrics)
