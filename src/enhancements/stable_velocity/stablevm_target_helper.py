from __future__ import annotations

from collections import deque
import math
from typing import Any, Deque, Dict, Optional, Tuple

import torch

from common.logger import get_logger
from enhancements.repa.stable_velocity_weighting import normalize_timesteps

logger = get_logger(__name__)

ShapeKey = Tuple[int, ...]
ClassBankKey = Tuple[ShapeKey, int]


class StableVMTargetHelper:
    """Train-time StableVM target helper with per-label FIFO banks.

    This helper is intentionally scoped to training target construction only.
    It does not alter inference code paths.
    """

    def __init__(self, args: Any) -> None:
        self.enabled = bool(
            getattr(args, "enable_stable_velocity", False)
            and getattr(args, "stable_velocity_stablevm_enable_target", False)
        )
        self.path_type = str(
            getattr(args, "stable_velocity_stablevm_path_type", "linear")
        ).lower()
        self.label_source = str(
            getattr(args, "stable_velocity_stablevm_label_source", "auto")
        ).lower()
        self.t_min = float(getattr(args, "stable_velocity_stablevm_t_min", 0.5))
        self.blend = float(getattr(args, "stable_velocity_stablevm_blend", 1.0))
        self.bank_capacity = int(
            getattr(args, "stable_velocity_stablevm_bank_capacity_per_label", 256)
        )
        self.refs_per_sample = int(
            getattr(args, "stable_velocity_stablevm_refs_per_sample", 64)
        )
        self.min_refs = int(getattr(args, "stable_velocity_stablevm_min_refs", 8))
        self.ref_chunk_size = int(
            getattr(args, "stable_velocity_stablevm_ref_chunk_size", 16)
        )
        self.use_global_fallback = bool(
            getattr(args, "stable_velocity_stablevm_use_global_fallback", True)
        )
        self.eps = float(getattr(args, "stable_velocity_stablevm_numerical_eps", 1e-8))
        self.log_interval = int(
            getattr(args, "stable_velocity_stablevm_log_interval", 100)
        )
        self.max_timestep = float(getattr(args, "max_timestep", 1000))

        self._class_banks: Dict[ClassBankKey, Deque[torch.Tensor]] = {}
        self._global_banks: Dict[ShapeKey, Deque[torch.Tensor]] = {}
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def _shape_key(self, tensor: torch.Tensor) -> ShapeKey:
        return tuple(int(v) for v in tensor.shape[1:])

    def _make_bank(self) -> Deque[torch.Tensor]:
        return deque(maxlen=self.bank_capacity)

    def _resolve_labels(
        self, batch: Dict[str, Any], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        def _coerce(name: str) -> Optional[torch.Tensor]:
            values = batch.get(name)
            if values is None:
                return None
            if torch.is_tensor(values):
                out = values.detach().to(device=device, dtype=torch.long).view(-1)
            else:
                try:
                    out = torch.tensor(values, device=device, dtype=torch.long).view(-1)
                except Exception:
                    return None
            if out.numel() < batch_size:
                pad = out.new_full((batch_size - out.numel(),), -1)
                out = torch.cat([out, pad], dim=0)
            return out[:batch_size]

        concept = _coerce("concept_id")
        dataset = _coerce("dataset_index")
        fallback = torch.full((batch_size,), -1, device=device, dtype=torch.long)

        if self.label_source == "concept_id":
            return concept if concept is not None else fallback
        if self.label_source == "dataset_index":
            return dataset if dataset is not None else fallback

        if concept is not None and bool((concept >= 0).any().item()):
            return concept
        if dataset is not None and bool((dataset >= 0).any().item()):
            return dataset
        return concept if concept is not None else (dataset if dataset is not None else fallback)

    def _interpolant(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.path_type == "linear":
            alpha_t = 1.0 - t
            sigma_t = t
            d_alpha_t = -torch.ones_like(t)
            d_sigma_t = torch.ones_like(t)
        elif self.path_type == "cosine":
            angle = t * (math.pi / 2.0)
            alpha_t = torch.cos(angle)
            sigma_t = torch.sin(angle)
            d_alpha_t = -(math.pi / 2.0) * torch.sin(angle)
            d_sigma_t = (math.pi / 2.0) * torch.cos(angle)
        else:
            raise ValueError(
                f"Unsupported stable_velocity_stablevm_path_type: {self.path_type!r}"
            )
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def _sample_references(
        self, shape_key: ShapeKey, label: int, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], bool]:
        class_bank = self._class_banks.get((shape_key, label))
        use_global = False
        pool: Optional[Deque[torch.Tensor]] = None

        if label >= 0 and class_bank is not None and len(class_bank) >= self.min_refs:
            pool = class_bank
        elif self.use_global_fallback:
            global_bank = self._global_banks.get(shape_key)
            if global_bank is not None and len(global_bank) >= self.min_refs:
                pool = global_bank
                use_global = True

        if pool is None or len(pool) < self.min_refs:
            return None, use_global

        count = min(self.refs_per_sample, len(pool))
        if count <= 0:
            return None, use_global

        if len(pool) <= count:
            selected = list(pool)
        else:
            indices = torch.randperm(len(pool))[:count].tolist()
            selected = [pool[i] for i in indices]

        refs = torch.stack(selected, dim=0).to(device=device, dtype=torch.float32)
        return refs, use_global

    @torch.no_grad()
    def _compute_stable_target(
        self,
        noisy_latents: torch.Tensor,
        t_norm: torch.Tensor,
        ref_latents: torch.Tensor,
    ) -> torch.Tensor:
        n = int(noisy_latents.shape[0])
        if n == 0:
            return noisy_latents.new_zeros(noisy_latents.shape, dtype=torch.float32)

        xt = noisy_latents.detach().to(dtype=torch.float32)
        refs = ref_latents.detach().to(device=xt.device, dtype=torch.float32)
        t = t_norm.detach().to(device=xt.device, dtype=torch.float32).clamp(
            min=self.eps, max=1.0 - self.eps
        )

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._interpolant(t)
        sigma_sq = (sigma_t * sigma_t).clamp_min(self.eps).unsqueeze(1)

        view_shape = (n,) + (1,) * (xt.dim() - 1)
        alpha_exp = alpha_t.view(view_shape)
        sigma_exp = sigma_t.view(view_shape)
        d_alpha_exp = d_alpha_t.view(view_shape)
        d_sigma_exp = d_sigma_t.view(view_shape)

        log_prob_chunks = []
        for start in range(0, refs.shape[0], self.ref_chunk_size):
            ref_chunk = refs[start : start + self.ref_chunk_size]
            scaled = ref_chunk.unsqueeze(0) * alpha_exp.unsqueeze(1)
            diff = xt.unsqueeze(1) - scaled
            dist = diff.flatten(2).pow(2).sum(dim=2)
            log_prob = (-dist / (2.0 * sigma_sq)).clamp(min=-1e4, max=0.0)
            log_prob_chunks.append(log_prob)

        logits = torch.cat(log_prob_chunks, dim=1)
        logits = logits - logits.max(dim=1, keepdim=True).values
        logits = logits.clamp(min=-50.0, max=0.0)
        weights = torch.softmax(logits, dim=1)

        if torch.isnan(weights).any():
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            weights = weights / denom

        target = torch.zeros_like(xt, dtype=torch.float32)
        d_sigma_ratio = d_sigma_exp / (sigma_exp + self.eps)

        for start in range(0, refs.shape[0], self.ref_chunk_size):
            ref_chunk = refs[start : start + self.ref_chunk_size]
            w_chunk = weights[:, start : start + ref_chunk.shape[0]]
            w_chunk = w_chunk.view((n, ref_chunk.shape[0]) + (1,) * (xt.dim() - 1))

            b_scale = (
                d_alpha_exp.unsqueeze(1) - d_sigma_ratio.unsqueeze(1) * alpha_exp.unsqueeze(1)
            ) * ref_chunk.unsqueeze(0)
            x_scale = d_sigma_ratio.unsqueeze(1) * xt.unsqueeze(1)
            target = target + (w_chunk * (x_scale + b_scale)).sum(dim=1)

        return target

    @torch.no_grad()
    def _update_banks(self, clean_latents: torch.Tensor, labels: torch.Tensor) -> float:
        batch = clean_latents.detach().to(device="cpu", dtype=torch.float16)
        labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
        shape_key = self._shape_key(batch)

        global_bank = self._global_banks.get(shape_key)
        if global_bank is None:
            global_bank = self._make_bank()
            self._global_banks[shape_key] = global_bank

        for idx in range(batch.shape[0]):
            sample = batch[idx].contiguous()
            global_bank.append(sample)

            label = int(labels_cpu[idx].item())
            if label < 0:
                continue
            class_key = (shape_key, label)
            class_bank = self._class_banks.get(class_key)
            if class_bank is None:
                class_bank = self._make_bank()
                self._class_banks[class_key] = class_bank
            class_bank.append(sample)

        return float(len(global_bank)) / float(max(1, self.bank_capacity))

    @torch.no_grad()
    def apply_to_target(
        self,
        *,
        base_target: torch.Tensor,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch: Dict[str, Any],
        global_step: Optional[int],
        update_bank: bool,
    ) -> torch.Tensor:
        self.last_metrics = {}
        if not self.enabled:
            return base_target

        batch_size = int(base_target.shape[0])
        device = noisy_latents.device
        labels = self._resolve_labels(batch, batch_size, device)
        t_norm = normalize_timesteps(timesteps, max_timestep=self.max_timestep).to(
            device=device, dtype=torch.float32
        )
        active_mask = t_norm >= self.t_min

        out_target = base_target
        applied_count = 0
        fallback_count = 0
        global_fallback_count = 0
        ref_count_sum = 0.0
        group_count = 0

        if bool(active_mask.any().item()):
            shape_key = self._shape_key(clean_latents)
            base_target_fp32 = base_target.detach().to(device=device, dtype=torch.float32)
            mixed_target_fp32 = base_target_fp32.clone()

            active_indices = active_mask.nonzero(as_tuple=True)[0]
            unique_labels = labels[active_indices].unique()

            for label_tensor in unique_labels:
                label = int(label_tensor.item())
                label_mask = labels == label_tensor
                sample_idx = (active_mask & label_mask).nonzero(as_tuple=True)[0]
                if sample_idx.numel() == 0:
                    continue

                refs, used_global = self._sample_references(shape_key, label, device)
                if refs is None:
                    continue

                ref_count_sum += float(refs.shape[0])
                group_count += 1
                if used_global:
                    global_fallback_count += int(sample_idx.numel())

                stable_target = self._compute_stable_target(
                    noisy_latents=noisy_latents[sample_idx],
                    t_norm=t_norm[sample_idx],
                    ref_latents=refs,
                )

                base_slice = base_target_fp32[sample_idx]
                blended = (1.0 - self.blend) * base_slice + self.blend * stable_target

                finite_mask = torch.isfinite(blended.flatten(1)).all(dim=1)
                if not bool(finite_mask.all().item()):
                    fallback_count += int((~finite_mask).sum().item())
                    blended[~finite_mask] = base_slice[~finite_mask]

                mixed_target_fp32[sample_idx] = blended
                applied_count += int(sample_idx.numel())

            out_target = mixed_target_fp32.to(dtype=base_target.dtype)

        bank_fill_ratio = 0.0
        if update_bank:
            try:
                bank_fill_ratio = self._update_banks(clean_latents, labels)
            except Exception as exc:
                logger.debug("StableVM bank update skipped due to error: %s", exc)

        should_log = (
            global_step is None
            or self.log_interval <= 1
            or global_step % self.log_interval == 0
        )
        if should_log:
            denom = max(1, batch_size)
            applied_ratio = float(applied_count) / float(denom)
            fallback_ratio = float(fallback_count) / float(max(1, applied_count))
            global_fallback_ratio = float(global_fallback_count) / float(max(1, applied_count))
            mean_refs = ref_count_sum / float(max(1, group_count))
            self.last_metrics = {
                "stable_velocity_target_applied_ratio": base_target.new_tensor(applied_ratio),
                "stable_velocity_target_mean_refs": base_target.new_tensor(mean_refs),
                "stable_velocity_target_fallback_ratio": base_target.new_tensor(
                    fallback_ratio
                ),
                "stable_velocity_target_global_fallback_ratio": base_target.new_tensor(
                    global_fallback_ratio
                ),
                "stable_velocity_target_bank_fill_ratio": base_target.new_tensor(
                    bank_fill_ratio
                ),
            }

        return out_target
