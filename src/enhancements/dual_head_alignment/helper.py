"""Paper-inspired dual-head alignment helper for WAN LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualHeadAlignmentConfig:
    enabled: bool
    global_weight: float
    local_weight: float
    window_frames: int
    window_stride: int
    local_recon_weight: float
    local_behavior_weight: float
    local_kl_weight: float
    local_div_forward_weight: float
    local_div_reverse_weight: float
    temperature: float
    teacher_mode: str
    head_lr_scale: float
    start_step: int
    teacher_interval_steps: int
    local_weight_ramp_steps: int
    window_sampling: str
    max_windows: int
    random_seed: int

    @classmethod
    def from_args(cls, args: Any) -> "DualHeadAlignmentConfig":
        return cls(
            enabled=bool(getattr(args, "enable_dual_head_alignment", False)),
            global_weight=float(getattr(args, "dual_head_alignment_global_weight", 1.0)),
            local_weight=float(getattr(args, "dual_head_alignment_local_weight", 1.0)),
            window_frames=int(getattr(args, "dual_head_alignment_window_frames", 0)),
            window_stride=int(getattr(args, "dual_head_alignment_window_stride", 0)),
            local_recon_weight=float(
                getattr(args, "dual_head_alignment_local_recon_weight", 0.0)
            ),
            local_behavior_weight=float(
                getattr(args, "dual_head_alignment_local_behavior_weight", 1.0)
            ),
            local_kl_weight=float(
                getattr(args, "dual_head_alignment_local_kl_weight", 0.0)
            ),
            local_div_forward_weight=float(
                getattr(args, "dual_head_alignment_local_div_forward_weight", 0.0)
            ),
            local_div_reverse_weight=float(
                getattr(args, "dual_head_alignment_local_div_reverse_weight", 1.0)
            ),
            temperature=float(getattr(args, "dual_head_alignment_temperature", 1.0)),
            teacher_mode=str(
                getattr(args, "dual_head_alignment_teacher_mode", "base_model")
            ).lower(),
            head_lr_scale=float(
                getattr(args, "dual_head_alignment_head_lr_scale", 1.0)
            ),
            start_step=int(getattr(args, "dual_head_alignment_start_step", 0)),
            teacher_interval_steps=int(
                getattr(args, "dual_head_alignment_teacher_interval_steps", 1)
            ),
            local_weight_ramp_steps=int(
                getattr(args, "dual_head_alignment_local_weight_ramp_steps", 0)
            ),
            window_sampling=str(
                getattr(args, "dual_head_alignment_window_sampling", "all")
            ).lower(),
            max_windows=int(getattr(args, "dual_head_alignment_max_windows", 0)),
            random_seed=int(getattr(args, "dual_head_alignment_random_seed", 42)),
        )


class AffineVelocityHead(nn.Module):
    """Minimal trainable velocity head with shape-agnostic affine transform."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.scale + self.bias


class DualHeadAlignmentHelper(nn.Module):
    """Training-only helper for a global/local auxiliary objective."""

    def __init__(self, args: Any):
        super().__init__()
        self.config = DualHeadAlignmentConfig.from_args(args)
        self.enabled = self.config.enabled
        self.fm_head = AffineVelocityHead()
        self.dm_head = AffineVelocityHead()

    def setup_hooks(self) -> None:
        """No-op hook API for consistency with other enhancement helpers."""
        return

    def remove_hooks(self) -> None:
        """No-op hook API for consistency with other enhancement helpers."""
        return

    def get_trainable_params(self) -> list[nn.Parameter]:
        if not self.enabled:
            return []
        return [p for p in self.parameters() if p.requires_grad]

    def _weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape for weighted MSE.")
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

    def _temporal_windows(
        self,
        tensor: torch.Tensor,
        global_step: int,
    ) -> tuple[torch.Tensor, int]:
        window_frames = self.config.window_frames
        window_stride = self.config.window_stride
        if tensor.ndim < 5 or window_frames <= 0:
            return tensor, 1

        total_frames = int(tensor.shape[2])
        if total_frames <= window_frames:
            return tensor, 1

        stride = window_stride if window_stride > 0 else window_frames
        max_start = total_frames - window_frames
        starts = list(range(0, max_start + 1, stride))
        if starts[-1] != max_start:
            starts.append(max_start)
        if self.config.max_windows > 0 and len(starts) > self.config.max_windows:
            if self.config.window_sampling == "all":
                starts = starts[: self.config.max_windows]
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(self.config.random_seed + max(global_step, 0)))
                perm = torch.randperm(len(starts), generator=gen).tolist()
                chosen = sorted(perm[: self.config.max_windows])
                starts = [starts[idx] for idx in chosen]
        windows = [tensor[:, :, s : s + window_frames, ...] for s in starts]
        return torch.cat(windows, dim=0), len(starts)

    def _expand_weighting(
        self,
        weighting: Optional[torch.Tensor],
        windows_per_sample: int,
    ) -> Optional[torch.Tensor]:
        if weighting is None or windows_per_sample <= 1:
            return weighting
        return weighting.repeat_interleave(windows_per_sample, dim=0)

    def _kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        s = student_logits.view(student_logits.size(0), -1) / temp
        t = teacher_logits.view(teacher_logits.size(0), -1) / temp
        s_log_prob = F.log_softmax(s, dim=-1)
        t_prob = F.softmax(t, dim=-1)
        return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp**2)

    def _reverse_kl(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        s = student_logits.view(student_logits.size(0), -1) / temp
        t = teacher_logits.view(teacher_logits.size(0), -1) / temp
        s_prob = F.softmax(s, dim=-1)
        t_log_prob = F.log_softmax(t, dim=-1)
        return F.kl_div(t_log_prob, s_prob, reduction="batchmean") * (temp**2)

    def should_apply_step(self, global_step: int) -> bool:
        if global_step < self.config.start_step:
            return False
        if self.config.teacher_interval_steps <= 1:
            return True
        offset = global_step - self.config.start_step
        return offset % self.config.teacher_interval_steps == 0

    def local_weight_scale(self, global_step: int) -> float:
        if global_step < self.config.start_step:
            return 0.0
        if self.config.local_weight_ramp_steps <= 0:
            return 1.0
        progressed = global_step - self.config.start_step + 1
        if progressed <= 0:
            return 0.0
        return float(min(1.0, progressed / float(self.config.local_weight_ramp_steps)))

    def compute_loss(
        self,
        *,
        student_pred: torch.Tensor,
        target: torch.Tensor,
        teacher_pred: Optional[torch.Tensor],
        weighting: Optional[torch.Tensor],
        global_step: int,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        if not self.enabled:
            return None, {}
        if teacher_pred is None:
            return None, {}
        if not self.should_apply_step(global_step):
            return None, {}

        fm_pred = self.fm_head(student_pred)
        dm_pred = self.dm_head(student_pred)

        global_loss = self._weighted_mse(fm_pred, target.detach(), weighting)

        dm_student, windows_per_sample = self._temporal_windows(dm_pred, global_step)
        dm_teacher, _ = self._temporal_windows(teacher_pred.detach(), global_step)
        dm_target, _ = self._temporal_windows(target.detach(), global_step)
        dm_weighting = self._expand_weighting(weighting, windows_per_sample)

        local_recon = self._weighted_mse(dm_student, dm_target, dm_weighting)
        local_behavior = self._weighted_mse(dm_student, dm_teacher, dm_weighting)

        local_kl = torch.tensor(0.0, device=student_pred.device, dtype=student_pred.dtype)
        if self.config.local_kl_weight > 0.0:
            local_kl = self._kl(
                dm_student,
                dm_teacher,
                self.config.temperature,
            )

        local_div_forward = torch.tensor(
            0.0, device=student_pred.device, dtype=student_pred.dtype
        )
        if self.config.local_div_forward_weight > 0.0:
            local_div_forward = self._kl(
                dm_student,
                dm_target,
                self.config.temperature,
            )

        local_div_reverse = torch.tensor(
            0.0, device=student_pred.device, dtype=student_pred.dtype
        )
        if self.config.local_div_reverse_weight > 0.0:
            local_div_reverse = self._reverse_kl(
                dm_teacher,
                dm_student,
                self.config.temperature,
            )

        local_total = (
            self.config.local_recon_weight * local_recon
            + self.config.local_behavior_weight * local_behavior
            + self.config.local_kl_weight * local_kl
            + self.config.local_div_forward_weight * local_div_forward
            + self.config.local_div_reverse_weight * local_div_reverse
        )
        local_scale = self.local_weight_scale(global_step)
        total = self.config.global_weight * global_loss + (
            self.config.local_weight * local_scale * local_total
        )

        metrics = {
            "dual_head/global_loss": float(global_loss.detach().item()),
            "dual_head/local_loss": float(local_total.detach().item()),
            "dual_head/local_recon": float(local_recon.detach().item()),
            "dual_head/local_behavior": float(local_behavior.detach().item()),
            "dual_head/local_kl": float(local_kl.detach().item()),
            "dual_head/local_div_forward": float(local_div_forward.detach().item()),
            "dual_head/local_div_reverse": float(local_div_reverse.detach().item()),
            "dual_head/windows_per_sample": float(windows_per_sample),
            "dual_head/local_weight_scale": float(local_scale),
            "dual_head/total": float(total.detach().item()),
        }
        return total, metrics

