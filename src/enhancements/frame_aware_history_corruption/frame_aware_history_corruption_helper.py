from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class FrameAwareHistoryCorruptionState:
    applied: bool
    history_frame_start: int
    history_frame_end: int
    exposure_frames: int
    noise_frames: int
    blur_frames: int
    clean_frames: int
    first_frame_preserved: bool


class FrameAwareHistoryCorruptionHelper:
    """Train-time frame-aware history corruption (LoRA-only, gated, default off)."""

    def __init__(self, args: Any) -> None:
        self.enabled = bool(
            getattr(args, "enable_frame_aware_history_corruption", False)
        )
        self.keep_first_frame = bool(
            getattr(args, "frame_aware_history_corruption_keep_first_frame", True)
        )
        self.exposure_prob = float(
            getattr(args, "frame_aware_history_corruption_exposure_prob", 0.0)
        )
        self.noise_prob = float(
            getattr(args, "frame_aware_history_corruption_noise_prob", 0.0)
        )
        self.blur_prob = float(
            getattr(args, "frame_aware_history_corruption_blur_prob", 0.0)
        )
        self.clean_prob = float(
            getattr(args, "frame_aware_history_corruption_clean_prob", 1.0)
        )
        self.exposure_min = float(
            getattr(args, "frame_aware_history_corruption_exposure_min", 0.85)
        )
        self.exposure_max = float(
            getattr(args, "frame_aware_history_corruption_exposure_max", 1.15)
        )
        self.noise_min = float(
            getattr(args, "frame_aware_history_corruption_noise_min", 0.01)
        )
        self.noise_max = float(
            getattr(args, "frame_aware_history_corruption_noise_max", 0.08)
        )
        self.downsample_min = float(
            getattr(args, "frame_aware_history_corruption_downsample_min", 1.25)
        )
        self.downsample_max = float(
            getattr(args, "frame_aware_history_corruption_downsample_max", 2.5)
        )
        self.history_start_frame = int(
            getattr(args, "frame_aware_history_corruption_history_start_frame", 0)
        )
        self.history_exclude_tail_frames = int(
            getattr(
                args,
                "frame_aware_history_corruption_history_exclude_tail_frames",
                1,
            )
        )
        self.blend = float(
            getattr(args, "frame_aware_history_corruption_blend", 1.0)
        )
        self.log_interval = int(
            getattr(args, "frame_aware_history_corruption_log_interval", 50)
        )
        self._warned_invalid_shape = False

    def setup_hooks(self) -> None:
        """Reserved for parity with other enhancement helpers."""

    def remove_hooks(self) -> None:
        """Reserved for parity with other enhancement helpers."""

    def _resolve_history_slice(self, num_frames: int) -> Tuple[int, int]:
        start = max(0, self.history_start_frame)
        end = max(start, num_frames - self.history_exclude_tail_frames)
        end = min(end, num_frames)
        return start, end

    def _build_state(
        self,
        *,
        applied: bool,
        history_frame_start: int,
        history_frame_end: int,
        exposure_frames: int = 0,
        noise_frames: int = 0,
        blur_frames: int = 0,
        clean_frames: int = 0,
        first_frame_preserved: bool = False,
    ) -> FrameAwareHistoryCorruptionState:
        return FrameAwareHistoryCorruptionState(
            applied=applied,
            history_frame_start=history_frame_start,
            history_frame_end=history_frame_end,
            exposure_frames=exposure_frames,
            noise_frames=noise_frames,
            blur_frames=blur_frames,
            clean_frames=clean_frames,
            first_frame_preserved=first_frame_preserved,
        )

    def _sample_mode(self, device: torch.device) -> str:
        probs = torch.tensor(
            [
                self.exposure_prob,
                self.noise_prob,
                self.blur_prob,
                self.clean_prob,
            ],
            device=device,
            dtype=torch.float32,
        )
        mode_idx = int(torch.multinomial(probs, num_samples=1).item())
        modes = ("exposure", "noise", "blur", "clean")
        return modes[mode_idx]

    def _apply_to_frame(self, *, frame: torch.Tensor, mode: str) -> torch.Tensor:
        orig_dtype = frame.dtype
        frame_fp32 = frame.to(torch.float32)

        if mode == "clean":
            return frame

        if mode == "exposure":
            scale = torch.empty(
                (frame_fp32.shape[0], 1, 1, 1, 1),
                device=frame_fp32.device,
                dtype=torch.float32,
            ).uniform_(self.exposure_min, self.exposure_max)
            return (frame_fp32 * scale).to(orig_dtype)

        if mode == "noise":
            noise_level = torch.empty(
                (frame_fp32.shape[0], 1, 1, 1, 1),
                device=frame_fp32.device,
                dtype=torch.float32,
            ).uniform_(self.noise_min, self.noise_max)
            noisy = frame_fp32 + torch.randn_like(frame_fp32) * noise_level
            return noisy.to(orig_dtype)

        if mode == "blur":
            if frame_fp32.ndim != 5 or frame_fp32.shape[2] != 1:
                return frame
            height = int(frame_fp32.shape[-2])
            width = int(frame_fp32.shape[-1])
            factor = float(
                torch.empty((), device=frame_fp32.device, dtype=torch.float32).uniform_(
                    self.downsample_min,
                    self.downsample_max,
                )
            )
            down_h = max(1, int(round(height / max(factor, 1.0))))
            down_w = max(1, int(round(width / max(factor, 1.0))))
            frame_2d = frame_fp32[:, :, 0, :, :]
            down = F.interpolate(
                frame_2d,
                size=(down_h, down_w),
                mode="bilinear",
                align_corners=False,
            )
            up = F.interpolate(
                down,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            return up.unsqueeze(2).to(orig_dtype)

        return frame

    def apply_to_inputs(
        self,
        noisy_model_input: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[FrameAwareHistoryCorruptionState]]:
        del global_step
        if not self.enabled:
            return noisy_model_input, None

        if noisy_model_input.dim() != 5:
            if not self._warned_invalid_shape:
                logger.warning(
                    "Frame-aware history corruption skipped: expected 5D tensor, got noisy=%s",
                    tuple(noisy_model_input.shape),
                )
                self._warned_invalid_shape = True
            return noisy_model_input, self._build_state(
                applied=False,
                history_frame_start=0,
                history_frame_end=0,
            )

        history_start, history_end = self._resolve_history_slice(
            noisy_model_input.shape[2]
        )
        if history_end <= history_start:
            return noisy_model_input, self._build_state(
                applied=False,
                history_frame_start=history_start,
                history_frame_end=history_end,
            )

        counts = {
            "exposure": 0,
            "noise": 0,
            "blur": 0,
            "clean": 0,
        }
        corrupted = noisy_model_input.clone()
        batch_size = int(corrupted.shape[0])
        for frame_idx in range(history_start, history_end):
            if self.keep_first_frame and frame_idx == 0:
                continue
            for sample_idx in range(batch_size):
                mode = self._sample_mode(corrupted.device)
                frame_slice = corrupted[
                    sample_idx : sample_idx + 1,
                    :,
                    frame_idx : frame_idx + 1,
                    ...,
                ]
                corrupted[
                    sample_idx : sample_idx + 1,
                    :,
                    frame_idx : frame_idx + 1,
                    ...,
                ] = self._apply_to_frame(frame=frame_slice, mode=mode)
                counts[mode] += 1

        blended = noisy_model_input.clone()
        src_slice = noisy_model_input[:, :, history_start:history_end, ...]
        corrupted_slice = corrupted[:, :, history_start:history_end, ...]
        if self.blend >= 1.0:
            blended[:, :, history_start:history_end, ...] = corrupted_slice
        else:
            blended[:, :, history_start:history_end, ...] = (
                (1.0 - self.blend) * src_slice + self.blend * corrupted_slice
            )

        return blended, self._build_state(
            applied=(
                counts["exposure"]
                + counts["noise"]
                + counts["blur"]
                + counts["clean"]
            )
            > 0,
            history_frame_start=history_start,
            history_frame_end=history_end,
            exposure_frames=counts["exposure"],
            noise_frames=counts["noise"],
            blur_frames=counts["blur"],
            clean_frames=counts["clean"],
            first_frame_preserved=(
                self.keep_first_frame and history_start <= 0 < history_end
            ),
        )

    def state_to_metrics(
        self,
        state: Optional[FrameAwareHistoryCorruptionState],
    ) -> Dict[str, float]:
        if state is None:
            return {}
        history_span = max(0, state.history_frame_end - state.history_frame_start)
        return {
            "frame_aware_history_corruption/applied": 1.0 if state.applied else 0.0,
            "frame_aware_history_corruption/history_span_frames": float(history_span),
            "frame_aware_history_corruption/exposure_frames": float(
                state.exposure_frames
            ),
            "frame_aware_history_corruption/noise_frames": float(state.noise_frames),
            "frame_aware_history_corruption/blur_frames": float(state.blur_frames),
            "frame_aware_history_corruption/clean_frames": float(state.clean_frames),
            "frame_aware_history_corruption/first_frame_preserved": (
                1.0 if state.first_frame_preserved else 0.0
            ),
        }
