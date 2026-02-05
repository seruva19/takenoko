from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class VideoConsistencyDistanceLossConfig:
    use_amplitude: bool = True
    use_phase: bool = True
    amplitude_weight: float = 1.0
    phase_weight: float = 1.0
    num_sampled_frames: int = 1
    random_frame_sampling: bool = True
    use_temporal_weight: bool = True
    start_step: int = 0
    end_step: Optional[int] = None
    warmup_steps: int = 0
    apply_every_n_steps: int = 1
    feature_layers: Sequence[int] = (1, 6, 11, 20, 29)
    feature_resolution: int = 224
    max_coeffs: int = 16384
    random_coeff_sampling: bool = True
    use_pretrained_vgg: bool = True
    detach_conditioning_frame: bool = True
    assume_neg_one_to_one: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "VideoConsistencyDistanceLossConfig":
        return cls(
            use_amplitude=bool(getattr(args, "vcd_use_amplitude", True)),
            use_phase=bool(getattr(args, "vcd_use_phase", True)),
            amplitude_weight=float(getattr(args, "vcd_amplitude_weight", 1.0)),
            phase_weight=float(getattr(args, "vcd_phase_weight", 1.0)),
            num_sampled_frames=int(getattr(args, "vcd_num_sampled_frames", 1)),
            random_frame_sampling=bool(
                getattr(args, "vcd_random_frame_sampling", True)
            ),
            use_temporal_weight=bool(getattr(args, "vcd_use_temporal_weight", True)),
            start_step=int(getattr(args, "vcd_start_step", 0)),
            end_step=getattr(args, "vcd_end_step", None),
            warmup_steps=int(getattr(args, "vcd_warmup_steps", 0)),
            apply_every_n_steps=int(getattr(args, "vcd_apply_every_n_steps", 1)),
            feature_layers=tuple(getattr(args, "vcd_feature_layers", [1, 6, 11, 20, 29])),
            feature_resolution=int(getattr(args, "vcd_feature_resolution", 224)),
            max_coeffs=int(getattr(args, "vcd_max_coeffs", 16384)),
            random_coeff_sampling=bool(getattr(args, "vcd_random_coeff_sampling", True)),
            use_pretrained_vgg=bool(getattr(args, "vcd_use_pretrained_vgg", True)),
            detach_conditioning_frame=bool(
                getattr(args, "vcd_detach_conditioning_frame", True)
            ),
            assume_neg_one_to_one=bool(getattr(args, "vcd_assume_neg_one_to_one", True)),
        )


class VideoConsistencyDistanceLoss:
    """VCD loss inspired by arXiv:2510.19193v2 Eq. (2).

    This helper computes Wasserstein distances between frequency-domain
    amplitude/phase distributions of shallow VGG19 frame features.
    """

    def __init__(
        self,
        config: VideoConsistencyDistanceLossConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.layer_indices = sorted(set(int(x) for x in config.feature_layers))
        self.max_layer_index = max(self.layer_indices)
        self._warned_non_rgb = False
        self._warned_vgg_fallback = False
        self._feature_extractor = self._build_feature_extractor()
        self._device = torch.device("cpu")
        if device is not None:
            self.to(device)

    @classmethod
    def from_args(
        cls,
        args: Any,
        device: Optional[torch.device] = None,
    ) -> "VideoConsistencyDistanceLoss":
        return cls(VideoConsistencyDistanceLossConfig.from_args(args), device=device)

    def _build_feature_extractor(self) -> torch.nn.Module:
        try:
            from torchvision.models import VGG19_Weights, vgg19
        except Exception as exc:
            raise RuntimeError(
                "torchvision is required for VCD. Install torchvision to use "
                "enable_video_consistency_distance."
            ) from exc

        weights = None
        if self.config.use_pretrained_vgg:
            weights = VGG19_Weights.IMAGENET1K_V1
        try:
            features = vgg19(weights=weights).features
        except Exception as exc:
            if weights is None:
                raise
            logger.warning(
                "VCD could not load pretrained VGG19 weights (%s). Falling back to "
                "randomly initialized VGG19 features.",
                exc,
            )
            self._warned_vgg_fallback = True
            features = vgg19(weights=None).features

        features.eval()
        for param in features.parameters():
            param.requires_grad_(False)
        return features

    def to(self, device: torch.device) -> None:
        self._feature_extractor = self._feature_extractor.to(device=device, dtype=torch.float32)
        self._device = device

    def _ensure_device(self, device: torch.device) -> None:
        if self._device != device:
            self.to(device)

    def _should_apply(self, step: int) -> bool:
        if step < self.config.start_step:
            return False
        if self.config.end_step is not None and step > int(self.config.end_step):
            return False
        if step % max(1, self.config.apply_every_n_steps) != 0:
            return False
        return True

    def _step_multiplier(self, step: int) -> float:
        if not self._should_apply(step):
            return 0.0
        warmup_steps = max(0, self.config.warmup_steps)
        if warmup_steps == 0:
            return 1.0
        if step < self.config.start_step + warmup_steps:
            progress = (step - self.config.start_step) / float(warmup_steps)
            return float(max(0.0, min(1.0, progress)))
        return 1.0

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(torch.float32)
        if self.config.assume_neg_one_to_one:
            x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)

    def _ensure_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        channels = int(frames.shape[1])
        if channels == 3:
            return frames
        if channels == 1:
            return frames.repeat(1, 3, 1, 1)
        if channels > 3:
            if not self._warned_non_rgb:
                logger.warning(
                    "VCD received %d-channel frames; using the first 3 channels for VGG19.",
                    channels,
                )
                self._warned_non_rgb = True
            return frames[:, :3, :, :]
        raise ValueError(f"VCD requires at least 1 channel, got {channels}")

    def _resize_if_needed(self, frames: torch.Tensor) -> torch.Tensor:
        size = int(self.config.feature_resolution)
        if size <= 0:
            return frames
        if frames.shape[-2] == size and frames.shape[-1] == size:
            return frames
        return F.interpolate(
            frames,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )

    def _extract_selected_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = x
        selected: List[torch.Tensor] = []
        for idx, layer in enumerate(self._feature_extractor):
            out = layer(out)
            if idx in self.layer_indices:
                selected.append(out)
            if idx >= self.max_layer_index:
                break
        return selected

    def _match_coefficient_count(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != y.shape[1]:
            count = min(x.shape[1], y.shape[1])
            x = x[:, :count]
            y = y[:, :count]

        coeff_count = x.shape[1]
        max_coeffs = int(self.config.max_coeffs)
        if max_coeffs <= 0 or coeff_count <= max_coeffs:
            return x, y

        if self.config.random_coeff_sampling:
            idx = torch.randperm(coeff_count, device=x.device)[:max_coeffs]
        else:
            idx = torch.linspace(
                0,
                coeff_count - 1,
                max_coeffs,
                device=x.device,
            ).long()
        x = x.index_select(1, idx)
        y = y.index_select(1, idx)
        return x, y

    def _wasserstein_1d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        x_flat, y_flat = self._match_coefficient_count(x_flat, y_flat)
        x_sorted = torch.sort(x_flat, dim=1).values
        y_sorted = torch.sort(y_flat, dim=1).values
        return torch.mean(torch.abs(x_sorted - y_sorted), dim=1)

    def _per_layer_vcd(self, cond_feat: torch.Tensor, frame_feat: torch.Tensor) -> torch.Tensor:
        cond_fft = torch.fft.fft2(cond_feat.to(torch.float32), dim=(-2, -1))
        frame_fft = torch.fft.fft2(frame_feat.to(torch.float32), dim=(-2, -1))

        per_sample = torch.zeros(cond_feat.shape[0], device=cond_feat.device, dtype=torch.float32)
        total_weight = 0.0

        if self.config.use_amplitude and self.config.amplitude_weight > 0.0:
            amp_distance = self._wasserstein_1d(torch.abs(cond_fft), torch.abs(frame_fft))
            per_sample = per_sample + float(self.config.amplitude_weight) * amp_distance
            total_weight += float(self.config.amplitude_weight)

        if self.config.use_phase and self.config.phase_weight > 0.0:
            phase_distance = self._wasserstein_1d(torch.angle(cond_fft), torch.angle(frame_fft))
            per_sample = per_sample + float(self.config.phase_weight) * phase_distance
            total_weight += float(self.config.phase_weight)

        if total_weight > 0.0:
            per_sample = per_sample / total_weight
        return per_sample

    def _sample_frame_indices(self, num_frames: int, device: torch.device) -> torch.Tensor:
        if num_frames <= 1:
            return torch.empty(0, dtype=torch.long, device=device)
        candidates = torch.arange(1, num_frames, device=device)
        if candidates.numel() <= self.config.num_sampled_frames:
            return candidates
        if self.config.random_frame_sampling:
            perm = torch.randperm(candidates.numel(), device=device)
            return candidates.index_select(0, perm[: self.config.num_sampled_frames])
        return candidates[: self.config.num_sampled_frames]

    def compute(
        self,
        *,
        pred_frames: torch.Tensor,
        conditioning_frame: torch.Tensor,
        step: int,
    ) -> Optional[torch.Tensor]:
        multiplier = self._step_multiplier(int(step))
        if multiplier <= 0.0:
            return None

        if pred_frames.dim() != 5:
            raise ValueError(
                f"VCD expects pred_frames shape (B, T, C, H, W), got {pred_frames.shape}"
            )

        num_frames = int(pred_frames.shape[1])
        sampled_indices = self._sample_frame_indices(num_frames, pred_frames.device)
        if sampled_indices.numel() == 0:
            return None

        self._ensure_device(pred_frames.device)

        cond = conditioning_frame
        if cond.dim() != 4:
            raise ValueError(
                "VCD conditioning_frame must have shape (B, C, H, W), got "
                f"{conditioning_frame.shape}"
            )

        if self.config.detach_conditioning_frame:
            cond = cond.detach()

        cond = cond.to(pred_frames.device)
        cond = self._resize_if_needed(self._ensure_rgb(self._normalize_frames(cond)))

        cond_features: List[torch.Tensor]
        if self.config.detach_conditioning_frame:
            with torch.no_grad():
                cond_features = self._extract_selected_features(cond)
        else:
            cond_features = self._extract_selected_features(cond)

        frame_losses: list[torch.Tensor] = []
        for frame_idx in sampled_indices.tolist():
            frame = pred_frames[:, frame_idx, :, :, :]
            frame = self._resize_if_needed(self._ensure_rgb(self._normalize_frames(frame)))
            frame_features = self._extract_selected_features(frame)

            layer_losses = []
            for cond_feat, frame_feat in zip(cond_features, frame_features):
                layer_losses.append(self._per_layer_vcd(cond_feat, frame_feat))
            if not layer_losses:
                continue

            per_sample_loss = torch.stack(layer_losses, dim=0).mean(dim=0)
            if self.config.use_temporal_weight:
                temporal_weight = (num_frames - frame_idx) / float(num_frames)
                per_sample_loss = per_sample_loss * temporal_weight
            frame_losses.append(per_sample_loss.mean())

        if not frame_losses:
            return None

        total = torch.stack(frame_losses).mean()
        return total * float(multiplier)
