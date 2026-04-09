from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class FleXRegularizer:
    """Fourier-based train-time regularizer over unmerged LoRA tensors."""

    def __init__(self, args: Any) -> None:
        self.enabled = bool(getattr(args, "enable_flex", False))
        self.lambda_value = float(getattr(args, "flex_lambda", 0.02))
        self.frequency_threshold = float(
            getattr(args, "flex_frequency_threshold", 0.5)
        )
        self.phi_low = float(getattr(args, "flex_phi_low", 1.0))
        self.phi_high = float(getattr(args, "flex_phi_high", 0.1))
        self.include_text_encoder = bool(
            getattr(args, "flex_include_text_encoder", False)
        )
        self.target_tensors = str(getattr(args, "flex_target_tensors", "both")).lower()
        self.include_patterns = self._compile_patterns(
            getattr(args, "flex_include_patterns", None)
        )
        self.exclude_patterns = self._compile_patterns(
            getattr(args, "flex_exclude_patterns", None)
        )
        self._last_metrics: Dict[str, float] = {}

    def setup_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        return

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

    def _compile_patterns(
        self, patterns: Optional[List[str]]
    ) -> Optional[List[Pattern[str]]]:
        if not patterns:
            return None
        return [re.compile(pattern) for pattern in patterns]

    def _matches_any(
        self, patterns: Optional[List[Pattern[str]]], value: str
    ) -> bool:
        if not patterns:
            return False
        return any(pattern.search(value) for pattern in patterns)

    def _should_include_module(self, lora_name: str) -> bool:
        if self.exclude_patterns and self._matches_any(self.exclude_patterns, lora_name):
            return False
        if self.include_patterns:
            return self._matches_any(self.include_patterns, lora_name)
        return True

    def _iter_lora_modules(
        self, network: Any
    ) -> Iterable[Tuple[str, Optional[Tensor], Optional[Tensor]]]:
        candidates = list(getattr(network, "unet_loras", []) or [])
        if self.include_text_encoder:
            candidates.extend(list(getattr(network, "text_encoder_loras", []) or []))

        for lora_module in candidates:
            lora_name = str(getattr(lora_module, "lora_name", ""))
            if not lora_name or not self._should_include_module(lora_name):
                continue

            lora_up = getattr(lora_module, "lora_up", None)
            lora_down = getattr(lora_module, "lora_down", None)
            if isinstance(lora_up, nn.ModuleList) or isinstance(lora_down, nn.ModuleList):
                continue

            up_weight = getattr(lora_up, "weight", None)
            down_weight = getattr(lora_down, "weight", None)
            yield (
                lora_name,
                up_weight if torch.is_tensor(up_weight) else None,
                down_weight if torch.is_tensor(down_weight) else None,
            )

    def _iter_target_tensors(
        self, network: Any
    ) -> Iterable[Tuple[str, str, Tensor]]:
        for lora_name, up_weight, down_weight in self._iter_lora_modules(network):
            if self.target_tensors in {"both", "up"} and up_weight is not None:
                yield lora_name, "up", up_weight
            if self.target_tensors in {"both", "down"} and down_weight is not None:
                yield lora_name, "down", down_weight

    def _build_frequency_weights(
        self, num_bins: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if num_bins <= 0:
            return torch.zeros(0, device=device, dtype=dtype)
        freq_idx = torch.arange(num_bins, device=device, dtype=dtype)
        denom = max(float(num_bins) * self.frequency_threshold, 1.0)
        ramp = torch.clamp(freq_idx / denom, max=1.0)
        phi = self.phi_low + (self.phi_high - self.phi_low) * ramp
        return 1.0 - phi

    def _compute_tensor_penalty(self, weight: Tensor) -> Optional[Tuple[Tensor, float]]:
        if not torch.is_floating_point(weight) or weight.numel() < 2:
            return None

        flat = weight.reshape(-1).to(dtype=torch.float32)
        spectrum = torch.fft.rfft(flat)
        power = spectrum.real.square() + spectrum.imag.square()
        if power.numel() <= 0:
            return None

        rho = self._build_frequency_weights(
            num_bins=int(power.numel()),
            device=power.device,
            dtype=power.dtype,
        )
        # FleX defines a summed Fourier penalty over weighted frequency bins.
        penalty = torch.sum(rho * power)

        threshold_bin = min(
            int(power.numel()),
            max(0, int(power.numel() * self.frequency_threshold)),
        )
        total_energy = float(power.sum().detach().item())
        if total_energy <= 0.0 or threshold_bin >= int(power.numel()):
            high_freq_ratio = 0.0
        else:
            high_freq_ratio = float(
                (power[threshold_bin:].sum().detach().item()) / total_energy
            )

        return penalty, high_freq_ratio

    def compute_loss(self, *, network: Any, global_step: Optional[int]) -> Optional[Tensor]:
        if not self.enabled or network is None or self.lambda_value <= 0.0:
            return None

        penalties: List[Tensor] = []
        high_freq_ratios: List[float] = []
        modules = set()

        for lora_name, _tensor_role, weight in self._iter_target_tensors(network):
            if weight is None or not getattr(weight, "requires_grad", False):
                continue
            penalty_result = self._compute_tensor_penalty(weight)
            if penalty_result is None:
                continue
            penalty, high_freq_ratio = penalty_result
            penalties.append(penalty)
            high_freq_ratios.append(high_freq_ratio)
            modules.add(lora_name)

        if not penalties:
            self._last_metrics = {
                "flex/active_modules": float(len(modules)),
                "flex/active_tensors": 0.0,
                "flex/high_freq_energy_ratio": 0.0,
                "flex/unscaled_penalty": 0.0,
                "flex/global_step": float(global_step or 0),
            }
            return None

        base_penalty = torch.stack(penalties).sum()
        self._last_metrics = {
            "flex/active_modules": float(len(modules)),
            "flex/active_tensors": float(len(penalties)),
            "flex/high_freq_energy_ratio": (
                sum(high_freq_ratios) / float(len(high_freq_ratios))
                if high_freq_ratios
                else 0.0
            ),
            "flex/unscaled_penalty": float(base_penalty.detach().item()),
            "flex/global_step": float(global_step or 0),
        }
        return self.lambda_value * base_penalty
