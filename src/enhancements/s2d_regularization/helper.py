from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from common.logger import get_logger


logger = get_logger(__name__)


@dataclass
class _S2DCacheEntry:
    u: Tensor
    v: Tensor
    sigma: Tensor
    k_target: int
    step_cached: int
    pcdr_kmax: float
    base_component: Optional[Tensor] = None


class S2DRegularizer:
    """Selective Spectral Decay regularizer adapted for LoRA training.

    Paper-faithful parts retained in this helper:
    - PCDR-based selective targeting of dominant singular components.
    - Amortized SVD refresh every `m` steps.
    - Dominant-spectrum weighting controlled by `n` and `lambda`.

    Due to LoRA-only training constraints, this implementation applies the
    regularization to LoRA update factors while using layer input activations
    (captured via hooks) for PCDR-based layer/component selection.
    """

    _VALID_SELECTION_MODES = {
        "pcdr_activation",
        "sigma_mass",
        "pcdr_dense_full_svd",
    }
    _VALID_PCDR_VARIANTS = {"outlier_neuron", "projection_mass"}

    def __init__(self, args: Any) -> None:
        self.enabled = bool(getattr(args, "enable_s2d", False))
        self.decay_lambda = float(getattr(args, "s2d_decay_lambda", 5e-4))
        self.power_n = float(getattr(args, "s2d_power_n", 2.0))
        self.pcdr_tau = float(getattr(args, "s2d_pcdr_tau", 0.95))
        self.k_max = int(getattr(args, "s2d_k_max", 3))
        self.update_interval = int(getattr(args, "s2d_update_interval", 100))
        self.max_modules = int(getattr(args, "s2d_max_modules", 0))
        self.include_text_encoder = bool(
            getattr(args, "s2d_include_text_encoder", False)
        )
        self.use_abs_response = bool(getattr(args, "s2d_use_abs_response", True))

        selection_mode = str(
            getattr(args, "s2d_selection_mode", "pcdr_activation")
        ).strip().lower()
        if selection_mode not in self._VALID_SELECTION_MODES:
            selection_mode = "pcdr_activation"
        self.selection_mode = selection_mode

        pcdr_variant = str(
            getattr(args, "s2d_pcdr_variant", "outlier_neuron")
        ).strip().lower()
        if pcdr_variant not in self._VALID_PCDR_VARIANTS:
            pcdr_variant = "outlier_neuron"
        self.pcdr_variant = pcdr_variant

        self.activation_samples = int(getattr(args, "s2d_activation_samples", 512))
        self.hook_batch_samples = int(getattr(args, "s2d_hook_batch_samples", 64))
        self.selection_fallback_to_kmax = bool(
            getattr(args, "s2d_selection_fallback_to_kmax", False)
        )

        self.eps = 1e-8

        self._last_refresh_step: Optional[int] = None
        self._cache: Dict[str, _S2DCacheEntry] = {}
        self._last_metrics: Dict[str, float] = {}

        self._hook_handles: List[RemovableHandle] = []
        self._hooked_network_id: Optional[int] = None
        self._activation_buffers: Dict[str, Tensor] = {}

    def _selection_uses_activation(self) -> bool:
        return self.selection_mode in {"pcdr_activation", "pcdr_dense_full_svd"}

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []
        self._hooked_network_id = None
        self._activation_buffers = {}

    def _iter_lora_modules(
        self,
        network: Any,
    ) -> Iterable[Tuple[str, Any, Tensor, Tensor]]:
        candidates = list(getattr(network, "unet_loras", []) or [])
        if self.include_text_encoder:
            candidates.extend(list(getattr(network, "text_encoder_loras", []) or []))

        yielded = 0
        for lora_module in candidates:
            lora_name = str(getattr(lora_module, "lora_name", ""))
            lora_up = getattr(lora_module, "lora_up", None)
            lora_down = getattr(lora_module, "lora_down", None)
            if lora_up is None or lora_down is None:
                continue
            if isinstance(lora_up, nn.ModuleList) or isinstance(lora_down, nn.ModuleList):
                continue
            if not hasattr(lora_up, "weight") or not hasattr(lora_down, "weight"):
                continue

            up_weight = lora_up.weight
            down_weight = lora_down.weight
            if not torch.is_tensor(up_weight) or not torch.is_tensor(down_weight):
                continue
            if up_weight.ndim != 2 or down_weight.ndim != 2:
                continue
            if up_weight.shape[1] != down_weight.shape[0]:
                continue
            if up_weight.shape[1] <= 0:
                continue

            yield lora_name, lora_module, up_weight, down_weight
            yielded += 1
            if self.max_modules > 0 and yielded >= self.max_modules:
                return

    def _flatten_activations(
        self,
        x: Tensor,
        expected_in_features: int,
    ) -> Optional[Tensor]:
        if not torch.is_tensor(x):
            return None
        if x.numel() == 0:
            return None

        if x.ndim >= 2 and int(x.shape[-1]) == int(expected_in_features):
            flat = x.reshape(-1, int(expected_in_features))
            return flat

        if x.ndim == 2 and int(x.shape[1]) == int(expected_in_features):
            return x

        # Some modules may pass data in (B, C, ...)-style layout.
        if x.ndim >= 3 and int(x.shape[1]) == int(expected_in_features):
            permute_order = list(range(x.ndim))
            permute_order = permute_order[0:1] + permute_order[2:] + [1]
            moved = x.permute(*permute_order).contiguous()
            return moved.reshape(-1, int(expected_in_features))

        return None

    def _append_activation_samples(self, lora_name: str, flat_x: Tensor) -> None:
        if flat_x.numel() == 0:
            return

        sample_cap = max(1, self.activation_samples)
        per_hook_cap = max(1, min(self.hook_batch_samples, sample_cap))

        num_rows = int(flat_x.shape[0])
        if num_rows > per_hook_cap:
            idx = torch.randperm(num_rows, device=flat_x.device)[:per_hook_cap]
            sample = flat_x.index_select(0, idx)
        else:
            sample = flat_x

        sample = sample.detach().to(dtype=torch.float32, device="cpu")

        prev = self._activation_buffers.get(lora_name)
        if prev is None:
            self._activation_buffers[lora_name] = sample[:sample_cap]
            return

        merged = torch.cat([prev, sample], dim=0)
        if merged.shape[0] > sample_cap:
            idx = torch.randperm(merged.shape[0])[:sample_cap]
            merged = merged.index_select(0, idx)
        self._activation_buffers[lora_name] = merged

    def _build_activation_hook(
        self,
        lora_name: str,
        expected_in_features: int,
    ):
        def _hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            if not self.enabled or len(inputs) == 0:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            try:
                flat_x = self._flatten_activations(x, expected_in_features)
                if flat_x is None:
                    return
                self._append_activation_samples(lora_name, flat_x)
            except Exception:
                return

        return _hook

    def _ensure_hooks(self, network: Any) -> None:
        network_id = id(network)
        if self._hooked_network_id == network_id and self._hook_handles:
            return

        self.remove_hooks()

        for lora_name, lora_module, _up_weight, _down_weight in self._iter_lora_modules(
            network
        ):
            lora_down = getattr(lora_module, "lora_down", None)
            in_features = getattr(lora_down, "in_features", None)
            if in_features is None:
                continue
            try:
                in_features_int = int(in_features)
            except Exception:
                continue
            if in_features_int <= 0:
                continue

            try:
                handle = lora_module.register_forward_pre_hook(
                    self._build_activation_hook(lora_name, in_features_int)
                )
            except Exception:
                continue
            self._hook_handles.append(handle)

        self._hooked_network_id = network_id

    def _get_org_module(self, lora_module: Any) -> Optional[nn.Module]:
        org_module_ref = getattr(lora_module, "org_module_ref", None)
        if isinstance(org_module_ref, list) and org_module_ref:
            candidate = org_module_ref[0]
            if isinstance(candidate, nn.Module):
                return candidate

        org_forward = getattr(lora_module, "org_forward", None)
        bound_self = getattr(org_forward, "__self__", None)
        if isinstance(bound_self, nn.Module):
            return bound_self

        return None

    def _get_lora_scale_factor(self, lora_module: Any) -> float:
        multiplier = getattr(lora_module, "multiplier", 1.0)
        scale = getattr(lora_module, "scale", 1.0)
        try:
            multiplier_f = float(multiplier)
        except Exception:
            multiplier_f = 1.0
        try:
            if torch.is_tensor(scale):
                scale_f = float(scale.detach().float().item())
            else:
                scale_f = float(scale)
        except Exception:
            scale_f = 1.0
        return multiplier_f * scale_f

    @torch.no_grad()
    def _dense_effective_weight(
        self,
        lora_module: Any,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor, float]]:
        org_module = self._get_org_module(lora_module)
        if org_module is None or not hasattr(org_module, "weight"):
            return None

        base_weight = getattr(org_module, "weight", None)
        if not torch.is_tensor(base_weight):
            return None
        if base_weight.ndim != 2:
            return None

        base_fp32 = base_weight.detach().to(dtype=torch.float32)
        up_fp32 = up_weight.detach().to(dtype=torch.float32)
        down_fp32 = down_weight.detach().to(dtype=torch.float32)
        if up_fp32.shape[0] != base_fp32.shape[0] or down_fp32.shape[1] != base_fp32.shape[1]:
            return None

        scale_factor = self._get_lora_scale_factor(lora_module)
        delta = up_fp32 @ down_fp32
        if scale_factor != 1.0:
            delta = delta * scale_factor

        return base_fp32 + delta, base_fp32, scale_factor

    @torch.no_grad()
    def _full_svd_from_dense_weight(
        self,
        dense_weight: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w = dense_weight.detach().to(dtype=torch.float32)
        u, sigma, v_t = torch.linalg.svd(w, full_matrices=False)
        v = v_t.transpose(0, 1)
        return u, sigma, v

    @torch.no_grad()
    def _thin_svd_from_factors(
        self,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # W = up @ down where up:(out,r), down:(r,in). Non-zero singular
        # spectrum can be recovered efficiently via reduced QR + small core SVD.
        up = up_weight.detach().to(dtype=torch.float32)
        down = down_weight.detach().to(dtype=torch.float32)

        q_u, r_u = torch.linalg.qr(up, mode="reduced")
        q_v, r_v = torch.linalg.qr(down.transpose(0, 1), mode="reduced")
        core = r_u @ r_v.transpose(0, 1)
        u_core, sigma, v_core_t = torch.linalg.svd(core, full_matrices=False)

        u = q_u @ u_core
        v = q_v @ v_core_t.transpose(0, 1)
        return u, sigma, v

    @torch.no_grad()
    def _compute_pcdr_curve(
        self,
        activations_cpu: Tensor,
        u: Tensor,
        sigma: Tensor,
        v: Tensor,
    ) -> Optional[Tensor]:
        if activations_cpu.numel() == 0 or sigma.numel() == 0:
            return None

        acts = activations_cpu.to(device=v.device, dtype=torch.float32)
        if acts.shape[0] > self.activation_samples:
            idx = torch.randperm(acts.shape[0], device=acts.device)[: self.activation_samples]
            acts = acts.index_select(0, idx)

        proj = acts @ v
        weighted = proj * sigma.unsqueeze(0)
        abs_weighted = weighted.abs()

        if self.pcdr_variant == "outlier_neuron":
            # Use the top-|activation| neuron per sample, then measure component
            # dominance for that outlier pathway.
            outputs = weighted @ u.transpose(0, 1)
            outlier_idx = outputs.abs().argmax(dim=1)
            abs_u = u.abs()
            selected_u = abs_u.index_select(0, outlier_idx)
            contrib = abs_weighted * selected_u
        else:
            contrib = abs_weighted

        denom = contrib.sum(dim=1, keepdim=True)
        valid = denom.squeeze(1) > self.eps
        if not torch.any(valid):
            return None

        contrib = contrib[valid]
        denom = denom[valid]

        pcdr_curve = torch.cumsum(contrib, dim=1) / (denom + self.eps)
        return pcdr_curve.mean(dim=0)

    @torch.no_grad()
    def _select_k_target(
        self,
        *,
        sigma: Tensor,
        pcdr_curve: Optional[Tensor],
    ) -> Tuple[Optional[int], float]:
        if sigma.numel() == 0:
            return None, 0.0

        limit = max(1, min(self.k_max, int(sigma.numel())))

        if self.selection_mode == "sigma_mass" or pcdr_curve is None:
            sigma_abs = sigma.abs()
            total_sigma = float(sigma_abs.sum().item())
            if total_sigma <= self.eps:
                return None, 0.0

            top_sigma = sigma_abs[:limit]
            ratio = torch.cumsum(top_sigma, dim=0) / (total_sigma + self.eps)
            reached = torch.nonzero(ratio >= self.pcdr_tau, as_tuple=False)
            if reached.numel() == 0:
                if self.selection_fallback_to_kmax:
                    return limit, float(ratio[-1].item())
                return None, float(ratio[-1].item())
            return int(reached[0].item()) + 1, float(ratio[min(limit, ratio.numel()) - 1].item())

        pcdr_limited = pcdr_curve[:limit]
        if pcdr_limited.numel() == 0:
            return None, 0.0

        reached = torch.nonzero(pcdr_limited >= self.pcdr_tau, as_tuple=False)
        if reached.numel() == 0:
            if self.selection_fallback_to_kmax:
                return limit, float(pcdr_limited[-1].item())
            return None, float(pcdr_limited[-1].item())

        return int(reached[0].item()) + 1, float(pcdr_limited[-1].item())

    @torch.no_grad()
    def _refresh_cache(self, network: Any, global_step: int) -> None:
        new_cache: Dict[str, _S2DCacheEntry] = {}

        selected_layers = 0
        skipped_no_activation = 0
        skipped_missing_dense_base = 0
        skipped_dense_svd_failure = 0
        skipped_below_tau = 0
        pcdr_kmax_sum = 0.0
        dense_svd_layers = 0

        for lora_name, lora_module, up_weight, down_weight in self._iter_lora_modules(
            network
        ):
            base_component: Optional[Tensor] = None
            scale_factor = self._get_lora_scale_factor(lora_module)

            if self.selection_mode == "pcdr_dense_full_svd":
                dense_tuple = self._dense_effective_weight(
                    lora_module=lora_module,
                    up_weight=up_weight,
                    down_weight=down_weight,
                )
                if dense_tuple is None:
                    skipped_missing_dense_base += 1
                    continue

                dense_weight, base_weight, _ = dense_tuple
                try:
                    u, sigma, v = self._full_svd_from_dense_weight(dense_weight)
                except Exception:
                    skipped_dense_svd_failure += 1
                    continue

                dense_svd_layers += 1

                limit = max(1, min(self.k_max, int(sigma.numel())))
                u_l = u[:, :limit]
                v_l = v[:, :limit]
                base_component = torch.sum(
                    (base_weight @ v_l) * u_l,
                    dim=0,
                ).detach().cpu()
            else:
                try:
                    u, sigma, v = self._thin_svd_from_factors(up_weight, down_weight)
                except Exception:
                    continue

            if sigma.numel() == 0:
                continue

            pcdr_curve: Optional[Tensor] = None
            if self._selection_uses_activation():
                activations = self._activation_buffers.get(lora_name)
                if activations is None or activations.numel() == 0:
                    skipped_no_activation += 1
                    continue
                pcdr_curve = self._compute_pcdr_curve(activations, u, sigma, v)
                if pcdr_curve is None:
                    skipped_no_activation += 1
                    continue

            k_target, pcdr_kmax = self._select_k_target(sigma=sigma, pcdr_curve=pcdr_curve)
            if k_target is None:
                skipped_below_tau += 1
                continue

            selected_layers += 1
            pcdr_kmax_sum += pcdr_kmax

            sigma_abs = sigma.abs()
            new_cache[lora_name] = _S2DCacheEntry(
                u=u[:, :k_target].detach().cpu(),
                v=v[:, :k_target].detach().cpu(),
                sigma=sigma_abs[:k_target].detach().cpu(),
                k_target=k_target,
                step_cached=int(global_step),
                pcdr_kmax=float(pcdr_kmax),
                base_component=(
                    base_component[:k_target].detach().cpu()
                    if base_component is not None
                    else None
                ),
            )

        self._cache = new_cache
        if (
            self._last_refresh_step is None
            and self._selection_uses_activation()
            and selected_layers <= 0
            and skipped_no_activation > 0
        ):
            # Warmup case: hooks were just attached and activations are not yet
            # available. Keep refresh pending for the next step.
            self._last_refresh_step = None
        else:
            self._last_refresh_step = int(global_step)
        # Use a fresh activation window for the next amortized interval.
        self._activation_buffers = {}

        self._last_metrics.update(
            {
                "s2d/selected_layers": float(selected_layers),
                "s2d/skipped_no_activation": float(skipped_no_activation),
                "s2d/skipped_missing_dense_base": float(skipped_missing_dense_base),
                "s2d/skipped_dense_svd_failure": float(skipped_dense_svd_failure),
                "s2d/skipped_below_tau": float(skipped_below_tau),
                "s2d/dense_svd_layers": float(dense_svd_layers),
                "s2d/mean_pcdr_kmax": (
                    pcdr_kmax_sum / max(1, selected_layers)
                ),
                "s2d/cached_layers": float(len(self._cache)),
            }
        )

    def _should_refresh(self, global_step: int) -> bool:
        if self._last_refresh_step is None:
            return True
        return (int(global_step) - int(self._last_refresh_step)) >= int(
            self.update_interval
        )

    def compute_loss(
        self,
        *,
        network: Any,
        global_step: Optional[int],
    ) -> Optional[Tensor]:
        if not self.enabled or network is None:
            return None

        if self._selection_uses_activation():
            self._ensure_hooks(network)
        elif self._hook_handles:
            self.remove_hooks()

        step = int(global_step) if global_step is not None else 0
        if self._should_refresh(step):
            self._refresh_cache(network, step)

        total_loss: Optional[Tensor] = None
        active_layers = 0
        mean_component_norm = 0.0

        for lora_name, lora_module, up_weight, down_weight in self._iter_lora_modules(network):
            cache_entry = self._cache.get(lora_name, None)
            if cache_entry is None:
                continue

            device = up_weight.device
            u = cache_entry.u.to(device=device, dtype=torch.float32)
            v = cache_entry.v.to(device=device, dtype=torch.float32)
            sigma = cache_entry.sigma.to(device=device, dtype=torch.float32)

            up = up_weight.to(dtype=torch.float32)
            down = down_weight.to(dtype=torch.float32)

            # component_response_j = u_j^T (up @ down) v_j.
            left = up.transpose(0, 1) @ u
            right = down @ v
            component_response = torch.sum(left * right, dim=0)
            scale_factor = self._get_lora_scale_factor(lora_module)
            if scale_factor != 1.0:
                component_response = component_response * float(scale_factor)
            if cache_entry.base_component is not None:
                component_response = component_response + cache_entry.base_component.to(
                    device=device, dtype=torch.float32
                )
            if self.use_abs_response:
                component_response = component_response.abs()

            mean_component_norm += float(component_response.mean().detach().item())

            coeff = torch.pow(torch.clamp(sigma, min=self.eps), self.power_n)
            layer_loss = torch.sum(coeff * component_response)

            total_loss = layer_loss if total_loss is None else total_loss + layer_loss
            active_layers += 1

        self._last_metrics.update(
            {
                "s2d/active_layers": float(active_layers),
                "s2d/hooked_layers": float(len(self._hook_handles)),
                "s2d/buffered_layers": float(len(self._activation_buffers)),
                "s2d/mean_component_response": (
                    mean_component_norm / max(1, active_layers)
                ),
                "s2d/last_refresh_step": (
                    float(self._last_refresh_step)
                    if self._last_refresh_step is not None
                    else -1.0
                ),
            }
        )

        if total_loss is None or active_layers <= 0:
            return None
        return self.decay_lambda * (total_loss / float(active_layers))
