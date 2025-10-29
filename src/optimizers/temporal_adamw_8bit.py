"""TemporalAdamW8bit optimizer wrapper.

This optimizer wraps bitsandbytes AdamW8bit and applies temporal gradient
preprocessing (EMA smoothing, optional cosine-consistency scaling, and
EMA-based noise adaptation) before delegating to the 8-bit optimizer.

Design goals:
- Keep Adam states (m, v) in 8-bit via bitsandbytes for memory efficiency.
- Maintain small, lazily-allocated feature buffers in param dtype (optionally fp16)
  for temporal features.
- Avoid modifying Adam's internal bias correction by not changing betas over time;
  instead approximate adaptive momentum via gradient scaling.

Usage example:
    optimizer_type = "TemporalAdamW8bit"
    optimizer_args = [
      "betas=(0.9, 0.999)",
      "eps=1e-8",
      "weight_decay=0.01",
      "temporal_smoothing=0.0",           # disabled by default (Adam already smooths)
      "adaptive_momentum=False",           # disabled by default
      "consistency_interval=2",
      "consistency_threshold=0.5",         # apply scaling only when cos_sim > threshold
      "consistency_scale=0.1",             # scale range [0.9, 1.1]
      "noise_adaptation=False",            # disabled by default
      "noise_ema_alpha=0.10",
      "warmup_steps=50",
    ]
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List
import torch
from torch.optim.optimizer import Optimizer


class TemporalAdamW8bit(Optimizer):
    """Gradient-preprocessing wrapper around bitsandbytes AdamW8bit.

    Parameters mirror AdamW and Temporal options. Any additional kwargs are
    forwarded to bitsandbytes AdamW8bit.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        temporal_smoothing: float = 0.0,
        adaptive_momentum: bool = False,
        consistency_interval: int = 1,
        noise_adaptation: bool = False,
        noise_ema_alpha: float = 0.1,
        warmup_steps: int = 50,
        consistency_threshold: float = 0.5,
        consistency_scale: float = 0.1,
        feature_dtype: Optional[torch.dtype] = None,
        **bnb_kwargs: Any,
    ) -> None:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as err:  # pragma: no cover - import-time failure
            raise ImportError(
                "bitsandbytes is required for TemporalAdamW8bit. Install bitsandbytes."
            ) from err

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= temporal_smoothing <= 1.0:
            raise ValueError(f"Invalid temporal_smoothing: {temporal_smoothing}")
        if consistency_interval <= 0:
            raise ValueError("consistency_interval must be >= 1")
        if not (0.0 < noise_ema_alpha <= 1.0):
            raise ValueError("noise_ema_alpha must be in (0, 1]")
        if not (0.0 <= consistency_threshold <= 1.0):
            raise ValueError("consistency_threshold must be in [0, 1]")
        if not (0.0 < consistency_scale <= 0.25):
            raise ValueError("consistency_scale must be in (0, 0.25]")

        # Store feature settings
        self.temporal_smoothing: float = float(temporal_smoothing)
        self.adaptive_momentum: bool = bool(adaptive_momentum)
        self.consistency_interval: int = int(consistency_interval)
        self.noise_adaptation: bool = bool(noise_adaptation)
        self.noise_ema_alpha: float = float(noise_ema_alpha)
        self.warmup_steps: int = int(warmup_steps)
        self.consistency_threshold: float = float(consistency_threshold)
        self.consistency_scale: float = float(consistency_scale)
        # Normalize feature_dtype to a real torch.dtype if provided as string
        if isinstance(feature_dtype, str):
            dtype_map = {
                "torch.float16": torch.float16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "torch.float32": torch.float32,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            feature_dtype = dtype_map.get(feature_dtype.strip(), None)
        self.feature_dtype: Optional[torch.dtype] = feature_dtype

        # Construct inner 8-bit AdamW
        self.inner = bnb.optim.AdamW8bit(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **bnb_kwargs,
        )

        # Make this object Optimizer-like
        # Note: We don't use torch.Optimizer's init to avoid duplicating param groups.
        self.param_groups = self.inner.param_groups  # type: ignore[attr-defined]
        self.defaults: Dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # Expose underlying optimizer for logging utilities
        # (codebase checks optimizer.optimizer to unwrap)
        self.optimizer = self.inner  # type: ignore[assignment]

        # Per-parameter temporal states (lazy allocation)
        self.state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        self.global_step: int = 0

    def _ordered_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for group in self.param_groups:  # type: ignore[operator]
            for p in group["params"]:
                params.append(p)
        return params

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        # Preprocess gradients in-place
        for group in self.param_groups:  # type: ignore[operator]
            eps = group.get("eps", 1e-8)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "TemporalAdamW8bit does not support sparse gradients"
                    )

                st = self.state.setdefault(p, {"step": 0})
                st["step"] = int(st.get("step", 0)) + 1
                step_i: int = st["step"]

                # Cast buffers to optional feature dtype for memory savings
                feat_dtype = self.feature_dtype or grad.dtype

                g_used = grad

                # Temporal smoothing (EMA of gradients)
                if self.temporal_smoothing > 0.0:
                    smoothed = st.get("smoothed_grad")
                    if smoothed is None:
                        smoothed = torch.zeros_like(p, dtype=feat_dtype)
                        st["smoothed_grad"] = smoothed
                    smoothed.mul_(self.temporal_smoothing).add_(
                        grad, alpha=1.0 - self.temporal_smoothing
                    )
                    g_used = smoothed.to(dtype=grad.dtype)

                # Adaptive momentum approximation via gradient scaling
                if self.adaptive_momentum and step_i % self.consistency_interval == 0:
                    prev = st.get("prev_grad")
                    if prev is None:
                        prev = torch.zeros_like(p, dtype=feat_dtype)
                        st["prev_grad"] = prev
                    # Cast prev to grad dtype for math to avoid dtype mismatch
                    prev_for_math = prev.to(dtype=grad.dtype)
                    denom = (grad.norm() * prev_for_math.norm()).clamp_min(1e-12)
                    if denom > 0:
                        cos_sim = (
                            torch.dot(grad.flatten(), prev_for_math.flatten()) / denom
                        )
                        # Apply only when similarity is strongly positive
                        if float(cos_sim.item()) > self.consistency_threshold:
                            amp = self.consistency_scale
                            scale = float(
                                torch.clamp(
                                    1.0 + amp * cos_sim, 1.0 - amp, 1.0 + amp
                                ).item()
                            )
                            g_used = (g_used * scale).to(dtype=grad.dtype)

                # Noise adaptation using EMA mean/var (downscale only in high-noise regions)
                if self.noise_adaptation:
                    gmean = st.get("grad_mean_ema")
                    gsqmean = st.get("grad_sq_mean_ema")
                    if gmean is None:
                        gmean = torch.zeros_like(p, dtype=feat_dtype)
                        st["grad_mean_ema"] = gmean
                    if gsqmean is None:
                        gsqmean = torch.zeros_like(p, dtype=feat_dtype)
                        st["grad_sq_mean_ema"] = gsqmean
                    alpha = self.noise_ema_alpha
                    gmean.mul_(1.0 - alpha).add_(grad.to(dtype=feat_dtype), alpha=alpha)
                    gsqmean.mul_(1.0 - alpha).addcmul_(
                        grad.to(dtype=feat_dtype),
                        grad.to(dtype=feat_dtype),
                        value=alpha,
                    )

                    if step_i > self.warmup_steps:
                        var = (gsqmean - gmean.pow(2)).clamp_min(0.0)
                        # Per-element inverse-SNR scaling: 1/sqrt(1 + var/signal)
                        denom = gmean.abs() + eps
                        inv_snr_scale = 1.0 / torch.sqrt(1.0 + (var / denom))
                        inv_snr_scale = inv_snr_scale.clamp(0.2, 1.0)
                        g_used = (g_used * inv_snr_scale.to(dtype=grad.dtype)).to(
                            dtype=grad.dtype
                        )

                # Write back processed gradient
                p.grad = g_used

                # Update prev_grad if used
                if self.adaptive_momentum:
                    prev = st.get("prev_grad")
                    if prev is None:
                        prev = torch.zeros_like(p, dtype=feat_dtype)
                        st["prev_grad"] = prev
                    prev.copy_(grad.to(dtype=feat_dtype))

        # Delegate to inner optimizer
        self.inner.step()
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:  # type: ignore[override]
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        # Persist temporal buffers by parameter order to enable robust restore
        ordered_params = self._ordered_params()
        temporal_state_by_index: List[Dict[str, Any]] = []
        for p in ordered_params:
            s = self.state.get(p, {})
            entry: Dict[str, Any] = {}
            if "step" in s:
                entry["step"] = int(s["step"])
            for key in (
                "smoothed_grad",
                "prev_grad",
                "grad_mean_ema",
                "grad_sq_mean_ema",
            ):
                if key in s and isinstance(s[key], torch.Tensor):
                    entry[key] = s[key]
            temporal_state_by_index.append(entry)

        return {
            "inner": self.inner.state_dict(),
            "temporal_state": {
                int(id(p)): s for p, s in self.state.items()
            },  # legacy, best-effort
            "temporal_state_by_index": temporal_state_by_index,
            "global_step": self.global_step,
            "temporal_cfg": {
                "temporal_smoothing": self.temporal_smoothing,
                "adaptive_momentum": self.adaptive_momentum,
                "consistency_interval": self.consistency_interval,
                "consistency_threshold": self.consistency_threshold,
                "consistency_scale": self.consistency_scale,
                "noise_adaptation": self.noise_adaptation,
                "noise_ema_alpha": self.noise_ema_alpha,
                "warmup_steps": self.warmup_steps,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        self.inner.load_state_dict(state_dict.get("inner", {}))
        # Best-effort restore of temporal state (non-strict, since params may differ)
        self.global_step = int(state_dict.get("global_step", 0))
        # Prefer order-based restoration if available
        saved_list = state_dict.get("temporal_state_by_index")
        if isinstance(saved_list, list):
            current_params = self._ordered_params()
            if len(saved_list) == len(current_params):
                for p, saved in zip(current_params, saved_list):
                    if not isinstance(saved, dict):
                        continue
                    st: Dict[str, Any] = self.state.setdefault(p, {"step": 0})
                    if "step" in saved:
                        st["step"] = int(saved["step"])
                    # Restore tensors lazily in feature dtype
                    feat_dtype = self.feature_dtype or p.dtype
                    for key in (
                        "smoothed_grad",
                        "prev_grad",
                        "grad_mean_ema",
                        "grad_sq_mean_ema",
                    ):
                        t = saved.get(key)
                        if isinstance(t, torch.Tensor):
                            buf = torch.zeros_like(p, dtype=feat_dtype)
                            buf.copy_(t.to(dtype=feat_dtype, device=p.device))
                            st[key] = buf
                return
        # Fallback: Temporal buffers will be lazily re-created.
        return
