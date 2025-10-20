"""Forward-pass helpers shared by the rCM distillation pipeline.

These utilities mirror the scaling and denoising formulae used in NVIDIA's
reference implementation while remaining lightweight enough to plug into the
existing Takenoko WAN trainer.  They provide rectified-flow projections as well
as JVP-powered tangents used by the distillation loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.autograd import functional as autograd_functional


@dataclass(slots=True)
class RCMDenoisePrediction:
    """Container capturing rCM-style predictions for a single forward pass.

    Attributes:
        trigflow_t: The trigflow time parameter (radians) used for scaling.
        c_skip/c_out/c_in/c_noise: Scaling coefficients applied to the DiT inputs.
        x0: Reconstructed clean sample.
        f: Flow-field prediction (a.k.a. rectified-flow velocity).
    """

    trigflow_t: torch.Tensor
    c_skip: torch.Tensor
    c_out: torch.Tensor
    c_in: torch.Tensor
    c_noise: torch.Tensor
    x0: torch.Tensor
    f: torch.Tensor


@dataclass(slots=True)
class RCMDenoiseResult:
    """Aggregates student/teacher predictions returned by ``rcm_compute_forward``."""

    prediction: RCMDenoisePrediction
    raw_model_output: torch.Tensor


def rcm_scaling(
    trigflow_t: torch.Tensor,
    *,
    t_scaling_factor: float = 1000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rectified-flow scaling wrapper used across the rCM codebase.

    The formula matches ``RectifiedFlow_TrigFlowWrapper`` from NVIDIA's reference
    implementation.  We keep operations in float64 for numerical stability before
    casting back to the caller's dtype.
    """

    dtype = trigflow_t.dtype
    device = trigflow_t.device
    trigflow_t64 = trigflow_t.to(torch.float64)
    denom = torch.cos(trigflow_t64) + torch.sin(trigflow_t64)
    # Avoid division-by-zero for tiny angles by clamping denominators.
    denom = torch.clamp(denom, min=1e-6)

    c_skip = 1.0 / denom
    c_out = -torch.sin(trigflow_t64) / denom
    c_in = 1.0 / denom
    c_noise = (torch.sin(trigflow_t64) / denom) * t_scaling_factor

    return (
        c_skip.to(dtype=dtype, device=device),
        c_out.to(dtype=dtype, device=device),
        c_in.to(dtype=dtype, device=device),
        c_noise.to(dtype=dtype, device=device),
    )


def rcm_compute_forward(
    *,
    xt: torch.Tensor,
    model_output: torch.Tensor,
    trigflow_t: torch.Tensor,
    t_scaling_factor: float = 1000.0,
) -> RCMDenoiseResult:
    """Compute rCM-style predictions from a DiT forward pass.

    Args:
        xt: The noisy sample fed to the network (already on-device/dtype).
        model_output: Raw DiT output tensor (same shape as ``xt``).
        trigflow_t: Trigflow time parameter expressed in radians.
        t_scaling_factor: Optional scaling factor for ``c_noise`` (defaults to 1000).

    Returns:
        ``RCMDenoiseResult`` containing scaling coefficients, the reconstructed
        clean sample, and the flow-field prediction.
    """

    if xt.shape != model_output.shape:
        raise ValueError(
            f"rcm_compute_forward expects xt and model_output to share the same shape, "
            f"got {xt.shape} vs {model_output.shape}"
        )

    (
        c_skip,
        c_out,
        c_in,  # unused for now but returned for completeness
        c_noise,
    ) = rcm_scaling(trigflow_t, t_scaling_factor=t_scaling_factor)

    # Broadcasting helpers expect time tensors with singleton spatial dims.
    # The runner provides shapes (B, 1, T, 1, 1); if not, unsqueeze accordingly.
    while trigflow_t.dim() < xt.dim():
        trigflow_t = trigflow_t.unsqueeze(-1)
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
        c_in = c_in.unsqueeze(-1)
        c_noise = c_noise.unsqueeze(-1)

    x0_pred = c_skip * xt + c_out * model_output

    # Avoid division by zero when sin(t) ~ 0 (early timesteps) by clamping.
    sin_t = torch.sin(trigflow_t).clamp(min=1e-6)
    cos_t = torch.cos(trigflow_t)
    f_pred = (cos_t * xt - x0_pred) / sin_t

    prediction = RCMDenoisePrediction(
        trigflow_t=trigflow_t,
        c_skip=c_skip,
        c_out=c_out,
        c_in=c_in,
        c_noise=c_noise,
        x0=x0_pred,
        f=f_pred,
    )

    return RCMDenoiseResult(prediction=prediction, raw_model_output=model_output)


def estimate_trigflow_from_timesteps(
    timesteps: torch.Tensor,
    *,
    base_num_train_timesteps: Optional[int] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Heuristic converter mapping discrete WAN timesteps to trigflow angles.

    This provides a temporary bridge until Phase 2 introduces the true trigflow
    sampler.  We normalise the integer timestep to `[0, 1]` using the scheduler's
    configured train steps (defaulting to 1000) and then scale by ``pi/2``.
    """

    if base_num_train_timesteps is None or base_num_train_timesteps <= 0:
        base_num_train_timesteps = 1000

    timesteps = timesteps.to(torch.float32)
    normalised = torch.clamp(timesteps / float(base_num_train_timesteps), min=0.0, max=1.0 - eps)
    return normalised * (math.pi / 2.0 - eps)


def student_f_with_t(
    model_fn: Callable[..., torch.Tensor],
    *,
    xt: torch.Tensor,
    xt_tangent: torch.Tensor,
    trigflow_t: torch.Tensor,
    trigflow_t_tangent: torch.Tensor,
    t_scaling_factor: float = 1000.0,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``F`` and its directional derivative via a JVP over the student model.

    Args:
        model_fn: Callable that receives ``xt`` and ``trigflow_t`` (plus optional
            keyword arguments) and returns the raw model output tensor.
        xt: Noisy student input ``x_t``.
        xt_tangent: Directional derivative of ``x_t`` (``t_xt`` in the paper).
        trigflow_t: Trigflow angle (radians) used for the current forward pass.
        trigflow_t_tangent: Directional derivative of the trigflow angle.
        t_scaling_factor: Optional scaling factor for the ``rcm_compute_forward`` helper.
        model_kwargs: Additional keyword arguments to forward into ``model_fn``.

    Returns:
        Tuple of the rectified-flow prediction ``F`` and its directional derivative
        w.r.t. the supplied tangents.
    """

    if model_kwargs is None:
        model_kwargs = {}

    if xt.shape != xt_tangent.shape:
        raise ValueError("xt_tangent must match xt shape for JVP computation.")
    if trigflow_t.shape != trigflow_t_tangent.shape:
        raise ValueError("trigflow_t_tangent must match trigflow_t shape for JVP computation.")

    def _forward(inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        xt_in, trigflow_in = inputs
        model_output = model_fn(xt_in, trigflow_in, **model_kwargs)
        result = rcm_compute_forward(
            xt=xt_in,
            model_output=model_output,
            trigflow_t=trigflow_in,
            t_scaling_factor=t_scaling_factor,
        )
        return result.prediction.f

    f_value, f_tangent = autograd_functional.jvp(
        _forward,
        (xt, trigflow_t),
        (xt_tangent, trigflow_t_tangent),
        create_graph=False,
        strict=True,
    )

    return f_value, f_tangent.detach()


def placeholder_student_f_with_t(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    raise RuntimeError(
        "placeholder_student_f_with_t has been removed; call 'student_f_with_t' directly with explicit arguments."
    )
