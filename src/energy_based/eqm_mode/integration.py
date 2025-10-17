from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from common.logger import get_logger
from .config import EqMModeConfig
from .energy import compute_energy_from_features
from ..eqm_transport.path import expand_t_like_x
from ..eqm_transport.path import expand_t_like_x
from ..eqm_transport.transport import create_transport, ModelType, Transport
from ..eqm_transport.utils import EasyDict, mean_flat

logger = get_logger(__name__)


@dataclass
class EqMModeContext:
    """Holds persistent state for EqM training mode."""

    config: EqMModeConfig
    transport: Transport
    warning_emitted: bool = False


@dataclass
class EqMStepResult:
    """Outputs produced for a single EqM training step."""

    loss_components: EasyDict
    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    noisy_latents: torch.Tensor
    intermediate: Optional[torch.Tensor]
    energy: Optional[torch.Tensor]


def _velocity_to_model_space(
    transport: Transport,
    velocity: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    model_type = transport.model_type
    if model_type == ModelType.VELOCITY:
        return velocity
    path_sampler = transport.path_sampler
    if model_type == ModelType.SCORE:
        result = path_sampler.get_score_from_velocity(velocity, xt, t)
        return _align_tensor_shape(result, velocity)
    if model_type == ModelType.NOISE:
        result = path_sampler.get_noise_from_velocity(velocity, xt, t)
        return _align_tensor_shape(result, velocity)
    raise ValueError(f"Unsupported EqM model type: {model_type}")


def _model_space_to_velocity(
    transport: Transport,
    prediction: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    model_type = transport.model_type
    if model_type == ModelType.VELOCITY:
        return prediction
    path_sampler = transport.path_sampler
    if model_type == ModelType.SCORE:
        result = path_sampler.get_velocity_from_score(prediction, xt, t)
        return _align_tensor_shape(result, xt)
    if model_type == ModelType.NOISE:
        result = path_sampler.get_velocity_from_noise(prediction, xt, t)
        return _align_tensor_shape(result, xt)
    raise ValueError(f"Unsupported EqM model type: {model_type}")


def _align_tensor_shape(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.shape == reference.shape:
        return tensor
    if tensor.dim() + 1 == reference.dim() and reference.shape[0] == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == reference.dim() + 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def setup_eqm_mode(args: Any) -> EqMModeContext:
    """Create the EqM transport context for the current run."""
    cfg = getattr(args, "eqm_mode_config", None)
    if cfg is None:
        cfg = EqMModeConfig.from_args(args)
    transport = create_transport(
        path_type=cfg.path_type,
        prediction=cfg.prediction,
        loss_weight=cfg.transport_weighting,
        train_eps=cfg.train_eps,
        sample_eps=cfg.sample_eps,
    )
    if cfg.energy_head or cfg.prediction.lower() == "energy":
        logger.info("EqM energy head active (mode=%s)", cfg.energy_mode)
    return EqMModeContext(config=cfg, transport=transport)


def _forward_model(
    *,
    transformer: Any,
    xt: torch.Tensor,
    t: torch.Tensor,
    context_tokens: list[torch.Tensor],
    accelerator: Accelerator,
    network_dtype: torch.dtype,
    return_act: bool,
    patch_size: tuple[int, int, int],
    eqm_config: EqMModeConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = accelerator.device
    use_energy_head = (
        eqm_config.energy_head or eqm_config.prediction.lower() == "energy"
    )

    if use_energy_head:
        xt_forward = xt.detach().clone().to(device=device, dtype=network_dtype)
        xt_forward.requires_grad_(True)
    else:
        xt_forward = xt.to(device=device, dtype=network_dtype)

    t = t.to(device=device, dtype=network_dtype)

    tokens = [tensor.to(device=device, dtype=network_dtype) for tensor in context_tokens]
    spatial_dims = xt_forward.shape[2:]
    if len(patch_size) >= len(spatial_dims):
        patch_dims = patch_size[-len(spatial_dims):]
    else:
        patch_dims = (patch_size + (1,) * len(spatial_dims))[-len(spatial_dims):]

    seq_len = 1
    for dim, patch_dim in zip(spatial_dims, patch_dims):
        divisor = max(int(patch_dim), 1)
        seq_len *= max(int(dim) // divisor, 1)

    with accelerator.autocast():
        outputs = transformer(
            xt_forward,
            t=t,
            context=tokens,
            clip_fea=None,
            seq_len=seq_len,
            y=None,
            dispersive_loss_target_block=None,
            return_intermediate=return_act,
        )

    intermediates = None
    if return_act and isinstance(outputs, tuple) and len(outputs) == 2:
        outputs, intermediates = outputs

    preds = torch.stack(outputs, dim=0)
    if preds.dim() > xt_forward.dim():
        preds = preds.mean(dim=0)
    energy_values: Optional[torch.Tensor] = None

    if use_energy_head:
        grad, energy_values = compute_energy_from_features(
            features=preds,
            latents=xt_forward,
            mode=eqm_config.energy_mode,
            create_graph=transformer.training,
        )
        preds = grad

    return preds, intermediates, energy_values




def predict_velocity(
    context: EqMModeContext,
    *,
    transformer: Any,
    latents: torch.Tensor,
    t: torch.Tensor,
    context_tokens: list[torch.Tensor],
    accelerator: Accelerator,
    network_dtype: torch.dtype,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    device = accelerator.device
    xt = latents.to(device=device, dtype=network_dtype)
    t_projected = t.to(device=device, dtype=network_dtype)
    preds, _, _ = _forward_model(
        transformer=transformer,
        xt=xt,
        t=t_projected,
        context_tokens=context_tokens,
        accelerator=accelerator,
        network_dtype=network_dtype,
        return_act=False,
        patch_size=patch_size,
        eqm_config=context.config,
    )
    velocity = _model_space_to_velocity(
        context.transport,
        preds,
        xt,
        t_projected,
    )
    return _align_tensor_shape(velocity, latents)
def compute_eqm_step(
    context: EqMModeContext,
    *,
    transformer: Any,
    latents: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
    accelerator: Accelerator,
    network_dtype: torch.dtype,
    patch_size: tuple[int, int, int],
) -> EqMStepResult:
    """Compute the transport loss and model outputs for EqM mode."""
    transport = context.transport
    x1 = latents.to(device=accelerator.device, dtype=network_dtype)
    t, x0, x1 = transport.sample(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    ct = transport.get_ct(t).view(-1, *([1] * (ut.ndim - 1)))
    ut = ut * ct

    if (
        not context.warning_emitted
        and (
            getattr(args, "enable_control_lora", False)
            or getattr(args, "enable_controlnet", False)
        )
    ):
        logger.warning(
            "EqM mode currently ignores ControlNet / Control LoRA signals."
        )
        context.warning_emitted = True

    context_inputs = batch.get("t5", [])
    if context_inputs is None:
        context_inputs = []
    if not isinstance(context_inputs, list):
        context_inputs = [context_inputs]

    return_act = bool(getattr(args, "enable_dispersive_loss", False))
    model_pred, intermediates, energy_vals = _forward_model(
        transformer=transformer,
        xt=xt,
        t=t,
        context_tokens=context_inputs,
        accelerator=accelerator,
        network_dtype=network_dtype,
        return_act=return_act,
        patch_size=patch_size,
        eqm_config=context.config,
    )

    target = _velocity_to_model_space(transport, ut, xt, t)
    target = _align_tensor_shape(target, model_pred)
    model_velocity = _model_space_to_velocity(transport, model_pred, xt, t)
    model_velocity = _align_tensor_shape(model_velocity, ut)

    dispersive_term = torch.zeros(
        (), device=model_pred.device, dtype=model_pred.dtype
    )
    if return_act and intermediates is not None:
        try:
            acts = intermediates
            if isinstance(acts, list) and acts:
                dispersive_term = transport.disp_loss(acts[-1])
            elif isinstance(acts, torch.Tensor):
                dispersive_term = transport.disp_loss(acts)
        except Exception as exc:
            logger.debug(f"EqM dispersive loss computation failed: {exc}")

    error_sq = (model_pred - target) ** 2
    per_sample_base = mean_flat(error_sq)
    base_loss = per_sample_base.mean()

    weight_map: Optional[torch.Tensor] = None
    weighting_mode = None
    if context.config.transport_weighting:
        weighting_mode = str(context.config.transport_weighting).lower()
        if weighting_mode in {"velocity", "likelihood"}:
            try:
                _, drift_var = transport.path_sampler.compute_drift(xt, t)
                sigma_t, _ = transport.path_sampler.compute_sigma_t(
                    expand_t_like_x(t, xt)
                )
                sigma_t = sigma_t.clamp_min(1e-8)
                if weighting_mode == "velocity":
                    weight_map = (drift_var / sigma_t) ** 2
                else:
                    weight_map = drift_var / (sigma_t ** 2)
                weight_map = weight_map.to(dtype=model_pred.dtype)
            except Exception as exc:
                logger.debug("EqM transport weighting failed: %s", exc)
                weight_map = None

    if weight_map is not None:
        weighted_error = weight_map * error_sq
        per_sample_weighted = mean_flat(weighted_error)
        loss_value = per_sample_weighted.mean()
    else:
        weighted_error = error_sq
        loss_value = base_loss

    loss = loss_value + 0.5 * dispersive_term
    loss = loss * float(context.config.loss_weight)

    loss_components = EasyDict(
        {
            "loss": loss,
            "total_loss": loss,
            "base_loss": base_loss.detach(),
        }
    )
    transport_stats = {
        "transport_t_mean": t.detach().mean(),
        "transport_ct_mean": ct.detach().mean(),
        "transport_target_norm": mean_flat(ut.detach() ** 2),
        "transport_pred_norm": mean_flat(model_velocity.detach() ** 2),
        "transport_model_target_norm": mean_flat(target.detach() ** 2),
        "transport_weight_scale": torch.tensor(
            float(context.config.loss_weight),
            device=loss.device,
            dtype=loss.dtype,
        ),
    }
    if weight_map is not None:
        transport_stats.update(
            transport_weight_mean=weight_map.detach().mean(),
            transport_weight_max=weight_map.detach().max(),
        )
    if dispersive_term is not None:
        loss_components.update(
            dispersive_loss=dispersive_term.detach()
            if dispersive_term.requires_grad
            else dispersive_term
        )
    if energy_vals is not None:
        loss_components.update(
            eqm_energy=energy_vals.detach()
        )
        transport_stats.update(
            eqm_energy_mean=energy_vals.detach().mean()
        )
    loss_components.update(transport_stats)

    return EqMStepResult(
        loss_components=loss_components,
        model_pred=model_pred,
        target=ut,
        timesteps=t,
        noise=x0,
        noisy_latents=xt,
        intermediate=intermediates,
        energy=energy_vals.detach() if energy_vals is not None else None,
    )
