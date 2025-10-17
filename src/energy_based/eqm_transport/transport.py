"""Transport loss utilities adapted from the EqM release."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from . import path
from .integrators import ODEIntegrator, SDEIntegrator
from .path import GVPCPlan, ICPlan, VPCPlan, expand_t_like_x
from .utils import EasyDict, mean_flat


class ModelType(enum.Enum):
    """Type of quantity predicted by the model."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    """Interpolation path to use between data and noise."""

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Weighting strategy for the regression target."""

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


@dataclass
class ODELikelihoodResult:
    logp: torch.Tensor
    prior_logp: torch.Tensor
    delta_logp: List[torch.Tensor]
    final_state: torch.Tensor


class Transport:
    """Core transport loss helper."""

    def __init__(
        self,
        *,
        model_type: ModelType,
        path_type: PathType,
        loss_type: WeightType,
        train_eps: float,
        sample_eps: float,
    ) -> None:
        path_options = {
            PathType.LINEAR: ICPlan,
            PathType.GVP: GVPCPlan,
            PathType.VP: VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    @staticmethod
    def prior_logp(z: torch.Tensor) -> torch.Tensor:
        """Log density of the standard normal prior."""
        flat = z.view(z.size(0), -1)
        dim = flat.size(1)
        return -0.5 * (dim * math.log(2 * math.pi) + (flat**2).sum(dim=1))

    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        eval: bool = False,
        last_step_size: float = 0.0,
    ) -> tuple[float, float]:
        """Match EqM's timestep interval handling."""
        t0 = 0.0
        t1 = 1.0
        eps = train_eps if not eval else sample_eps

        if isinstance(self.path_sampler, VPCPlan):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif isinstance(self.path_sampler, (ICPlan, GVPCPlan)) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random timestep and noise latent."""
        device = x1.device
        dtype = x1.dtype
        x0 = torch.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = torch.rand((x1.size(0),), device=device, dtype=dtype) * (t1 - t0) + t0
        return t, x0, x1

    @staticmethod
    def disp_loss(z: torch.Tensor) -> torch.Tensor:
        """Official InfoNCE-style dispersive loss on activations."""
        if z is None or z.size(0) <= 1:
            return torch.zeros((), device=z.device if z is not None else "cpu")

        flat = z.reshape(z.size(0), -1)
        feature_dim = max(flat.size(1), 1)
        diff = torch.nn.functional.pdist(flat).pow(2) / feature_dim
        zeros = torch.zeros(flat.size(0), device=flat.device, dtype=flat.dtype)
        diff = torch.cat((diff, diff, zeros), dim=0)
        return torch.log(torch.exp(-diff).mean() + 1e-12)

    @staticmethod
    def get_ct(t: torch.Tensor) -> torch.Tensor:
        """Energy-compatible target scaling from EqM."""
        interp = 0.8
        start = 1.0
        ct = torch.minimum(
            start - (start - 1) / interp * t,
            1 / (1 - interp) - 1 / (1 - interp) * t,
        )
        return ct * 4.0

    def _compute_weight_map(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return an optional weighting tensor for transport losses."""
        if self.loss_type not in {WeightType.VELOCITY, WeightType.LIKELIHOOD}:
            return None

        try:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))
        except Exception:
            return None

        sigma_t = sigma_t.clamp_min(1e-8)
        drift_var = drift_var.to(dtype=sigma_t.dtype)

        if self.loss_type == WeightType.VELOCITY:
            weight = (drift_var / sigma_t) ** 2
        else:
            weight = drift_var / (sigma_t ** 2)

        return weight.to(dtype=xt.dtype)

    def training_losses(
        self,
        model,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the EqM transport loss dictionary."""
        model_kwargs = dict(model_kwargs or {})

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        ct = self.get_ct(t).view(-1, *([1] * (ut.ndim - 1)))
        ut = ut * ct

        model_output = model(xt, t, **model_kwargs)
        dispersive_term = torch.zeros((), device=xt.device, dtype=xt.dtype)

        if model_kwargs.get("return_act"):
            model_output, activations = model_output
            if activations:
                dispersive_term = self.disp_loss(activations[-1])

        batch_size = xt.size(0)
        assert model_output.shape == (batch_size, *xt.shape[1:]), (
            "Model output must match xt dimensions"
        )

        weight_map = self._compute_weight_map(xt, t)

        if self.model_type == ModelType.VELOCITY:
            error = (model_output - ut) ** 2
            if weight_map is not None:
                loss = mean_flat(weight_map * error)
            else:
                loss = mean_flat(error)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))
            sigma_t = sigma_t.clamp_min(1e-8)
            if self.loss_type == WeightType.VELOCITY:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type == WeightType.LIKELIHOOD:
                weight = drift_var / (sigma_t**2)
            else:
                weight = torch.tensor(1.0, device=xt.device, dtype=xt.dtype)

            if isinstance(weight, torch.Tensor):
                weight = weight.to(dtype=xt.dtype)

            if self.model_type == ModelType.NOISE:
                diff = (model_output - x0) ** 2
            else:
                diff = ((model_output * sigma_t) + x0) ** 2

            loss = mean_flat(weight * diff)

        loss = loss + 0.5 * dispersive_term
        return {"pred": model_output, "loss": loss}

    def get_drift(self):
        """Return the drift function for the probability flow ODE."""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            output = drift_fn(x, t, model, **model_kwargs)
            if output.shape != x.shape:
                raise ValueError("ODE drift must preserve the state shape")
            return output

        return body_fn

    def get_score(self):
        """Return the score function for reconciling model outputs."""
        if self.model_type == ModelType.NOISE:
            return lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(
                path.expand_t_like_x(t, x)
            )[0]
        if self.model_type == ModelType.SCORE:
            return lambda x, t, model, **kwargs: model(x, t, **kwargs)
        if self.model_type == ModelType.VELOCITY:
            return lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        raise NotImplementedError()


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: Optional[str] = None,
    train_eps: Optional[float] = None,
    sample_eps: Optional[float] = None,
) -> Transport:
    """Factory mirroring the EqM CLI defaults."""
    path_map = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    if path_type not in path_map:
        raise ValueError(f"Unknown path type: {path_type}")

    prediction = prediction.lower()
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        weight_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        weight_type = WeightType.LIKELIHOOD
    else:
        weight_type = WeightType.NONE

    path_choice = path_map[path_type]

    if path_choice == PathType.VP:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_choice in {PathType.GVP, PathType.LINEAR} and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0.0 if train_eps is None else train_eps
        sample_eps = 0.0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=path_choice,
        loss_type=weight_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )


class Sampler:
    """Sampler class wrapping SDE/ODE integrators for EqM transport objects."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _build_sde_terms(
        self,
        *,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
    ):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        def sde_drift(x, t, model, **kwargs):
            return self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        return sde_drift, diffusion_fn

    def _final_step(
        self,
        sde_drift,
        *,
        last_step: Optional[str],
        last_step_size: float,
    ):
        if last_step is None:
            return lambda x, t, model, **model_kwargs: x

        last_step = last_step.lower()
        if last_step == "mean":
            return lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size

        if last_step == "tweedie":
            alpha_fn = self.transport.path_sampler.compute_alpha_t
            sigma_fn = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **model_kwargs: x / alpha_fn(t)[0][0] + (
                sigma_fn(t)[0][0] ** 2
            ) / alpha_fn(t)[0][0] * self.score(x, t, model, **model_kwargs)

        if last_step == "euler":
            return lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size

        raise ValueError(f"Unknown SDE last step '{last_step}'.")

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        num_steps: int = 250,
    ):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self._build_sde_terms(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        integrator = SDEIntegrator(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )
        final_step = self._final_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init: torch.Tensor, model: Callable[..., torch.Tensor], **model_kwargs):
            xs = integrator.sample(init, model, **model_kwargs)
            if not xs:
                xs = [init]
            t_final = torch.ones(init.size(0), device=init.device, dtype=init.dtype) * t1
            x_final = final_step(xs[-1], t_final, model, **model_kwargs)
            xs.append(x_final)
            if len(xs) != num_steps:
                raise RuntimeError("SDE sampler produced unexpected number of steps")
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
    ):
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(
                x, torch.ones_like(t) * (1 - t), model, **kwargs
            )
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        integrator = ODEIntegrator(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return integrator.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        trace_samples: int = 1,
    ):
        if trace_samples < 1:
            raise ValueError("trace_samples must be >= 1 for likelihood estimation.")

        def _likelihood_drift(
            state: tuple[torch.Tensor, torch.Tensor],
            t: torch.Tensor,
            model: Callable[..., torch.Tensor],
            **model_kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x_state, _ = state
            reverse_t = torch.ones_like(t) * (1 - t)
            divergence = torch.zeros(
                x_state.size(0), device=x_state.device, dtype=x_state.dtype
            )
            with torch.enable_grad():
                x_grad = x_state.detach().clone().requires_grad_(True)
                velocity = self.drift(x_grad, reverse_t, model, **model_kwargs)
                for trace_index in range(trace_samples):
                    noise = (
                        torch.randint(
                            low=0,
                            high=2,
                            size=x_grad.shape,
                            device=x_grad.device,
                            dtype=torch.int32,
                        ).to(x_grad.dtype)
                        * 2
                        - 1
                    )
                    hvp = torch.autograd.grad(
                        (velocity * noise).sum(),
                        x_grad,
                        retain_graph=trace_index < trace_samples - 1,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    divergence = divergence + (hvp * noise).reshape(hvp.size(0), -1).sum(dim=1)
            divergence = divergence / trace_samples
            return (-velocity.detach(), divergence.detach())

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        integrator = ODEIntegrator(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            requires_grad=True,
        )

        def _sample(
            init: torch.Tensor,
            model: Callable[..., torch.Tensor],
            **model_kwargs,
        ) -> EasyDict:
            initial_state = (
                init,
                torch.zeros(init.size(0), device=init.device, dtype=init.dtype),
            )
            trajectory = integrator.sample(initial_state, model, **model_kwargs)
            if isinstance(trajectory, tuple):
                xs_series, log_series = trajectory
                xs = [xs_series[i].detach() for i in range(xs_series.size(0))]
                delta = [log_series[i].detach() for i in range(log_series.size(0))]
            else:
                xs = [item[0].detach() for item in trajectory]
                delta = [item[1].detach() for item in trajectory]
            final_state = xs[-1]
            cumulative = delta[-1]
            prior_logp = self.transport.prior_logp(final_state)
            logp = prior_logp - cumulative
            return EasyDict(
                {
                    "states": xs,
                    "delta_logp": delta,
                    "final_state": final_state,
                    "prior_logp": prior_logp,
                    "logp": logp,
                }
            )

        return _sample
