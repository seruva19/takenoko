from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

import torch

try:
    from torchdiffeq import odeint  # type: ignore
except ImportError:
    odeint = None


TensorLike = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class SDEIntegrator:
    """Minimal SDE solver supporting Euler-Maruyama and Heun schemes."""

    def __init__(
        self,
        drift: Callable[[torch.Tensor, torch.Tensor, Callable[..., torch.Tensor], Sequence], torch.Tensor],
        diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        t0: float,
        t1: float,
        num_steps: int,
        sampler_type: str = "Euler",
    ) -> None:
        if t0 >= t1:
            raise ValueError("SDE integrator expects t0 < t1")
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")

        self.drift = drift
        self.diffusion = diffusion
        self.timesteps = torch.linspace(t0, t1, num_steps)
        self.dt = self.timesteps[1] - self.timesteps[0]
        sampler_type = sampler_type.lower()
        if sampler_type not in {"euler", "heun"}:
            raise ValueError(f"Unsupported SDE sampler '{sampler_type}' (use 'Euler' or 'Heun').")
        self.sampler_type = sampler_type

    def _euler_step(
        self,
        x: torch.Tensor,
        mean_x: torch.Tensor,
        t: float,
        model: Callable[..., torch.Tensor],
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        t_batch = torch.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        dw = noise * torch.sqrt(self.dt)
        drift_val = self.drift(x, t_batch, model, **model_kwargs)
        diffusion_val = self.diffusion(x, t_batch)
        mean_x = x + drift_val * self.dt
        x_next = mean_x + torch.sqrt(2 * diffusion_val) * dw
        return x_next, mean_x

    def _heun_step(
        self,
        x: torch.Tensor,
        _: torch.Tensor,
        t: float,
        model: Callable[..., torch.Tensor],
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        dw = noise * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        diffusion = self.diffusion(x, t_cur)

        x_hat = x + torch.sqrt(2 * diffusion) * dw
        k1 = self.drift(x_hat, t_cur, model, **model_kwargs)
        x_pred = x_hat + self.dt * k1
        k2 = self.drift(x_pred, t_cur + self.dt, model, **model_kwargs)
        return x_hat + 0.5 * self.dt * (k1 + k2), x_hat

    def sample(
        self,
        init: torch.Tensor,
        model: Callable[..., torch.Tensor],
        **model_kwargs,
    ) -> List[torch.Tensor]:
        """Integrate forward and return the trajectory (excluding the initial state)."""
        x = init
        mean_x = init
        samples: List[torch.Tensor] = []
        step_fn = self._heun_step if self.sampler_type == "heun" else self._euler_step

        for t_i in self.timesteps[:-1]:
            with torch.no_grad():
                x, mean_x = step_fn(x, mean_x, float(t_i), model, **model_kwargs)
                samples.append(x)

        return samples


class ODEIntegrator:
    """Deterministic ODE solver with Euler/Heun or Dopri5 (if torchdiffeq is available)."""

    def __init__(
        self,
        drift: Callable[[TensorLike, torch.Tensor, Callable[..., TensorLike], Sequence], TensorLike],
        *,
        t0: float,
        t1: float,
        sampler_type: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        requires_grad: bool = False,
    ) -> None:
        if t0 >= t1:
            raise ValueError("ODE integrator expects t0 < t1")
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")

        self.drift = drift
        self.t0 = t0
        self.t1 = t1
        self.num_steps = num_steps
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type.lower()
        self.timesteps = torch.linspace(t0, t1, num_steps)
        self.requires_grad = requires_grad

    @staticmethod
    def _primary(state: TensorLike) -> torch.Tensor:
        return state[0] if isinstance(state, tuple) else state

    @staticmethod
    def _clone_state(state: TensorLike) -> TensorLike:
        if isinstance(state, tuple):
            return tuple(component.detach().clone() for component in state)  # type: ignore[return-value]
        return state.detach().clone()

    @staticmethod
    def _add_state(state: TensorLike, delta: TensorLike) -> TensorLike:
        if isinstance(state, tuple):
            assert isinstance(delta, tuple) and len(delta) == len(state)
            return tuple(
                (component + delta_component).detach()
                for component, delta_component in zip(state, delta)
            )  # type: ignore[return-value]
        if isinstance(delta, tuple):
            raise TypeError("State/delta type mismatch in ODEIntegrator.")
        return (state + delta).detach()

    @staticmethod
    def _scale_state(state: TensorLike, factor: float) -> TensorLike:
        if isinstance(state, tuple):
            return tuple(component * factor for component in state)  # type: ignore[return-value]
        return state * factor

    @staticmethod
    def _combine_states(a: TensorLike, b: TensorLike, *, average: bool = False) -> TensorLike:
        weight = 0.5 if average else 1.0
        if isinstance(a, tuple):
            assert isinstance(b, tuple) and len(a) == len(b)
            return tuple(
                weight * (a_i + b_i) for a_i, b_i in zip(a, b)
            )  # type: ignore[return-value]
        if isinstance(b, tuple):
            raise TypeError("State type mismatch in ODEIntegrator.")
        return weight * (a + b)

    def _t_batch(self, state: TensorLike, t_value: float) -> torch.Tensor:
        primary = self._primary(state)
        return torch.ones(
            primary.size(0), device=primary.device, dtype=primary.dtype
        ) * t_value

    def _numerical_loop(
        self,
        init: TensorLike,
        model: Callable[..., TensorLike],
        **model_kwargs,
    ) -> List[TensorLike]:
        samples: List[TensorLike] = []
        state = self._clone_state(init)
        step_ctx = torch.enable_grad() if self.requires_grad else torch.no_grad()

        for step_index, t_i in enumerate(self.timesteps[:-1]):
            dt = float(self.timesteps[step_index + 1] - t_i)
            with step_ctx:
                t_batch = self._t_batch(state, float(t_i))
                drift_val = self.drift(state, t_batch, model, **model_kwargs)

                if self.sampler_type == "heun":
                    state_euler = self._add_state(state, self._scale_state(drift_val, dt))
                    t_next = float(self.timesteps[step_index + 1])
                    drift_next = self.drift(
                        state_euler,
                        self._t_batch(state_euler, t_next),
                        model,
                        **model_kwargs,
                    )
                    increment = self._scale_state(
                        self._combine_states(drift_val, drift_next, average=True),
                        dt,
                    )
                else:
                    increment = self._scale_state(drift_val, dt)

            state = self._add_state(state, increment)
            samples.append(self._clone_state(state))

        samples.append(self._clone_state(state))
        return samples

    def sample(
        self,
        init: TensorLike,
        model: Callable[..., TensorLike],
        **model_kwargs,
    ) -> Sequence[TensorLike]:
        if self.sampler_type in {"euler", "heun"} or odeint is None:
            if isinstance(init, tuple):
                return self._numerical_loop(init, model, **model_kwargs)
            return self._numerical_loop(init, model, **model_kwargs)

        device = init[0].device if isinstance(init, tuple) else init.device

        def _fn(t: torch.Tensor, state: TensorLike) -> TensorLike:
            if isinstance(state, tuple):
                batch = torch.ones(state[0].size(0), device=device, dtype=state[0].dtype) * t
            else:
                batch = torch.ones(state.size(0), device=device, dtype=state.dtype) * t
            return self.drift(state, batch, model, **model_kwargs)

        t = self.timesteps.to(device)
        atol = [self.atol] * len(init) if isinstance(init, tuple) else self.atol
        rtol = [self.rtol] * len(init) if isinstance(init, tuple) else self.rtol
        result = odeint(_fn, init, t, method=self.sampler_type, atol=atol, rtol=rtol)
        if isinstance(init, tuple):
            steps = result[0].shape[0]
            return [tuple(component[step].detach() for component in result) for step in range(steps)]
        return [result[step].detach() for step in range(result.shape[0])]
