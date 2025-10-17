"""Helper utilities for constructing EqM sampler override payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_sampler_overrides(
    *,
    sampler: str,
    step_size: Optional[float] = None,
    momentum: Optional[float] = None,
    ode_method: Optional[str] = None,
    ode_steps: Optional[int] = None,
    ode_atol: Optional[float] = None,
    ode_rtol: Optional[float] = None,
    ode_likelihood_atol: Optional[float] = None,
    ode_likelihood_rtol: Optional[float] = None,
    ode_likelihood_trace_samples: Optional[int] = None,
    sde_method: Optional[str] = None,
    sde_steps: Optional[int] = None,
    sde_last_step: Optional[str] = None,
    sde_last_step_size: Optional[float] = None,
    sde_diffusion_form: Optional[str] = None,
    sde_diffusion_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a dictionary of EqM sampler overrides for embedding into configs."""
    sampler = sampler.lower()
    if sampler not in {"gd", "ngd", "ode", "sde", "ode_likelihood"}:
        raise ValueError(f"Unsupported sampler '{sampler}'")

    overrides: Dict[str, Any] = {"eqm_sampler": sampler}

    if step_size is not None:
        overrides["eqm_step_size"] = step_size
    if momentum is not None:
        overrides["eqm_momentum"] = momentum

    if sampler in {"ode", "ode_likelihood"}:
        if ode_method is not None:
            overrides["eqm_ode_method"] = ode_method
        if ode_steps is not None:
            overrides["eqm_ode_steps"] = ode_steps
        if ode_atol is not None:
            overrides["eqm_ode_atol"] = ode_atol
        if ode_rtol is not None:
            overrides["eqm_ode_rtol"] = ode_rtol
        if sampler == "ode_likelihood":
            if ode_likelihood_atol is not None:
                overrides["eqm_ode_likelihood_atol"] = ode_likelihood_atol
            if ode_likelihood_rtol is not None:
                overrides["eqm_ode_likelihood_rtol"] = ode_likelihood_rtol
            if ode_likelihood_trace_samples is not None:
                overrides["eqm_ode_likelihood_trace_samples"] = ode_likelihood_trace_samples

    if sampler == "sde":
        if sde_method is not None:
            overrides["eqm_sde_method"] = sde_method
        if sde_steps is not None:
            overrides["eqm_sde_steps"] = sde_steps
        if sde_last_step is not None:
            overrides["eqm_sde_last_step"] = sde_last_step
        if sde_last_step_size is not None:
            overrides["eqm_sde_last_step_size"] = sde_last_step_size
        if sde_diffusion_form is not None:
            overrides["eqm_sde_diffusion_form"] = sde_diffusion_form
        if sde_diffusion_norm is not None:
            overrides["eqm_sde_diffusion_norm"] = sde_diffusion_norm

    return overrides


def format_overrides_as_toml(
    overrides: Dict[str, Any],
    *,
    wrap_sample_block: bool = False,
    comment: Optional[str] = "EqM overrides generated programmatically",
) -> str:
    """Render an overrides dictionary into TOML snippet form."""
    lines = []
    if wrap_sample_block:
        lines.append("[[sample_prompts]]")
    if comment:
        prefix = "  " if wrap_sample_block else ""
        lines.append(f"{prefix}# {comment}")
    for key, value in overrides.items():
        prefix = "  " if wrap_sample_block else ""
        if isinstance(value, str):
            lines.append(f'{prefix}{key} = "{value}"')
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
            lines.append(f"{prefix}{key} = {rendered}")
        else:
            lines.append(f"{prefix}{key} = {value}")
    return "\n".join(lines)


def format_overrides_as_json(overrides: Dict[str, Any]) -> str:
    """Render an overrides dictionary as a JSON string."""
    import json

    return json.dumps(overrides, indent=2)
