"""Configuration for Differential Guidance enhancement.

Differential Guidance amplifies the difference between model predictions and
training targets to potentially accelerate convergence by causing the model
to "overshoot" the target instead of incrementally approaching it.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DifferentialGuidanceConfig:
    """Configuration for differential guidance target transformation."""

    # Main feature flag
    enable_differential_guidance: bool = False

    # Scale factor for difference amplification
    # new_target = model_pred + scale * (target - model_pred)
    differential_guidance_scale: float = 3.0

    # Validation bounds for scale
    differential_guidance_min_scale: float = 0.0
    differential_guidance_max_scale: float = 10.0

    # Step range for applying differential guidance
    differential_guidance_start_step: int = 0
    differential_guidance_end_step: Optional[int] = None

    @classmethod
    def from_args(cls, args) -> "DifferentialGuidanceConfig":
        """Create config from command line arguments."""
        config = cls(
            enable_differential_guidance=getattr(
                args, "enable_differential_guidance", False
            ),
            differential_guidance_scale=getattr(
                args, "differential_guidance_scale", 3.0
            ),
            differential_guidance_min_scale=getattr(
                args, "differential_guidance_min_scale", 0.0
            ),
            differential_guidance_max_scale=getattr(
                args, "differential_guidance_max_scale", 10.0
            ),
            differential_guidance_start_step=getattr(
                args, "differential_guidance_start_step", 0
            ),
            differential_guidance_end_step=getattr(
                args, "differential_guidance_end_step", None
            ),
        )

        # Validate scale is within bounds
        if config.enable_differential_guidance:
            if not (
                config.differential_guidance_min_scale
                <= config.differential_guidance_scale
                <= config.differential_guidance_max_scale
            ):
                raise ValueError(
                    f"differential_guidance_scale ({config.differential_guidance_scale}) "
                    f"must be between {config.differential_guidance_min_scale} and "
                    f"{config.differential_guidance_max_scale}"
                )

        return config

    def is_enabled_for_step(self, step: int) -> bool:
        """Check if differential guidance should be applied at this step."""
        if not self.enable_differential_guidance:
            return False

        if step < self.differential_guidance_start_step:
            return False

        if (
            self.differential_guidance_end_step is not None
            and step >= self.differential_guidance_end_step
        ):
            return False

        return True
