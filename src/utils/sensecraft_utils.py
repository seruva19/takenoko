"""Helpers for optional SenseCraft integrations."""

from __future__ import annotations

from typing import Any, Dict

import torch

VALID_SENSECRAFT_METRICS = ("psnr", "ssim", "ms_ssim", "lpips")
VALID_SENSECRAFT_LPIPS_NETS = ("alex", "vgg", "squeeze")


def validate_sensecraft_loss_config(
    raw_config: Any, *, config_name: str
) -> list[Dict[str, Any]]:
    """Validate TOML-friendly SenseCraft compound loss configuration."""

    if raw_config is None:
        return []
    if not isinstance(raw_config, list):
        raise ValueError(f"{config_name} must be a list of one-key inline tables")

    normalized: list[Dict[str, Any]] = []
    for index, item in enumerate(raw_config):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                f"{config_name}[{index}] must be a one-key table like {{ charbonnier = 1.0 }}"
            )

        loss_name, value = next(iter(item.items()))
        if not isinstance(loss_name, str) or not loss_name.strip():
            raise ValueError(f"{config_name}[{index}] has an invalid loss name")

        if isinstance(value, (int, float)):
            normalized.append({loss_name: float(value)})
            continue

        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], dict)
        ):
            normalized.append({loss_name: [float(value[0]), dict(value[1])]})
            continue

        raise ValueError(
            f"{config_name}[{index}] must map to a float or [float, {{...}}] pair"
        )

    return normalized


def validate_sensecraft_input_range(
    raw_range: Any,
    *,
    config_name: str,
    default: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Validate a two-value numeric range used by SenseCraft."""

    if raw_range is None:
        return default
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError(f"{config_name} must be a two-item array like [0.0, 1.0]")

    try:
        min_value = float(raw_range[0])
        max_value = float(raw_range[1])
    except (TypeError, ValueError):
        raise ValueError(f"{config_name} values must be numeric") from None

    if max_value <= min_value:
        raise ValueError(f"{config_name} must satisfy max > min")

    return (min_value, max_value)


def validate_sensecraft_metrics(
    raw_metrics: Any,
    *,
    config_name: str,
) -> list[str]:
    """Validate and normalize SenseCraft metric names."""

    if raw_metrics is None:
        return []
    if not isinstance(raw_metrics, list):
        raise ValueError(
            f"{config_name} must be a list containing any of {VALID_SENSECRAFT_METRICS}"
        )

    normalized: list[str] = []
    for index, metric in enumerate(raw_metrics):
        metric_name = str(metric).strip().lower()
        if metric_name not in VALID_SENSECRAFT_METRICS:
            raise ValueError(
                f"{config_name}[{index}] must be one of {VALID_SENSECRAFT_METRICS}"
            )
        if metric_name not in normalized:
            normalized.append(metric_name)

    return normalized


def create_sensecraft_loss(
    *,
    loss_config: list[Dict[str, Any]],
    input_range: tuple[float, float],
    mode: str,
) -> torch.nn.Module:
    """Instantiate SenseCraftLoss lazily with a clear optional-dependency error."""

    try:
        from sensecraft.loss import SenseCraftLoss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SenseCraft is not available. Install project dependencies to enable this integration."
        ) from exc

    return SenseCraftLoss(
        loss_config=loss_config,
        input_range=input_range,
        mode=mode,
    )


def get_sensecraft_metric_functions() -> Dict[str, Any]:
    """Import the functional SenseCraft metrics lazily."""

    try:
        from sensecraft.metrics import lpips, ms_ssim, psnr, ssim
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SenseCraft metrics are not available. Install project dependencies to enable them."
        ) from exc

    return {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
    }


def prepare_sensecraft_loss_tensors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt Takenoko BCHW/BCFHW tensors to the layout expected by SenseCraft."""

    if predicted.shape != target.shape:
        raise ValueError(
            f"SenseCraft inputs must share shape, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )

    if mode == "3d":
        if predicted.dim() != 5:
            raise ValueError(
                f"SenseCraft 3d mode expects a 5D tensor, got shape {tuple(predicted.shape)}"
            )
        predicted = predicted.permute(0, 2, 1, 3, 4).contiguous()
        target = target.permute(0, 2, 1, 3, 4).contiguous()
        return predicted, target

    if mode == "2d":
        if predicted.dim() == 5:
            batch, channels, frames, height, width = predicted.shape
            predicted = predicted.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
            target = target.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
        elif predicted.dim() != 4:
            raise ValueError(
                f"SenseCraft 2d mode expects a 4D or 5D tensor, got shape {tuple(predicted.shape)}"
            )
        return predicted.contiguous(), target.contiguous()

    raise ValueError(f"Unsupported SenseCraft mode '{mode}'")
