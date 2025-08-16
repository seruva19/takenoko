## Based on https://github.com/TheDenk/wan2.1-dilated-controlnet/blob/main/train/train_controlnet.py (Apache)

"""Modular ControlNet for WAN, integrated with Takenoko's network system.

This implementation provides a lightweight ControlNet that is compatible with
Takenoko's training pipeline. It is intentionally self-contained (no external
Diffusers dependency) and focuses on producing per-layer control states that
are injected into the main `WanModel` at configurable intervals.

Design notes:
- The network encodes the control hint (e.g., edges) via a small 3D CNN stack
  to downscale and aggregate spatiotemporal information.
- The control hint features are concatenated with the current noisy latents and
  patch-embedded into token space with the same inner dimension as the WAN
  transformer (num_heads * head_dim).
- A set of zero-initialized linear layers (one per injected layer) maps token
  embeddings to per-layer residual states that are added to the main model.

The result is an effective, zero-disturbance-at-init control signal path that
learns to steer generation during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.logger import get_logger

logger = get_logger(__name__)


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


@dataclass
class ControlNetWanConfig:
    """Configuration for ControlNetWanNetwork.

    Attributes:
        patch_size: 3D patch size used by the main WAN model.
        num_attention_heads: Number of attention heads in WAN model.
        attention_head_dim: Dimension per attention head in WAN model.
        vae_channels: Number of channels in WAN latents (in_dim of WanModel).
        control_in_channels: Number of channels in the control hint (e.g., 1 or 3).
        num_layers: Number of per-layer control outputs to produce.
        downscale_coef: Spatial downscale factor for the control encoder.
        out_proj_dim: Dimension of per-token hidden state in WAN (heads * head_dim).
    """

    patch_size: Tuple[int, int, int]
    num_attention_heads: int
    attention_head_dim: int
    vae_channels: int
    control_in_channels: int = 3
    num_layers: int = 20
    downscale_coef: int = 8
    out_proj_dim: Optional[int] = None

    def resolve(self) -> None:
        if self.out_proj_dim is None:
            self.out_proj_dim = self.num_attention_heads * self.attention_head_dim


class ControlNetWanNetwork(nn.Module):
    """Lightweight ControlNet producing per-layer control states for WAN.

    Forward signature mirrors the reference shape expectations:
    - hidden_states: (B, vae_channels, F, H, W)
    - controlnet_states: (B, C_control, F, H, W)
    - encoder_hidden_states: (B, L_text, C_text) [currently unused, reserved]
    - timestep: (B,) int64 [currently unused, reserved]

    Returns a tuple(list) of per-layer token tensors:
      ( (B, L_tokens, out_proj_dim), ... ) with `num_layers` elements.
    """

    def __init__(self, cfg: ControlNetWanConfig) -> None:
        super().__init__()
        cfg.resolve()
        self.cfg = cfg

        start_channels = cfg.control_in_channels * (cfg.downscale_coef**2)
        channels = [start_channels, start_channels // 2, start_channels // 4]

        self.control_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        cfg.control_in_channels,
                        channels[0],
                        kernel_size=(3, cfg.downscale_coef + 1, cfg.downscale_coef + 1),
                        stride=(1, cfg.downscale_coef, cfg.downscale_coef),
                        padding=(1, cfg.downscale_coef // 2, cfg.downscale_coef // 2),
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, channels[0]),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        channels[0],
                        channels[1],
                        kernel_size=3,
                        stride=(2, 1, 1),
                        padding=1,
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, channels[1]),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        channels[1],
                        channels[2],
                        kernel_size=3,
                        stride=(2, 1, 1),
                        padding=1,
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, channels[2]),
                ),
            ]
        )

        inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.patch_embedding = nn.Conv3d(
            cfg.vae_channels + channels[2],
            inner_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

        # One zero-initialized projection per injection layer
        self.controlnet_blocks = nn.ModuleList(
            [_zero_module(nn.Linear(inner_dim, cfg.out_proj_dim)) for _ in range(cfg.num_layers)]  # type: ignore[arg-type]
        )

        self.gradient_checkpointing: bool = False

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        controlnet_states: torch.Tensor,
        return_dict: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, ...]] | Tuple[torch.Tensor, ...]:
        # Encode control video
        x_control = controlnet_states
        for enc in self.control_encoder:
            x_control = enc(x_control)

        # Concatenate with latents and patch-embed to tokens
        x = torch.cat([hidden_states, x_control], dim=1)
        x = self.patch_embedding(x)  # (B, inner_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, L_tokens, inner_dim)

        # Produce per-layer residuals
        outputs: List[torch.Tensor] = []
        for proj in self.controlnet_blocks:
            outputs.append(proj(x))  # (B, L_tokens, out_proj_dim)

        result: Tuple[torch.Tensor, ...] = tuple(outputs)
        if return_dict:
            # Keep parity with diffusers-like API shape without strict typing
            return (result,)
        return (result,)

    # Takenoko network API compatibility ------------------------------------
    def prepare_optimizer_params(
        self, unet_lr: float, **_: Any
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Return optimizer param group for this ControlNet.

        Args:
            unet_lr: Base learning rate for WAN model; reused here.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            return [], []
        return ([{"params": trainable, "lr": float(unet_lr)}], ["controlnet"])


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: Optional[nn.Module],
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    **kwargs: Any,
) -> ControlNetWanNetwork:
    """Factory for ControlNetWanNetwork matching Takenoko's create_network API.

    Unused args are kept for signature compatibility with other network modules.
    ControlNet-specific settings are expected in kwargs, e.g.:
      - controlnet_transformer_num_layers: int
      - downscale_coef: int
      - control_in_channels: int
    """
    # Extract WAN backbone properties
    num_heads: int = int(getattr(unet, "num_heads", 16))
    dim: int = int(getattr(unet, "dim", num_heads * 128))
    head_dim: int = dim // num_heads
    vae_channels: int = int(getattr(unet, "in_dim", 16))
    patch_size: Tuple[int, int, int] = tuple(getattr(unet, "patch_size", (1, 2, 2)))  # type: ignore[assignment]

    # ControlNet-specific params
    num_layers = int(kwargs.get("controlnet_transformer_num_layers", 20))
    downscale_coef = int(kwargs.get("downscale_coef", 8))
    control_in_channels = int(kwargs.get("control_in_channels", 3))
    out_proj_dim = int(kwargs.get("out_proj_dim", dim))

    cfg = ControlNetWanConfig(
        patch_size=patch_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        vae_channels=vae_channels,
        control_in_channels=control_in_channels,
        num_layers=num_layers,
        downscale_coef=downscale_coef,
        out_proj_dim=out_proj_dim,
    )

    net = ControlNetWanNetwork(cfg)
    logger.info(
        "Created ControlNetWanNetwork: heads=%d, head_dim=%d, layers=%d, vae_ch=%d, out_proj_dim=%d",
        num_heads,
        head_dim,
        num_layers,
        vae_channels,
        cfg.out_proj_dim or -1,
    )
    return net


# Optional alias for clarity when imported from model manager
def create_controlnet(*args: Any, **kwargs: Any) -> ControlNetWanNetwork:
    return create_network(*args, **kwargs)
