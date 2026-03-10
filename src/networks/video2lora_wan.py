from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from networks.lora_wan import LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)


def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    table = torch.zeros(n_position, d_hid, dtype=torch.float32)
    position = torch.arange(0, n_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_hid, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, d_hid))
    )
    table[:, 0::2] = torch.sin(position * div_term)
    if d_hid > 1:
        table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table.unsqueeze(0)


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
        return default
    return bool(value)


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_video_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 5:
        return None
    if tensor.shape[1] <= 32:
        return tensor
    if tensor.shape[-1] <= 32:
        return tensor.permute(0, 4, 1, 2, 3).contiguous()
    if tensor.shape[2] <= 32:
        return tensor.permute(0, 2, 1, 3, 4).contiguous()
    return tensor


def _choose_attention_heads(weight_dim: int, requested_heads: int) -> int:
    heads = max(1, int(requested_heads))
    while heads > 1 and weight_dim % heads != 0:
        heads -= 1
    return heads


class Video2LoRAModule(nn.Module):
    """LightLoRA adapter whose per-sample rank weights are predicted at runtime."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        video2lora_aux_down_dim: int = 128,
        video2lora_aux_up_dim: int = 64,
        **_: Any,
    ) -> None:
        super().__init__()
        in_dim = getattr(org_module, "in_features", None)
        out_dim = getattr(org_module, "out_features", None)
        if in_dim is None or out_dim is None:
            raise RuntimeError(
                f"Video2LoRA only supports linear-like modules, got {type(org_module).__name__}"
            )

        self.lora_name = lora_name
        self.lora_dim = int(lora_dim)
        self.multiplier = float(multiplier)
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.enabled = True
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.video2lora_aux_down_dim = int(video2lora_aux_down_dim)
        self.video2lora_aux_up_dim = int(video2lora_aux_up_dim)
        alpha_value = float(
            self.lora_dim if alpha is None or float(alpha) == 0.0 else alpha
        )
        self.scale = alpha_value / float(max(1, self.lora_dim))
        self.register_buffer("alpha", torch.tensor(alpha_value))

        self.down_aux = nn.Parameter(
            torch.empty(self.video2lora_aux_down_dim, self.in_dim)
        )
        self.up_aux = nn.Parameter(
            torch.empty(self.out_dim, self.video2lora_aux_up_dim)
        )
        nn.init.orthogonal_(self.down_aux)
        nn.init.orthogonal_(self.up_aux)

        self.org_module = org_module
        self._runtime_weight_embedding: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.down_aux.device

    @property
    def dtype(self) -> torch.dtype:
        return self.down_aux.dtype

    @property
    def weight_embedding_dim(self) -> int:
        return self.lora_dim * (
            self.video2lora_aux_down_dim + self.video2lora_aux_up_dim
        )

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def clear_runtime_weight_embedding(self) -> None:
        self._runtime_weight_embedding = None

    def set_runtime_weight_embedding(self, weight_embedding: Optional[torch.Tensor]) -> None:
        if weight_embedding is None:
            self._runtime_weight_embedding = None
            return
        expected = self.weight_embedding_dim
        if weight_embedding.dim() == 1 and int(weight_embedding.shape[0]) != expected:
            raise ValueError(
                f"{self.lora_name} expected runtime embedding dim {expected}, "
                f"got {tuple(weight_embedding.shape)}"
            )
        if weight_embedding.dim() == 2 and int(weight_embedding.shape[1]) != expected:
            raise ValueError(
                f"{self.lora_name} expected runtime embedding dim {expected}, "
                f"got {tuple(weight_embedding.shape)}"
            )
        if weight_embedding.dim() not in {1, 2}:
            raise ValueError(
                f"{self.lora_name} runtime embedding must be rank-1 or rank-2."
            )
        self._runtime_weight_embedding = weight_embedding

    def _materialize_runtime_weights(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        down_width = self.lora_dim * self.video2lora_aux_down_dim
        down_part, up_part = embedding.split(
            [down_width, self.lora_dim * self.video2lora_aux_up_dim],
            dim=-1,
        )

        if embedding.dim() == 1:
            down_part = down_part.reshape(self.lora_dim, self.video2lora_aux_down_dim)
            up_part = up_part.reshape(self.video2lora_aux_up_dim, self.lora_dim)
            down = down_part @ self.down_aux
            up = self.up_aux @ up_part
            return down, up

        down_part = down_part.reshape(
            embedding.shape[0], self.lora_dim, self.video2lora_aux_down_dim
        )
        up_part = up_part.reshape(
            embedding.shape[0], self.video2lora_aux_up_dim, self.lora_dim
        )
        down = torch.einsum("bra,ai->bri", down_part, self.down_aux)
        up = torch.einsum("oa,bar->bor", self.up_aux, up_part)
        return down, up

    def _apply_rank_dropout(self, hidden: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.rank_dropout is None or not self.training:
            return hidden, 1.0

        batch = int(hidden.shape[0]) if hidden.dim() > 0 else 1
        mask = torch.rand((batch, self.lora_dim), device=hidden.device) > float(
            self.rank_dropout
        )
        if hidden.dim() == 3:
            mask = mask.unsqueeze(1)
        elif hidden.dim() > 3:
            for _ in range(hidden.dim() - 2):
                mask = mask.unsqueeze(-2)
        hidden = hidden * mask.to(dtype=hidden.dtype)
        keep = max(1e-6, 1.0 - float(self.rank_dropout))
        return hidden, 1.0 / keep

    def _apply_dynamic_delta(
        self, x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
    ) -> torch.Tensor:
        x_work = x.to(dtype=down.dtype, device=down.device)
        if down.dim() == 2:
            hidden = F.linear(x_work, down)
            if self.dropout is not None and self.training:
                hidden = F.dropout(hidden, p=float(self.dropout))
            hidden, dropout_scale = self._apply_rank_dropout(hidden)
            delta = F.linear(hidden, up)
            return delta * dropout_scale

        batch = int(down.shape[0])
        if x_work.dim() == 2:
            if batch != 1:
                raise ValueError(
                    f"{self.lora_name} received batched weights without a batch axis in the input."
                )
            hidden = F.linear(x_work, down[0])
            if self.dropout is not None and self.training:
                hidden = F.dropout(hidden, p=float(self.dropout))
            hidden, dropout_scale = self._apply_rank_dropout(hidden.unsqueeze(0))
            delta = F.linear(hidden.squeeze(0), up[0])
            return delta * dropout_scale

        if int(x_work.shape[0]) != batch:
            if batch == 1:
                down = down.expand(int(x_work.shape[0]), -1, -1)
                up = up.expand(int(x_work.shape[0]), -1, -1)
                batch = int(x_work.shape[0])
            else:
                raise ValueError(
                    f"{self.lora_name} batch mismatch between input {tuple(x_work.shape)} "
                    f"and runtime weights {tuple(down.shape)}"
                )

        flat = x_work.reshape(batch, -1, self.in_dim)
        hidden = torch.einsum("bri,bti->btr", down, flat)
        if self.dropout is not None and self.training:
            hidden = F.dropout(hidden, p=float(self.dropout))
        hidden, dropout_scale = self._apply_rank_dropout(hidden)
        delta = torch.einsum("bor,btr->bto", up, hidden)
        delta = delta.reshape(*x_work.shape[:-1], self.out_dim)
        return delta * dropout_scale

    def merge_to(
        self,
        sd: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        non_blocking: bool = False,
    ) -> None:
        del sd, dtype, device, non_blocking
        raise RuntimeError(
            f"{self.lora_name} is runtime-conditioned and cannot be merged statically."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        org_forwarded = self.org_forward(x)
        if not self.enabled or self._runtime_weight_embedding is None:
            return org_forwarded

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < float(self.module_dropout):
                return org_forwarded

        down, up = self._materialize_runtime_weights(self._runtime_weight_embedding)
        delta = self._apply_dynamic_delta(x, down, up)
        delta = delta.to(dtype=org_forwarded.dtype, device=org_forwarded.device)
        return org_forwarded + delta * float(self.multiplier) * float(self.scale)


class Video2LoRAHyperNet(nn.Module):
    def __init__(
        self,
        weight_dim: int,
        weight_num: int,
        reference_feature_dim: int = 16,
        feature_dim: int = 256,
        decoder_blocks: int = 4,
        attention_heads: int = 4,
        sample_iters: int = 4,
        max_reference_frames: int = 0,
        spatial_pool_size: int = 0,
    ) -> None:
        super().__init__()
        self.weight_dim = int(weight_dim)
        self.weight_num = int(weight_num)
        self.reference_feature_dim = int(reference_feature_dim)
        self.feature_dim = int(feature_dim)
        self.sample_iters = int(sample_iters)
        self.max_reference_frames = int(max_reference_frames)
        self.spatial_pool_size = int(spatial_pool_size)

        heads = _choose_attention_heads(self.weight_dim, attention_heads)
        if heads != int(attention_heads):
            logger.info(
                "Video2LoRA adjusted attention heads from %s to %s to match weight_dim=%s.",
                attention_heads,
                heads,
                self.weight_dim,
            )

        self.register_buffer(
            "block_pos_emb",
            _get_sinusoid_encoding_table(self.weight_num * 2, self.weight_dim),
        )
        self.feature_proj = nn.Linear(
            self.reference_feature_dim,
            self.weight_dim,
            bias=False,
        )
        self.pos_emb_proj = nn.Linear(self.weight_dim, self.weight_dim, bias=False)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.weight_dim,
            nhead=heads,
            dim_feedforward=self.feature_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=max(1, int(decoder_blocks)),
        )
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(self.weight_dim),
            nn.Linear(self.weight_dim, self.weight_dim, bias=False),
        )
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.xavier_uniform_(self.pos_emb_proj.weight)
        nn.init.normal_(self.delta_proj[1].weight, std=1e-3)

    def _prepare_reference_tokens(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.dim() != 5:
            raise ValueError(
                f"Video2LoRA expected reference latents with shape (B, C, T, H, W), got {tuple(reference.shape)}"
            )
        if int(reference.shape[1]) != self.reference_feature_dim:
            raise ValueError(
                "Video2LoRA reference feature dim mismatch: "
                f"expected {self.reference_feature_dim}, got {int(reference.shape[1])}"
            )

        if self.max_reference_frames > 0 and int(reference.shape[2]) > self.max_reference_frames:
            reference = reference[:, :, : self.max_reference_frames]
        if self.spatial_pool_size > 0:
            reference = F.adaptive_avg_pool3d(
                reference,
                output_size=(
                    int(reference.shape[2]),
                    self.spatial_pool_size,
                    self.spatial_pool_size,
                ),
            )
        return reference.to(torch.float32).flatten(2).transpose(1, 2).contiguous()

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        memory = self._prepare_reference_tokens(reference)
        memory = self.feature_proj(memory)
        pos_emb = self.pos_emb_proj(
            self.block_pos_emb[:, : self.weight_num].to(
                device=memory.device,
                dtype=memory.dtype,
            )
        )
        weight_tokens = torch.zeros(
            reference.shape[0],
            self.weight_num,
            self.weight_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        for _ in range(max(1, self.sample_iters)):
            decoded = self.decoder(tgt=weight_tokens + pos_emb, memory=memory)
            weight_tokens = weight_tokens + self.delta_proj(decoded)
        return weight_tokens


class Video2LoRANetwork(LoRANetwork):
    def __init__(
        self,
        *args: Any,
        video2lora_aux_down_dim: int = 128,
        video2lora_aux_up_dim: int = 64,
        video2lora_reference_feature_dim: int = 16,
        video2lora_feature_dim: int = 256,
        video2lora_decoder_blocks: int = 4,
        video2lora_attention_heads: int = 4,
        video2lora_sample_iters: int = 4,
        video2lora_max_reference_frames: int = 0,
        video2lora_spatial_pool_size: int = 0,
        video2lora_runtime_source: str = "auto",
        video2lora_require_reference: bool = True,
        video2lora_reference_dropout_p: float = 0.0,
        video2lora_hypernet_lr_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.video2lora_aux_down_dim = int(video2lora_aux_down_dim)
        self.video2lora_aux_up_dim = int(video2lora_aux_up_dim)
        self.video2lora_reference_feature_dim = int(video2lora_reference_feature_dim)
        self.video2lora_feature_dim = int(video2lora_feature_dim)
        self.video2lora_decoder_blocks = int(video2lora_decoder_blocks)
        self.video2lora_attention_heads = int(video2lora_attention_heads)
        self.video2lora_sample_iters = int(video2lora_sample_iters)
        self.video2lora_max_reference_frames = int(video2lora_max_reference_frames)
        self.video2lora_spatial_pool_size = int(video2lora_spatial_pool_size)
        self.video2lora_runtime_source = str(video2lora_runtime_source).strip().lower()
        self.video2lora_require_reference = bool(video2lora_require_reference)
        self.video2lora_reference_dropout_p = float(video2lora_reference_dropout_p)
        self.video2lora_hypernet_lr_ratio = float(video2lora_hypernet_lr_ratio)

        def _module_factory(
            lora_name: str,
            org_module: nn.Module,
            multiplier: float,
            lora_dim: int,
            alpha: float,
            **module_kwargs: Any,
        ) -> Video2LoRAModule:
            return Video2LoRAModule(
                lora_name=lora_name,
                org_module=org_module,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                video2lora_aux_down_dim=self.video2lora_aux_down_dim,
                video2lora_aux_up_dim=self.video2lora_aux_up_dim,
                **module_kwargs,
            )

        super().__init__(*args, module_class=_module_factory, **kwargs)

        self._video2lora_modules: List[Video2LoRAModule] = [
            module
            for module in getattr(self, "unet_loras", [])
            if isinstance(module, Video2LoRAModule)
        ]
        if not self._video2lora_modules:
            raise ValueError("Video2LoRA network did not find any target modules.")

        weight_dim = self._video2lora_modules[0].weight_embedding_dim
        self.video2lora_hypernet = Video2LoRAHyperNet(
            weight_dim=weight_dim,
            weight_num=len(self._video2lora_modules),
            reference_feature_dim=self.video2lora_reference_feature_dim,
            feature_dim=self.video2lora_feature_dim,
            decoder_blocks=self.video2lora_decoder_blocks,
            attention_heads=self.video2lora_attention_heads,
            sample_iters=self.video2lora_sample_iters,
            max_reference_frames=self.video2lora_max_reference_frames,
            spatial_pool_size=self.video2lora_spatial_pool_size,
        )
        self._last_runtime_source: Optional[str] = None

    def clear_video2lora_runtime_condition(self) -> None:
        self._last_runtime_source = None
        for module in self._video2lora_modules:
            module.clear_runtime_weight_embedding()

    def on_step_start(self) -> None:
        super().on_step_start()
        self.clear_video2lora_runtime_condition()

    def _pick_runtime_reference(
        self,
        latents: Optional[torch.Tensor],
        control_signal: Optional[torch.Tensor],
        pixels: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[str]]:
        del latents, pixels
        normalized = _normalize_video_tensor(control_signal)
        if normalized is None:
            return None, None
        return normalized, "control_signal"

    def set_video2lora_runtime_condition(
        self,
        latents: Optional[torch.Tensor],
        control_signal: Optional[torch.Tensor],
        pixels: Optional[torch.Tensor],
        timesteps: Optional[torch.Tensor] = None,
    ) -> bool:
        del timesteps
        reference, source_name = self._pick_runtime_reference(
            latents=latents,
            control_signal=control_signal,
            pixels=pixels,
        )
        if reference is None:
            self.clear_video2lora_runtime_condition()
            return not self.video2lora_require_reference

        if self.video2lora_reference_dropout_p > 0.0 and self.training:
            drop_mask = torch.rand(
                (reference.shape[0],), device=reference.device
            ) < float(self.video2lora_reference_dropout_p)
            if bool(drop_mask.any().item()):
                reference = reference.clone()
                reference[drop_mask] = 0.0

        predicted = self.video2lora_hypernet(reference)
        for module, embedding in zip(self._video2lora_modules, predicted.unbind(dim=1)):
            module.set_runtime_weight_embedding(embedding)

        self._last_runtime_source = source_name
        return True

    def prepare_optimizer_params(
        self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], list[str]]:
        all_params, lr_descriptions = super().prepare_optimizer_params(
            unet_lr=unet_lr,
            input_lr_scale=input_lr_scale,
            **kwargs,
        )
        hypernet_params = [
            param for param in self.video2lora_hypernet.parameters() if param.requires_grad
        ]
        if hypernet_params:
            all_params.append(
                {
                    "params": hypernet_params,
                    "lr": float(unet_lr) * float(self.video2lora_hypernet_lr_ratio),
                }
            )
            lr_descriptions.append("video2lora hypernet")
        return all_params, lr_descriptions

    def is_mergeable(self) -> bool:
        return False

    def merge_to(
        self,
        text_encoders: Any,
        unet: Any,
        weights_sd: Dict[str, torch.Tensor],
        dtype: Any = None,
        device: Any = None,
        non_blocking: bool = False,
    ) -> None:
        del text_encoders, unet, weights_sd, dtype, device, non_blocking
        raise RuntimeError(
            "Video2LoRA is runtime-conditioned and cannot be merged statically. "
            "Load it as a network module and provide paired reference latents at runtime."
        )


def _parse_video2lora_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "video2lora_aux_down_dim": _parse_int(
            kwargs.get("video2lora_aux_down_dim", 128), 128
        ),
        "video2lora_aux_up_dim": _parse_int(
            kwargs.get("video2lora_aux_up_dim", 64), 64
        ),
        "video2lora_reference_feature_dim": _parse_int(
            kwargs.get("video2lora_reference_feature_dim", 16),
            16,
        ),
        "video2lora_feature_dim": _parse_int(
            kwargs.get("video2lora_feature_dim", 256), 256
        ),
        "video2lora_decoder_blocks": _parse_int(
            kwargs.get("video2lora_decoder_blocks", 4), 4
        ),
        "video2lora_attention_heads": _parse_int(
            kwargs.get("video2lora_attention_heads", 4), 4
        ),
        "video2lora_sample_iters": _parse_int(
            kwargs.get("video2lora_sample_iters", 4), 4
        ),
        "video2lora_max_reference_frames": _parse_int(
            kwargs.get("video2lora_max_reference_frames", 0), 0
        ),
        "video2lora_spatial_pool_size": _parse_int(
            kwargs.get("video2lora_spatial_pool_size", 0), 0
        ),
        "video2lora_runtime_source": str(
            kwargs.get("video2lora_runtime_source", "auto")
        ).strip().lower(),
        "video2lora_require_reference": _parse_bool(
            kwargs.get("video2lora_require_reference", True), True
        ),
        "video2lora_reference_dropout_p": _parse_float(
            kwargs.get("video2lora_reference_dropout_p", 0.0), 0.0
        ),
        "video2lora_hypernet_lr_ratio": _parse_float(
            kwargs.get("video2lora_hypernet_lr_ratio", 1.0), 1.0
        ),
    }


def _infer_hypernet_weight_dim(weights_sd: Dict[str, torch.Tensor]) -> int:
    for key, value in weights_sd.items():
        if key.endswith("video2lora_hypernet.block_pos_emb"):
            return int(value.shape[-1])
    raise ValueError("Video2LoRA checkpoint is missing video2lora_hypernet.block_pos_emb")


def _infer_reference_feature_dim(weights_sd: Dict[str, torch.Tensor]) -> int:
    for key, value in weights_sd.items():
        if key.endswith("video2lora_hypernet.feature_proj.weight"):
            return int(value.shape[1])
    raise ValueError(
        "Video2LoRA checkpoint is missing video2lora_hypernet.feature_proj.weight"
    )


def _infer_hypernet_feature_dim(weights_sd: Dict[str, torch.Tensor]) -> int:
    for key, value in weights_sd.items():
        if key.endswith("video2lora_hypernet.decoder.layers.0.linear1.weight"):
            return int(value.shape[0])
    return 256


def _infer_decoder_blocks(weights_sd: Dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"video2lora_hypernet\.decoder\.layers\.(\d+)\.")
    max_index = -1
    for key in weights_sd:
        match = pattern.search(key)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1 if max_index >= 0 else 4


def _infer_aux_dims(weights_sd: Dict[str, torch.Tensor]) -> tuple[int, int]:
    down_dim = None
    up_dim = None
    for key, value in weights_sd.items():
        if key.endswith(".down_aux"):
            down_dim = int(value.shape[0])
        elif key.endswith(".up_aux"):
            up_dim = int(value.shape[1])
        if down_dim is not None and up_dim is not None:
            return down_dim, up_dim
    raise ValueError("Video2LoRA checkpoint is missing auxiliary matrices.")


def _infer_rank_and_alpha(weights_sd: Dict[str, torch.Tensor]) -> tuple[int, float]:
    down_dim, up_dim = _infer_aux_dims(weights_sd)
    weight_dim = _infer_hypernet_weight_dim(weights_sd)
    denom = down_dim + up_dim
    if denom <= 0 or weight_dim % denom != 0:
        raise ValueError(
            "Video2LoRA checkpoint has inconsistent hypernetwork/query dimensions."
        )
    rank = max(1, weight_dim // denom)
    alpha = float(rank)
    for key, value in weights_sd.items():
        if key.endswith(".alpha"):
            alpha = float(value.detach().reshape(-1)[0].item())
            break
    return rank, alpha


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs: Any,
) -> Video2LoRANetwork:
    del vae
    rank = int(network_dim if network_dim is not None else 1)
    alpha = float(network_alpha if network_alpha is not None else rank)
    parsed_kwargs = _parse_video2lora_kwargs(kwargs)
    network_kwargs = dict(kwargs)
    network_kwargs.update(parsed_kwargs)

    network = Video2LoRANetwork(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="video2lora_unet",
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        dropout=neuron_dropout,
        **network_kwargs,
    )
    logger.info(
        "Video2LoRA initialized: rank=%s, aux=%s/%s, ref_feat_dim=%s, sample_iters=%s, modules=%s, source=%s",
        rank,
        network.video2lora_aux_down_dim,
        network.video2lora_aux_up_dim,
        network.video2lora_reference_feature_dim,
        network.video2lora_sample_iters,
        len(network.unet_loras),
        network.video2lora_runtime_source,
    )
    return network


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs: Any,
) -> Video2LoRANetwork:
    return create_arch_network(
        multiplier=multiplier,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> Video2LoRANetwork:
    del for_inference
    rank, alpha = _infer_rank_and_alpha(weights_sd)
    down_dim, up_dim = _infer_aux_dims(weights_sd)
    reference_feature_dim = _infer_reference_feature_dim(weights_sd)
    hypernet_feature_dim = _infer_hypernet_feature_dim(weights_sd)
    decoder_blocks = _infer_decoder_blocks(weights_sd)
    kwargs = dict(kwargs)
    kwargs.setdefault("video2lora_aux_down_dim", down_dim)
    kwargs.setdefault("video2lora_aux_up_dim", up_dim)
    kwargs.setdefault("video2lora_reference_feature_dim", reference_feature_dim)
    kwargs.setdefault("video2lora_feature_dim", hypernet_feature_dim)
    kwargs.setdefault("video2lora_decoder_blocks", decoder_blocks)

    network = create_arch_network(
        multiplier=multiplier,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,  # type: ignore[arg-type]
        text_encoders=text_encoders or [],
        unet=unet,  # type: ignore[arg-type]
        neuron_dropout=None,
        **kwargs,
    )
    info = network.load_state_dict(weights_sd, strict=False)
    logger.info(
        "Loaded Video2LoRA checkpoint from weights (missing=%s, unexpected=%s).",
        len(info.missing_keys),
        len(info.unexpected_keys),
    )
    return network


def create_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> Video2LoRANetwork:
    return create_arch_network_from_weights(
        multiplier=multiplier,
        weights_sd=weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )
