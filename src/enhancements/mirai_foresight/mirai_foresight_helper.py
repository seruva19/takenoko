"""Mirai Foresight training-only future representation alignment.

This helper ports the inference-free part of Video-Mirai/Foresight-Forcing:
current DiT hidden states are projected toward detached future hidden states
from the same training video clip. The projector and hooks are used only during
training and are not part of the exported LoRA inference graph.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class ResidualMLPProjector(nn.Module):
    """Identity-initialized MLP projector for stable feature prediction."""

    def __init__(self, dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        modules: List[nn.Module] = [nn.LayerNorm(dim)]
        width = dim
        for _ in range(max(1, layers - 1)):
            modules.append(nn.Linear(width, hidden_dim))
            modules.append(nn.SiLU())
            width = hidden_dim
        final = nn.Linear(width, dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, delta: int = 1) -> torch.Tensor:
        del delta
        return x + self.net(x)


class DeltaEmbedder(nn.Module):
    """Sinusoidal delta embedding followed by a small MLP."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 16,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max(2, int(max_period))
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, delta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, device=delta.device, dtype=torch.float32)
            / max(1, half)
        )
        args = delta[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.frequency_embedding_size:
            emb = F.pad(emb, (0, self.frequency_embedding_size - emb.shape[-1]))
        return self.mlp(emb.to(dtype=dtype))


class ForesightDiTLayer(nn.Module):
    """Small DiT-style self-attention layer with AdaLN modulation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        gate_init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        if gate_init_value != 0.0:
            with torch.no_grad():
                bias = self.adaLN_modulation[-1].bias
                gate_dim = bias.shape[0] // 6
                bias[2 * gate_dim : 3 * gate_dim] = gate_init_value
                bias[5 * gate_dim : 6 * gate_dim] = gate_init_value

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)

        y = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn

        y = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ffn(y)
        return x


class MiraiForesightDiTProjector(nn.Module):
    """DiT-style future hidden-state projector."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        max_delta: int,
    ) -> None:
        super().__init__()
        self.delta_embedder = DeltaEmbedder(dim, max_period=max_delta + 1)
        self.layers = nn.ModuleList(
            [
                ForesightDiTLayer(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    gate_init_value=0.0,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, delta: int = 1) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 4:
            batch, frames, tokens, dim = x.shape
            x = x.reshape(batch, frames * tokens, dim)
        batch = x.shape[0]
        delta_tensor = torch.full(
            (batch,),
            int(delta),
            device=x.device,
            dtype=torch.long,
        )
        cond = self.delta_embedder(delta_tensor, dtype=x.dtype)
        h = x
        for layer in self.layers:
            h = layer(h, cond)
        if len(original_shape) == 4:
            h = h.reshape(original_shape)
        return h


class MiraiForesightHelper(nn.Module):
    """Train-time future-hidden alignment helper."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []
        self._shape_warning_logged = False
        self._batch_warning_logged = False

        raw_depths = getattr(args, "mirai_foresight_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            self.alignment_depths = [int(value) for value in raw_depths]
        else:
            self.alignment_depths = [
                int(getattr(args, "mirai_foresight_alignment_depth", 18))
            ]
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        self.loss_lambda = float(getattr(args, "mirai_foresight_loss_lambda", 0.1))
        self.delta = int(getattr(args, "mirai_foresight_delta", 1))
        self.num_frame_per_block = int(
            getattr(args, "mirai_foresight_num_frame_per_block", 1)
        )
        self.include_current = bool(
            getattr(args, "mirai_foresight_include_current", False)
        )
        self.delta_mean_pool = bool(
            getattr(args, "mirai_foresight_delta_mean_pool", False)
        )
        self.loss_type = str(
            getattr(args, "mirai_foresight_loss_type", "negative_cosine")
        ).lower()
        self.max_spatial_tokens = int(
            getattr(args, "mirai_foresight_max_spatial_tokens", 256)
        )
        self.detach_teacher = bool(
            getattr(args, "mirai_foresight_detach_teacher", True)
        )
        self.last_mirai_metrics: dict[str, torch.Tensor] = {}

        projector_type = str(
            getattr(args, "mirai_foresight_projector_type", "mlp")
        ).lower()
        layers = int(getattr(args, "mirai_foresight_projector_layers", 2))
        hidden_dim = int(getattr(args, "mirai_foresight_projector_hidden_dim", 0))
        if hidden_dim <= 0:
            hidden_dim = int(
                self.hidden_dim
                * float(getattr(args, "mirai_foresight_ffn_ratio", 4.0))
            )
        if projector_type == "dit":
            num_heads = int(getattr(args, "mirai_foresight_num_heads", 0))
            if num_heads <= 0:
                num_heads = max(1, self.hidden_dim // 64)
            while self.hidden_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            self.projectors = nn.ModuleList(
                [
                    MiraiForesightDiTProjector(
                        dim=self.hidden_dim,
                        num_heads=num_heads,
                        ffn_dim=hidden_dim,
                        num_layers=layers,
                        max_delta=self.delta,
                    )
                    for _ in self.alignment_depths
                ]
            )
        else:
            self.projectors = nn.ModuleList(
                [
                    ResidualMLPProjector(
                        dim=self.hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=layers,
                    )
                    for _ in self.alignment_depths
                ]
            )

        logger.info(
            "Mirai Foresight: initialized %s projector(s) at depths=%s "
            "(dim=%d, delta=%d, lambda=%.4f).",
            projector_type,
            self.alignment_depths,
            self.hidden_dim,
            self.delta,
            self.loss_lambda,
        )

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("Mirai Foresight: falling back to hidden_dim=1024")
        return 1024

    def _locate_blocks(self) -> Sequence[Any]:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("Mirai Foresight: could not locate transformer block list")

    def _get_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            features = _first_tensor(output)
            if features is not None:
                self.captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        try:
            for i, depth in enumerate(self.alignment_depths):
                if depth >= num_blocks:
                    raise ValueError(
                        f"Mirai Foresight alignment depth {depth} exceeds "
                        f"available blocks ({num_blocks})"
                    )
                handle = blocks[depth].register_forward_hook(self._get_hook(i))
                self.hook_handles.append(handle)
                logger.info("Mirai Foresight: hook attached to layer %d.", depth)
        except Exception:
            self.remove_hooks()
            raise

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()
        self.captured_features = [None] * len(self.alignment_depths)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.projectors.parameters())

    def _zero_loss(self, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            return next(self.parameters()).new_tensor(0.0)
        except StopIteration:
            if isinstance(reference, torch.Tensor):
                return reference.new_tensor(0.0)
            return torch.tensor(0.0)

    def _frame_candidates(
        self,
        clean_pixels: torch.Tensor,
        latents: Optional[torch.Tensor],
    ) -> List[int]:
        candidates: List[int] = []
        if isinstance(latents, torch.Tensor) and latents.dim() == 5:
            patch_size = getattr(self.args, "patch_size", None)
            patch_t = 1
            if isinstance(patch_size, (list, tuple)) and len(patch_size) > 0:
                patch_t = max(1, int(patch_size[0]))
            elif isinstance(patch_size, int):
                patch_t = max(1, int(patch_size))
            candidates.append(max(1, int(latents.shape[2]) // patch_t))
        if clean_pixels.dim() == 5:
            candidates.append(int(clean_pixels.shape[2]))
        return list(dict.fromkeys(value for value in candidates if value > 0))

    def _coerce_features(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        if features.dim() == 4:
            features = features.reshape(features.shape[0], -1, features.shape[-1])
        elif features.dim() != 3:
            raise ValueError(
                f"expected hidden features [B, Seq, C], got {tuple(features.shape)}"
            )

        if features.shape[0] != batch_size:
            if features.shape[0] > batch_size:
                if not self._batch_warning_logged:
                    logger.warning(
                        "Mirai Foresight: hidden batch %d differs from pixel batch %d; using first %d rows.",
                        features.shape[0],
                        batch_size,
                        batch_size,
                    )
                    self._batch_warning_logged = True
                features = features[:batch_size]
            else:
                raise ValueError(
                    f"hidden batch {features.shape[0]} is smaller than pixel batch {batch_size}"
                )
        if features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden dim {features.shape[-1]} does not match projector dim {self.hidden_dim}"
            )
        return features

    def _reshape_to_frames(
        self,
        features: torch.Tensor,
        frame_candidates: Sequence[int],
    ) -> Optional[torch.Tensor]:
        seq_len = int(features.shape[1])
        min_required = self.delta * self.num_frame_per_block + 1
        for frames in frame_candidates:
            if frames < min_required:
                continue
            if seq_len % frames == 0:
                tokens = seq_len // frames
                return features.reshape(features.shape[0], frames, tokens, features.shape[-1])

        if not self._shape_warning_logged:
            logger.warning(
                "Mirai Foresight: could not reshape hidden sequence length %d into candidate frame counts %s. Skipping loss.",
                seq_len,
                list(frame_candidates),
            )
            self._shape_warning_logged = True
        return None

    def _cap_spatial_tokens(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        max_tokens = self.max_spatial_tokens
        if max_tokens < 0 or frame_tokens.shape[2] <= max_tokens:
            return frame_tokens
        indices = torch.linspace(
            0,
            frame_tokens.shape[2] - 1,
            max_tokens,
            device=frame_tokens.device,
        ).long()
        return frame_tokens.index_select(2, indices)

    def _build_future_pairs(
        self,
        frame_tokens: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        block_stride = max(1, self.num_frame_per_block)
        max_offset = self.delta * block_stride
        frames = frame_tokens.shape[1]
        if frames <= max_offset:
            return None

        valid_frames = frames - max_offset
        source = frame_tokens[:, :valid_frames]

        if self.delta_mean_pool:
            start_delta = 0 if self.include_current else 1
            offsets = [delta * block_stride for delta in range(start_delta, self.delta + 1)]
            offsets = [offset for offset in offsets if offset <= max_offset]
            if not offsets:
                offsets = [max_offset]
            targets = [
                frame_tokens[:, offset : offset + valid_frames]
                for offset in offsets
            ]
            target = torch.stack(targets, dim=0).mean(dim=0)
            projector_delta = 0 if self.include_current else self.delta
        else:
            target = frame_tokens[:, max_offset : max_offset + valid_frames]
            if self.include_current:
                target = 0.5 * (target + source.detach())
            projector_delta = self.delta

        if self.detach_teacher:
            target = target.detach()
        return source, target, projector_delta

    def _loss(self, projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        projected = projected.reshape(-1, projected.shape[-1])
        target = target.reshape(-1, target.shape[-1]).to(dtype=projected.dtype)
        if self.loss_type == "negative_cosine":
            return -F.cosine_similarity(projected, target, dim=-1).mean()
        if self.loss_type == "cosine":
            return (1.0 - F.cosine_similarity(projected, target, dim=-1)).mean()
        if self.loss_type == "mse":
            return F.mse_loss(projected, target)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(projected, target)
        raise ValueError(f"Unsupported Mirai Foresight loss type: {self.loss_type}")

    def get_repa_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
        item_info: Optional[Any] = None,
        latents: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del vae, item_info, timesteps
        if clean_pixels is None or not isinstance(clean_pixels, torch.Tensor):
            return self._zero_loss()
        if clean_pixels.dim() != 5:
            return self._zero_loss(clean_pixels)
        if not any(feature is not None for feature in self.captured_features):
            return self._zero_loss(clean_pixels)

        batch_size = int(clean_pixels.shape[0])
        frame_candidates = self._frame_candidates(clean_pixels, latents)
        if not frame_candidates:
            self.captured_features = [None] * len(self.alignment_depths)
            return self._zero_loss(clean_pixels)

        losses: List[torch.Tensor] = []
        token_counts: List[int] = []
        for idx, raw_features in enumerate(self.captured_features):
            if raw_features is None:
                continue
            try:
                features = self._coerce_features(raw_features, batch_size)
                frame_tokens = self._reshape_to_frames(features, frame_candidates)
                if frame_tokens is None:
                    continue
                frame_tokens = self._cap_spatial_tokens(frame_tokens)
                pair = self._build_future_pairs(frame_tokens)
                if pair is None:
                    continue
                source, target, projector_delta = pair
                projector = self.projectors[idx]
                projector_dtype = next(projector.parameters()).dtype
                projected = projector(source.to(dtype=projector_dtype), delta=projector_delta)
                losses.append(self._loss(projected, target))
                token_counts.append(int(source.shape[1] * source.shape[2]))
            except Exception as exc:
                logger.warning("Mirai Foresight layer %d loss skipped: %s", idx, exc)

        self.captured_features = [None] * len(self.alignment_depths)
        if not losses:
            return self._zero_loss(clean_pixels)

        raw_loss = torch.stack(losses).mean()
        total_loss = raw_loss * self.loss_lambda
        self.last_mirai_metrics = {
            "mirai_foresight_raw_loss": raw_loss.detach(),
            "mirai_foresight_loss": total_loss.detach(),
            "mirai_foresight_token_count": total_loss.detach().new_tensor(
                float(sum(token_counts) / max(1, len(token_counts)))
            ),
        }
        return total_loss
