from __future__ import annotations

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


def _interpolate_tokens(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token features to match token count."""
    if features.shape[1] == target_tokens:
        return features

    bsz, src_tokens, dim = features.shape
    src_side = int(math.isqrt(src_tokens))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_tokens:
        x = features.permute(0, 2, 1).reshape(bsz, dim, src_side, src_side)
        x = F.interpolate(
            x,
            size=(tgt_side, tgt_side),
            mode="bilinear",
            align_corners=False,
        )
        return x.reshape(bsz, dim, target_tokens).permute(0, 2, 1)

    x = features.permute(0, 2, 1)
    x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


class VaeRepaHelper(nn.Module):
    """VAE-REPA helper: aligns diffusion hidden states to VAE latent features."""

    def __init__(
        self, diffusion_model: Any, args: Any, vae: Optional[Any] = None
    ) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        self.vae_feature_dim = self._infer_vae_feature_dim(vae)
        self.alignment_depths = self._parse_alignment_depths()
        self.auto_depth = bool(getattr(args, "vae_repa_auto_depth", False))
        self.loss_lambda = float(getattr(args, "vae_repa_loss_lambda", 1.0))
        self.loss_beta = float(getattr(args, "vae_repa_loss_beta", 0.05))
        self.alignment_loss = str(
            getattr(args, "vae_repa_alignment_loss", "smooth_l1")
        ).lower()
        self.spatial_align = bool(getattr(args, "vae_repa_spatial_align", True))
        self.use_full_video = bool(getattr(args, "vae_repa_use_full_video", False))
        self.timestep_min = float(getattr(args, "vae_repa_timestep_min", 0.0))
        self.timestep_max = float(getattr(args, "vae_repa_timestep_max", 1.0))
        self.projector_hidden_mult = int(
            getattr(args, "vae_repa_projector_hidden_mult", 4)
        )
        self.projector_layers = int(getattr(args, "vae_repa_projector_layers", 5))
        projector_hidden = self.hidden_dim * self.projector_hidden_mult

        self.projectors = nn.ModuleList(
            [
                self._build_projector(projector_hidden)
                for _ in self.alignment_depths
            ]
        )
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        logger.info(
            "VAE-REPA: initialized (depths=%s, auto_depth=%s, hidden_dim=%d, vae_dim=%d, lambda=%.4f, beta=%.4f, loss=%s, proj_layers=%d, t_range=[%.3f, %.3f]).",
            ("paper_auto" if self.auto_depth else self.alignment_depths),
            self.auto_depth,
            self.hidden_dim,
            self.vae_feature_dim,
            self.loss_lambda,
            self.loss_beta,
            self.alignment_loss,
            self.projector_layers,
            self.timestep_min,
            self.timestep_max,
        )

    def _build_projector(self, projector_hidden: int) -> nn.Module:
        """Build a configurable MLP projector (paper default: 5 layers)."""
        if self.projector_layers <= 2:
            return nn.Sequential(
                nn.Linear(self.hidden_dim, projector_hidden),
                nn.SiLU(),
                nn.Linear(projector_hidden, self.vae_feature_dim),
            )

        layers: List[nn.Module] = [
            nn.Linear(self.hidden_dim, projector_hidden),
            nn.SiLU(),
        ]
        # Internal hidden blocks before output layer.
        hidden_blocks = max(0, self.projector_layers - 2)
        for _ in range(hidden_blocks):
            layers.append(nn.Linear(projector_hidden, projector_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(projector_hidden, self.vae_feature_dim))
        return nn.Sequential(*layers)

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("VAE-REPA: could not infer hidden dim; falling back to 1024.")
        return 1024

    def _infer_vae_feature_dim(self, vae: Optional[Any]) -> int:
        override_dim = int(getattr(self.args, "vae_repa_target_dim", 0))
        if override_dim > 0:
            return override_dim

        if vae is not None:
            candidates = [
                getattr(vae, "z_dim", None),
                getattr(getattr(vae, "model", None), "z_dim", None),
                getattr(getattr(vae, "config", None), "latent_channels", None),
                getattr(getattr(vae, "config", None), "z_channels", None),
            ]
            for dim in candidates:
                if dim is None:
                    continue
                try:
                    dim_int = int(dim)
                except Exception:
                    continue
                if dim_int > 0:
                    return dim_int

        logger.warning(
            "VAE-REPA: could not infer VAE latent channel dim; falling back to 4. "
            "Set vae_repa_target_dim to override."
        )
        return 4

    def _parse_alignment_depths(self) -> List[int]:
        if bool(getattr(self.args, "vae_repa_auto_depth", False)):
            # Placeholder depth resolved against actual model block count in setup_hooks().
            return [-1]
        raw_depths = getattr(self.args, "vae_repa_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            return [int(v) for v in raw_depths]
        return [int(getattr(self.args, "vae_repa_alignment_depth", 2))]

    @staticmethod
    def _select_paper_auto_depth(num_blocks: int) -> int:
        """
        Paper-inspired depth mapping:
        - smaller backbones -> early depth 2
        - larger backbones -> depth 8
        """
        if num_blocks <= 0:
            return 0
        target = 2 if num_blocks <= 20 else 8
        return max(0, min(target, num_blocks - 1))

    def _locate_blocks(self) -> Any:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("VAE-REPA: could not locate transformer block list")

    def _get_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        depths = self.alignment_depths
        if self.auto_depth:
            depths = [self._select_paper_auto_depth(num_blocks)]
            self.alignment_depths = depths
            self.captured_features = [None] * len(depths)
        try:
            for i, depth in enumerate(depths):
                if depth >= num_blocks:
                    raise ValueError(
                        f"VAE-REPA alignment depth {depth} exceeds available blocks ({num_blocks})"
                    )
                handle = blocks[depth].register_forward_hook(self._get_hook(i))
                self.hook_handles.append(handle)
                logger.info("VAE-REPA: hook attached to layer %d.", depth)
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

    @staticmethod
    def _coerce_latent_tokens(latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 5:
            # (B, C, F, H, W) -> (B*F, H*W, C)
            bsz, channels, frames, height, width = latents.shape
            return (
                latents.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(bsz * frames, height * width, channels)
            )
        if latents.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            bsz, channels, height, width = latents.shape
            return (
                latents.permute(0, 2, 3, 1)
                .contiguous()
                .view(bsz, height * width, channels)
            )
        if latents.dim() == 3:
            return latents
        raise ValueError(f"VAE-REPA: unsupported latent shape {tuple(latents.shape)}")

    @staticmethod
    def _reduce_token_axis(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def _normalize_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timestep tensor into [0, 1] using training max_timestep when available."""
        t = timesteps.detach().float()
        if t.dim() > 1:
            t = t.view(t.shape[0], -1).mean(dim=1)
        max_ts = float(getattr(self.args, "max_timestep", 1.0) or 1.0)
        if max_ts > 1.0:
            t = t / max_ts
        elif t.numel() > 0 and float(t.max().item()) > 1.0:
            t = t / max(1.0, float(t.max().item()))
        return t.clamp(0.0, 1.0)

    @staticmethod
    def _mask_for_batch(mask_bsz: torch.Tensor, target_bsz: int) -> torch.Tensor:
        """Broadcast/reduce per-sample boolean mask to target batch size."""
        if target_bsz <= 0:
            return mask_bsz.new_zeros((0,), dtype=torch.bool)
        if mask_bsz.numel() == target_bsz:
            return mask_bsz
        if target_bsz % mask_bsz.numel() == 0:
            factor = target_bsz // mask_bsz.numel()
            return mask_bsz.repeat_interleave(factor)
        if mask_bsz.numel() % target_bsz == 0:
            factor = mask_bsz.numel() // target_bsz
            return mask_bsz.view(target_bsz, factor).any(dim=1)
        if target_bsz < mask_bsz.numel():
            return mask_bsz[:target_bsz]
        repeats = int(math.ceil(target_bsz / float(mask_bsz.numel())))
        return mask_bsz.repeat(repeats)[:target_bsz]

    def _align_batch_dims(
        self, projected: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected.shape[0] == target.shape[0]:
            return projected, target

        p_bsz = projected.shape[0]
        t_bsz = target.shape[0]
        if t_bsz % p_bsz == 0:
            factor = t_bsz // p_bsz
            target = target.view(p_bsz, factor, *target.shape[1:]).mean(dim=1)
            return projected, target
        if p_bsz % t_bsz == 0:
            factor = p_bsz // t_bsz
            projected = projected.view(t_bsz, factor, *projected.shape[1:]).mean(dim=1)
            return projected, target

        min_bsz = min(p_bsz, t_bsz)
        return projected[:min_bsz], target[:min_bsz]

    def get_repa_loss(
        self,
        clean_pixels: Optional[torch.Tensor],
        vae: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not any(feat is not None for feat in self.captured_features):
            base_device = (
                clean_pixels.device
                if torch.is_tensor(clean_pixels)
                else next(self.projectors.parameters()).device
            )
            return torch.tensor(0.0, device=base_device)

        target_latents = kwargs.get("latents", None)
        if not torch.is_tensor(target_latents):
            logger.warning_once(
                "VAE-REPA: latent targets were not provided. Returning zero loss."
            )
            base_device = (
                clean_pixels.device
                if torch.is_tensor(clean_pixels)
                else next(self.projectors.parameters()).device
            )
            self.captured_features = [None] * len(self.alignment_depths)
            return torch.tensor(0.0, device=base_device)

        if target_latents.dim() == 5 and not self.use_full_video:
            target_latents = target_latents[:, :, :1, :, :]
        target_tokens = self._coerce_latent_tokens(target_latents)
        target_tokens = target_tokens.to(dtype=torch.float32)
        timestep_mask: Optional[torch.Tensor] = None
        raw_timesteps = kwargs.get("timesteps", None)
        if torch.is_tensor(raw_timesteps):
            t_norm = self._normalize_timesteps(raw_timesteps).to(target_tokens.device)
            timestep_mask = (t_norm >= self.timestep_min) & (t_norm <= self.timestep_max)
            if not timestep_mask.any():
                self.captured_features = [None] * len(self.alignment_depths)
                return target_tokens.new_tensor(0.0)

        total_loss = target_tokens.new_tensor(0.0)
        valid_layers = 0

        for layer_idx, diffusion_features in enumerate(self.captured_features):
            if diffusion_features is None:
                continue

            if diffusion_features.dim() != 3:
                logger.warning_once(
                    "VAE-REPA: expected diffusion features [B, Seq, C], got %s. Skipping layer.",
                    tuple(diffusion_features.shape),
                )
                continue

            projected = self.projectors[layer_idx](diffusion_features)

            if target_latents.dim() == 5 and self.use_full_video:
                bsz, _, frames, _, _ = target_latents.shape
                if projected.shape[0] == bsz and projected.shape[1] % frames == 0:
                    tokens_per_frame = projected.shape[1] // frames
                    projected = projected.view(bsz, frames, tokens_per_frame, -1)
                    projected = projected.view(bsz * frames, tokens_per_frame, -1)

            projected, layer_target = self._align_batch_dims(projected, target_tokens)
            if timestep_mask is not None:
                layer_mask = self._mask_for_batch(timestep_mask, projected.shape[0]).to(
                    projected.device
                )
                if not layer_mask.any():
                    continue
                projected = projected[layer_mask]
                layer_target = layer_target[layer_mask]

            shapes_match = projected.shape == layer_target.shape
            if (
                not shapes_match
                and projected.dim() == 3
                and layer_target.dim() == 3
                and self.spatial_align
            ):
                proj_tokens = projected.shape[1]
                tgt_tokens = layer_target.shape[1]
                if proj_tokens != tgt_tokens:
                    if proj_tokens > tgt_tokens:
                        projected = _interpolate_tokens(projected, tgt_tokens)
                    else:
                        layer_target = _interpolate_tokens(layer_target, proj_tokens)
                shapes_match = projected.shape == layer_target.shape

            if not shapes_match:
                projected = self._reduce_token_axis(projected)
                layer_target = self._reduce_token_axis(layer_target)

            projected_f = projected.to(dtype=torch.float32)
            layer_target_f = layer_target.to(dtype=torch.float32)
            if self.alignment_loss == "smooth_l1":
                layer_loss = F.smooth_l1_loss(
                    projected_f,
                    layer_target_f,
                    beta=self.loss_beta,
                    reduction="mean",
                )
            elif self.alignment_loss == "cosine":
                proj_norm = F.normalize(projected_f, dim=-1)
                tgt_norm = F.normalize(layer_target_f, dim=-1)
                layer_loss = (1.0 - (proj_norm * tgt_norm).sum(dim=-1)).mean()
            elif self.alignment_loss == "l1":
                layer_loss = F.l1_loss(projected_f, layer_target_f, reduction="mean")
            else:
                layer_loss = F.mse_loss(projected_f, layer_target_f, reduction="mean")
            total_loss = total_loss + layer_loss
            valid_layers += 1

        self.captured_features = [None] * len(self.alignment_depths)
        if valid_layers == 0:
            return total_loss
        return (total_loss / float(valid_layers)) * self.loss_lambda
