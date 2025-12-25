from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from common.logger import get_logger
from utils.train_utils import get_sigmas
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from enhancements.repa.enhanced_repa_helper import interpolate_features_spatial

logger = get_logger(__name__, level=logging.INFO)


class RegProjectionHead(nn.Module):
    """Simple projection head for REG alignment."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RegHelper(nn.Module):
    """REG helper for class-token entanglement and alignment loss."""

    def __init__(self, diffusion_model: nn.Module, args: Any):
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handle: Optional[Any] = None
        self.captured_features: Optional[torch.Tensor] = None

        device = next(diffusion_model.parameters()).device
        self.encoder_manager = EncoderManager(device=device)

        encoders, encoder_types, _ = self.encoder_manager.load_encoders(
            args.reg_encoder_name, resolution=getattr(args, "reg_input_resolution", 256)
        )
        self.encoders = nn.ModuleList(encoders)
        self.encoder_types = encoder_types
        self.encoder_dims = [enc.embed_dim for enc in encoders]

        self.primary_encoder = self.encoders[0]
        self.primary_encoder_type = self.encoder_types[0]
        self.primary_encoder_dim = self.encoder_dims[0]

        self.diffusion_hidden_dim = getattr(diffusion_model, "dim", None)
        if self.diffusion_hidden_dim is None:
            raise ValueError("REG helper could not determine diffusion hidden dim.")

        self.projection_heads = nn.ModuleList(
            [
                RegProjectionHead(self.diffusion_hidden_dim, dim)
                for dim in self.encoder_dims
            ]
        )

        self.spatial_align = bool(getattr(args, "reg_spatial_align", True))
        self.alignment_depth = int(getattr(args, "reg_alignment_depth", 8))

    def attach_to_model(self, model: nn.Module) -> None:
        """Attach REG projections to the diffusion model."""
        cls_dim = int(getattr(self.args, "reg_cls_dim", 0) or 0)
        if cls_dim <= 0:
            cls_dim = self.primary_encoder_dim
            setattr(self.args, "reg_cls_dim", cls_dim)
        elif cls_dim != self.primary_encoder_dim:
            logger.warning(
                "REG cls_dim=%d does not match encoder dim=%d; using encoder dim.",
                cls_dim,
                self.primary_encoder_dim,
            )
            cls_dim = self.primary_encoder_dim
            setattr(self.args, "reg_cls_dim", cls_dim)

        if hasattr(model, "configure_reg"):
            model.configure_reg(cls_dim)

    def _get_hook(self):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            self.captured_features = output

        return hook

    def setup_hooks(self) -> None:
        """Attach hook at the configured alignment depth."""
        depth = self.alignment_depth
        target_module = None
        if hasattr(self.diffusion_model, "blocks"):
            if depth >= len(self.diffusion_model.blocks):
                raise ValueError(
                    f"reg_alignment_depth {depth} exceeds block count {len(self.diffusion_model.blocks)}"
                )
            target_module = self.diffusion_model.blocks[depth]
        elif hasattr(self.diffusion_model, "transformer_blocks"):
            target_module = self.diffusion_model.transformer_blocks[depth]
        else:
            raise ValueError("REG helper could not locate diffusion blocks.")

        self.hook_handle = target_module.register_forward_hook(self._get_hook())
        logger.info("REG: Hook attached to layer %d of the diffusion model.", depth)

    def remove_hooks(self) -> None:
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("REG: Hook removed successfully.")

    def _extract_encoder_tokens(
        self, features: Any
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cls_token = None
        patch_tokens = None
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                cls_token = features["x_norm_clstoken"]
            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
            if patch_tokens is None:
                for value in features.values():
                    if torch.is_tensor(value) and value.dim() == 3:
                        patch_tokens = value
                        break
        elif torch.is_tensor(features):
            if features.dim() == 2:
                cls_token = features
            elif features.dim() == 3:
                patch_tokens = features

        if cls_token is None and patch_tokens is not None:
            cls_token = patch_tokens.mean(dim=1)
        if patch_tokens is None and cls_token is not None:
            patch_tokens = cls_token.unsqueeze(1)

        return cls_token, patch_tokens

    def _prepare_images(self, clean_pixels: torch.Tensor, enc_type: str) -> torch.Tensor:
        images = (clean_pixels + 1) / 2.0
        images = images * 255.0
        return preprocess_raw_image(images, enc_type)

    def prepare_class_token_inputs(
        self,
        clean_pixels: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return REG class-token input and target for denoising loss."""
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]

        images = self._prepare_images(clean_pixels.to(device), self.primary_encoder_type)
        with torch.no_grad():
            features = self.primary_encoder.forward_features(images)
        cls_token, _ = self._extract_encoder_tokens(features)
        if cls_token is None:
            raise ValueError("REG encoder did not return class token features.")

        cls_token = cls_token.to(device=device, dtype=dtype)
        if timesteps.dim() > 1:
            timesteps_scalar = timesteps.float().mean(dim=1)
        else:
            timesteps_scalar = timesteps

        sigmas = get_sigmas(
            noise_scheduler,
            timesteps_scalar,
            device,
            n_dim=2,
            dtype=dtype,
            source="reg/class_token",
            timestep_layout="per_sample",
        )
        noise = torch.randn_like(cls_token)
        cls_input = (1.0 - sigmas) * cls_token + sigmas * noise
        target_type = getattr(self.args, "reg_target_type", "flow")
        if target_type == "flow":
            cls_target = noise - cls_token
        elif target_type == "velocity":
            cls_target = noise - cls_token
        else:
            raise ValueError(
                f"Invalid reg_target_type '{target_type}'. "
                "Expected 'flow' or 'velocity'."
            )
        return cls_input, cls_target

    def get_alignment_loss(
        self, clean_pixels: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute REG alignment loss against encoder class+patch tokens."""
        if self.captured_features is None:
            return torch.tensor(0.0, device=clean_pixels.device), None

        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]
        encoder_device = next(self.encoders.parameters()).device
        clean_pixels = clean_pixels.to(encoder_device)

        losses: List[torch.Tensor] = []
        sim_values: List[torch.Tensor] = []
        for encoder, enc_type, proj in zip(
            self.encoders, self.encoder_types, self.projection_heads
        ):
            images = self._prepare_images(clean_pixels, enc_type)
            with torch.no_grad():
                features = encoder.forward_features(images)
            cls_token, patch_tokens = self._extract_encoder_tokens(features)
            if cls_token is None or patch_tokens is None:
                continue

            target_tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
            projected = proj(self.captured_features)

            proj_cls = projected[:, :1, :]
            proj_patch = projected[:, 1:, :]
            tgt_cls = target_tokens[:, :1, :]
            tgt_patch = target_tokens[:, 1:, :]

            if proj_patch.shape[1] != tgt_patch.shape[1]:
                if self.spatial_align:
                    proj_patch = interpolate_features_spatial(
                        proj_patch, tgt_patch.shape[1]
                    )
                else:
                    proj_patch = proj_patch.mean(dim=1, keepdim=True)
                    tgt_patch = tgt_patch.mean(dim=1, keepdim=True)

            projected_tokens = torch.cat([proj_cls, proj_patch], dim=1)
            target_tokens = torch.cat([tgt_cls, tgt_patch], dim=1)

            similarity_fn = getattr(self.args, "reg_similarity_fn", "cosine")
            if similarity_fn == "cosine":
                proj_norm = F.normalize(projected_tokens, dim=-1)
                tgt_norm = F.normalize(target_tokens, dim=-1)
                sim = (proj_norm * tgt_norm).sum(dim=-1)
                loss = -sim.mean()
                sim_values.append(sim.mean())
            else:
                loss = F.mse_loss(projected_tokens, target_tokens)

            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=clean_pixels.device), None

        total_loss = torch.stack(losses).mean()
        similarity = torch.stack(sim_values).mean() if sim_values else None
        return total_loss, similarity
