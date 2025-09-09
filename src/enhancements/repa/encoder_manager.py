import os
import math
import warnings
import torch
import torch.nn as nn
import torch.hub
import timm
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torchvision import transforms
from torchvision.datasets.utils import download_url
import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# ImageNet default normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# CLIP default normalization
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Cut & paste from PyTorch official master until it's in a few official releases - RW"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def fix_mocov3_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Fix MoCo v3 checkpoint state dict naming issues."""
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder"):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder.") :]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if "head" not in new_k and new_k.split(".")[0] != "fc":
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if "pos_embed" in state_dict.keys():
        state_dict["pos_embed"] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict["pos_embed"],
            [16, 16],
        )
    return state_dict


def preprocess_raw_image(x: torch.Tensor, enc_type: str) -> torch.Tensor:
    """
    Preprocess raw images for different encoder types.

    Args:
        x: Input tensor in range [0, 255] with shape (B, C, H, W)
        enc_type: Encoder type string (e.g., 'dinov2', 'clip', 'mocov3', etc.)

    Returns:
        Preprocessed tensor ready for encoder input
    """
    resolution = x.shape[-1]

    if "clip" in enc_type:
        x = x / 255.0
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )
        x = transforms.Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif "mocov3" in enc_type or "mae" in enc_type:
        x = x / 255.0
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "dinov2" in enc_type:
        x = x / 255.0
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )
    elif "dinov1" in enc_type:
        x = x / 255.0
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "jepa" in enc_type:
        x = x / 255.0
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )

    return x


class EnhancedEncoder(nn.Module):
    """Wrapper for different encoder architectures with unified interface."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_type: str,
        architecture: str,
        model_config: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config

        # Determine embed_dim
        if hasattr(encoder, "embed_dim"):
            self.embed_dim = encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            self.embed_dim = encoder.num_features
        else:
            # Try to infer from a forward pass
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.forward_features(dummy_input)
                    if isinstance(features, dict) and "x_norm_patchtokens" in features:
                        self.embed_dim = features["x_norm_patchtokens"].shape[-1]
                    elif isinstance(features, torch.Tensor):
                        self.embed_dim = features.shape[-1]
                    else:
                        raise ValueError("Could not determine embed_dim")
            except Exception:
                # Fallback
                self.embed_dim = 768

    def forward_features(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract features from encoder."""
        if hasattr(self.encoder, "forward_features"):
            return self.encoder.forward_features(x)
        else:
            # Fallback for encoders without explicit forward_features
            return self.encoder(x)


class EncoderManager:
    """Advanced encoder loading and management system based on REPA utilities."""

    def __init__(self, device: Union[str, torch.device], cache_dir: str = "models"):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def load_encoders(
        self, enc_type: str, resolution: int = 256
    ) -> Tuple[List[EnhancedEncoder], List[str], List[str]]:
        """
        Load multiple encoders from specification string.

        Args:
            enc_type: Comma-separated encoder specifications like "dinov2-vit-b,clip-vit-L"
            resolution: Input image resolution (256 or 512)

        Returns:
            Tuple of (encoders, encoder_types, architectures)
        """
        assert resolution in [256, 512], f"Unsupported resolution: {resolution}"

        enc_names = enc_type.split(",")
        encoders: List[EnhancedEncoder] = []
        architectures: List[str] = []
        encoder_types: List[str] = []

        for enc_name in enc_names:
            enc_name = enc_name.strip()
            logger.info(f"Loading encoder: {enc_name}")

            try:
                encoder_type, architecture, model_config = enc_name.split("-")
            except ValueError:
                logger.warning(f"Invalid encoder specification: {enc_name}. Skipping.")
                continue

            # Currently, we only support 512x512 experiments with DINOv2 encoders.
            if resolution == 512 and encoder_type != "dinov2":
                logger.warning(
                    f"Resolution 512 only supported with DINOv2 encoders, skipping {enc_name}"
                )
                continue

            architectures.append(architecture)
            encoder_types.append(encoder_type)

            # Load specific encoder type
            if encoder_type == "mocov3":
                encoder = self._load_mocov3_encoder(architecture, model_config)
            elif encoder_type == "dinov2":
                encoder = self._load_dinov2_encoder(model_config, resolution)
            elif encoder_type == "dinov1":
                encoder = self._load_dinov1_encoder(model_config)
            elif encoder_type == "clip":
                encoder = self._load_clip_encoder(model_config)
            elif encoder_type == "mae":
                encoder = self._load_mae_encoder(model_config)
            elif encoder_type == "jepa":
                encoder = self._load_jepa_encoder(model_config)
            else:
                logger.warning(f"Unknown encoder type: {encoder_type}. Skipping.")
                # Remove from lists since we're skipping
                architectures.pop()
                encoder_types.pop()
                continue

            # Wrap in EnhancedEncoder and add to list
            enhanced_encoder = EnhancedEncoder(
                encoder, encoder_type, architecture, model_config
            )
            enhanced_encoder = enhanced_encoder.to(self.device)
            enhanced_encoder.eval()
            encoders.append(enhanced_encoder)

            logger.info(
                f"Successfully loaded {enc_name} with embed_dim={enhanced_encoder.embed_dim}"
            )

        if not encoders:
            raise ValueError(f"No valid encoders could be loaded from: {enc_type}")

        return encoders, encoder_types, architectures

    def _load_mocov3_encoder(self, architecture: str, model_config: str) -> nn.Module:
        """Load MoCo v3 encoder."""
        if architecture != "vit":
            raise NotImplementedError(f"MoCo v3 {architecture} not supported")

        # Import here to avoid dependency issues
        try:
            from enhancements.repa.models import mocov3_vit
        except ImportError:
            logger.error(
                "mocov3_vit model not found. Please ensure REPA models are available."
            )
            raise

        if model_config == "s":
            encoder = mocov3_vit.vit_small()
        elif model_config == "b":
            encoder = mocov3_vit.vit_base()
        elif model_config == "l":
            encoder = mocov3_vit.vit_large()
        else:
            raise ValueError(f"Unsupported MoCo v3 config: {model_config}")

        # Load checkpoint
        ckpt_path = os.path.join(self.cache_dir, f"mocov3_vit{model_config}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MoCo v3 checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = fix_mocov3_state_dict(ckpt["state_dict"])

        # Remove and replace head
        if hasattr(encoder, "head"):
            del encoder.head
        encoder.load_state_dict(state_dict, strict=True)
        encoder.head = torch.nn.Identity()

        return encoder

    def _load_dinov2_encoder(self, model_config: str, resolution: int) -> nn.Module:
        """Load DINOv2 encoder."""
        reg_variant = "reg" in model_config
        config_clean = model_config.replace("reg", "") if reg_variant else model_config

        if reg_variant:
            encoder = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_vit{config_clean}14_reg"
            )
        else:
            encoder = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_vit{config_clean}14"
            )

        # Remove classification head
        if hasattr(encoder, "head"):
            del encoder.head

        # Resize positional embeddings for different resolutions
        patch_resolution = 16 * (resolution // 256)
        if hasattr(encoder, "pos_embed") and encoder.pos_embed is not None:
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data,
                [patch_resolution, patch_resolution],
            )

        encoder.head = torch.nn.Identity()
        return encoder

    def _load_dinov1_encoder(self, model_config: str) -> nn.Module:
        """Load DINOv1 encoder."""
        try:
            from enhancements.repa.models.vision_transformer import vit_base
        except ImportError:
            logger.error(
                "dinov1 model not found. Please ensure REPA models are available."
            )
            raise

        encoder = vit_base()

        # Load checkpoint
        ckpt_path = os.path.join(self.cache_dir, f"dinov1_vit{model_config}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"DINOv1 checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Resize positional embeddings if needed
        if "pos_embed" in ckpt.keys():
            ckpt["pos_embed"] = timm.layers.pos_embed.resample_abs_pos_embed(
                ckpt["pos_embed"],
                [16, 16],
            )

        # Remove and replace head
        if hasattr(encoder, "head"):
            del encoder.head
        encoder.head = torch.nn.Identity()
        encoder.load_state_dict(ckpt, strict=True)

        # Set forward_features method
        encoder.forward_features = encoder.forward

        return encoder

    def _load_clip_encoder(self, model_config: str) -> nn.Module:
        """Load CLIP encoder."""
        try:
            import clip  # type: ignore
            from enhancements.repa.models.clip_vit import UpdatedVisionTransformer
        except ImportError:
            logger.error(
                "CLIP dependencies not found. Please ensure CLIP and REPA models are available."
            )
            raise

        encoder_ = clip.load(f"ViT-{model_config}/14", device="cpu")[0].visual
        encoder = UpdatedVisionTransformer(encoder_)
        encoder.embed_dim = encoder.model.transformer.width
        encoder.forward_features = encoder.forward

        return encoder

    def _load_mae_encoder(self, model_config: str) -> nn.Module:
        """Load MAE encoder."""
        try:
            from enhancements.repa.models.mae_vit import vit_large_patch16
        except ImportError:
            logger.error(
                "MAE model not found. Please ensure REPA models are available."
            )
            raise

        kwargs = dict(img_size=256)
        encoder = vit_large_patch16(**kwargs)

        # Load checkpoint
        ckpt_path = os.path.join(self.cache_dir, f"mae_vit{model_config}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")

        with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        # Resize positional embeddings if needed
        if "pos_embed" in state_dict["model"].keys():
            state_dict["model"]["pos_embed"] = (
                timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]["pos_embed"],
                    [16, 16],
                )
            )

        encoder.load_state_dict(state_dict["model"])

        # Additional positional embedding resize
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data,
            [16, 16],
        )

        return encoder

    def _load_jepa_encoder(self, model_config: str) -> nn.Module:
        """Load I-JEPA encoder."""
        try:
            from enhancements.repa.models.jepa import vit_huge
        except ImportError:
            logger.error(
                "I-JEPA model not found. Please ensure REPA models are available."
            )
            raise

        kwargs = dict(img_size=[224, 224], patch_size=14)
        encoder = vit_huge(**kwargs)

        # Load checkpoint
        ckpt_path = os.path.join(self.cache_dir, f"ijepa_vit{model_config}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"I-JEPA checkpoint not found: {ckpt_path}")

        with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f, map_location=self.device)

        # Extract encoder state dict
        new_state_dict = dict()
        for key, value in state_dict["encoder"].items():
            new_state_dict[key[7:]] = value  # Remove 'module.' prefix

        encoder.load_state_dict(new_state_dict)
        encoder.forward_features = encoder.forward

        return encoder


def get_preprocessing_transform(encoder_types: List[str]) -> transforms.Compose:
    """
    Get appropriate preprocessing transform for given encoder types.

    Args:
        encoder_types: List of encoder type strings

    Returns:
        Composed preprocessing transform
    """
    # Determine the most appropriate normalization
    # If we have CLIP encoders, prefer CLIP normalization, otherwise use ImageNet
    if any("clip" in enc_type for enc_type in encoder_types):
        mean, std = CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
    else:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    return transforms.Compose([transforms.Normalize(mean=mean, std=std)])
