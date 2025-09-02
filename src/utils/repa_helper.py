import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from typing import Optional, Any, Union
import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class RepaHelper(nn.Module):
    """
    A helper module to encapsulate all logic for REPA (Representation Alignment).

    This module handles:
    1. Loading a frozen pretrained visual encoder (e.g., DINOv2)
    2. Creating an MLP projection head to align diffusion model features
    3. Setting up forward hooks to capture hidden states
    4. Computing the REPA loss for representation alignment
    """

    def __init__(self, diffusion_model: Any, args: Any):
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model

        # 1. Load the frozen, pretrained visual encoder (e.g., DINOv2)
        logger.info(
            f"REPA: Loading pretrained visual encoder: {args.repa_encoder_name}"
        )
        try:
            self.visual_encoder: nn.Module = timm.create_model(
                args.repa_encoder_name,
                pretrained=True,
                cache_dir="models",
                num_classes=0,  # Remove the classification head
            ).eval()
            self.visual_encoder.requires_grad_(False)

            # Get the feature dimension from the encoder
            if hasattr(self.visual_encoder, "embed_dim"):
                embed_dim = getattr(self.visual_encoder, "embed_dim")
                self.encoder_feature_dim = (
                    int(embed_dim) if embed_dim is not None else 768
                )
            elif hasattr(self.visual_encoder, "num_features"):
                num_features = getattr(self.visual_encoder, "num_features")
                self.encoder_feature_dim = (
                    int(num_features) if num_features is not None else 768
                )
            else:
                # Fallback: try to infer from the model
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.visual_encoder.forward_features(dummy_input)  # type: ignore
                    if isinstance(features, dict) and "x_norm_patchtokens" in features:
                        self.encoder_feature_dim = features["x_norm_patchtokens"].shape[
                            -1
                        ]
                    else:
                        # Try to get the last dimension
                        if isinstance(features, torch.Tensor):
                            self.encoder_feature_dim = features.shape[-1]
                        else:
                            raise ValueError(
                                f"Could not determine feature dimension for {args.repa_encoder_name}"
                            )

            logger.info(
                f"REPA: Visual encoder loaded successfully. Feature dim: {self.encoder_feature_dim}"
            )

        except Exception as e:
            logger.error(
                f"REPA ERROR: Failed to load visual encoder {args.repa_encoder_name}: {e}"
            )
            raise

        # 2. Create the MLP projection head
        # Get the diffusion model's hidden dimension
        if hasattr(diffusion_model, "dim"):
            self.diffusion_hidden_dim = int(diffusion_model.dim)
        elif hasattr(diffusion_model, "hidden_size"):
            self.diffusion_hidden_dim = int(diffusion_model.hidden_size)
        else:
            # Try to infer from the model structure
            for module in diffusion_model.modules():
                if hasattr(module, "in_features"):
                    self.diffusion_hidden_dim = int(module.in_features)
                    break
            else:
                # Fallback to a reasonable default
                self.diffusion_hidden_dim = 1024
                logger.warning(
                    f"REPA: Could not determine diffusion model hidden dim, using default: {self.diffusion_hidden_dim}"
                )

        self.projection_head = self._create_projection_head()

        # 3. Image normalization transform for the visual encoder
        # DINOv2 and most timm models expect this standard ImageNet normalization
        self.image_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # 4. Placeholders for hooks and captured features
        self.hook_handle: Optional[Any] = None
        self.captured_features: Optional[torch.Tensor] = None

        logger.info(
            f"REPA: Initialized with diffusion hidden dim: {self.diffusion_hidden_dim}, encoder feature dim: {self.encoder_feature_dim}"
        )

    def _create_projection_head(self) -> nn.Module:
        """Creates a 3-layer MLP as described in the REPA paper."""
        return nn.Sequential(
            nn.Linear(self.diffusion_hidden_dim, self.diffusion_hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.diffusion_hidden_dim * 4, self.encoder_feature_dim),
        )

    def _get_hook(self):
        """Closure to capture the hidden state from the target layer."""

        def hook(model, input, output):
            # The output of a transformer block is typically a tuple (hidden_state, ...).
            # We are interested in the first element.
            self.captured_features = output[0] if isinstance(output, tuple) else output

        return hook

    def setup_hooks(self) -> None:
        """Finds the target layer and attaches the forward hook."""
        depth = self.args.repa_alignment_depth
        try:
            # The target module is inside the `blocks` list of your `WanModel`
            if hasattr(self.diffusion_model, "blocks"):
                target_module = self.diffusion_model.blocks[depth]
            elif hasattr(self.diffusion_model, "layers"):
                target_module = self.diffusion_model.layers[depth]
            elif hasattr(self.diffusion_model, "transformer_blocks"):
                target_module = self.diffusion_model.transformer_blocks[depth]
            else:
                # Try to find blocks in a different way
                blocks = []
                for name, module in self.diffusion_model.named_modules():
                    if "block" in name.lower() or "layer" in name.lower():
                        blocks.append(module)
                if len(blocks) > depth:
                    target_module = blocks[depth]
                else:
                    raise ValueError(
                        f"Could not find blocks in diffusion model. Available modules: {list(self.diffusion_model.named_modules())}"
                    )

            self.hook_handle = target_module.register_forward_hook(self._get_hook())
            logger.info(f"REPA: Hook attached to layer {depth} of the diffusion model.")

        except Exception as e:
            logger.error(
                f"REPA ERROR: Could not attach hook to layer {depth}. Please check `repa_alignment_depth`. Error: {e}"
            )
            raise

    def remove_hooks(self) -> None:
        """Removes the forward hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("REPA: Hook removed successfully.")

    def get_repa_loss(
        self, clean_pixels: torch.Tensor, vae: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Calculates the REPA loss for a given batch.

        Args:
            clean_pixels: Clean pixel values in range [-1, 1], shape (B, C, H, W) or (B, C, F, H, W) for video
            vae: VAE model (not used in current implementation but kept for compatibility)

        Returns:
            REPA loss tensor
        """
        if self.captured_features is None:
            return torch.tensor(0.0, device=clean_pixels.device)

        # If clean_pixels is a video batch (B, C, F, H, W), take the first frame
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]  # Take the first frame

        with torch.no_grad():
            # The visual encoder expects clean images, not latents.
            # `clean_pixels` are already normalized to [-1, 1]. We need to convert them to [0, 1].
            images = (clean_pixels + 1) / 2.0
            images = self.image_transform(images)  # Apply ImageNet normalization

            # Get target representations from the visual encoder
            # The output is patch features, not a global feature.
            target_features = self.visual_encoder.forward_features(images)  # type: ignore

            # Handle different output formats from timm models
            if isinstance(target_features, dict):
                if "x_norm_patchtokens" in target_features:
                    target_features = target_features["x_norm_patchtokens"]
                elif "x_norm_clstoken" in target_features:
                    target_features = target_features["x_norm_clstoken"]
                else:
                    # Try to find any patch-like features
                    for key, value in target_features.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 2:
                            target_features = value
                            break
                    else:
                        raise ValueError(
                            f"Could not find suitable features in encoder output: {target_features.keys()}"
                        )

            # Ensure target_features is a tensor at this point
            if not isinstance(target_features, torch.Tensor):
                raise ValueError(
                    f"Expected tensor output from visual encoder, got {type(target_features)}"
                )

            # Ensure we have the right shape: (B, N, D) where N is number of patches/tokens, D is feature dim
            if target_features.dim() == 2:
                # If we got (B, D), add a dimension to make it (B, 1, D)
                target_features = target_features.unsqueeze(1)
            elif target_features.dim() > 3:
                # If we got (B, C, H, W), flatten spatial dimensions
                B, C, H, W = target_features.shape
                target_features = target_features.view(B, C, H * W).transpose(
                    1, 2
                )  # (B, H*W, C)

        # Project the captured hidden state from the diffusion model
        projected_features = self.projection_head(self.captured_features)

        # Ensure both tensors have the same shape for comparison
        if projected_features.shape != target_features.shape:
            # Try to match the shapes by taking the mean over spatial dimensions if needed
            if projected_features.dim() == 3 and target_features.dim() == 3:
                # Both are (B, N, D), but N might be different
                if projected_features.shape[1] != target_features.shape[1]:
                    # Take mean over the spatial dimension to get (B, D)
                    projected_features = projected_features.mean(dim=1)
                    target_features = target_features.mean(dim=1)
            elif projected_features.dim() == 2 and target_features.dim() == 3:
                # projected_features is (B, D), target_features is (B, N, D)
                target_features = target_features.mean(dim=1)  # (B, D)
            elif projected_features.dim() == 3 and target_features.dim() == 2:
                # projected_features is (B, N, D), target_features is (B, D)
                projected_features = projected_features.mean(dim=1)  # (B, D)

        # Calculate similarity loss (negative cosine similarity is common)
        if self.args.repa_similarity_fn == "cosine":
            # Ensure both tensors are 2D for cosine similarity
            if projected_features.dim() > 2:
                projected_features = projected_features.mean(dim=1)
            if target_features.dim() > 2:
                target_features = target_features.mean(dim=1)

            # Normalize both tensors for cosine similarity
            projected_features = F.normalize(projected_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)

            # Cosine similarity: dot product of normalized vectors
            similarity = (projected_features * target_features).sum(dim=-1)
            loss = -similarity.mean()  # Negative because we want to maximize similarity

        else:
            raise NotImplementedError(
                f"REPA similarity function '{self.args.repa_similarity_fn}' not implemented."
            )

        # Clear captured features for the next step
        self.captured_features = None

        return loss * self.args.repa_loss_lambda
