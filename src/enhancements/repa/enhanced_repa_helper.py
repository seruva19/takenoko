import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, Any, Union, List, Dict, Tuple
import logging
from common.logger import get_logger

from enhancements.repa.encoder_manager import (
    EncoderManager,
    EnhancedEncoder,
    preprocess_raw_image,
)

logger = get_logger(__name__, level=logging.INFO)


class MultiEncoderProjectionHead(nn.Module):
    """Multi-encoder projection head that handles ensemble of encoders."""

    def __init__(
        self,
        diffusion_hidden_dim: int,
        encoder_dims: List[int],
        ensemble_mode: str = "individual",
        shared_projection: bool = False,
    ):
        super().__init__()
        self.diffusion_hidden_dim = diffusion_hidden_dim
        self.encoder_dims = encoder_dims
        self.ensemble_mode = ensemble_mode
        self.shared_projection = shared_projection

        if shared_projection:
            # Single shared projection head for all encoders
            # Project to average encoder dimension
            avg_dim = int(sum(encoder_dims) / len(encoder_dims))
            self.projection = self._create_projection_head(
                diffusion_hidden_dim, avg_dim
            )
            self.projections = None
        else:
            # Individual projection heads for each encoder
            self.projections = nn.ModuleList(
                [
                    self._create_projection_head(diffusion_hidden_dim, dim)
                    for dim in encoder_dims
                ]
            )
            self.projection = None

    def _create_projection_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """Creates a 3-layer MLP as described in the REPA paper."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.SiLU(),
            nn.Linear(input_dim * 4, output_dim),
        )

    def forward(self, diffusion_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Project diffusion features to match each encoder's feature space.

        Args:
            diffusion_features: Hidden states from diffusion model

        Returns:
            List of projected features for each encoder
        """
        if self.shared_projection:
            # Use shared projection for all encoders
            projected = self.projection(diffusion_features)
            # Replicate for each encoder (may need adaptation layers in future)
            return [projected for _ in self.encoder_dims]
        else:
            # Use individual projections
            return [proj(diffusion_features) for proj in self.projections]


class EnhancedRepaHelper(nn.Module):
    """
    Enhanced REPA helper with multi-encoder support and advanced features.

    This module extends the original REPA implementation with:
    1. Multi-encoder ensemble support
    2. Advanced encoder loading and management
    3. Encoder-specific preprocessing
    4. Multi-layer alignment options
    5. Flexible loss computation strategies
    """

    def __init__(self, diffusion_model: Any, args: Any):
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model

        # Initialize encoder manager
        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = getattr(args, "model_cache_dir", "models")
        self.encoder_manager = EncoderManager(device, cache_dir)

        # Parse encoder specification (can be comma-separated for multiple encoders)
        encoder_spec = getattr(args, "repa_encoder_name", "dinov2_vitb14")
        resolution = getattr(args, "repa_input_resolution", 256)

        logger.info(f"REPA: Loading encoder ensemble: {encoder_spec}")

        try:
            # Load encoders using the advanced manager
            self.encoders, self.encoder_types, self.architectures = (
                self.encoder_manager.load_encoders(encoder_spec, resolution)
            )

            # Get encoder dimensions
            self.encoder_dims = [enc.embed_dim for enc in self.encoders]
            logger.info(
                f"REPA: Loaded {len(self.encoders)} encoders with dims: {self.encoder_dims}"
            )

        except Exception as e:
            logger.error(f"REPA ERROR: Failed to load encoders {encoder_spec}: {e}")
            raise

        # Get the diffusion model's hidden dimension
        self.diffusion_hidden_dim = self._infer_diffusion_hidden_dim()

        # Create multi-encoder projection heads
        ensemble_mode = getattr(args, "repa_ensemble_mode", "individual")
        shared_proj = getattr(args, "repa_shared_projection", False)

        self.projection_heads = MultiEncoderProjectionHead(
            self.diffusion_hidden_dim,
            self.encoder_dims,
            ensemble_mode=ensemble_mode,
            shared_projection=shared_proj,
        )

        # Setup preprocessing transforms for each encoder type
        self._setup_preprocessing()

        # Multi-layer alignment support
        self.alignment_depths = self._parse_alignment_depths()

        # Placeholders for hooks and captured features
        self.hook_handles: List[Any] = []
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        logger.info(
            f"REPA: Enhanced helper initialized - "
            f"{len(self.encoders)} encoders, "
            f"{len(self.alignment_depths)} alignment layers, "
            f"diffusion dim: {self.diffusion_hidden_dim}"
        )

    def _infer_diffusion_hidden_dim(self) -> int:
        """Infer the hidden dimension of the diffusion model."""
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        elif hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        else:
            # Try to infer from the model structure
            for module in self.diffusion_model.modules():
                if hasattr(module, "in_features"):
                    return int(module.in_features)
            # Fallback to a reasonable default
            logger.warning(
                f"REPA: Could not determine diffusion model hidden dim, using default: 1024"
            )
            return 1024

    def _parse_alignment_depths(self) -> List[int]:
        """Parse alignment depth specification."""
        depth_spec = getattr(self.args, "repa_alignment_depth", 8)

        if isinstance(depth_spec, (int, str)):
            # Single depth
            return [int(depth_spec)]
        elif isinstance(depth_spec, (list, tuple)):
            # Multiple depths
            return [int(d) for d in depth_spec]
        else:
            # Parse string like "8,12,16"
            try:
                return [int(d.strip()) for d in str(depth_spec).split(",")]
            except ValueError:
                logger.warning(
                    f"Invalid depth specification: {depth_spec}, using default [8]"
                )
                return [8]

    def _setup_preprocessing(self):
        """Setup preprocessing transforms for each encoder type."""
        self.preprocessing_transforms = []

        for encoder_type in self.encoder_types:
            if "clip" in encoder_type:
                transform = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
            else:
                # Default to ImageNet normalization for DINOv2, etc.
                transform = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            self.preprocessing_transforms.append(transform)

    def _get_hook(self, layer_idx: int):
        """Closure to capture the hidden state from the target layer."""

        def hook(model, input, output):
            # The output of a transformer block is typically a tuple (hidden_state, ...).
            # We are interested in the first element.
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        """Finds the target layers and attaches forward hooks."""
        try:
            for i, depth in enumerate(self.alignment_depths):
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

                hook_handle = target_module.register_forward_hook(self._get_hook(i))
                self.hook_handles.append(hook_handle)
                logger.info(
                    f"REPA: Hook attached to layer {depth} (index {i}) of the diffusion model."
                )

        except Exception as e:
            logger.error(
                f"REPA ERROR: Could not attach hooks to layers {self.alignment_depths}. Error: {e}"
            )
            raise

    def remove_hooks(self) -> None:
        """Removes all forward hooks."""
        for handle in self.hook_handles:
            if handle:
                handle.remove()
        self.hook_handles.clear()
        logger.info("REPA: All hooks removed successfully.")

    def get_repa_loss(
        self, clean_pixels: torch.Tensor, vae: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Calculates the enhanced REPA loss for a given batch.

        Args:
            clean_pixels: Clean pixel values in range [-1, 1], shape (B, C, H, W) or (B, C, F, H, W) for video
            vae: VAE model (not used in current implementation but kept for compatibility)

        Returns:
            REPA loss tensor
        """
        # Check if we have captured features from any layer
        if not any(feat is not None for feat in self.captured_features):
            return torch.tensor(0.0, device=clean_pixels.device)

        # If clean_pixels is a video batch (B, C, F, H, W), take the first frame
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]  # Take the first frame

        total_loss = torch.tensor(0.0, device=clean_pixels.device)

        with torch.no_grad():
            # Convert pixels from [-1, 1] to [0, 1] and then to [0, 255]
            images = ((clean_pixels + 1) / 2.0).clamp(0, 1) * 255.0

            # Get target representations from all encoders
            target_features_list = []
            for i, (encoder, encoder_type) in enumerate(
                zip(self.encoders, self.encoder_types)
            ):
                # Apply encoder-specific preprocessing
                processed_images = preprocess_raw_image(images, encoder_type)

                # Get features from encoder
                target_features = encoder.forward_features(processed_images)

                # Handle different output formats
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
                                f"Could not find suitable features in encoder {i} output: {target_features.keys()}"
                            )

                # Ensure target_features is a tensor
                if not isinstance(target_features, torch.Tensor):
                    raise ValueError(
                        f"Expected tensor output from encoder {i}, got {type(target_features)}"
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

                target_features_list.append(target_features)

        # Compute loss for each alignment layer and encoder combination
        num_valid_layers = sum(1 for feat in self.captured_features if feat is not None)

        for layer_idx, diffusion_features in enumerate(self.captured_features):
            if diffusion_features is None:
                continue

            # Project diffusion features for all encoders
            projected_features_list = self.projection_heads(diffusion_features)

            # Compute loss against each encoder's target features
            layer_loss = torch.tensor(0.0, device=clean_pixels.device)

            for encoder_idx, (projected_features, target_features) in enumerate(
                zip(projected_features_list, target_features_list)
            ):
                # Ensure both tensors have compatible shapes
                if projected_features.shape != target_features.shape:
                    # Match shapes by taking mean over spatial dimensions if needed
                    if projected_features.dim() == 3 and target_features.dim() == 3:
                        if projected_features.shape[1] != target_features.shape[1]:
                            projected_features = projected_features.mean(dim=1)
                            target_features = target_features.mean(dim=1)
                    elif projected_features.dim() == 2 and target_features.dim() == 3:
                        target_features = target_features.mean(dim=1)
                    elif projected_features.dim() == 3 and target_features.dim() == 2:
                        projected_features = projected_features.mean(dim=1)

                # Calculate similarity loss
                similarity_fn = getattr(self.args, "repa_similarity_fn", "cosine")

                if similarity_fn == "cosine":
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
                    encoder_loss = (
                        -similarity.mean()
                    )  # Negative because we want to maximize similarity

                elif similarity_fn == "mse":
                    # Mean squared error between projected and target features
                    encoder_loss = F.mse_loss(projected_features, target_features)

                else:
                    raise NotImplementedError(
                        f"REPA similarity function '{similarity_fn}' not implemented."
                    )

                layer_loss += encoder_loss

            # Average over encoders for this layer
            layer_loss = layer_loss / len(self.encoders)
            total_loss += layer_loss

        # Average over alignment layers
        if num_valid_layers > 0:
            total_loss = total_loss / num_valid_layers

        # Clear captured features for the next step
        self.captured_features = [None] * len(self.alignment_depths)

        # Apply loss weight
        loss_lambda = getattr(self.args, "repa_loss_lambda", 0.5)
        return total_loss * loss_lambda

    def get_repa_metrics(
        self, clean_pixels: torch.Tensor, vae: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates detailed REPA metrics for logging and analysis.

        Args:
            clean_pixels: Clean pixel values in range [-1, 1], shape (B, C, H, W) or (B, C, F, H, W) for video
            vae: VAE model (not used in current implementation but kept for compatibility)

        Returns:
            Dictionary containing detailed REPA metrics
        """
        # Check if we have captured features from any layer
        if not any(feat is not None for feat in self.captured_features):
            return {
                "repa/total_loss": torch.tensor(0.0, device=clean_pixels.device),
                "repa/num_valid_layers": torch.tensor(0.0, device=clean_pixels.device),
            }

        # If clean_pixels is a video batch (B, C, F, H, W), take the first frame
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]  # Take the first frame

        total_loss = torch.tensor(0.0, device=clean_pixels.device)
        metrics = {}

        with torch.no_grad():
            # Convert pixels from [-1, 1] to [0, 1] and then to [0, 255]
            images = ((clean_pixels + 1) / 2.0).clamp(0, 1) * 255.0

            # Get target representations from all encoders
            target_features_list = []
            for i, (encoder, encoder_type) in enumerate(
                zip(self.encoders, self.encoder_types)
            ):
                # Apply encoder-specific preprocessing
                processed_images = preprocess_raw_image(images, encoder_type)

                # Get features from encoder
                target_features = encoder.forward_features(processed_images)

                # Handle different output formats
                if isinstance(target_features, dict):
                    if "x_norm_patchtokens" in target_features:
                        target_features = target_features["x_norm_patchtokens"]
                    elif "x_norm_clstoken" in target_features:
                        target_features = target_features["x_norm_clstoken"]
                    else:
                        # Try to find any patch-like features
                        for key in target_features.keys():
                            if "patch" in key.lower() or "token" in key.lower():
                                target_features = target_features[key]
                                break
                        else:
                            target_features = target_features[list(target_features.keys())[0]]

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

                target_features_list.append(target_features)

                # Log encoder-specific metrics
                metrics[f"repa/encoder_{encoder_type}/feature_norm"] = torch.norm(target_features).mean()
                metrics[f"repa/encoder_{encoder_type}/feature_mean"] = target_features.mean()
                metrics[f"repa/encoder_{encoder_type}/feature_std"] = target_features.std()

        # Compute loss for each alignment layer and encoder combination
        num_valid_layers = sum(1 for feat in self.captured_features if feat is not None)
        layer_losses = []

        for layer_idx, diffusion_features in enumerate(self.captured_features):
            if diffusion_features is None:
                continue

            # Project diffusion features for all encoders
            projected_features_list = self.projection_heads(diffusion_features)

            # Compute loss against each encoder's target features
            layer_loss = torch.tensor(0.0, device=clean_pixels.device)
            encoder_losses = []

            for encoder_idx, (projected_features, target_features) in enumerate(
                zip(projected_features_list, target_features_list)
            ):
                encoder_type = self.encoder_types[encoder_idx]

                # Ensure both tensors have compatible shapes
                if projected_features.shape != target_features.shape:
                    # Match shapes by taking mean over spatial dimensions if needed
                    if projected_features.dim() == 3 and target_features.dim() == 3:
                        if projected_features.shape[1] != target_features.shape[1]:
                            projected_features = projected_features.mean(dim=1)
                            target_features = target_features.mean(dim=1)
                    elif projected_features.dim() == 2 and target_features.dim() == 3:
                        target_features = target_features.mean(dim=1)
                    elif projected_features.dim() == 3 and target_features.dim() == 2:
                        projected_features = projected_features.mean(dim=1)

                # Calculate similarity loss
                similarity_fn = getattr(self.args, "repa_similarity_fn", "cosine")

                if similarity_fn == "cosine":
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
                    encoder_loss = (
                        -similarity.mean()
                    )  # Negative because we want to maximize similarity

                    # Log detailed similarity metrics
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_mean"] = similarity.mean()
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_std"] = similarity.std()
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_min"] = similarity.min()
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_max"] = similarity.max()

                elif similarity_fn == "mse":
                    # Mean squared error between projected and target features
                    encoder_loss = F.mse_loss(projected_features, target_features)

                    # Log MSE metrics
                    mse_per_sample = F.mse_loss(projected_features, target_features, reduction='none')
                    mse_per_sample = mse_per_sample.mean(dim=tuple(range(1, mse_per_sample.dim())))
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/mse_mean"] = mse_per_sample.mean()
                    metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/mse_std"] = mse_per_sample.std()

                else:
                    raise NotImplementedError(
                        f"REPA similarity function '{similarity_fn}' not implemented."
                    )

                layer_loss += encoder_loss
                encoder_losses.append(encoder_loss)

                # Log per-encoder loss for this layer
                metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/loss"] = encoder_loss

            # Average over encoders for this layer
            layer_loss = layer_loss / len(self.encoders)
            layer_losses.append(layer_loss)
            total_loss += layer_loss

            # Log layer-specific metrics
            metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/loss"] = layer_loss
            metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_loss_mean"] = torch.stack(encoder_losses).mean()
            metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_loss_std"] = torch.stack(encoder_losses).std()

        # Average over alignment layers
        if num_valid_layers > 0:
            total_loss = total_loss / num_valid_layers

        # Clear captured features for the next step
        self.captured_features = [None] * len(self.alignment_depths)

        # Apply loss weight
        loss_lambda = getattr(self.args, "repa_loss_lambda", 0.5)
        weighted_total_loss = total_loss * loss_lambda

        # Add summary metrics
        metrics["repa/total_loss"] = weighted_total_loss
        metrics["repa/unweighted_loss"] = total_loss
        metrics["repa/num_valid_layers"] = torch.tensor(float(num_valid_layers), device=clean_pixels.device)
        metrics["repa/loss_lambda"] = torch.tensor(loss_lambda, device=clean_pixels.device)

        if layer_losses:
            metrics["repa/layer_loss_mean"] = torch.stack(layer_losses).mean()
            metrics["repa/layer_loss_std"] = torch.stack(layer_losses).std()

        return metrics
