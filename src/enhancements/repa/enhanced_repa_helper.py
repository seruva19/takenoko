import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, Any, Union, List, Dict, Tuple
import logging
import math
from common.logger import get_logger

from enhancements.repa.encoder_manager import (
    EncoderManager,
    EnhancedEncoder,
    preprocess_raw_image,
)
from enhancements.repa.stable_velocity_weighting import (
    align_weights_to_batch,
    compute_stable_velocity_weights,
    normalize_timesteps,
    weighted_mean,
)

logger = get_logger(__name__, level=logging.INFO)


def interpolate_features_spatial(
    features: torch.Tensor, target_tokens: int
) -> torch.Tensor:
    """
    Spatially interpolate token features to match target token count.

    Assumes tokens are arranged in a square grid (sqrt(N) x sqrt(N)).

    Args:
        features: (B, N_tokens, D) tensor
        target_tokens: Target number of tokens

    Returns:
        Interpolated features with shape (B, target_tokens, D)
    """
    B, N, D = features.shape

    # Compute spatial dimensions (assume square)
    src_size = int(math.sqrt(N))
    tgt_size = int(math.sqrt(target_tokens))

    if src_size * src_size != N or tgt_size * tgt_size != target_tokens:
        # Non-square token counts, fall back to 1D interpolation
        features_1d = features.permute(0, 2, 1)  # (B, D, N)
        interpolated = F.interpolate(
            features_1d, size=target_tokens, mode="linear", align_corners=False
        )
        return interpolated.permute(0, 2, 1)  # (B, target_tokens, D)

    # Reshape to 2D spatial grid: (B, N, D) -> (B, D, H, W)
    features_2d = features.permute(0, 2, 1).view(B, D, src_size, src_size)

    # Bilinear interpolation to target size
    interpolated_2d = F.interpolate(
        features_2d, size=(tgt_size, tgt_size), mode="bilinear", align_corners=False
    )

    # Reshape back: (B, D, H', W') -> (B, N', D)
    return interpolated_2d.view(B, D, -1).permute(0, 2, 1)


class MultiEncoderProjectionHead(nn.Module):
    """Multi-encoder projection head that handles ensemble of encoders."""

    def __init__(
        self,
        diffusion_hidden_dim: int,
        encoder_dims: List[int],
        ensemble_mode: str = "individual",
        shared_projection: bool = False,
        projection_type: str = "mlp",
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.diffusion_hidden_dim = diffusion_hidden_dim
        self.encoder_dims = encoder_dims
        self.ensemble_mode = ensemble_mode
        self.shared_projection = shared_projection
        self.projection_type = projection_type
        self.conv_kernel = conv_kernel
        self._conv_warning_logged = False

        if shared_projection:
            # Single shared projection head for all encoders
            # Project to average encoder dimension
            avg_dim = int(sum(encoder_dims) / len(encoder_dims))
            self.projection = self._create_projection_head(
                diffusion_hidden_dim, avg_dim
            )
            self.fallback_linear = (
                nn.Linear(diffusion_hidden_dim, avg_dim)
                if self.projection_type == "conv"
                else None
            )
            self.projections = None
            self.fallback_linears = None
        else:
            # Individual projection heads for each encoder
            self.projections = nn.ModuleList(
                [
                    self._create_projection_head(diffusion_hidden_dim, dim)
                    for dim in encoder_dims
                ]
            )
            self.fallback_linears = (
                nn.ModuleList(
                    [nn.Linear(diffusion_hidden_dim, dim) for dim in encoder_dims]
                )
                if self.projection_type == "conv"
                else None
            )
            self.projection = None
            self.fallback_linear = None

    def _create_projection_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """Creates a projection head (MLP or conv) for REPA/iREPA."""
        if self.projection_type == "conv":
            padding = self.conv_kernel // 2
            return nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=self.conv_kernel,
                padding=padding,
            )
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.SiLU(),
            nn.Linear(input_dim * 4, output_dim),
        )

    def _project_with_layer(
        self,
        features: torch.Tensor,
        proj_layer: nn.Module,
        fallback_layer: Optional[nn.Module],
    ) -> torch.Tensor:
        """
        Apply either conv or MLP projection. Conv expects a square token grid; falls back to linear if not.
        """
        B, tokens, dim = features.shape
        if self.projection_type == "conv":
            side = int(math.isqrt(tokens))
            if side * side == tokens:
                x = features.view(B, side, side, dim).permute(0, 3, 1, 2).contiguous()
                y = proj_layer(x)
                return y.permute(0, 2, 3, 1).contiguous().view(B, tokens, -1)
            if not self._conv_warning_logged:
                logger.warning(
                    "iREPA conv projection falling back to linear: token grid is not square (tokens=%d).",
                    tokens,
                )
                self._conv_warning_logged = True
        flat = features.view(B * tokens, dim)
        if fallback_layer is not None:
            projected = fallback_layer(flat)
            return projected.view(B, tokens, -1)
        # Fallback to identity if no projection is available (should not happen for MLP path)
        return proj_layer(flat).view(B, tokens, -1)

    def forward(self, diffusion_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Project diffusion features to match each encoder's feature space.

        Args:
            diffusion_features: Hidden states from diffusion model

        Returns:
            List of projected features for each encoder
        """
        if self.projection_type == "conv":
            if self.shared_projection:
                projected = self._project_with_layer(
                    diffusion_features, self.projection, self.fallback_linear
                )
                return [projected for _ in self.encoder_dims]
            return [
                self._project_with_layer(
                    diffusion_features,
                    proj_layer,
                    self.fallback_linears[idx] if self.fallback_linears else None,
                )
                for idx, proj_layer in enumerate(self.projections)
            ]

        if self.shared_projection:
            projected = self.projection(diffusion_features)
            return [projected for _ in self.encoder_dims]
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

        # iREPA settings (conv projector + spatial normalization)
        self.irepa_enabled = bool(getattr(args, "enable_irepa", False))
        self.irepa_projection_type = getattr(args, "irepa_projection_type", "conv")
        self.irepa_proj_kernel = int(getattr(args, "irepa_proj_kernel", 3))
        self.irepa_spatial_norm = getattr(args, "irepa_spatial_norm", "zscore")
        self.irepa_zscore_alpha = float(getattr(args, "irepa_zscore_alpha", 1.0))

        # Create multi-encoder projection heads
        ensemble_mode = getattr(args, "repa_ensemble_mode", "individual")
        shared_proj = getattr(args, "repa_shared_projection", False)

        self.projection_heads = MultiEncoderProjectionHead(
            self.diffusion_hidden_dim,
            self.encoder_dims,
            ensemble_mode=ensemble_mode,
            shared_projection=shared_proj,
            projection_type=(
                self.irepa_projection_type if self.irepa_enabled else "mlp"
            ),
            conv_kernel=self.irepa_proj_kernel,
        )

        # Setup preprocessing transforms for each encoder type
        self._setup_preprocessing()

        # Multi-layer alignment support
        self.alignment_depths = self._parse_alignment_depths()

        # CREPA settings
        self.crepa_enabled = getattr(args, "crepa_enabled", False)
        self.crepa_adjacency = getattr(args, "crepa_adjacency", 1)
        self.crepa_temperature = getattr(args, "crepa_temperature", 1.0)
        self.crepa_normalize_by_frames = getattr(
            args, "crepa_normalize_by_frames", True
        )

        # Spatial alignment settings
        self.spatial_align = getattr(args, "repa_spatial_align", True)
        self.stable_velocity_enabled = bool(
            getattr(args, "enable_stable_velocity", False)
            and getattr(args, "stable_velocity_repa_enabled", True)
        )
        self.stable_velocity_repa_weight_schedule = str(
            getattr(args, "stable_velocity_repa_weight_schedule", "sigmoid")
        ).lower()
        self.stable_velocity_repa_tau = float(
            getattr(args, "stable_velocity_repa_tau", 0.7)
        )
        self.stable_velocity_repa_k = float(
            getattr(args, "stable_velocity_repa_k", 20.0)
        )
        self.stable_velocity_repa_path_type = str(
            getattr(args, "stable_velocity_repa_path_type", "linear")
        ).lower()
        self.stable_velocity_repa_min_weight = float(
            getattr(args, "stable_velocity_repa_min_weight", 0.0)
        )
        self.stable_velocity_repa_log_interval = int(
            getattr(args, "stable_velocity_repa_log_interval", 100)
        )
        self.last_stable_velocity_metrics: Dict[str, torch.Tensor] = {}

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
        if self.stable_velocity_enabled:
            logger.info(
                "REPA: StableVelocity weighting enabled (schedule=%s, tau=%.3f, k=%.3f, path=%s, min_weight=%.3f).",
                self.stable_velocity_repa_weight_schedule,
                self.stable_velocity_repa_tau,
                self.stable_velocity_repa_k,
                self.stable_velocity_repa_path_type,
                self.stable_velocity_repa_min_weight,
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

    def _apply_spatial_norm(self, features: torch.Tensor) -> torch.Tensor:
        """Optional spatial normalization (iREPA) over patch tokens."""
        if not self.irepa_enabled or self.irepa_spatial_norm == "none":
            return features
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        return (features - self.irepa_zscore_alpha * mean) / (std + 1e-6)

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

    def _get_stable_velocity_weights(
        self,
        raw_timesteps: Any,
        device: torch.device,
        sample_count: int,
    ) -> Optional[torch.Tensor]:
        self.last_stable_velocity_metrics = {}
        if (
            not self.stable_velocity_enabled
            or not torch.is_tensor(raw_timesteps)
            or sample_count <= 0
        ):
            return None

        t_norm = normalize_timesteps(
            raw_timesteps,
            float(getattr(self.args, "max_timestep", 1000.0) or 1000.0),
        )
        weights = compute_stable_velocity_weights(
            t_norm=t_norm,
            schedule=self.stable_velocity_repa_weight_schedule,
            tau=self.stable_velocity_repa_tau,
            k=self.stable_velocity_repa_k,
            path_type=self.stable_velocity_repa_path_type,
            min_weight=self.stable_velocity_repa_min_weight,
        )
        weights = align_weights_to_batch(weights, sample_count).to(
            device=device, dtype=torch.float32
        )
        self.last_stable_velocity_metrics = {
            "stable_velocity_weight_mean": weights.mean().detach(),
            "stable_velocity_active_ratio": (weights > 1e-6).float().mean().detach(),
        }
        return weights

    def get_repa_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
        **kwargs: Any,
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
        self.last_stable_velocity_metrics = {}

        total_loss = torch.tensor(0.0, device=clean_pixels.device)

        # Determine if we are in video mode with CREPA enabled
        is_video = clean_pixels.dim() == 5
        use_crepa = is_video and self.crepa_enabled

        if is_video and not use_crepa:
            # Legacy behavior: Take the first frame for standard REPA on video
            # This avoids massive compute overhead if user didn't ask for full video alignment
            clean_pixels = clean_pixels[:, :, 0, :, :]

        # Helper to process features
        # If video (B, C, F, H, W), we flatten to (B*F, C, H, W) for encoder
        if is_video and use_crepa:
            B, C, F_frames, H, W = clean_pixels.shape
            # Flatten frames into batch dimension
            images_input = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
                B * F_frames, C, H, W
            )
        else:
            images_input = clean_pixels
            B = clean_pixels.shape[0]
            F_frames = 1

        sample_weights = self._get_stable_velocity_weights(
            kwargs.get("timesteps"),
            clean_pixels.device,
            B,
        )
        if sample_weights is not None and float(sample_weights.sum().item()) <= 1e-8:
            self.captured_features = [None] * len(self.alignment_depths)
            return torch.tensor(0.0, device=clean_pixels.device)

        with torch.no_grad():
            # Convert pixels from [-1, 1] to [0, 1] and then to [0, 255]
            images = ((images_input + 1) / 2.0).clamp(0, 1) * 255.0

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

                # Ensure we have the right shape: (N_samples, N_tokens, D)
                if target_features.dim() == 2:
                    target_features = target_features.unsqueeze(1)
                elif target_features.dim() > 3:
                    # If we got (N_samples, C, H, W), flatten spatial dimensions
                    N_s, C_feat, H_feat, W_feat = target_features.shape
                    target_features = target_features.view(
                        N_s, C_feat, H_feat * W_feat
                    ).transpose(1, 2)

                target_features = self._apply_spatial_norm(target_features)

                # For CREPA, we generally want global frame representations (or at least aligned structure)
                # Standard REPA usually pools if shapes don't match.
                # We'll handle pooling in the loop.
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
                # Handle CREPA reshaping if needed
                if is_video and use_crepa:
                    # diffusion_features / projected_features are likely (B, SeqLen, D)
                    # We need to reshape to (B, F, TokensPerFrame, D) or (B*F, TokensPerFrame, D)
                    # We assume SeqLen is divisible by F

                    # projected_features: (B, SeqLen, D_proj)
                    # target_features: (B*F, N_enc_tokens, D_enc)

                    # First, check if we need to pool target features (common in REPA)
                    # If shapes mismatch, we usually global pool.

                    # Reshape projected to match target batch-wise
                    # (B, SeqLen, D) -> (B, F, Tokens, D) -> (B*F, Tokens, D)
                    if (
                        projected_features.shape[0] == B
                        and projected_features.dim() == 3
                    ):
                        seq_len = projected_features.shape[1]
                        if seq_len % F_frames != 0:
                            # Fallback: cannot align frames, treat as one blob (might fail)
                            pass
                        else:
                            tokens_per_frame = seq_len // F_frames
                            projected_features = projected_features.view(
                                B, F_frames, tokens_per_frame, -1
                            )
                            projected_features = projected_features.view(
                                B * F_frames, tokens_per_frame, -1
                            )

                # Official REPA computes per-patch similarity, then averages.
                # projected: (N_samples, N_proj_tokens, D)
                # target: (N_samples, N_enc_tokens, D)
                # If token counts differ, we can either:
                # 1. Spatially interpolate to match (preserves spatial alignment)
                # 2. Pool to (N_samples, D) (fallback)

                similarity_fn = getattr(self.args, "repa_similarity_fn", "cosine")

                if similarity_fn == "cosine":
                    # Check if shapes match
                    shapes_match = (
                        projected_features.shape == target_features.shape
                        and projected_features.dim() == target_features.dim()
                    )

                    if (
                        not shapes_match
                        and projected_features.dim() == 3
                        and target_features.dim() == 3
                    ):
                        # Token counts differ - try spatial interpolation if enabled
                        if self.spatial_align:
                            # Interpolate to the smaller token count (usually encoder's)
                            proj_tokens = projected_features.shape[1]
                            tgt_tokens = target_features.shape[1]

                            if proj_tokens != tgt_tokens:
                                # Interpolate the larger one to match the smaller
                                if proj_tokens > tgt_tokens:
                                    projected_features = interpolate_features_spatial(
                                        projected_features, tgt_tokens
                                    )
                                else:
                                    target_features = interpolate_features_spatial(
                                        target_features, proj_tokens
                                    )
                            shapes_match = True  # Now they match

                    if not shapes_match:
                        # Fallback: Pool to (N_samples, D)
                        if projected_features.dim() == 3:
                            projected_features = projected_features.mean(dim=1)
                        if target_features.dim() == 3:
                            target_features = target_features.mean(dim=1)

                        projected_norm = F.normalize(projected_features, dim=-1)
                        target_norm = F.normalize(target_features, dim=-1)

                        if use_crepa and is_video:
                            # CREPA with pooled features: (B*F, D) -> (B, F, D)
                            h_f = projected_norm.view(B, F_frames, -1)
                            y_f = target_norm.view(B, F_frames, -1)

                            # Self-frame similarity
                            self_sim = (h_f * y_f).sum(dim=-1)  # (B, F)

                            # Neighbor similarities
                            neighbor_sim = torch.zeros_like(self_sim)
                            d = self.crepa_adjacency
                            tau = self.crepa_temperature
                            weight = math.exp(-d / tau)

                            if d < F_frames:
                                left_sim = (h_f[:, d:, :] * y_f[:, :-d, :]).sum(dim=-1)
                                neighbor_sim[:, d:] += weight * left_sim
                                right_sim = (h_f[:, :-d, :] * y_f[:, d:, :]).sum(dim=-1)
                                neighbor_sim[:, :-d] += weight * right_sim

                            total_sim = self_sim + neighbor_sim
                            if self.crepa_normalize_by_frames:
                                # Normalize by frames for consistent scale across video lengths
                                per_sample_loss = -total_sim.mean(dim=1)
                            else:
                                # Sum over frames (stronger signal for longer videos)
                                per_sample_loss = -total_sim.sum(dim=1)
                            encoder_loss = weighted_mean(per_sample_loss, sample_weights)
                        else:
                            # Standard REPA (pooled)
                            similarity = (projected_norm * target_norm).sum(dim=-1)
                            per_sample_loss = -similarity
                            encoder_loss = weighted_mean(per_sample_loss, sample_weights)
                    else:
                        # Official REPA: per-patch similarity (no pooling needed)
                        # Normalize per-patch: (N_samples, N_patches, D)
                        projected_norm = F.normalize(projected_features, dim=-1)
                        target_norm = F.normalize(target_features, dim=-1)

                        # Per-patch cosine similarity: (N_samples, N_patches)
                        patch_sim = (projected_norm * target_norm).sum(dim=-1)

                        if use_crepa and is_video:
                            # CREPA: Cross-frame Representation Alignment (Eq. 6)
                            # L = -E[ Σ_f (sim(ȳ_f, h_f) + Σ_{k∈K} e^{-|k-f|/τ} · sim(ȳ_k, h_f)) ]

                            # Reshape: (B*F, N_patches, D) -> (B, F, N_patches, D)
                            N_patches = projected_norm.shape[1]
                            h_f = projected_norm.view(B, F_frames, N_patches, -1)
                            y_f = target_norm.view(B, F_frames, N_patches, -1)

                            # Self-frame: mean over patches first, then we have (B, F)
                            self_sim = (h_f * y_f).sum(dim=-1).mean(dim=-1)  # (B, F)

                            # Neighbor similarities
                            neighbor_sim = torch.zeros_like(self_sim)
                            d = self.crepa_adjacency
                            tau = self.crepa_temperature
                            weight = math.exp(-d / tau)

                            if d < F_frames:
                                # Left neighbor
                                left_sim = (
                                    (h_f[:, d:, :, :] * y_f[:, :-d, :, :])
                                    .sum(dim=-1)
                                    .mean(dim=-1)
                                )
                                neighbor_sim[:, d:] += weight * left_sim
                                # Right neighbor
                                right_sim = (
                                    (h_f[:, :-d, :, :] * y_f[:, d:, :, :])
                                    .sum(dim=-1)
                                    .mean(dim=-1)
                                )
                                neighbor_sim[:, :-d] += weight * right_sim

                            total_sim = self_sim + neighbor_sim
                            if self.crepa_normalize_by_frames:
                                # Normalize by frames for consistent scale across video lengths
                                per_sample_loss = -total_sim.mean(dim=1)
                            else:
                                # Sum over frames (stronger signal for longer videos)
                                per_sample_loss = -total_sim.sum(dim=1)
                            encoder_loss = weighted_mean(per_sample_loss, sample_weights)
                        else:
                            # Standard REPA: mean over patches, then mean over samples
                            # This matches official: mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                            per_sample_loss = -patch_sim.mean(dim=-1)
                            encoder_loss = weighted_mean(per_sample_loss, sample_weights)

                elif similarity_fn == "mse":
                    # Mean squared error
                    mse_per_sample = (projected_features - target_features).pow(2)
                    mse_per_sample = mse_per_sample.reshape(
                        mse_per_sample.shape[0], -1
                    ).mean(dim=1)
                    encoder_loss = weighted_mean(mse_per_sample, sample_weights)
                    # Note: CREPA isn't typically defined with MSE in the paper, but we could extrapolate.
                    # For now, ignoring CREPA for MSE to stay safe, or implement if requested.
                    if use_crepa:
                        logger.warning_once(
                            "CREPA is currently only implemented for cosine similarity."
                        )

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
                            target_features = target_features[
                                list(target_features.keys())[0]
                            ]

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

                target_features = self._apply_spatial_norm(target_features)

                target_features_list.append(target_features)

                # Log encoder-specific metrics
                metrics[f"repa/encoder_{encoder_type}/feature_norm"] = torch.norm(
                    target_features
                ).mean()
                metrics[f"repa/encoder_{encoder_type}/feature_mean"] = (
                    target_features.mean()
                )
                metrics[f"repa/encoder_{encoder_type}/feature_std"] = (
                    target_features.std()
                )

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
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_mean"
                    ] = similarity.mean()
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_std"
                    ] = similarity.std()
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_min"
                    ] = similarity.min()
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/similarity_max"
                    ] = similarity.max()

                elif similarity_fn == "mse":
                    # Mean squared error between projected and target features
                    encoder_loss = F.mse_loss(projected_features, target_features)

                    # Log MSE metrics
                    mse_per_sample = F.mse_loss(
                        projected_features, target_features, reduction="none"
                    )
                    mse_per_sample = mse_per_sample.mean(
                        dim=tuple(range(1, mse_per_sample.dim()))
                    )
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/mse_mean"
                    ] = mse_per_sample.mean()
                    metrics[
                        f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/mse_std"
                    ] = mse_per_sample.std()

                else:
                    raise NotImplementedError(
                        f"REPA similarity function '{similarity_fn}' not implemented."
                    )

                layer_loss += encoder_loss
                encoder_losses.append(encoder_loss)

                # Log per-encoder loss for this layer
                metrics[
                    f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_{encoder_type}/loss"
                ] = encoder_loss

            # Average over encoders for this layer
            layer_loss = layer_loss / len(self.encoders)
            layer_losses.append(layer_loss)
            total_loss += layer_loss

            # Log layer-specific metrics
            metrics[f"repa/layer_{self.alignment_depths[layer_idx]}/loss"] = layer_loss
            metrics[
                f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_loss_mean"
            ] = torch.stack(encoder_losses).mean()
            metrics[
                f"repa/layer_{self.alignment_depths[layer_idx]}/encoder_loss_std"
            ] = torch.stack(encoder_losses).std()

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
        metrics["repa/num_valid_layers"] = torch.tensor(
            float(num_valid_layers), device=clean_pixels.device
        )
        metrics["repa/loss_lambda"] = torch.tensor(
            loss_lambda, device=clean_pixels.device
        )

        if layer_losses:
            metrics["repa/layer_loss_mean"] = torch.stack(layer_losses).mean()
            metrics["repa/layer_loss_std"] = torch.stack(layer_losses).std()

        return metrics
