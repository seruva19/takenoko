"""
FVD (Fréchet Video Distance) implementations for Takenoko.

This module provides two FVD implementations:
1. Reference FVD: Uses I3D backbone (canonical implementation)
2. Fast FVD: Uses R3D-18 backbone (speed-optimized approximation)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import logging

try:
    import scipy.linalg
except ImportError:
    scipy = None

logger = logging.getLogger(__name__)


class FVDMetric:
    """Base class for FVD metric implementations."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._backbone = None

    def _preprocess_videos(
        self, videos: torch.Tensor, target_resolution: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """Preprocess videos for the backbone model.

        Args:
            videos: [B, C, F, H, W] tensor in [0, 1] range
            target_resolution: (width, height) target resolution

        Returns:
            Preprocessed videos ready for backbone
        """
        raise NotImplementedError

    def _extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features from videos using the backbone.

        Args:
            videos: Preprocessed videos

        Returns:
            Feature tensor
        """
        raise NotImplementedError

    def compute_fvd(
        self, real_videos: torch.Tensor, generated_videos: torch.Tensor
    ) -> float:
        """Compute FVD between real and generated videos.

        Args:
            real_videos: [B, C, F, H, W] real videos in [0, 1] range
            generated_videos: [B, C, F, H, W] generated videos in [0, 1] range

        Returns:
            FVD score
        """
        raise NotImplementedError


class FastFVDMetric(FVDMetric):
    """Fast FVD implementation using R3D-18 backbone (current Takenoko approach)."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__(device, dtype)

    def _load_backbone(self):
        """Load R3D-18 backbone."""
        if self._backbone is None:
            try:
                import torchvision

                self._backbone = torchvision.models.video.r3d_18(
                    weights=torchvision.models.video.R3D_18_Weights.DEFAULT
                ).to(self.device, dtype=self.dtype)
                self._backbone.eval()
                # Strip classifier to get penultimate features
                if hasattr(self._backbone, "fc"):
                    self._backbone.fc = nn.Identity()
            except ImportError as e:
                raise ImportError(f"torchvision not available: {e}")

    def _preprocess_videos(
        self, videos: torch.Tensor, target_resolution: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """Preprocess videos for R3D-18."""
        # R3D-18 expects [B, C, T, H, W] in [0, 1] range, float32
        return videos.to(dtype=self.dtype)

    def _extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features using R3D-18."""
        self._load_backbone()
        with torch.no_grad():
            return self._backbone(videos)

    def _compute_frechet_distance(
        self, real_features: torch.Tensor, generated_features: torch.Tensor
    ) -> float:
        """Compute Fréchet distance using simplified approximation."""

        def mean_and_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x2 = x.flatten(1)
            mu = x2.mean(dim=0)
            xc = x2 - mu
            cov = (xc.T @ xc) / max(x2.size(0), 1)
            return mu, cov

        mu_real, cov_real = mean_and_cov(real_features)
        mu_gen, cov_gen = mean_and_cov(generated_features)

        # FID-style distance: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*(C1*C2)^{1/2})
        diff = mu_gen - mu_real
        diff_term = diff.dot(diff)

        # Approximate trace_sqrt using eigenvalue decomposition
        try:
            C1 = cov_gen.detach().cpu()
            C2 = cov_real.detach().cpu()

            w1, _ = np.linalg.eigh(C1.numpy())
            w2, _ = np.linalg.eigh(C2.numpy())

            # Crude approximation: trace_sqrt(C1*C2) ~ sum(sqrt(w1))*sum(sqrt(w2)) / dim
            trace_sqrt_approx = (
                np.sqrt(np.maximum(w1, 0.0)).sum()
                * np.sqrt(np.maximum(w2, 0.0)).sum()
                / max(len(w1), 1)
            )
            cov_term = float(
                C1.trace().item() + C2.trace().item() - 2.0 * trace_sqrt_approx
            )
        except Exception:
            cov_term = float(cov_gen.trace().item() + cov_real.trace().item())

        return float(diff_term + cov_term)

    def compute_fvd(
        self, real_videos: torch.Tensor, generated_videos: torch.Tensor
    ) -> float:
        """Compute fast FVD approximation."""
        real_processed = self._preprocess_videos(real_videos)
        gen_processed = self._preprocess_videos(generated_videos)

        real_features = self._extract_features(real_processed)
        gen_features = self._extract_features(gen_processed)

        return self._compute_frechet_distance(real_features, gen_features)


class ReferenceFVDMetric(FVDMetric):
    """Reference FVD implementation using I3D backbone (canonical approach)."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__(device, dtype)

    def _load_backbone(self):
        """Load I3D backbone from TensorFlow Hub."""
        if self._backbone is None:
            try:
                # pip install tensorflow tensorflow-hub tensorflow-gan scipy
                import tensorflow as tf  # type: ignore
                import tensorflow_hub as hub  # type: ignore

                # Load I3D model from TF Hub
                module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
                self._backbone = hub.load(module_spec)

                # Create a simple wrapper for PyTorch compatibility
                class I3DWrapper:
                    def __init__(self, tf_model):
                        self.tf_model = tf_model

                    def __call__(self, videos):
                        # Convert PyTorch tensor to TF tensor
                        # I3D expects [B, T, H, W, C] format
                        if videos.dim() == 5:
                            videos = videos.permute(
                                0, 2, 3, 4, 1
                            )  # [B, C, T, H, W] -> [B, T, H, W, C]

                        # Convert to TF tensor
                        tf_videos = tf.convert_to_tensor(videos.detach().cpu().numpy())

                        # Get I3D features (global average pooling)
                        features = self.tf_model(
                            tf_videos, signature="default", as_dict=True
                        )
                        return torch.from_numpy(features["default"].numpy()).to(
                            videos.device
                        )

                self._backbone = I3DWrapper(self._backbone)

            except ImportError as e:
                raise ImportError(f"TensorFlow/TF Hub not available: {e}")

    def _preprocess_videos(
        self, videos: torch.Tensor, target_resolution: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """Preprocess videos for I3D (reference implementation preprocessing)."""
        # Reference implementation preprocessing:
        # 1. Resize to target resolution
        # 2. Scale from [0, 1] to [-1, 1] range

        B, C, F, H, W = videos.shape

        # Resize if needed
        if H != target_resolution[1] or W != target_resolution[0]:
            videos = torch.nn.functional.interpolate(
                videos.view(-1, C, H, W),  # [B*F, C, H, W]
                size=target_resolution,
                mode="bilinear",
                align_corners=False,
            ).view(B, C, F, target_resolution[1], target_resolution[0])

        # Scale from [0, 1] to [-1, 1]
        videos = 2.0 * videos - 1.0

        return videos.to(dtype=self.dtype)

    def _extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features using I3D."""
        self._load_backbone()
        with torch.no_grad():
            return self._backbone(videos)

    def _compute_frechet_distance(
        self, real_features: torch.Tensor, generated_features: torch.Tensor
    ) -> float:
        """Compute exact Fréchet distance using TensorFlow GAN implementation."""
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_gan as tfgan  # type: ignore

            # Convert to TF tensors
            real_tf = tf.convert_to_tensor(real_features.detach().cpu().numpy())
            gen_tf = tf.convert_to_tensor(generated_features.detach().cpu().numpy())

            # Use TF GAN's exact Fréchet distance implementation
            fvd = tfgan.eval.frechet_classifier_distance_from_activations(
                real_tf, gen_tf
            )

            return float(fvd.numpy())

        except ImportError:
            # Fallback to numpy implementation if TF GAN not available
            logger.warning(
                "TensorFlow GAN not available, using numpy fallback for Fréchet distance"
            )
            return self._compute_frechet_distance_numpy(
                real_features, generated_features
            )

    def _compute_frechet_distance_numpy(
        self, real_features: torch.Tensor, generated_features: torch.Tensor
    ) -> float:
        """Fallback Fréchet distance implementation using numpy."""

        def compute_statistics(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            features_np = features.detach().cpu().numpy()
            mu = np.mean(features_np, axis=0)
            sigma = np.cov(features_np, rowvar=False)
            return mu, sigma

        mu_real, sigma_real = compute_statistics(real_features)
        mu_gen, sigma_gen = compute_statistics(generated_features)

        # Compute Fréchet distance: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*(C1*C2)^{1/2})
        diff = mu_gen - mu_real
        diff_term = np.dot(diff, diff)

        # Compute matrix square root term
        if scipy is not None:
            try:
                covmean = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                cov_term = np.trace(sigma_real + sigma_gen - 2 * covmean)
            except:
                # Fallback if matrix sqrt fails
                cov_term = np.trace(sigma_real + sigma_gen)
        else:
            # Fallback if scipy not available
            cov_term = np.trace(sigma_real + sigma_gen)

        return float(diff_term + cov_term)

    def compute_fvd(
        self, real_videos: torch.Tensor, generated_videos: torch.Tensor
    ) -> float:
        """Compute reference FVD using I3D."""
        real_processed = self._preprocess_videos(real_videos)
        gen_processed = self._preprocess_videos(generated_videos)

        real_features = self._extract_features(real_processed)
        gen_features = self._extract_features(gen_processed)

        return self._compute_frechet_distance(real_features, gen_features)


def create_fvd_metric(
    model_type: str, device: torch.device, dtype: torch.dtype = torch.float32
) -> FVDMetric:
    """Factory function to create FVD metric based on model type.

    Args:
        model_type: "torchvision_r3d_18" for fast FVD, "reference_i3d" for reference FVD
        device: Device to run on
        dtype: Data type

    Returns:
        FVDMetric instance
    """
    if model_type == "torchvision_r3d_18":
        return FastFVDMetric(device, dtype)
    elif model_type == "reference_i3d":
        return ReferenceFVDMetric(device, dtype)
    else:
        raise ValueError(
            f"Unknown FVD model type: {model_type}. "
            f"Supported: 'torchvision_r3d_18', 'reference_i3d'"
        )
