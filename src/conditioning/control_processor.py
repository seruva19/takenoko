## Based on: https://github.com/spacepxl/WanTraining/blob/main/train_wan_lora.py (Apache)

"""
Control signal processor for LoRA training.
Handles different types of control signals like tile, canny, depth, etc.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Optional, Dict, Any, Tuple
import numpy as np
import random

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class ControlSignalProcessor:
    """
    Processor for different types of control signals.
    Supports tile, canny, depth, and other control types.
    Aligned with reference implementation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.control_type = self.config.get("control_lora_type", "tile")
        self.preprocessing = self.config.get("control_preprocessing", "blur")
        self.blur_kernel_size = self.config.get("control_blur_kernel_size", 15)
        self.blur_sigma = self.config.get("control_blur_sigma", 4.0)
        self.scale_factor = self.config.get("control_scale_factor", 1.0)
        self.concatenation_dim = self.config.get("control_concatenation_dim", -2)

        logger.info(
            f"Control signal processor initialized with type: {self.control_type}"
        )

    def process_control_signal(
        self,
        control_signal: torch.Tensor,
        target_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process control signal according to the specified type and preprocessing.
        Aligned with reference implementation.

        Args:
            control_signal: Input control signal tensor
            target_shape: Target shape (frames, height, width)
            device: Target device
            dtype: Target dtype

        Returns:
            Processed control signal tensor
        """
        if self.control_type == "tile":
            return self._process_tile_control(
                control_signal, target_shape, device, dtype
            )
        elif self.control_type == "canny":
            return self._process_canny_control(
                control_signal, target_shape, device, dtype
            )
        elif self.control_type == "depth":
            return self._process_depth_control(
                control_signal, target_shape, device, dtype
            )
        else:
            logger.warning(
                f"Unknown control type: {self.control_type}, using default processing"
            )
            return self._process_default_control(
                control_signal, target_shape, device, dtype
            )

    def _process_tile_control(
        self,
        control_signal: torch.Tensor,
        target_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process tile control signal with blur preprocessing.
        This is typically used for upscaling tasks.
        Aligned with reference implementation.
        """
        # Ensure control signal is in the right format (CFHW -> BFCHW)
        if control_signal.dim() == 4:  # B, C, H, W
            control_signal = control_signal.movedim(0, 1).unsqueeze(0)  # CFHW -> BFCHW
        elif control_signal.dim() == 5:  # B, C, F, H, W
            control_signal = control_signal.movedim(1, 2)  # BCFHW -> BFCHW

        # Apply preprocessing with random sigma for better generalization
        if self.preprocessing == "blur":
            height, width = control_signal.shape[-2:]

            # Use random sigma like in reference implementation
            sigma = random.uniform(3, 6) if self.blur_sigma == 4.0 else self.blur_sigma

            blur = v2.Compose(
                [
                    v2.Resize(size=(height // 4, width // 4)),
                    v2.Resize(size=(height, width)),
                    v2.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=sigma),
                ]
            )

            control_signal = torch.clamp(
                torch.nan_to_num(blur(control_signal)), min=-1, max=1
            )
            control_signal = control_signal[0].movedim(0, 1)  # BFCHW -> CFHW

        # Resize to target shape if needed
        frames, height, width = target_shape
        if control_signal.shape[-2:] != (height, width):
            control_signal = self._resize_control_signal(
                control_signal, (height, width)
            )

        # Apply scale factor
        control_signal = control_signal * self.scale_factor

        return control_signal.to(device=device, dtype=dtype)

    def _process_canny_control(
        self,
        control_signal: torch.Tensor,
        target_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process canny edge control signal.
        """
        # Convert to grayscale if needed
        if control_signal.shape[1] == 3:  # RGB
            control_signal = self._rgb_to_grayscale(control_signal)

        # Apply canny edge detection
        control_signal = self._apply_canny_edge_detection(control_signal)

        # Resize to target shape
        frames, height, width = target_shape
        control_signal = self._resize_control_signal(control_signal, (height, width))

        # Normalize to [0, 1] range for edge maps
        control_signal = torch.clamp(control_signal, min=0, max=1)

        # Apply scale factor
        control_signal = control_signal * self.scale_factor

        return control_signal.to(device=device, dtype=dtype)

    def _process_depth_control(
        self,
        control_signal: torch.Tensor,
        target_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process depth control signal.
        """
        # Convert to grayscale if needed
        if control_signal.shape[1] == 3:  # RGB
            control_signal = self._rgb_to_grayscale(control_signal)

        # Resize to target shape
        frames, height, width = target_shape
        control_signal = self._resize_control_signal(control_signal, (height, width))

        # Normalize to [0, 1] range for depth maps
        control_signal = torch.clamp(control_signal, min=0, max=1)

        # Apply scale factor
        control_signal = control_signal * self.scale_factor

        return control_signal.to(device=device, dtype=dtype)

    def _process_default_control(
        self,
        control_signal: torch.Tensor,
        target_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Default control signal processing.
        """
        # Resize to target shape
        frames, height, width = target_shape
        control_signal = self._resize_control_signal(control_signal, (height, width))

        # Normalize to [-1, 1] range
        control_signal = control_signal * 2 - 1
        control_signal = torch.clamp(torch.nan_to_num(control_signal), min=-1, max=1)

        # Apply scale factor
        control_signal = control_signal * self.scale_factor

        return control_signal.to(device=device, dtype=dtype)

    def _apply_blur_preprocessing(self, control_signal: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur preprocessing."""
        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.GaussianBlur(
                    kernel_size=self.blur_kernel_size, sigma=self.blur_sigma
                ),
            ]
        )

        # Apply transform to each frame
        if control_signal.dim() == 5:  # B, C, F, H, W
            B, C, F, H, W = control_signal.shape
            control_signal = control_signal.view(B * F, C, H, W)
            control_signal = transform(control_signal)
            control_signal = control_signal.view(B, F, C, H, W)
        else:
            control_signal = transform(control_signal)

        return control_signal

    def _apply_canny_edge_detection(self, control_signal: torch.Tensor) -> torch.Tensor:
        """Apply Canny edge detection to control signal."""
        # Convert to numpy for OpenCV processing
        if control_signal.dim() == 5:  # B, C, F, H, W
            B, C, F, H, W = control_signal.shape
            control_signal = control_signal.view(B * F, C, H, W)

            # Process each frame
            processed_frames = []
            for i in range(control_signal.shape[0]):
                frame = control_signal[i].cpu().numpy().transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)

                # Apply edge detection
                edge_frame = self._simple_edge_detection(frame)
                processed_frames.append(torch.from_numpy(edge_frame).float() / 255.0)

            control_signal = torch.stack(processed_frames).view(B, F, C, H, W)
        else:
            # Single frame processing
            frame = control_signal.cpu().numpy().transpose(1, 2, 0)
            frame = (frame * 255).astype(np.uint8)
            edge_frame = self._simple_edge_detection(frame)
            control_signal = torch.from_numpy(edge_frame).float() / 255.0

        return control_signal

    def _simple_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Simple edge detection using Sobel operators."""
        if len(frame.shape) == 3:
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = frame

        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply convolution
        grad_x = self._convolve2d(gray, sobel_x)
        grad_y = self._convolve2d(gray, sobel_y)

        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255)

        return magnitude

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution."""
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape

        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1

        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(
                    image[i : i + kernel_height, j : j + kernel_width] * kernel
                )

        return output

    def _rgb_to_grayscale(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Convert RGB tensor to grayscale."""
        # Use standard RGB to grayscale conversion weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device)
        grayscale = torch.sum(
            rgb_tensor * weights.view(1, 3, 1, 1), dim=1, keepdim=True
        )
        return grayscale

    def _resize_control_signal(
        self, control_signal: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize control signal to target size."""
        height, width = target_size

        if control_signal.dim() == 5:  # B, C, F, H, W
            B, C, F, H, W = control_signal.shape
            control_signal = control_signal.view(B * F, C, H, W)
            control_signal = torch.nn.functional.interpolate(
                control_signal,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            control_signal = control_signal.view(B, F, C, height, width)
        else:
            control_signal = torch.nn.functional.interpolate(
                control_signal,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        return control_signal

    def concatenate_with_latents(
        self, control_latents: torch.Tensor, noisy_latents: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate control latents with noisy latents.

        Behavior:
        - If tensors are CFHW (4D), default concat on channel dim (0)
        - If tensors are BCFHW (5D), default concat on channel dim (1)
        - If config contains `control_concatenation_dim`, use it when valid.

        Args:
            control_latents: Control tensor (4D CFHW or 5D BCFHW)
            noisy_latents: Noisy input tensor (same shape layout as control_latents)

        Returns:
            Concatenated tensor along the appropriate channel dimension
        """
        # Ensure both tensors are on the same device and dtype
        device = noisy_latents.device
        dtype = noisy_latents.dtype
        control_latents = control_latents.to(device=device, dtype=dtype)

        ndim = noisy_latents.dim()
        configured_dim = getattr(self, "concatenation_dim", None)
        # Determine concat dim with robust defaults
        if configured_dim is None:
            concat_dim = 1 if ndim == 5 else 0
        else:
            # Common mapping: CFHW channel dim(0) roughly corresponds to BCFHW dim(1)
            if ndim == 5 and configured_dim in (0, -2):
                concat_dim = 1
            elif isinstance(configured_dim, int) and -ndim <= configured_dim < ndim:
                # Normalize negative dims
                concat_dim = configured_dim % ndim
            else:
                concat_dim = 1 if ndim == 5 else 0

        return torch.cat([noisy_latents, control_latents], dim=concat_dim)

    def inject_noise(
        self, control_latents: torch.Tensor, noise_strength: float
    ) -> torch.Tensor:
        """
        Inject random noise into control latents.
        Aligned with reference implementation.
        """
        if noise_strength <= 0:
            return control_latents

        # Generate random noise strength for each sample
        inject_strength = torch.rand(1).item() * noise_strength
        noise = torch.randn_like(control_latents) * inject_strength

        return control_latents + noise


def create_control_processor(
    config: Optional[Dict[str, Any]] = None,
) -> ControlSignalProcessor:
    """
    Create a control signal processor with the given configuration.
    """
    return ControlSignalProcessor(config)
