"""
Mask Utilities for Takenoko

This module provides utilities for creating, processing, and validating masks
for masked training in the Takenoko. Masked training allows you to train
the model only on specific parts of images or videos while ignoring other regions.

Mask Format:
- White (255) = Train on this pixel
- Black (0) = Ignore this pixel (mask it out)
- Gray values = Partial training weight (0.0 to 1.0)

Usage:
    from src.utils.mask_utils import create_center_mask, validate_mask

    # Create a center mask for 1024x1024 image
    mask = create_center_mask(1024, 1024, center_ratio=0.5)
    mask.save("center_mask.png")

    # Validate a mask
    is_valid = validate_mask("path/to/mask.png", (1024, 1024))
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_center_mask(
    width: int, height: int, center_ratio: float = 0.5
) -> Image.Image:
    """
    Create a mask that trains only on the center region of an image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        center_ratio: Ratio of center region to train on (0.0 to 1.0)

    Returns:
        PIL Image with white center region and black background

    Example:
        >>> mask = create_center_mask(1024, 1024, center_ratio=0.5)
        >>> mask.save("center_mask.png")
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Calculate center region
    center_w = int(width * center_ratio)
    center_h = int(height * center_ratio)
    start_w = (width - center_w) // 2
    start_h = (height - center_h) // 2

    # Set center region to white (255)
    mask[start_h : start_h + center_h, start_w : start_w + center_w] = 255

    return Image.fromarray(mask)


def create_gradient_mask(
    width: int, height: int, gradient_type: str = "radial"
) -> Image.Image:
    """
    Create a gradient mask with smooth transitions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        gradient_type: Type of gradient ("radial", "linear_h", "linear_v")

    Returns:
        PIL Image with gradient mask

    Example:
        >>> mask = create_gradient_mask(1024, 1024, "radial")
        >>> mask.save("gradient_mask.png")
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if gradient_type == "radial":
        # Create radial gradient from center
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                intensity = max(0, 255 * (1 - distance / max_distance))
                mask[y, x] = int(intensity)

    elif gradient_type == "linear_h":
        # Horizontal linear gradient
        for x in range(width):
            intensity = int(255 * (x / width))
            mask[:, x] = intensity

    elif gradient_type == "linear_v":
        # Vertical linear gradient
        for y in range(height):
            intensity = int(255 * (y / height))
            mask[y, :] = intensity

    return Image.fromarray(mask)


def create_selective_mask(
    width: int, height: int, regions: List[Tuple[int, int, int, int, int]]
) -> Image.Image:
    """
    Create a mask with specific regions to train on.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        regions: List of (x1, y1, x2, y2, intensity) tuples

    Returns:
        PIL Image with selective regions

    Example:
        >>> regions = [
        ...     (100, 100, 300, 300, 255),  # White region
        ...     (400, 400, 500, 500, 128),  # Gray region
        ... ]
        >>> mask = create_selective_mask(1024, 1024, regions)
        >>> mask.save("selective_mask.png")
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        x1, y1, x2, y2, intensity = region
        mask[y1:y2, x1:x2] = intensity

    return Image.fromarray(mask)


def create_border_mask(width: int, height: int, border_width: int = 50) -> Image.Image:
    """
    Create a mask that trains only on the border region.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        border_width: Width of border region in pixels

    Returns:
        PIL Image with white border and black center
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Set border regions to white
    mask[:border_width, :] = 255  # Top border
    mask[-border_width:, :] = 255  # Bottom border
    mask[:, :border_width] = 255  # Left border
    mask[:, -border_width:] = 255  # Right border

    return Image.fromarray(mask)


def create_corner_mask(width: int, height: int, corner_size: int = 200) -> Image.Image:
    """
    Create a mask that trains only on the four corners.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        corner_size: Size of corner regions in pixels

    Returns:
        PIL Image with white corners and black center
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Set corner regions to white
    mask[:corner_size, :corner_size] = 255  # Top-left
    mask[:corner_size, -corner_size:] = 255  # Top-right
    mask[-corner_size:, :corner_size] = 255  # Bottom-left
    mask[-corner_size:, -corner_size:] = 255  # Bottom-right

    return Image.fromarray(mask)


def validate_mask(
    mask_path: str, expected_size: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Validate a mask image for training.

    Args:
        mask_path: Path to mask image
        expected_size: Expected (width, height) of mask

    Returns:
        True if mask is valid, False otherwise

    Example:
        >>> is_valid = validate_mask("mask.png", (1024, 1024))
        >>> print(f"Mask is valid: {is_valid}")
    """
    try:
        # Check if file exists
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found: {mask_path}")
            return False

        # Load and check image
        with Image.open(mask_path) as img:
            # Convert to grayscale
            img = img.convert("L")

            # Check size if specified
            if expected_size:
                if img.size != expected_size:
                    logger.error(
                        f"Mask size {img.size} doesn't match expected {expected_size}"
                    )
                    return False

            # Check value range
            img_array = np.array(img)
            min_val = img_array.min()
            max_val = img_array.max()

            if min_val < 0 or max_val > 255:
                logger.error(
                    f"Mask values out of range [0, 255]: [{min_val}, {max_val}]"
                )
                return False

            logger.info(f"Mask validation passed: {mask_path}")
            return True

    except Exception as e:
        logger.error(f"Error validating mask {mask_path}: {e}")
        return False


def batch_create_masks(
    image_directory: str, mask_directory: str, mask_type: str = "center", **kwargs
) -> None:
    """
    Batch create masks for all images in a directory.

    Args:
        image_directory: Directory containing training images
        mask_directory: Directory to save mask images
        mask_type: Type of mask to create ("center", "gradient", "border", "corner")
        **kwargs: Additional arguments for mask creation functions

    Example:
        >>> batch_create_masks(
        ...     "path/to/images",
        ...     "path/to/masks",
        ...     mask_type="center",
        ...     center_ratio=0.5
        ... )
    """
    # Create mask directory if it doesn't exist
    os.makedirs(mask_directory, exist_ok=True)

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Get all image files
    image_files = []
    for filename in os.listdir(image_directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)

    logger.info(f"Found {len(image_files)} images to process")

    for filename in image_files:
        # Get image path and determine mask path
        image_path = os.path.join(image_directory, filename)
        base_name = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_directory, f"{base_name}.png")

        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Create mask based on type
            if mask_type == "center":
                mask = create_center_mask(width, height, **kwargs)
            elif mask_type == "gradient":
                mask = create_gradient_mask(width, height, **kwargs)
            elif mask_type == "border":
                mask = create_border_mask(width, height, **kwargs)
            elif mask_type == "corner":
                mask = create_corner_mask(width, height, **kwargs)
            else:
                logger.error(f"Unknown mask type: {mask_type}")
                continue

            # Save mask
            mask.save(mask_path)
            logger.info(f"Created mask: {mask_path}")

        except Exception as e:
            logger.error(f"Error creating mask for {filename}: {e}")


def analyze_mask_coverage(mask_path: str) -> dict:
    """
    Analyze mask coverage and statistics.

    Args:
        mask_path: Path to mask image

    Returns:
        Dictionary with mask statistics

    Example:
        >>> stats = analyze_mask_coverage("mask.png")
        >>> print(f"Coverage: {stats['coverage']:.2%}")
    """
    try:
        with Image.open(mask_path) as img:
            img = img.convert("L")
            img_array = np.array(img)

            # Calculate statistics
            total_pixels = img_array.size
            white_pixels = np.sum(img_array == 255)
            black_pixels = np.sum(img_array == 0)
            gray_pixels = total_pixels - white_pixels - black_pixels

            coverage = white_pixels / total_pixels
            avg_intensity = img_array.mean()

            return {
                "coverage": coverage,
                "total_pixels": total_pixels,
                "white_pixels": white_pixels,
                "black_pixels": black_pixels,
                "gray_pixels": gray_pixels,
                "avg_intensity": avg_intensity,
                "min_intensity": img_array.min(),
                "max_intensity": img_array.max(),
            }

    except Exception as e:
        logger.error(f"Error analyzing mask {mask_path}: {e}")
        return {}


def create_mask_from_alpha(image_path: str, threshold: int = 128) -> Image.Image:
    """
    Create a mask from the alpha channel of an image.

    Args:
        image_path: Path to image with alpha channel
        threshold: Threshold for alpha values (0-255)

    Returns:
        PIL Image mask based on alpha channel
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA"):
            # Extract alpha channel
            alpha = img.split()[-1]
            # Convert to binary mask based on threshold
            mask_array = np.array(alpha) > threshold
            mask_array = mask_array.astype(np.uint8) * 255
            return Image.fromarray(mask_array)
        else:
            raise ValueError(f"Image {image_path} doesn't have an alpha channel")


def augment_mask(
    mask: np.ndarray, augmentation_factor: float = 0.1, noise_type: str = "gaussian"
) -> np.ndarray:
    """
    Apply augmentation to a mask.

    Args:
        mask: Mask as numpy array
        augmentation_factor: Strength of augmentation (0.0 to 1.0)
        noise_type: Type of noise ("gaussian", "uniform")

    Returns:
        Augmented mask as numpy array
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, augmentation_factor * 255, mask.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(
            -augmentation_factor * 255, augmentation_factor * 255, mask.shape
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    augmented_mask = np.clip(mask.astype(float) + noise, 0, 255)
    return augmented_mask.astype(np.uint8)


# Example usage and testing functions
def create_example_masks(output_directory: str = "example_masks") -> None:
    """
    Create example masks for demonstration.

    Args:
        output_directory: Directory to save example masks
    """
    os.makedirs(output_directory, exist_ok=True)

    # Create various example masks
    masks = [
        ("center_mask.png", create_center_mask(1024, 1024, 0.5)),
        ("gradient_radial.png", create_gradient_mask(1024, 1024, "radial")),
        ("gradient_linear_h.png", create_gradient_mask(1024, 1024, "linear_h")),
        ("border_mask.png", create_border_mask(1024, 1024, 100)),
        ("corner_mask.png", create_corner_mask(1024, 1024, 200)),
    ]

    for filename, mask in masks:
        mask_path = os.path.join(output_directory, filename)
        mask.save(mask_path)
        logger.info(f"Created example mask: {mask_path}")

        # Analyze the mask
        stats = analyze_mask_coverage(mask_path)
        logger.info(f"Coverage: {stats['coverage']:.2%}")
