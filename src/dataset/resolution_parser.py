"""
Resolution parsing utilities for flexible resolution specifications.

Supports multiple formats:
- [width, height] - explicit resolution like [960, 544]
- scalar - square resolution like 512 -> (512, 512)
- [width, None] or [width, ] - width-constrained, height derived from aspect ratio
- [None, height] or [, height] - height-constrained, width derived from aspect ratio
- "480p", "720p", "1080p", etc. - standard video resolutions
"""

import logging
from typing import Tuple, Union, List, Optional
from common.logger import get_logger
from common.constants import RESOLUTION_STEPS_WAN_2

logger = get_logger(__name__, level=logging.INFO)

# Standard resolution presets (height -> common width for 16:9 aspect ratio)
RESOLUTION_PRESETS = {
    # SD resolutions
    "240p": (426, 240),
    "256p": (456, 256),
    "360p": (640, 360),
    "384p": (688, 384),
    "480p": (854, 480),
    "512p": (912, 512),
    "540p": (960, 540),
    "640p": (1136, 640),

    # HD resolutions
    "720p": (1280, 720),
    "768p": (1360, 768),
    "900p": (1600, 900),
    "1080p": (1920, 1080),

    # 2K/QHD resolutions
    "1440p": (2560, 1440),
    "2k": (2048, 1080),

    # 4K/UHD resolutions
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
    "uhd": (3840, 2160),

    # Other common resolutions
    "vga": (640, 480),
    "svga": (800, 600),
    "xga": (1024, 768),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}

# Default aspect ratio for min-dimension mode (16:9 is most common for video)
DEFAULT_ASPECT_RATIO = 16.0 / 9.0


class ResolutionSpec:
    """
    Represents a resolution specification that can be either:
    - Fixed: both dimensions specified
    - Width-constrained: width specified, height derived
    - Height-constrained: height specified, width derived
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: str = "fixed",
        aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    ):
        """
        Args:
            width: Target width (if specified)
            height: Target height (if specified)
            mode: One of "fixed", "width_constrained", "height_constrained"
            aspect_ratio: Target aspect ratio for constrained modes
        """
        self.width = width
        self.height = height
        self.mode = mode
        self.aspect_ratio = aspect_ratio

    def resolve(self, resolution_step: int = 16) -> Tuple[int, int]:
        """
        Resolve the specification to concrete (width, height).

        Args:
            resolution_step: Step size for rounding (default 16 for WAN2)

        Returns:
            Tuple of (width, height)
        """
        if self.mode == "fixed":
            if self.width is None or self.height is None:
                raise ValueError("Fixed mode requires both width and height")
            # Round both dimensions to resolution step
            width = _round_to_step(self.width, resolution_step)
            height = _round_to_step(self.height, resolution_step)
            logger.info(
                f"📐 Resolved fixed resolution: [{self.width}, {self.height}] "
                f"→ ({width}, {height}) (both dimensions divisible by {resolution_step})"
            )
            return (width, height)

        elif self.mode == "width_constrained":
            if self.width is None:
                raise ValueError("Width-constrained mode requires width")
            # Round width to resolution step first
            width = _round_to_step(self.width, resolution_step)
            # Calculate height from width and aspect ratio
            height = int(width / self.aspect_ratio)
            # Round height to resolution step
            height = _round_to_step(height, resolution_step)
            logger.info(
                f"📐 Resolved width-constrained resolution: [{self.width}, ] "
                f"→ ({width}, {height}) [AR={width/height:.3f}] (both dimensions divisible by {resolution_step})"
            )
            return (width, height)

        elif self.mode == "height_constrained":
            if self.height is None:
                raise ValueError("Height-constrained mode requires height")
            # Round height to resolution step first
            height = _round_to_step(self.height, resolution_step)
            # Calculate width from height and aspect ratio
            width = int(height * self.aspect_ratio)
            # Round width to resolution step
            width = _round_to_step(width, resolution_step)
            logger.info(
                f"📐 Resolved height-constrained resolution: [ ,{self.height}] "
                f"→ ({width}, {height}) [AR={width/height:.3f}] (both dimensions divisible by {resolution_step})"
            )
            return (width, height)

        else:
            raise ValueError(f"Unknown resolution mode: {self.mode}")


def _round_to_step(value: int, step: int) -> int:
    """Round value to nearest multiple of step."""
    return int(round(value / step) * step)


def parse_resolution_string(value: str) -> Union[Tuple[int, int], Tuple[Tuple[int, int], str, int]]:
    """
    Parse resolution from string format.

    Supports formats like:
    - "480p", "720p", "1080p" - standard presets
    - "WxH" - explicit like "960x544"
    - "Wx" - width-constrained like "512x" (NEW!)
    - "xH" - height-constrained like "x512" (NEW!)

    Args:
        value: Resolution string

    Returns:
        Tuple of (width, height) or ((width, height), constraint_type, value) for constrained modes
    """
    value_lower = value.lower().strip()

    # Check if it's a preset
    if value_lower in RESOLUTION_PRESETS:
        width, height = RESOLUTION_PRESETS[value_lower]
        # Round to resolution step to ensure compatibility
        width = _round_to_step(width, RESOLUTION_STEPS_WAN_2)
        height = _round_to_step(height, RESOLUTION_STEPS_WAN_2)
        logger.info(f"📺 Using preset resolution '{value}' → ({width}, {height}) (divisible by {RESOLUTION_STEPS_WAN_2})")
        return (width, height)

    # Check if it's constrained format: "512x" or "x512"
    if 'x' in value_lower:
        parts = value_lower.split('x')
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()

            # Width-constrained: "512x" (right side empty)
            if left and not right:
                try:
                    width = int(left)
                    # Return as constrained format
                    # Will be processed by parse_resolution
                    return ([width, None], "width", width)  # type: ignore
                except ValueError:
                    pass

            # Height-constrained: "x512" (left side empty)
            elif not left and right:
                try:
                    height = int(right)
                    # Return as constrained format
                    return ([None, height], "height", height)  # type: ignore
                except ValueError:
                    pass

            # Explicit WxH: "960x544"
            elif left and right:
                try:
                    width = int(left)
                    height = int(right)
                    logger.info(f"📐 Parsed resolution string '{value}' → ({width}, {height})")
                    return (width, height)
                except ValueError:
                    pass

    raise ValueError(
        f"Invalid resolution string: '{value}'. "
        f"Supported formats: '480p', '720p', '1080p', '4k', 'WxH' (e.g. '960x544'), "
        f"'Wx' (width-constrained, e.g. '512x'), 'xH' (height-constrained, e.g. 'x512')"
    )


def parse_resolution(
    value: Union[int, str, List, Tuple],
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    resolution_step: int = 16,
    _return_constraint_info: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], str, Optional[int]]]:
    """
    Parse resolution from various formats and return (width, height).

    Supported formats:
    - int: 512 → (512, 512)
    - str: "480p", "720p", "1080p", "4k", "960x544"
    - [int, int]: [960, 544] → (960, 544)
    - [int, None]: [512, None] → (512, height_from_AR)
    - [None, int]: [None, 512] → (width_from_AR, 512)

    Args:
        value: Resolution specification
        aspect_ratio: Default aspect ratio for constrained modes (default 16:9)
        resolution_step: Rounding step for derived dimensions (default 16)

    Returns:
        Tuple of (width, height)
    """
    # String format (presets, WxH, Wx, or xH)
    if isinstance(value, str):
        result = parse_resolution_string(value)
        # Check if it returned a constrained format
        if isinstance(result, tuple) and len(result) == 3:
            # It's a constrained format, recurse with the list format
            list_format, constraint_type_str, constrained_val = result
            # Recurse to parse the list format (without constraint info flag)
            final_result = parse_resolution(list_format, aspect_ratio, resolution_step)
            # Return with constraint info if needed
            if _return_constraint_info:
                return (final_result, constraint_type_str, constrained_val)
            return final_result
        # Regular string format (preset or WxH)
        if _return_constraint_info:
            return (result, "none", None)
        return result

    # Scalar - square resolution
    if isinstance(value, int):
        # Round to resolution step
        size = _round_to_step(value, resolution_step)
        logger.info(f"📐 Using square resolution: {value} → ({size}, {size}) (divisible by {resolution_step})")
        result = (size, size)
        if _return_constraint_info:
            return (result, "none", None)
        return result

    # List or tuple format
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"Resolution list must have exactly 2 elements, got {len(value)}: {value}"
            )

        width, height = value

        # Both specified - fixed resolution
        if width is not None and height is not None:
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError(
                    f"Width and height must be integers, got {type(width)} and {type(height)}"
                )
            # Round both dimensions to resolution step
            width = _round_to_step(width, resolution_step)
            height = _round_to_step(height, resolution_step)
            result = (width, height)
            if _return_constraint_info:
                return (result, "none", None)
            return result

        # Width constrained - [512, None] or [512, ]
        elif width is not None and height is None:
            if not isinstance(width, int):
                raise ValueError(f"Width must be integer, got {type(width)}")
            width = _round_to_step(width, resolution_step)
            spec = ResolutionSpec(
                width=width,
                height=None,
                mode="width_constrained",
                aspect_ratio=aspect_ratio,
            )
            result = spec.resolve(resolution_step)
            if _return_constraint_info:
                return (result, "width", width)
            return result

        # Height constrained - [None, 512] or [ , 512]
        elif width is None and height is not None:
            if not isinstance(height, int):
                raise ValueError(f"Height must be integer, got {type(height)}")
            height = _round_to_step(height, resolution_step)
            spec = ResolutionSpec(
                width=None,
                height=height,
                mode="height_constrained",
                aspect_ratio=aspect_ratio,
            )
            result = spec.resolve(resolution_step)
            if _return_constraint_info:
                return (result, "height", height)
            return result

        # Both None - invalid
        else:
            raise ValueError(
                "At least one dimension must be specified in resolution: [width, height]"
            )

    raise ValueError(
        f"Unsupported resolution format: {type(value)}. "
        f"Use int, str ('480p', '720p', etc.), or [width, height]"
    )


def validate_and_parse_resolution(
    value: Union[int, str, List, Tuple],
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    return_constraint_info: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], str, Optional[int]]]:
    """
    Validate and parse resolution with improved error messages.

    This is the main function to use from config parsing.

    Args:
        value: Resolution specification in any supported format
        aspect_ratio: Default aspect ratio for min-dimension modes
        return_constraint_info: If True, return (resolution, constraint_type, constrained_value)

    Returns:
        If return_constraint_info=False: Tuple of (width, height)
        If return_constraint_info=True: Tuple of ((width, height), constraint_type, constrained_value)
            - constraint_type: "none", "width", or "height"
            - constrained_value: The fixed dimension value, or None if constraint_type="none"

    Raises:
        ValueError: If resolution format is invalid
    """
    try:
        result = parse_resolution(value, aspect_ratio=aspect_ratio, _return_constraint_info=True)

        # Unpack the result
        if isinstance(result, tuple) and len(result) == 3:
            resolution, constraint_type, constrained_value = result
        else:
            # Shouldn't happen, but handle it
            resolution = result
            constraint_type = "none"
            constrained_value = None

        # Validate result
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Resolution dimensions must be positive, got ({width}, {height})"
            )

        # Warn if resolution is not divisible by 16 (WAN2 requirement)
        if width % 16 != 0 or height % 16 != 0:
            logger.warning(
                f"⚠️  Resolution ({width}, {height}) is not divisible by 16. "
                f"This may cause issues with WAN2 architecture. "
                f"Recommended: use multiples of 16 for both dimensions."
            )

        if return_constraint_info:
            return (resolution, constraint_type, constrained_value)
        else:
            return resolution

    except Exception as e:
        logger.error(f"❌ Failed to parse resolution: {value}")
        logger.error(f"   Error: {e}")
        logger.error("")
        logger.error("💡 Supported resolution formats:")
        logger.error("   - Square: 512")
        logger.error("   - Explicit: [960, 544]")
        logger.error("   - Width-only: [512, None] or [512, ]")
        logger.error("   - Height-only: [None, 512] or [ , 512]")
        logger.error("   - Presets: '480p', '720p', '1080p', '4k'")
        logger.error("   - String: '960x544'")
        raise
