# Slider Training Configuration Handler
# Manages slider-specific configuration and validation

from typing import Optional, Dict, Any
import argparse
from dataclasses import dataclass

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class SliderConfig:
    """Configuration class for slider training parameters."""

    # Core slider parameters
    guidance_strength: float = 3.0
    anchor_strength: float = 1.0

    # Advanced guidance parameters (for fine-tuning)
    guidance_scale: float = 1.0                  # Base classifier-free guidance scale
    guidance_embedding_scale: float = 1.0       # Embedding-level guidance scaling
    target_guidance_scale: float = 1.0          # Separate scale for target predictions

    # Prompt configuration
    positive_prompt: str = ""
    negative_prompt: str = ""
    target_class: str = ""
    anchor_class: Optional[str] = None

    # Training parameters
    slider_learning_rate_multiplier: float = 1.0
    slider_cache_embeddings: bool = True

    # T5 text encoder settings (follows Takenoko's pattern)
    slider_t5_device: str = "cpu"                # Device for T5 encoder ("cpu" or "cuda")
    slider_cache_on_init: bool = True            # Cache embeddings during initialization

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate slider configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.positive_prompt or not self.positive_prompt.strip():
            raise ValueError("slider_positive_prompt is required and cannot be empty")

        if not self.negative_prompt or not self.negative_prompt.strip():
            raise ValueError("slider_negative_prompt is required and cannot be empty")

        if not self.target_class or not self.target_class.strip():
            raise ValueError("slider_target_class is required and cannot be empty")

        # Check for identical prompts (would cause zero concept direction)
        if self.positive_prompt.strip() == self.negative_prompt.strip():
            raise ValueError("positive_prompt and negative_prompt cannot be identical")

        if self.positive_prompt.strip() == self.target_class.strip():
            raise ValueError("positive_prompt and target_class should be different")

        # Validate anchor class if provided
        if self.anchor_class is not None and not self.anchor_class.strip():
            raise ValueError("anchor_class cannot be empty string (use None to disable)")

        # Validate numeric parameters
        if self.guidance_strength <= 0:
            raise ValueError("guidance_strength must be positive")

        if self.anchor_strength < 0:
            raise ValueError("anchor_strength must be non-negative")

        if self.slider_learning_rate_multiplier <= 0:
            raise ValueError("slider_learning_rate_multiplier must be positive")

        # Validate scale parameters
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")

        if self.guidance_embedding_scale <= 0:
            raise ValueError("guidance_embedding_scale must be positive")

        if self.target_guidance_scale <= 0:
            raise ValueError("target_guidance_scale must be positive")

        # Validate reasonable ranges
        if self.guidance_strength > 100:
            raise ValueError("guidance_strength seems too high (>100), check configuration")

        if self.anchor_strength > 100:
            raise ValueError("anchor_strength seems too high (>100), check configuration")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'guidance_strength': self.guidance_strength,
            'anchor_strength': self.anchor_strength,
            'positive_prompt': self.positive_prompt,
            'negative_prompt': self.negative_prompt,
            'target_class': self.target_class,
            'anchor_class': self.anchor_class,
            'slider_learning_rate_multiplier': self.slider_learning_rate_multiplier,
            'slider_cache_embeddings': self.slider_cache_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SliderConfig":
        """Create SliderConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class SliderConfigHandler:
    """
    Simplified handler for slider training configuration management.

    Now only provides utility methods since slider detection is based on network_module.
    """

    @staticmethod
    def is_slider_network(network_module: str) -> bool:
        """
        Check if the network module indicates slider training.

        Args:
            network_module: Network module string

        Returns:
            True if this is a slider network module
        """
        return 'slider' in network_module.lower()

    @staticmethod
    def extract_slider_config(args: argparse.Namespace) -> SliderConfig:
        """
        Extract slider configuration from parsed arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            SliderConfig instance
        """
        return SliderConfig(
            guidance_strength=getattr(args, 'slider_guidance_strength', 3.0),
            anchor_strength=getattr(args, 'slider_anchor_strength', 1.0),
            positive_prompt=getattr(args, 'slider_positive_prompt', ''),
            negative_prompt=getattr(args, 'slider_negative_prompt', ''),
            target_class=getattr(args, 'slider_target_class', ''),
            anchor_class=getattr(args, 'slider_anchor_class', None),
            slider_learning_rate_multiplier=getattr(args, 'slider_learning_rate_multiplier', 1.0),
            slider_cache_embeddings=getattr(args, 'slider_cache_embeddings', True),
        )

    @staticmethod
    def log_slider_configuration(config: SliderConfig) -> None:
        """
        Log slider training configuration for debugging.

        Args:
            config: SliderConfig instance to log
        """
        logger.info("🎚️  Slider Training Configuration:")
        logger.info("=" * 50)
        logger.info(f"  Guidance Strength: {config.guidance_strength}")
        logger.info(f"  Anchor Strength: {config.anchor_strength}")
        logger.info(f"  Positive Prompt: '{config.positive_prompt}'")
        logger.info(f"  Negative Prompt: '{config.negative_prompt}'")
        logger.info(f"  Target Class: '{config.target_class}'")
        logger.info(f"  Anchor Class: '{config.anchor_class}'")
        logger.info(f"  Learning Rate Multiplier: {config.slider_learning_rate_multiplier}")
        logger.info(f"  Cache Embeddings: {config.slider_cache_embeddings}")
        logger.info("=" * 50)