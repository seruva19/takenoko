"""DenseDPO configuration schema with validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DenseDPOConfig:
    """Configuration for DenseDPO training."""

    densedpo_partial_noise_eta: float = 0.5
    densedpo_num_inference_steps: int = 50
    densedpo_segment_frames: int = 16
    densedpo_beta: float = 0.1
    densedpo_label_source: str = "reward"
    densedpo_segment_preference_key: str = "densedpo_segment_preferences"

    densedpo_reward_model_name: str = "hps"
    densedpo_reward_model_dtype: str = "float32"
    densedpo_reward_frame_strategy: str = "first"
    densedpo_reward_num_frames: int = 1
    densedpo_reward_aggregation: str = "mean"

    densedpo_vlm_model_path: Optional[str] = None
    densedpo_vlm_dtype: str = "bfloat16"
    densedpo_vlm_prompt: str = (
        "Rate the visual quality and motion consistency of this short video clip "
        "on a scale of 1 to 10. Respond with a single number."
    )
    densedpo_vlm_max_new_tokens: int = 8
    densedpo_vlm_temperature: float = 0.0
    densedpo_vlm_cache_dir: Optional[str] = None
    densedpo_vlm_max_frames: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.densedpo_partial_noise_eta <= 1.0):
            raise ValueError(
                "densedpo_partial_noise_eta must be in [0, 1]."
            )
        if self.densedpo_num_inference_steps < 2:
            raise ValueError("densedpo_num_inference_steps must be >= 2.")
        if self.densedpo_segment_frames < 1:
            raise ValueError("densedpo_segment_frames must be >= 1.")
        if self.densedpo_beta <= 0.0:
            raise ValueError("densedpo_beta must be > 0.")
        if self.densedpo_label_source not in ("reward", "provided", "vlm"):
            raise ValueError(
                "densedpo_label_source must be 'reward', 'provided', or 'vlm'."
            )
        if not self.densedpo_segment_preference_key:
            raise ValueError(
                "densedpo_segment_preference_key must be non-empty."
            )
        if (
            self.densedpo_label_source == "vlm"
            and not self.densedpo_vlm_model_path
        ):
            raise ValueError(
                "densedpo_vlm_model_path is required when densedpo_label_source='vlm'."
            )

        valid_reward_models = ["hps", "pickscore", "aesthetic"]
        if self.densedpo_reward_model_name not in valid_reward_models:
            raise ValueError(
                f"Invalid densedpo_reward_model_name='"
                f"{self.densedpo_reward_model_name}'. "
                f"Must be one of {valid_reward_models}"
            )

        valid_reward_dtypes = ["float32", "bfloat16", "float16"]
        if self.densedpo_reward_model_dtype not in valid_reward_dtypes:
            raise ValueError(
                f"Invalid densedpo_reward_model_dtype='"
                f"{self.densedpo_reward_model_dtype}'. "
                f"Must be one of {valid_reward_dtypes}"
            )

        if self.densedpo_vlm_dtype not in valid_reward_dtypes:
            raise ValueError(
                f"Invalid densedpo_vlm_dtype='"
                f"{self.densedpo_vlm_dtype}'. "
                f"Must be one of {valid_reward_dtypes}"
            )

        valid_strategies = ["first", "uniform", "all", "boundary"]
        if self.densedpo_reward_frame_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid densedpo_reward_frame_strategy='"
                f"{self.densedpo_reward_frame_strategy}'. "
                f"Must be one of {valid_strategies}"
            )

        if self.densedpo_reward_num_frames < 1:
            raise ValueError("densedpo_reward_num_frames must be >= 1.")

        valid_aggregations = ["mean", "min", "max", "weighted"]
        if self.densedpo_reward_aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid densedpo_reward_aggregation='"
                f"{self.densedpo_reward_aggregation}'. "
                f"Must be one of {valid_aggregations}"
            )

        if self.densedpo_vlm_max_new_tokens < 1:
            raise ValueError("densedpo_vlm_max_new_tokens must be >= 1.")
        if self.densedpo_vlm_temperature < 0.0:
            raise ValueError("densedpo_vlm_temperature must be >= 0.")
        if self.densedpo_vlm_max_frames < 1:
            raise ValueError("densedpo_vlm_max_frames must be >= 1.")

        logger.info(
            "DenseDPO config validated (segments=%s, beta=%s, label_source=%s).",
            self.densedpo_segment_frames,
            self.densedpo_beta,
            self.densedpo_label_source,
        )
