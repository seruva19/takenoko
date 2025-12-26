from typing import Any, Optional
import logging
import numpy as np

from common.logger import get_logger


logger = get_logger(__name__, level=logging.INFO)


class ItemInfo:
    # TODO: Consider using dataclass or Pydantic model for better type safety and validation
    # TODO: bucket_size type hint is too generic - should be tuple[int, int] or tuple[int, int, int]
    # TODO: Consider separating image and video item info into different classes
    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: tuple[int, int],
        bucket_size: Optional[tuple[Any]] = None,
        frame_count: Optional[int] = None,
        content: Optional[np.ndarray] = None,
        latent_cache_path: Optional[str] = None,
        weight: Optional[float] = 1.0,
        control_content: Optional[np.ndarray] = None,
        mask_content: Optional[np.ndarray] = None,
        is_reg: bool = False,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.frame_count = frame_count
        self.content = content
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None
        self.semantic_encoder_output_cache_path: Optional[str] = None
        self.weight = weight
        self.control_content = control_content
        self.mask_content = mask_content
        self.is_reg = is_reg

    def __str__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            + f"frame_count={self.frame_count}, latent_cache_path={self.latent_cache_path}, content={self.content.shape if self.content is not None else None})"
        )
