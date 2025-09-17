import os
from typing import Optional, Callable
from typing import TYPE_CHECKING
import logging
import numpy as np
from PIL import Image

from dataset.buckets import BucketSelector
from dataset.datasource_utils import glob_images, glob_videos, load_video


from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Global set to track missing caption files we've already warned about
_missing_caption_warnings = set()
# Global flag to track if we've shown the general missing caption warning
_missing_caption_global_warning_shown = False


class ContentDataSource:
    # TODO: REFACTOR - This should be an abstract base class (ABC)
    # Consider using ABC and @abstractmethod decorators for better interface definition
    def __init__(self):
        self.caption_only = False
        self.load_control = False
        self.control_suffix = "_control"
        self.load_mask = False
        self.mask_path = None
        self.default_mask_file = None
        # Probability to drop the caption entirely (0.0-1.0). When >0, captions are set to '' with this probability.
        self.caption_dropout_rate: float = 0.0

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def set_control_settings(
        self, load_control: bool, control_suffix: str = "_control"
    ):
        """Set control signal loading settings."""
        self.load_control = load_control
        self.control_suffix = control_suffix

    def set_caption_dropout_rate(self, rate: float):
        try:
            r = float(rate)
        except Exception:
            r = 0.0
        # clamp to [0, 1]
        self.caption_dropout_rate = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)

    def set_mask_settings(
        self,
        load_mask: bool,
        mask_path: Optional[str] = None,
        default_mask_file: Optional[str] = None,
    ):
        """Set mask loading settings for masked training. default_mask_file is deprecated and ignored."""
        self.load_mask = load_mask
        self.mask_path = mask_path
        self.default_mask_file = None

    @property
    def has_control(self) -> bool:
        """Check if control signals are available."""
        return self.load_control

    def is_indexable(self):
        raise NotImplementedError

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDataSource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str]:
        raise NotImplementedError


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(self, image_directory: str, caption_extension: Optional[str] = None):
        # TODO: REFACTOR - Complex initialization logic should be split into separate methods
        # TODO: REFACTOR - Control image matching logic is complex and should be extracted
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory)
        logger.info(f"found {len(self.image_paths)} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(
        self, idx: int
    ) -> tuple[str, Image.Image, str, Optional[Image.Image], Optional[Image.Image]]:
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        _, caption = self.get_caption(idx)

        # Load control image if enabled
        control_image = None
        if self.load_control:
            control_path = self._get_control_path(image_path)
            if os.path.exists(control_path):
                try:
                    with Image.open(control_path) as img:
                        control_image = img.convert("RGB")
                    logger.debug(f"Loaded control image: {control_path}")
                except Exception as e:
                    logger.warning(f"Failed to load control image {control_path}: {e}")
            else:
                logger.debug(f"Control image not found: {control_path}")

        # Load mask image if enabled
        mask_image = None
        if self.load_mask:
            mask_path = self._get_mask_path(image_path)
            if os.path.exists(mask_path):
                try:
                    with Image.open(mask_path) as img:
                        mask_image = img.convert("L")  # Convert to grayscale
                    logger.debug(f"Loaded mask image: {mask_path}")
                except Exception as e:
                    logger.warning(f"Failed to load mask image {mask_path}: {e}")
            else:
                logger.debug(f"Mask image not found: {mask_path}")

        return image_path, image, caption, control_image, mask_image

    def _get_control_path(self, image_path: str) -> str:
        """Get the path for the control image based on the original image path."""
        base_path = os.path.splitext(image_path)[0]
        return f"{base_path}{self.control_suffix}.png"

    def _get_mask_path(self, image_path: str) -> str:
        """Get the path for the mask image based on the original image path."""
        if self.mask_path is None:
            # If no mask path is specified, look for mask in same directory as image
            base_path = os.path.splitext(image_path)[0]
            return f"{base_path}_mask.png"
        else:
            # If mask path is specified, look for mask with same name in mask directory
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            return os.path.join(self.mask_path, f"{base_name}.png")

    def get_caption(self, idx: int) -> tuple[str, str]:
        global _missing_caption_warnings, _missing_caption_global_warning_shown
        image_path = self.image_paths[idx]

        # If no caption extension is specified, return empty caption
        if not self.caption_extension:
            return image_path, ""

        caption_path = os.path.splitext(image_path)[0] + self.caption_extension

        # Check if caption file exists
        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                # Apply caption dropout if configured
                if self.caption_dropout_rate > 0.0:
                    try:
                        import random as _random

                        if _random.random() < self.caption_dropout_rate:
                            caption = ""
                    except Exception:
                        pass
                return image_path, caption
            except Exception as e:
                # Only warn once per file for read errors
                if caption_path not in _missing_caption_warnings:
                    logger.warning(f"Failed to read caption file {caption_path}: {e}")
                    _missing_caption_warnings.add(caption_path)
                # Return empty string as fallback
                return image_path, ""
        else:
            # Show global warning only once
            if not _missing_caption_global_warning_shown:
                logger.warning(
                    "\nCaption files not found - using empty strings as captions"
                )
                logger.info("To disable this warning, either:")
                logger.info("  1. Create caption files for your images, or")
                logger.info(
                    "  2. Remove or comment out 'caption_extension' in your config"
                )
                _missing_caption_global_warning_shown = True

            # Return empty string as fallback
            return image_path, ""

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_image_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_image_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoDatasource(ContentDataSource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[np.ndarray], str]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = (
            bucket_selector if bucket_selector is not None else self.bucket_selector
        )

        video = load_video(
            video_path,
            start_frame,
            end_frame,
            bucket_selector,
            source_fps=self.source_fps,
            target_fps=self.target_fps,
        )
        return video_path, video, ""

    def set_start_and_end_frame(
        self, start_frame: Optional[int], end_frame: Optional[int]
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(
        self, source_fps: Optional[float], target_fps: Optional[float]
    ):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDataSource(VideoDatasource):
    def __init__(
        self,
        video_directory: str,
        caption_extension: Optional[str] = None,
    ):
        super().__init__()
        self.video_directory = video_directory
        self.caption_extension = caption_extension
        self.current_idx = 0

        # glob videos
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[
        str,
        list[np.ndarray],
        str,
        Optional[list[np.ndarray]],
        Optional[list[np.ndarray]],
    ]:
        video_path = self.video_paths[idx]
        _, video_frames, _ = self.get_video_data_from_path(
            video_path, start_frame, end_frame, bucket_selector
        )

        _, caption = self.get_caption(idx)

        # Load control video if enabled
        control_frames = None
        if self.load_control:
            control_path = self._get_control_path(video_path)
            if os.path.exists(control_path):
                try:
                    _, control_frames, _ = self.get_video_data_from_path(
                        control_path, start_frame, end_frame, bucket_selector
                    )
                    logger.debug(f"Loaded control video: {control_path}")
                except Exception as e:
                    logger.warning(f"Failed to load control video {control_path}: {e}")
            else:
                logger.debug(f"Control video not found: {control_path}")

        # Load mask video if enabled
        mask_frames = None
        if self.load_mask:
            mask_path = self._get_mask_path(video_path)
            if os.path.exists(mask_path):
                try:
                    _, mask_frames, _ = self.get_video_data_from_path(
                        mask_path, start_frame, end_frame, bucket_selector
                    )
                    # Convert to grayscale masks
                    mask_frames = [
                        frame[:, :, 0] if frame.ndim == 3 else frame
                        for frame in mask_frames
                    ]
                    logger.debug(f"Loaded mask video: {mask_path}")
                except Exception as e:
                    logger.warning(f"Failed to load mask video {mask_path}: {e}")
            else:
                logger.debug(f"Mask video not found: {mask_path}")

        return video_path, video_frames, caption, control_frames, mask_frames

    def _get_control_path(self, video_path: str) -> str:
        """Get the path for the control video based on the original video path."""
        base_path = os.path.splitext(video_path)[0]
        return f"{base_path}{self.control_suffix}.mp4"

    def _get_mask_path(self, video_path: str) -> str:
        """Get the path for the mask video based on the original video path."""
        if self.mask_path is None:
            # If no mask path is specified, look for mask in same directory as video
            base_path = os.path.splitext(video_path)[0]
            return f"{base_path}_mask.mp4"
        else:
            # If mask path is specified, look for mask with same name in mask directory
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]
            return os.path.join(self.mask_path, f"{base_name}.mp4")

    def get_caption(self, idx: int) -> tuple[str, str]:
        global _missing_caption_warnings, _missing_caption_global_warning_shown
        video_path = self.video_paths[idx]

        # If no caption extension is specified, return empty caption
        if not self.caption_extension:
            return video_path, ""

        caption_path = os.path.splitext(video_path)[0] + self.caption_extension

        # Check if caption file exists
        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                # Apply caption dropout if configured
                if self.caption_dropout_rate > 0.0:
                    try:
                        import random as _random

                        if _random.random() < self.caption_dropout_rate:
                            caption = ""
                    except Exception:
                        pass
                return video_path, caption
            except Exception as e:
                # Only warn once per file for read errors
                if caption_path not in _missing_caption_warnings:
                    logger.warning(f"Failed to read caption file {caption_path}: {e}")
                    _missing_caption_warnings.add(caption_path)
                # Return empty string as fallback
                return video_path, ""
        else:
            # Show global warning only once (shared with image datasets)
            if not _missing_caption_global_warning_shown:
                logger.warning(
                    "\nCaption files not found - using empty strings as captions"
                )
                logger.info("To disable this warning, either:")
                logger.info("  1. Create caption files for your videos, or")
                logger.info(
                    "  2. Remove or comment out 'caption_extension' in your config"
                )
                _missing_caption_global_warning_shown = True

            # Return empty string as fallback
            return video_path, ""

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher
