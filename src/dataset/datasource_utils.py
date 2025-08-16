## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/image_video_dataset.py (Apache)

import glob
import os
from typing import Optional, Union
import logging
import numpy as np
from PIL import Image
import cv2
import av

from dataset.buckets import BucketSelector
from dataset.extensions import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(
                glob.glob(os.path.join(glob.escape(directory), base + ext))
            )
        else:
            img_paths.extend(
                glob.glob(glob.escape(os.path.join(directory, base + ext)))
            )
    img_paths = list(set(img_paths))  # remove duplicates
    img_paths.sort()
    return img_paths


def glob_videos(directory, base="*"):
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        if base == "*":
            video_paths.extend(
                glob.glob(os.path.join(glob.escape(directory), base + ext))
            )
        else:
            video_paths.extend(
                glob.glob(glob.escape(os.path.join(directory, base + ext)))
            )
    video_paths = list(set(video_paths))  # remove duplicates
    video_paths.sort()
    return video_paths


def resize_image_to_bucket(
    image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]
) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image

        image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)  # type: ignore
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(  # type: ignore
            image, (image_width, image_height), interpolation=cv2.INTER_AREA  # type: ignore
        )

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[
        crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width
    ]
    return image


def resize_image_to_bucket_lossless(
    image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]
) -> np.ndarray:
    """
    Resize the image to the bucket resolution using nearest-neighbor interpolation.
    Preserves exact pixel values when scaling by integer factors.

    Args:
        image: PIL Image or numpy array
        bucket_reso: Target resolution as (width, height)

    Returns:
        numpy.ndarray: Resized and cropped image

    Raises:
        ValueError: If image dimensions are invalid or bucket resolution is invalid
    """
    # Input validation
    if not isinstance(bucket_reso, tuple) or len(bucket_reso) != 2:
        raise ValueError("bucket_reso must be a tuple of (width, height)")

    bucket_width, bucket_height = bucket_reso
    if bucket_width <= 0 or bucket_height <= 0:
        raise ValueError("Bucket dimensions must be positive")

    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be PIL Image or numpy array")
        if len(image.shape) < 2:
            raise ValueError("Image must have at least 2 dimensions")
        image_height, image_width = image.shape[:2]

    # Validate image dimensions
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # If already at target size, return as numpy array
    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image.copy()

    # Calculate scaling factors
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)

    # Calculate target dimensions preserving integer ratios where possible
    target_width = int(image_width * scale + 0.5)
    target_height = int(image_height * scale + 0.5)

    # Ensure minimum dimensions
    target_width = max(target_width, bucket_width)
    target_height = max(target_height, bucket_height)

    # Use nearest-neighbor interpolation for both upscaling and downscaling
    if is_pil_image:
        resized_image = image.resize(
            (target_width, target_height), Image.Resampling.NEAREST
        )
        resized_image = np.array(resized_image)
    else:
        resized_image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    # Center crop to exact bucket size
    crop_left = max(0, (target_width - bucket_width) // 2)
    crop_top = max(0, (target_height - bucket_height) // 2)

    # Ensure crop coordinates are within bounds
    crop_left = min(crop_left, target_width - bucket_width)
    crop_top = min(crop_top, target_height - bucket_height)

    return resized_image[
        crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width
    ]


def load_video(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    bucket_selector: Optional[BucketSelector] = None,
    bucket_reso: Optional[tuple[int, int]] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
) -> list[np.ndarray]:
    # TODO: REFACTOR - This function is too complex and handles multiple responsibilities
    # Consider splitting into separate functions for video loading, frame extraction, and resizing
    # TODO: REFACTOR - Duplicate code blocks for video file vs directory handling should be extracted
    """
    bucket_reso: if given, resize the video to the bucket resolution, (width, height)
    """
    if source_fps is None or target_fps is None:
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(
                        frame.size
                    )  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            for i in range(len(image_files)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(
                        image.size
                    )  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)
    else:
        # drop frames to match the target fps TODO commonize this code with the above if this works
        frame_index_delta = target_fps / source_fps  # example: 16 / 30 = 0.5333
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i, frame in enumerate(container.decode(video=0)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(
                        frame.size
                    )  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i in range(len(image_files)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(
                        image.size
                    )  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)

    return video
