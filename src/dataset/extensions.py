import logging
from common.logger import get_logger
from common.dependencies import setup_pillow_extensions

logger = get_logger(__name__, level=logging.INFO)

# Base image extensions that are always supported
IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]

# Use centralized optional dependency management to prevent duplicate warnings
pillow_extensions = setup_pillow_extensions()

# Add extensions based on what's available
if pillow_extensions["pillow_avif"] is not None:
    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])

if pillow_extensions["jxlpy"] is not None:
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if pillow_extensions["pillow_jxl"] is not None and pillow_extensions["jxlpy"] is None:
    # Only add JXL extensions if jxlpy didn't already add them
    if ".jxl" not in IMAGE_EXTENSIONS:
        IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".MP4",
    ".WEBM",
    ".AVI",
    ".MKV",
    ".MOV",
    ".FLV",
    ".WMV",
    ".M4V",
    ".MPG",
    ".MPEG",
]
