"""
Simplified model downloader for direct file downloads.
"""

import os
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def download_model_if_needed(
    path_or_url: str, cache_dir: Optional[str] = None, **kwargs
) -> str:
    """
    Download a model if it's a URL, otherwise return local path.

    Args:
        path_or_url: Local path or URL to the model
        cache_dir: Directory to cache downloaded models (defaults to 'models')
        **kwargs: Additional arguments (ignored for simplicity)

    Returns:
        Path to the model file
    """
    # If it's a local path and exists, return as-is
    if not path_or_url.startswith(("http://", "https://")) and os.path.exists(
        path_or_url
    ):
        logger.info(f"Using local file: {path_or_url}")
        return path_or_url

    # If it's not a URL, raise error
    if not path_or_url.startswith(("http://", "https://")):
        raise FileNotFoundError(f"Model file not found: {path_or_url}")

    # Set up models directory
    models_dir = Path(cache_dir) if cache_dir else Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from URL
    parsed = urlparse(path_or_url)
    filename = os.path.basename(parsed.path)
    if not filename or "." not in filename:
        # Fallback: use hash of URL as filename
        url_hash = hashlib.md5(path_or_url.encode()).hexdigest()[:16]
        filename = f"model_{url_hash}.bin"

    file_path = models_dir / filename

    # Check if already downloaded
    if file_path.exists():
        logger.info(f"Model located: {file_path}")
        return str(file_path)

    # Download the file
    logger.info(f"Downloading: {path_or_url}")
    logger.info(f"Saving to: {file_path}")

    try:
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()

        # Get file size for progress tracking
        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0
        start_time = time.time()
        last_progress_time = start_time

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Update progress every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_progress_time >= 0.5:
                        elapsed_time = current_time - start_time
                        speed = (
                            downloaded_size / elapsed_time if elapsed_time > 0 else 0
                        )

                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            # Create simple progress bar
                            bar_length = 30
                            filled_length = int(
                                bar_length * downloaded_size // total_size
                            )
                            bar = "█" * filled_length + "░" * (
                                bar_length - filled_length
                            )

                            # Format file sizes
                            downloaded_mb = downloaded_size / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            speed_mb = speed / (1024 * 1024)

                            # Estimate time remaining
                            if speed > 0:
                                remaining_bytes = total_size - downloaded_size
                                eta_seconds = remaining_bytes / speed
                                eta_str = f" ETA: {int(eta_seconds)}s"
                            else:
                                eta_str = ""

                            print(
                                f"\r[{bar}] {progress:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f}MB) {speed_mb:.1f}MB/s{eta_str}",
                                end="",
                                flush=True,
                            )
                        else:
                            # No total size available
                            downloaded_mb = downloaded_size / (1024 * 1024)
                            speed_mb = speed / (1024 * 1024)
                            print(
                                f"\rDownloaded: {downloaded_mb:.1f}MB @ {speed_mb:.1f}MB/s",
                                end="",
                                flush=True,
                            )

                        last_progress_time = current_time

        # Final progress update
        elapsed_time = time.time() - start_time
        avg_speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
        avg_speed_mb = avg_speed / (1024 * 1024)
        downloaded_mb = downloaded_size / (1024 * 1024)

        print()  # New line after progress bar
        logger.info(f"Successfully downloaded: {file_path}")
        logger.info(
            f"Total: {downloaded_mb:.1f}MB in {elapsed_time:.1f}s (avg: {avg_speed_mb:.1f}MB/s)"
        )
        return str(file_path)

    except Exception as e:
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Failed to download {path_or_url}: {e}")
        raise
