import os
import torch
from typing import Optional

from wan.modules.vae import WanVAE
from common.model_downloader import download_model_if_needed
from common.logger import get_logger

logger = get_logger(__name__)


def load_vae(args, config, device: torch.device, dtype: torch.dtype) -> WanVAE:
    """load VAE model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dtype: data type for the model

    Returns:
        WanVAE: loaded VAE model
    """
    vae_path = (
        args.vae
        if args.vae is not None
        else os.path.join(args.ckpt_dir, config.vae_checkpoint)
    )

    # Download model if it's a URL
    if vae_path.startswith(("http://", "https://")):
        logger.info(f"Detected URL for VAE model, downloading: {vae_path}")
        cache_dir = getattr(args, "model_cache_dir", None)
        vae_path = download_model_if_needed(vae_path, cache_dir=cache_dir)
        logger.info(f"Downloaded VAE model to: {vae_path}")

    logger.info(f"Loading VAE model from {vae_path}")
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(
        vae_path=vae_path, device=device, dtype=dtype, cache_device=cache_device  # type: ignore
    )
    return vae
