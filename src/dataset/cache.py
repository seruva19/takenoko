## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/image_video_dataset.py (Apache)

import os
from typing import Optional
import logging
import torch
from safetensors.torch import save_file

from dataset.item_info import ItemInfo
from utils import safetensors_utils
from utils.model_utils import dtype_to_str

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
):
    """Wan architecture only"""
    assert (
        latent.dim() == 4
    ), "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    save_latent_cache_common(item_info, sd)


def save_latent_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor]):
    metadata = {
        "architecture": "wan21",
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(
                f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0"
            )
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)  # type: ignore
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)  # type: ignore


def save_text_encoder_output_cache_wan(
    item_info: ItemInfo,
    embed: torch.Tensor,
    preservation_embed: Optional[torch.Tensor] = None,
):
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    if preservation_embed is not None:
        # Save the preservation embedding with a distinct key
        # The key format is chosen to be easy to parse later
        sd[f"varlen_{text_encoder_type}_preservation_{dtype_str}"] = (
            preservation_embed.detach().cpu()
        )

    save_text_encoder_output_cache_common(item_info, sd)


def save_text_encoder_output_cache_common(
    item_info: ItemInfo, sd: dict[str, torch.Tensor]
):
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(
                f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0"
            )
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": "wan21",
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):  # type: ignore
        # load existing cache and update metadata
        with safetensors_utils.MemoryEfficientSafeOpen(
            item_info.text_encoder_output_cache_path
        ) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if (
                    key not in sd
                ):  # avoid overwriting by existing cache, we keep the new one
                    sd[key] = f.get_tensor(key)

        assert (
            existing_metadata["architecture"] == metadata["architecture"]
        ), "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(
                f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite"
            )
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(
            existing_metadata
        )  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(
            item_info.text_encoder_output_cache_path  # type: ignore
        )
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(
        sd, item_info.text_encoder_output_cache_path, metadata=metadata  # type: ignore
    )
