## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/hv_generate_video.py (Apache)

import os
from typing import Union

import numpy as np
import torch
import torchvision
from transformers.models.llama import LlamaModel
import av
from einops import rearrange
from PIL import Image

from utils.vae_utils import load_vae

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype

import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def clean_memory_on_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    # Safety check for empty or None path
    if not path or not path.strip():
        raise ValueError(
            f"Invalid video save path: '{path}'. Path cannot be empty or None."
        )

    logger.info(f"save_videos_grid called with path='{path}'")

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # Safety check for directory creation
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if dirname is not empty
        logger.info(f"Creating directory: {dirname}")
        os.makedirs(dirname, exist_ok=True)
    else:
        logger.warning(f"Empty directory name for path: {path}")

    # # save video with av
    # container = av.open(path, "w")
    # stream = container.add_stream("libx264", rate=fps)
    # for x in outputs:
    #     frame = av.VideoFrame.from_ndarray(x, format="rgb24")
    #     packet = stream.encode(frame)
    #     container.mux(packet)
    # packet = stream.encode(None)
    # container.mux(packet)
    # container.close()

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width  # type: ignore
    stream.height = height  # type: ignore
    stream.pix_fmt = pixel_format  # type: ignore
    stream.bit_rate = 4000000  # type: ignore # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)  # type: ignore
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():  # type: ignore
        container.mux(packet)

    container.close()


def save_images_grid(
    videos: torch.Tensor,
    parent_dir: str,
    image_name: str,
    rescale: bool = False,
    n_rows: int = 1,
    create_subdir=True,
):
    # Safety check for empty or None paths
    if not parent_dir or not parent_dir.strip():
        raise ValueError(
            f"Invalid parent_dir: '{parent_dir}'. Parent directory cannot be empty or None."
        )
    if not image_name or not image_name.strip():
        raise ValueError(
            f"Invalid image_name: '{image_name}'. Image name cannot be empty or None."
        )

    logger.info(
        f"save_images_grid called with parent_dir='{parent_dir}', image_name='{image_name}'"
    )

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if create_subdir:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        logger.info(f"Saving image to: {image_path}")
        image = Image.fromarray(x)
        image.save(image_path)


def prepare_vae(args, device):
    vae_dtype = (
        torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    )
    vae = load_vae(args, config=None, device=device, dtype=vae_dtype)
    vae.eval()
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # set chunk_size to CausalConv3d recursively
    chunk_size = args.vae_chunk_size
    if chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(chunk_size)  # type: ignore
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)  # type: ignore
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size  # type: ignore
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8  # type: ignore
    # elif args.vae_tiling:
    else:
        vae.enable_spatial_tiling(True)  # type: ignore

    return vae, vae_dtype


def encode_to_latents(args, video, device):
    vae, vae_dtype = prepare_vae(args, device)

    video = video.to(device=device, dtype=vae_dtype)
    video = video * 2 - 1  # 0, 1 -> -1, 1
    with torch.no_grad():
        latents = vae.encode(video).latent_dist.sample()  # type: ignore

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:  # type: ignore
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor  # type: ignore
    else:
        latents = latents * vae.config.scaling_factor  # type: ignore

    return latents


def decode_latents(args, latents, device):
    vae, vae_dtype = prepare_vae(args, device)

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
        pass
    else:
        raise ValueError(
            f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
        )

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:  # type: ignore
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor  # type: ignore
    else:
        latents = latents / vae.config.scaling_factor  # type: ignore

    latents = latents.to(device=device, dtype=vae_dtype)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]  # type: ignore

    if expand_temporal_dim:
        image = image.squeeze(2)

    image = (image / 2 + 0.5).clamp(0, 1)
    # always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().float()

    return image


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
        )
    return height, width, video_length
