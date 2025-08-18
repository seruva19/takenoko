## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/cache_latents.py (Apache)

import argparse
import os
from typing import Any, Optional, Union
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch

from dataset.cache import save_latent_cache_wan
from dataset.image_video_dataset import BaseDataset
from dataset.item_info import ItemInfo
from wan.modules.vae import WanVAE

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def show_image(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],  # type: ignore
) -> int:
    """Display image using OpenCV window"""
    import cv2

    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3)
        or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")  # type: ignore
        else:
            print(f"Image: {img.shape}")  # type: ignore
        cv2_img = np.array(img) if isinstance(img, Image.Image) else img
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == ord("q") or k == ord("d"):
            return k
    return k


def show_console(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],  # type: ignore
    width: int,
    back: str,
    interactive: bool = False,
) -> int:
    """Display image using ASCII art in console"""
    from ascii_magic import from_pillow_image, Back  # type: ignore

    # Map provided background name to ascii_magic Back if specified
    if back is not None:
        back = getattr(Back, back.upper())

    k = None
    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3)
        or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")  # type: ignore
        else:
            print(f"Image: {img.shape}")  # type: ignore
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        ascii_img = from_pillow_image(pil_img)
        ascii_img.to_terminal(columns=width, back=back)

        if interactive:
            k = input("Press q to quit, d to next dataset, other key to next: ")
            if k == "q" or k == "d":
                return ord(k)

    if not interactive:
        return ord(" ")
    return ord(k) if k else ord(" ")


def save_video(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],  # type: ignore
    cache_path: str,
    fps: int = 24,
):
    """Save video or image to file"""
    import av

    directory = os.path.dirname(cache_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(
        image, Image.Image
    ):
        # save image
        image_path = cache_path.replace(".safetensors", ".jpg")
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        img.save(image_path)
        print(f"Saved image: {image_path}")
    else:
        imgs = image
        print(f"Number of images: {len(imgs)}")
        # save video
        video_path = cache_path.replace(".safetensors", ".mp4")
        # Determine frame dimensions for stream robustly
        first_frame = imgs[0]
        if isinstance(first_frame, Image.Image):
            first_np = np.array(first_frame)
            height, width = first_np.shape[0:2]
        else:
            height, width = first_frame.shape[0:2]  # numpy array (H, W, C)

        # create output container
        container = av.open(video_path, mode="w")

        # create video stream
        codec = "libx264"
        pixel_format = "yuv420p"
        stream = container.add_stream(codec, rate=fps)
        stream.width = width  # type: ignore
        stream.height = height  # type: ignore
        stream.pix_fmt = pixel_format  # type: ignore
        stream.bit_rate = 1000000  # type: ignore # 1Mbit/s for preview quality

        for frame_img in imgs:
            if isinstance(frame_img, Image.Image):
                frame = av.VideoFrame.from_image(frame_img)
            else:
                frame = av.VideoFrame.from_ndarray(frame_img, format="rgb24")
            packets = stream.encode(frame)  # type: ignore
            for packet in packets:
                container.mux(packet)

        for packet in stream.encode():  # type: ignore
            container.mux(packet)

        container.close()

        print(f"Saved video: {video_path}")


def show_datasets(
    datasets: list[BaseDataset],
    debug_mode: str,
    console_width: int,
    console_back: str,
    console_num_images: Optional[int],
    fps: int = 24,
):
    """Display datasets for debugging purposes"""
    if debug_mode != "video":
        print(f"d: next dataset, q: quit")

    num_workers = max(1, os.cpu_count() - 1)  # type: ignore
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        batch_index = 0
        num_images_to_show = console_num_images
        k = None
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {key}, count: {len(batch)}")
            for j, item_info in enumerate(batch):
                item_info: ItemInfo
                print(f"{batch_index}-{j}: {item_info}")
                if debug_mode == "image":
                    k = show_image(item_info.content)  # type: ignore
                elif debug_mode == "console":
                    k = show_console(
                        item_info.content,  # type: ignore
                        console_width,
                        console_back,
                        console_num_images is None,
                    )
                    if num_images_to_show is not None:
                        num_images_to_show -= 1
                        if num_images_to_show == 0:
                            k = ord("d")  # next dataset
                elif debug_mode == "video":
                    save_video(item_info.content, item_info.latent_cache_path, fps)  # type: ignore
                    k = None  # save next video

                if k == ord("q"):
                    return
                elif k == ord("d"):
                    break
            if k == ord("d"):
                break
            batch_index += 1


def encode_and_save_batch(
    vae: WanVAE,
    batch: list[ItemInfo],
    args: Optional[argparse.Namespace] = None,
):
    """Encode and save a batch of items using WAN VAE"""
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(
            f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}"
        )

    # Encode main content
    with (
        torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype),  # type: ignore
        torch.no_grad(),
    ):
        latent = vae.encode(contents)  # list of Tensor[C, F, H, W]
    latent = torch.stack(latent, dim=0)  # B, C, F, H, W
    latent = latent.to(vae.dtype)  # convert to bfloat16

    # Handle control signals
    control_latent = None
    if hasattr(batch[0], "control_content") and batch[0].control_content is not None:
        # Process control signals
        control_contents = torch.stack(
            [torch.from_numpy(item.control_content) for item in batch]
        )
        if len(control_contents.shape) == 4:
            control_contents = control_contents.unsqueeze(
                1
            )  # B, H, W, C -> B, F, H, W, C

        control_contents = control_contents.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # B, C, F, H, W
        control_contents = control_contents.to(vae.device, dtype=vae.dtype)
        control_contents = control_contents / 127.5 - 1.0  # normalize to [-1, 1]

        # Apply blurring preprocessing before VAE encoding (matching training behavior)
        control_lora_type = getattr(args, "control_lora_type", "tile")
        control_preprocessing = getattr(args, "control_preprocessing", "blur")
        control_blur_kernel_size = getattr(args, "control_blur_kernel_size", 15)
        control_blur_sigma = getattr(args, "control_blur_sigma", 3.0)

        if control_lora_type == "tile" and control_preprocessing == "blur":
            # Apply blur preprocessing like in training
            from torchvision.transforms import v2

            # Convert to BFCHW format for preprocessing
            control_contents = control_contents.movedim(1, 2)  # BCFHW -> BFCHW

            # Apply blur preprocessing
            height, width = control_contents.shape[-2:]
            blur = v2.Compose(
                [
                    v2.Resize(size=(height // 4, width // 4)),
                    v2.Resize(size=(height, width)),
                    v2.GaussianBlur(
                        kernel_size=control_blur_kernel_size,
                        sigma=control_blur_sigma,
                    ),
                ]
            )

            # Apply blur to each batch item
            blurred_contents = []
            for i in range(control_contents.shape[0]):
                blurred_item = blur(control_contents[i])
                blurred_contents.append(blurred_item)

            control_contents = torch.stack(blurred_contents, dim=0)
            control_contents = torch.clamp(
                torch.nan_to_num(control_contents), min=-1, max=1
            )

            # Convert back to BCFHW format for VAE encoding
            control_contents = control_contents.movedim(1, 2)  # BFCHW -> BCFHW

        with torch.no_grad():
            control_latent = vae.encode(control_contents)
            control_latent = torch.stack(control_latent, dim=0)  # list to tensor

    # Save main latents
    for i, item in enumerate(batch):
        l = latent[i]
        cctx = None
        y_i = None
        save_latent_cache_wan(item, l, cctx, y_i)

    # Save control latents if available
    if control_latent is not None:
        for item, cl in zip(batch, control_latent):
            control_cache_path = item.latent_cache_path.replace(  # type: ignore
                ".safetensors", "_control.safetensors"
            )
            # Create a temporary ItemInfo for control latent
            control_item = ItemInfo(
                item_key=item.item_key,
                caption=item.caption,
                original_size=item.original_size,
                bucket_size=item.bucket_size,
                frame_count=item.frame_count,
                content=item.content,
                latent_cache_path=control_cache_path,
                weight=item.weight,
            )
            save_latent_cache_wan(control_item, cl, None, None)

    # Save mask tensors if provided (per-pixel weights), to be used at training time
    # Expect shape per item: (F, H, W) or (H, W) for images.
    if hasattr(batch[0], "mask_content") and batch[0].mask_content is not None:
        from safetensors.torch import save_file as _save_file

        for item in batch:
            try:
                mask_obj = item.mask_content
                mask_array: Optional[np.ndarray] = None  # type: ignore
                if isinstance(mask_obj, np.ndarray):
                    mask_array = mask_obj
                elif isinstance(mask_obj, (list, tuple)) and len(mask_obj) > 0 and isinstance(mask_obj[0], np.ndarray):  # type: ignore
                    mask_array = np.stack(mask_obj, axis=0)
                else:
                    # Unsupported or empty
                    continue

                # Ensure shape (F, H, W)
                if mask_array.ndim == 2:
                    mask_array = mask_array[None, ...]

                mask_tensor = torch.from_numpy(mask_array).to(torch.float32)
                # Normalize to [0,1] if values appear in [0,255]
                if torch.isfinite(mask_tensor).any() and mask_tensor.max() > 1.0:
                    mask_tensor = mask_tensor / 255.0

                F, H, W = (
                    int(mask_tensor.shape[0]),
                    int(mask_tensor.shape[-2]),
                    int(mask_tensor.shape[-1]),
                )
                key_name = f"latents_{F}x{H}x{W}_float32"

                # Persist alongside latent cache for the same item
                mask_cache_path = item.latent_cache_path.replace(".safetensors", "_mask.safetensors")  # type: ignore
                _save_file({key_name: mask_tensor.cpu()}, mask_cache_path)
            except Exception as e:
                logger.warning(f"Failed to save mask cache for {item.item_key}: {e}")


def encode_datasets(
    datasets: list[BaseDataset], encode: callable, args: argparse.Namespace  # type: ignore
):
    """Encode all datasets using the provided encode function"""
    num_workers = (
        args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)  # type: ignore
    )
    # Optional purge step: remove all existing latent cache files in each dataset cache dir
    if getattr(args, "purge_before_run", False):
        total_purged = 0
        for dataset in datasets:
            try:
                for cache_file in dataset.get_all_latent_cache_files():
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        total_purged += 1
            except Exception as e:
                logger.warning(f"Failed to purge latent cache files: {e}")
        logger.info(f"🧹 Purged {total_purged} latent cache files before caching")
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            all_latent_cache_paths.extend([item.latent_cache_path for item in batch])

            if args.skip_existing:
                filtered_batch = [
                    item for item in batch if not os.path.exists(item.latent_cache_path)
                ]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])

        # normalize paths
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)

        # remove old cache files not in the dataset
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")
