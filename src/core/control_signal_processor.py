## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/hv_train_network.py (Apache)
## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/wan_train_network.py (Apache)
## Based on: https://github.com/spacepxl/WanTraining/blob/main/train_wan_lora.py (Apache)

"""Control signal processing for control LoRA training.

This module handles all control LoRA signal processing, preprocessing, and video saving.
Extracted from wan_network_trainer.py to improve code organization and maintainability.
"""

import argparse
import hashlib
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from accelerate import Accelerator
from torchvision.transforms import v2

import logging
from common.logger import get_logger
from utils.train_utils import clean_memory_on_device
from generation.sampling import save_videos_grid

logger = get_logger(__name__, level=logging.INFO)


class ControlSignalProcessor:
    """Handles control LoRA signal processing and related operations."""

    def __init__(self):
        self.vae = None  # Will be set by the trainer when VAE is loaded
        # Integrate utility processor for reusable ops (preprocess/concat/noise)
        try:
            from conditioning.control_processor import ControlSignalProcessor as _UtilsCSP  # type: ignore

            self._utils_proc = _UtilsCSP()
        except Exception:
            self._utils_proc = None

    def process_control_signal(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        batch: Dict[str, torch.Tensor],
        latents: torch.Tensor,
        network_dtype: torch.dtype,
        vae: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """
        Process control signal for control LoRA training.
        Simplified to match reference implementation exactly.

        Args:
            args: Training arguments
            accelerator: Accelerator instance
            batch: Training batch
            latents: Input latents
            network_dtype: Network dtype
            vae: VAE model for encoding control signals

        Returns:
            Processed control latents or None if not available
        """
        # DISABLED: Cached control signal path - forcing on-the-fly generation for video saving
        # The cached control signal path doesn't support video saving, so we skip it entirely
        # to ensure all control processing goes through the on-the-fly path where video saving happens

        # if "control_signal" in batch:
        #     logger.info("🎯 Found cached control signal in batch")
        #     # ... cached control signal processing (DISABLED)
        #     # return cached_control_latents

        # On-the-fly control generation from raw pixels
        if "pixels" in batch and vae is not None:
            logger.info("🎯 Found pixels in batch for on-the-fly control generation")
            pixels_data = batch["pixels"]

            if isinstance(pixels_data, list) and len(pixels_data) > 0:
                logger.info(
                    f"🎯 Generating control latents on-the-fly from {len(pixels_data)} pixel tensors"
                )

                control_pixels = []
                for pixel_tensor in pixels_data:
                    # Use VAE dtype to avoid dtype mismatch
                    vae_dtype = vae.dtype if vae is not None else torch.float16
                    one_pixels = pixel_tensor.to(device=latents.device, dtype=vae_dtype)
                    if self._utils_proc is not None:
                        # Sync config fields
                        self._utils_proc.control_type = getattr(
                            args, "control_lora_type", "tile"
                        )
                        self._utils_proc.preprocessing = getattr(
                            args, "control_preprocessing", "blur"
                        )
                        self._utils_proc.blur_kernel_size = getattr(
                            args, "control_blur_kernel_size", 15
                        )
                        self._utils_proc.blur_sigma = getattr(
                            args, "control_blur_sigma", 4.0
                        )
                        self._utils_proc.scale_factor = getattr(
                            args, "control_scale_factor", 1.0
                        )
                        self._utils_proc.concatenation_dim = getattr(
                            args, "control_concatenation_dim", -2
                        )

                        # Target shape as (F, H, W); prefer from latents
                        if latents.dim() == 5:
                            target_shape = (
                                latents.shape[2],
                                latents.shape[3],
                                latents.shape[4],
                            )
                        else:
                            target_shape = (
                                latents.shape[1],
                                latents.shape[2],
                                latents.shape[3],
                            )

                        control_pixel = self._utils_proc._process_tile_control(
                            one_pixels, target_shape, latents.device, vae_dtype
                        )
                    else:
                        control_pixel = self.apply_blur_preprocessing_on_the_fly(
                            one_pixels,
                            args,
                        )
                    control_pixels.append(control_pixel)

                vae_device = vae.device
                try:
                    vae.to(latents.device)
                    with torch.no_grad():
                        control_latents = vae.encode(control_pixels)

                        # Optional noise injection
                        if getattr(args, "control_inject_noise", 0.0) > 0:
                            if self._utils_proc is not None:
                                control_latents = [
                                    self._utils_proc.inject_noise(
                                        cl, args.control_inject_noise
                                    )
                                    for cl in control_latents
                                ]
                            else:
                                for i in range(len(control_latents)):
                                    strength = (
                                        torch.rand(1).item() * args.control_inject_noise
                                    )
                                    control_latents[i] += (
                                        torch.randn_like(control_latents[i]) * strength
                                    )

                        if isinstance(control_latents, list):
                            control_latents = torch.stack(control_latents)

                        control_latents = control_latents.to(
                            device=latents.device, dtype=network_dtype
                        )
                        logger.debug(
                            f"On-the-fly control latents shape: {control_latents.shape}"
                        )
                        return control_latents
                finally:
                    vae.to(vae_device)

            else:
                logger.warning(f"Unexpected pixels format: {type(pixels_data)}")

        # No control signal available
        logger.debug("No control signal found in batch")
        return None

    def preprocess_control_reference_style(
        self, pixels: torch.Tensor, args: argparse.Namespace
    ) -> torch.Tensor:
        """
        Apply control preprocessing exactly like in reference implementation.
        Reference: preprocess_control() function in reference_train_wan_lora.py
        """
        control_lora_type = getattr(args, "control_lora_type", "tile")
        control_preprocessing = getattr(args, "control_preprocessing", "blur")

        logger.info(
            f"🎯 preprocess_control_reference_style called with type={control_lora_type}, preprocessing={control_preprocessing}"
        )

        if control_lora_type == "tile" and control_preprocessing == "blur":
            # Reference implementation format conversion: CFHW -> BFCHW
            control = pixels.movedim(0, 1).unsqueeze(0)  # CFHW -> BFCHW
            height, width = control.shape[-2:]

            # Apply blur like in reference implementation
            blur = v2.Compose(
                [
                    v2.Resize(size=(height // 4, width // 4)),
                    v2.Resize(size=(height, width)),
                    v2.GaussianBlur(
                        kernel_size=getattr(args, "control_blur_kernel_size", 15),
                        sigma=getattr(args, "control_blur_sigma", 3.0),
                    ),  # Changed from sigma=4 to sigma=3 to match reference
                ]
            )

            control = torch.clamp(torch.nan_to_num(blur(control)), min=-1, max=1)
            control = control[0].movedim(0, 1)

            # Save control video if enabled using unified path
            if getattr(args, "save_control_videos", False):
                try:
                    self.save_control_video(
                        control,
                        args,
                        f"{control_lora_type}_{control_preprocessing}_reference",
                    )
                except Exception as e:
                    logger.warning(f"Failed to save control video: {e}")

            return control

        else:
            # For other preprocessing types, just return original pixels
            logger.warning(
                f"Control preprocessing '{control_preprocessing}' for type '{control_lora_type}' not implemented, using original pixels"
            )
            return pixels

    def modify_model_for_control_lora(
        self, transformer: Any, args: argparse.Namespace
    ) -> None:
        """
        Modify the model's patch embedding layer to accept additional channels for control LoRA.
        This aligns with the reference implementation.
        """
        # Re-entrancy guard – return early if already patched
        if getattr(transformer, "_control_lora_patched", False):
            logger.debug("Control LoRA patch already applied – skipping.")
            return

        if hasattr(transformer, "patch_embedding"):
            with torch.no_grad():
                in_cls = transformer.patch_embedding.__class__  # nn.Conv3d
                old_in_dim = transformer.in_dim  # 16
                new_in_dim = old_in_dim * 2  # Double the input channels

                new_in = in_cls(
                    in_channels=new_in_dim,
                    out_channels=transformer.patch_embedding.out_channels,
                    kernel_size=transformer.patch_embedding.kernel_size,
                    stride=transformer.patch_embedding.stride,
                    padding=transformer.patch_embedding.padding,
                ).to(
                    device=transformer.patch_embedding.weight.device,
                    dtype=transformer.patch_embedding.weight.dtype,
                )

                new_in.weight.zero_()
                # Copy original weights to first half of new weights
                new_in.weight[:, :old_in_dim, :, :, :] = (
                    transformer.patch_embedding.weight
                )
                # Copy original bias so the behaviour matches the reference implementation
                if transformer.patch_embedding.bias is not None:
                    new_in.bias.copy_(transformer.patch_embedding.bias)

                # Replace the original patch embedding
                transformer.patch_embedding = new_in
                transformer.in_dim = new_in_dim

                # Update HuggingFace config so that any model save/load cycle retains the new input channel size
                if hasattr(transformer, "register_to_config"):
                    # WanModel may inherit from ConfigMixin in some versions
                    transformer.register_to_config(in_dim=new_in_dim)

                logger.info(
                    f"Modified model for control LoRA: input channels {old_in_dim} -> {new_in_dim}"
                )

                # Ensure gradients are enabled for the new patch_embedding so it can learn
                transformer.patch_embedding.requires_grad_(True)

                # mark patched
                transformer._control_lora_patched = True

    def generate_control_signal_on_the_fly(
        self,
        args: argparse.Namespace,
        pixels: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Generate control signal on-the-fly from input pixels.
        Aligned with reference implementation.
        """
        control_type = getattr(args, "control_lora_type", "tile")
        preprocessing = getattr(args, "control_preprocessing", "blur")

        if control_type == "tile" and preprocessing == "blur":
            # Apply blur preprocessing like in reference implementation
            return self.apply_blur_preprocessing_on_the_fly(pixels, args)
        else:
            logger.warning(
                f"On-the-fly control generation not implemented for type: {control_type}, preprocessing: {preprocessing}"
            )
            return None

    def apply_blur_preprocessing_on_the_fly(
        self, pixels: torch.Tensor, args: argparse.Namespace
    ) -> torch.Tensor:
        """
        Apply blur preprocessing on-the-fly like in reference implementation.
        """
        # Convert to CFHW format like in reference implementation
        if pixels.dim() == 4:  # B, C, H, W
            pixels = pixels.movedim(0, 1).unsqueeze(0)  # CFHW -> BFCHW
        elif pixels.dim() == 5:  # B, C, F, H, W
            pixels = pixels.movedim(1, 2)  # BCFHW -> BFCHW

        # Apply blur preprocessing like in reference implementation
        height, width = pixels.shape[-2:]

        # Use configurable sigma
        sigma = getattr(args, "control_blur_sigma", 3.0)
        kernel_size = getattr(args, "control_blur_kernel_size", 15)

        blur = v2.Compose(
            [
                v2.Resize(size=(height // 4, width // 4)),
                v2.Resize(size=(height, width)),
                v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            ]
        )

        # Apply blur to the entire tensor like in reference
        blurred = blur(pixels)

        # Clamp to [-1, 1] like in reference
        result = torch.clamp(torch.nan_to_num(blurred), min=-1, max=1)

        # Convert back to CFHW format like in reference
        result = result[0].movedim(0, 1)  # BFCHW -> CFHW

        # Save control video if enabled
        if hasattr(args, "save_control_videos") and args.save_control_videos:
            logger.info("🎯 Control video saving is enabled")
            save_all = getattr(args, "control_video_save_all", False)
            should_save = False

            if save_all:
                # Save all control videos (every video processed)
                should_save = True
                control_lora_type = getattr(args, "control_lora_type", "tile")
                control_preprocessing = getattr(args, "control_preprocessing", "blur")
                logger.info(
                    f"🎥 Control video saved (save_all mode) for on-the-fly preprocessing: {control_lora_type}_{control_preprocessing}"
                )
            else:
                # Save only one control video per unique input video
                if not hasattr(args, "_control_videos_saved_onthefly"):
                    args._control_videos_saved_onthefly = set()

                control_lora_type = getattr(args, "control_lora_type", "tile")
                control_preprocessing = getattr(args, "control_preprocessing", "blur")

                # Create a unique identifier based on input video content
                video_hash = hashlib.md5(pixels.cpu().numpy().tobytes()).hexdigest()[:8]
                video_id = (
                    f"{control_lora_type}_{control_preprocessing}_onthefly_{video_hash}"
                )

                if video_id not in args._control_videos_saved_onthefly:
                    should_save = True
                    args._control_videos_saved_onthefly.add(video_id)
                    logger.info(
                        f"🎥 Control video saved (new video {video_hash}) for on-the-fly preprocessing: {control_lora_type}_{control_preprocessing}"
                    )
                else:
                    should_save = False
                    logger.debug(
                        f"Skipping control video save (already saved video {video_hash}) for: {control_lora_type}_{control_preprocessing}"
                    )

            if should_save:
                control_lora_type = getattr(args, "control_lora_type", "tile")
                control_preprocessing = getattr(args, "control_preprocessing", "blur")
                self.save_control_video(
                    result,
                    args,
                    f"{control_lora_type}_{control_preprocessing}_onthefly",
                )

        return result

    def save_control_video(
        self, control_tensor: torch.Tensor, args: argparse.Namespace, suffix: str
    ) -> None:
        """
        Save control video to disk for debugging/inspection purposes.

        Args:
            control_tensor: Control tensor in various formats (CFHW, BFCHW, etc.)
            args: Training arguments containing save configuration
            suffix: Suffix to append to filename (e.g., "tile_blur")
        """
        try:
            # Get save directory from args
            save_dir = getattr(args, "control_video_save_dir", "tmp/control_videos")

            # Create absolute path relative to Takenoko root
            if not os.path.isabs(save_dir):
                # Get the project root directory (parent of src)
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                save_dir = os.path.join(project_root, save_dir)

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # According to the code analysis, control_tensor should ALWAYS be in CFHW format
            logger.debug(f"Control tensor shape: {control_tensor.shape}")
            logger.debug(f"Control tensor dtype: {control_tensor.dtype}")

            # Validate that we have the expected CFHW format
            if control_tensor.dim() != 4:
                logger.error(
                    f"BUG: Expected 4D tensor in CFHW format, got {control_tensor.dim()}D tensor with shape {control_tensor.shape}"
                )
                logger.error(f"This indicates a bug in the tensor processing pipeline!")
                logger.error(
                    f"Please check the preprocessing methods preprocess_control_reference_style and apply_blur_preprocessing_on_the_fly"
                )

                # Emergency handling for debugging - try to fix common issues
                if control_tensor.dim() == 3:
                    if control_tensor.shape == (
                        288,
                        512,
                        17,
                    ):  # Your reported case: HWF
                        logger.warning(
                            "Emergency fix: Detected HWF format (288, 512, 17), converting to CFHW"
                        )
                        # HWF -> FHW -> CFHW (add channel dimension)
                        control_tensor = (
                            control_tensor.permute(2, 0, 1)
                            .unsqueeze(0)
                            .repeat(3, 1, 1, 1)
                        )
                        logger.warning(f"After emergency fix: {control_tensor.shape}")
                    else:
                        logger.error(
                            f"Cannot handle 3D tensor with shape {control_tensor.shape}"
                        )
                        raise ValueError(
                            f"Unexpected 3D tensor shape: {control_tensor.shape}. Expected 4D CFHW format."
                        )
                else:
                    raise ValueError(
                        f"Cannot handle {control_tensor.dim()}D tensor. Expected 4D CFHW format."
                    )

            # Extract CFHW dimensions
            C, F, H, W = control_tensor.shape
            logger.debug(f"CFHW format: C={C}, F={F}, H={H}, W={W}")

            # Sanity check: channels should be reasonable (1-32), frames should be > 0
            if C > 32:
                logger.warning(
                    f"Unusual number of channels: {C}. This might indicate wrong tensor format!"
                )
            if F == 0:
                logger.error(f"Zero frames detected: F={F}")
                raise ValueError("Invalid frame count: 0")

            # Convert tensor format for saving: CFHW -> BCTHW (save_videos_grid expects BCTHW)
            control_video = control_tensor.unsqueeze(
                0
            )  # CFHW -> BCFHW (Frame=Time, so this is BCTHW)

            # Check current value range and normalize appropriately
            min_val = control_video.min().item()
            max_val = control_video.max().item()
            logger.debug(f"Control video value range: [{min_val:.3f}, {max_val:.3f}]")

            if min_val >= 0 and max_val <= 1:
                # Already in [0, 1] range
                logger.debug("Control video already in [0, 1] range")
            elif min_val >= -1 and max_val <= 1:
                # In [-1, 1] range, normalize to [0, 1]
                logger.debug("Normalizing from [-1, 1] to [0, 1] range")
                control_video = (control_video + 1.0) / 2.0
            else:
                # Unknown range, normalize by min-max scaling
                logger.debug(
                    f"Normalizing from [{min_val:.3f}, {max_val:.3f}] to [0, 1] range"
                )
                control_video = (control_video - min_val) / (max_val - min_val + 1e-8)

            control_video = torch.clamp(control_video, 0, 1)

            # Convert to float32 and move to CPU for saving (video saving requires float32 CPU tensors)
            control_video = control_video.to(dtype=torch.float32).cpu()

            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"control_{suffix}_{timestamp}.mp4"
            save_path = os.path.join(save_dir, filename)

            logger.debug(f"Saving control video with shape: {control_video.shape}")
            logger.debug(f"Control video dtype: {control_video.dtype}")
            logger.debug(f"Control video device: {control_video.device}")

            # Save video using WAN target FPS (16fps) instead of hardcoded 8fps
            try:
                target_fps = 16  # TARGET_FPS_WAN constant from VideoDataset
                save_videos_grid(
                    control_video, save_path, rescale=False, fps=target_fps
                )
                logger.info(f"Control video saved to: {save_path} (fps={target_fps})")
            except Exception as save_error:
                logger.error(f"Error in save_videos_grid: {save_error}")
                logger.debug(
                    f"Tensor info - shape: {control_video.shape}, dtype: {control_video.dtype}"
                )

                # Try alternative: save as individual frames
                try:
                    import torchvision

                    logger.info("Attempting to save as individual frames instead...")
                    frames_dir = save_path.replace(".mp4", "_frames")
                    os.makedirs(frames_dir, exist_ok=True)

                    # Extract frames: BCTHW -> individual frames
                    B, C, T, H, W = control_video.shape
                    for t in range(T):
                        frame = control_video[0, :, t]  # CHW
                        frame_path = os.path.join(frames_dir, f"frame_{t:04d}.png")
                        torchvision.utils.save_image(frame, frame_path)

                    logger.info(f"Control frames saved to: {frames_dir}")
                except Exception as frame_error:
                    logger.warning(f"Failed to save frames: {frame_error}")
                    raise save_error

        except Exception as e:
            logger.warning(f"⚠️  Failed to save control video: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
