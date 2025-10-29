"""Sampling and inference management for WAN network trainer.

This module handles all image/video sampling, inference, and related functionality.
Extracted from wan_network_trainer.py to improve code organization and maintainability.
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from tqdm import tqdm
from accelerate import Accelerator, PartialState

import logging
from dataset.image_video_dataset import TARGET_FPS_WAN
from common.logger import get_logger
from utils.train_utils import clean_memory_on_device, should_sample_images, load_prompts
from common.model_downloader import download_model_if_needed
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from generation.sampling import save_images_grid, save_videos_grid
from torch.utils.tensorboard.writer import SummaryWriter
from energy_based.eqm_mode.integration import EqMModeContext, setup_eqm_mode
from energy_based.eqm_mode.eval import save_npz_from_samples
from energy_based.eqm_mode.sampling_helper import EqMSamplingResult, run_eqm_sampling

logger = get_logger(__name__, level=logging.INFO)


class SamplingManager:
    """Handles image/video sampling and inference operations."""

    def __init__(self, config: Dict[str, Any], default_guidance_scale: float = 5.0):
        self.config = config
        self.default_guidance_scale = default_guidance_scale
        self._vae_config = None  # Store VAE config for lazy loading
        self._last_eqm_likelihood: Optional[Dict[str, torch.Tensor]] = None

    def set_vae_config(self, vae_config: Dict[str, Any]) -> None:
        """Set VAE configuration for lazy loading."""
        self._vae_config = vae_config

    def _load_vae_lazy(self) -> Optional[WanVAE]:
        """Load VAE on-demand for sampling."""
        if self._vae_config is None:
            return None

        logger.info("Loading VAE on-demand for sampling...")
        from core.model_manager import ModelManager

        model_manager = ModelManager()

        vae = model_manager.load_vae(
            self._vae_config["args"],
            vae_dtype=self._vae_config["vae_dtype"],
            vae_path=self._vae_config["vae_path"],
        )
        vae.requires_grad_(False)
        vae.eval()

        return vae

    def _unload_vae(self, vae: Optional[WanVAE]) -> None:
        """Unload VAE from memory after use."""
        if vae is not None:
            logger.info("Unloading VAE from memory after sampling...")
            vae.to("cpu")
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()

    def sample_images(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        epoch: Optional[int],
        steps: int,
        vae: WanVAE,
        transformer: Any,
        sample_parameters: Optional[List[Dict[str, Any]]],
        dit_dtype: torch.dtype,
        dual_model_manager: Optional[Any] = None,
    ) -> None:
        """Architecture independent sample images generation."""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info(f"🖼️ Generating sample images at step: {steps}")
        if sample_parameters is None:
            logger.warning("No sample parameters available, skipping sample generation")
            return

        # Handle lazy VAE loading if vae is None
        should_unload_vae = False
        if vae is None and self._vae_config is not None:
            vae = self._load_vae_lazy()
            should_unload_vae = True
            if vae is None:
                logger.warning("Failed to load VAE for sampling, skipping...")
                return

        distributed_state = (
            PartialState()
        )  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

        # Use the unwrapped model
        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()

        # Safety check for output_dir
        if not args.output_dir or not args.output_dir.strip():
            logger.error(
                f"args.output_dir is empty or None: '{args.output_dir}'. Cannot save samples."
            )
            return

        save_dir = os.path.join(args.output_dir, "sample")
        logger.info(f"save_dir={save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        # save random state to restore later
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
        except Exception:
            pass

        if distributed_state.num_processes <= 1:
            # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
            with torch.no_grad(), accelerator.autocast():  # type: ignore[misc]
                # Create a single SummaryWriter instance for all videos to avoid file locking issues
                writer = None
                if hasattr(args, "logging_dir"):
                    try:
                        from tensorboardX import SummaryWriter

                        use_tensorboardx = True
                    except ImportError:
                        from torch.utils.tensorboard.writer import SummaryWriter

                        use_tensorboardx = False
                    writer = SummaryWriter(log_dir=args.logging_dir)

                for sample_parameter in sample_parameters:
                    self.sample_image_inference(
                        accelerator,
                        args,
                        transformer,
                        dit_dtype,
                        vae,
                        save_dir,
                        sample_parameter,
                        epoch,
                        steps,
                        writer=writer,
                        use_tensorboardx=use_tensorboardx if writer else False,
                        dual_model_manager=dual_model_manager,
                    )
                    clean_memory_on_device(accelerator.device)

                # Close the writer after all videos are logged
                if writer:
                    writer.close()
                    import gc

                    gc.collect()
        else:
            # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
            # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
            per_process_params = []  # list of lists
            for i in range(distributed_state.num_processes):
                per_process_params.append(
                    sample_parameters[i :: distributed_state.num_processes]
                )

            with torch.no_grad():
                # Create a single SummaryWriter instance for all videos to avoid file locking issues
                writer = None
                if hasattr(args, "logging_dir"):
                    try:
                        from tensorboardX import SummaryWriter

                        use_tensorboardx = True
                    except ImportError:
                        from torch.utils.tensorboard.writer import SummaryWriter

                        use_tensorboardx = False
                    writer = SummaryWriter(log_dir=args.logging_dir)

                with distributed_state.split_between_processes(
                    per_process_params
                ) as sample_parameter_lists:
                    for sample_parameter in sample_parameter_lists[0]:
                        self.sample_image_inference(
                            accelerator,
                            args,
                            transformer,
                            dit_dtype,
                            vae,
                            save_dir,
                            sample_parameter,  # type: ignore
                            epoch,
                            steps,
                            writer=writer,
                            use_tensorboardx=use_tensorboardx if writer else False,
                            dual_model_manager=dual_model_manager,
                        )
                        clean_memory_on_device(accelerator.device)

                # Close the writer after all videos are logged
                if writer:
                    writer.close()
                    import gc

                    gc.collect()

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

        # Unload VAE if it was loaded lazily
        if should_unload_vae:
            self._unload_vae(vae)

    def sample_image_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer: Any,
        dit_dtype: torch.dtype,
        vae: WanVAE,
        save_dir: str,
        sample_parameter: Dict[str, Any],
        epoch: Optional[int],
        steps: int,
        writer: Optional[Any] = None,
        use_tensorboardx: bool = False,
        dual_model_manager: Optional[Any] = None,
    ) -> None:
        """Architecture independent sample images inference. Logs generated video to TensorBoard if possible."""
        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get(
            "width", 256
        )  # make smaller for faster and memory saving inference
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 1)
        guidance_scale = sample_parameter.get(
            "guidance_scale", self.default_guidance_scale
        )
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 7)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get(
            "cfg_scale", None
        )  # None for architecture default
        # If cfg_scale not explicitly provided, fall back to guidance_scale from prompt,
        # then to global args.guidance_scale if present
        if cfg_scale is None:
            if guidance_scale is not None:
                cfg_scale = guidance_scale
            elif hasattr(args, "guidance_scale"):
                cfg_scale = getattr(args, "guidance_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)

        # round width and height to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        frame_count = (frame_count - 1) // 4 * 4 + 1  # 1, 5, 9, 13, ...

        image_path = None
        control_video_path = None

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # True random sample image generation
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"💬 prompt: {prompt}")
        logger.info(f"🖼️ height: {height}")
        logger.info(f"🖼️ width: {width}")
        logger.info(f"🎥 frame count: {frame_count}")
        logger.info(f"⚡ sample steps: {sample_steps}")
        logger.info(f"🎯 guidance scale: {guidance_scale}")
        logger.info(f"🔄 discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"🎲 seed: {seed}")

        do_classifier_free_guidance = False
        if negative_prompt is not None:
            do_classifier_free_guidance = True
            logger.info(f"🚫 negative prompt: {negative_prompt}")
            logger.info(f"⚙️ cfg scale: {cfg_scale}")

        video = self.do_inference(
            accelerator,
            args,
            sample_parameter,
            vae,
            dit_dtype,
            transformer,
            discrete_flow_shift,
            sample_steps,
            width,
            height,
            frame_count,
            generator,
            do_classifier_free_guidance,
            guidance_scale,
            cfg_scale,
            image_path=image_path,
            control_video_path=control_video_path,
            dual_model_manager=dual_model_manager,
        )

        # Save video
        if video is None:
            logger.error("No video was generated for the given sample parameters.")
            return

        # --- TensorBoardX video logging ---

        try:
            if writer is not None:
                # Create a copy for TensorBoard logging to avoid modifying the original video tensor
                # TensorBoard expects (N, T, C, H, W) format
                video_to_log = (
                    video.permute(0, 2, 1, 3, 4).detach().cpu().float().clamp(0, 1)
                )

                # Make tag unique by including prompt index to avoid overwriting multiple videos at same step
                prompt_idx = sample_parameter.get("enum", 0)
                tag = f"generated/sample_video_step_{steps}_prompt_{prompt_idx:02d}"

                if use_tensorboardx:
                    # TensorBoardX supports better video logging
                    # Convert to uint8 format for better compatibility
                    video_to_log_uint8 = (
                        (video_to_log * 255).clamp(0, 255).to(torch.uint8)
                    )
                    writer.add_video(
                        tag, video_to_log_uint8, global_step=steps, fps=TARGET_FPS_WAN
                    )
                    logger.info(f"🎬 Logged video to TensorBoardX: {tag}")
                else:
                    # Standard TensorBoard video logging
                    writer.add_video(
                        tag, video_to_log, global_step=steps, fps=TARGET_FPS_WAN
                    )
                    logger.info(f"🎬 Logged video to TensorBoard: {tag}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to log video to TensorBoard: {e}")

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)

        # Safety check for output_name
        output_name_prefix = ""
        if args.output_name is not None and args.output_name.strip():
            output_name_prefix = args.output_name + "_"
        else:
            logger.warning(f"args.output_name is None or empty: '{args.output_name}'")

        save_path = (
            f"{output_name_prefix}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        logger.info(f"save_path={save_path}")
        logger.info(f"video.shape={video.shape}")
        logger.info(f"save_dir={save_dir}")

        if video.shape[2] == 1:
            logger.info(f"Saving as images grid")
            save_images_grid(video, save_dir, save_path, create_subdir=False)
        else:
            logger.info(
                f"Saving as video grid to {os.path.join(save_dir, save_path) + '.mp4'}"
            )
            save_videos_grid(video, os.path.join(save_dir, save_path) + ".mp4")

        # Check if EqM mode is enabled for NPZ export
        eqm_enabled = getattr(args, "enable_eqm_mode", False)
        if eqm_enabled and getattr(args, "eqm_save_npz", False):
            try:
                npz_dir = getattr(args, "eqm_npz_dir", save_dir) or save_dir
                prefix = f"{save_path}_step{steps}"
                limit = getattr(args, "eqm_npz_limit", None)
                save_npz_from_samples(
                    video.squeeze(0), npz_dir, prefix=prefix, limit=limit
                )
                logger.info(f"EqM NPZ exported to {npz_dir}")
            except Exception as exc:
                logger.warning(f"Failed to export EqM NPZ: {exc}")

        # Move models back to initial state
        vae.to("cpu")
        clean_memory_on_device(device)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: Union[str, List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Process sample prompts and prepare embeddings."""
        config = self.config
        device = accelerator.device
        t5_path, clip_path, fp8_t5 = args.t5, args.clip, args.fp8_t5

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        logger.info(f"sample_prompts type = {type(sample_prompts)}")
        logger.info(f"sample_prompts value = {sample_prompts}")

        # Handle both file paths and direct lists of prompt dictionaries
        if isinstance(sample_prompts, str):
            # If it's a file path, load prompts from file
            if not sample_prompts.strip():
                logger.warning(
                    "Empty sample_prompts string provided, skipping sample generation"
                )
                return None
            logger.info(f"Loading prompts from file: {sample_prompts}")
            prompts = load_prompts(sample_prompts)
        elif isinstance(sample_prompts, list):
            # If it's already a list of prompt dictionaries, use it directly
            logger.info(f"Using direct list of prompts, count: {len(sample_prompts)}")
            prompts = []
            for i, prompt_dict in enumerate(sample_prompts):
                # Convert the new structure to the expected format
                converted_dict = {}
                if "text" in prompt_dict:
                    converted_dict["prompt"] = prompt_dict["text"]
                if "width" in prompt_dict:
                    converted_dict["width"] = prompt_dict["width"]
                if "height" in prompt_dict:
                    converted_dict["height"] = prompt_dict["height"]
                if "frames" in prompt_dict:
                    converted_dict["frame_count"] = prompt_dict["frames"]
                if "seed" in prompt_dict:
                    converted_dict["seed"] = prompt_dict["seed"]
                if "step" in prompt_dict:
                    converted_dict["sample_steps"] = prompt_dict["step"]
                if "control_path" in prompt_dict:
                    converted_dict["control_path"] = prompt_dict["control_path"]
                if "control_video_path" in prompt_dict:
                    converted_dict["control_path"] = prompt_dict["control_video_path"]
                # Guidance / CFG controls
                if "guidance_scale" in prompt_dict:
                    converted_dict["guidance_scale"] = prompt_dict["guidance_scale"]
                if "cfg_scale" in prompt_dict:
                    converted_dict["cfg_scale"] = prompt_dict["cfg_scale"]
                # Add other fields as needed
                converted_dict["enum"] = i
                prompts.append(converted_dict)
        else:
            raise ValueError(
                f"sample_prompts must be a string (file path) or list, got {type(sample_prompts)}"
            )

        if not prompts:
            logger.warning("No prompts found, skipping sample generation")
            return None

        logger.info(f"Processed {len(prompts)} prompts")

        def encode_for_text_encoder(text_encoder):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            # with accelerator.autocast(), torch.no_grad(): # this causes NaN if dit_dtype is fp16
            t5_dtype = config.t5_dtype  # type: ignore
            with (
                torch.autocast(device_type=device.type, dtype=t5_dtype),
                torch.no_grad(),
            ):
                for prompt_dict in prompts:
                    if "negative_prompt" not in prompt_dict:
                        prompt_dict["negative_prompt"] = self.config[
                            "sample_neg_prompt"
                        ]
                    for p in [
                        prompt_dict.get("prompt", ""),
                        prompt_dict.get("negative_prompt", None),
                    ]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            prompt_outputs = text_encoder([p], device)
                            sample_prompts_te_outputs[p] = prompt_outputs

            return sample_prompts_te_outputs

        # Download T5 model if it's a URL
        if t5_path.startswith(("http://", "https://")):
            logger.info(f"Detected URL for T5 model, downloading: {t5_path}")
            cache_dir = getattr(args, "model_cache_dir", None)
            t5_path = download_model_if_needed(t5_path, cache_dir=cache_dir)
            logger.info(f"Downloaded T5 model to: {t5_path}")

        # Load Text Encoder 1 and encode
        logger.info(f"loading T5: {t5_path}")
        t5 = T5EncoderModel(
            text_len=config.text_len,  # type: ignore
            dtype=config.t5_dtype,  # type: ignore
            device=device,
            weight_path=t5_path,
            fp8=fp8_t5,
        )

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(t5)
        del t5

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_t5_embeds"] = te_outputs_1[p][0]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict[str, Any],
        vae: WanVAE,
        dit_dtype: torch.dtype,
        transformer: WanModel,
        discrete_flow_shift: float,
        sample_steps: int,
        width: int,
        height: int,
        frame_count: int,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        image_path: Optional[str] = None,
        control_video_path: Optional[str] = None,
        dual_model_manager: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        model: WanModel = transformer
        device = accelerator.device
        if cfg_scale is None:
            cfg_scale = 5.0
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # Ensure model is properly modified for control LoRA during inference
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            # Prefer centralized patching via ControlSignalProcessor if available
            try:
                from trainer.control_signal_processor import ControlSignalProcessor as _CSP  # type: ignore

                _CSP().modify_model_for_control_lora(model, args)
            except Exception:
                # Fallback to local implementation
                if not getattr(model, "_control_lora_patched", False):
                    logger.info("Modifying model for control LoRA during inference...")
                    self._modify_model_for_control_lora(model, args)

            # Validate patch embedding channel count
            if hasattr(model, "patch_embedding") and hasattr(
                model.patch_embedding, "in_channels"
            ):
                expected_channels = getattr(
                    model, "in_dim", model.patch_embedding.in_channels
                )
                actual_channels = model.patch_embedding.in_channels
                assert actual_channels == expected_channels, (
                    f"Patch embedding in_channels={actual_channels} but model.in_dim={expected_channels}. "
                    "Control LoRA patching may have been skipped or applied multiple times."
                )

        # Calculate latent video length based on VAE version
        latent_video_length = (frame_count - 1) // self.config["vae_stride"][0] + 1

        # Get embeddings
        context = sample_parameter["t5_embeds"].to(device=device)
        if do_classifier_free_guidance:
            context_null = sample_parameter["negative_t5_embeds"].to(device=device)
        else:
            context_null = None

        num_channels_latents = 16  # model.in_dim
        vae_scale_factor = self.config["vae_stride"][1]

        # Initialize latents
        lat_h = height // vae_scale_factor
        lat_w = width // vae_scale_factor
        shape_or_frame = (1, num_channels_latents, 1, lat_h, lat_w)
        latents = []
        for _ in range(latent_video_length):
            latents.append(
                torch.randn(
                    shape_or_frame,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            )
        latents = torch.cat(latents, dim=2)

        image_latents = None

        # Check if control LoRA is enabled and needs control signal
        control_latents = None
        extracted_properties = None
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            control_latents, extracted_properties = (
                self._generate_control_latents_for_inference(
                    args, vae, device, width, height, frame_count, sample_parameter
                )
            )

            # Use extracted properties to override parameters if available
            if (
                extracted_properties
                and extracted_properties["source"] != "fallback_zero_signal"
            ):
                # Only override frame_count if not explicitly specified in sample_parameter
                if "frame_count" not in sample_parameter or sample_parameter.get(
                    "auto_extract_from_control", True
                ):
                    logger.info(
                        f"🎯 Using frame count from control signal: {extracted_properties['frame_count']} (was {frame_count})"
                    )
                    frame_count = extracted_properties["frame_count"]
                    # Recalculate latent dimensions with new frame count
                    latent_video_length = (frame_count - 1) // self.config[
                        "vae_stride"
                    ][0] + 1

                # Override resolution if not explicitly specified in sample_parameter
                if "width" not in sample_parameter or sample_parameter.get(
                    "auto_extract_from_control", True
                ):
                    logger.info(
                        f"🎯 Using width from control signal: {extracted_properties['width']} (was {width})"
                    )
                    width = extracted_properties["width"]

                if "height" not in sample_parameter or sample_parameter.get(
                    "auto_extract_from_control", True
                ):
                    logger.info(
                        f"🎯 Using height from control signal: {extracted_properties['height']} (was {height})"
                    )
                    height = extracted_properties["height"]

                # Recalculate latent dimensions with new resolution
                vae_scale_factor = self.config["vae_stride"][1]
                lat_h = height // vae_scale_factor
                lat_w = width // vae_scale_factor

        eqm_enabled = getattr(args, "enable_eqm_mode", False)

        # Generate noise for the required number of frames only (keep on device)
        noise = torch.randn(
            16,
            latent_video_length,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=generator,
            device=device,
        )

        patch_size = tuple(self.config["patch_size"])  # type: ignore

        if eqm_enabled:
            eqm_context = setup_eqm_mode(args)
            if not eqm_context.warning_emitted and (
                getattr(args, "enable_control_lora", False)
                or getattr(args, "enable_controlnet", False)
            ):
                logger.warning(
                    "EqM sampling currently ignores ControlNet / Control LoRA signals."
                )
                eqm_context.warning_emitted = True
            latent = self._eqm_sample_latents(
                eqm_context=eqm_context,
                accelerator=accelerator,
                transformer=model,
                initial_latent=noise,
                context_tokens=context,
                negative_context=context_null,
                sample_parameter=sample_parameter,
                sample_steps=sample_steps,
                cfg_scale=cfg_scale,
                device=device,
                network_dtype=dit_dtype,
                patch_size=patch_size,
                args=args,
            )
        else:
            # use the default value for num_train_timesteps (1000)
            scheduler = FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(
                sample_steps, device=device, shift=discrete_flow_shift
            )
            timesteps = scheduler.timesteps

            # prepare the model input
            # Align seq_len computation with training (divide by full 3D patch volume)
            max_seq_len = (
                latent_video_length
                * lat_h
                * lat_w
                // (patch_size[0] * patch_size[1] * patch_size[2])
            )
            arg_c = {"context": [context], "seq_len": max_seq_len}
            arg_null = {"context": [context_null], "seq_len": max_seq_len}

            # Wrap the inner loop with tqdm to track progress over timesteps
            prompt_idx = sample_parameter.get("enum", 0)
            latent = noise
            # Prepare timestep boundary (normalize int 0..1000 to float 0..1 if needed)
            boundary_cfg = getattr(args, "timestep_boundary", 875)
            boundary = float(boundary_cfg)
            if boundary > 1.0:
                boundary = boundary / 1000.0

            with torch.no_grad():
                for i, t in enumerate(
                    tqdm(
                        timesteps,
                        desc=f"🎥 Sampling timesteps for prompt {prompt_idx+1}",
                    )
                ):
                    # Dual-mode inference: swap base weights if crossing boundary
                    if dual_model_manager is not None:
                        # Align with reference normalization: t/1000.0
                        t_norm = float((t.item()) / 1000.0)
                        try:
                            dual_model_manager.next_model_is_high_noise = (
                                t_norm >= boundary
                            )
                            dual_model_manager.swap_if_needed(accelerator)
                            model = (
                                dual_model_manager.active_model
                            )  # keep local reference fresh
                        except Exception as _inf_swap_err:
                            logger.debug(
                                f"Dual swap during inference skipped: {_inf_swap_err}"
                            )
                    # Prepare model input - concatenate control latents if available
                    if (
                        hasattr(args, "enable_control_lora")
                        and args.enable_control_lora
                    ):
                        # Always concatenate along the channel dimension to match training
                        channel_dim = 0  # CFHW format at sampling time
                        if control_latents is not None:
                            # Debug logging
                            logger.debug(f"Latent shape: {latent.shape}")
                            logger.debug(
                                f"Control latents shape: {control_latents.shape}"
                            )
                            # Ensure device/dtype alignment
                            cat_latent = torch.cat(
                                [
                                    latent.to(device=device),
                                    control_latents.to(
                                        device=device, dtype=latent.dtype
                                    ),
                                ],
                                dim=channel_dim,
                            )
                            latent_model_input = [cat_latent]
                            logger.debug(
                                f"Concatenated input shape: {latent_model_input[0].shape}"
                            )
                        else:
                            # Fallback: use current latent as control signal (matches training fallback)
                            logger.warning(
                                "Control LoRA enabled but no control signal available, using latent clone as fallback"
                            )
                            fallback_control = latent.detach().clone()
                            cat_latent = torch.cat(
                                [
                                    latent.to(device=device),
                                    fallback_control.to(device=device),
                                ],
                                dim=channel_dim,
                            )
                            latent_model_input = [cat_latent]
                            logger.debug(
                                f"Concatenated input shape (fallback clone): {latent_model_input[0].shape}"
                            )
                    else:
                        # Control LoRA not enabled - use original latent
                        latent_model_input = [latent.to(device=device)]
                    timestep = t.unsqueeze(0)

                    with accelerator.autocast():
                        # Apply T-LoRA rank mask at inference time if supported
                        try:
                            unwrapped_net = accelerator.unwrap_model(transformer)
                            if hasattr(
                                unwrapped_net, "update_rank_mask_from_timesteps"
                            ):
                                unwrapped_net.update_rank_mask_from_timesteps(
                                    timestep, max_timestep=1000, device=device
                                )
                        except Exception:
                            pass

                        noise_pred_cond = model(
                            latent_model_input, t=timestep, **arg_c
                        )[0]
                        if do_classifier_free_guidance:
                            noise_pred_uncond = model(
                                latent_model_input, t=timestep, **arg_null
                            )[0]
                        else:
                            noise_pred_uncond = None

                    if do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + cfg_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latent = temp_x0.squeeze(0)

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latent.shape}")
        latent = latent.unsqueeze(0)  # add batch dim
        latent = latent.to(device=device)

        with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            video = vae.decode(latent)[0]  # vae returns list
        video = video.unsqueeze(0)  # add batch dim
        del latent

        logger.info(f"Decoding complete")
        video = video.to(torch.float32).cpu()
        video = (video / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1

        vae.to("cpu")
        clean_memory_on_device(device)

        return video

    def _eqm_sample_latents(
        self,
        *,
        eqm_context: EqMModeContext,
        accelerator: Accelerator,
        transformer: WanModel,
        initial_latent: torch.Tensor,
        context_tokens: torch.Tensor,
        negative_context: Optional[torch.Tensor],
        sample_parameter: Dict[str, Any],
        sample_steps: int,
        cfg_scale: float,
        device: torch.device,
        network_dtype: torch.dtype,
        patch_size: Tuple[int, int, int],
        args: argparse.Namespace,
    ) -> torch.Tensor:
        result: EqMSamplingResult = run_eqm_sampling(
            eqm_context=eqm_context,
            accelerator=accelerator,
            transformer=transformer,
            initial_latent=initial_latent,
            context_tokens=context_tokens,
            negative_context=negative_context,
            sample_parameter=sample_parameter,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            device=device,
            network_dtype=network_dtype,
            patch_size=patch_size,
            args=args,
        )

        if result.likelihood is not None:
            self._last_eqm_likelihood = result.likelihood
            if getattr(accelerator, "is_main_process", True):
                logp_mean = float(result.likelihood["logp"].mean().item())
                prior_mean = float(result.likelihood["prior_logp"].mean().item())
                logger.info(
                    "EqM ODE likelihood stats: logp_mean=%.4f prior_mean=%.4f",
                    logp_mean,
                    prior_mean,
                )
        else:
            self._last_eqm_likelihood = None

        return result.latent

    def _eqm_integrator_sample(
        self, *args, **kwargs
    ) -> torch.Tensor:  # pragma: no cover
        """Deprecated placeholder retained for backward compatibility."""
        raise NotImplementedError(
            "_eqm_integrator_sample has been superseded by research.eqm_mode.sampling_helper.run_eqm_sampling"
        )

    def _generate_control_latents_for_inference(
        self,
        args: argparse.Namespace,
        vae: WanVAE,
        device: torch.device,
        width: int,
        height: int,
        frame_count: int,
        sample_parameter: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Generate control latents for inference, similar to the reference implementation.
        Returns both control latents and extracted video properties.
        """
        import os
        from torchvision.transforms import v2

        # Check if control video/image path is specified in sample parameter
        control_path = sample_parameter.get("control_path", None)
        logger.info(f"Sample parameter keys: {list(sample_parameter.keys())}")
        logger.info(f"Control path from sample parameter: {control_path}")

        if control_path is None:
            # Try to get from args
            control_path = getattr(args, "control_video_path", None)
            logger.info(f"Control path from args: {control_path}")

        if control_path is None:
            logger.warning(
                "No control path specified for control LoRA inference; will fall back to latent clone during sampling"
            )
            # Return None to trigger per-step latent-clone fallback, keep properties for logging
            extracted_properties = {
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "source": "fallback_zero_signal",
            }
            return None, extracted_properties

        # Load control video/image
        if control_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Load video
            try:
                import decord

                decord.bridge.set_bridge("torch")

                # Check if file exists first
                logger.info(f"Checking if control video file exists: {control_path}")
                if not os.path.exists(control_path):
                    logger.error(f"Control video file not found: {control_path}")
                    logger.error(
                        "Please check that the path is correct and the file exists"
                    )
                    raise FileNotFoundError(
                        f"Control video file not found: {control_path}"
                    )
                else:
                    logger.info(f"Control video file found: {control_path}")

                logger.info(f"Loading control video from: {control_path}")
                vr = decord.VideoReader(control_path)
                logger.info(
                    f"Video loaded successfully. Total frames: {len(vr)}, requesting: {frame_count}"
                )

                control_pixels = vr[:frame_count]
                control_pixels = control_pixels.movedim(3, 1).unsqueeze(
                    0
                )  # FHWC -> FCHW -> BFCHW
                logger.info(
                    f"Control pixels shape after loading: {control_pixels.shape}"
                )

            except ImportError:
                logger.error(
                    "decord not available for video loading, using PIL for first frame"
                )
                from PIL import Image
                import torchvision.transforms.functional as TF

                # Fallback to using first frame repeated
                with Image.open(control_path) as img:
                    img = img.convert("RGB")
                control_pixels = (
                    TF.to_tensor(img).unsqueeze(0).unsqueeze(0)
                )  # CHW -> BFCHW
                control_pixels = control_pixels.repeat(
                    1, frame_count, 1, 1, 1
                )  # Repeat frame
            except FileNotFoundError as e:
                logger.error(f"Control video file not found: {e}")
                logger.error("Falling back to zero control signal")
                # Return zero control signal as fallback
                return self._create_zero_control_signal_fallback(
                    width, height, frame_count, device
                )
            except Exception as e:
                logger.error(f"Error loading control video '{control_path}': {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error("Falling back to zero control signal")
                # Return zero control signal as fallback
                return self._create_zero_control_signal_fallback(
                    width, height, frame_count, device
                )
        else:
            # Load image and repeat for video
            try:
                from PIL import Image
                import torchvision.transforms.functional as TF

                # Check if file exists first
                if not os.path.exists(control_path):
                    logger.error(f"Control image file not found: {control_path}")
                    logger.error(
                        "Please check that the path is correct and the file exists"
                    )
                    raise FileNotFoundError(
                        f"Control image file not found: {control_path}"
                    )

                logger.info(f"Loading control image from: {control_path}")
                with Image.open(control_path) as img:
                    img = img.convert("RGB")
                control_pixels = (
                    TF.to_tensor(img).unsqueeze(0).unsqueeze(0)
                )  # CHW -> BFCHW
                control_pixels = control_pixels.repeat(
                    1, frame_count, 1, 1, 1
                )  # Repeat frame
                logger.info(
                    f"Control pixels shape after loading: {control_pixels.shape}"
                )

            except FileNotFoundError as e:
                logger.error(f"Control image file not found: {e}")
                logger.error("Falling back to zero control signal")
                # Return zero control signal as fallback
                return self._create_zero_control_signal_fallback(
                    width, height, frame_count, device
                )
            except Exception as e:
                logger.error(f"Error loading control image '{control_path}': {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error("Falling back to zero control signal")
                # Return zero control signal as fallback
                return self._create_zero_control_signal_fallback(
                    width, height, frame_count, device
                )

        # Apply preprocessing similar to reference implementation
        control_lora_type = getattr(args, "control_lora_type", "tile")
        control_preprocessing = getattr(args, "control_preprocessing", "blur")

        if control_lora_type == "tile" and control_preprocessing == "blur":
            transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize(size=(height // 4, width // 4)),
                    v2.Resize(size=(height, width)),
                    v2.GaussianBlur(
                        kernel_size=getattr(args, "control_blur_kernel_size", 15),
                        sigma=getattr(args, "control_blur_sigma", 3.0),
                    ),  # Changed from sigma=4 to sigma=3 to match reference
                ]
            )

            control_pixels = transform(control_pixels) * 2 - 1  # Scale to [-1, 1]
            control_pixels = torch.clamp(
                torch.nan_to_num(control_pixels), min=-1, max=1
            )
        else:
            # Default preprocessing: just scale to [-1, 1]
            control_pixels = control_pixels * 2 - 1
            control_pixels = torch.clamp(control_pixels, min=-1, max=1)

        # Convert to CFHW format like in reference implementation
        control_pixels = control_pixels[0].movedim(0, 1)  # BFCHW -> CFHW

        # Encode with VAE
        vae.to(device)
        try:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=vae.dtype):
                    control_latents = vae.encode(
                        [control_pixels.to(dtype=vae.dtype, device=device)]
                    )[0]
                    control_latents = control_latents.to(device)
        finally:
            vae.to("cpu")
            clean_memory_on_device(device)

        # Extract properties from control video/image
        actual_frame_count = (
            control_pixels.shape[1] if control_pixels.dim() == 4 else frame_count
        )
        actual_height, actual_width = control_pixels.shape[-2:]

        extracted_properties = {
            "frame_count": actual_frame_count,
            "width": actual_width,
            "height": actual_height,
            "source": (
                "control_video"
                if control_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
                else "control_image"
            ),
        }

        logger.info(f"Generated control latents with shape: {control_latents.shape}")
        logger.info(f"Extracted properties from control signal: {extracted_properties}")
        return control_latents, extracted_properties

    def _create_zero_control_signal_fallback(
        self, width: int, height: int, frame_count: int, device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Create zero control signal as fallback when control signal loading fails."""
        latent_height = height // self.config["vae_stride"][1]
        latent_width = width // self.config["vae_stride"][1]
        latent_frames = (frame_count - 1) // self.config["vae_stride"][0] + 1

        extracted_properties = {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "source": "fallback_zero_signal",
        }

        # For control LoRA, we need to return a tensor that matches the expected input channels
        # The model expects 32 channels (16 original + 16 control), so we need to provide 16 control channels
        control_channels = 16  # This matches the original latent channels

        return (
            torch.zeros(
                control_channels,
                latent_frames,
                latent_height,
                latent_width,
                dtype=torch.float32,
                device=device,
            ),
            extracted_properties,
        )

    def _modify_model_for_control_lora(
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
                    f"✅ Modified model for control LoRA: input channels {old_in_dim} -> {new_in_dim}"
                )

                # Ensure gradients are enabled for the new patch_embedding so it can learn
                transformer.patch_embedding.requires_grad_(True)

                # mark patched
                transformer._control_lora_patched = True
