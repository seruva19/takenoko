"""Reward-based training core for WAN Reward LoRA.

Implements a generate-then-reward backprop training loop similar to the
reference approach, reusing local reward functions in `criteria.reward_fn`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from einops import rearrange
import logging
from common.logger import get_logger
from generation.sampling import save_videos_grid
from utils.train_utils import clean_memory_on_device
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


logger = get_logger(__name__)


def _load_prompts(args: argparse.Namespace) -> List[str]:
    """Load training prompts strictly from TOML-derived args.

    Only uses args.reward_prompts (list[str]) which must be populated from the
    TOML (either as a list or via enumerated reward_promptN keys). No files.
    """
    prompts: List[str] = list(getattr(args, "reward_prompts", []) or [])
    if len(prompts) == 0:
        raise ValueError(
            "No reward prompts found in config (reward_prompts or reward_promptN)"
        )
    return prompts


def _parse_reward_fn(
    device: torch.device,
    dtype: torch.dtype,
    fn_name: str,
    fn_kwargs_json: Optional[str],
):
    from reward import reward_fn as reward_lib

    if not hasattr(reward_lib, fn_name):
        raise ValueError(f"Unknown reward function: {fn_name}")
    kwargs: Dict[str, Any] = {}
    if fn_kwargs_json:
        try:
            kwargs = json.loads(fn_kwargs_json)
        except Exception as e:
            raise ValueError(f"Invalid reward_fn_kwargs JSON: {e}")
    # normalize dtype for external libs
    return getattr(reward_lib, fn_name)(device=str(device), dtype=dtype, **kwargs)


def _compute_seq_len(latents: torch.Tensor, patch_size: Tuple[int, int, int]) -> int:
    _, _, lat_f, lat_h, lat_w = latents.shape
    pt, ph, pw = patch_size
    return max(1, (lat_f * lat_h * lat_w) // (pt * ph * pw))


class RewardTrainingCore:
    """Reward-based trainer for WAN Reward LoRA."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _ensure_vae(self, vae: Optional[WanVAE]) -> WanVAE:
        if vae is None:
            raise ValueError("VAE is required for reward training but not loaded")
        return vae

    def _build_t5(
        self, accelerator: Accelerator, args: argparse.Namespace
    ) -> T5EncoderModel:
        # Select config by task
        text_len = self.config["text_len"]
        t5_dtype = self.config["t5_dtype"]
        t5_path = args.t5
        if t5_path is None or not str(t5_path).strip():
            raise ValueError("T5 checkpoint path (t5) is required for reward training")
        logger.info(f"Loading T5 encoder for reward training: {t5_path}")
        return T5EncoderModel(
            text_len=text_len,
            dtype=t5_dtype,
            device=accelerator.device,
            weight_path=t5_path,
            fp8=getattr(args, "fp8_t5", False),
        )

    def _encode_prompts(
        self, t5: T5EncoderModel, prompts: List[str], device: torch.device
    ) -> List[torch.Tensor]:
        embeds_list: List[torch.Tensor] = []
        with torch.no_grad():
            for p in prompts:
                out = t5([p], device)
                embeds_list.append(out[0])  # (L, D)
        return embeds_list

    def run_reward_training_loop(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: WanModel,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        trainable_params: List[Dict[str, Any]],
        save_model: Any,
        remove_model: Any,
        vae: Optional[WanVAE] = None,
        is_main_process: bool = False,
        global_step: int = 0,
    ) -> Tuple[int, torch.nn.Module]:
        """Main reward-based training loop.

        Notes:
        - Generates latents per step, performs per-timestep denoising with optional CFG,
          decodes a subset of frames, computes reward loss, and backprops on selected steps.
        - Saves checkpoints periodically according to standard args.save_every_* settings.
        """

        device = accelerator.device
        network_dtype = (
            torch.bfloat16
            if args.mixed_precision == "bf16"
            else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
        )

        vae = self._ensure_vae(vae)
        reward_batch_size = int(getattr(args, "reward_train_batch_size", 1))
        height = int(getattr(args, "reward_train_sample_height", 256))
        width = int(getattr(args, "reward_train_sample_width", 256))
        video_len = int(getattr(args, "reward_video_length", 49))
        guidance_scale = float(getattr(args, "reward_guidance_scale", 6.0))
        num_inference_steps = int(getattr(args, "reward_num_inference_steps", 50))
        num_decoded_latents = int(getattr(args, "reward_num_decoded_latents", 1))
        validation_steps = int(getattr(args, "reward_validation_steps", 10000))

        # Prepare scheduler (Flow UniPC multi-step)
        scheduler = FlowUniPCMultistepScheduler(
            shift=int(getattr(args, "discrete_flow_shift", 1))
        )

        # Build T5 encoder and load prompts
        t5 = self._build_t5(accelerator, args)
        prompt_pool = _load_prompts(args)

        # Reward function
        weight_dtype = network_dtype
        reward_fn_name = getattr(args, "reward_fn", "HPSReward")
        reward_fn_kwargs = getattr(args, "reward_fn_kwargs", None)
        reward_fn = _parse_reward_fn(
            device, weight_dtype, reward_fn_name, reward_fn_kwargs
        )

        # Trainer loop size
        max_steps = int(getattr(args, "max_train_steps", 10000))
        checkpoint_every = int(getattr(args, "save_every_n_steps", 1000) or 1000)

        # Latent/patch sizing
        vae_stride_t, vae_stride_hw = (
            self.config["vae_stride"][0],
            self.config["vae_stride"][1],
        )
        lat_f = 1 if video_len == 1 else (video_len - 1) // vae_stride_t + 1
        lat_h, lat_w = height // vae_stride_hw, width // vae_stride_hw
        in_channels = getattr(transformer, "in_dim", 16)

        # Determine backprop steps strategy
        backprop_enabled = bool(getattr(args, "reward_backprop", True))
        strategy = str(getattr(args, "reward_backprop_strategy", "last"))
        tail_k = int(getattr(args, "reward_backprop_num_steps", 5))
        manual_steps: Optional[List[int]] = getattr(
            args, "reward_backprop_step_list", None
        )
        random_range = (
            int(getattr(args, "reward_backprop_random_start_step", 0)),
            int(
                getattr(
                    args, "reward_backprop_random_end_step", num_inference_steps - 1
                )
            ),
        )
        stop_grad_latent_input = bool(
            getattr(args, "reward_stop_latent_model_input_gradient", False)
        )

        # Training
        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_training()

        progress = range(global_step, max_steps)
        for step_idx in progress:
            # Batch selection
            batch_prompts = random.choices(prompt_pool, k=reward_batch_size)

            # Encode prompts and negative prompts
            prompt_embeds = self._encode_prompts(t5, batch_prompts, device)
            negative_embeds = self._encode_prompts(t5, [""] * reward_batch_size, device)

            # Prepare latents
            latents = torch.randn(
                reward_batch_size,
                in_channels,
                lat_f,
                lat_h,
                lat_w,
                device=device,
                dtype=network_dtype,
            )

            # Prepare scheduler
            scheduler.set_timesteps(
                num_inference_steps,
                device=device,
                shift=int(getattr(args, "discrete_flow_shift", 1)),
            )
            timesteps = scheduler.timesteps

            # Per-step denoising
            for i, t in enumerate(timesteps):
                # Prepare model input
                latent_input = latents
                if stop_grad_latent_input:
                    latent_input = latent_input.detach()

                # seq len for DiT
                seq_len = _compute_seq_len(
                    latent_input, tuple(self.config["patch_size"])
                )

                # Two passes for CFG
                with accelerator.autocast():
                    # cond
                    cond_pred_list = transformer(
                        latent_input,
                        t=t.unsqueeze(0).repeat(reward_batch_size),
                        context=[
                            x.to(device=device, dtype=network_dtype)
                            for x in prompt_embeds
                        ],
                        seq_len=seq_len,
                        y=None,
                        clip_fea=None,
                    )
                    cond_pred = torch.stack(cond_pred_list, dim=0)  # (B, C, F, H, W)

                    if guidance_scale > 1.0:
                        uncond_pred_list = transformer(
                            latent_input,
                            t=t.unsqueeze(0).repeat(reward_batch_size),
                            context=[
                                x.to(device=device, dtype=network_dtype)
                                for x in negative_embeds
                            ],
                            seq_len=seq_len,
                            y=None,
                            clip_fea=None,
                        )
                        uncond_pred = torch.stack(uncond_pred_list, dim=0)
                        noise_pred = uncond_pred + guidance_scale * (
                            cond_pred - uncond_pred
                        )
                    else:
                        noise_pred = cond_pred

                # Determine if backprop applies for this step
                do_backprop = False
                if backprop_enabled:
                    if manual_steps is not None:
                        do_backprop = i in manual_steps
                    elif strategy == "last":
                        do_backprop = i == (num_inference_steps - 1)
                    elif strategy == "tail":
                        do_backprop = i >= max(0, num_inference_steps - tail_k)
                    elif strategy == "uniform":
                        interval = max(1, num_inference_steps // max(1, tail_k))
                        do_backprop = (i % interval) == 0
                    elif strategy == "random":
                        rs, re = random_range
                        selected = set(
                            random.sample(
                                list(range(rs, re + 1)),
                                k=min(tail_k, max(1, re - rs + 1)),
                            )
                        )
                        do_backprop = i in selected

                if not do_backprop:
                    noise_pred = noise_pred.detach()

                # Scheduler step
                latents = scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    generator=None,
                )[0]

            # Decode subset of frames for reward
            with torch.autocast(device.type, dtype=vae.dtype):
                # Select first N temporal slices along latent T
                indices = list(range(min(num_decoded_latents, latents.shape[2])))
                sub_latents = latents[:, :, indices, :, :]
                decoded_out = vae.decode(sub_latents)
                # decoded_out can be a list of length B with tensors of shape (T, C, H, W)
                # or a tensor (B, T, C, H, W). Normalize to (B, T, C, H, W).
                if isinstance(decoded_out, list):
                    decoded_btchw = torch.stack(decoded_out, dim=0)
                else:
                    decoded_btchw = decoded_out
            # Scale to [0,1]
            decoded_btchw = (decoded_btchw / 2 + 0.5).clamp(0, 1)
            # For reward: (B, C, T, H, W)
            decoded = decoded_btchw.permute(0, 2, 1, 3, 4).contiguous()

            # Compute reward loss
            # Reward expects (B, C, T, H, W)
            loss, reward_val = reward_fn(
                decoded.to(device=device, dtype=weight_dtype), batch_prompts
            )

            # Log and backward
            accelerator.backward(loss)
            if args.max_grad_norm and float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(
                    [p for p in network.parameters() if p.requires_grad],
                    float(args.max_grad_norm),
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Periodic checkpointing
            global_step += 1
            if (
                is_main_process
                and checkpoint_every
                and (global_step % checkpoint_every == 0)
            ):
                from utils import train_utils as _tu

                ckpt_name = _tu.get_step_ckpt_name(args.output_name, global_step)
                save_model(
                    ckpt_name, accelerator.unwrap_model(network), global_step, None
                )

            # Periodic validation sample (save tiny mp4 of decoded subset)
            if validation_steps and (global_step % validation_steps == 0):
                try:
                    out_dir = os.path.join(args.output_dir, "train_sample")
                    os.makedirs(out_dir, exist_ok=True)
                    # decoded_btchw holds (B, T, C, H, W)
                    save_videos_grid(
                        decoded_btchw.to(torch.float32).detach().cpu(),
                        os.path.join(out_dir, f"sample-{global_step}.mp4"),
                        fps=16,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save reward training sample video: {e}")

            # Tracker logs
            if accelerator.is_main_process and len(accelerator.trackers) > 0:
                accelerator.log(
                    {
                        "train_loss": float(loss.detach().item()),
                        "train_reward": float(reward_val.detach().item()),
                    },
                    step=global_step,
                )

            # Early exit
            if global_step >= max_steps:
                break

            # Clean memory
            clean_memory_on_device(device)

        transformer.switch_block_swap_for_training()
        return global_step, network
