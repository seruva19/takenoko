"""
SRPO training core - main training loop.

This module orchestrates the complete SRPO training pipeline:
1. Load and setup reward models, Direct-Align engine
2. Run online rollout (generate videos from prompts)
3. Apply Direct-Align algorithm (noise injection, single step, recovery)
4. Compute rewards and backpropagate gradients
5. Handle logging, validation, and checkpointing

This is a STANDALONE training loop - it does NOT inherit from TrainingCore.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import logging
from pathlib import Path
import random

from srpo.srpo_config_schema import SRPOConfig
from srpo.srpo_reward_models import create_reward_model
from srpo.srpo_direct_align import DirectAlignEngine
from srpo.srpo_video_rewards import VideoRewardAggregator

logger = logging.getLogger(__name__)


class SRPOTrainingCore:
    """
    SRPO training core - orchestrates the complete training pipeline.

    This class is instantiated ONLY when enable_srpo_training=true.
    It replaces the standard training loop with SRPO-specific logic.
    """

    def __init__(
        self,
        srpo_config: SRPOConfig,
        model_config: Dict[str, Any],
        accelerator,
        transformer: nn.Module,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        vae: nn.Module,
        text_encoder: nn.Module,
        args,
    ):
        self.srpo_config = srpo_config
        self.model_config = model_config
        self.accelerator = accelerator
        self.transformer = transformer
        self.network = network
        self.optimizer = optimizer
        self.vae = vae
        self.text_encoder = text_encoder
        self.args = args

        # Initialize reward model
        reward_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[srpo_config.srpo_reward_model_dtype]

        self.reward_model = create_reward_model(
            reward_model_name=srpo_config.srpo_reward_model_name,
            device=accelerator.device,
            dtype=reward_dtype,
        )
        # CRITICAL: Freeze reward model to prevent gradient computation
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()
        logger.info(f"✓ Reward model frozen (no gradients)")

        # Initialize video-specific rewards if enabled
        if srpo_config.srpo_enable_video_rewards:
            self.video_reward_aggregator = VideoRewardAggregator(
                device=accelerator.device,
                dtype=reward_dtype,
                temporal_consistency_weight=srpo_config.srpo_temporal_consistency_weight,
                optical_flow_weight=srpo_config.srpo_optical_flow_weight,
                motion_quality_weight=srpo_config.srpo_motion_quality_weight,
            )
            logger.info(f"✓ Video-specific rewards enabled")
        else:
            self.video_reward_aggregator = None

        # Initialize Direct-Align engine
        self.direct_align_engine = DirectAlignEngine(
            num_inference_steps=srpo_config.srpo_num_inference_steps,
            sigma_min=srpo_config.srpo_sigma_interpolation_min,
            sigma_max=srpo_config.srpo_sigma_interpolation_max,
            interpolation_method=srpo_config.srpo_sigma_interpolation_method,
            enable_time_shift=srpo_config.srpo_enable_sd3_time_shift,
            time_shift_value=srpo_config.srpo_sd3_time_shift_value,
            discount_denoise_min=srpo_config.srpo_discount_denoise_min,
            discount_denoise_max=srpo_config.srpo_discount_denoise_max,
            discount_inversion_start=srpo_config.srpo_discount_inversion_start,
            discount_inversion_end=srpo_config.srpo_discount_inversion_end,
            device=accelerator.device,
        )

        # Freeze VAE (should already be frozen, but double-check)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Set transformer to train mode (needed for LoRA dropout, etc.)
        self.transformer.train()

        # Enable LoRA network for training
        # CRITICAL: Must enable network to inject LoRA into transformer forward pass
        if hasattr(network, "prepare_network"):
            network.prepare_network(transformer)
            logger.info("✓ LoRA network prepared and enabled for transformer")
        else:
            logger.warning(
                "Network doesn't have prepare_network method - LoRA may not be active!"
            )

        # Verify LoRA network has trainable parameters
        lora_param_count = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )
        if lora_param_count == 0:
            raise RuntimeError(
                "❌ LoRA network has NO trainable parameters! "
                "Check that network_module='networks.lora_wan' and LoRA config is correct."
            )
        logger.info(f"✓ LoRA network has {lora_param_count:,} trainable parameters")

        # Verify frozen models have NO trainable parameters
        reward_param_count = sum(
            p.numel() for p in self.reward_model.parameters() if p.requires_grad
        )
        vae_param_count = sum(
            p.numel() for p in self.vae.parameters() if p.requires_grad
        )
        if reward_param_count > 0:
            logger.warning(
                f"⚠️ Reward model has {reward_param_count} trainable parameters (should be 0)"
            )
        if vae_param_count > 0:
            logger.warning(
                f"⚠️ VAE has {vae_param_count} trainable parameters (should be 0)"
            )

        logger.info(
            f"Initialized SRPO training with reward model: {srpo_config.srpo_reward_model_name}"
        )

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts using T5 encoder.

        VERIFIED: T5EncoderModel.__call__ returns List[Tensor] where each
        tensor has shape (seq_len, hidden_dim). We need to pad and stack.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of shape [B, max_seq_len, hidden_dim]
        """
        # Call T5 encoder (returns list of tensors)
        embeddings_list = self.text_encoder(prompts)  # List[Tensor(seq_len, dim)]

        # Find max sequence length
        max_seq_len = max(emb.shape[0] for emb in embeddings_list)

        # Pad and stack
        B = len(prompts)
        hidden_dim = embeddings_list[0].shape[1]
        padded_embeddings = torch.zeros(
            B, max_seq_len, hidden_dim, device=self.accelerator.device
        )

        for i, emb in enumerate(embeddings_list):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len, :] = emb

        return padded_embeddings

    def _decode_latents_to_images(
        self, latents: torch.Tensor, num_frames: int = 1
    ) -> torch.Tensor:
        """
        Decode latents to images using VAE.

        NOTE: WAN VAE performs internal per-channel normalization using its
        own mean/std tensors. VAE.decode() returns a list of tensors in [-1, 1]
        range. We convert to [0, 1] for reward models.

        Args:
            latents: Tensor of shape [B, C, F, H, W]
            num_frames: Number of frames to decode (for multi-frame rewards)

        Returns:
            If num_frames == 1: Images of shape [B, C, H_out, W_out]
            If num_frames > 1: Images of shape [B, num_frames, C, H_out, W_out]
        """
        with torch.no_grad():
            B, C, F, H, W = latents.shape

            # Determine which frames to decode based on strategy
            frame_indices = self._select_reward_frames(F, num_frames)

            if len(frame_indices) == 1:
                # Single frame decoding (original behavior)
                frame_latents = latents[:, :, frame_indices[0], :, :]  # [B, C, H, W]

                # Convert batched tensor to list of unbatched tensors (WAN VAE API)
                latents_list = [frame_latents[i] for i in range(frame_latents.shape[0])]

                # WAN VAE decodes and normalizes internally, returns list in [-1, 1]
                images_list = self.vae.decode(latents_list)  # List of [C, H, W]

                # Stack back to batch tensor
                images = torch.stack(images_list)  # [B, C, H, W]

                # Convert from [-1, 1] to [0, 1] range for reward models
                images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0.0, 1.0)

                return images
            else:
                # Multi-frame decoding
                all_frames = []
                for frame_idx in frame_indices:
                    frame_latents = latents[:, :, frame_idx, :, :]  # [B, C, H, W]

                    # Convert to list for VAE
                    latents_list = [
                        frame_latents[i] for i in range(frame_latents.shape[0])
                    ]

                    # Decode
                    images_list = self.vae.decode(latents_list)

                    # Stack and normalize
                    images = torch.stack(images_list)  # [B, C, H, W]
                    images = (images + 1.0) / 2.0
                    images = torch.clamp(images, 0.0, 1.0)

                    all_frames.append(images)

                # Stack frames: [B, num_frames, C, H, W]
                all_frames = torch.stack(all_frames, dim=1)

                return all_frames

    def _select_reward_frames(self, total_frames: int, num_frames: int) -> list:
        """
        Select which frames to use for reward computation based on strategy.

        Args:
            total_frames: Total number of frames in video
            num_frames: Number of frames to select

        Returns:
            List of frame indices
        """
        if num_frames >= total_frames:
            return list(range(total_frames))

        strategy = self.srpo_config.srpo_reward_frame_strategy

        if strategy == "first":
            return [0]
        elif strategy == "uniform":
            # Uniformly sample frames across the video
            indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
            return indices
        elif strategy == "boundary":
            # Sample first, last, and middle frames
            if num_frames == 1:
                return [0]
            elif num_frames == 2:
                return [0, total_frames - 1]
            else:
                # First, last, and evenly spaced middle frames
                indices = [0]
                middle_frames = num_frames - 2
                if middle_frames > 0:
                    middle_indices = (
                        torch.linspace(1, total_frames - 2, middle_frames)
                        .long()
                        .tolist()
                    )
                    indices.extend(middle_indices)
                indices.append(total_frames - 1)
                return indices
        elif strategy == "all":
            return list(range(total_frames))
        else:
            return [0]

    def _online_rollout(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Perform online rollout - generate videos from prompts using current LoRA.

        This is a SIMPLIFIED version of the WAN generation process,
        running a full Euler integration loop.

        Args:
            prompts: List of text prompts
            num_frames: Number of frames to generate
            height: Video height
            width: Video width

        Returns:
            Clean latents of shape [B, C, F, H, W]
        """
        B = len(prompts)
        C = self.srpo_config.srpo_latent_channels
        vae_scale = self.srpo_config.srpo_vae_scale_factor

        # Latent dimensions
        F = num_frames
        H = height // vae_scale
        W = width // vae_scale

        # Encode prompts
        context_tensor = self._encode_prompts(prompts)  # [B, max_seq_len, hidden_dim]
        context_list = [context_tensor[i] for i in range(B)]

        # Calculate sequence length
        seq_len = F * H * W

        # Initialize with noise
        latents = torch.randn(
            B, C, F, H, W, device=self.accelerator.device, dtype=self.transformer.dtype
        )

        # Run Euler integration
        sigma_schedule = self.direct_align_engine.sigma_schedule

        for i in range(len(sigma_schedule) - 1):
            sigma_current = sigma_schedule[i]
            sigma_next = sigma_schedule[i + 1]

            # Prepare timesteps
            timesteps = sigma_current.unsqueeze(0).expand(B)

            # Convert latents to list format for WAN (unbatched: [C, F, H, W] per element)
            latents_list = [
                latents[j] for j in range(B)
            ]  # List of [C, F, H, W] tensors

            # Call WAN transformer
            with autocast(enabled=True, dtype=self.transformer.dtype):
                model_pred_list = self.transformer(
                    latents_list,
                    t=timesteps,
                    context=context_list,
                    seq_len=seq_len,
                )

            # Convert back to tensor (stack, not cat, since each element is unbatched)
            model_pred = torch.stack(model_pred_list, dim=0)  # [B, C, F, H, W]

            # Euler step
            sigma_current_expanded = sigma_current.view(1, 1, 1, 1, 1).expand(
                B, 1, 1, 1, 1
            )
            sigma_next_expanded = sigma_next.view(1, 1, 1, 1, 1).expand(B, 1, 1, 1, 1)

            latents = self.direct_align_engine.single_euler_step(
                latents,
                model_pred,
                sigma_current_expanded,
                sigma_next_expanded,
                branch="denoise",
            )

        return latents

    def _compute_srpo_loss(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Compute SRPO loss for a batch of prompts using Direct-Align.

        Steps:
        1. Online rollout to generate clean latents
        2. Sample random noise
        3. Inject noise at random sigma
        4. Single Euler step (random branch: denoise or inversion)
        5. Recover clean image using alternating branch
        6. Compute SRP reward
        7. Apply discount and return negative reward as loss

        Args:
            prompts: List of text prompts
            num_frames: Number of frames
            height: Video height
            width: Video width

        Returns:
            Scalar loss tensor
        """
        B = len(prompts)

        # Step 1: Online rollout
        clean_latents = self._online_rollout(prompts, num_frames, height, width)

        # Step 2: Sample random noise
        noise = torch.randn_like(clean_latents)

        # Step 3: Inject noise at random sigma
        noisy_latents, sigma, step_idx = (
            self.direct_align_engine.inject_noise_at_random_sigma(
                clean_latents,
                noise,
                sigma_min=self.srpo_config.srpo_sigma_interpolation_min,
                sigma_max=self.srpo_config.srpo_sigma_interpolation_max,
            )
        )

        # Step 4: Random branch selection
        branch = random.choice(["denoise", "inversion"])

        # Encode prompts for model call
        context_tensor = self._encode_prompts(prompts)
        context_list = [context_tensor[i] for i in range(B)]
        seq_len = (
            num_frames
            * (height // self.srpo_config.srpo_vae_scale_factor)
            * (width // self.srpo_config.srpo_vae_scale_factor)
        )

        # Get model prediction
        timesteps = sigma.view(B)
        latents_list = [
            noisy_latents[i] for i in range(B)
        ]  # List of [C, F, H, W] tensors

        with autocast(enabled=True, dtype=self.transformer.dtype):
            model_pred_list = self.transformer(
                latents_list,
                t=timesteps,
                context=context_list,
                seq_len=seq_len,
            )

        model_pred = torch.stack(model_pred_list, dim=0)  # [B, C, F, H, W]

        # Single Euler step
        sigma_next = (
            self.direct_align_engine.sigma_schedule[step_idx + 1]
            if step_idx + 1 < len(self.direct_align_engine.sigma_schedule)
            else torch.tensor(0.0, device=self.accelerator.device)
        )
        sigma_next_expanded = sigma_next.view(1, 1, 1, 1, 1).expand(B, 1, 1, 1, 1)

        latents_after_step = self.direct_align_engine.single_euler_step(
            noisy_latents, model_pred, sigma, sigma_next_expanded, branch=branch
        )

        # Step 5: Recover clean image
        # Recovered clean latents of shape [B, C, F, H, W]
        # Note: lambda converts batched tensor to WAN's expected list format
        recovered_latents = self.direct_align_engine.recover_image(
            latents_after_step,
            sigma_next_expanded,
            lambda x, **kwargs: torch.stack(
                self.transformer(
                    [
                        x[i] for i in range(x.shape[0])
                    ],  # Unbatched list: each element is [C, F, H, W]
                    **kwargs,
                ),
                dim=0,
            ),
            context_list,
            seq_len,
            branch=branch,
        )

        # Step 6: Compute rewards
        # Decode frames for reward computation
        num_reward_frames = self.srpo_config.srpo_reward_num_frames
        recovered_images = self._decode_latents_to_images(
            recovered_latents, num_reward_frames
        )

        # Compute image-based rewards
        if num_reward_frames == 1:
            # Single frame: use standard image reward
            image_rewards = self.reward_model.compute_srp_rewards(
                recovered_images,  # [B, C, H, W]
                prompts,
                positive_words=self.srpo_config.srpo_srp_positive_words,
                negative_words=self.srpo_config.srpo_srp_negative_words,
                k=self.srpo_config.srpo_srp_control_weight,
            )
        else:
            # Multi-frame: compute rewards per frame and aggregate
            frame_rewards = []
            for frame_idx in range(num_reward_frames):
                frame_images = recovered_images[:, frame_idx, :, :, :]  # [B, C, H, W]
                frame_reward = self.reward_model.compute_srp_rewards(
                    frame_images,
                    prompts,
                    positive_words=self.srpo_config.srpo_srp_positive_words,
                    negative_words=self.srpo_config.srpo_srp_negative_words,
                    k=self.srpo_config.srpo_srp_control_weight,
                )
                frame_rewards.append(frame_reward)

            # Aggregate frame rewards
            frame_rewards = torch.stack(frame_rewards, dim=1)  # [B, num_frames]

            aggregation = self.srpo_config.srpo_reward_aggregation
            if aggregation == "mean":
                image_rewards = frame_rewards.mean(dim=1)
            elif aggregation == "min":
                image_rewards = frame_rewards.min(dim=1)[0]
            elif aggregation == "max":
                image_rewards = frame_rewards.max(dim=1)[0]
            elif aggregation == "weighted":
                # Weighted average: more weight to later frames
                weights = torch.linspace(
                    0.5, 1.0, num_reward_frames, device=frame_rewards.device
                )
                weights = weights / weights.sum()
                image_rewards = (frame_rewards * weights.unsqueeze(0)).sum(dim=1)
            else:
                image_rewards = frame_rewards.mean(dim=1)

        # Add video-specific rewards if enabled
        if self.video_reward_aggregator is not None:
            # Reshape for video rewards: need [B, C, F, H, W]
            # recovered_latents is already in this format
            # Decode ALL frames for video-specific rewards
            with torch.no_grad():
                all_frames = self._decode_latents_to_images(
                    recovered_latents,
                    num_frames=recovered_latents.shape[2],  # Decode all frames
                )
                # all_frames: [B, F, C, H, W] -> need [B, C, F, H, W]
                all_frames = all_frames.permute(0, 2, 1, 3, 4)

                video_rewards = self.video_reward_aggregator.compute_reward(all_frames)

            # Combine image and video rewards
            rewards = image_rewards + video_rewards
        else:
            rewards = image_rewards

        # Step 7: Apply discount
        if branch == "denoise":
            discount = self.direct_align_engine.discount_denoise[step_idx]
        else:
            discount = self.direct_align_engine.discount_inversion[step_idx]

        discounted_rewards = discount * rewards

        # Loss is negative reward (we want to maximize reward)
        loss = -discounted_rewards.mean()

        return loss

    def run_training_loop(self):
        """
        Run the SRPO training loop.

        This is the main entry point called from wan_network_trainer.py.
        """
        logger.info("Starting SRPO training loop...")

        # Setup output directories
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_dir = output_dir / "validation"
        validation_dir.mkdir(exist_ok=True)

        # Training hyperparameters
        num_training_steps = self.srpo_config.srpo_num_training_steps
        gradient_accumulation_steps = self.srpo_config.srpo_gradient_accumulation_steps
        batch_size = self.srpo_config.srpo_batch_size

        # Get video parameters from args
        num_frames = getattr(self.args, "num_frames", 81)
        height = getattr(self.args, "height", 480)
        width = getattr(self.args, "width", 720)

        # Check for auto-resume
        starting_step = 0
        if getattr(self.args, "auto_resume", False):
            starting_step = self._try_resume_training(output_dir)
            if starting_step > 0:
                logger.info(f"✓ Resumed training from step {starting_step}")
            else:
                logger.info("No checkpoint found, starting from scratch")

        # Load dataset for prompts
        if hasattr(self.args, "datasets") and self.args.datasets:
            # Use prompts from dataset
            dataset_config = (
                self.args.datasets[0]
                if isinstance(self.args.datasets, list)
                else self.args.datasets
            )
            dataset_path = Path(dataset_config.get("path", "data/prompts.txt"))
            if dataset_path.exists():
                with open(dataset_path, "r") as f:
                    all_prompts = [line.strip() for line in f if line.strip()]
            else:
                logger.warning(
                    f"Dataset path {dataset_path} not found, using default prompts"
                )
                all_prompts = [
                    "a dog running in a field",
                    "a beautiful sunset over mountains",
                    "a futuristic city with flying cars",
                    "a cat playing with a ball of yarn",
                ]
        else:
            # Use default prompts
            all_prompts = [
                "a dog running in a field",
                "a beautiful sunset over mountains",
                "a futuristic city with flying cars",
                "a cat playing with a ball of yarn",
            ]

        logger.info(f"Loaded {len(all_prompts)} prompts for training")

        # Training loop
        global_step = starting_step
        self.optimizer.zero_grad()

        for step in range(starting_step, num_training_steps):
            # Sample prompts
            prompts = random.sample(all_prompts, min(batch_size, len(all_prompts)))

            # Compute loss (with gradient accumulation)
            for _ in range(gradient_accumulation_steps):
                loss = self._compute_srpo_loss(prompts, num_frames, height, width)

                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps

                # Backward pass
                self.accelerator.backward(loss)

            # CRITICAL: Verify gradients on first step
            if global_step == starting_step:
                lora_params_with_grad = sum(
                    1
                    for p in self.network.parameters()
                    if p.requires_grad and p.grad is not None
                )
                lora_params_total = sum(
                    1 for p in self.network.parameters() if p.requires_grad
                )

                if lora_params_with_grad == 0:
                    raise RuntimeError(
                        "❌ CRITICAL: No gradients computed for LoRA network! "
                        "Training will not work. Check that LoRA is properly enabled."
                    )

                if lora_params_with_grad < lora_params_total:
                    logger.warning(
                        f"⚠️ Only {lora_params_with_grad}/{lora_params_total} "
                        f"LoRA parameters received gradients"
                    )
                else:
                    logger.info(
                        f"✓ Gradient flow verified: {lora_params_with_grad}/{lora_params_total} "
                        f"LoRA parameters have gradients"
                    )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                logger.info(
                    f"Step {global_step}/{num_training_steps} | Loss: {loss.item():.4f}"
                )

            # Validation
            if global_step % self.srpo_config.srpo_validation_frequency == 0:
                self._run_validation(
                    validation_dir, global_step, height, width, num_frames
                )

            # Checkpoint saving
            if global_step % 100 == 0:
                checkpoint_path = (
                    output_dir / f"checkpoint_step_{global_step}.safetensors"
                )
                self._save_checkpoint(checkpoint_path)

        logger.info("SRPO training completed!")

    def _run_validation(
        self, validation_dir: Path, step: int, height: int, width: int, num_frames: int
    ):
        """Run validation and save results."""
        logger.info(f"Running validation at step {step}...")

        # Switch to eval mode for validation
        self.transformer.eval()
        self.network.eval()

        validation_prompts = self.srpo_config.srpo_validation_prompts

        # Generate videos
        with torch.no_grad():
            clean_latents = self._online_rollout(
                validation_prompts, num_frames, height, width
            )

            # Decode frames for reward computation
            num_reward_frames = self.srpo_config.srpo_reward_num_frames
            images = self._decode_latents_to_images(clean_latents, num_reward_frames)

            # Compute rewards (same logic as training)
            if num_reward_frames == 1:
                rewards = self.reward_model.compute_rewards(images, validation_prompts)
            else:
                # Multi-frame rewards
                frame_rewards = []
                for frame_idx in range(num_reward_frames):
                    frame_images = images[:, frame_idx, :, :, :]
                    frame_reward = self.reward_model.compute_rewards(
                        frame_images, validation_prompts
                    )
                    frame_rewards.append(frame_reward)

                frame_rewards = torch.stack(frame_rewards, dim=1)

                aggregation = self.srpo_config.srpo_reward_aggregation
                if aggregation == "mean":
                    rewards = frame_rewards.mean(dim=1)
                elif aggregation == "min":
                    rewards = frame_rewards.min(dim=1)[0]
                elif aggregation == "max":
                    rewards = frame_rewards.max(dim=1)[0]
                elif aggregation == "weighted":
                    weights = torch.linspace(
                        0.5, 1.0, num_reward_frames, device=frame_rewards.device
                    )
                    weights = weights / weights.sum()
                    rewards = (frame_rewards * weights.unsqueeze(0)).sum(dim=1)
                else:
                    rewards = frame_rewards.mean(dim=1)

            # Add video-specific rewards if enabled
            if self.video_reward_aggregator is not None:
                all_frames = self._decode_latents_to_images(
                    clean_latents, num_frames=clean_latents.shape[2]
                )
                all_frames = all_frames.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                video_rewards = self.video_reward_aggregator.compute_reward(all_frames)
                rewards = rewards + video_rewards

            logger.info(
                f"Validation rewards: {rewards.mean().item():.4f} "
                f"(min: {rewards.min().item():.4f}, max: {rewards.max().item():.4f})"
            )

            # Save validation outputs
            if self.srpo_config.srpo_save_validation_videos:
                if self.srpo_config.srpo_save_validation_as_images:
                    # Save as PNG images
                    import torchvision.utils as vutils

                    for i, prompt in enumerate(validation_prompts):
                        img_path = (
                            validation_dir / f"step_{step}_prompt_{i}_{prompt[:30]}.png"
                        )
                        vutils.save_image(images[i], img_path)
                else:
                    # Save as videos (existing Takenoko video save logic)
                    from generation.sampling import save_videos_grid

                    for i, prompt in enumerate(validation_prompts):
                        video_path = (
                            validation_dir / f"step_{step}_prompt_{i}_{prompt[:30]}.mp4"
                        )
                        # Decode latents and save
                        # save_videos_grid expects [1, C, T, H, W] tensor
                        save_videos_grid(
                            self.vae.decode([clean_latents[i]]),
                            str(video_path),
                            rescale=False,
                        )

        # Switch back to train mode
        self.transformer.train()
        self.network.train()

    def _save_checkpoint(self, checkpoint_path: Path):
        """Save LoRA checkpoint."""
        logger.info(f"Saving checkpoint to {checkpoint_path}...")

        # Save LoRA weights using accelerator
        # Note: This saves the entire training state including optimizer
        self.accelerator.save_state(
            str(checkpoint_path.parent / f"state_{checkpoint_path.stem}")
        )

        # For LoRA-only weights (safetensors), unwrap and save directly
        import safetensors.torch as safetensors_torch

        unwrapped_network = self.accelerator.unwrap_model(self.network)
        state_dict = unwrapped_network.state_dict()

        safetensors_torch.save_file(state_dict, str(checkpoint_path))

    def _try_resume_training(self, output_dir: Path) -> int:
        """
        Try to resume training from the latest checkpoint.

        Returns:
            Starting step number (0 if no checkpoint found)
        """
        # Look for state checkpoints (full training state with optimizer)
        state_dirs = list(output_dir.glob("state_checkpoint_step_*"))

        if not state_dirs:
            return 0

        # Find the latest checkpoint by step number
        latest_step = 0
        latest_state_dir = None

        for state_dir in state_dirs:
            try:
                step_str = state_dir.name.replace("state_checkpoint_step_", "")
                step = int(step_str)
                if step > latest_step:
                    latest_step = step
                    latest_state_dir = state_dir
            except ValueError:
                continue

        if latest_state_dir is None:
            return 0

        logger.info(f"Found checkpoint at step {latest_step}, resuming...")

        try:
            # Load full training state (includes optimizer, network, RNG states)
            self.accelerator.load_state(str(latest_state_dir))
            logger.info(f"✓ Loaded training state from {latest_state_dir}")

            return latest_step
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting training from scratch")
            return 0
