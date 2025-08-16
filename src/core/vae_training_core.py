"""VAE-specific training core for WAN network trainer.

This module handles VAE training logic, including reconstruction loss and KL divergence.
Separate from the main training_core.py to handle the different training paradigm.
"""

import argparse
import math
import numpy as np
import time
from multiprocessing import Value
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm
import accelerate
from accelerate import Accelerator, PartialState
import torch.nn.functional as F

from utils.train_utils import (
    clean_memory_on_device,
    should_sample_images,
    LossRecorder,
)

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class VaeTrainingCore:
    """Handles VAE-specific training logic."""

    def __init__(self, config: Any):
        self.config = config

    def compute_vae_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        kl_weight: float = 1e-6,
        reconstruction_loss_type: str = "mse",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VAE loss including reconstruction loss and KL divergence."""

        # Reconstruction loss
        if reconstruction_loss_type == "mse":
            reconstruction_loss = F.mse_loss(reconstructed, original, reduction="mean")
        elif reconstruction_loss_type == "l1":
            reconstruction_loss = F.l1_loss(reconstructed, original, reduction="mean")
        elif reconstruction_loss_type == "huber":
            reconstruction_loss = F.smooth_l1_loss(
                reconstructed, original, reduction="mean"
            )
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {reconstruction_loss_type}"
            )

        # KL divergence loss (if VAE outputs mu and logvar)
        kl_loss = torch.tensor(0.0, device=reconstructed.device)
        if mu is not None and logvar is not None:
            # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / original.numel()  # Normalize by number of elements

        # Total loss
        total_loss = reconstruction_loss + kl_weight * kl_loss

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def call_vae(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        vae: Any,
        images: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Call the VAE model and compute reconstruction."""

        # Ensure images are in the right format and dtype
        images = images.to(device=accelerator.device, dtype=network_dtype)

        with accelerator.autocast():
            # For VAE training, we need the full forward pass
            if hasattr(vae, "encode") and hasattr(vae, "decode"):
                # Standard VAE with separate encode/decode
                encoded = vae.encode(images)

                # Handle different VAE output formats
                if isinstance(encoded, tuple):
                    # (mu, logvar) format
                    mu, logvar = encoded
                    # Reparameterization trick
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                else:
                    # Direct latent format
                    z = encoded
                    mu, logvar = None, None

                # Decode
                reconstructed = vae.decode(z)
            else:
                # Direct forward pass
                output = vae(images)
                if isinstance(output, tuple):
                    reconstructed, mu, logvar = output
                else:
                    reconstructed = output
                    mu, logvar = None, None

        return reconstructed, images, mu, logvar

    def validate_vae(
        self,
        accelerator: Accelerator,
        vae: Any,
        val_dataloader: Any,
        args: argparse.Namespace,
    ) -> float:
        """Run VAE validation and return average validation loss."""
        logger.info("Running VAE validation...")

        vae.eval()
        losses = []

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                # For VAE validation, we need original images, not latents
                if "images" in batch:
                    images = batch["images"]
                elif "latents" in batch:
                    # If we only have latents, we can't validate properly
                    logger.warning(
                        "VAE validation requires original images, but only latents found. Skipping validation."
                    )
                    return 0.0
                else:
                    logger.warning("No images found in validation batch. Skipping.")
                    continue

                images = images.to(accelerator.device, dtype=vae.dtype)

                # Forward pass
                reconstructed, original, mu, logvar = self.call_vae(
                    args, accelerator, vae, images, vae.dtype
                )

                # Compute loss
                val_loss, _ = self.compute_vae_loss(
                    reconstructed,
                    original,
                    mu,
                    logvar,
                    kl_weight=getattr(args, "vae_kl_weight", 1e-6),
                    reconstruction_loss_type=getattr(
                        args, "vae_reconstruction_loss", "mse"
                    ),
                )

                losses.append(val_loss)

        if losses:
            losses_tensor = torch.stack(losses)
            gathered_losses = accelerator.gather_for_metrics(losses_tensor)
            final_avg_loss = gathered_losses.mean().item()  # type: ignore
        else:
            final_avg_loss = 0.0

        vae.train()
        logger.info(f"VAE validation finished. Average loss: {final_avg_loss:.5f}")
        return final_avg_loss

    def run_vae_training_loop(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        vae: Any,
        network: Any,
        training_model: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        lr_descriptions: List[str],
        train_dataloader: Any,
        val_dataloader: Optional[Any],
        network_dtype: torch.dtype,
        num_train_epochs: int,
        global_step: int,
        progress_bar: tqdm,
        metadata: Dict[str, str],
        loss_recorder: LossRecorder,
        current_epoch: Optional[Value] = None,  # type: ignore
        current_step: Optional[Value] = None,  # type: ignore
        optimizer_train_fn: Optional[callable] = None,  # type: ignore
        optimizer_eval_fn: Optional[callable] = None,  # type: ignore
        save_model: Optional[callable] = None,  # type: ignore
        remove_model: Optional[callable] = None,  # type: ignore
        is_main_process: bool = False,
    ) -> Tuple[int, Any]:
        """Run the VAE training loop."""

        # VAE-specific training parameters
        kl_weight = getattr(args, "vae_kl_weight", 1e-6)
        reconstruction_loss_type = getattr(args, "vae_reconstruction_loss", "mse")

        logger.info(
            f"Starting VAE training with KL weight: {kl_weight}, reconstruction loss: {reconstruction_loss_type}"
        )

        for epoch in range(num_train_epochs):
            accelerator.print(f"\nVAE epoch {epoch+1}/{num_train_epochs}")
            if current_epoch is not None:
                current_epoch.value = epoch + 1

            metadata["takenoko_epoch"] = str(epoch + 1)

            for step, batch in enumerate(train_dataloader):
                if current_step is not None:
                    current_step.value = global_step

                with accelerator.accumulate(training_model):
                    # For VAE training, we need original images
                    if "images" in batch:
                        images = batch["images"]
                    else:
                        logger.error(
                            "VAE training requires original images in batch. Please ensure dataset provides 'images' key."
                        )
                        continue

                    # Apply sample weights if present
                    sample_weights = batch.get("weight", None)

                    # Forward pass through VAE
                    reconstructed, original, mu, logvar = self.call_vae(
                        args, accelerator, vae, images, network_dtype
                    )

                    # Compute VAE loss
                    loss, loss_dict = self.compute_vae_loss(
                        reconstructed,
                        original,
                        mu,
                        logvar,
                        kl_weight=kl_weight,
                        reconstruction_loss_type=reconstruction_loss_type,
                    )

                    # Apply sample weights if present
                    if sample_weights is not None:
                        sample_weights = sample_weights.to(
                            device=accelerator.device, dtype=network_dtype
                        )
                        loss = (
                            loss * sample_weights.mean()
                        )  # Apply weight to total loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.max_grad_norm != 0.0:
                            params_to_clip = network.get_trainable_params()
                            accelerator.clip_grad_norm_(
                                params_to_clip, args.max_grad_norm
                            )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Validation
                    should_validating = (
                        args.validate_every_n_steps is not None
                        and global_step % args.validate_every_n_steps == 0
                        and val_dataloader is not None
                    )

                    # Model saving
                    should_saving = (
                        args.save_every_n_steps is not None
                        and global_step % args.save_every_n_steps == 0
                    )

                    if should_validating or should_saving:
                        if optimizer_eval_fn:
                            optimizer_eval_fn()

                        if should_validating:
                            val_loss = self.validate_vae(
                                accelerator, vae, val_dataloader, args
                            )
                            accelerator.print(
                                f"[Step {global_step}] VAE val_loss={val_loss:0.5f}"
                            )
                            accelerator.log({"val_loss": val_loss}, step=global_step)

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process and save_model:
                                from utils import train_utils

                                ckpt_name = train_utils.get_step_ckpt_name(
                                    args.output_name, global_step
                                )
                                save_model(ckpt_name, network, global_step, epoch + 1)

                        if optimizer_train_fn:
                            optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch + 1, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average

                # Log detailed VAE losses
                logs = {
                    "loss/total": current_loss,
                    "loss/average": avr_loss,
                    "loss/reconstruction": loss_dict["reconstruction_loss"].item(),
                    "loss/kl": loss_dict["kl_loss"].item(),
                }

                # Add learning rates
                lrs = lr_scheduler.get_last_lr()
                for i, lr in enumerate(lrs):
                    if i < len(lr_descriptions):
                        logs[f"lr/{lr_descriptions[i]}"] = lr
                    else:
                        logs[f"lr/group{i}"] = lr

                progress_bar.set_postfix(logs)

                if len(accelerator.trackers) > 0:
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

            # End of epoch validation
            should_validate_on_epoch_end = getattr(args, "validate_on_epoch_end", False)
            if val_dataloader is not None and should_validate_on_epoch_end:
                val_loss = self.validate_vae(accelerator, vae, val_dataloader, args)
                accelerator.print(f"[Epoch {epoch+1}] VAE val_loss={val_loss:0.5f}")
                accelerator.log({"val_loss": val_loss}, step=global_step)
            elif val_dataloader is not None and not should_validate_on_epoch_end:
                accelerator.print(
                    f"\n[Epoch {epoch+1}] VAE epoch-end validation disabled"
                )

            # Save model at end of epoch if needed
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs
                if is_main_process and saving and save_model:
                    from utils import train_utils

                    ckpt_name = train_utils.get_epoch_ckpt_name(
                        args.output_name, epoch + 1
                    )
                    save_model(ckpt_name, network, global_step, epoch + 1)

        return global_step, network
