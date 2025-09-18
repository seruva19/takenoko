"""VAE-specific training core for WAN network trainer.

This module handles VAE training logic, including reconstruction loss and KL divergence.
Separate from the main training_core.py to handle the different training paradigm.
"""

import argparse
import math
import numpy as np
import os
import time
from collections import deque
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
    config_key_provided,
)

from utils.perceptual import get_lpips_model, sobel_edges

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class MedianLossBalancer:
    """Windowed median balancer to stabilise weighted loss contributions."""

    def __init__(
        self,
        desired_weights: Dict[str, float],
        window_steps: int,
        percentile: float,
    ) -> None:
        self.window = max(int(window_steps), 0)
        self.percentile = float(percentile)
        self.desired_weights = {
            key: max(float(value), 0.0) for key, value in desired_weights.items()
        }
        self.total_weight = sum(self.desired_weights.values())
        if self.total_weight <= 0:
            self.total_weight = 0.0

        if self.total_weight > 0:
            self.normalised = {
                key: value / self.total_weight
                for key, value in self.desired_weights.items()
                if value > 0
            }
        else:
            self.normalised = {}

        if self.window > 0:
            self.buffers: Dict[str, deque[float]] = {
                key: deque(maxlen=self.window) for key in self.normalised
            }
        else:
            self.buffers = {key: deque() for key in self.normalised}

        self.last_coeffs: Dict[str, float] = {}
        self.last_medians: Dict[str, float] = {}

    def update_and_total(
        self, abs_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """Update rolling statistics and return balanced total loss."""

        if not abs_losses:
            raise ValueError("abs_losses must contain at least one entry")

        # Select a reference tensor for dtype/device creation
        ref_tensor = next(iter(abs_losses.values()))
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        coeffs: Dict[str, float] = {}
        medians: Dict[str, float] = {}

        # When no balancing window or no active weights, revert to classic weighting
        if self.window == 0 or not self.normalised:
            total = torch.zeros((), device=device, dtype=dtype)
            for key, tensor in abs_losses.items():
                weight = self.desired_weights.get(key, 0.0)
                coeffs[key] = weight
                medians[key] = float(
                    tensor.detach().abs().float().cpu().item()
                )
                if weight > 0:
                    total = total + tensor * weight

            self.last_coeffs = coeffs
            self.last_medians = medians
            return total, coeffs, medians

        # Update buffers with most recent absolute losses
        for key, tensor in abs_losses.items():
            if key not in self.normalised:
                continue
            scalar = float(tensor.detach().abs().float().cpu().item())
            self.buffers[key].append(scalar)

        # Compute percentiles and derive coefficients
        for key, ratio in self.normalised.items():
            buffer = list(self.buffers.get(key, []))
            if len(buffer) == 0:
                median = 1.0
            else:
                median = float(np.percentile(buffer, self.percentile))
                if not np.isfinite(median) or median <= 0:
                    median = 1.0
            medians[key] = median
            coeff_value = (ratio / max(median, 1e-12)) * self.total_weight
            coeffs[key] = coeff_value

        total = torch.zeros((), device=device, dtype=dtype)
        for key, tensor in abs_losses.items():
            weight = coeffs.get(key, 0.0)
            if weight > 0:
                total = total + tensor * weight

        self.last_coeffs = coeffs
        self.last_medians = medians
        return total, coeffs, medians


class VaeTrainingCore:
    """Handles VAE-specific training logic."""

    def __init__(self, config: Any):
        self.config = config
        self.loss_balancer: Optional[MedianLossBalancer] = None
        self._lpips_model: Optional[torch.nn.Module] = None
        self._warned_missing_lpips = False
        self._loss_weights: Dict[str, float] = {}
        self._sobel_fn: Callable[[torch.Tensor], torch.Tensor] = sobel_edges
        self._use_latent_mean: bool = False
        self._fixed_sample_batch: Optional[torch.Tensor] = None
        self._sample_images_dir: Optional[str] = None
        self._sample_max_frames: int = 16

    def compute_vae_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        loss_balancer: Optional[MedianLossBalancer] = None,
        lpips_model: Optional[torch.nn.Module] = None,
        use_latent_mean: bool = False,
        reconstruction_loss_type: str = "mse",
        sobel_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute VAE losses including optional perceptual terms."""

        weights = loss_weights or {
            "mse": 1.0,
            "mae": 0.0,
            "lpips": 0.0,
            "edge": 0.0,
            "kl": 1e-6,
        }

        primary = reconstruction_loss_type.lower()
        if primary == "l1":
            primary_fn = lambda x, y: F.l1_loss(x, y, reduction="mean")
        elif primary == "huber":
            primary_fn = lambda x, y: F.smooth_l1_loss(x, y, reduction="mean")
        else:
            primary_fn = lambda x, y: F.mse_loss(x, y, reduction="mean")

        original_f32 = original.to(torch.float32)
        reconstructed_f32 = reconstructed.to(torch.float32)

        loss_dict: Dict[str, Any] = {}
        abs_losses: Dict[str, torch.Tensor] = {}

        if weights.get("mse", 0.0) > 0.0:
            mse_loss = primary_fn(reconstructed_f32, original_f32)
            loss_dict["mse"] = mse_loss
            abs_losses["mse"] = mse_loss
        else:
            loss_dict["mse"] = reconstructed_f32.new_zeros(())

        if weights.get("mae", 0.0) > 0.0:
            mae_loss = F.l1_loss(reconstructed_f32, original_f32, reduction="mean")
            loss_dict["mae"] = mae_loss
            abs_losses["mae"] = mae_loss
        else:
            loss_dict["mae"] = reconstructed_f32.new_zeros(())

        if weights.get("lpips", 0.0) > 0.0 and lpips_model is not None:
            lpips_a = reconstructed_f32
            lpips_b = original_f32
            if lpips_a.dim() == 5:
                b, c, t, h, w = lpips_a.shape
                lpips_a = lpips_a.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                lpips_b = lpips_b.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            lpips_loss = lpips_model(lpips_a, lpips_b).mean()
            loss_dict["lpips"] = lpips_loss
            abs_losses["lpips"] = lpips_loss
        else:
            loss_dict["lpips"] = reconstructed_f32.new_zeros(())

        if weights.get("edge", 0.0) > 0.0 and sobel_fn is not None:
            edge_loss = F.l1_loss(
                sobel_fn(reconstructed_f32), sobel_fn(original_f32), reduction="mean"
            )
            loss_dict["edge"] = edge_loss
            abs_losses["edge"] = edge_loss
        else:
            loss_dict["edge"] = reconstructed_f32.new_zeros(())

        kl_weight = weights.get("kl", 0.0)
        if (
            kl_weight > 0.0
            and mu is not None
            and logvar is not None
            and not use_latent_mean
        ):
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss_dict["kl"] = kl_loss
            abs_losses["kl"] = kl_loss
        else:
            loss_dict["kl"] = reconstructed_f32.new_zeros(())

        coeffs: Dict[str, float] = {}
        medians: Dict[str, float] = {}
        if loss_balancer is not None and abs_losses:
            total_loss, coeffs, medians = loss_balancer.update_and_total(abs_losses)
        else:
            total_loss = reconstructed_f32.new_zeros(())
            for key, tensor in abs_losses.items():
                weight = weights.get(key, 0.0)
                coeffs[key] = weight
                medians[key] = float(tensor.detach().abs().float().cpu().item())
                total_loss = total_loss + tensor * weight

        loss_dict["total"] = total_loss
        loss_dict["coeffs"] = coeffs
        loss_dict["medians"] = medians

        return total_loss, loss_dict

    def _extract_latent_stats(
        self,
        encoded: Any,
        use_latent_mean: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Normalize different encoder outputs to (latent, mu, logvar)."""

        mu: Optional[torch.Tensor] = None
        logvar: Optional[torch.Tensor] = None

        if isinstance(encoded, list):
            if len(encoded) == 0:
                raise ValueError("Encoder returned an empty list")
            if isinstance(encoded[0], torch.Tensor):
                latent = torch.stack(encoded, dim=0)
                return latent, None, None
            latent = encoded
            return latent, None, None

        if isinstance(encoded, tuple):
            if len(encoded) >= 3:
                latent = encoded[0]
                mu = encoded[1]
                logvar = encoded[2]
                if use_latent_mean and mu is not None:
                    latent = mu
            elif len(encoded) == 2:
                mu = encoded[0]
                logvar = encoded[1]
                if use_latent_mean:
                    latent = mu
                else:
                    std = torch.exp(0.5 * logvar)
                    latent = mu + torch.randn_like(std) * std
            else:
                latent = encoded[0]
        elif isinstance(encoded, torch.Tensor):
            latent = encoded
        elif hasattr(encoded, "latent_dist"):
            latent_dist = encoded.latent_dist
            mu = getattr(latent_dist, "mean", None)
            if hasattr(latent_dist, "logvar"):
                logvar = latent_dist.logvar
            elif hasattr(latent_dist, "variance"):
                logvar = torch.log(latent_dist.variance + 1e-12)
            elif hasattr(latent_dist, "std"):
                logvar = 2 * torch.log(latent_dist.std + 1e-12)

            if use_latent_mean and mu is not None:
                latent = mu
            elif hasattr(latent_dist, "sample"):
                latent = latent_dist.sample()
            else:
                latent = mu if mu is not None else encoded
        else:
            latent = encoded

        return latent, mu, logvar

    def _decode_output(self, decoded: Any) -> torch.Tensor:
        if isinstance(decoded, tuple):
            return decoded[0]
        if hasattr(decoded, "sample"):
            return decoded.sample
        if isinstance(decoded, list):
            if len(decoded) == 0:
                raise ValueError("Decoder returned an empty list")
            if isinstance(decoded[0], torch.Tensor):
                return torch.stack(decoded, dim=0)
        return decoded

    def _ensure_sample_dir(self, args: argparse.Namespace) -> str:
        if self._sample_images_dir is not None:
            return self._sample_images_dir

        base_dir = getattr(args, "sample_output_dir", None)
        if base_dir is None:
            base_dir = os.path.join(
                getattr(args, "output_dir", os.getcwd()), "samples", "vae"
            )
        else:
            base_dir = os.path.join(base_dir, "vae")

        os.makedirs(base_dir, exist_ok=True)
        self._sample_images_dir = base_dir
        return base_dir

    def maybe_log_samples(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        vae: Any,
        network_dtype: torch.dtype,
        global_step: int,
        current_batch: Optional[torch.Tensor],
    ) -> None:
        if not accelerator.is_main_process:
            return

        if self._fixed_sample_batch is None and isinstance(current_batch, torch.Tensor):
            if current_batch.numel() > 0:
                sample_count = min(4, current_batch.shape[0])
                try:
                    self._fixed_sample_batch = (
                        current_batch[:sample_count]
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .clone()
                    )
                except Exception:
                    self._fixed_sample_batch = None

        if self._fixed_sample_batch is None:
            return

        sample_dir = self._ensure_sample_dir(args)

        with torch.no_grad():
            sample = self._fixed_sample_batch.to(
                device=accelerator.device, dtype=network_dtype
            )
            reconstructed, reference, _, _ = self.call_vae(
                args,
                accelerator,
                vae,
                sample,
                network_dtype,
                use_latent_mean=self._use_latent_mean,
            )

        def _flatten_video(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 5:
                b, c, t, h, w = tensor.shape
                tensor = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                if tensor.shape[0] > self._sample_max_frames:
                    tensor = tensor[: self._sample_max_frames]
            return tensor

        try:
            import torchvision.utils as vutils  # type: ignore
        except Exception:
            logger.warning(
                "torchvision is required to save VAE samples. Install torchvision to enable sample logging."
            )
            return

        orig = _flatten_video(reference.detach().to(torch.float32).cpu())
        recon = _flatten_video(reconstructed.detach().to(torch.float32).cpu())

        nrow = max(1, min(4, orig.shape[0]))
        try:
            orig_path = os.path.join(
                sample_dir, f"vae_step_{global_step:08d}_orig.png"
            )
            recon_path = os.path.join(
                sample_dir, f"vae_step_{global_step:08d}_recon.png"
            )
            vutils.save_image(
                orig,
                orig_path,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1),
            )
            vutils.save_image(
                recon,
                recon_path,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1),
            )
        except Exception as exc:  # pragma: no cover - file system issues
            logger.warning(f"Failed to save VAE samples: {exc}")

        sample_logs: Dict[str, float] = {"vae/sample_saved": 1.0}

        if self._lpips_model is not None:
            try:
                lpips_a = reconstructed.to(torch.float32)
                lpips_b = reference.to(torch.float32)
                if lpips_a.dim() == 5:
                    b, c, t, h, w = lpips_a.shape
                    lpips_a = lpips_a.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                    lpips_b = lpips_b.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                lpips_val = (
                    self._lpips_model(lpips_a, lpips_b)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                sample_logs["vae/sample_lpips"] = float(lpips_val)
            except Exception:
                pass

        if sample_logs:
            try:
                from utils.tensorboard_utils import (
                    apply_direction_hints_to_logs as _adh,
                )

                sample_logs = _adh(args, sample_logs)
            except Exception:
                pass

            accelerator.log(sample_logs, step=global_step)

    def call_vae(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        vae: Any,
        images: torch.Tensor,
        network_dtype: torch.dtype,
        use_latent_mean: bool,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Call the VAE model and compute reconstruction."""

        images = images.to(device=accelerator.device, dtype=network_dtype)

        vae_module = vae
        if hasattr(vae_module, "vae") and hasattr(vae_module.vae, "encode"):
            vae_module = vae_module.vae
        if hasattr(vae_module, "module"):
            vae_module = vae_module.module

        with accelerator.autocast():
            if hasattr(vae_module, "encode") and hasattr(vae_module, "decode"):
                encoded = vae_module.encode(images)
                latent, mu, logvar = self._extract_latent_stats(
                    encoded, use_latent_mean
                )
                reconstructed = self._decode_output(vae_module.decode(latent))
            else:
                output = vae_module(images)
                if isinstance(output, tuple):
                    reconstructed = output[0]
                    mu = output[1] if len(output) > 1 else None
                    logvar = output[2] if len(output) > 2 else None
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
                    args,
                    accelerator,
                    vae,
                    images,
                    vae.dtype,
                    use_latent_mean=self._use_latent_mean,
                )

                val_loss, _ = self.compute_vae_loss(
                    reconstructed,
                    original,
                    mu,
                    logvar,
                    loss_weights=self._loss_weights,
                    loss_balancer=None,
                    lpips_model=self._lpips_model,
                    use_latent_mean=self._use_latent_mean,
                    reconstruction_loss_type=getattr(
                        args, "vae_reconstruction_loss", "mse"
                    ),
                    sobel_fn=self._sobel_fn,
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

        reconstruction_loss_type = getattr(args, "vae_reconstruction_loss", "mse")

        loss_weights = {
            "mse": float(getattr(args, "vae_mse_weight", 1.0)),
            "mae": float(getattr(args, "vae_mae_weight", 0.0)),
            "lpips": float(getattr(args, "vae_lpips_weight", 0.0)),
            "edge": float(getattr(args, "vae_edge_weight", 0.0)),
            "kl": float(getattr(args, "vae_kl_weight", 1e-6)),
        }

        # Legacy compatibility: respect reconstruction_loss switch when weights untouched
        if (
            reconstruction_loss_type.lower() == "l1"
            and not config_key_provided(args, "vae_mae_weight")
            and not config_key_provided(args, "vae_mse_weight")
        ):
            loss_weights["mse"] = 0.0
            loss_weights["mae"] = 1.0

        self._loss_weights = loss_weights
        self._use_latent_mean = bool(getattr(args, "vae_decoder_latent_mean", False))

        balancer_window = max(int(getattr(args, "vae_loss_balancer_window", 0)), 0)
        balancer_percentile = float(
            getattr(args, "vae_loss_balancer_percentile", 95.0)
        )

        if balancer_window > 0 and any(v > 0 for v in loss_weights.values()):
            self.loss_balancer = MedianLossBalancer(
                loss_weights, balancer_window, balancer_percentile
            )
            logger.info(
                "Median loss balancer enabled (window=%s, percentile=%s)",
                balancer_window,
                balancer_percentile,
            )
        else:
            self.loss_balancer = None

        if loss_weights.get("lpips", 0.0) > 0.0:
            try:
                self._lpips_model = get_lpips_model(accelerator.device)
            except RuntimeError as exc:
                if not self._warned_missing_lpips:
                    logger.warning(
                        "LPIPS weighting requested but unavailable: %s. The term will be disabled.",
                        exc,
                    )
                    self._warned_missing_lpips = True
                self._lpips_model = None
                self._loss_weights["lpips"] = 0.0
        else:
            self._lpips_model = None

        logger.info(
            "Starting VAE training with weights %s, reconstruction loss=%s, latent_mean=%s",
            self._loss_weights,
            reconstruction_loss_type,
            self._use_latent_mean,
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

                    if (
                        self._fixed_sample_batch is None
                        and accelerator.is_main_process
                        and isinstance(images, torch.Tensor)
                        and images.numel() > 0
                    ):
                        sample_count = min(4, images.shape[0])
                        try:
                            self._fixed_sample_batch = (
                                images[:sample_count]
                                .detach()
                                .to(torch.float32)
                                .cpu()
                                .clone()
                            )
                        except Exception:
                            self._fixed_sample_batch = None

                    # Apply sample weights if present
                    sample_weights = batch.get("weight", None)

                    # Forward pass through VAE
                    reconstructed, original, mu, logvar = self.call_vae(
                        args,
                        accelerator,
                        vae,
                        images,
                        network_dtype,
                        use_latent_mean=self._use_latent_mean,
                    )

                    loss, loss_dict = self.compute_vae_loss(
                        reconstructed,
                        original,
                        mu,
                        logvar,
                        loss_weights=self._loss_weights,
                        loss_balancer=self.loss_balancer,
                        lpips_model=self._lpips_model,
                        use_latent_mean=self._use_latent_mean,
                        reconstruction_loss_type=reconstruction_loss_type,
                        sobel_fn=self._sobel_fn,
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

                        # Update adaptive LR schedulers with gradient stats (before optimizer.step)
                        try:
                            _sched = lr_scheduler
                            if hasattr(_sched, "update_gradient_stats"):
                                _sched.update_gradient_stats()  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    optimizer.step()

                    # Update adaptive LR schedulers with loss (post-backward, pre-step is also fine)
                    try:
                        _sched = lr_scheduler
                        _loss_val = float(loss.detach().item())
                        if hasattr(_sched, "update_training_stats"):
                            _sched.update_training_stats(_loss_val)  # type: ignore[attr-defined]
                        elif hasattr(_sched, "update_metrics"):
                            _sched.update_metrics(_loss_val)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    should_sampling = should_sample_images(
                        args, global_step, epoch=epoch + 1
                    )

                    if should_sampling:
                        if accelerator.is_main_process:
                            self.maybe_log_samples(
                                args,
                                accelerator,
                                vae,
                                network_dtype,
                                global_step,
                                original,
                            )
                        accelerator.wait_for_everyone()

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
                            try:
                                from utils.tensorboard_utils import (
                                    apply_direction_hints_to_logs as _adh,
                                )

                                _logs = _adh(args, {"vae/val_loss": val_loss})
                            except Exception:
                                _logs = {"vae/val_loss": val_loss}
                            accelerator.log(_logs, step=global_step)

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
                    "vae/loss/total": current_loss,
                    "vae/loss/average": avr_loss,
                }

                for term in ("mse", "mae", "lpips", "edge", "kl"):
                    value = loss_dict.get(term)
                    if isinstance(value, torch.Tensor):
                        logs[f"vae/loss/{term}"] = float(value.detach().item())

                coeffs = loss_dict.get("coeffs", {}) or {}
                for key, value in coeffs.items():
                    logs[f"vae/loss/coeff_{key}"] = float(value)

                medians = loss_dict.get("medians", {}) or {}
                for key, value in medians.items():
                    logs[f"vae/loss/median_{key}"] = float(value)

                # Add learning rates
                lrs = lr_scheduler.get_last_lr()
                for i, lr in enumerate(lrs):
                    if i < len(lr_descriptions):
                        logs[f"lr/{lr_descriptions[i]}"] = lr
                    else:
                        logs[f"lr/group{i}"] = lr

                progress_bar.set_postfix(logs)

                if len(accelerator.trackers) > 0:
                    try:
                        from utils.tensorboard_utils import (
                            apply_direction_hints_to_logs as _adh,
                        )

                        logs = _adh(args, logs)
                    except Exception:
                        pass
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
                try:
                    from utils.tensorboard_utils import (
                        apply_direction_hints_to_logs as _adh,
                    )

                    _logs = _adh(args, {"vae/val_loss": val_loss})
                except Exception:
                    _logs = {"vae/val_loss": val_loss}
                accelerator.log(_logs, step=global_step)
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
