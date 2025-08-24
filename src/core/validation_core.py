"""Core validation logic for WAN network trainer.

This module handles validation, model evaluation, and validation metrics computation.
Extracted from training_core.py to improve code organization and maintainability.
"""

import argparse
import logging
from typing import Any, Dict, Optional
import torch
import numpy as np
from accelerate import Accelerator

from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import get_sigmas
from common.logger import get_logger
from core.metrics import get_throughput_metrics, get_total_runtime

logger = get_logger(__name__, level=logging.INFO)


def compute_snr(
    noise_scheduler: FlowMatchDiscreteScheduler, timesteps: torch.Tensor
) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio (SNR) for given timesteps.

    Args:
        noise_scheduler: The noise scheduler instance
        timesteps: Tensor of timesteps

    Returns:
        SNR values for each timestep
    """
    # Get sigma values for the timesteps
    sigmas = get_sigmas(
        noise_scheduler,
        timesteps,
        timesteps.device,
        n_dim=4,  # BCFHW format
        dtype=timesteps.dtype,
        source="validation/snr",
    )

    # SNR = 1 / sigma^2 (signal power / noise power)
    snr = 1.0 / (sigmas**2)
    return snr


class ValidationCore:
    """Handles validation logic, model evaluation, and validation metrics."""

    def __init__(self, config: Any, fluxflow_config: Dict[str, Any]):
        self.config = config
        self.fluxflow_config = fluxflow_config
        self._setup_validation_timesteps()

    def _setup_validation_timesteps(self) -> None:
        """Parse and setup validation timesteps from configuration only.

        Accepts either a single integer (as string or int) or a comma-separated
        list of integers. Any invalid value defaults to [600, 700, 800, 900].
        """
        raw = getattr(self.config, "validation_timesteps", "600,700,800,900")
        try:
            # Single integer path
            if isinstance(raw, int):
                self.validation_timesteps = [raw]
                return
            raw_str = str(raw).strip()
            if "," in raw_str:
                self.validation_timesteps = [
                    int(t.strip()) for t in raw_str.split(",") if t.strip()
                ]
            else:
                self.validation_timesteps = [int(raw_str)]
        except Exception:
            logger.warning(
                f"Invalid validation_timesteps format: {raw}. Using default [600, 700, 800, 900]"
            )
            self.validation_timesteps = [600, 700, 800, 900]

        logger.info(f"Validation timesteps: {self.validation_timesteps}")
        # Store dynamic mode defaults from config if present (used when args missing)
        self.validation_timesteps_mode = getattr(
            self.config, "validation_timesteps_mode", "fixed"
        )
        self.validation_timesteps_count = int(
            getattr(self.config, "validation_timesteps_count", 4)
        )
        self.validation_timesteps_min = getattr(
            self.config, "validation_timesteps_min", None
        )
        self.validation_timesteps_max = getattr(
            self.config, "validation_timesteps_max", None
        )
        self.validation_timesteps_jitter = int(
            getattr(self.config, "validation_timesteps_jitter", 0)
        )

    def _compute_and_gather_metrics(
        self,
        accelerator: Accelerator,
        loss_tensors_list: list[torch.Tensor],
        metric_name: str,
    ) -> Dict[str, float]:
        """Concatenates, gathers, and computes statistics for a list of loss tensors.

        Args:
            accelerator: The accelerator instance for distributed training
            loss_tensors_list: List of loss tensors to process
            metric_name: Base name for the metrics (e.g., "velocity_loss")

        Returns:
            Dictionary containing computed statistics (mean, std, min, max)
        """
        if not loss_tensors_list:
            return {}

        # 1. Concatenate ONCE
        local_tensor = torch.cat(loss_tensors_list)

        # 2. Gather ONCE
        gathered_tensor = accelerator.gather_for_metrics(local_tensor)

        # 3. Compute all stats from the final gathered tensor
        if accelerator.is_main_process:
            if isinstance(gathered_tensor, torch.Tensor):
                stats: Dict[str, float] = {
                    f"{metric_name}_mean": gathered_tensor.mean().item(),
                    f"{metric_name}_std": gathered_tensor.std().item(),
                    f"{metric_name}_min": gathered_tensor.min().item(),
                    f"{metric_name}_max": gathered_tensor.max().item(),
                }
                # Quantile statistics (defensive in case torch.quantile not available)
                try:
                    median_val = torch.quantile(gathered_tensor, 0.5).item()
                    p90_val = torch.quantile(gathered_tensor, 0.9).item()
                    stats[f"{metric_name}_p50"] = median_val
                    stats[f"{metric_name}_p90"] = p90_val
                except Exception:
                    try:
                        stats[f"{metric_name}_p50"] = torch.median(
                            gathered_tensor
                        ).item()
                    except Exception:
                        pass
                return stats
            else:
                # Handle case where gathered_tensor is not a tensor
                gathered_tensor = torch.tensor(gathered_tensor)
                stats: Dict[str, float] = {
                    f"{metric_name}_mean": gathered_tensor.mean().item(),
                    f"{metric_name}_std": gathered_tensor.std().item(),
                    f"{metric_name}_min": gathered_tensor.min().item(),
                    f"{metric_name}_max": gathered_tensor.max().item(),
                }
                try:
                    median_val = torch.quantile(gathered_tensor, 0.5).item()
                    p90_val = torch.quantile(gathered_tensor, 0.9).item()
                    stats[f"{metric_name}_p50"] = median_val
                    stats[f"{metric_name}_p90"] = p90_val
                except Exception:
                    try:
                        stats[f"{metric_name}_p50"] = torch.median(
                            gathered_tensor
                        ).item()
                    except Exception:
                        pass
                return stats
        return {}  # Return empty dict on non-main processes

    def validate(
        self,
        accelerator: Accelerator,
        transformer: Any,
        val_dataloader: Any,
        noise_scheduler: FlowMatchDiscreteScheduler,
        args: argparse.Namespace,
        control_signal_processor: Optional[Any] = None,
        vae: Optional[Any] = None,
        global_step: Optional[int] = None,
        show_progress: bool = True,
    ) -> tuple[float, float]:
        # Determine validation timesteps according to mode
        try:
            mode = str(
                getattr(
                    args, "validation_timesteps_mode", self.validation_timesteps_mode
                )
            ).lower()
        except Exception:
            mode = "fixed"

        # Base fixed list from args/config
        base_list = None
        try:
            raw = getattr(args, "validation_timesteps", None)
            if raw is None:
                raw = getattr(self.config, "validation_timesteps", None)
            if raw is not None:
                raw_str = str(raw).strip()
                if "," in raw_str:
                    base_list = [
                        int(t.strip()) for t in raw_str.split(",") if t.strip()
                    ]
                else:
                    base_list = [int(raw_str)]
        except Exception:
            base_list = None

        if base_list is None:
            base_list = getattr(
                self,
                "validation_timesteps",
                [100, 200, 300, 400, 500, 600, 700, 800, 900],
            )

        # Bounds for random/jitter
        min_t = getattr(args, "validation_timesteps_min", self.validation_timesteps_min)
        max_t = getattr(args, "validation_timesteps_max", self.validation_timesteps_max)
        if min_t is None:
            min_t = getattr(args, "min_timestep", 0)
        if max_t is None:
            max_t = getattr(args, "max_timestep", 1000)
        min_t = int(min_t)
        max_t = int(max_t)

        if mode == "random":
            count = int(
                getattr(
                    args, "validation_timesteps_count", self.validation_timesteps_count
                )
            )
            count = max(1, count)
            # Sample without replacement within [min_t, max_t]
            if max_t < min_t:
                min_t, max_t = max_t, min_t
            rng = torch.Generator(device=accelerator.device)
            rng.manual_seed(42)  # deterministic across processes
            # Use torch.randint then unique to avoid dependency on numpy
            samples = []
            attempts = 0
            while len(samples) < count and attempts < count * 5:
                t = int(torch.randint(min_t, max_t + 1, (1,), generator=rng).item())
                if t not in samples:
                    samples.append(t)
                attempts += 1
            self.validation_timesteps = samples
        elif mode == "jitter":
            jitter = int(
                getattr(
                    args,
                    "validation_timesteps_jitter",
                    self.validation_timesteps_jitter,
                )
            )
            if jitter > 0:
                jittered = []
                rng = torch.Generator(device=accelerator.device)
                rng.manual_seed(
                    42 + (global_step or 0)
                )  # change over time but deterministic
                for t in base_list:
                    low = max(min_t, t - jitter)
                    high = min(max_t, t + jitter)
                    if high < low:
                        low, high = high, low
                    jt = int(torch.randint(low, high + 1, (1,), generator=rng).item())
                    jittered.append(jt)
                self.validation_timesteps = jittered
            else:
                self.validation_timesteps = base_list
        else:
            # fixed
            self.validation_timesteps = base_list
        """Run validation and return average validation loss for both velocity and direct noise prediction.

        Args:
            global_step: Current training step for logging per-timestep metrics
            show_progress: Whether to show progress bars

        Note:
            Noise generation behavior is controlled by args.use_unique_noise_per_batch:
            - True (recommended): Each batch gets unique but deterministic noise (seed = 42 + batch_idx)
            - False (legacy): All batches use the same noise pattern (seed = 42)

            Using unique noise per batch provides more reliable validation metrics.

        Returns:
            tuple[float, float]: (velocity_loss, direct_noise_loss)

        Example:
            velocity_loss, direct_noise_loss = validation_core.validate(...)
            print(f"Velocity loss: {velocity_loss:.5f}")
            print(f"Direct noise loss: {direct_noise_loss:.5f}")
        """
        logger.info("Running validation...")
        unwrapped_model = accelerator.unwrap_model(transformer)
        unwrapped_model.switch_block_swap_for_inference()
        unwrapped_model.eval()

        fixed_seed = 42
        total_velocity_loss = 0.0
        total_direct_noise_loss = 0.0
        num_timesteps = len(self.validation_timesteps)

        # Collect timestep losses and SNRs for statistics (across all timesteps)
        all_timestep_avg_velocity_losses = []
        all_timestep_avg_direct_noise_losses = []
        all_timestep_snrs = []

        if num_timesteps == 0:
            logger.warning(
                "Validation timesteps list is empty. Skipping validation and returning 0 loss."
            )
            # Switch back to train mode before returning
            unwrapped_model.train()
            unwrapped_model.switch_block_swap_for_training()
            return 0.0, 0.0

        logger.info(
            f"Validating across {num_timesteps} timesteps: {self.validation_timesteps}"
        )

        # Setup nested progress bars for better UX
        if show_progress and accelerator.is_main_process:
            from tqdm import tqdm

            timestep_pbar = tqdm(
                self.validation_timesteps,
                desc="Validating Timesteps",
                disable=not accelerator.is_main_process,
                leave=True,
            )
        else:
            timestep_pbar = self.validation_timesteps

        with torch.no_grad():
            # Validate across all timesteps
            for timestep_idx, current_timestep in enumerate(timestep_pbar):
                if show_progress and accelerator.is_main_process:
                    dataloader_pbar = tqdm(
                        val_dataloader,
                        desc=f"Timestep {current_timestep}",
                        leave=False,
                        disable=not accelerator.is_main_process,
                    )
                else:
                    dataloader_pbar = val_dataloader

                # These lists collect per-batch losses for the CURRENT timestep
                batch_velocity_losses = []
                batch_direct_noise_losses = []
                batch_item_counts = []  # number of items per-batch for coverage

                # Calculate SNR only once for the current timestep (optimization)
                fixed_timesteps_tensor = torch.full(
                    (1,),
                    current_timestep,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                snr_for_timestep = (
                    compute_snr(noise_scheduler, fixed_timesteps_tensor).mean().item()
                )

                # Optional perceptual SNR (SSIM-based) on a very small subsample for cost control
                # Requires decoded frames from VAE and reference pixels in batch
                enable_psnr = bool(getattr(args, "enable_perceptual_snr", False))
                max_psnr_items = int(getattr(args, "perceptual_snr_max_items", 4))

                # Track sample-weighted (per-example) sums across all batches/timesteps
                if timestep_idx == 0:
                    # Initialize accumulators once (on first timestep)
                    local_velocity_loss_sum = torch.tensor(
                        0.0, device=accelerator.device, dtype=torch.float32
                    )
                    local_velocity_loss_count = torch.tensor(
                        0.0, device=accelerator.device, dtype=torch.float32
                    )
                    local_direct_noise_loss_sum = torch.tensor(
                        0.0, device=accelerator.device, dtype=torch.float32
                    )
                    local_direct_noise_loss_count = torch.tensor(
                        0.0, device=accelerator.device, dtype=torch.float32
                    )

                for step, batch in enumerate(dataloader_pbar):
                    latents = batch["latents"]

                    # Text embedding preparation
                    if "t5" in batch:
                        llm_embeds = [
                            t.to(device=accelerator.device, dtype=unwrapped_model.dtype)
                            for t in batch["t5"]
                        ]
                    else:
                        t5_keys = [
                            k for k in batch.keys() if k.startswith("varlen_t5_")
                        ]
                        if t5_keys:
                            llm_embeds = batch[t5_keys[0]]
                        else:
                            raise ValueError(
                                "No text encoder outputs found in validation batch."
                            )

                    latents = latents.to(accelerator.device, dtype=torch.float32)
                    if not isinstance(llm_embeds, list):
                        llm_embeds = llm_embeds.to(
                            accelerator.device, dtype=unwrapped_model.dtype
                        )

                    # Deterministic noise generation with unique seed per batch for proper validation
                    with torch.random.fork_rng(devices=[accelerator.device]):
                        use_unique_noise = True
                        if hasattr(args, "use_unique_noise_per_batch"):
                            use_unique_noise = bool(args.use_unique_noise_per_batch)
                        if not use_unique_noise and accelerator.is_main_process:
                            logger.warning(
                                "use_unique_noise_per_batch=False is deprecated for validation and may lead to less reliable metrics; defaulting to True in future releases."
                            )

                        if use_unique_noise:
                            # Use unique seed per batch for proper validation (recommended)
                            batch_seed = fixed_seed + step
                            torch.manual_seed(batch_seed)
                            if accelerator.device.type == "cuda":
                                torch.cuda.manual_seed(batch_seed)
                        else:
                            # Legacy behavior: same seed for all batches (not recommended)
                            torch.manual_seed(fixed_seed)
                            if accelerator.device.type == "cuda":
                                torch.cuda.manual_seed(fixed_seed)
                        noise = torch.randn_like(latents)

                    timesteps = torch.full(
                        (latents.size(0),),
                        current_timestep,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )

                    sigma = get_sigmas(
                        noise_scheduler,
                        timesteps,
                        accelerator.device,
                        latents.dim(),
                        latents.dtype,
                        source="validation",
                    )
                    noisy_model_input = sigma * noise + (1.0 - sigma) * latents

                    # Apply FluxFlow temporal augmentation if enabled (same as training)
                    perturbed_latents_for_target = latents.clone()
                    if self.fluxflow_config.get("enable_fluxflow", False):
                        frame_dim = self.fluxflow_config.get("frame_dim_in_batch", 2)
                        if latents.ndim > frame_dim and latents.shape[frame_dim] > 1:
                            import utils.fluxflow_augmentation as fluxflow_augmentation

                            perturbed_latents_for_target = (
                                fluxflow_augmentation.apply_fluxflow_to_batch(
                                    perturbed_latents_for_target, self.fluxflow_config
                                )
                            )

                    # Control LoRA processing (same as training)
                    control_latents = None
                    if (
                        hasattr(args, "enable_control_lora")
                        and args.enable_control_lora
                    ):
                        if control_signal_processor is not None:
                            # Match training network dtype for consistency
                            network_dtype = noisy_model_input.dtype
                            control_latents = (
                                control_signal_processor.process_control_signal(
                                    args,
                                    accelerator,
                                    batch,
                                    latents,
                                    network_dtype,
                                    vae,
                                )
                            )

                        # Fallback to using image latents as control signal
                        if control_latents is None:
                            control_latents = latents.detach().clone()

                    # Calculate sequence length
                    lat_f, lat_h, lat_w = latents.shape[2:5]
                    seq_len = (
                        lat_f
                        * lat_h
                        * lat_w
                        // (
                            self.config.patch_size[0]
                            * self.config.patch_size[1]
                            * self.config.patch_size[2]
                        )
                    )

                    # Prepare model input with control signal if control LoRA is enabled
                    if (
                        hasattr(args, "enable_control_lora")
                        and args.enable_control_lora
                    ):
                        if control_latents is not None:
                            control_latents = control_latents.to(
                                device=noisy_model_input.device,
                                dtype=noisy_model_input.dtype,
                            )
                            # Concatenate along the channel dimension (dim=1 for BCFHW)
                            model_input = torch.cat(
                                [noisy_model_input, control_latents], dim=1
                            )
                        else:
                            # Fallback: create zero tensor for control part
                            zero_control = torch.zeros_like(noisy_model_input)
                            model_input = torch.cat(
                                [noisy_model_input, zero_control], dim=1
                            )
                    else:
                        model_input = noisy_model_input

                    # Forward pass with consistent model calling
                    pred = unwrapped_model(
                        model_input,
                        t=timesteps,
                        context=llm_embeds,
                        seq_len=seq_len,
                        clip_fea=None,
                    )

                    # Model returns a list of output tensors; stack them for consistent processing
                    pred = torch.stack(pred, dim=0)

                    # Calculate both velocity and direct noise prediction losses

                    # 1. Velocity prediction (existing approach)
                    velocity_target = noise - perturbed_latents_for_target.to(
                        device=accelerator.device, dtype=torch.float32
                    )
                    velocity_loss = torch.nn.functional.mse_loss(
                        pred, velocity_target, reduction="none"
                    )
                    velocity_loss = velocity_loss.mean(
                        dim=[1, 2, 3, 4]
                    )  # Per-batch item mean
                    batch_velocity_losses.append(velocity_loss)
                    batch_item_counts.append(float(velocity_loss.numel()))
                    # Update sample-weighted accumulators
                    local_velocity_loss_sum = (
                        local_velocity_loss_sum + velocity_loss.sum()
                    )
                    local_velocity_loss_count = (
                        local_velocity_loss_count
                        + torch.tensor(
                            float(velocity_loss.numel()),
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    )

                    # 2. Direct noise prediction (new approach - matches your example)
                    direct_noise_target = noise.to(
                        device=accelerator.device, dtype=torch.float32
                    )
                    direct_noise_loss = torch.nn.functional.mse_loss(
                        pred, direct_noise_target, reduction="none"
                    )
                    direct_noise_loss = direct_noise_loss.mean(
                        dim=[1, 2, 3, 4]
                    )  # Per-batch item mean
                    batch_direct_noise_losses.append(direct_noise_loss)
                    local_direct_noise_loss_sum = (
                        local_direct_noise_loss_sum + direct_noise_loss.sum()
                    )
                    local_direct_noise_loss_count = (
                        local_direct_noise_loss_count
                        + torch.tensor(
                            float(direct_noise_loss.numel()),
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    )

                    # Perceptual SNR (SSIM): compute for a tiny subsample only when enabled
                    if enable_psnr and vae is not None and "pixels" in batch:
                        try:
                            # Decode predicted latents to frames
                            with torch.autocast(
                                device_type=str(accelerator.device).split(":")[0],
                                enabled=False,
                            ):
                                pred_latents = pred.to(torch.float32)
                                decoded = vae.decode(
                                    pred_latents / getattr(vae, "scaling_factor", 1.0)
                                )
                                pred_frames = (
                                    torch.stack(decoded, dim=0)
                                    if isinstance(decoded, list)
                                    else decoded
                                )
                            # Reference frames from batch
                            ref_frames = torch.stack(batch["pixels"], dim=0).to(
                                device=pred_frames.device, dtype=pred_frames.dtype
                            )
                            # Limit items for cost
                            b = min(
                                pred_frames.shape[0],
                                ref_frames.shape[0],
                                max_psnr_items,
                            )
                            if b > 0:
                                pf = pred_frames[:b, :, :, :, :]
                                rf = ref_frames[:b, :, :, :, :]
                                # Convert to grayscale-ish luminance for SSIM stability
                                pf_y = pf.mean(dim=2)
                                rf_y = rf.mean(dim=2)
                                # Compute SSIM per-frame quickly
                                # Using windowless approximation: 1 - ((pf - rf)^2 / (pf^2 + rf^2 + eps))
                                eps = 1e-6
                                num = (pf_y - rf_y) ** 2
                                den = pf_y**2 + rf_y**2 + eps
                                ssim_approx = 1.0 - (num / den)
                                ssim_approx = torch.clamp(ssim_approx, 0.0, 1.0)
                                # Perceptual SNR (log-scale)
                                psnr_ssim_vals = 10.0 * torch.log10(
                                    1.0 / ((1.0 - ssim_approx) ** 2 + eps)
                                )
                                psnr_ssim_mean = float(psnr_ssim_vals.mean().item())
                                # Route PSNR to snr_other if splitting is enabled
                                psnr_prefix = (
                                    "snr_other"
                                    if getattr(args, "snr_split_namespaces", False)
                                    else "snr"
                                )
                                accelerator.log(
                                    {
                                        f"{psnr_prefix}/val/psnr_ssim_t{current_timestep}": psnr_ssim_mean
                                    },
                                    step=global_step,
                                )
                        except Exception:
                            pass

                # Gather and compute metrics efficiently (refactored approach)
                velocity_metrics = self._compute_and_gather_metrics(
                    accelerator, batch_velocity_losses, "velocity_loss"
                )
                noise_metrics = self._compute_and_gather_metrics(
                    accelerator, batch_direct_noise_losses, "direct_noise_loss"
                )

                # SNR-binned loss and coverage per timestep (quantile bins on gathered per-sample loss)
                if accelerator.is_main_process and batch_velocity_losses:
                    try:
                        # Concatenate gathered per-sample losses for velocity
                        vel_cat = torch.cat(batch_velocity_losses)
                        vel_gathered = accelerator.gather_for_metrics(vel_cat)
                        if not isinstance(vel_gathered, torch.Tensor):
                            vel_gathered = torch.tensor(vel_gathered)

                        # Repeat SNR value to match samples
                        snr_tensor = torch.full_like(vel_gathered, snr_for_timestep)

                        # Define 5 quantile bins on SNR (degenerate here since single timestep)
                        # Keep interface stable: bin 0 holds all samples for this T
                        bin_indices = [torch.ones_like(vel_gathered, dtype=torch.bool)]
                        bin_names = ["bin0"]

                        snr_logs = {}
                        snr_ns = (
                            "snr_other"
                            if getattr(args, "snr_split_namespaces", False)
                            else "snr"
                        )
                        for idx, mask in enumerate(bin_indices):
                            if mask.sum() > 0:
                                mean_loss = float(vel_gathered[mask].mean().item())
                                snr_logs[
                                    f"{snr_ns}/val/snr_{bin_names[idx]}_loss_t{current_timestep}"
                                ] = mean_loss

                        # Coverage: all samples fall into this timestep's bin
                        total_items = float(vel_gathered.numel())
                        snr_logs[f"{snr_ns}/val/coverage_t{current_timestep}"] = (
                            total_items
                        )

                        accelerator.log(snr_logs, step=global_step)
                    except Exception:
                        pass

                # The main process now holds all the computed stats
                if accelerator.is_main_process:
                    timestep_velocity_avg = velocity_metrics.get(
                        "velocity_loss_mean", 0.0
                    )
                    timestep_direct_noise_avg = noise_metrics.get(
                        "direct_noise_loss_mean", 0.0
                    )

                    total_velocity_loss += timestep_velocity_avg
                    total_direct_noise_loss += timestep_direct_noise_avg

                    # Store timestep average losses and SNR for statistics (across all timesteps)
                    all_timestep_avg_velocity_losses.append(timestep_velocity_avg)
                    all_timestep_avg_direct_noise_losses.append(
                        timestep_direct_noise_avg
                    )
                    all_timestep_snrs.append(snr_for_timestep)

                    # Log per-timestep metrics
                    if global_step is not None:
                        # SNR per-timestep under snr_other when splitting is enabled (to limit essentials)
                        log_dict = {}
                        per_ns = (
                            "snr_other"
                            if getattr(args, "snr_split_namespaces", False)
                            else "snr"
                        )
                        log_dict[f"{per_ns}/val/snr_t{current_timestep}"] = (
                            snr_for_timestep
                        )

                        # Dynamically add all computed metrics to the log under timesteps category
                        for key, value in velocity_metrics.items():
                            log_dict[f"val_timesteps/{key}_t{current_timestep}"] = value
                        for key, value in noise_metrics.items():
                            log_dict[f"val_timesteps/{key}_t{current_timestep}"] = value

                        accelerator.log(log_dict, step=global_step)

                    logger.info(
                        f"Timestep {current_timestep} - Velocity loss: {timestep_velocity_avg:.5f}, "
                        f"Direct noise loss: {timestep_direct_noise_avg:.5f}, "
                        f"SNR: {snr_for_timestep:.3f}"
                    )

        # Calculate final average losses across all timesteps (timestep-weighted)
        velocity_final_avg_loss = total_velocity_loss / num_timesteps
        direct_noise_final_avg_loss = total_direct_noise_loss / num_timesteps

        # Calculate sample-weighted (per-example) averages across all batches and timesteps
        velocity_weighted_avg_loss = 0.0
        direct_noise_weighted_avg_loss = 0.0
        try:
            gathered_vel_sum = accelerator.gather_for_metrics(local_velocity_loss_sum)
            gathered_vel_cnt = accelerator.gather_for_metrics(local_velocity_loss_count)
            gathered_dir_sum = accelerator.gather_for_metrics(
                local_direct_noise_loss_sum
            )
            gathered_dir_cnt = accelerator.gather_for_metrics(
                local_direct_noise_loss_count
            )

            if accelerator.is_main_process:
                # Handle potential non-tensor returns defensively
                if not isinstance(gathered_vel_sum, torch.Tensor):
                    gathered_vel_sum = torch.tensor(gathered_vel_sum)
                if not isinstance(gathered_vel_cnt, torch.Tensor):
                    gathered_vel_cnt = torch.tensor(gathered_vel_cnt)
                if not isinstance(gathered_dir_sum, torch.Tensor):
                    gathered_dir_sum = torch.tensor(gathered_dir_sum)
                if not isinstance(gathered_dir_cnt, torch.Tensor):
                    gathered_dir_cnt = torch.tensor(gathered_dir_cnt)

                total_vel_sum = gathered_vel_sum.sum().item()
                total_vel_cnt = max(gathered_vel_cnt.sum().item(), 1.0)
                total_dir_sum = gathered_dir_sum.sum().item()
                total_dir_cnt = max(gathered_dir_cnt.sum().item(), 1.0)

                velocity_weighted_avg_loss = total_vel_sum / total_vel_cnt
                direct_noise_weighted_avg_loss = total_dir_sum / total_dir_cnt
        except Exception as ex:
            if accelerator.is_main_process:
                logger.warning(
                    f"Failed to compute sample-weighted validation metrics: {ex}"
                )

        # Log average losses and correlation statistics to tensorboard under val category
        if global_step is not None and accelerator.is_main_process:
            # Calculate statistics from collected timestep losses
            velocity_loss_std = 0.0
            direct_noise_loss_std = 0.0

            # Only calculate std if we have at least 2 values
            if len(all_timestep_avg_velocity_losses) > 1:
                velocity_loss_std = torch.std(
                    torch.tensor(all_timestep_avg_velocity_losses)
                ).item()
            if len(all_timestep_avg_direct_noise_losses) > 1:
                direct_noise_loss_std = torch.std(
                    torch.tensor(all_timestep_avg_direct_noise_losses)
                ).item()

            # Calculate loss consistency metrics
            velocity_loss_range = 0.0
            direct_noise_loss_range = 0.0

            # Only calculate range if we have at least 2 values
            if len(all_timestep_avg_velocity_losses) > 1:
                velocity_loss_range = max(all_timestep_avg_velocity_losses) - min(
                    all_timestep_avg_velocity_losses
                )
            if len(all_timestep_avg_direct_noise_losses) > 1:
                direct_noise_loss_range = max(
                    all_timestep_avg_direct_noise_losses
                ) - min(all_timestep_avg_direct_noise_losses)

            # Calculate relative performance ratio
            loss_ratio = (
                velocity_final_avg_loss / direct_noise_final_avg_loss
                if direct_noise_final_avg_loss > 0
                else 0.0
            )

            # Calculate Loss/SNR Correlation
            velocity_loss_snr_corr = 0.0
            noise_loss_snr_corr = 0.0
            # Correlation requires at least 2 data points
            if num_timesteps > 1:
                # --- For velocity loss ---
                corr_matrix_vel = np.corrcoef(
                    all_timestep_snrs, all_timestep_avg_velocity_losses
                )
                # Handle NaN case (if one input has zero variance)
                if not np.isnan(corr_matrix_vel[0, 1]):
                    velocity_loss_snr_corr = corr_matrix_vel[0, 1]

                # --- For direct noise loss ---
                corr_matrix_noise = np.corrcoef(
                    all_timestep_snrs, all_timestep_avg_direct_noise_losses
                )
                # Handle NaN case
                if not np.isnan(corr_matrix_noise[0, 1]):
                    noise_loss_snr_corr = corr_matrix_noise[0, 1]

            # Additional across-timestep diagnostics
            velocity_cv_across_timesteps = 0.0
            noise_cv_across_timesteps = 0.0
            if velocity_final_avg_loss > 0:
                velocity_cv_across_timesteps = velocity_loss_std / max(
                    velocity_final_avg_loss, 1e-12
                )
            if direct_noise_final_avg_loss > 0:
                noise_cv_across_timesteps = direct_noise_loss_std / max(
                    direct_noise_final_avg_loss, 1e-12
                )

            # Percentiles of per-timestep average losses (p25/p50/p75)
            vel_p25 = vel_p50 = vel_p75 = 0.0
            noi_p25 = noi_p50 = noi_p75 = 0.0
            if len(all_timestep_avg_velocity_losses) > 0:
                try:
                    vel_tensor = torch.tensor(all_timestep_avg_velocity_losses)
                    vel_p25 = torch.quantile(vel_tensor, 0.25).item()
                    vel_p50 = torch.quantile(vel_tensor, 0.50).item()
                    vel_p75 = torch.quantile(vel_tensor, 0.75).item()
                except Exception:
                    vel_p50 = float(np.median(all_timestep_avg_velocity_losses))
            if len(all_timestep_avg_direct_noise_losses) > 0:
                try:
                    noi_tensor = torch.tensor(all_timestep_avg_direct_noise_losses)
                    noi_p25 = torch.quantile(noi_tensor, 0.25).item()
                    noi_p50 = torch.quantile(noi_tensor, 0.50).item()
                    noi_p75 = torch.quantile(noi_tensor, 0.75).item()
                except Exception:
                    noi_p50 = float(np.median(all_timestep_avg_direct_noise_losses))

            # Best/Worst timesteps by loss
            best_timestep_velocity = worst_timestep_velocity = 0
            best_timestep_direct = worst_timestep_direct = 0
            best_velocity_loss = worst_velocity_loss = 0.0
            best_direct_loss = worst_direct_loss = 0.0
            if len(all_timestep_avg_velocity_losses) > 0:
                vel_losses_np = np.array(all_timestep_avg_velocity_losses)
                best_idx_vel = int(np.argmin(vel_losses_np))
                worst_idx_vel = int(np.argmax(vel_losses_np))
                best_timestep_velocity = int(self.validation_timesteps[best_idx_vel])
                worst_timestep_velocity = int(self.validation_timesteps[worst_idx_vel])
                best_velocity_loss = float(vel_losses_np[best_idx_vel])
                worst_velocity_loss = float(vel_losses_np[worst_idx_vel])
            if len(all_timestep_avg_direct_noise_losses) > 0:
                dir_losses_np = np.array(all_timestep_avg_direct_noise_losses)
                best_idx_dir = int(np.argmin(dir_losses_np))
                worst_idx_dir = int(np.argmax(dir_losses_np))
                best_timestep_direct = int(self.validation_timesteps[best_idx_dir])
                worst_timestep_direct = int(self.validation_timesteps[worst_idx_dir])
                best_direct_loss = float(dir_losses_np[best_idx_dir])
                worst_direct_loss = float(dir_losses_np[worst_idx_dir])

            # Correlation with raw timestep values
            velocity_loss_t_corr = 0.0
            noise_loss_t_corr = 0.0
            if num_timesteps > 1:
                ts_array = np.array(self.validation_timesteps, dtype=np.float32)
                try:
                    corr_vel_t = np.corrcoef(
                        ts_array, np.array(all_timestep_avg_velocity_losses)
                    )
                    if not np.isnan(corr_vel_t[0, 1]):
                        velocity_loss_t_corr = float(corr_vel_t[0, 1])
                except Exception:
                    pass
                try:
                    corr_noise_t = np.corrcoef(
                        ts_array, np.array(all_timestep_avg_direct_noise_losses)
                    )
                    if not np.isnan(corr_noise_t[0, 1]):
                        noise_loss_t_corr = float(corr_noise_t[0, 1])
                except Exception:
                    pass

            # Add evaluation throughput metrics if enabled
            eval_throughput_metrics = {}
            eval_runtime = 0.0
            if getattr(args, "log_throughput_metrics", True):
                eval_throughput_metrics = get_throughput_metrics()
                eval_runtime = get_total_runtime()

            # Compute SNR distribution stats and slope-style sensitivities for snr/ namespace
            snr_mean = snr_std = snr_min = snr_max = snr_p50 = snr_p90 = 0.0
            _snr_std_val = 0.0
            try:
                snr_tensor = torch.tensor(all_timestep_snrs)
                if snr_tensor.numel() > 0:
                    snr_mean = float(snr_tensor.mean().item())
                    _snr_std_val = float(snr_tensor.std(unbiased=False).item())
                    snr_std = _snr_std_val
                    snr_min = float(snr_tensor.min().item())
                    snr_max = float(snr_tensor.max().item())
                    try:
                        snr_p50 = float(torch.quantile(snr_tensor, 0.5).item())
                        snr_p90 = float(torch.quantile(snr_tensor, 0.9).item())
                    except Exception:
                        snr_p50 = float(torch.median(snr_tensor).item())
                        k = max(1, int(0.1 * snr_tensor.numel()))
                        snr_p90 = float(snr_tensor.topk(k).values.min().item())
            except Exception:
                pass

            vel_slope = 0.0
            noise_slope = 0.0
            if _snr_std_val > 0.0:
                if velocity_loss_std > 0:
                    vel_slope = float(
                        velocity_loss_snr_corr * (velocity_loss_std / _snr_std_val)
                    )
                if direct_noise_loss_std > 0:
                    noise_slope = float(
                        noise_loss_snr_corr * (direct_noise_loss_std / _snr_std_val)
                    )

            # Split SNR metrics into essential snr/ and others snr_other/ if enabled
            snr_essential = {}
            snr_other = {}
            if getattr(args, "snr_split_namespaces", False):
                # Keep only 4-5 essentials in snr/
                snr_essential["snr/val/mean"] = snr_mean
                snr_essential["snr/val/std"] = snr_std
                snr_essential["snr/val/p90"] = snr_p90
                snr_essential["snr/val/velocity_loss_correlation"] = (
                    velocity_loss_snr_corr
                )
                snr_essential["snr/val/direct_noise_loss_correlation"] = (
                    noise_loss_snr_corr
                )

                # Route remaining SNR metrics to snr_other/
                snr_other.update(
                    {
                        "snr_other/val/min": snr_min,
                        "snr_other/val/max": snr_max,
                        "snr_other/val/p50": snr_p50,
                        "snr_other/val/velocity_loss_slope": vel_slope,
                        "snr_other/val/direct_noise_loss_slope": noise_slope,
                    }
                )
            else:
                snr_essential.update(
                    {
                        "snr/val/velocity_loss_correlation": velocity_loss_snr_corr,
                        "snr/val/direct_noise_loss_correlation": noise_loss_snr_corr,
                        "snr/val/mean": snr_mean,
                        "snr/val/std": snr_std,
                        "snr/val/min": snr_min,
                        "snr/val/max": snr_max,
                        "snr/val/p50": snr_p50,
                        "snr/val/p90": snr_p90,
                        "snr/val/velocity_loss_slope": vel_slope,
                        "snr/val/direct_noise_loss_slope": noise_slope,
                    }
                )

            # Split validation metrics into essential val/ and detailed val_other/ if enabled
            val_essential = {}
            val_other = {}

            if getattr(args, "val_split_namespaces", True):
                # Keep the top-line 6 charts users care about most
                val_essential.update(
                    {
                        "val/velocity_loss_avg": velocity_final_avg_loss,
                        "val/direct_noise_loss_avg": direct_noise_final_avg_loss,
                        "val/velocity_loss_std": velocity_loss_std,
                        "val/direct_noise_loss_std": direct_noise_loss_std,
                        "val/best_velocity_loss": best_velocity_loss,
                        "val/best_direct_loss": best_direct_loss,
                    }
                )

                # Route remaining diagnostics to val_other/
                val_other.update(
                    {
                        "val_other/velocity_loss_avg_weighted": velocity_weighted_avg_loss,
                        "val_other/direct_noise_loss_avg_weighted": direct_noise_weighted_avg_loss,
                        "val_other/velocity_loss_range": velocity_loss_range,
                        "val_other/direct_noise_loss_range": direct_noise_loss_range,
                        "val_other/loss_ratio": loss_ratio,
                        "val_other/velocity_loss_cv_across_timesteps": velocity_cv_across_timesteps,
                        "val_other/direct_noise_loss_cv_across_timesteps": noise_cv_across_timesteps,
                        "val_other/velocity_loss_avg_p25": vel_p25,
                        "val_other/velocity_loss_avg_p50": vel_p50,
                        "val_other/velocity_loss_avg_p75": vel_p75,
                        "val_other/direct_noise_loss_avg_p25": noi_p25,
                        "val_other/direct_noise_loss_avg_p50": noi_p50,
                        "val_other/direct_noise_loss_avg_p75": noi_p75,
                        "val_other/best_timestep_velocity": best_timestep_velocity,
                        "val_other/worst_timestep_velocity": worst_timestep_velocity,
                        "val_other/worst_velocity_loss": worst_velocity_loss,
                        "val_other/best_timestep_direct": best_timestep_direct,
                        "val_other/worst_timestep_direct": worst_timestep_direct,
                        "val_other/worst_direct_loss": worst_direct_loss,
                        "val_other/velocity_loss_timestep_correlation": velocity_loss_t_corr,
                        "val_other/noise_loss_timestep_correlation": noise_loss_t_corr,
                    }
                )
            else:
                # Original behavior: keep everything under val/
                val_essential.update(
                    {
                        "val/velocity_loss_avg": velocity_final_avg_loss,
                        "val/direct_noise_loss_avg": direct_noise_final_avg_loss,
                        "val/velocity_loss_avg_weighted": velocity_weighted_avg_loss,
                        "val/direct_noise_loss_avg_weighted": direct_noise_weighted_avg_loss,
                        "val/velocity_loss_std": velocity_loss_std,
                        "val/direct_noise_loss_std": direct_noise_loss_std,
                        "val/velocity_loss_range": velocity_loss_range,
                        "val/direct_noise_loss_range": direct_noise_loss_range,
                        "val/loss_ratio": loss_ratio,
                        "val/velocity_loss_cv_across_timesteps": velocity_cv_across_timesteps,
                        "val/direct_noise_loss_cv_across_timesteps": noise_cv_across_timesteps,
                        "val/velocity_loss_avg_p25": vel_p25,
                        "val/velocity_loss_avg_p50": vel_p50,
                        "val/velocity_loss_avg_p75": vel_p75,
                        "val/direct_noise_loss_avg_p25": noi_p25,
                        "val/direct_noise_loss_avg_p50": noi_p50,
                        "val/direct_noise_loss_avg_p75": noi_p75,
                        "val/best_timestep_velocity": best_timestep_velocity,
                        "val/worst_timestep_velocity": worst_timestep_velocity,
                        "val/best_velocity_loss": best_velocity_loss,
                        "val/worst_velocity_loss": worst_velocity_loss,
                        "val/best_timestep_direct": best_timestep_direct,
                        "val/worst_timestep_direct": worst_timestep_direct,
                        "val/best_direct_loss": best_direct_loss,
                        "val/worst_direct_loss": worst_direct_loss,
                        "val/velocity_loss_timestep_correlation": velocity_loss_t_corr,
                        "val/noise_loss_timestep_correlation": noise_loss_t_corr,
                    }
                )

            # Merge and log
            merged_logs = {}
            merged_logs.update(val_essential)
            merged_logs.update(val_other)
            merged_logs.update(snr_essential)
            merged_logs.update(snr_other)

            accelerator.log(merged_logs, step=global_step)

        # Switch back to train mode
        unwrapped_model.train()
        unwrapped_model.switch_block_swap_for_training()

        logger.info(
            f"Validation finished. Average velocity loss: {velocity_final_avg_loss:.5f}, Average direct noise loss: {direct_noise_final_avg_loss:.5f}"
        )
        return velocity_final_avg_loss, direct_noise_final_avg_loss

    def sync_validation_epoch(
        self,
        val_dataloader: Any,
        val_epoch_step_sync: Optional[tuple[Any, Any]],
        current_epoch_value: int,
        current_step_value: int,
    ) -> None:
        """Synchronize validation dataset epochs before validation runs."""
        if val_epoch_step_sync is None:
            return

        val_current_epoch, val_current_step = val_epoch_step_sync

        # Update validation epoch/step tracking to match training
        val_current_epoch.value = current_epoch_value
        val_current_step.value = current_step_value

        # Force update validation datasets to prevent unwanted shuffling
        if hasattr(val_dataloader, "dataset"):
            dataset = val_dataloader.dataset
            if hasattr(dataset, "set_current_epoch"):
                dataset.set_current_epoch(
                    current_epoch_value,
                    force_shuffle=False,  # Validation should never shuffle
                    reason="validation_sync",
                )

    def should_validate(
        self,
        args: argparse.Namespace,
        global_step: int,
        val_dataloader: Optional[Any],
        last_validated_step: int,
    ) -> bool:
        """Check if validation should be performed at the current step."""
        if val_dataloader is None:
            return False

        # Check if validation already occurred at this step
        if last_validated_step == global_step:
            return False

        # Check if step-based validation is enabled
        if (
            args.validate_every_n_steps is not None
            and global_step % args.validate_every_n_steps == 0
        ):
            return True

        return False

    def log_validation_results(
        self,
        accelerator: Accelerator,
        val_loss: tuple[float, float],
        global_step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log validation results to console only.

        Note: Tensorboard logging is now handled within the validate() method
        to avoid redundancy and provide more detailed per-timestep metrics.
        """
        velocity_loss, direct_noise_loss = val_loss

        if epoch is not None:
            accelerator.print(
                f"[Epoch {epoch}] velocity_loss={velocity_loss:0.5f}, direct_noise_loss={direct_noise_loss:0.5f}"
            )
        else:
            accelerator.print(
                f"[Step {global_step}] velocity_loss={velocity_loss:0.5f}, direct_noise_loss={direct_noise_loss:0.5f}"
            )

    def validate_with_progress(
        self,
        accelerator: Accelerator,
        transformer: Any,
        val_dataloader: Any,
        noise_scheduler: FlowMatchDiscreteScheduler,
        args: argparse.Namespace,
        control_signal_processor: Optional[Any] = None,
        vae: Optional[Any] = None,
        global_step: Optional[int] = None,
        show_progress: bool = True,
    ) -> tuple[float, float]:
        """Run validation with optional progress bar.

        Args:
            global_step: Current training step for logging per-timestep metrics
            show_progress: Whether to show progress bars
        """
        return self.validate(
            accelerator,
            transformer,
            val_dataloader,
            noise_scheduler,
            args,
            control_signal_processor,
            vae,
            global_step,
            show_progress,
        )
