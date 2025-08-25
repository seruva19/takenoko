## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/hv_train_network.py (Apache)
## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/wan_train_network.py (Apache)

"""Core training logic for WAN network trainer.

This module handles the main training loop, model calling, loss computation, and validation.
Extracted from wan_network_trainer.py to improve code organization and maintainability.
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

import utils.fluxflow_augmentation as fluxflow_augmentation
import scheduling.fvdm as fvdm

from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import (
    compute_loss_weighting_for_sd3,
    get_sigmas,
    clean_memory_on_device,
    should_sample_images,
    LossRecorder,
)
from scheduling.timestep_distribution import (
    TimestepDistribution,
    should_use_precomputed_timesteps,
)
from scheduling.timestep_logging import (
    log_initial_timestep_distribution,
    log_live_timestep_distribution,
    log_loss_scatterplot,
    log_show_timesteps_figure_unconditional,
)
from scheduling.timestep_utils import (
    get_noisy_model_input_and_timesteps,
    initialize_timestep_distribution,
    time_shift,
    get_lin_function,
)
from core.validation_core import ValidationCore
from criteria.dispersive_loss import dispersive_loss_info_nce
from criteria.training_loss import TrainingLossComputer, LossComponents


from scheduling.fopp import (
    FoPPScheduler,
    get_alpha_bar_schedule,
    apply_asynchronous_noise,
)

# Enhanced optimizer logging
from optimizers.enhanced_logging import (
    get_enhanced_metrics,
    get_histogram_data,
    is_supported,
)

# Performance logging
from common.performance_logger import (
    start_step_timing,
    start_forward_pass_timing,
    end_forward_pass_timing,
    start_backward_pass_timing,
    end_backward_pass_timing,
    start_optimizer_step_timing,
    end_optimizer_step_timing,
    end_step_timing,
    get_timing_metrics,
    get_model_statistics,
    get_hardware_metrics,
    log_performance_summary,
    configure_verbosity,
)

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TrainingCore:
    """Handles core training logic, model calling, and validation."""

    def __init__(self, config: Any, fluxflow_config: Dict[str, Any]):
        self.config = config
        self.fluxflow_config = fluxflow_config

        # Initialize validation core
        self.validation_core = ValidationCore(config, fluxflow_config)

        # EMA loss tracking for smoother TensorBoard visualization
        self.ema_loss: Optional[float] = None
        self.ema_beta: float = 0.98  # Smoothing factor (0.9-0.99 are good choices)
        self.ema_step_count: int = 0
        self.ema_bias_warmup_steps: int = (
            100  # defer bias correction for early readability
        )

        # Parameter statistics tracking
        self.last_param_log_step: int = -1

        # Per-source loss tracking
        self.per_source_losses: Dict[str, List[float]] = {}

        # Pre-computed timestep distribution (initialized when needed)
        self.timestep_distribution = TimestepDistribution()
        self._timestep_logging_initialized: bool = False

        # Gradient norm tracking
        self.gradient_norms: List[float] = []

        # TensorBoardX enhanced logging
        self.tensorboardx_writer = None
        self.use_tensorboardx = False

        # Centralized loss computation
        self.loss_computer = TrainingLossComputer(self.config)

    def set_ema_beta(self, beta: float) -> None:
        """Set the EMA smoothing factor. Higher values (closer to 1.0) = more smoothing."""
        if not 0.0 < beta < 1.0:
            raise ValueError("EMA beta must be between 0.0 and 1.0")
        self.ema_beta = beta
        logger.info(f"EMA beta set to {beta}")

    def configure_ema_from_args(self, args: argparse.Namespace) -> None:
        """Configure EMA hyperparameters from args if provided."""
        try:
            if hasattr(args, "ema_loss_beta"):
                self.set_ema_beta(float(args.ema_loss_beta))
            if hasattr(args, "ema_loss_bias_warmup_steps"):
                self.ema_bias_warmup_steps = int(args.ema_loss_bias_warmup_steps)
        except Exception:
            pass

    def configure_advanced_logging(self, args: argparse.Namespace) -> None:
        """Configure advanced logging settings including parameter stats, per-source losses, and gradient norms.

        Sets default values for various logging options if not already set in args.
        """
        # Performance logging verbosity
        if not hasattr(args, "performance_verbosity"):
            args.performance_verbosity = "standard"  # Default verbosity level

        # Configure performance logger verbosity
        configure_verbosity(args.performance_verbosity)
        logger.info(
            f"Performance logging verbosity set to: {args.performance_verbosity}"
        )

        # Parameter statistics logging
        if not hasattr(args, "log_param_stats"):
            args.log_param_stats = False  # Disabled by default
        if not hasattr(args, "param_stats_every_n_steps"):
            args.param_stats_every_n_steps = 100  # Log every 100 steps
        if not hasattr(args, "max_param_stats_logged"):
            args.max_param_stats_logged = 20  # Log top 20 parameters by norm

        # Per-source loss logging
        if not hasattr(args, "log_per_source_loss"):
            args.log_per_source_loss = False  # Disabled by default

        # Gradient norm logging
        if not hasattr(args, "log_gradient_norm"):
            args.log_gradient_norm = False  # Disabled by default

        # Extra train metrics (periodic)
        if not hasattr(args, "log_extra_train_metrics"):
            args.log_extra_train_metrics = True  # Enabled by default
        if not hasattr(args, "train_metrics_interval"):
            args.train_metrics_interval = 50  # Log every 50 steps by default

        # Report enabled features
        enabled_features = []

        if args.log_param_stats:
            enabled_features.append("Parameter Statistics")
            logger.info(f"Parameter statistics logging enabled:")
            logger.info(f"  - Logging every {args.param_stats_every_n_steps} steps")
            logger.info(
                f"  - Tracking top {args.max_param_stats_logged} parameters by norm"
            )
            logger.info(
                f"  - Will create TensorBoard metrics: param_norm/*, grad_norm/*, param_stats/*"
            )

        if args.log_per_source_loss:
            enabled_features.append("Per-Source Loss")
            logger.info("Per-source loss logging enabled:")
            logger.info("  - Will attempt to detect video vs image sources")
            logger.info("  - Will create TensorBoard metrics: loss/video, loss/image")

        if args.log_gradient_norm:
            enabled_features.append("Gradient Norm")
            logger.info("Gradient norm logging enabled:")
            logger.info("  - Will create TensorBoard metric: grad_norm")

        if enabled_features:
            logger.info(
                f"Advanced logging features enabled: {', '.join(enabled_features)}"
            )
        else:
            logger.info(
                "No advanced logging features enabled (use configure_advanced_logging to enable)"
            )

    def update_ema_loss(self, current_loss: float) -> float:
        """Update EMA loss; warm-start and defer bias correction for early steps."""
        self.ema_step_count += 1

        if self.ema_loss is None:
            # Warm-start from current loss to avoid huge initial spike
            self.ema_loss = float(current_loss)
        else:
            self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * float(
                current_loss
            )

        # Defer bias correction for a short warmup window to improve readability
        if self.ema_step_count <= max(0, int(self.ema_bias_warmup_steps)):
            return float(self.ema_loss)

        corrected_ema = self.ema_loss / (1 - self.ema_beta**self.ema_step_count)
        return float(corrected_ema)

    # Metric helpers moved to trainer.metrics
    # kept methods as thin wrappers for backward compatibility
    def generate_parameter_stats(
        self,
        model: Any,
        global_step: int,
        log_every_n_steps: int = 100,
        max_params_to_log: int = 20,
    ) -> Dict[str, float]:
        from core.metrics import generate_parameter_stats as _gps

        # gate by local last_param_log_step to keep same behavior
        if global_step - self.last_param_log_step < log_every_n_steps:
            return {}
        self.last_param_log_step = global_step
        return _gps(model, global_step, log_every_n_steps, max_params_to_log)

    def compute_per_source_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        batch: Dict[str, Any],
        weighting: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        from core.metrics import compute_per_source_loss as _cpsl

        return _cpsl(model_pred, target, batch, weighting, sample_weights)

    def compute_gradient_norm(
        self, model: Any, max_norm: Optional[float] = None, norm_type: float = 2.0
    ) -> float:
        from core.metrics import compute_gradient_norm as _cgn

        return _cgn(model, max_norm, norm_type)

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss: float,
        avr_loss: float,
        lr_scheduler: Any,
        lr_descriptions: List[str],
        optimizer: Optional[torch.optim.Optimizer] = None,
        keys_scaled: Optional[int] = None,
        mean_norm: Optional[float] = None,
        maximum_norm: Optional[float] = None,
        ema_loss: Optional[float] = None,
        model: Optional[Any] = None,
        global_step: Optional[int] = None,
        per_source_losses: Optional[Dict[str, float]] = None,
        gradient_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        from core.metrics import generate_step_logs as _gsl

        # Include max-norm details if available
        logs = _gsl(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            ema_loss,
            model,
            global_step,
            per_source_losses,
            gradient_norm,
        )
        return logs

    def record_training_step(self, batch_size: int) -> None:
        """Record a training step for throughput tracking."""
        from core.metrics import record_training_step as _rts

        _rts(batch_size)

    def initialize_throughput_tracker(self, args: Any) -> None:
        """Initialize throughput tracker with configuration."""
        from core.metrics import initialize_throughput_tracker as _init_tracker

        window_size = getattr(args, "throughput_window_size", 100)
        _init_tracker(window_size)

    def generate_safe_progress_metrics(
        self,
        args: argparse.Namespace,
        current_loss: float,
        avr_loss: float,
        lr_scheduler: Any,
        epoch: int,
        global_step: int,
        keys_scaled: Optional[int] = None,
        mean_norm: Optional[float] = None,
        maximum_norm: Optional[float] = None,
        current_step_in_epoch: Optional[int] = None,
        total_steps_in_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        from core.metrics import generate_safe_progress_metrics as _gspm

        return _gspm(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            epoch,
            global_step,
            keys_scaled,
            mean_norm,
            maximum_norm,
            current_step_in_epoch,
            total_steps_in_epoch,
        )

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale and shift latents if needed."""
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: Any,
        latents: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        control_signal_processor: Optional[Any] = None,
        controlnet: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Call the DiT model and compute target for loss calculation."""
        model = transformer

        current_logging_level = getattr(args, "logging_level", "INFO").upper()
        perturbed_latents_for_target = latents.clone()  # Start with a clone
        if self.fluxflow_config.get("enable_fluxflow", False):
            frame_dim = self.fluxflow_config.get("frame_dim_in_batch", 2)
            if latents.ndim > frame_dim and latents.shape[frame_dim] > 1:
                # Always log FluxFlow application at INFO level so user can see it
                if accelerator.is_main_process:
                    logger.info(
                        f"FLUXFLOW: Applying temporal augmentation - mode={self.fluxflow_config.get('mode','frame')}, "
                        f"latents shape={latents.shape}, frame_dim={frame_dim}, "
                        f"num_frames={latents.shape[frame_dim]}"
                    )

                perturbed_latents_for_target = (
                    fluxflow_augmentation.apply_fluxflow_to_batch(
                        perturbed_latents_for_target, self.fluxflow_config
                    )
                )

                # Log if perturbation was successful
                if accelerator.is_main_process:
                    perturbation_applied = not torch.equal(
                        latents, perturbed_latents_for_target
                    )
                    logger.info(
                        f"FLUXFLOW: Temporal perturbation applied: {perturbation_applied}"
                    )

            else:
                # Always log why FluxFlow is being skipped
                if accelerator.is_main_process:
                    logger.info(
                        f"FLUXFLOW: Skipping temporal augmentation - latents shape={latents.shape}, "
                        f"frame_dim={frame_dim}, num_frames={latents.shape[frame_dim] if latents.ndim > frame_dim else 'N/A'}, "
                        f"reason={'not enough dimensions' if latents.ndim <= frame_dim else 'only 1 frame'}"
                    )

        # I2V training and Control training
        image_latents = None
        clip_fea = None

        context = [
            t.to(device=accelerator.device, dtype=network_dtype) for t in batch["t5"]
        ]

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in context:
                t.requires_grad_(True)

        # Control LoRA processing (aligned with reference implementation)
        control_latents = None
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            # Log that control LoRA is engaged
            logger.info("🎯 Control LoRA processing engaged in training loop")

            # Pass VAE to control signal processing (must be provided by training loop)
            vae = (
                getattr(control_signal_processor, "vae", None)
                if control_signal_processor
                else None
            )
            if vae is None:
                logger.error(
                    "VAE not available for control LoRA training - this will cause training to fail"
                )

            control_latents = (
                control_signal_processor.process_control_signal(
                    args, accelerator, batch, latents, network_dtype, vae
                )
                if control_signal_processor
                else None
            )

            # If control signal could not be generated, fall back to using the image
            # latents themselves (same behaviour as V1 implementation).
            if control_latents is None:
                logger.warning(
                    "No control signal or pixels found; using image latents as control signal fallback"
                )
                control_latents = latents.detach().clone()

        # Optional: run ControlNet to get per-layer control states
        control_states = None
        try:
            if (
                hasattr(args, "enable_controlnet")
                and args.enable_controlnet
                and controlnet is not None
            ):
                # Control hint: prefer explicit control_signal (CFHW) -> BCFHW, else None
                control_pixels = None
                if (
                    "pixels" in batch
                    and isinstance(batch["pixels"], list)
                    and len(batch["pixels"]) > 0
                ):
                    # Pixels list entries are CFHW in our dataset; stack to BCFHW then move C back
                    cfhw_list = [
                        p.to(device=accelerator.device, dtype=network_dtype)
                        for p in batch["pixels"]
                    ]
                    control_pixels = torch.stack(cfhw_list, dim=0)  # B, C, F, H, W
                elif "control_signal" in batch:
                    # Already batched (B, C, F, H, W)
                    control_pixels = batch["control_signal"].to(
                        device=accelerator.device, dtype=network_dtype
                    )

                if control_pixels is not None:
                    control_states_tuple = controlnet(
                        hidden_states=noisy_model_input.to(
                            device=accelerator.device, dtype=network_dtype
                        ),
                        timestep=timesteps.to(device=accelerator.device),
                        encoder_hidden_states=(
                            torch.stack(context, dim=0)
                            if isinstance(context, list)
                            else context
                        ),
                        controlnet_states=control_pixels,
                        return_dict=False,
                    )
                    # controlnet returns ((layer_states,...),)
                    if (
                        isinstance(control_states_tuple, tuple)
                        and len(control_states_tuple) > 0
                    ):
                        control_states = control_states_tuple[0]
        except Exception as e:
            logger.warning(
                f"ControlNet forward failed; continuing without control. Error: {e}"
            )

        # call DiT
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
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(
            device=accelerator.device, dtype=network_dtype
        )

        # Prepare model input with control signal if control LoRA is enabled
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            if control_latents is not None:
                control_latents = control_latents.to(
                    device=noisy_model_input.device, dtype=noisy_model_input.dtype
                )

                # Honor control_concatenation_dim when valid; default to channel dim
                concat_dim = getattr(args, "control_concatenation_dim", None)
                if concat_dim is None:
                    concat_dim = 1 if noisy_model_input.dim() == 5 else 0
                else:
                    if noisy_model_input.dim() == 5 and concat_dim in (0, -2):
                        concat_dim = 1
                    else:
                        # Normalize negative dims and clamp
                        ndim = noisy_model_input.dim()
                        if not isinstance(concat_dim, int) or not (
                            -ndim <= concat_dim < ndim
                        ):
                            concat_dim = 1 if ndim == 5 else 0
                        else:
                            concat_dim = concat_dim % ndim

                model_input = torch.cat(
                    [noisy_model_input, control_latents], dim=concat_dim
                )
            else:
                logger.error(
                    "Control LoRA is enabled but control_latents is None. This will likely fail."
                )
                zero_control = torch.zeros_like(noisy_model_input)
                concat_dim = 1 if noisy_model_input.dim() == 5 else 0
                model_input = torch.cat(
                    [noisy_model_input, zero_control], dim=concat_dim
                )
        else:
            model_input = noisy_model_input

        with accelerator.autocast():
            # Build force_keep_mask for TREAD masked training (preserve masked tokens)
            force_keep_mask = None
            try:
                if (
                    getattr(args, "enable_tread", False)
                    and getattr(args, "tread_config", None) is not None
                    and "mask_signal" in batch
                    and batch["mask_signal"] is not None
                ):
                    # mask_signal shape could be (B, T, H, W) or (B, 1, T, H, W) or similar
                    ms = batch["mask_signal"]
                    if ms.dim() == 4:
                        # (B, T, H, W) -> (B, 1, T, H, W)
                        ms = ms.unsqueeze(1)
                    # Normalize to [0,1] if in [-1,1]
                    if ms.min() < 0 or ms.max() > 1:
                        ms = (ms + 1) / 2

                    # Compute token grid sizes from latent shape and patch size
                    lat_f, lat_h, lat_w = latents.shape[2:5]
                    pt, ph, pw = self.config.patch_size
                    t_tokens = max(1, lat_f // pt)
                    h_tokens = max(1, lat_h // ph)
                    w_tokens = max(1, lat_w // pw)

                    # Downsample to token grid
                    ms_tok = F.interpolate(
                        ms.float(),
                        size=(t_tokens, h_tokens, w_tokens),
                        mode="trilinear",
                        align_corners=False,
                    )
                    # Flatten with same T->H->W order as patch embedding
                    force_keep_mask = ms_tok.squeeze(1).flatten(1) > 0.5
            except Exception:
                force_keep_mask = None

            model_pred = model(
                model_input,
                t=timesteps,
                context=context,
                clip_fea=clip_fea,
                seq_len=seq_len,
                y=image_latents,
                force_keep_mask=force_keep_mask,
                controlnet_states=control_states,
                controlnet_weight=getattr(args, "controlnet_weight", 1.0),
                controlnet_stride=int(getattr(args, "controlnet_stride", 1)),
                dispersive_loss_target_block=getattr(
                    args, "dispersive_loss_target_block", None
                ),
                return_intermediate=bool(
                    getattr(args, "enable_dispersive_loss", False)
                ),
            )
        # Unpack optional intermediate
        intermediate_z: Optional[torch.Tensor] = None
        if isinstance(model_pred, tuple) and len(model_pred) == 2:
            model_pred, intermediate_z = model_pred  # type: ignore
        model_pred = torch.stack(model_pred, dim=0)  # list to tensor

        if model_pred.grad_fn is None:
            print(
                "model_pred is detached from the graph before returning from call_dit"
            )

        # flow matching loss - compute target using perturbed latents if fluxflow is enabled
        target = noise - perturbed_latents_for_target.to(
            device=accelerator.device, dtype=network_dtype
        )

        return model_pred, target, intermediate_z

    def run_training_loop(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: Any,
        network: Any,
        training_model: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        lr_descriptions: List[str],
        train_dataloader: Any,
        val_dataloader: Optional[Any],
        noise_scheduler: FlowMatchDiscreteScheduler,
        network_dtype: torch.dtype,
        dit_dtype: torch.dtype,
        num_train_epochs: int,
        global_step: int,
        progress_bar: tqdm,
        metadata: Dict[str, str],
        loss_recorder: LossRecorder,
        sampling_manager: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        control_signal_processor: Optional[Any] = None,
        current_epoch: Optional[Any] = None,
        current_step: Optional[Any] = None,
        optimizer_train_fn: Optional[Callable] = None,
        optimizer_eval_fn: Optional[Callable] = None,
        vae: Optional[Any] = None,
        sample_parameters: Optional[Any] = None,
        save_model: Optional[Callable] = None,
        remove_model: Optional[Callable] = None,
        is_main_process: bool = False,
        val_epoch_step_sync: Optional[Tuple[Any, Any]] = None,
        repa_helper: Optional[Any] = None,
        controlnet: Optional[Any] = None,
        dual_model_manager: Optional[Any] = None,
    ) -> Tuple[int, Any]:
        # Configure EMA hyperparameters from args (non-fatal)
        try:
            self.configure_ema_from_args(args)
        except Exception:
            pass
        """Run the main training loop."""

        # Configure attention metrics from args (non-fatal)
        try:
            from common import attention_metrics as _attn_metrics

            _attn_metrics.configure_from_args(args)
        except Exception:
            pass

        # Calculate starting epoch when resuming from checkpoint
        if global_step > 0:
            # We're resuming from a checkpoint - calculate which epoch we should be in
            steps_per_epoch = len(train_dataloader)
            epoch_to_start = global_step // steps_per_epoch

            # If we're exactly at the end of an epoch, we should start the next epoch
            if global_step % steps_per_epoch == 0 and global_step > 0:
                # We completed the previous epoch, start the next one
                pass  # epoch_to_start is already correct

            logger.info(
                f"Resuming training from step {global_step}, starting at epoch {epoch_to_start + 1}"
            )
        else:
            epoch_to_start = 0

        # Track the last step where sampling occurred to prevent duplicates
        # When resuming from a checkpoint, initialize based on whether sampling would have occurred at the current step
        if global_step > 0:
            # We're resuming from a checkpoint
            # Check if sampling would occur at the current global_step
            if should_sample_images(args, global_step, epoch=None):
                # Sampling would have occurred at this step before checkpoint was saved
                last_sampled_step = global_step
            else:
                last_sampled_step = -1

            # Check if validation would occur at the current global_step
            if (
                args.validate_every_n_steps is not None
                and global_step % args.validate_every_n_steps == 0
            ):
                last_validated_step = global_step
            else:
                last_validated_step = -1
        else:
            # Fresh training start
            last_sampled_step = -1
            last_validated_step = -1

        # Initialize throughput tracker with configuration
        self.initialize_throughput_tracker(args)

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            if current_epoch is not None:
                current_epoch.value = epoch + 1

            metadata["takenoko_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)

            # Calculate step offset when resuming in the middle of an epoch
            step_offset = 0
            if global_step > 0 and epoch == epoch_to_start:
                steps_per_epoch = len(train_dataloader)
                step_offset = global_step % steps_per_epoch
                if step_offset > 0:
                    logger.info(
                        f"Skipping first {step_offset} batches in epoch {epoch + 1} (resuming from middle of epoch)"
                    )

            for step, batch in enumerate(train_dataloader):
                # Start performance timing for this step
                start_step_timing()

                # Begin gated attention-metrics window (no-op if disabled)
                try:
                    from common import attention_metrics as _attn_metrics

                    _attn_metrics.begin_step(global_step)
                except Exception:
                    pass

                # Skip batches when resuming in the middle of an epoch
                if epoch == epoch_to_start and step < step_offset:
                    continue
                latents = batch["latents"]
                bsz = latents.shape[0]
                if current_step is not None:
                    current_step.value = global_step

                # Initialize metrics for this step
                per_source_losses = {}
                gradient_norm = None

                with accelerator.accumulate(training_model):
                    accelerator.unwrap_model(network).on_step_start()

                    latents = self.scale_shift_latents(latents)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    if hasattr(args, "enable_fvdm") and args.enable_fvdm:
                        # This inner 'if' is ONLY for the one-time log message
                        if accelerator.is_main_process and step == 0:
                            logger.info(
                                "FVDM training enabled. Using FVDM timestep sampling."
                            )

                        # This function call happens on EVERY step when FVDM is enabled
                        noisy_model_input, timesteps, sigmas = (
                            fvdm.get_noisy_model_input_and_timesteps_fvdm(
                                args,
                                noise,
                                latents,
                                noise_scheduler,
                                accelerator.device,
                                dit_dtype,
                            )
                        )
                    else:
                        # Initialize timestep distribution if needed
                        initialize_timestep_distribution(
                            args, self.timestep_distribution
                        )

                        # Optionally log the initial expected timestep distribution
                        if (
                            accelerator.is_main_process
                            and not self._timestep_logging_initialized
                            and getattr(args, "log_timestep_distribution_init", True)
                            and args.log_with in ["tensorboard", "all"]
                        ):
                            try:
                                log_initial_timestep_distribution(
                                    accelerator, args, self.timestep_distribution
                                )
                            finally:
                                self._timestep_logging_initialized = True

                        try:
                            if accelerator.is_main_process:
                                log_show_timesteps_figure_unconditional(
                                    accelerator,
                                    args,
                                    self.timestep_distribution,
                                    noise_scheduler,
                                )
                        except Exception:
                            pass

                        # calculate model input and timesteps
                        if dual_model_manager is not None:
                            noisy_model_input, timesteps, sigmas = (
                                dual_model_manager.determine_and_prepare_batch(
                                    args=args,
                                    noise=noise,
                                    latents=latents,
                                    noise_scheduler=noise_scheduler,
                                    device=accelerator.device,
                                    dtype=dit_dtype,
                                    timestep_distribution=self.timestep_distribution,
                                    presampled_uniform=(
                                        None
                                        if (
                                            hasattr(args, "use_precomputed_timesteps")
                                            and getattr(
                                                args, "use_precomputed_timesteps", False
                                            )
                                        )
                                        else batch.get("timesteps", None)
                                    ),
                                )
                            )
                            # swap base weights if regime changed
                            try:
                                dual_model_manager.swap_if_needed(accelerator)
                            except Exception as _swap_err:
                                logger.warning(
                                    f"DualModelManager swap failed: {_swap_err}"
                                )
                                # proceed without swap to avoid breaking training
                        else:
                            # If dataset provided per-batch pre-sampled uniform t values and
                            # precomputed timesteps are NOT enabled, map them through the
                            # selected sampling strategy. Otherwise, ignore and use current path.
                            batch_timesteps_uniform = None
                            try:
                                if hasattr(
                                    args, "use_precomputed_timesteps"
                                ) and getattr(args, "use_precomputed_timesteps", False):
                                    batch_timesteps_uniform = None
                                else:
                                    bt = batch.get("timesteps", None)
                                    if bt is not None:
                                        # bt may be list[float] length B
                                        batch_timesteps_uniform = torch.tensor(
                                            bt,
                                            device=accelerator.device,
                                            dtype=torch.float32,
                                        )
                            except Exception:
                                batch_timesteps_uniform = None

                            # Centralized utility with optional presampled uniform
                            noisy_model_input, timesteps, sigmas = (
                                get_noisy_model_input_and_timesteps(
                                    args,
                                    noise,
                                    latents,
                                    noise_scheduler,
                                    accelerator.device,
                                    dit_dtype,
                                    self.timestep_distribution,
                                    batch_timesteps_uniform,
                                )
                            )

                    # Optional: If the network supports TLora-style masking, update mask from timesteps
                    try:
                        unwrapped_net = accelerator.unwrap_model(network)
                        if hasattr(unwrapped_net, "update_rank_mask_from_timesteps"):
                            unwrapped_net.update_rank_mask_from_timesteps(
                                timesteps, max_timestep=1000, device=accelerator.device
                            )
                    except Exception:
                        pass

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme,
                        noise_scheduler,
                        timesteps,
                        accelerator.device,
                        dit_dtype,
                    )

                    # Start forward pass timing
                    start_forward_pass_timing()

                    # choose active transformer when dual mode is enabled
                    active_transformer = transformer
                    if dual_model_manager is not None:
                        active_transformer = dual_model_manager.active_model

                    model_result = self.call_dit(
                        args,
                        accelerator,
                        active_transformer,
                        latents,
                        batch,
                        noise,
                        noisy_model_input,
                        timesteps,
                        network_dtype,
                        control_signal_processor,
                        controlnet,
                    )

                    # End forward pass timing
                    end_forward_pass_timing()

                    # Handle case where control LoRA failed to process
                    if model_result is None or model_result[0] is None:
                        logger.warning(
                            "Skipping batch due to control LoRA processing failure"
                        )
                        continue

                    model_pred, target, intermediate_z = model_result

                    # Loss will be computed later by centralized loss computer

                    # Compute per-source losses if enabled (before backprop), separate from loss
                    if (
                        hasattr(args, "log_per_source_loss")
                        and args.log_per_source_loss
                    ):
                        try:
                            sample_weights_local = batch.get("weight", None)
                            if sample_weights_local is not None:
                                sample_weights_local = sample_weights_local.to(
                                    device=accelerator.device, dtype=network_dtype
                                )
                            per_source_losses = self.compute_per_source_loss(
                                model_pred.to(network_dtype),
                                target,
                                batch,
                                weighting,
                                sample_weights_local,
                            )
                        except Exception as e:
                            logger.debug(f"⚠️  Failed to compute per-source losses: {e}")
                            per_source_losses = {}

                    # Centralized training loss computation (includes DOP/REPA/Dispersive/OpticalFlow)
                    loss_components = self.loss_computer.compute_training_loss(
                        args=args,
                        accelerator=accelerator,
                        latents=latents,
                        noise=noise,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        network_dtype=network_dtype,
                        model_pred=model_pred,
                        target=target,
                        weighting=weighting,
                        batch=batch,
                        intermediate_z=intermediate_z,
                        vae=vae,
                        transformer=active_transformer,
                        network=network,
                        control_signal_processor=control_signal_processor,
                        repa_helper=repa_helper,
                        raft=getattr(self, "raft", None),
                        warp_fn=getattr(self, "warp", None),
                    )

                    # Start backward pass timing
                    start_backward_pass_timing()
                    accelerator.backward(loss_components.total_loss)

                    # End backward pass timing
                    end_backward_pass_timing()

                    if accelerator.is_main_process:
                        # Check if ANY trainable parameter has a gradient
                        has_grad = any(
                            p.grad is not None
                            for p in network.parameters()
                            if p.requires_grad
                        )
                        if not has_grad:
                            print(
                                "WARNING: No gradients were computed for any parameter. The computation graph is broken."
                            )
                            # raise RuntimeError("Computation graph broken: no gradients found.")

                    # Compute gradient norm if enabled (before clipping)
                    gradient_norm = None
                    if (
                        accelerator.sync_gradients
                        and hasattr(args, "log_gradient_norm")
                        and args.log_gradient_norm
                    ):
                        try:
                            gradient_norm = self.compute_gradient_norm(network)
                        except Exception as e:
                            logger.debug(f"⚠️ Failed to compute gradient norm: {e}")

                if accelerator.sync_gradients:
                    # sync DDP grad manually
                    state = accelerate.PartialState()
                    if state.distributed_type != accelerate.DistributedType.NO:
                        for param in network.parameters():
                            if param.grad is not None:
                                param.grad = accelerator.reduce(
                                    param.grad, reduction="mean"
                                )

                    # Update GGPO gradient norms once gradients are synchronized
                    try:
                        if hasattr(network, "update_grad_norms"):
                            accelerator.unwrap_model(network).update_grad_norms()
                    except Exception:
                        pass

                    # Optional separate clipping for ControlNet
                    try:
                        if (
                            hasattr(args, "controlnet_max_grad_norm")
                            and args.controlnet_max_grad_norm is not None
                            and float(args.controlnet_max_grad_norm) > 0.0
                            and controlnet is not None
                        ):
                            accelerator.clip_grad_norm_(
                                controlnet.parameters(),
                                float(args.controlnet_max_grad_norm),
                            )
                    except Exception:
                        # Non-fatal: fall back to global clipping only
                        pass

                    if args.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(
                            network
                        ).get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    # Update adaptive LR schedulers that need gradient stats (AMP/DDP-safe point)
                    try:
                        _sched = lr_scheduler
                        if hasattr(_sched, "update_gradient_stats"):
                            _sched.update_gradient_stats()  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    # Start optimizer step timing
                    start_optimizer_step_timing()
                    try:
                        optimizer.step()
                    except RuntimeError as e:
                        logger.error(
                            "🚨 Make sure you're not adding empty parameter groups to the optimizer"
                        )
                        raise e
                    # End optimizer step timing
                    end_optimizer_step_timing()

                    # Update GGPO weight norms after parameter update
                    try:
                        if hasattr(network, "update_norms"):
                            accelerator.unwrap_model(network).update_norms()
                    except Exception:
                        pass

                    # Update adaptive LR schedulers with loss trend when available
                    try:
                        _sched = lr_scheduler
                        if hasattr(_sched, "update_training_stats"):
                            _loss_val = float(
                                loss_components.total_loss.detach().item()
                            )
                            _sched.update_training_stats(_loss_val)  # type: ignore[attr-defined]
                        elif hasattr(_sched, "update_metrics"):
                            _loss_val = float(
                                loss_components.total_loss.detach().item()
                            )
                            _sched.update_metrics(_loss_val)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(
                        network
                    ).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {
                        "Keys Scaled": keys_scaled,
                        "Average key norm": mean_norm,
                    }
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Lightweight periodic self-correction cache refresh (fully gated)
                    try:
                        if (
                            bool(getattr(args, "self_correction_enabled", False))
                            and global_step
                            > int(getattr(args, "self_correction_warmup_steps", 1000))
                            and int(
                                getattr(args, "self_correction_update_frequency", 1000)
                            )
                            > 0
                            and global_step
                            % int(
                                getattr(args, "self_correction_update_frequency", 1000)
                            )
                            == 0
                        ):
                            # Access manager via trainer if present
                            mgr = getattr(
                                accelerator.state, "_self_correction_manager", None
                            )
                            # Fallback: some callers may attach it to the transformer
                            if mgr is None:
                                try:
                                    mgr = getattr(
                                        transformer, "_self_correction_manager", None
                                    )
                                except Exception:
                                    mgr = None
                            if mgr is not None:
                                # Switch to eval for generation
                                try:
                                    training_was = training_model.training
                                    training_model.eval()
                                except Exception:
                                    training_was = None
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    mgr.update_cache(accelerator.unwrap_model(transformer))  # type: ignore
                                accelerator.wait_for_everyone()
                                # Restore mode
                                try:
                                    if training_was is True:
                                        training_model.train()
                                except Exception:
                                    pass
                    except Exception as _sc_err:
                        # Non-fatal path: continue training
                        if accelerator.is_main_process:
                            logger.debug(f"Self-correction update skipped: {_sc_err}")

                    # to avoid calling optimizer_eval_fn() too frequently, we call it only when we need to sample images, validate, or save the model
                    should_sampling = should_sample_images(
                        args, global_step, epoch=epoch + 1
                    )
                    should_saving = (
                        args.save_every_n_steps is not None
                        and global_step % args.save_every_n_steps == 0
                    )
                    should_validating = self.validation_core.should_validate(
                        args, global_step, val_dataloader, last_validated_step
                    )

                    if should_sampling or should_saving or should_validating:
                        if optimizer_eval_fn:
                            optimizer_eval_fn()
                        if should_sampling and sampling_manager:
                            # Use epoch-based naming only if sampling was triggered by epoch, not steps
                            # This prevents filename conflicts when resuming training in the same epoch
                            epoch_for_naming = None
                            if (
                                args.sample_every_n_epochs is not None
                                and args.sample_every_n_epochs > 0
                                and (epoch + 1) % args.sample_every_n_epochs == 0
                            ):
                                # This sampling was triggered by epoch boundary
                                epoch_for_naming = epoch + 1
                            # Otherwise, leave epoch_for_naming as None to use step-based naming

                            sampling_manager.sample_images(
                                accelerator,
                                args,
                                epoch_for_naming,  # Use None for step-based sampling
                                global_step,
                                vae,
                                transformer,
                                sample_parameters,
                                dit_dtype,
                            )
                            # Track that sampling occurred at this step
                            last_sampled_step = global_step

                        if should_validating:
                            # Sync validation datasets before validation runs
                            self.validation_core.sync_validation_epoch(
                                val_dataloader,
                                val_epoch_step_sync,
                                current_epoch.value if current_epoch else epoch + 1,
                                global_step,
                            )

                            # Determine if validation metrics require a VAE and lazily load if needed
                            requires_vae_for_val = any(
                                [
                                    bool(getattr(args, "enable_perceptual_snr", False)),
                                    bool(getattr(args, "enable_temporal_ssim", False)),
                                    bool(getattr(args, "enable_temporal_lpips", False)),
                                    bool(
                                        getattr(args, "enable_flow_warped_ssim", False)
                                    ),
                                    bool(getattr(args, "enable_fvd", False)),
                                    bool(getattr(args, "enable_vmaf", False)),
                                ]
                            )
                            temp_val_vae = None
                            val_vae_to_use = vae
                            if val_vae_to_use is None and requires_vae_for_val:
                                try:
                                    if sampling_manager is not None:
                                        if accelerator.is_main_process:
                                            logger.info(
                                                "🔄 Loading VAE temporarily for validation metrics..."
                                            )
                                        temp_val_vae = sampling_manager._load_vae_lazy()  # type: ignore[attr-defined]
                                        val_vae_to_use = temp_val_vae
                                    else:
                                        if accelerator.is_main_process:
                                            logger.warning(
                                                "Validation metrics requiring a VAE are enabled but no SamplingManager is available to lazy-load one. Metrics may be skipped."
                                            )
                                except Exception as e:
                                    if accelerator.is_main_process:
                                        logger.warning(
                                            f"Failed to load VAE for validation metrics: {e}"
                                        )

                            val_loss = self.validation_core.validate(
                                accelerator,
                                transformer,
                                val_dataloader,
                                noise_scheduler,
                                args,
                                control_signal_processor,
                                val_vae_to_use,
                                global_step,
                            )
                            self.validation_core.log_validation_results(
                                accelerator, val_loss, global_step
                            )

                            # Unload temporary VAE if it was loaded for validation
                            if (
                                temp_val_vae is not None
                                and sampling_manager is not None
                            ):
                                try:
                                    sampling_manager._unload_vae(temp_val_vae)  # type: ignore[attr-defined]
                                    if accelerator.is_main_process:
                                        logger.info(
                                            "🧹 Unloaded temporary VAE after validation"
                                        )
                                except Exception as e:
                                    if accelerator.is_main_process:
                                        logger.debug(
                                            f"Failed to unload temporary VAE after validation: {e}"
                                        )

                            # Track that validation occurred at this step
                            last_validated_step = global_step

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process and save_model:
                                from utils import train_utils

                                ckpt_name = train_utils.get_step_ckpt_name(
                                    args.output_name, global_step
                                )
                                save_model(
                                    ckpt_name,
                                    accelerator.unwrap_model(network),
                                    global_step,
                                    epoch + 1,
                                )

                                if args.save_state:
                                    train_utils.save_and_remove_state_stepwise(
                                        args, accelerator, global_step
                                    )

                                remove_step_no = train_utils.get_remove_step_no(
                                    args, global_step
                                )
                                if remove_step_no is not None and remove_model:
                                    remove_ckpt_name = train_utils.get_step_ckpt_name(
                                        args.output_name, remove_step_no
                                    )
                                    remove_model(remove_ckpt_name)
                        if optimizer_train_fn:
                            optimizer_train_fn()

                current_loss = float(loss_components.total_loss.detach().item())
                loss_recorder.add(epoch=epoch + 1, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average

                # Record training step for throughput tracking
                self.record_training_step(bsz)

                # Update EMA loss for TensorBoard logging
                ema_loss_value = self.update_ema_loss(current_loss)

                # Generate enhanced progress bar metrics safely if enabled
                if getattr(args, "enhanced_progress_bar", True):
                    try:
                        # Calculate epoch progress information
                        current_step_in_epoch = step + 1  # step is 0-indexed
                        total_steps_in_epoch = len(train_dataloader)

                        enhanced_logs = self.generate_safe_progress_metrics(
                            args,
                            current_loss,
                            avr_loss,
                            lr_scheduler,
                            epoch,
                            global_step,
                            keys_scaled,
                            mean_norm,
                            maximum_norm,
                            current_step_in_epoch,
                            total_steps_in_epoch,
                        )

                        # Add hardware metrics to progress bar
                        hardware_metrics = get_hardware_metrics()
                        enhanced_logs.update(hardware_metrics)
                        progress_bar.set_postfix(enhanced_logs)
                    except Exception:
                        # Fallback to original simple display if enhanced metrics fail
                        logs = {"avr_loss": avr_loss}
                        progress_bar.set_postfix(logs)
                        if args.scale_weight_norms:
                            progress_bar.set_postfix({**max_mean_logs, **logs})
                else:
                    # Use original simple progress bar
                    logs = {"avr_loss": avr_loss}
                    progress_bar.set_postfix(logs)
                    if args.scale_weight_norms:
                        progress_bar.set_postfix({**max_mean_logs, **logs})

                # Only the main process should handle logging and saving
                if accelerator.is_main_process and len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        ema_loss_value,
                        network,  # Pass the model for parameter stats
                        global_step,  # Pass global_step for parameter stats
                        per_source_losses,  # Pass per-source losses
                        gradient_norm,  # Pass gradient norm
                    )

                    # Add performance metrics
                    if accelerator.sync_gradients:
                        # Get timing metrics
                        timing_metrics = get_timing_metrics()
                        logs.update(timing_metrics)

                        # Get model statistics
                        model_stats = get_model_statistics(
                            model_pred.to(network_dtype),
                            target,
                            accelerator.is_main_process,
                            timesteps,
                            global_step,
                        )
                        logs.update(model_stats)

                        # Log performance summary
                        log_performance_summary(global_step, timing_metrics)

                    # Attach GGPO metrics if available
                    try:
                        if hasattr(network, "grad_norms") and hasattr(
                            network, "combined_weight_norms"
                        ):
                            gn = accelerator.unwrap_model(network).grad_norms()
                            wn = accelerator.unwrap_model(
                                network
                            ).combined_weight_norms()
                            if gn is not None:
                                logs["norm/avg_grad_norm"] = float(gn.item())
                            if wn is not None:
                                logs["norm/avg_combined_norm"] = float(wn.item())
                    except Exception:
                        pass

                    # Attach component losses if available
                    base_loss_val = getattr(loss_components, "base_loss", None)
                    if base_loss_val is not None:
                        logs["loss/mse"] = float(base_loss_val.item())
                    if loss_components.dispersive_loss is not None:
                        logs["loss/dispersive"] = float(
                            loss_components.dispersive_loss.item()
                        )
                    if loss_components.dop_loss is not None:
                        logs["loss/dop"] = float(loss_components.dop_loss.item())
                    if loss_components.optical_flow_loss is not None:
                        logs["loss/optical_flow"] = float(
                            loss_components.optical_flow_loss.item()
                        )
                    if loss_components.repa_loss is not None:
                        logs["loss/repa"] = float(loss_components.repa_loss.item())

                    # Optionally compute extra training metrics periodically
                    try:
                        if (
                            getattr(args, "log_extra_train_metrics", True)
                            and (args.train_metrics_interval or 0) > 0
                            and (global_step % int(args.train_metrics_interval) == 0)
                        ):
                            extra_metrics = (
                                self.loss_computer.compute_extra_train_metrics(
                                    model_pred=model_pred,
                                    target=target,
                                    noise=noise,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    accelerator=accelerator,
                                )
                            )
                            if extra_metrics:
                                logs.update(extra_metrics)
                    except Exception:
                        pass

                    # Log scalar attention metrics and optional heatmap via helper
                    try:
                        from common.attention_logging import (
                            attach_attention_metrics_and_maybe_heatmap as _attn_log_helper,
                        )

                        _attn_log_helper(accelerator, args, logs, global_step)
                    except Exception:
                        pass

                    try:
                        from utils.tensorboard_utils import (
                            apply_direction_hints_to_logs as _adh,
                        )

                        logs = _adh(args, logs)
                    except Exception:
                        pass
                    accelerator.log(logs, step=global_step)

                    # Enhanced optimizer-specific histogram logging
                    if optimizer is not None and is_supported(optimizer):
                        try:
                            histogram_data = get_histogram_data(optimizer)
                            if histogram_data:
                                metric_name, tensor_data = histogram_data
                                # Make sure tensor_data is not empty
                                if tensor_data.numel() > 0:
                                    # Log histogram directly to TensorBoard writer
                                    # TensorBoard expects histograms to be logged via add_histogram
                                    for tracker in accelerator.trackers:
                                        if tracker.name == "tensorboard":
                                            tracker.writer.add_histogram(
                                                metric_name, tensor_data, global_step
                                            )
                                            break
                        except Exception as e:
                            logger.debug(
                                f"Failed to log enhanced optimizer histogram: {e}"
                            )

                    # Periodic live histogram of used timesteps (1..1000)
                    try:
                        log_live_timestep_distribution(
                            accelerator, args, timesteps, global_step
                        )
                    except Exception:
                        pass

                    # Periodic loss-vs-timestep scatter figure
                    try:
                        log_loss_scatterplot(
                            accelerator,
                            args,
                            timesteps,
                            model_pred,
                            target,
                            global_step,
                        )
                    except Exception:
                        pass

                # End step timing
                end_step_timing()

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process and len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_recorder.moving_average}
                try:
                    from utils.tensorboard_utils import (
                        apply_direction_hints_to_logs as _adh,
                    )

                    logs = _adh(args, logs)
                except Exception:
                    pass
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # save model at the end of epoch if needed
            if optimizer_eval_fn:
                optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs
                if is_main_process and saving and save_model:
                    from utils import train_utils

                    ckpt_name = train_utils.get_epoch_ckpt_name(
                        args.output_name, epoch + 1
                    )
                    save_model(
                        ckpt_name,
                        accelerator.unwrap_model(network),
                        global_step,
                        epoch + 1,
                    )

                    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None and remove_model:
                        remove_ckpt_name = train_utils.get_epoch_ckpt_name(
                            args.output_name, remove_epoch_no
                        )
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        from utils import train_utils

                        train_utils.save_and_remove_state_on_epoch_end(
                            args, accelerator, epoch + 1
                        )

            # Only sample at end of epoch if epoch-based sampling is enabled AND it's not already sampled during the last step
            # This prevents double sampling when the last step of an epoch also triggers sampling
            should_sample_at_epoch_end = (
                args.sample_every_n_epochs is not None
                and args.sample_every_n_epochs > 0
                and (epoch + 1) % args.sample_every_n_epochs == 0
            )
            # Only sample if epoch-based sampling is enabled AND we haven't already sampled at this step
            if (
                should_sample_at_epoch_end
                and last_sampled_step != global_step
                and sampling_manager
            ):
                sampling_manager.sample_images(
                    accelerator,
                    args,
                    epoch + 1,
                    global_step,
                    vae,
                    transformer,
                    sample_parameters,
                    dit_dtype,
                )
            if optimizer_train_fn:
                optimizer_train_fn()

            # Do validation only if validation dataloader exists, validation hasn't already run at this step, and epoch-end validation is enabled
            should_validate_on_epoch_end = getattr(args, "validate_on_epoch_end", False)
            if (
                val_dataloader is not None
                and last_validated_step != global_step
                and should_validate_on_epoch_end
            ):
                # Sync validation datasets before validation runs
                self.validation_core.sync_validation_epoch(
                    val_dataloader,
                    val_epoch_step_sync,
                    current_epoch.value if current_epoch else epoch + 1,
                    global_step,
                )

                val_loss = self.validation_core.validate(
                    accelerator,
                    transformer,
                    val_dataloader,
                    noise_scheduler,
                    args,
                    control_signal_processor,
                    vae,
                    global_step,
                )
                self.validation_core.log_validation_results(
                    accelerator, val_loss, global_step, epoch + 1
                )
            elif val_dataloader is None:
                accelerator.print(
                    f"\n[Epoch {epoch+1}] No validation dataset configured"
                )
            elif not should_validate_on_epoch_end:
                accelerator.print(f"\n[Epoch {epoch+1}] Epoch-end validation disabled")
            else:
                accelerator.print(
                    f"\n[Epoch {epoch+1}] Validation already performed at step {global_step}"
                )

            # end of epoch

        return global_step, network
