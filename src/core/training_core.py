"""Core training logic for Takenoko.
This module handles the main training loop, model calling, loss computation, and validation.
"""

import argparse
import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
import torch.nn.functional as F
from easydict import EasyDict

from utils.ema import ExponentialMovingAverage

import logging
from common.logger import get_logger
from memory.safe_memory_manager import SafeMemoryManager

logger = get_logger(__name__, level=logging.INFO)

from enhancements.temporal_consistency.training_integration import (
    enhance_loss_with_temporal_consistency,
)


from enhancements.differential_guidance.training_integration import (
    create_differential_guidance_integration,
    transform_target_with_differential_guidance,
)

from enhancements.slider.slider_integration import compute_slider_loss_if_enabled
import utils.fluxflow_augmentation as fluxflow_augmentation
from transition.manager import TransitionTrainingManager
from transition.pipeline import (
    prepare_transition_batch,
    finalize_batch_if_enabled,
    update_teacher_if_needed,
)
from core.training_inputs import prepare_standard_training_inputs
from energy_based.eqm_mode.training_helper import EqMTrainingHelper


from scheduling.fvdm_manager import create_fvdm_manager

from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import (
    compute_loss_weighting_for_sd3,
    should_sample_images,
    LossRecorder,
)
from scheduling.timestep_distribution import TimestepDistribution
from scheduling.timestep_logging import (
    log_initial_timestep_distribution,
    log_show_timesteps_figure_unconditional,
)
from scheduling.timestep_utils import (
    get_noisy_model_input_and_timesteps,
    initialize_timestep_distribution,
)
from core.validation_core import ValidationCore
from criteria.training_loss import TrainingLossComputer
from common.context_memory_manager import ContextMemoryManager

from junctions.training_events import trigger_event

from core import ema_helper
from core.handlers.logging_config import (
    configure_advanced_logging as _configure_advanced_logging,
)
from core.handlers.adaptive_config import (
    initialize_adaptive_timestep_sampling as _initialize_adaptive_timestep_sampling,
)
from core.handlers.misc import (
    scale_shift_latents as _scale_shift_latents,
    record_training_step as _record_training_step,
    initialize_throughput_tracker as _initialize_throughput_tracker,
)
from core.handlers.ema_utils import (
    validate_ema_beta,
    configure_ema_from_args as _configure_ema_from_args,
    update_ema_loss as _update_ema_loss,
    update_iter_time_ema as _update_iter_time_ema,
)
from core.handlers.metrics_utils import (
    generate_parameter_stats as _generate_parameter_stats,
    should_generate_parameter_stats,
    compute_per_source_loss as _compute_per_source_loss,
    compute_gradient_norm as _compute_gradient_norm,
)
from core.handlers.progress_bar_handler import process_enhanced_progress_bar
from core.handlers.logging_handler import collect_and_log_training_metrics
from core.handlers.sampling_handler import (
    handle_training_sampling_with_accelerator,
    handle_epoch_end_sampling_with_accelerator,
)
from core.handlers.validation_handler import (
    handle_step_validation,
    handle_epoch_end_validation,
)
from core.handlers.saving_handler import (
    handle_step_saving,
    handle_epoch_end_saving,
)
from core.handlers.adaptive_handler import handle_adaptive_timestep_sampling
from core.handlers.self_correction_handler import handle_self_correction_update
from core.handlers.vram_validation_handler import (
    handle_vram_validation_if_enabled,
    handle_windows_shared_memory_check,
)

from common.performance_logger import (
    start_step_timing,
    start_forward_pass_timing,
    end_forward_pass_timing,
    start_backward_pass_timing,
    end_backward_pass_timing,
    start_optimizer_step_timing,
    end_optimizer_step_timing,
    end_step_timing,
)

from distillation.rcm_core.forward import (
    RCMDenoiseResult,
    estimate_trigflow_from_timesteps,
    rcm_compute_forward,
)


class TrainingCore:
    """Handles core training logic, model calling, and validation."""

    def __init__(self, config: Any, fluxflow_config: Dict[str, Any]):
        self.config = config
        self.fluxflow_config = fluxflow_config

        # Initialize validation core
        self.validation_core = ValidationCore(config, fluxflow_config)

        # Initialize adaptive timestep manager (initially None)
        self.adaptive_manager: Optional[Any] = None

        # Noise scheduler reference (set in run_training_loop)
        self.noise_scheduler: Optional[Any] = None

        # EMA loss tracking for smoother TensorBoard visualization
        self.ema_loss: Optional[float] = None
        self.ema_beta: float = 0.98  # Smoothing factor (0.9-0.99 are good choices)

        self.ema_step_count: int = 0
        self.ema_bias_warmup_steps: int = (
            100  # defer bias correction for early readability
        )
        self.ema_loss_zero_started: Optional[float] = None

        # Parameter statistics tracking
        self.last_param_log_step: int = -1

        # Per-source loss tracking
        self.per_source_losses: Dict[str, List[float]] = {}

        # Pre-computed timestep distribution (initialized when needed)
        self.timestep_distribution = TimestepDistribution()
        self.transition_manager: Optional[TransitionTrainingManager] = None
        self._timestep_logging_initialized: bool = False

        # Gradient norm tracking
        self.gradient_norms: List[float] = []

        # TensorBoardX enhanced logging
        self.tensorboardx_writer = None
        self.use_tensorboardx = False

        # Centralized loss computation
        self.loss_computer = TrainingLossComputer(self.config)

        # Context memory components (all-in-one manager)
        self.context_memory_manager = ContextMemoryManager(self.config.__dict__)

        # Training-safe memory optimization manager
        self.memory_manager = SafeMemoryManager(self.config.__dict__)

        # Toggle for alternating progress postfix (iter_ms vs peak VRAM)
        self._perf_display_toggle: bool = False

        # One-time warning flag for skipping perceptual metrics without pixels
        self._warned_no_val_pixels_for_perceptual = False

        # EMA for iteration time display (seconds)
        self._iter_time_ema_sec: Optional[float] = None
        self._iter_time_ema_beta: float = 0.95

        # EqM transport helper (experimental)
        self.eqm_training_helper: Optional[EqMTrainingHelper] = None

        # Initialize temporal consistency enhancement
        self.temporal_consistency_integration = None

        # Initialize differential guidance enhancement
        self.differential_guidance_integration = None

        # Weight EMA tracking (initialized when enabled)
        self.weight_ema: Optional[ExponentialMovingAverage] = None
        self.weight_ema_start_step: int = 0
        self.weight_ema_use_for_eval: bool = True
        self.weight_ema_update_interval: int = 1
        self.weight_ema_device: str = "accelerator"
        self.weight_ema_eval_mode: str = "off"
        self.weight_ema_save_separately: bool = False
        self._weight_ema_initialized: bool = False

    def set_ema_beta(self, beta: float) -> None:
        """Set the EMA smoothing factor. Higher values (closer to 1.0) = more smoothing."""
        validate_ema_beta(beta)
        self.ema_beta = beta
        logger.info(f"EMA beta set to {beta}")

    def configure_ema_from_args(self, args: argparse.Namespace) -> None:
        """Configure EMA hyperparameters from args if provided."""
        ema_beta, ema_bias_warmup_steps = _configure_ema_from_args(args)
        if ema_beta is not None:
            self.set_ema_beta(ema_beta)
        if ema_bias_warmup_steps is not None:
            self.ema_bias_warmup_steps = ema_bias_warmup_steps

    def configure_advanced_logging(self, args: argparse.Namespace) -> None:
        """Configure advanced logging settings including parameter stats, per-source losses, and gradient norms.

        Sets default values for various logging options if not already set in args.
        """
        _configure_advanced_logging(args)

    def initialize_weight_ema(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network: Any,
        register_checkpoint: bool = True,
    ) -> None:
        """Optionally initialize model weight EMA for eval/saving."""
        try:
            ema_helper.initialize_weight_ema(
                owner=self,
                args=args,
                accelerator=accelerator,
                network=network,
                register_checkpoint=register_checkpoint,
            )
        except Exception as exc:
            logger.warning("Weight EMA setup skipped: %s", exc)

    def _weight_ema_eval_context(self):
        """Return context manager that swaps EMA weights in for eval/saving when enabled."""
        return ema_helper.weight_ema_eval_context(self)

    def _update_weight_ema_if_needed(
        self, accelerator: Accelerator, global_step: int
    ) -> None:
        """Update weight EMA if enabled and at the correct interval."""
        try:
            ema_helper.update_weight_ema_if_needed(self, accelerator, global_step)
        except Exception as exc:
            logger.warning("Disabling weight EMA due to update failure: %s", exc)
            self.weight_ema = None

    def initialize_adaptive_timestep_sampling(self, args: argparse.Namespace) -> None:
        """Initialize adaptive timestep sampling and FVDM manager."""
        self.adaptive_manager = _initialize_adaptive_timestep_sampling(args)

        # Initialize FVDM manager after adaptive manager is available
        self.fvdm_manager = create_fvdm_manager(
            args,
            device=torch.device("cpu"),  # Temporary device, will be updated
            adaptive_manager=self.adaptive_manager,
        )

        # Connect adaptive manager to timestep distribution
        if (
            self.adaptive_manager
            and hasattr(self, "timestep_distribution")
            and self.timestep_distribution
        ):
            if hasattr(self.timestep_distribution, "set_adaptive_manager"):
                self.timestep_distribution.set_adaptive_manager(self.adaptive_manager)

    def initialize_temporal_consistency_enhancement(
        self, args: argparse.Namespace
    ) -> None:
        """Initialize temporal consistency enhancement if enabled."""
        from enhancements.temporal_consistency.main import (
            create_temporal_consistency_integration,
        )

        self.temporal_consistency_integration = create_temporal_consistency_integration(
            args
        )

    def initialize_differential_guidance(self, args: argparse.Namespace) -> None:
        """Initialize differential guidance enhancement if enabled."""
        self.differential_guidance_integration = (
            create_differential_guidance_integration(args)
        )

    def update_ema_loss(self, current_loss: float) -> Tuple[float, float]:
        """Update EMA loss; warm-start and defer bias correction for early steps."""
        (
            reported_ema,
            new_ema_loss,
            new_zero_started_ema,
            new_step_count,
            debiased_ema,
        ) = _update_ema_loss(
            current_loss,
            self.ema_loss,
            self.ema_step_count,
            self.ema_beta,
            self.ema_bias_warmup_steps,
            self.ema_loss_zero_started,
        )

        # Update instance state
        self.ema_loss = new_ema_loss
        self.ema_loss_zero_started = new_zero_started_ema
        self.ema_step_count = new_step_count

        return reported_ema, debiased_ema

    def _update_iter_time_ema(self, last_iter_seconds: float) -> float:
        """Update EMA of iteration time in seconds.

        Args:
            last_iter_seconds: Duration of last iteration in seconds (>0)

        Returns:
            The updated EMA value in seconds.
        """
        new_ema = _update_iter_time_ema(
            last_iter_seconds, self._iter_time_ema_sec, self._iter_time_ema_beta
        )
        self._iter_time_ema_sec = new_ema
        return new_ema

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
        *,
        rcm_mode: bool = False,
        rcm_trigflow: Optional[torch.Tensor] = None,
        rcm_t_scaling_factor: Optional[float] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[RCMDenoiseResult],
        ],
    ]:
        """Call the DiT model and compute target for loss calculation.

        Args:
            rcm_mode: When ``True`` the method will additionally compute
                rectified-flow predictions and return them as a fourth element.
                The default behaviour (``False``) preserves the historical
                3-tuple return contract.
            rcm_trigflow: Optional pre-computed trigflow time tensor.  When
                ``None`` a heuristic conversion from discrete timesteps is used.
            rcm_t_scaling_factor: Allows overriding the scaling factor applied
                to ``c_noise``; falls back to 1000 following the reference
                implementation.
        """
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
            logger.info(
                "Р РѓР Р‡Р С›Р С— Control LoRA processing engaged in training loop"
            )

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
                clip_fea=None,
                seq_len=seq_len,
                y=None,
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

        rcm_result: Optional[RCMDenoiseResult] = None
        if rcm_mode:
            base_train_steps = None
            if getattr(self, "noise_scheduler", None) is not None:
                base_train_steps = getattr(
                    getattr(self.noise_scheduler, "config", {}),
                    "num_train_timesteps",
                    None,
                )
            trigflow = (
                rcm_trigflow
                if rcm_trigflow is not None
                else estimate_trigflow_from_timesteps(
                    timesteps, base_num_train_timesteps=base_train_steps
                )
            )
            scaling_factor = (
                float(rcm_t_scaling_factor)
                if rcm_t_scaling_factor is not None
                else float(getattr(args, "rcm_t_scaling_factor", 1000.0))
            )
            try:
                rcm_result = rcm_compute_forward(
                    xt=noisy_model_input,
                    model_output=model_pred,
                    trigflow_t=trigflow,
                    t_scaling_factor=scaling_factor,
                )
            except Exception as exc:  # pragma: no cover - rCM path is optional
                if accelerator.is_local_main_process:
                    logger.warning("RCM forward pass helper failed: %s", exc)
                rcm_result = None

        # flow matching loss - compute target using perturbed latents if fluxflow is enabled
        # Support custom loss target computation
        enable_custom_target = getattr(args, "enable_custom_loss_target", False)
        if (
            enable_custom_target
            and self.noise_scheduler is not None
            and hasattr(self.noise_scheduler, "get_loss_target")
        ):
            try:
                target = self.noise_scheduler.get_loss_target(
                    noise=noise,
                    latents=perturbed_latents_for_target.to(
                        device=accelerator.device, dtype=network_dtype
                    ),
                    timesteps=timesteps,
                ).detach()
            except Exception as e:
                logger.warning(
                    f"Custom get_loss_target() failed, falling back to standard flow matching: {e}"
                )
                target = noise - perturbed_latents_for_target.to(
                    device=accelerator.device, dtype=network_dtype
                )
        else:
            target = noise - perturbed_latents_for_target.to(
                device=accelerator.device, dtype=network_dtype
            )

        if rcm_mode:
            return model_pred, target, intermediate_z, rcm_result
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
        sara_helper: Optional[Any] = None,
        layer_sync_helper: Optional[Any] = None,
        controlnet: Optional[Any] = None,
        dual_model_manager: Optional[Any] = None,
    ) -> Tuple[int, Any]:
        """Run the main training loop."""

        # Store noise scheduler reference for call_dit
        self.noise_scheduler = noise_scheduler

        transition_cfg = getattr(args, "transition_training", None)
        if transition_cfg and getattr(transition_cfg, "enabled", False):
            self.transition_manager = TransitionTrainingManager(
                args=args, config=transition_cfg
            )
            try:
                self.transition_manager.attach_teacher(
                    accelerator.unwrap_model(transformer)
                )
            except Exception:
                pass
        else:
            self.transition_manager = None

        self.eqm_training_helper = EqMTrainingHelper.maybe_create(args, noise_scheduler)
        eqm_enabled = self.eqm_training_helper is not None
        if eqm_enabled and self.eqm_training_helper is not None:
            metadata = self.eqm_training_helper.log_startup()
            logger.info(
                "EqM mode enabled (prediction=%s, path=%s).",
                metadata.get("prediction"),
                metadata.get("path_type"),
            )
        else:
            self.eqm_training_helper = None

        # Configure EMA hyperparameters from args (non-fatal)
        try:
            self.configure_ema_from_args(args)
        except Exception:
            pass

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

            # Check if fast validation would occur at the current global_step
            if (
                hasattr(args, "validate_fast_every_n_steps")
                and args.validate_fast_every_n_steps is not None
                and global_step % args.validate_fast_every_n_steps == 0
            ):
                last_fast_validated_step = global_step
            else:
                last_fast_validated_step = -1
        else:
            # Fresh training start
            last_sampled_step = -1
            last_validated_step = -1
            last_fast_validated_step = -1

        # Initialize throughput tracker with configuration
        _initialize_throughput_tracker(args)

        # Wire Temporal Consistency TensorBoard logger once (integration handles cadence)
        try:
            if (
                self.temporal_consistency_integration is not None
                and self.temporal_consistency_integration.is_enabled()
            ):

                def _tc_tb_logger(logs: Dict[str, float], step: int) -> None:
                    accelerator.log(logs, step=step)

                tb_every = getattr(args, "freq_temporal_tb_log_every_steps", None)
                self.temporal_consistency_integration.set_tensorboard_logger(
                    _tc_tb_logger, every_steps=tb_every
                )
        except Exception:
            pass

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            if current_epoch is not None:
                current_epoch.value = epoch + 1

            metadata["takenoko_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)

            # Trigger epoch_start junction event
            trigger_event(
                "epoch_start",
                args=args,
                accelerator=accelerator,
                epoch=epoch + 1,
                transformer=transformer,
                network=network,
                global_step=global_step,
                metadata=metadata,
            )

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
                # Update FVDM manager device on first step (when accelerator is available)
                if (
                    step == 0
                    and hasattr(self, "fvdm_manager")
                    and self.fvdm_manager.device != accelerator.device
                ):
                    self.fvdm_manager.device = accelerator.device
                    if (
                        hasattr(self.fvdm_manager, "metrics")
                        and self.fvdm_manager.metrics
                    ):
                        self.fvdm_manager.metrics.device = accelerator.device

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

                    # Trigger step_start junction event
                    trigger_event(
                        "step_start",
                        args=args,
                        accelerator=accelerator,
                        epoch=epoch + 1,
                        step=step,
                        global_step=global_step,
                        batch=batch,
                        latents=latents,
                        network=network,
                    )

                    latents = _scale_shift_latents(latents)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    fvdm_sampling_metadata = None
                    if self.fvdm_manager.enabled:
                        # Initialize timestep distribution if needed (same as regular path)
                        initialize_timestep_distribution(
                            args, self.timestep_distribution
                        )

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

                        # FVDM is enabled - use FVDM manager with full parameter alignment
                        (
                            noisy_model_input,
                            timesteps,
                            sigmas,
                            fvdm_sampling_metadata,
                        ) = self.fvdm_manager.get_noisy_input_and_timesteps(
                            noise,
                            latents,
                            noise_scheduler,
                            dit_dtype,
                            step,
                            timestep_distribution=self.timestep_distribution,
                            presampled_uniform=None,  # Could be added later for advanced use
                        )

                        # Compute weighting for FVDM timesteps
                        weighting = compute_loss_weighting_for_sd3(
                            args.weighting_scheme,
                            noise_scheduler,
                            timesteps,
                            accelerator.device,
                            dit_dtype,
                        )

                        self.fvdm_manager.log_timestep_distribution_comparison(
                            accelerator=accelerator,
                            latents=latents,
                            timesteps=timesteps,
                            global_step=global_step,
                            noise_scheduler=noise_scheduler,
                            timestep_distribution=self.timestep_distribution,
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

                        # Calculate model input and timesteps (standard path)
                        noisy_model_input, timesteps, sigmas, weighting = (
                            prepare_standard_training_inputs(
                                args,
                                accelerator,
                                latents,
                                noise,
                                noise_scheduler,
                                dit_dtype,
                                self.timestep_distribution,
                                dual_model_manager,
                                batch,
                            )
                        )

                    transition_reference_timesteps = None
                    need_intermediate = False
                    active_transition_manager = None
                    prepared_transition = None
                    if (
                        self.transition_manager is not None
                        and dual_model_manager is None
                        and not eqm_enabled
                    ):
                        active_transition_manager = self.transition_manager
                        prepared_transition = prepare_transition_batch(
                            active_transition_manager,
                            accelerator,
                            latents,
                            noise,
                        )
                        if prepared_transition is not None:
                            noisy_model_input = prepared_transition.noisy_latents
                            timesteps = prepared_transition.timesteps.to(
                                device=accelerator.device, dtype=dit_dtype
                            )
                            transition_reference_timesteps = (
                                prepared_transition.reference_timesteps.to(
                                    device=accelerator.device, dtype=dit_dtype
                                )
                            )
                            weighting = prepared_transition.weights.to(
                                device=accelerator.device, dtype=network_dtype
                            )
                            sigmas = None
                            need_intermediate = prepared_transition.need_directional
                    # Optional: If the network supports TLora-style masking, update mask from timesteps
                    try:
                        unwrapped_net = accelerator.unwrap_model(network)
                        if hasattr(unwrapped_net, "update_rank_mask_from_timesteps"):
                            unwrapped_net.update_rank_mask_from_timesteps(
                                timesteps, max_timestep=1000, device=accelerator.device
                            )
                    except Exception:
                        pass

                    if weighting is None:
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

                    if eqm_enabled:
                        trigger_event(
                            "before_forward",
                            args=args,
                            accelerator=accelerator,
                            latents=latents,
                            batch=batch,
                            noise=noise,
                            noisy_model_input=noisy_model_input,
                            timesteps=timesteps,
                            network_dtype=network_dtype,
                            transformer=active_transformer,
                            weighting=weighting,
                        )

                        if self.eqm_training_helper is None:
                            raise RuntimeError(
                                "EqM training helper was not initialised."
                            )

                        eqm_result = self.eqm_training_helper.prepare_forward(
                            transformer=active_transformer,
                            latents=latents,
                            batch=batch,
                            args=args,
                            accelerator=accelerator,
                            network_dtype=network_dtype,
                            patch_size=self.config.patch_size,
                            global_step=global_step,
                        )
                        end_forward_pass_timing()

                        model_pred = eqm_result.model_pred
                        target = eqm_result.target
                        timesteps = eqm_result.timesteps
                        noise = eqm_result.noise
                        noisy_model_input = eqm_result.noisy_latents
                        intermediate_z = eqm_result.intermediate
                        context_memory_loss = torch.zeros(
                            (), device=accelerator.device, dtype=network_dtype
                        )
                        context_stats = {}
                        loss_components = eqm_result.loss_components
                        if eqm_result.metrics and accelerator.is_main_process:
                            accelerator.log(eqm_result.metrics, step=global_step)
                        weighting = None
                        per_source_losses = {}
                        transition_loss_context = {"transition_training_enabled": False}

                        trigger_event(
                            "after_forward",
                            args=args,
                            accelerator=accelerator,
                            model_pred=model_pred,
                            target=target,
                            intermediate_z=intermediate_z,
                            latents=latents,
                            batch=batch,
                            noise=noise,
                            noisy_model_input=noisy_model_input,
                            timesteps=timesteps,
                            network_dtype=network_dtype,
                            weighting=weighting,
                        )
                    else:
                        # Trigger before_forward junction event
                        trigger_event(
                            "before_forward",
                            args=args,
                            accelerator=accelerator,
                            latents=latents,
                            batch=batch,
                            noise=noise,
                            noisy_model_input=noisy_model_input,
                            timesteps=timesteps,
                            network_dtype=network_dtype,
                            transformer=active_transformer,
                            weighting=weighting,
                        )

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

                        # Unpack model result - handle both 3-tuple and 4-tuple returns
                        # (4-tuple is returned when rcm_mode=True, includes rcm_result)
                        if len(model_result) == 4:
                            model_pred, target, intermediate_z, _ = model_result
                        else:
                            model_pred, target, intermediate_z = model_result

                        transition_loss_context = {"transition_training_enabled": False}
                        per_sample_transition_loss = None

                        if active_transition_manager is not None:

                            def _transition_forward(
                                model_input: torch.Tensor, step_tensor: torch.Tensor
                            ) -> torch.Tensor:
                                local_ctx = (
                                    active_transition_manager.lora_modulation_context(
                                        accelerator.unwrap_model(network),
                                        step_tensor,
                                        step_tensor,
                                    )
                                )
                                with local_ctx:
                                    pred, _, _ = self.call_dit(
                                        args,
                                        accelerator,
                                        active_transformer,
                                        latents,
                                        batch,
                                        noise,
                                        model_input,
                                        step_tensor,
                                        network_dtype,
                                        control_signal_processor,
                                        controlnet,
                                        return_intermediate=False,
                                        override_target=True,
                                    )
                                return pred

                            (
                                transition_target,
                                transition_weighting,
                                transition_metrics,
                                transition_directional_loss,
                                per_sample_transition_loss,
                            ) = finalize_batch_if_enabled(
                                active_transition_manager,
                                accelerator,
                                model_pred,
                                _transition_forward,
                                accelerator.unwrap_model(active_transformer),
                                need_intermediate,
                                intermediate_z,
                            )

                            if transition_target is not None:
                                target = transition_target
                                if transition_weighting is not None:
                                    weighting = transition_weighting
                                if transition_metrics and accelerator.is_main_process:
                                    accelerator.log(
                                        transition_metrics, step=global_step
                                    )
                                transition_loss_context = {
                                    "transition_training_enabled": True,
                                    "per_sample_loss": per_sample_transition_loss,
                                    "directional_loss": transition_directional_loss,
                                    "directional_weight": getattr(
                                        self.transition_manager,
                                        "directional_weight",
                                        0.0,
                                    ),
                                }
                        # Context memory processing
                        context_memory_loss, context_stats = (
                            self.context_memory_manager.process_training_step(
                                latents=latents,
                                global_step=global_step,
                                step=step,
                                args=args,
                                accelerator=accelerator,
                                temporal_consistency_loss_fn=self.context_memory_manager.create_temporal_consistency_loss_fn(),
                                batch=batch,
                            )
                        )

                        # Log context memory stats if available
                        if context_stats:
                            self.context_memory_manager.log_stats_to_accelerator(
                                context_stats, accelerator, global_step
                            )

                        # Trigger after_forward junction event
                        trigger_event(
                            "after_forward",
                            args=args,
                            accelerator=accelerator,
                            model_pred=model_pred,
                            target=target,
                            intermediate_z=intermediate_z,
                            latents=latents,
                            batch=batch,
                            noise=noise,
                            noisy_model_input=noisy_model_input,
                            timesteps=timesteps,
                            network_dtype=network_dtype,
                            weighting=weighting,
                        )

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
                                per_source_losses = _compute_per_source_loss(
                                    model_pred.to(network_dtype),
                                    target,
                                    batch,
                                    weighting,
                                    sample_weights_local,
                                )
                            except Exception as e:
                                logger.debug(
                                    f"⚠️  Failed to compute per-source losses: {e}"
                                )
                                per_source_losses = {}

                        if not eqm_enabled:
                            # Apply differential guidance to target if enabled
                            target = transform_target_with_differential_guidance(
                                self.differential_guidance_integration,
                                target=target,
                                model_pred=model_pred,
                                step=global_step,
                            )

                            # Centralized training loss computation (includes DOP/REPA/Dispersive/OpticalFlow/Slider)
                            loss_components = compute_slider_loss_if_enabled(
                                loss_computer=self.loss_computer,
                                transformer=active_transformer,
                                network=network,
                                noisy_latents=noisy_model_input,
                                timesteps=timesteps,
                                batch=batch,
                                noise=noise,
                                noise_scheduler=noise_scheduler,
                                args=args,
                                accelerator=accelerator,
                                latents=latents,
                                network_dtype=network_dtype,
                                model_pred=model_pred,
                                target=target,
                                weighting=weighting,
                                intermediate_z=intermediate_z,
                                vae=vae,
                                control_signal_processor=control_signal_processor,
                                repa_helper=(
                                    repa_helper if sara_helper is None else None
                                ),
                                sara_helper=sara_helper,
                                layer_sync_helper=layer_sync_helper,
                                raft=getattr(self, "raft", None),
                                warp_fn=getattr(self, "warp", None),
                                adaptive_manager=self.adaptive_manager,
                                transition_loss_context=transition_loss_context,
                            )

                    # Integrate context memory loss into total loss
                    self.context_memory_manager.integrate_context_loss(
                        loss_components=loss_components,
                        context_memory_loss=context_memory_loss,
                        config=self.config.__dict__,
                        accelerator=accelerator,
                        global_step=global_step,
                    )

                    # FVDM: Record training metrics and integrate additional loss components
                    if self.fvdm_manager.enabled:
                        try:
                            # Record FVDM training metrics
                            self.fvdm_manager.record_training_step(
                                frames=latents,
                                timesteps=timesteps,
                                loss=loss_components["loss"],
                                step=global_step,
                                prediction=model_pred,
                                sampling_metadata=fvdm_sampling_metadata,
                            )

                            # Compute and integrate FVDM additional loss components
                            fvdm_additional_loss, fvdm_loss_details = (
                                self.fvdm_manager.get_additional_loss(
                                    frames=latents,
                                    timesteps=timesteps,
                                    prediction=model_pred,
                                )
                            )

                            if fvdm_additional_loss.item() > 0:
                                # Add FVDM loss components to the total loss
                                loss_components["loss"] = (
                                    loss_components["loss"] + fvdm_additional_loss
                                )
                                loss_components.update(
                                    {
                                        f"fvdm_{k}": v
                                        for k, v in fvdm_loss_details.items()
                                    }
                                )

                        except Exception as e:
                            logger.debug(f"FVDM integration warning: {e}")

                    # Trigger after_loss_computation junction event
                    trigger_event(
                        "after_loss_computation",
                        args=args,
                        accelerator=accelerator,
                        loss_components=loss_components,
                        model_pred=model_pred,
                        target=target,
                        timesteps=timesteps,
                        batch=batch,
                        noise=noise,
                        adaptive_manager=self.adaptive_manager,
                    )

                    # Handle adaptive timestep sampling
                    handle_adaptive_timestep_sampling(
                        self.adaptive_manager,
                        accelerator,
                        training_model,
                        latents,
                        noise_scheduler,
                        model_pred,
                        target,
                        timesteps,
                    )

                    # Trigger before_backward junction event
                    trigger_event(
                        "before_backward",
                        args=args,
                        accelerator=accelerator,
                        loss_components=loss_components,
                        model_pred=model_pred,
                        target=target,
                        network=network,
                        global_step=global_step,
                    )

                    # Start backward pass timing
                    start_backward_pass_timing()
                    loss_for_backward = enhance_loss_with_temporal_consistency(
                        self.temporal_consistency_integration,
                        base_loss=loss_components.total_loss,
                        model_pred=model_pred,
                        target=target,
                        step=global_step,
                    )
                    accelerator.backward(loss_for_backward)

                    # End backward pass timing
                    end_backward_pass_timing()

                    # Trigger after_backward junction event
                    trigger_event(
                        "after_backward",
                        args=args,
                        accelerator=accelerator,
                        loss_components=loss_components,
                        network=network,
                        global_step=global_step,
                    )

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
                            gradient_norm = _compute_gradient_norm(network)
                        except Exception as e:
                            logger.debug(
                                f"РЎвЂљР Р„Р В°РЎРЏРІвЂўвЂўР Сџ Failed to compute gradient norm: {e}"
                            )

                if accelerator.sync_gradients:
                    # Trigger before_gradient_clipping junction event
                    trigger_event(
                        "before_gradient_clipping",
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        gradient_norm=gradient_norm,
                        global_step=global_step,
                    )
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
                        update_teacher_if_needed(
                            self.transition_manager, accelerator, transformer
                        )
                    except RuntimeError as e:
                        logger.error(
                            "Р РѓР Р‡Р Р„Р С‘ Make sure you're not adding empty parameter groups to the optimizer"
                        )
                        raise e
                    # End optimizer step timing
                    end_optimizer_step_timing()

                    # Trigger after_optimizer_step junction event
                    trigger_event(
                        "after_optimizer_step",
                        args=args,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        network=network,
                        global_step=global_step,
                    )

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

                    # Trigger after_sync_gradients junction event
                    trigger_event(
                        "after_sync_gradients",
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                    )

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
                    max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Handle self-correction cache refresh
                    handle_self_correction_update(
                        args,
                        global_step,
                        accelerator,
                        transformer,
                        training_model,
                    )

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
                    should_fast_validating = self.validation_core.should_validate_fast(
                        args,
                        global_step,
                        val_dataloader,
                        last_validated_step,
                        last_fast_validated_step,
                    )

                    if (
                        should_sampling
                        or should_saving
                        or should_validating
                        or should_fast_validating
                    ):
                        if optimizer_eval_fn:
                            optimizer_eval_fn()

                        # Conditionally save checkpoint before or after sampling/validation
                        save_before = getattr(
                            args, "save_checkpoint_before_sampling", False
                        )

                        if save_before:
                            # Save checkpoint first, then sample and validate
                            handle_step_saving(
                                should_saving,
                                accelerator,
                                save_model,
                                remove_model,
                                args,
                                network,
                                global_step,
                                epoch,
                            )

                        # Handle training sampling
                        last_sampled_step = handle_training_sampling_with_accelerator(
                            should_sampling,
                            sampling_manager,
                            args,
                            accelerator,
                            epoch,
                            global_step,
                            vae,
                            transformer,
                            sample_parameters,
                            dit_dtype,
                            last_sampled_step,
                        )

                        # Handle step validation (regular and fast)
                        (
                            last_validated_step,
                            self._warned_no_val_pixels_for_perceptual,
                            last_fast_validated_step,
                        ) = handle_step_validation(
                            should_validating,
                            self.validation_core,
                            val_dataloader,
                            val_epoch_step_sync,
                            current_epoch,
                            epoch,
                            global_step,
                            args,
                            accelerator,
                            transformer,
                            noise_scheduler,
                            control_signal_processor,
                            vae,
                            sampling_manager,
                            self._warned_no_val_pixels_for_perceptual,
                            last_validated_step,
                            self.timestep_distribution,
                            should_fast_validating,
                            last_fast_validated_step,
                        )

                        if not save_before:
                            # Save checkpoint after sampling and validation (original behavior)
                            handle_step_saving(
                                should_saving,
                                accelerator,
                                save_model,
                                remove_model,
                                args,
                                network,
                                global_step,
                                epoch,
                            )

                        # Handle independent state-only saving at step level
                        from utils.train_utils import (
                            should_save_state_at_step,
                            save_state_only_at_step,
                        )

                        if should_save_state_at_step(args, global_step):
                            save_state_only_at_step(args, accelerator, global_step)

                        if optimizer_train_fn:
                            optimizer_train_fn()

                current_loss = float(loss_components.total_loss.detach().item())
                loss_recorder.add(epoch=epoch + 1, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average

                # Record training step for throughput tracking
                _record_training_step(bsz)

                # Update EMA loss for TensorBoard logging
                ema_loss_value, ema_loss_debiased = self.update_ema_loss(current_loss)

                # Generate enhanced progress bar metrics safely if enabled
                enhanced_logs, self._perf_display_toggle, self._iter_time_ema_sec = (
                    process_enhanced_progress_bar(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        epoch,
                        global_step,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        self._perf_display_toggle,
                        self._iter_time_ema_sec,
                        self._iter_time_ema_beta,
                        progress_bar,
                        max_mean_logs,
                        step
                        + 1,  # step is 0-indexed, but we want 1-indexed for display
                        len(train_dataloader),
                        transformer,
                        network,
                    )
                )

                # Collect and log all training metrics
                collect_and_log_training_metrics(
                    args,
                    accelerator,
                    current_loss,
                    avr_loss,
                    lr_scheduler,
                    lr_descriptions,
                    optimizer,
                    keys_scaled,
                    mean_norm,
                    maximum_norm,
                    ema_loss_value,
                    ema_loss_debiased,
                    network,
                    global_step,
                    per_source_losses,
                    gradient_norm,
                    model_pred,
                    target,
                    network_dtype,
                    timesteps,
                    loss_components,
                    noise,
                    noise_scheduler,
                    self.adaptive_manager,
                    self.loss_computer,
                )

                # Temporal Consistency: logging handled inside integration (no per-step wiring here)

                # FVDM: Log additional metrics if enabled and appropriate
                if self.fvdm_manager.should_log_metrics(global_step):
                    try:
                        fvdm_metrics = self.fvdm_manager.get_metrics_for_logging(
                            global_step
                        )
                        if fvdm_metrics:
                            accelerator.log(fvdm_metrics, step=global_step)
                    except Exception as e:
                        logger.debug(f"FVDM metrics logging warning: {e}")

                # Trigger step_end junction event
                trigger_event(
                    "step_end",
                    args=args,
                    accelerator=accelerator,
                    epoch=epoch + 1,
                    step=step,
                    global_step=global_step,
                    current_loss=current_loss,
                    avr_loss=avr_loss,
                    network=network,
                    batch_size=bsz,
                )

                # End step timing
                end_step_timing()

                # Validate VRAM estimate vs actual (if enabled, only on first step)
                handle_vram_validation_if_enabled(args, global_step, accelerator)

                # Check for Windows shared GPU memory usage (if enabled, periodic)
                handle_windows_shared_memory_check(args, global_step, accelerator)

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

            # Handle epoch-end saving
            if optimizer_eval_fn:
                optimizer_eval_fn()
            handle_epoch_end_saving(
                args,
                epoch,
                num_train_epochs,
                is_main_process,
                save_model,
                remove_model,
                accelerator,
                network,
                global_step,
            )

            # Handle independent state-only saving at epoch level
            if is_main_process:
                from utils.train_utils import (
                    should_save_state_at_epoch,
                    save_state_only_at_epoch,
                )

                if should_save_state_at_epoch(args, epoch + 1):
                    save_state_only_at_epoch(args, accelerator, epoch + 1)

            # Only sample at end of epoch if epoch-based sampling is enabled AND it's not already sampled during the last step
            # Handle epoch-end sampling
            should_sample_at_epoch_end = (
                args.sample_every_n_epochs is not None
                and args.sample_every_n_epochs > 0
                and (epoch + 1) % args.sample_every_n_epochs == 0
            )
            handle_epoch_end_sampling_with_accelerator(
                should_sample_at_epoch_end,
                last_sampled_step,
                global_step,
                sampling_manager,
                args,
                accelerator,
                epoch,
                vae,
                transformer,
                sample_parameters,
                dit_dtype,
            )
            if optimizer_train_fn:
                optimizer_train_fn()

            # Handle epoch-end validation
            should_validate_on_epoch_end = getattr(args, "validate_on_epoch_end", False)
            handle_epoch_end_validation(
                should_validate_on_epoch_end,
                val_dataloader,
                last_validated_step,
                global_step,
                self.validation_core,
                val_epoch_step_sync,
                current_epoch,
                epoch,
                args,
                accelerator,
                transformer,
                noise_scheduler,
                control_signal_processor,
                vae,
            )

            # Trigger epoch_end junction event
            trigger_event(
                "epoch_end",
                args=args,
                accelerator=accelerator,
                epoch=epoch + 1,
                global_step=global_step,
                network=network,
                loss_recorder=loss_recorder,
            )

            # end of epoch

        return global_step, network
