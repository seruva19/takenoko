"""Refactored WAN network trainer orchestrator.

This is the main orchestrator class that coordinates all the refactored training components.
Much cleaner and more maintainable than the original monolithic implementation.
"""

import argparse
import math
import os
import random
import time
from multiprocessing import Value
from typing import Any, Optional
import torch
from tqdm import tqdm
from accelerate.utils import set_seed

import utils.fluxflow_augmentation as fluxflow_augmentation
from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import (
    collator_class,
    prepare_accelerator,
    clean_memory_on_device,
    should_sample_images,
    LossRecorder,
)
from utils import model_utils
from wan.configs.config import WAN_CONFIGS
from utils.tread.tread_router import TREADRouter

# Import all our refactored components
from core.trainer_config import TrainerConfig
from core.optimizer_manager import OptimizerManager
from core.model_manager import ModelManager
from core.sampling_manager import SamplingManager
from core.control_signal_processor import ControlSignalProcessor
from core.checkpoint_manager import CheckpointManager
from core.training_core import TrainingCore
from memory.safe_memory_manager import SafeMemoryManager

from core.vae_training_core import VaeTrainingCore
from reward.reward_training_core import RewardTrainingCore
from enhancements.repa.repa_helper import RepaHelper
from scheduling.timestep_utils import (
    initialize_timestep_distribution,
    get_noisy_model_input_and_timesteps,
)
from energy_based.eqm_mode.config import EqMModeConfig
from energy_based.eqm_mode.energy import register_energy_head_metadata
import logging
from common.logger import get_logger
from common.performance_logger import snapshot_gpu_memory

logger = get_logger(__name__, level=logging.INFO)


class WanNetworkTrainer:
    """Main orchestrator class for WAN network training using refactored components."""

    def __init__(self):
        self.blocks_to_swap = None
        self.fluxflow_config = {}
        self.config = None
        # Store original config file content for saving with training states
        self.original_config_content = None
        self.original_config_path = None

        # Initialize all component managers
        self.trainer_config = TrainerConfig()
        self.optimizer_manager = OptimizerManager()
        self.model_manager = ModelManager()
        self.sampling_manager = None  # Will be initialized with config
        self.control_signal_processor = ControlSignalProcessor()
        self.checkpoint_manager = CheckpointManager()
        self.training_core = None  # Will be initialized with config
        self.vae_training_core = None  # Will be initialized for VAE training
        self.eqm_mode_config: Optional[EqMModeConfig] = None

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle model-specific arguments and configuration."""
        self.pos_embed_cache = {}
        self.config = WAN_CONFIGS[args.task]

        # Call model manager's handle_model_specific_args to handle downloads
        self.model_manager.handle_model_specific_args(args)

        # Get the dit_dtype from the model manager after downloads
        dit_dtype = self.model_manager.get_dit_dtype()

        if dit_dtype == torch.float16:
            assert args.mixed_precision in [
                "fp16",
                "no",
            ], "DiT weights are in fp16, mixed precision must be fp16 or no"
        elif dit_dtype == torch.bfloat16:
            assert args.mixed_precision in [
                "bf16",
                "no",
            ], "DiT weights are in bf16, mixed precision must be bf16 or no"

        if args.fp8_scaled and dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights"
            )

        # dit_dtype cannot be fp8, so we select the appropriate dtype
        if dit_dtype.itemsize == 1:
            dit_dtype = (
                torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            )

        args.dit_dtype = model_utils.dtype_to_str(dit_dtype)

        self.default_guidance_scale = 1.0  # not used
        self.fluxflow_config = fluxflow_augmentation.get_fluxflow_config_from_args(args)

        # Initialize training cores with config
        self.training_core = TrainingCore(self.config, self.fluxflow_config)
        self.vae_training_core = VaeTrainingCore(self.config)
        self.reward_training_core = RewardTrainingCore(self.config)

        if getattr(args, "enable_eqm_mode", False):
            self.eqm_mode_config = EqMModeConfig.from_args(args)
            args.eqm_mode_config = self.eqm_mode_config
            logger.info(
                "EqM mode enabled (prediction=%s, path=%s)",
                self.eqm_mode_config.prediction,
                self.eqm_mode_config.path_type,
            )
        else:
            self.eqm_mode_config = None

        # Configure advanced logging settings (progress bar, parameter stats, etc.)
        # This ensures TOML config values are actually used instead of hardcoded defaults
        self.training_core.configure_advanced_logging(args)

        # Re-initialize memory manager with training args so flags are honored
        try:
            self.training_core.memory_manager = SafeMemoryManager(args.__dict__)
        except Exception:
            pass

        # Initialize adaptive timestep sampling if available
        if hasattr(args, "enable_adaptive_timestep_sampling"):
            try:
                self.training_core.initialize_adaptive_timestep_sampling(args)
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive timestep sampling: {e}")

        # Initialize masked training if available
        try:
            self.training_core.loss_computer.initialize_masked_training(args)
        except Exception as e:
            logger.warning(f"Failed to initialize masked training: {e}")

        # Initialize temporal consistency enhancement if available
        try:
            self.training_core.initialize_temporal_consistency_enhancement(args)
        except Exception as e:
            logger.warning(
                f"Failed to initialize temporal consistency enhancement: {e}"
            )

        # Initialize slider training based on network module
        try:
            from enhancements.slider.slider_integration import (
                initialize_slider_integration,
            )

            initialize_slider_integration(args)
        except Exception as e:
            logger.warning(f"Failed to initialize slider training: {e}")

        # Initialize sampling manager now that we have config
        self.sampling_manager = SamplingManager(
            self.config, self.default_guidance_scale
        )

    def show_timesteps(self, args: argparse.Namespace) -> None:
        """Show timesteps distribution for debugging purposes."""
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=args.discrete_flow_shift, reverse=True, solver="euler"
        )

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1, 1, dtype=torch.float16)
        noise = torch.ones_like(latents)

        # sample timesteps
        sampled_timesteps = [0] * 1000  # Use fixed size instead of config access
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            # we use noise=1, so returned noisy_model_input is same as timestep
            # Initialize timestep distribution if needed

            # Ensure training_core is initialized
            if self.training_core is None:
                raise RuntimeError(
                    "Training core not initialized. Call handle_model_specific_args first."
                )

            initialize_timestep_distribution(
                args, self.training_core.timestep_distribution
            )

            actual_timesteps, _, _ = get_noisy_model_input_and_timesteps(
                args,
                noise,
                latents,
                noise_scheduler,
                torch.device("cpu"),
                torch.float16,
                self.training_core.timestep_distribution,
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                if 0 <= t < len(sampled_timesteps):
                    sampled_timesteps[t] += 1

        # sample weighting
        sampled_weighting = [0] * 1000  # Use fixed size
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            from utils.train_utils import compute_loss_weighting_for_sd3

            weighting = compute_loss_weighting_for_sd3(
                args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16
            )
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()  # type: ignore

        # show results
        if args.show_timesteps == "image":
            # Recompute using shared helper for consistency
            try:
                from scheduling.timestep_utils import (
                    compute_sampled_timesteps_and_weighting,
                )

                # Re-assert training_core is available for static typing
                training_core = self.training_core
                if training_core is None:
                    raise RuntimeError(
                        "Training core not initialized. Call handle_model_specific_args first."
                    )
                sampled_timesteps, sampled_weighting = (
                    compute_sampled_timesteps_and_weighting(
                        args,
                        training_core.timestep_distribution,
                        noise_scheduler,
                        num_samples=100000,
                        batch_size=1000,
                    )
                )
            except Exception:
                pass

            # show timesteps with matplotlib (non-interactive backend)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            # Also log this figure to TensorBoard if configured
            try:
                log_dir = getattr(args, "logging_dir", None)
                log_with = str(getattr(args, "log_with", "tensorboard")).lower()
                if log_dir and log_with in ("tensorboard", "all"):
                    os.makedirs(log_dir, exist_ok=True)
                    try:
                        from tensorboardX import SummaryWriter as _SummaryWriter  # type: ignore
                    except ImportError:
                        from torch.utils.tensorboard.writer import SummaryWriter as _SummaryWriter  # type: ignore
                    fig = plt.gcf()
                    writer = _SummaryWriter(log_dir=log_dir)
                    try:
                        writer.add_figure(
                            "timestep/show_timesteps_chart", fig, global_step=0
                        )
                        logger.info(
                            "Logged timestep distribution figure to TensorBoard"
                        )
                    finally:
                        writer.close()
            except Exception as _tb_err:
                logger.debug(
                    f"Failed to log show_timesteps figure to TensorBoard: {_tb_err}"
                )
            plt.show()

        else:
            import numpy as np

            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            # average per line
            sampled_timesteps = sampled_timesteps.reshape(
                -1, N_TIMESTEPS_PER_LINE
            ).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(
                -1, N_TIMESTEPS_PER_LINE
            ).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = (
                    f"{(i)*N_TIMESTEPS_PER_LINE:4d}-{(i+1)*N_TIMESTEPS_PER_LINE-1:4d}: "
                )
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i*N_TIMESTEPS_PER_LINE:4d}-{(i+1)*N_TIMESTEPS_PER_LINE-1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def train(self, args: argparse.Namespace) -> None:
        """Main training orchestration method"""

        trace_memory: bool = bool(getattr(args, "trace_memory", False))
        if trace_memory:
            snapshot_gpu_memory("train/start")

        # ========== Validation and Setup ==========
        # Check required arguments
        if args.dataset_config is None:
            raise ValueError("dataset_config is required")
        if args.dit is None:
            raise ValueError("path to DiT model is required")
        if args.output_dir is None or not args.output_dir.strip():
            raise ValueError("output_dir is required and cannot be empty")
        if args.output_name is None or not args.output_name.strip():
            raise ValueError("output_name is required and cannot be empty")
        if args.log_tracker_config is not None and not args.log_tracker_config.strip():
            logger.warning(f"log_tracker_config is empty, setting to None")
            args.log_tracker_config = None
        assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base"

        if args.sage_attn:
            raise ValueError(
                "SageAttention doesn't support training currently. Please use `--sdpa` or `--xformers` etc. instead."
            )

        if args.fp16_accumulation:
            logger.info("Enabling FP16 accumulation")
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                logger.warning(
                    "💡 Note: fp16 accumulation may degrade training quality"
                )
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                logger.warning(
                    "🚨 FP16 accumulation not available, requires at least PyTorch 2.7.0"
                )

        # Handle model-specific arguments
        self.handle_model_specific_args(args)

        # Show timesteps for debugging if requested
        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Create shared epoch/step counters BEFORE dataset creation
        # These are used for multiprocessing-safe epoch tracking and bucket shuffling
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args)

        # Conflict handling: prefer precomputed timesteps when both are requested
        try:
            if (
                getattr(args, "use_precomputed_timesteps", False)
                and getattr(args, "num_timestep_buckets", None) is not None
            ):
                logger.warning(
                    "💡 Both use_precomputed_timesteps and num_timestep_buckets are set; "
                    "preferring precomputed distribution and ignoring per-epoch timestep buckets."
                )
        except Exception:
            pass

        need_pixels_for_alignment = bool(
            getattr(args, "enable_control_lora", False)
            or getattr(args, "sara_enabled", False)
            or getattr(args, "enable_repa", False)
        )

        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.train_dataset_group,
            training=True,
            load_pixels_for_batches=need_pixels_for_alignment,
            prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
            num_timestep_buckets=(
                None
                if getattr(args, "use_precomputed_timesteps", False)
                else getattr(args, "num_timestep_buckets", None)
            ),
            shared_epoch=current_epoch,  # NEW: Pass shared epoch counter
        )

        # Log regularization information
        from utils.regularization_utils import (
            log_regularization_info,
            validate_regularization_config,
        )

        log_regularization_info(train_dataset_group)
        validate_regularization_config(args)

        # Only create validation dataset group if there are validation datasets
        val_dataset_group = None
        val_current_epoch = None
        val_current_step = None
        if len(blueprint.val_dataset_group.datasets) > 0:
            # Create separate epoch/step tracking for validation to prevent cross-contamination
            val_current_epoch = Value("i", 0)
            val_current_step = Value("i", 0)

            # For validation, we might want pixels available in batches without enabling Control LoRA.
            # When args.load_val_pixels is True, piggyback on the dataset preparation flag
            # that loads original pixels for control processing.
            val_need_pixels = bool(
                getattr(args, "enable_control_lora", False)
                or getattr(args, "sara_enabled", False)
                or getattr(args, "enable_repa", False)
                or getattr(args, "load_val_pixels", False)
            )

            val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
                blueprint.val_dataset_group,
                training=True,
                load_pixels_for_batches=val_need_pixels,
                prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
                num_timestep_buckets=(
                    None
                    if getattr(args, "use_precomputed_timesteps", False)
                    else getattr(args, "num_timestep_buckets", None)
                ),
                shared_epoch=val_current_epoch,  # NEW: Use separate shared_epoch for validation
            )

        # Optionally wrap with self-correction hybrid group (delegated to helper)
        try:
            from enhancements.self_correction.setup import (
                maybe_wrap_with_self_correction,
            )

            train_dataset_group = maybe_wrap_with_self_correction(
                args,
                blueprint_generator,
                user_config,
                self.config,
                train_dataset_group,
            )
        except Exception as _sc_wrap_err:
            logger.warning(f"Self-correction hybrid setup skipped: {_sc_wrap_err}")

        # Setup latent quality analysis if enabled (actual analysis runs later with TensorBoard or here if TB disabled)
        from dataset.latent_quality_analyzer import (
            setup_latent_quality_for_trainer,
            run_dataset_analysis_for_trainer,
        )

        if setup_latent_quality_for_trainer(args):
            # Only run initial analysis if TensorBoard logging is disabled
            if not getattr(args, "latent_quality_tensorboard", True):
                run_dataset_analysis_for_trainer(args, train_dataset_group)

        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        # ========== Accelerator Setup ==========
        # Reset Accelerator state to avoid errors if an Accelerator was created earlier in the same Python process (e.g. during latent/text caching)
        try:
            from accelerate.state import AcceleratorState  # type: ignore

            # Private API – safe to use here because we are in a controlled environment
            AcceleratorState._reset_state()  # pylint: disable=protected-access
            logger.debug("Accelerator state reset successfully")
        except Exception as reset_err:  # pragma: no cover
            # If reset fails, we continue; an error will surface when creating Accelerator if truly incompatible
            logger.debug(f"Unable to reset Accelerator state: {reset_err}")

        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if trace_memory:
            snapshot_gpu_memory("train/after_accelerator")

        # ========== Precision Setup ==========
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        dit_dtype = (
            torch.bfloat16
            if args.dit_dtype is None
            else model_utils.str_to_dtype(args.dit_dtype)
        )
        dit_weight_dtype = (
            (None if args.fp8_scaled else torch.float8_e4m3fn)
            if args.fp8_base
            else dit_dtype
        )
        logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

        vae_dtype = (
            torch.float16
            if args.vae_dtype is None
            else model_utils.str_to_dtype(args.vae_dtype)
        )

        # ========== Model Loading ==========
        sample_parameters = None
        vae = None

        # Load VAE when it will be needed ------------------------------------------------
        need_vae_for_control = (
            hasattr(args, "enable_control_lora") and args.enable_control_lora
        )
        need_vae_for_sampling = args.sample_prompts is not None

        if need_vae_for_sampling and self.sampling_manager is not None:
            sample_parameters = self.sampling_manager.process_sample_prompts(
                args, accelerator, args.sample_prompts
            )

        # Decide VAE loading strategy
        if need_vae_for_control:
            # Hard fail early if Control LoRA is enabled without a VAE path
            if not getattr(args, "vae", None) or not str(args.vae).strip():
                raise ValueError(
                    "Control LoRA requires a VAE checkpoint. Set 'vae' in the config when enable_control_lora is True."
                )

            # Control-LoRA requires an actual VAE during training – load it now.
            vae = self.model_manager.load_vae(
                args, vae_dtype=vae_dtype, vae_path=args.vae
            )
            if vae is None:
                raise RuntimeError(
                    "Failed to load VAE for Control LoRA. Please verify the 'vae' path in the config."
                )
            vae.requires_grad_(False)
            vae.eval()
            # Expose to control-signal processor so it can encode control latents
            # Store on processor for later use without breaking static typing
            setattr(self.control_signal_processor, "vae", vae)
        else:
            # We may still need a VAE later for sampling; configure lazy load
            vae = None

        # Provide VAE config to SamplingManager for lazy loading when necessary
        self._vae_config = {
            "args": args,
            "vae_dtype": vae_dtype,
            "vae_path": args.vae,
        }
        if self.sampling_manager is not None:
            self.sampling_manager.set_vae_config(self._vae_config)

        # Load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        attn_mode = self.model_manager.get_attention_mode(args)
        transformer, dual_model_manager = self.model_manager.load_transformer(
            accelerator,
            args,
            args.dit,
            attn_mode,
            args.split_attn,
            loading_device,
            dit_weight_dtype,
            self.config,
        )
        if trace_memory:
            snapshot_gpu_memory("train/after_transformer_load")

        transformer.eval()
        transformer.requires_grad_(False)

        if self.eqm_mode_config and self.eqm_mode_config.energy_head:
            register_energy_head_metadata(
                transformer, mode=self.eqm_mode_config.energy_mode
            )
            logger.info(
                "EqM energy head enabled (mode=%s)", self.eqm_mode_config.energy_mode
            )

        # Configure TREAD routing if enabled and routes provided
        if getattr(args, "enable_tread", False):
            tread_cfg = getattr(args, "tread_config", None)
            routes = tread_cfg.get("routes") if isinstance(tread_cfg, dict) else None
            if routes and len(routes) > 0:
                try:
                    router = TREADRouter(
                        seed=getattr(args, "seed", 42) or 42,
                        device=accelerator.device,
                    )
                    # set on the raw module (not wrapped yet)
                    transformer.set_router(router, routes)  # type: ignore
                    # Store tread mode on the model for runtime gating
                    try:
                        setattr(
                            transformer,
                            "_tread_mode",
                            getattr(args, "tread_mode", "full"),
                        )
                        setattr(
                            transformer,
                            "row_tread_auto_fallback",
                            bool(getattr(args, "row_tread_auto_fallback", True)),
                        )
                    except Exception:
                        pass

                    logger.info("🛣️ TREAD routing enabled with %d route(s)", len(routes))
                except Exception as e:
                    logger.warning(f"Failed to enable TREAD routing: {e}")
            else:
                logger.info(
                    "enable_tread is True but no routes configured; TREAD disabled"
                )

        if blocks_to_swap > 0:
            logger.info(
                f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}"
            )
            transformer.enable_block_swap(
                blocks_to_swap,
                accelerator.device,
                supports_backward=True,
                config_args=args,
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # ========== Network Setup ==========
        network = self.model_manager.create_network(
            args, transformer, vae, self.control_signal_processor
        )
        controlnet = getattr(self.model_manager, "controlnet", None)
        if network is None:
            return

        # ========== Verbose Network Information ==========
        if getattr(args, "verbose_network", False):
            self._log_detailed_network_info(network, transformer, args)

        # ========== Optimizer and Scheduler Setup ==========
        # Check if we're using Lycoris network (which doesn't accept input_lr_scale)
        # Lycoris networks have a different prepare_optimizer_params signature that doesn't
        # include input_lr_scale parameter, unlike our custom network implementations
        is_lycoris_network = args.network_module == "lycoris.kohya"

        if is_lycoris_network:
            trainable_params, lr_descriptions = network.prepare_optimizer_params(
                unet_lr=args.learning_rate,
            )
        else:
            trainable_params, lr_descriptions = network.prepare_optimizer_params(
                unet_lr=args.learning_rate,
                input_lr_scale=getattr(args, "input_lr_scale", 1.0),
            )

        # If ControlNet is enabled, append its optimizer params
        if (
            hasattr(args, "enable_controlnet")
            and args.enable_controlnet
            and controlnet is not None
        ):
            cn_params, cn_desc = controlnet.prepare_optimizer_params(
                unet_lr=args.learning_rate
            )
            if cn_params:
                trainable_params.extend(cn_params)
                lr_descriptions.extend(cn_desc)

        # Add patch embedding parameters for control LoRA, ONLY if it's enabled.
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            logger.info("Adding patch embedding parameters for control LoRA")
            if hasattr(transformer, "patch_embedding"):
                patch_params = list(transformer.patch_embedding.parameters())
                if patch_params:
                    # For Control LoRA, the patch_embedding layer is replaced and should be trained
                    trainable_params.append(
                        {
                            "params": patch_params,
                            "lr": args.learning_rate
                            * getattr(args, "input_lr_scale", 1.0),
                        }
                    )
                    lr_descriptions.append("patch_embedding")

        (
            optimizer_name,
            optimizer_args,
            optimizer,
            optimizer_train_fn,
            optimizer_eval_fn,
        ) = self.optimizer_manager.get_optimizer(args, transformer, trainable_params)

        # ========== DataLoader Setup ==========
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        val_dataloader = None
        if val_dataset_group is not None:
            # Validation epoch/step tracking already created during dataset initialization
            val_collator = collator_class(
                val_current_epoch, val_current_step, ds_for_collator
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collator,
                num_workers=args.max_data_loader_n_workers,
            )

        # ========== Training Parameters ==========
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        lr_scheduler = self.optimizer_manager.get_lr_scheduler(
            args, optimizer, accelerator.num_processes
        )

        # ========== Model Preparation ==========
        network_dtype = torch.float32
        args.full_fp16 = args.full_bf16 = False

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            logger.info(f"casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        if blocks_to_swap > 0:
            transformer = accelerator.prepare(
                transformer, device_placement=[not blocks_to_swap > 0]
            )
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(
                accelerator.device
            )
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)
        if trace_memory:
            snapshot_gpu_memory("train/after_prepare_transformer")

        if controlnet is not None:
            network, controlnet, optimizer, train_dataloader, lr_scheduler = (
                accelerator.prepare(
                    network, controlnet, optimizer, train_dataloader, lr_scheduler
                )
            )
            # update prepared instance back to model_manager
            self.model_manager.controlnet = controlnet
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
        training_model = network

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        # ========== Checkpoint Hooks ==========
        self.checkpoint_manager.register_hooks(accelerator, args, transformer, network)

        # Resume from checkpoint if specified
        restored_step = self.checkpoint_manager.resume_from_local_if_specified(
            accelerator, args, transformer, self.control_signal_processor
        )

        # ========== Training Setup ==========
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator.print(f"🚀 Starting WAN Network Training")
        accelerator.print(f"📊 Training Configuration:")
        accelerator.print(
            f"   • Total training items: {train_dataset_group.num_train_items:,}"
        )
        accelerator.print(f"   • Batches per epoch: {len(train_dataloader):,}")
        accelerator.print(f"   • Number of epochs: {num_train_epochs:,}")
        accelerator.print(f"   • Total optimization steps: {args.max_train_steps:,}")
        accelerator.print(
            f"   • Gradient accumulation steps: {args.gradient_accumulation_steps}"
        )
        accelerator.print(
            f"   • Effective batch size: {args.gradient_accumulation_steps * sum(d.batch_size for d in train_dataset_group.datasets):,}"
        )
        accelerator.print(
            f"   • Batch sizes per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )

        accelerator.print(f"⚙️  Optimizer & Learning Rate:")
        accelerator.print(f"   • Optimizer: {optimizer_name}")
        if optimizer_args:
            accelerator.print(f"   • Optimizer args: {optimizer_args}")
        accelerator.print(f"   • Base learning rate: {args.learning_rate:.2e}")
        if hasattr(args, "lr_scheduler") and args.lr_scheduler:
            accelerator.print(f"   • LR scheduler: {args.lr_scheduler}")

        accelerator.print(f"🔧 Model Configuration:")
        accelerator.print(f"   • Model dtype: {dit_dtype}")
        accelerator.print(
            f"   • Gradient checkpointing: {'enabled' if args.gradient_checkpointing else 'disabled'}"
        )
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            accelerator.print(f"   • Control LoRA: enabled")
        if blocks_to_swap > 0:
            accelerator.print(f"   • Block swapping: {blocks_to_swap} blocks")

        accelerator.print(f"💾 Checkpoint & Logging:")
        accelerator.print(f"   • Save every {args.save_every_n_steps:,} steps")
        if hasattr(args, "sample_every_n_steps") and args.sample_every_n_steps:
            accelerator.print(f"   • Sample every {args.sample_every_n_steps:,} steps")
        if hasattr(args, "log_every_n_steps") and args.log_every_n_steps:
            accelerator.print(f"   • Log every {args.log_every_n_steps:,} steps")

        # ========== Metadata Setup ==========
        metadata = self.trainer_config.create_training_metadata(
            args,
            session_id,
            training_started_at,
            train_dataset_group,
            num_train_epochs,
            len(train_dataloader),
            optimizer_name,
            optimizer_args,
        )

        minimum_metadata = {}
        from utils.train_utils import TAKENOKO_METADATA_MINIMUM_KEYS

        for key in TAKENOKO_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # ========== Tracker Setup ==========
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None and args.log_tracker_config.strip():
                import toml

                try:
                    init_kwargs = toml.load(args.log_tracker_config)
                except Exception:
                    init_kwargs = {}
            try:
                accelerator.init_trackers(
                    (
                        "network_train"
                        if args.log_tracker_name is None
                        else args.log_tracker_name
                    ),
                    config=self.trainer_config.get_sanitized_config_or_none(args),
                    init_kwargs=init_kwargs,
                )
            except Exception as e:
                # Continue without trackers
                logger.warning(
                    f"⚠️  Tracker initialization failed, continuing without logging: {e}"
                )
                pass

            # Non-intrusive registration of TensorBoard metric descriptions
            try:
                from utils.tensorboard_utils import (
                    get_default_metric_descriptions,
                    register_metric_descriptions_non_intrusive,
                )

                tag_to_desc = get_default_metric_descriptions()
                register_metric_descriptions_non_intrusive(
                    accelerator, args, tag_to_desc
                )
            except Exception:
                logger.warning(
                    "⚠️  TensorBoard metric descriptions registration failed, continuing without metric descriptions."
                )
                pass

            # Run latent quality analysis with TensorBoard logging after trackers are ready
            try:
                from dataset.latent_quality_analyzer import (
                    run_tensorboard_analysis_for_trainer,
                )

                run_tensorboard_analysis_for_trainer(
                    args, train_dataset_group, accelerator, 0
                )
            except Exception as e:
                logger.warning(f"❌ Latent quality TensorBoard integration failed: {e}")
                import traceback

                logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Log dataset statistics to TensorBoard (only on first run, not on resume)
            try:
                from utils.dataset_stats_logger import log_dataset_stats_to_tensorboard

                # Note: global_step will be set after this block, so we check restored_step
                initial_step = restored_step if restored_step is not None else 0
                log_dataset_stats_to_tensorboard(
                    accelerator,
                    train_dataset_group,
                    val_dataset_group,
                    global_step=initial_step,
                )
            except Exception as e:
                logger.warning(f"Failed to log dataset stats to TensorBoard: {e}")

        # ========== Training Loop Setup ==========
        global_step = restored_step if restored_step is not None else 0
        # Customize tqdm bar format when enhanced progress bar is enabled to avoid
        # the default rate field's wide alignment ("s/it" spacing). We keep ETA
        # and elapsed, and move performance metrics to postfix via set_postfix.
        try:
            use_enhanced_bar = bool(getattr(args, "enhanced_progress_bar", True))
        except Exception:
            use_enhanced_bar = True
        custom_bar_format = (
            "{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            if use_enhanced_bar
            else None
        )
        progress_bar = tqdm(
            range(args.max_train_steps),
            initial=global_step,  # Ensure progress bar resumes at correct step
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="steps",
            dynamic_ncols=True,
            bar_format=custom_bar_format,
        )

        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=args.discrete_flow_shift, reverse=True, solver="euler"
        )

        loss_recorder = LossRecorder()
        del train_dataset_group

        # Create save/remove model functions
        save_model = self.checkpoint_manager.create_save_model_function(
            args, metadata, minimum_metadata, dit_dtype
        )
        remove_model = self.checkpoint_manager.create_remove_model_function(args)

        # Prepare validation epoch/step sync objects for training core
        val_epoch_step_sync = None
        if val_dataloader is not None:
            val_epoch_step_sync = (val_current_epoch, val_current_step)

        # ========== Initial Sampling ==========
        # Only run initial sampling if:
        # 1. We should sample at this step, AND
        # 2. We're NOT resuming from a step where sampling would have already occurred
        should_do_initial_sampling = should_sample_images(args, global_step, epoch=0)
        if restored_step is not None and should_do_initial_sampling:
            # We're resuming and sampling would occur - check if sampling happened before save
            # If save_every_n_steps matches sample_every_n_steps and we're at that step,
            # sampling would have occurred before the checkpoint was saved
            if (
                args.save_every_n_steps is not None
                and args.sample_every_n_steps is not None
                and global_step % args.save_every_n_steps == 0
                and global_step % args.sample_every_n_steps == 0
            ):
                logger.info(
                    f"Skipping initial sampling at step {global_step} - sampling already occurred before checkpoint was saved"
                )
                should_do_initial_sampling = False

        if should_do_initial_sampling:
            try:
                if optimizer_eval_fn:
                    optimizer_eval_fn()
                # Prefer the already-loaded VAE (if any); otherwise load lazily
                sampling_vae = (
                    vae
                    if vae is not None
                    else (
                        self.sampling_manager._load_vae_lazy()
                        if self.sampling_manager
                        else None
                    )
                )
                if sampling_vae is None:
                    logger.warning("No VAE available for sampling, skipping...")
                    return
                self.sampling_manager.sample_images(  # type: ignore
                    accelerator,
                    args,
                    0,
                    global_step,
                    sampling_vae,
                    transformer,
                    sample_parameters,
                    dit_dtype,
                    dual_model_manager=dual_model_manager,
                )
                # Unload only if we loaded lazily here
                if sampling_vae is not vae and self.sampling_manager is not None:
                    self.sampling_manager._unload_vae(sampling_vae)
                if optimizer_train_fn:
                    optimizer_train_fn()
            except Exception as e:
                logger.error(f"💥 Initial sampling failed: {e}")
                raise

        if len(accelerator.trackers) > 0:
            try:
                # Ensure trackers are ready; optionally write a no-op with decorated tags
                try:
                    from utils.tensorboard_utils import (
                        apply_direction_hints_to_logs as _adh,
                    )

                    _ = _adh(args, {})
                except Exception:
                    pass
                accelerator.log({}, step=0)

            except Exception as e:
                logger.error(f"💥 Accelerator logging failed: {e}")
                raise

        logger.info(f"DiT dtype: {transformer.dtype}, device: {transformer.device}")
        clean_memory_on_device(accelerator.device)

        # Determine if this is VAE training
        is_vae_training = args.network_module == "networks.vae_wan"

        enable_reward_training = bool(getattr(args, "enable_reward_lora", False))

        if is_vae_training:
            logger.info("🎨 Starting VAE training mode")

            # Log loss type information
            from criteria.loss_factory import log_loss_type_info

            log_loss_type_info(args)

            # Run VAE training loop
            global_step, network = self.vae_training_core.run_vae_training_loop(  # type: ignore
                args=args,
                accelerator=accelerator,
                vae=vae,  # Pass the VAE as the main model to train
                network=network,
                training_model=training_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                lr_descriptions=lr_descriptions,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                network_dtype=network_dtype,
                num_train_epochs=num_train_epochs,
                global_step=global_step,
                progress_bar=progress_bar,
                metadata=metadata,
                loss_recorder=loss_recorder,
                current_epoch=current_epoch,
                current_step=current_step,
                optimizer_train_fn=optimizer_train_fn,
                optimizer_eval_fn=optimizer_eval_fn,
                save_model=save_model,
                remove_model=remove_model,
                is_main_process=is_main_process,
            )
        elif enable_reward_training:
            logger.info("🏆 Starting Reward LoRA training mode")

            # Log loss type information
            from criteria.loss_factory import log_loss_type_info

            log_loss_type_info(args)

            # Ensure VAE is available for decoding
            if vae is None:
                # Attempt lazy load (reusing SamplingManager path)
                if self.sampling_manager is not None:
                    vae = self.sampling_manager._load_vae_lazy()
                if vae is None:
                    raise ValueError(
                        "Reward training requires a VAE checkpoint (set 'vae' in config)"
                    )

            # Run reward training loop
            global_step, network = self.reward_training_core.run_reward_training_loop(
                args=args,
                accelerator=accelerator,
                transformer=transformer,
                network=network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                trainable_params=trainable_params,
                save_model=save_model,
                remove_model=remove_model,
                vae=vae,
                is_main_process=is_main_process,
                global_step=global_step,
            )
        elif getattr(args, "enable_srpo_training", False):
            # SRPO (Semantic Relative Preference Optimization) training mode
            from srpo.srpo_setup import setup_srpo_training

            srpo_trainer = setup_srpo_training(
                args=args,
                accelerator=accelerator,
                transformer=transformer,
                network=network,
                optimizer=optimizer,
                vae=vae,
                model_config=self.config,
            )

            # Run SRPO training loop
            srpo_trainer.run_training_loop()

            # SRPO handles its own checkpointing
            logger.info("✅ SRPO training completed successfully")
        else:
            logger.info("🤖 Starting DiT training mode")

            # Log loss type information
            from criteria.loss_factory import log_loss_type_info

            log_loss_type_info(args)

            # ========== SARA / REPA Helper Setup ==========
            sara_helper = None
            repa_helper = None
            if getattr(args, "sara_enabled", False):
                from enhancements.sara.sara_helper import create_sara_helper

                logger.info("SARA is enabled. Setting up the helper module.")
                sara_helper = create_sara_helper(transformer, args)
                if sara_helper is not None:
                    if hasattr(sara_helper, "configure_accelerator"):
                        sara_helper.configure_accelerator(accelerator)
                    sara_helper.setup_hooks()
                    sara_helper = accelerator.prepare(sara_helper)
            elif getattr(args, "enable_repa", False):
                from enhancements.repa.enhanced_repa_helper import EnhancedRepaHelper

                logger.info("REPA is enabled. Setting up the enhanced helper module.")
                repa_helper = EnhancedRepaHelper(transformer, args)
                repa_helper.setup_hooks()
                repa_helper = accelerator.prepare(repa_helper)

            # Run the main training loop using TrainingCore
            # Attach a self-correction manager instance if enabled so the core can call it
            try:
                from enhancements.self_correction.setup import (
                    maybe_attach_self_correction_manager,
                )

                maybe_attach_self_correction_manager(
                    args,
                    accelerator,
                    self.sampling_manager,
                    blueprint,
                    vae_dtype,
                    transformer,
                )
            except Exception as _sc_init_err:
                logger.warning(f"Self-correction manager init failed: {_sc_init_err}")

            # Store dual_model_manager in args for advanced metrics (if enabled)
            if dual_model_manager is not None:
                args.dual_model_manager = dual_model_manager

            global_step, network = self.training_core.run_training_loop(  # type: ignore
                args=args,
                accelerator=accelerator,
                transformer=transformer,
                network=network,
                controlnet=controlnet,
                training_model=training_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                lr_descriptions=lr_descriptions,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                noise_scheduler=noise_scheduler,
                network_dtype=network_dtype,
                dit_dtype=dit_dtype,
                num_train_epochs=num_train_epochs,
                global_step=global_step,
                progress_bar=progress_bar,
                metadata=metadata,
                loss_recorder=loss_recorder,
                sampling_manager=self.sampling_manager,
                checkpoint_manager=self.checkpoint_manager,
                control_signal_processor=self.control_signal_processor,
                # ControlNet is managed as part of network_module option; pass through if created later
                current_epoch=current_epoch,
                current_step=current_step,
                optimizer_train_fn=optimizer_train_fn,
                optimizer_eval_fn=optimizer_eval_fn,
                # Pass the (possibly eager-loaded) VAE so SampleManager can reuse it
                vae=vae,
                sample_parameters=sample_parameters,
                save_model=save_model,
                remove_model=remove_model,
                is_main_process=is_main_process,
                val_epoch_step_sync=val_epoch_step_sync,
                repa_helper=repa_helper if sara_helper is None else None,
                sara_helper=sara_helper,
                dual_model_manager=dual_model_manager,
            )

        if "sara_helper" in locals() and sara_helper is not None:
            sara_helper.remove_hooks()
        if "repa_helper" in locals() and repa_helper is not None:
            repa_helper.remove_hooks()

        metadata["takenoko_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        if optimizer_eval_fn:
            optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            from utils import train_utils

            train_utils.save_state_on_train_end(args, accelerator)

        if is_main_process:
            from utils import train_utils

            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(
                ckpt_name,
                network,
                global_step,
                num_train_epochs,
                force_sync_upload=True,
            )

    def _log_detailed_network_info(
        self, network: Any, transformer: Any, args: argparse.Namespace
    ) -> None:
        """Log detailed information about the LoRA network and trainable parameters."""
        logger.info("🔍 Detailed LoRA Network Information:")
        logger.info("=" * 80)

        # Count different types of parameters
        total_params = 0
        trainable_params = 0
        lora_params = 0
        patch_embedding_params = 0

        # Log network modules if available
        if hasattr(network, "unet_loras"):
            logger.info(f"📌 LoRA Modules ({len(network.unet_loras)} modules):")
            for i, lora in enumerate(network.unet_loras):
                if hasattr(lora, "lora_name") and hasattr(lora, "lora_dim"):
                    logger.info(
                        f"  {i+1:3d}: {lora.lora_name:<50} dim={lora.lora_dim}, alpha={getattr(lora, 'alpha', 'N/A')}"
                    )
                else:
                    logger.info(f"  {i+1:3d}: {type(lora).__name__}")

        # Log detailed parameter information
        logger.info("\n📊 Trainable Parameters:")
        for name, param in network.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                trainable_params += param_count

                # Categorize parameters
                if "lora" in name.lower():
                    lora_params += param_count
                    param_type = "LoRA"
                elif "patch_embedding" in name.lower():
                    patch_embedding_params += param_count
                    param_type = "Patch"
                else:
                    param_type = "Other"

                logger.info(
                    f"  📍 {name:<60} {str(param.shape):<20} {param_count:>10,} [{param_type}]"
                )
            total_params += param.numel()

        # Log patch embedding parameters from transformer if control LoRA is enabled
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            if hasattr(transformer, "patch_embedding"):
                logger.info("\n🎯 Control LoRA Patch Embedding Parameters:")
                for name, param in transformer.patch_embedding.named_parameters():
                    if param.requires_grad:
                        param_count = param.numel()
                        patch_embedding_params += param_count
                        trainable_params += param_count
                        logger.info(
                            f"  📍 patch_embedding.{name:<45} {str(param.shape):<20} {param_count:>10,} [Patch]"
                        )

        # Summary
        logger.info("\n📈 Parameter Summary:")
        logger.info(f"  Total parameters:         {total_params:>12,}")
        logger.info(f"  Trainable parameters:     {trainable_params:>12,}")
        logger.info(f"    ├─ LoRA parameters:     {lora_params:>12,}")
        logger.info(f"    └─ Patch embedding:     {patch_embedding_params:>12,}")
        logger.info(
            f"  Trainable ratio:          {trainable_params/total_params*100:>11.2f}%"
        )

        # Network configuration
        logger.info("\n⚙️  Network Configuration:")
        logger.info(
            f"  Network module:           {getattr(args, 'network_module', 'N/A')}"
        )
        logger.info(
            f"  Network dimension (rank): {getattr(args, 'network_dim', 'N/A')}"
        )
        logger.info(
            f"  Network alpha:            {getattr(args, 'network_alpha', 'N/A')}"
        )
        logger.info(
            f"  Network dropout:          {getattr(args, 'network_dropout', 'N/A')}"
        )

        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            logger.info(f"  Control LoRA enabled:     ✅")
            logger.info(
                f"  Control type:             {getattr(args, 'control_lora_type', 'N/A')}"
            )
            logger.info(
                f"  Input LR scale:           {getattr(args, 'input_lr_scale', 'N/A')}"
            )
        else:
            logger.info(f"  Control LoRA enabled:     ❌")

        logger.info("=" * 80)
