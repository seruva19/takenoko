"""WanFinetune trainer for full model fine-tuning.

This trainer performs genuine full model fine-tuning:
- Trains transformer parameters directly
- Uses advanced memory optimizations for large model training
- Implements efficient training optimizations (full_bf16, fused_backward_pass, mem_eff_save)
"""

import argparse
import math
import os
import random
from multiprocessing import Value
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from utils.train_utils import (
    collator_class,
    prepare_accelerator,
    clean_memory_on_device,
    should_save_state_at_epoch,
    should_save_state_at_step,
    LossRecorder,
)
from utils import model_utils
from wan.configs.config import WAN_CONFIGS
from utils.tread.tread_router import TREADRouter

from core.trainer_config import TrainerConfig
from core.optimizer_manager import OptimizerManager
from core.model_manager import ModelManager
from core.sampling_manager import SamplingManager
from core.control_signal_processor import ControlSignalProcessor
from core.training_core import TrainingCore
from memory.safe_memory_manager import SafeMemoryManager

from core.vae_training_core import VaeTrainingCore
from reward.reward_training_core import RewardTrainingCore
from enhancements.repa.repa_helper import RepaHelper
from tqdm import tqdm

from finetune.checkpoint_utils import CheckpointUtils
from finetune.model_saving_utils import ModelSavingUtils
from finetune.logging_utils import (
    NetworkLoggingUtils,
    TrainingProgressLogger,
    TimestepDistributionLogger,
)
from finetune.weight_dynamics_analysis import WeightDynamicsAnalyzer

from scheduling.timestep_distribution import TimestepDistribution

import logging
from common.logger import get_logger
from common.performance_logger import snapshot_gpu_memory

try:
    from utils.stochastic_rounding.main import (
        lerp_,
        add_,
        CUDA_AVAILABLE as STOCH_CUDA_AVAILABLE,
    )

    STOCHASTIC_ROUNDING_AVAILABLE = True
except ImportError:
    STOCHASTIC_ROUNDING_AVAILABLE = False
    STOCH_CUDA_AVAILABLE = False


logger = get_logger(__name__, level=logging.INFO)


class WanFinetuneTrainer:
    """
    WanFinetune trainer for full model fine-tuning.

    Key features:
    1. Trains transformer parameters DIRECTLY
    2. Does NOT use network adapters or LoRA-style approach
    3. Optimizes ALL model parameters with proper gradient flow
    4. Implements memory-efficient training for large models
    """

    def __init__(self):
        self.blocks_to_swap = None
        self.fluxflow_config = {}
        self.config = None
        # Store original config file content for saving with training states
        self.original_config_content = None
        self.original_config_path = None

        # Initialize all component managers (reuse takenoko's existing)
        self.trainer_config = TrainerConfig()
        self.optimizer_manager = OptimizerManager()
        self.model_manager = ModelManager()
        self.sampling_manager = None  # Will be initialized with config
        self.sample_parameters = (
            None  # Preprocessed sampling params (persist across steps)
        )
        self.control_signal_processor = ControlSignalProcessor()
        # NOTE: CheckpointManager is for LoRA training - we use simple accelerator.save/load_state for full finetuning
        self.checkpoint_manager = None  # Not used for full finetuning
        self.training_core = None  # Will be initialized with config
        self.vae_training_core = None  # Will be initialized for VAE training

        # Full fine-tuning optimization settings
        self.full_bf16 = False
        self.fused_backward_pass = False
        self.mem_eff_save = True
        self.mixed_precision_dtype = torch.bfloat16

        # Stochastic rounding settings
        self.use_stochastic_rounding = False
        self.stochastic_rounding_available = STOCHASTIC_ROUNDING_AVAILABLE

        # Initialize utility classes (will be properly initialized after model setup)
        self.model_saving_utils = None

        # Timestep distribution for logging (reuse existing functionality)
        self.timestep_distribution = TimestepDistribution()
        self._timestep_logging_initialized = False

        # Weight dynamics analysis for research and monitoring
        self.weight_analyzer = WeightDynamicsAnalyzer()

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle model-specific arguments for full fine-tuning."""
        self.pos_embed_cache = {}
        self.config = WAN_CONFIGS[args.task]

        # For fine-tuning, we set dtype based on training config (like Qwen Image)
        # This allows using any mixed_precision regardless of checkpoint dtype
        logger.info(
            "🎯 WAN Fine-tuning: Using training-config-based dtype (bypassing checkpoint dtype validation)"
        )

        # Full fine-tuning optimization arguments
        if hasattr(args, "fused_backward_pass"):
            self.fused_backward_pass = args.fused_backward_pass

            # Validate optimizer compatibility with fused backward pass
            if self.fused_backward_pass and hasattr(args, "optimizer_type"):
                optimizer_type = getattr(args, "optimizer_type", "adafactor").lower()
                if optimizer_type != "adafactor":
                    logger.warning(
                        f"⚠️ fused_backward_pass=true only supports Adafactor optimizer. "
                        f"Current optimizer_type='{optimizer_type}' will be ignored and Adafactor will be used."
                    )
        if hasattr(args, "full_bf16"):
            self.full_bf16 = args.full_bf16
        if hasattr(args, "mem_eff_save"):
            self.mem_eff_save = args.mem_eff_save

        # Validate FP8 compatibility with fine-tuning
        if hasattr(args, "fp8_base") or hasattr(args, "fp8_scaled"):
            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                logger.warning(
                    "⚠️ FP8 training is not supported for fine-tuning. Disabling FP8 options."
                )
                if hasattr(args, "fp8_base"):
                    args.fp8_base = False
                if hasattr(args, "fp8_scaled"):
                    args.fp8_scaled = False

        # Initialize stochastic rounding
        if hasattr(args, "use_stochastic_rounding"):
            self.use_stochastic_rounding = args.use_stochastic_rounding
        if hasattr(args, "use_stochastic_rounding_cuda"):
            self.use_stochastic_rounding_cuda = args.use_stochastic_rounding_cuda

        if (
            hasattr(self, "use_stochastic_rounding")
            and self.use_stochastic_rounding
            and not self.stochastic_rounding_available
        ):
            logger.warning("=" * 80)
            logger.warning("⚠️  STOCHASTIC ROUNDING: CUDA Extension Not Available")
            logger.warning("=" * 80)
            logger.warning(
                "You enabled use_stochastic_rounding = true but the CUDA extension is not compiled."
            )
            logger.warning(
                "Falling back to deterministic rounding (standard behavior)."
            )
            logger.warning("")
            logger.warning("To enable stochastic rounding:")
            logger.warning("  cd extensions/stochastic_rounding")
            logger.warning("  python setup.py install")
            logger.warning("")
            logger.warning(
                "Note: CUDA extension must be compiled on each machine individually!"
            )
            logger.warning("=" * 80)
            self.use_stochastic_rounding = False
        elif hasattr(self, "use_stochastic_rounding") and self.use_stochastic_rounding:
            logger.info(
                f"🎲 Stochastic rounding enabled for BF16 training "
                f"(CUDA: {'✓' if STOCH_CUDA_AVAILABLE else '✗'})"
            )

        # Initialize utility classes now that settings are configured
        self.model_saving_utils = ModelSavingUtils(
            self.mixed_precision_dtype,
            self.full_bf16,
            self.fused_backward_pass,
            self.mem_eff_save,
        )

        # Apply full_bf16 logic for memory efficiency
        if self.full_bf16:
            if args.mixed_precision != "bf16":
                raise ValueError("full_bf16 requires mixed_precision='bf16'")
            logger.info("🔥 Enable full BF16 training for memory efficiency")
            args.dit_dtype = "bfloat16"
            self.mixed_precision_dtype = torch.bfloat16
        else:
            # For fine-tuning, set dtype based on mixed_precision, not checkpoint dtype
            if args.mixed_precision == "bf16":
                args.dit_dtype = "bfloat16"
                self.mixed_precision_dtype = torch.bfloat16
                logger.info(
                    "📈 Setting dit_dtype to bfloat16 based on mixed_precision=bf16"
                )
            elif args.mixed_precision == "fp16":
                args.dit_dtype = "float16"
                self.mixed_precision_dtype = torch.float16
                logger.info(
                    "📈 Setting dit_dtype to float16 based on mixed_precision=fp16"
                )
            else:
                args.dit_dtype = "float32"
                self.mixed_precision_dtype = torch.float32
                logger.info(
                    "📈 Setting dit_dtype to float32 based on mixed_precision=no"
                )

        # Handle downloads only - call model manager just for downloads, skip dtype validation
        try:
            # Temporarily store the mixed_precision to restore it later
            original_mixed_precision = args.mixed_precision
            # Set mixed_precision to match our target dtype to bypass validation
            if hasattr(self, "mixed_precision_dtype"):
                if self.mixed_precision_dtype == torch.bfloat16:
                    args.mixed_precision = "bf16"
                elif self.mixed_precision_dtype == torch.float16:
                    args.mixed_precision = "fp16"
                else:
                    args.mixed_precision = "no"

            self.model_manager.handle_model_specific_args(args)

            # Restore original mixed_precision
            args.mixed_precision = original_mixed_precision
        except Exception as e:
            logger.warning(
                f"Model manager dtype validation failed (expected for fine-tuning): {e}"
            )
            # Continue anyway - we've already set our dtype based on training config

        NetworkLoggingUtils.log_training_configuration(args, self)

    def show_timesteps(
        self, args: argparse.Namespace, accelerator: Optional[Accelerator] = None
    ) -> None:
        """Delegate to TrainingCore.show_timesteps without duplicating logic.

        This preserves existing behavior while reusing the centralized implementation
        in TrainingCore for timestep visualization and logging.
        """
        # Initialize minimal TrainingCore and an Accelerator for device/writer
        training_core = TrainingCore(self.config, self.fluxflow_config)
        acc = accelerator if accelerator is not None else Accelerator()
        training_core.show_timesteps(acc, args)

    def prepare_transformer_for_finetuning(
        self, args: argparse.Namespace, accelerator: Accelerator
    ) -> torch.nn.Module:
        """
        Load and prepare transformer for direct fine-tuning with optional memory-efficient resume.
        """
        # Memory tracing: snapshot before model loading
        if getattr(args, "trace_memory", False):
            from common.performance_logger import snapshot_gpu_memory

            snapshot_gpu_memory("before_transformer_load")

        # Standard loading path
        return self._prepare_transformer_standard(args, accelerator)

    def _prepare_transformer_standard(
        self, args: argparse.Namespace, accelerator: Accelerator
    ) -> torch.nn.Module:
        """
        Standard transformer preparation - loads base model weights normally.
        """
        # Load transformer using takenoko's model manager
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device
        dit_dtype = model_utils.str_to_dtype(args.dit_dtype)

        # Handle BF16 conversion if enabled
        if getattr(args, "use_or_convert_bf16", False):
            from finetune.model_saving_utils import ModelSavingUtils

            args.dit = ModelSavingUtils.resolve_bf16_checkpoint(args.dit, accelerator)

        logger.info(f"Loading transformer for direct fine-tuning: {args.dit}")

        # Determine attention mode
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            attn_mode = "torch"  # Default fallback

        # Unified checkpoint and model path resolution
        from wan.modules.model import load_wan_model
        from finetune.checkpoint_utils import CheckpointUtils

        # Step 1: Resolve checkpoint path (auto-resume or explicit)
        checkpoint_path = None
        if getattr(args, "auto_resume", False) and not args.resume:
            logger.info("🔍 Auto-resume: Searching for checkpoints...")
            latest_checkpoint = CheckpointUtils._find_latest_checkpoint(args)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                args.resume = latest_checkpoint  # Set for later use
                logger.info(
                    f"🔍 Auto-resume: Found latest checkpoint: {latest_checkpoint}"
                )
            else:
                logger.info(
                    "🔍 Auto-resume: No existing checkpoints found, starting fresh training"
                )
        elif args.resume:
            checkpoint_path = args.resume

        # Step 2: Determine model path (base model or checkpoint model)
        model_path = args.dit  # Default to base model
        if getattr(args, "direct_checkpoint_loading", False) and checkpoint_path:
            # Try to use checkpoint's model file instead of base model
            import os

            checkpoint_model_files = [
                os.path.join(checkpoint_path, "model.safetensors"),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
                os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors"),
            ]

            for candidate in checkpoint_model_files:
                if os.path.exists(candidate):
                    model_path = candidate
                    logger.info(f"🔄 Direct checkpoint loading from: {model_path}")
                    break

            if model_path == args.dit:  # No checkpoint model found
                logger.warning(
                    f"⚠️ No model file found in checkpoint: {checkpoint_path}"
                )
                logger.info("🔄 Falling back to base model loading")
        elif getattr(args, "direct_checkpoint_loading", False):
            logger.warning(
                "⚠️ Direct checkpoint loading enabled but no checkpoint available"
            )
            logger.info("🔄 Using base model loading")

        # Load model (either base model or checkpoint, depending on direct_checkpoint_loading)
        transformer = load_wan_model(
            self.config,
            accelerator.device,
            model_path,
            attn_mode,
            getattr(args, "split_attn", False),
            loading_device,
            dit_dtype,
            getattr(args, "fp8_scaled", False),
            enable_memory_mapping=bool(getattr(args, "enable_memory_mapping", False)),
            enable_zero_copy_loading=bool(
                getattr(args, "enable_zero_copy_loading", False)
            ),
            enable_non_blocking_transfers=bool(
                getattr(args, "enable_non_blocking_transfers", False)
            ),
            memory_mapping_threshold=int(
                getattr(args, "memory_mapping_threshold", 10 * 1024 * 1024)
            ),
        )

        # Configure TREAD routing if enabled and routes provided
        if getattr(args, "enable_tread", False):
            # Set tread_mode on transformer for proper routing behavior
            tread_mode = getattr(args, "tread_mode", "full")
            setattr(transformer, "_tread_mode", tread_mode)
            logger.info(f"TREAD mode set to: {tread_mode}")

            tread_cfg = getattr(args, "tread_config", None)
            routes = tread_cfg.get("routes") if isinstance(tread_cfg, dict) else None
            if routes and len(routes) > 0:
                try:
                    router = TREADRouter(
                        seed=getattr(args, "seed", 42) or 42,
                        device=accelerator.device,
                    )
                    transformer.set_router(router, routes)  # type: ignore
                    logger.info("🛣️ TREAD routing enabled with %d route(s)", len(routes))
                except Exception as e:
                    logger.warning(f"Failed to enable TREAD routing: {e}")
            else:
                logger.warning(
                    "enable_tread is True but no routes configured; TREAD disabled"
                )

        # Apply block swapping if needed for memory efficiency
        if blocks_to_swap > 0:
            logger.info(
                f"Enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}"
            )
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # Enable gradient checkpointing for memory efficiency
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload
            )

        # Memory tracing: snapshot after model preparation
        if getattr(args, "trace_memory", False):
            snapshot_gpu_memory("after_transformer_prep")

        return transformer

    def load_t5_encoder(
        self, args: argparse.Namespace, accelerator: Accelerator
    ) -> torch.nn.Module:
        """
        Load T5 text encoder for fine-tuning.
        Only called when finetune_text_encoder=True.
        """
        if not hasattr(args, "t5") or args.t5 is None:
            raise ValueError(
                "T5 path must be specified when finetune_text_encoder=True"
            )

        try:
            # Import T5 encoder from WAN modules
            from wan.modules.t5 import T5EncoderModel

            logger.info(f"Loading T5 encoder from {args.t5}")
            text_encoder = T5EncoderModel.from_pretrained(args.t5)

            # Set to training mode for fine-tuning
            text_encoder.train()

            # Apply gradient checkpointing if enabled
            if getattr(args, "gradient_checkpointing", False):
                text_encoder.gradient_checkpointing_enable()
                logger.info("T5 gradient checkpointing enabled")

            logger.info("T5 text encoder loaded successfully for fine-tuning")
            return text_encoder

        except Exception as e:
            logger.error(f"Failed to load T5 encoder: {e}")
            raise

    def prepare_optimizer_params(
        self,
        transformer: torch.nn.Module,
        args: argparse.Namespace,
        text_encoder: Optional[torch.nn.Module] = None,
    ) -> List[Dict]:
        """
        Prepare optimizer parameters for full fine-tuning.
        Supports both DiT-only and DiT+T5 training based on finetune_text_encoder flag.
        """
        # Always include DiT transformer parameters
        dit_params = list(transformer.named_parameters())

        params_to_optimize = []
        param_names = []

        # DiT parameters (primary training target)
        params_to_optimize.append(
            {"params": [p for _, p in dit_params], "lr": args.learning_rate}
        )
        param_names.append([n for n, _ in dit_params])

        # Conditionally add T5 text encoder parameters
        if getattr(args, "finetune_text_encoder", False) and text_encoder is not None:
            t5_params = list(text_encoder.named_parameters())
            # Use lower learning rate for text encoder (typically 10x lower)
            t5_lr = args.learning_rate * 0.1

            params_to_optimize.append(
                {"params": [p for _, p in t5_params], "lr": t5_lr}
            )
            param_names.append([n for n, _ in t5_params])

            logger.info(f"T5 text encoder fine-tuning enabled with LR: {t5_lr}")

        # Calculate total number of trainable parameters
        n_params = 0
        for group in params_to_optimize:
            for p in group["params"]:
                n_params += p.numel()

        logger.info(f"Total number of trainable parameters: {n_params:,}")

        # Count transformer parameters
        dit_params_count = sum(p.numel() for p in params_to_optimize[0]["params"])
        logger.info(f"DiT transformer parameters being trained: {dit_params_count:,}")

        # Log layer structure information
        try:
            from core.handlers.layer_info_utils import get_layer_structure_info

            layer_info = get_layer_structure_info(transformer)
            logger.info("Layer structure being trained:")
            logger.info(f"  📊 Total layers: {layer_info['total_layers']}")
            logger.info(f"  🔧 Trainable layers: {layer_info['trainable_layers']}")
            logger.info(
                f"  📈 Layer training percentage: {layer_info['layer_percentage']:.1f}%"
            )
            logger.info(f"  🎯 Training mode: {layer_info['training_mode']}")

            # Show layer type breakdown
            if layer_info["layer_types"]:
                logger.info("  Layer type breakdown:")
                for layer_type, count in layer_info["layer_types"].items():
                    logger.info(f"    - {layer_type}: {count}")

            # Show subcomponent breakdown (for WanAttentionBlock internals)
            if layer_info.get("subcomponent_types"):
                logger.info("  Subcomponent breakdown:")
                for component_type, count in layer_info["subcomponent_types"].items():
                    logger.info(f"    - {component_type}: {count}")

            # Show parameter statistics
            logger.info(f"  ⚖️  Total parameters: {layer_info['total_parameters']:,}")
            logger.info(
                f"  🎚️  Trainable parameters: {layer_info['trainable_parameters']:,}"
            )
            logger.info(
                f"  📊 Parameter training percentage: {layer_info['parameter_percentage']:.2f}%"
            )

            if layer_info["is_full_finetune"]:
                logger.info("  🔥 FULL MODEL FINE-TUNING DETECTED")
            else:
                logger.info("  ⚡ Partial fine-tuning mode")

        except Exception as e:
            logger.warning(f"Could not extract layer structure info: {e}")
            # Fallback to showing sample parameter names
            sample_param_names = param_names[0][:5]  # Show fewer as fallback
            logger.info("Sample trainable parameter names:")
            for name in sample_param_names:
                logger.info(f"  ✅ {name}")

        if getattr(args, "finetune_text_encoder", False) and text_encoder is not None:
            t5_params_count = sum(p.numel() for p in params_to_optimize[1]["params"])
            logger.info(f"T5 text encoder parameters: {t5_params_count:,}")

        # Validate parameters and log warnings
        NetworkLoggingUtils.log_parameter_validation_info(
            dit_params_count, param_names, args, text_encoder
        )

        return params_to_optimize, param_names

    def _convert_resolutions_to_tuples(self, blueprint):
        """
        Convert resolution lists to tuples in dataset blueprints to fix unhashable type error.
        This ensures bucket resolutions can be used as dictionary keys.
        """
        # Convert train dataset resolutions
        if hasattr(blueprint, "train_dataset_group") and blueprint.train_dataset_group:
            for dataset in blueprint.train_dataset_group.datasets:
                if hasattr(dataset, "params") and hasattr(dataset.params, "resolution"):
                    if isinstance(dataset.params.resolution, list):
                        dataset.params.resolution = tuple(dataset.params.resolution)

        # Convert validation dataset resolutions if present
        if hasattr(blueprint, "val_dataset_group") and blueprint.val_dataset_group:
            for dataset in blueprint.val_dataset_group.datasets:
                if hasattr(dataset, "params") and hasattr(dataset.params, "resolution"):
                    if isinstance(dataset.params.resolution, list):
                        dataset.params.resolution = tuple(dataset.params.resolution)

    def forward_pass(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        text_encoder: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Forward pass using transformer DIRECTLY.
        This is full fine-tuning where the transformer itself is trained.

        TODO: Support both cached embeddings (efficient) and live T5 encoding (for T5 fine-tuning).
        """
        device = accelerator.device

        # Extract batch data (adapt to WAN video format) - no early conversion
        latents = batch["latents"]

        # Use batch["t5"] context for text embeddings - no early conversion
        if "t5" in batch:
            # Use T5 embeddings from batch (convert later)
            context = batch["t5"]
        elif "text_embeds" in batch:
            # Fallback to cached embeddings for compatibility (convert later)
            context = [batch["text_embeds"]]
        else:
            raise ValueError(
                "No text embeddings found in batch (expected 't5' or 'text_embeds')"
            )

        # Generate noise (device/dtype conversion later)
        noise = torch.randn_like(latents)

        # Use Takenoko's timestep generation
        from scheduling.timestep_utils import get_noisy_model_input_and_timesteps

        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=getattr(args, "discrete_flow_shift", 3.0),
            reverse=True,
            solver="euler",
        )

        # Get noisy input using Takenoko's method with timestep distribution
        noisy_model_input, timesteps, _ = get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            noise_scheduler,
            device,
            transformer.dtype,
            timestep_distribution=self.timestep_distribution,
        )

        # Convert tensors to correct device/dtype right before model call
        context = [t.to(device=device, dtype=transformer.dtype) for t in context]
        latents = latents.to(device=device, dtype=transformer.dtype)
        noisy_model_input = noisy_model_input.to(device=device, dtype=transformer.dtype)
        noise = noise.to(device=device, dtype=transformer.dtype)
        timesteps = timesteps.to(device=device, dtype=transformer.dtype)

        # Set noise scheduler for custom loss target support
        self.training_core.noise_scheduler = noise_scheduler

        # Use TrainingCore's call_dit method
        with accelerator.autocast():
            model_pred, target, _ = self.training_core.call_dit(
                args,
                accelerator,
                transformer,
                latents,
                batch,
                noise,
                noisy_model_input,
                timesteps,
                transformer.dtype,
            )

        # Loss computation with weighting
        loss = torch.nn.functional.mse_loss(
            model_pred.to(transformer.dtype), target, reduction="none"
        )

        # Apply loss weighting if configured (critical for training quality)
        weighting_scheme = getattr(args, "weighting_scheme", None)
        if weighting_scheme:
            try:
                from utils.train_utils import compute_loss_weighting_for_sd3

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme,
                    noise_scheduler,
                    timesteps,
                    device,
                    transformer.dtype,
                )
                if weighting is not None:
                    loss = loss * weighting
            except ImportError:
                logger.warning("Loss weighting not available, using uniform weighting")

        loss = loss.mean()  # mean loss over all elements in batch

        return loss

    def train(self, args: argparse.Namespace) -> None:
        """
        Main training loop for full fine-tuning.
        """
        # Dataset and accelerator setup
        if args.dataset_config is None:
            raise ValueError("dataset_config is required")
        if args.dit is None:
            raise ValueError("path to DiT model is required")

        # Set seed
        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Validate training configuration and provide helpful warnings
        from common.training_validators import validate_training_config

        validate_training_config(args)

        # Handle model-specific arguments
        self.handle_model_specific_args(args)

        # Prepare accelerator for distributed training
        logger.info("Preparing accelerator")
        accelerator = prepare_accelerator(args)

        # Show timesteps for debugging if requested (use existing accelerator)
        if args.show_timesteps:
            self.show_timesteps(args, accelerator)
            return

        # Initialize training cores with config
        self.training_core = TrainingCore(self.config, self.fluxflow_config)
        self.vae_training_core = VaeTrainingCore(self.config)
        self.reward_training_core = RewardTrainingCore(self.config)

        # Configure advanced logging settings
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

        # Create shared epoch/step counters BEFORE dataset creation
        # These are used for multiprocessing-safe epoch tracking and bucket shuffling
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)

        # Load dataset configuration
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)

        blueprint = blueprint_generator.generate(user_config, args)

        # Convert resolution lists to tuples in blueprint before dataset generation
        self._convert_resolutions_to_tuples(
            blueprint
        )  # TODO: find out why this is needed only for finetuning

        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.train_dataset_group,
            training=True,
            load_pixels_for_batches=getattr(args, "enable_control_lora", False),
            prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
            num_timestep_buckets=(
                None
                if getattr(args, "use_precomputed_timesteps", False)
                else getattr(args, "num_timestep_buckets", None)
            ),
            shared_epoch=current_epoch,  # NEW: Pass shared epoch counter
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError("No training items found in the dataset")

        # Log regularization information
        from utils.regularization_utils import (
            log_regularization_info,
            validate_regularization_config,
        )

        log_regularization_info(train_dataset_group)
        validate_regularization_config(args)

        # Setup and run latent quality analysis if enabled
        from dataset.latent_quality_analyzer import (
            setup_latent_quality_for_trainer,
            run_dataset_analysis_for_trainer,
        )

        if setup_latent_quality_for_trainer(args):
            # Only run initial analysis if TensorBoard logging is disabled
            if not getattr(args, "latent_quality_tensorboard", True):
                run_dataset_analysis_for_trainer(args, train_dataset_group)

        # Handle validation dataset if available
        val_dataset_group = None
        val_current_epoch = None
        val_current_step = None
        if (
            hasattr(blueprint, "val_dataset_group")
            and blueprint.val_dataset_group is not None
            and len(blueprint.val_dataset_group.datasets) > 0
        ):
            logger.info("Loading validation dataset")
            # Create separate epoch/step tracking for validation to prevent cross-contamination
            val_current_epoch = Value("i", 0)
            val_current_step = Value("i", 0)

            val_enable_control_lora = bool(getattr(args, "enable_control_lora", False))
            if bool(getattr(args, "load_val_pixels", False)):
                val_enable_control_lora = True

            val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
                blueprint.val_dataset_group,
                training=True,
                load_pixels_for_batches=val_enable_control_lora,
                prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
                num_timestep_buckets=(
                    None
                    if getattr(args, "use_precomputed_timesteps", False)
                    else getattr(args, "num_timestep_buckets", None)
                ),
                shared_epoch=val_current_epoch,  # NEW: Use separate shared_epoch for validation
            )

        # Create validation dataloader if validation dataset exists
        val_dataloader = None
        val_epoch_step_sync = None
        if val_dataset_group is not None:
            # Validation epoch/step tracking already created during dataset initialization
            val_epoch_step_sync = (val_current_epoch, val_current_step)

            # Use the same collator class as training
            val_ds_for_collator = (
                val_dataset_group if args.max_data_loader_n_workers == 0 else None
            )
            val_collator = collator_class(
                val_current_epoch, val_current_step, val_ds_for_collator
            )

            import torch.utils.data

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collator,
                num_workers=args.max_data_loader_n_workers,
            )

        # Prepare data collator
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        # accelerator is already prepared above

        if args.mixed_precision is None:
            # Convert accelerator mixed_precision to string format if needed
            accel_mixed_precision = accelerator.mixed_precision
            if accel_mixed_precision is None:
                args.mixed_precision = "no"
            else:
                args.mixed_precision = str(accel_mixed_precision).lower()
            logger.info(f"Mixed precision set to {args.mixed_precision}")

        # Load transformer for direct fine-tuning
        transformer = self.prepare_transformer_for_finetuning(args, accelerator)

        # Conditionally load T5 text encoder for fine-tuning
        text_encoder = None
        if getattr(args, "finetune_text_encoder", False):
            logger.info("Loading T5 text encoder for fine-tuning")
            text_encoder = self.load_t5_encoder(args, accelerator)

        # Prepare optimizer parameters for full fine-tuning
        params_to_optimize, param_names = self.prepare_optimizer_params(
            transformer, args, text_encoder
        )

        # Log detailed network information if verbose mode is enabled
        if getattr(args, "verbose_network", False):
            NetworkLoggingUtils.log_detailed_network_info(transformer, args)

        # Create optimizer directly
        logger.info(f"Creating Adafactor optimizer directly")

        # Extract parameters for direct optimizer creation
        trainable_params = params_to_optimize[0]["params"]

        # Create Adafactor optimizer directly
        import transformers.optimization

        optimizer_kwargs = {
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
        }

        optimizer = transformers.optimization.Adafactor(
            trainable_params, lr=args.learning_rate, **optimizer_kwargs
        )

        # Set optimizer metadata for logging
        optimizer_name = "transformers.optimization.Adafactor"
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        logger.info(
            f"✅ Created Adafactor optimizer: {optimizer_name} | {optimizer_args}"
        )
        # Fused backward pass will be applied AFTER accelerator.prepare()

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

        # Prepare data loader
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # Initialize loss recorder for proper loss tracking
        loss_recorder = LossRecorder()

        # Load VAE for sampling if configured
        vae = None
        if getattr(args, "vae", None) is not None:
            logger.info(f"Loading VAE from {args.vae}")
            try:
                # Use ModelManager for proper URL handling and caching
                vae = self.model_manager.load_vae(
                    args, vae_dtype=self.mixed_precision_dtype, vae_path=args.vae
                )
                vae.eval()
                logger.info("✅ VAE loaded successfully for sampling")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load VAE: {e}")
                vae = None

        # Initialize sampling manager (but postpone T5 preprocessing until after config setup)
        if getattr(args, "sample_every_n_steps", None) or getattr(
            args, "sample_every_n_epochs", None
        ):
            # Initialize sampling manager now that we have config
            self.sampling_manager = SamplingManager(
                self.config, getattr(args, "guidance_scale", 7.5)
            )

            # Configure VAE for lazy loading if VAE failed to load initially
            if vae is None and getattr(args, "vae", None) is not None:
                vae_config = {
                    "args": args,
                    "vae_dtype": self.mixed_precision_dtype,
                    "vae_path": args.vae,
                }
                self.sampling_manager.set_vae_config(vae_config)
                logger.info("✅ Sampling manager initialized with lazy VAE loading")
            else:
                logger.info("✅ Sampling manager initialized")

            # Pre-process sample prompts from config so we don't fall back to defaults later
            try:
                sample_prompts_cfg = getattr(args, "sample_prompts", None)
                if sample_prompts_cfg:
                    self.sample_parameters = (
                        self.sampling_manager.process_sample_prompts(
                            args=args,
                            accelerator=accelerator,
                            sample_prompts=sample_prompts_cfg,
                        )
                    )
                    if self.sample_parameters:
                        logger.info(
                            f"✅ Prepared {len(self.sample_parameters)} sample prompt(s) from config"
                        )
                    else:
                        logger.warning(
                            "No sample parameters produced from config; sampling will be skipped"
                        )
                else:
                    logger.info(
                        "No sample_prompts provided in config; sampling will use default only if invoked"
                    )
            except Exception as _sp_err:
                logger.warning(
                    f"Failed to process sample_prompts from config, will use fallback if sampling occurs: {_sp_err}"
                )

        # Calculate maximum training steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"Override steps. Steps for {args.max_train_epochs} epochs: {args.max_train_steps}"
            )

        # Send max_train_steps to dataset group
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # Prepare learning rate scheduler (use optimizer manager for LR scheduler)
        lr_scheduler = self.optimizer_manager.get_lr_scheduler(
            args, optimizer, accelerator.num_processes
        )

        # Apply full_bf16 if enabled for memory efficiency
        if self.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed_precision='bf16'"
            accelerator.print("Enable full bf16 training")

        # Prepare models with accelerator
        if self.blocks_to_swap > 0:
            transformer = accelerator.prepare(
                transformer, device_placement=[not self.blocks_to_swap > 0]
            )
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(
                accelerator.device
            )
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        # Prepare T5 encoder if enabled
        if text_encoder is not None:
            text_encoder = accelerator.prepare(text_encoder)
            logger.info("T5 text encoder prepared with accelerator")

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )

        # Apply fused backward pass AFTER accelerator.prepare() when optimizer is wrapped
        if self.fused_backward_pass:
            logger.info("⚡ Enabling fused backward pass optimization (post-prepare)")
            try:
                import modules.adafactor_fused as adafactor_fused

                # Patch the underlying optimizer inside the AcceleratedOptimizer wrapper
                underlying_optimizer = getattr(optimizer, "optimizer", optimizer)
                adafactor_fused.patch_adafactor_fused(
                    underlying_optimizer,
                    self.use_stochastic_rounding,
                    getattr(self, "use_stochastic_rounding_cuda", False),
                )

                # Create gradient hooks for fused optimization
                for param_group, param_name_group in zip(
                    underlying_optimizer.param_groups, param_names
                ):
                    for parameter, param_name in zip(
                        param_group["params"], param_name_group
                    ):
                        if parameter.requires_grad:

                            def create_grad_hook(p_name, p_group):
                                def grad_hook(tensor: torch.Tensor):
                                    # Process gradient immediately
                                    if (
                                        accelerator.sync_gradients
                                        and getattr(args, "max_grad_norm", 0.0) != 0.0
                                    ):
                                        accelerator.clip_grad_norm_(
                                            tensor, args.max_grad_norm
                                        )

                                    # Apply optimizer step to this parameter immediately
                                    if accelerator.sync_gradients:
                                        underlying_optimizer.step_param(tensor, p_group)

                                    # Clear gradient immediately to free memory
                                    tensor.grad = None

                                return grad_hook

                            parameter.register_post_accumulate_grad_hook(
                                create_grad_hook(param_name, param_group)
                            )

                logger.info("✅ Fused backward pass enabled (post-prepare)")
            except ImportError:
                logger.warning(
                    "⚠️  Fused backward pass module not found, falling back to standard optimization"
                )
                self.fused_backward_pass = False

        # Register checkpoint hooks for proper fine-tuning save/load (matching LoRA approach)
        CheckpointUtils.register_hooks_for_finetuning(accelerator, args)

        # training_model is the transformer itself (full fine-tuning)
        training_model = transformer

        analysis_frequency = getattr(args, "verify_weight_dynamics_every_n_steps", 0)

        if analysis_frequency > 0:
            # Initialize weight dynamics analysis baseline
            self.weight_analyzer.initialize_baseline_statistics(training_model)

        # TODO: Initialize enhanced REPA helper if enabled
        repa_helper = None
        if getattr(args, "enable_repa", False):
            try:
                from enhancements.repa.enhanced_repa_helper import EnhancedRepaHelper

                repa_helper = EnhancedRepaHelper(transformer, args)
                logger.info("Enhanced REPA helper initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced REPA helper: {e}")
                # Fallback to basic REPA if enhanced fails
                try:
                    repa_helper = RepaHelper(transformer, args)
                    logger.info("Basic REPA helper initialized as fallback")
                except Exception as e2:
                    logger.warning(f"Failed to initialize basic REPA helper: {e2}")

        # Set training mode and ensure proper device placement
        # Always use training mode for fine-tuning to ensure weight updates
        transformer.train()

        # Ensure block swap is set for training (not forward-only)
        if hasattr(transformer, "switch_block_swap_for_training"):
            transformer.switch_block_swap_for_training()

        # Clean memory before training starts
        clean_memory_on_device(accelerator.device)

        # TensorBoard tracker setup
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None and args.log_tracker_config.strip():
                import toml

                try:
                    init_kwargs = toml.load(args.log_tracker_config)
                except Exception:
                    init_kwargs = {}
            try:
                # Filter config to only include valid types for tracker
                config_dict = {}
                for key, value in vars(args).items():
                    if isinstance(value, (int, float, str, bool)) or torch.is_tensor(
                        value
                    ):
                        config_dict[key] = value
                    elif value is None:
                        config_dict[key] = "None"
                    else:
                        # Convert other types to string representation
                        config_dict[key] = str(value)

                accelerator.init_trackers(
                    (
                        "wan_finetune_train"
                        if args.log_tracker_name is None
                        else args.log_tracker_name
                    ),
                    config=config_dict,  # Use filtered config for WAN fine-tuning
                    init_kwargs=init_kwargs,
                )
                logger.info("✅ TensorBoard tracker initialized successfully")
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
                logger.info("✅ TensorBoard metric descriptions registered")
            except Exception:
                logger.warning(
                    "⚠️  TensorBoard metric descriptions registration failed, continuing without metric descriptions."
                )
                pass

        # Handle checkpoint resume for FULL MODEL fine-tuning
        if getattr(args, "direct_checkpoint_loading", False):
            # Checkpoint already loaded directly via load_wan_model, just extract step info
            logger.info(
                "Direct checkpoint loading: Model already loaded with checkpoint weights, extracting step info"
            )
            restored_step = CheckpointUtils._extract_step_from_checkpoint(args.resume)
        else:
            # Use standard checkpoint loading
            restored_step = CheckpointUtils.resume_from_local_if_specified(
                accelerator, args
            )

        if restored_step is not None:
            logger.info(f"✅ Resumed from checkpoint: step {restored_step}")
        # Note: Auto-resume already logs if no checkpoint found, so we don't duplicate that here

        # Initialize epoch and step tracking with STRICT separation
        starting_epoch = 0
        starting_global_step = 0

        if restored_step is not None:
            # Get checkpoint type information
            checkpoint_number, checkpoint_type = (
                CheckpointUtils._extract_checkpoint_info(args.resume)
            )

            logger.info(
                f"✅ Successfully resumed from {checkpoint_type}-based checkpoint"
            )
            logger.info(f"   - {checkpoint_type.title()}: {checkpoint_number}")

            if checkpoint_type == "epoch":
                # EPOCH-BASED: Set starting epoch, calculate global step
                starting_epoch = checkpoint_number
                starting_global_step = starting_epoch * len(train_dataloader)
                logger.info(
                    f"🔄 Epoch checkpoint: starting from epoch {starting_epoch}"
                )
                logger.info(
                    f"🔄 Calculated global step: {starting_global_step} ({starting_epoch} × {len(train_dataloader)})"
                )

            elif checkpoint_type == "step":
                # STEP-BASED: Set global step, calculate starting epoch
                starting_global_step = checkpoint_number
                starting_epoch = starting_global_step // len(train_dataloader)
                logger.info(
                    f"🔄 Step checkpoint: starting from step {starting_global_step}"
                )
                logger.info(
                    f"🔄 Calculated starting epoch: {starting_epoch} (step {starting_global_step} ÷ {len(train_dataloader)})"
                )

            else:
                logger.error(f"❌ Unknown checkpoint type: {checkpoint_type}")
                logger.info("🆕 Falling back to fresh training")
                starting_epoch = 0
                starting_global_step = 0

        # Enhanced resume functionality - backward compatible with existing logic
        enhanced_initial_step, enhanced_epoch_to_start, should_skip_data, _ = (
            CheckpointUtils.prepare_enhanced_resume(
                args,
                len(train_dataloader),
                accelerator.num_processes,
                getattr(args, "gradient_accumulation_steps", 1),
            )
        )

        # Override starting values if enhanced resume is active (while preserving existing checkpoint logic)
        if enhanced_initial_step > 0:
            # Validate max_train_steps
            if getattr(args, "max_train_steps", 0) <= enhanced_initial_step:
                logger.error(
                    f"❌ max_train_steps ({args.max_train_steps}) must be greater than initial_step ({enhanced_initial_step})"
                )
                return

            # Use enhanced resume values (overrides existing checkpoint resume)
            starting_epoch = enhanced_epoch_to_start
            starting_global_step = enhanced_initial_step
            logger.info(
                f"🔄 Enhanced resume overriding checkpoint resume: epoch={starting_epoch}, step={starting_global_step}"
            )

        # Main training loop
        logger.info("🚀 Starting WAN full fine-tuning...")

        # Use the correctly calculated global step from the strict separation logic above
        global_step = starting_global_step

        # Log debug info only if enabled
        NetworkLoggingUtils.log_checkpoint_debug_info(
            restored_step, starting_epoch, global_step, args
        )

        last_sampled_step = -1  # Track last sampling step
        last_validated_step = -1  # Track last validation step
        warned_no_val_pixels_for_perceptual = (
            False  # Warning flag for validation metrics
        )

        # Test TensorBoard logging at the start
        if accelerator.is_main_process and len(accelerator.trackers) > 0:
            try:
                test_logs = {
                    "training_started": 1.0,
                    "initial_step": float(global_step),
                }
                accelerator.log(test_logs, step=global_step)
                logger.info("✅ TensorBoard logging test successful")

                # Run latent quality analysis with TensorBoard logging
                from dataset.latent_quality_analyzer import (
                    run_tensorboard_analysis_for_trainer,
                )

                run_tensorboard_analysis_for_trainer(
                    args, train_dataset_group, accelerator, global_step
                )
            except Exception as e:
                logger.warning(f"⚠️  TensorBoard test logging failed: {e}")

        # Calculate total steps for progress bar
        total_steps = (
            args.max_train_epochs * len(train_dataloader)
            if hasattr(args, "max_train_epochs")
            else len(train_dataloader)
        )
        progress_bar = tqdm(
            total=total_steps,
            initial=global_step,  # Resume progress bar at correct position
            desc="Training",
            disable=not accelerator.is_local_main_process,
        )

        # Calculate epoch range - start from resumed epoch if applicable
        max_epochs = (
            args.max_train_epochs
            if (hasattr(args, "max_train_epochs") and args.max_train_epochs is not None)
            else 1
        )

        # Debug logging for epoch range
        NetworkLoggingUtils.log_epoch_range_info(starting_epoch, max_epochs)

        if starting_epoch >= max_epochs:
            logger.error(
                f"❌ Starting epoch ({starting_epoch}) >= max epochs ({max_epochs})"
            )
            logger.error("   This means training is already complete!")
            logger.error(
                "   Either increase max_train_epochs or start from an earlier checkpoint."
            )
            return

        # Final debug before entering training loops
        NetworkLoggingUtils.log_training_loop_entry(starting_epoch, max_epochs)

        # Handle enhanced resume dataloader wrapping (minimal integration)
        active_dataloader, skipped_steps = (
            CheckpointUtils.create_enhanced_training_loop_wrapper(
                train_dataloader,
                accelerator,
                enhanced_initial_step,
                should_skip_data,
                getattr(args, "gradient_accumulation_steps", 1),
            )
        )

        for epoch in range(starting_epoch, max_epochs):
            logger.info(f"🔄 Starting epoch {epoch + 1}/{max_epochs}")
            training_model.train()

            # Update training state for checkpoint saving
            CheckpointUtils.update_training_state_for_saving(
                args, epoch + 1, global_step
            )

            for step, batch in enumerate(active_dataloader):
                # Initialize timestep distribution and logging (using centralized utility)
                if not self._timestep_logging_initialized:
                    TimestepDistributionLogger.initialize_and_log_timestep_distribution(
                        args, accelerator, self.timestep_distribution
                    )
                    self._timestep_logging_initialized = True
                with accelerator.accumulate(training_model):
                    # Forward pass using the transformer DIRECTLY
                    loss = self.forward_pass(
                        args, accelerator, training_model, batch, text_encoder
                    )

                    # Backward pass
                    accelerator.backward(loss)

                    # Optimizer step with proper synchronization
                    if not self.fused_backward_pass:
                        # Standard optimization path - only step when gradients are synchronized
                        if accelerator.sync_gradients:
                            # Gradient clipping if enabled
                            if getattr(args, "max_grad_norm", 0.0) != 0.0:
                                params_to_clip = training_model.parameters()
                                accelerator.clip_grad_norm_(
                                    params_to_clip, args.max_grad_norm
                                )

                            # Optimizer step
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        # Fused backward pass - optimizer.step() and zero_grad() handled by hooks
                        if accelerator.sync_gradients:
                            lr_scheduler.step()

                            # Store weight change info for progress bar (for debugging)
                            if global_step % 1 == 0:
                                sample_param = next(training_model.parameters())
                                weight_mean = sample_param.data.mean().item()

                                # Store first weight for comparison
                                if not hasattr(training_model, "_debug_first_weight"):
                                    training_model._debug_first_weight = weight_mean
                                    training_model._weight_change = 0.0
                                    training_model._weight_status = "init"
                                else:
                                    change = abs(
                                        weight_mean - training_model._debug_first_weight
                                    )
                                    training_model._weight_change = change
                                    if change < 1e-7:
                                        training_model._weight_status = "minimal"
                                    else:
                                        training_model._weight_status = "changing"

                # Enhanced logging with comprehensive metrics
                if global_step % getattr(args, "logging_steps", 10) == 0:
                    if accelerator.is_main_process and len(accelerator.trackers) > 0:
                        # Prepare comprehensive logs
                        logs = {
                            "train_loss": loss.item(),
                            "learning_rate": (
                                lr_scheduler.get_last_lr()[0]
                                if lr_scheduler
                                else args.learning_rate
                            ),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        }

                        # Add gradient norm if available
                        if (
                            hasattr(optimizer, "grad_norm")
                            and optimizer.grad_norm is not None
                        ):
                            logs["gradient_norm"] = float(optimizer.grad_norm)

                        # Apply TensorBoard direction hints if enabled
                        try:
                            from utils.tensorboard_utils import (
                                apply_direction_hints_to_logs as _adh,
                            )

                            logs = _adh(args, logs)
                        except Exception:
                            pass

                        # Log to TensorBoard
                        accelerator.log(logs, step=global_step)
                        logger.info(
                            f"Step {global_step}: Loss = {loss.item():.6f}, LR = {logs.get('learning_rate', 'N/A')}"
                        )
                    else:
                        logger.info(f"Step {global_step}: Loss = {loss.item():.6f}")

                # Record loss and update progress
                if accelerator.sync_gradients:
                    # Record loss for averaging
                    current_loss = float(loss.detach().item())
                    loss_recorder.add(epoch=epoch + 1, step=step, loss=current_loss)

                    # Update progress bar with weight change info
                    progress_bar.update(1)

                    # Prepare progress bar data
                    current_lr = (
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler
                        else args.learning_rate
                    )
                    postfix_data = {
                        "loss": f"{current_loss:.6f}",
                        "avg": f"{loss_recorder.moving_average:.6f}",
                        "ep": epoch + 1,
                        "lr": f"{current_lr:.2e}",
                    }

                    # Add weight change info if available
                    if hasattr(training_model, "_weight_change") and hasattr(
                        training_model, "_weight_status"
                    ):
                        if training_model._weight_status == "changing":
                            postfix_data["Δw"] = (
                                f"✅ {training_model._weight_change:.2e}"
                            )
                        elif training_model._weight_status == "minimal":
                            postfix_data["Δw"] = (
                                f"⚠️ {training_model._weight_change:.2e}"
                            )
                        else:
                            pass

                    progress_bar.set_postfix(postfix_data)

                    global_step += 1

                    # Update training state for checkpoint saving
                    CheckpointUtils.update_training_state_for_saving(
                        args, epoch + 1, global_step
                    )

                    # Weight dynamics analysis at configurable intervals
                    if analysis_frequency > 0 and global_step % analysis_frequency == 0:
                        analysis_result = (
                            self.weight_analyzer.analyze_parameter_evolution(
                                training_model, global_step
                            )
                        )
                        if analysis_result and accelerator.is_main_process:
                            self.weight_analyzer.log_dynamics_analysis_summary(
                                analysis_result
                            )

                    # Handle sampling, validation, and saving
                    if accelerator.is_main_process:
                        # Check if we should sample
                        should_sample = ModelSavingUtils.should_sample_images(
                            args, global_step, epoch + 1
                        )

                        # Debug sampling conditions
                        if should_sample:
                            TrainingProgressLogger.log_sampling_debug(
                                global_step,
                                should_sample,
                                self.sampling_manager is not None,
                                vae is not None,
                            )

                        if should_sample and self.sampling_manager:
                            if global_step != last_sampled_step:
                                try:
                                    logger.info(
                                        f"🎨 Generating samples at step {global_step}"
                                    )

                                    # Use pre-processed sample parameters
                                    if self.sample_parameters is None:
                                        # Fallback: create default parameters if none were preprocessed
                                        logger.info(
                                            "No pre-processed sample parameters, creating default..."
                                        )
                                        default_prompts = [
                                            {
                                                "text": "A beautiful landscape",
                                                "width": 512,
                                                "height": 512,
                                                "frames": 16,
                                            }
                                        ]
                                        processed_sample_parameters = self.sampling_manager.process_sample_prompts(
                                            args=args,
                                            accelerator=accelerator,
                                            sample_prompts=default_prompts,
                                        )
                                    else:
                                        # Use pre-processed parameters with T5 embeddings already computed
                                        processed_sample_parameters = (
                                            self.sample_parameters
                                        )

                                    if processed_sample_parameters is None:
                                        logger.warning(
                                            "No sample parameters available, skipping sampling"
                                        )
                                    else:
                                        # Generate samples using sampling manager with pre-processed parameters
                                        self.sampling_manager.sample_images(
                                            accelerator=accelerator,
                                            args=args,
                                            epoch=epoch,
                                            steps=global_step,
                                            vae=vae,
                                            transformer=transformer,
                                            sample_parameters=processed_sample_parameters,  # Use pre-processed parameters (no repeated T5 loading)
                                            dit_dtype=self.mixed_precision_dtype,
                                        )

                                    logger.info(
                                        f"✅ Sampling completed at step {global_step}"
                                    )
                                    last_sampled_step = global_step

                                except Exception as e:
                                    logger.warning(
                                        f"⚠️  Sampling failed at step {global_step}: {e}"
                                    )

                        # Check if we should validate
                        should_validate = (
                            val_dataloader is not None
                            and getattr(args, "validate_every_n_steps", None)
                            is not None
                            and global_step % args.validate_every_n_steps == 0
                            and global_step > 0
                            and last_validated_step != global_step
                        )

                        if should_validate:
                            from core.handlers.validation_handler import (
                                handle_step_validation,
                            )
                            from modules.scheduling_flow_match_discrete import (
                                FlowMatchDiscreteScheduler,
                            )

                            # Create noise scheduler for validation
                            validation_noise_scheduler = FlowMatchDiscreteScheduler(
                                shift=getattr(args, "discrete_flow_shift", 3.0),
                                reverse=True,
                                solver="euler",
                            )

                            last_validated_step, warned_no_val_pixels_for_perceptual = (
                                handle_step_validation(
                                    should_validating=True,
                                    validation_core=self.training_core.validation_core,
                                    val_dataloader=val_dataloader,
                                    val_epoch_step_sync=val_epoch_step_sync,
                                    current_epoch=current_epoch,
                                    epoch=epoch,
                                    global_step=global_step,
                                    args=args,
                                    accelerator=accelerator,
                                    transformer=transformer,
                                    noise_scheduler=validation_noise_scheduler,
                                    control_signal_processor=None,  # Not used in finetune trainer
                                    vae=vae,  # Pass the VAE if available
                                    sampling_manager=self.sampling_manager,
                                    warned_no_val_pixels_for_perceptual=warned_no_val_pixels_for_perceptual,
                                    last_validated_step=last_validated_step,
                                    timestep_distribution=None,  # Optional parameter
                                )
                            )

                        # Handle step-based saving (includes model and state)
                        self.model_saving_utils.handle_step_saving(
                            args, accelerator, training_model, global_step
                        )

                        # Handle additional state saving based on save_state_every_n_steps
                        if should_save_state_at_step(args, global_step):
                            output_dir = getattr(args, "output_dir", "output")
                            output_name = getattr(args, "output_name", "wan_finetune")

                            # Save state directory for resume functionality
                            state_dir = os.path.join(
                                output_dir, f"{output_name}-step{global_step:06d}-state"
                            )
                            logger.info(f"💾 Saving additional state to: {state_dir}")
                            accelerator.save_state(state_dir)

                            # Save step info for resume
                            step_file = os.path.join(state_dir, "step.txt")
                            with open(step_file, "w") as f:
                                f.write(str(global_step))

                if (
                    hasattr(args, "max_train_steps")
                    and global_step >= args.max_train_steps
                ):
                    break

            if (
                hasattr(args, "max_train_steps")
                and args.max_train_steps is not None
                and global_step >= args.max_train_steps
            ):
                break

            # Handle epoch-end saving (includes model and state)
            if accelerator.is_main_process:
                self.model_saving_utils.handle_epoch_end_saving(
                    args, epoch, accelerator, training_model, global_step
                )

                # Handle additional state saving based on save_state_every_n_epochs
                if should_save_state_at_epoch(args, epoch + 1):
                    output_dir = getattr(args, "output_dir", "output")
                    output_name = getattr(args, "output_name", "wan_finetune")

                    # Save state directory for resume functionality
                    state_dir = os.path.join(
                        output_dir, f"{output_name}-epoch{epoch+1:04d}-state"
                    )
                    logger.info(f"💾 Saving additional state to: {state_dir}")
                    accelerator.save_state(state_dir)

                    # Save step and epoch info for resume
                    step_file = os.path.join(state_dir, "step.txt")
                    with open(step_file, "w") as f:
                        f.write(str(global_step))
                    epoch_file = os.path.join(state_dir, "epoch.txt")
                    with open(epoch_file, "w") as f:
                        f.write(str(epoch + 1))

        # Close progress bar
        progress_bar.close()

        # Final save using checkpoint manager format
        if accelerator.is_main_process:
            output_dir = getattr(args, "output_dir", "output")
            output_name = getattr(args, "output_name", "wan_finetune")

            # Save final model as safetensors
            final_model_path = os.path.join(
                output_dir, f"{output_name}_final.safetensors"
            )
            self.model_saving_utils.save_model_safetensors(
                args,
                accelerator,
                training_model,
                final_model_path,
                global_step,
                final=True,
            )

            # Also save final state for resuming if needed
            final_state_dir = os.path.join(output_dir, f"{output_name}-final-state")
            logger.info(f"💾 Saving final checkpoint state to: {final_state_dir}")
            accelerator.save_state(final_state_dir)

            from utils.train_utils import save_step_to_state_dir

            save_step_to_state_dir(final_state_dir, global_step)
        logger.info("WanFinetune training completed!")
        logger.info("🔄 TRAINING METHOD COMPLETED NORMALLY")
