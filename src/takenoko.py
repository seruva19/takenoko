import argparse
import logging
import torch
import gc
import toml
import os
import sys
import subprocess
import atexit
import time
from typing import Dict, Any, Optional, Tuple, List

from config_parser import create_args_from_config
from dataset import config_utils
from wan.modules.vae import WanVAE
from utils.model_utils import str_to_dtype
from common.model_downloader import download_model_if_needed

from caching.cache_latents import (
    encode_and_save_batch,
    encode_datasets,
    show_datasets,
)

from caching.cache_text_encoder_outputs import (
    encode_and_save_text_encoder_output_batch,
    process_text_encoder_batches,
    post_process_cache_files,
    prepare_cache_files_and_paths,
)
from wan.modules.t5 import T5EncoderModel
from wan.configs import wan_t2v_14B


from core.wan_network_trainer import WanNetworkTrainer
from common.logger import get_logger
from common.performance_logger import (
    snapshot_gpu_memory,
    force_cuda_cleanup,
)
from common.global_seed import set_global_seed

logger = get_logger(__name__, level=logging.INFO)

from common.dependencies import (
    setup_flash_attention,
    setup_sageattention,
    setup_xformers,
)

from dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
)

try:
    from distillation.rcm_bridge import dispatch_rcm_pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency tree
    dispatch_rcm_pipeline = None

import accelerate
from utils.memory_utils import configure_cuda_from_config
from common.vram_estimator import (
    estimate_and_log_vram,
)
from modules.ramtorch_linear_factory import (
    configure_ramtorch_from_args,
)

# Import memory tracking manager
try:
    from utils.memory_tracking_manager import (
        initialize_memory_tracking,
        get_memory_tracking_manager,
        show_memory_diagnostics as show_memory_diagnostics_func,
        is_memory_tracking_available,
    )

    MEMORY_TRACKING_AVAILABLE = True
except ImportError:
    MEMORY_TRACKING_AVAILABLE = False
    logger.debug("Memory tracking utilities not available")


def load_training_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Load training configuration from TOML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()

    config = toml.loads(config_content)
    return config, config_content


# Global variable to store TensorBoard process
_tensorboard_process: Optional[subprocess.Popen] = None


def find_tensorboard_executable() -> Optional[str]:
    """Find the correct TensorBoard executable path"""
    # Try different common locations and methods
    candidates = []

    # Method 1: Direct 'tensorboard' command
    candidates.append("tensorboard")

    # Method 2: In virtual environment Scripts directory (Windows)
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # We're in a virtual environment
        if os.name == "nt":  # Windows
            venv_tensorboard = os.path.join(
                os.path.dirname(sys.executable), "Scripts", "tensorboard.exe"
            )
            candidates.append(venv_tensorboard)
        else:  # Unix-like
            venv_tensorboard = os.path.join(
                os.path.dirname(sys.executable), "tensorboard"
            )
            candidates.append(venv_tensorboard)

    # Method 3: Alongside Python executable
    if os.name == "nt":  # Windows
        python_dir_tensorboard = os.path.join(
            os.path.dirname(sys.executable), "tensorboard.exe"
        )
        candidates.append(python_dir_tensorboard)

    # Test each candidate
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--help"], capture_output=True, timeout=10, text=True
            )
            if result.returncode == 0:
                return candidate
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            continue

    return None


def launch_tensorboard_server(
    logdir: str, host: str = "127.0.0.1", port: int = 6006, auto_reload: bool = True
) -> Optional[subprocess.Popen]:
    """Launch TensorBoard server in background"""
    global _tensorboard_process

    # Check if TensorBoard is already running on this port
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            logger.warning(
                f"TensorBoard appears to be already running on {host}:{port}"
            )
            return None
    except Exception:
        pass  # Ignore socket check errors

    try:
        # Find the correct TensorBoard executable
        tensorboard_exe = find_tensorboard_executable()
        if not tensorboard_exe:
            logger.error("❌ Could not find TensorBoard executable")
            return None

        # Build the command
        cmd = [tensorboard_exe, "--logdir", logdir, "--host", host, "--port", str(port)]

        if auto_reload:
            cmd.extend(["--reload_interval", "30"])

        logger.info("Launching TensorBoard server...")
        logger.debug(f"TensorBoard command: {' '.join(cmd)}")
        logger.info(f"TensorBoard will be accessible at: http://{host}:{port}")

        # Start the process with minimal output
        _tensorboard_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        # Give TensorBoard a moment to start
        time.sleep(3)

        # Check if process is still running (didn't immediately crash)
        if _tensorboard_process.poll() is None:
            logger.info("✓ TensorBoard server launched successfully!")
            logger.info(f"  Access it at: http://{host}:{port}")
            logger.info(f"  Logdir: {logdir}")

            # Register cleanup function to run on exit
            atexit.register(stop_tensorboard_server)

            return _tensorboard_process
        else:
            # Process crashed, get error output
            _, stderr = _tensorboard_process.communicate()
            logger.error(f"Failed to launch TensorBoard: {stderr}")
            _tensorboard_process = None
            return None

    except Exception as e:
        logger.exception(f"Error launching TensorBoard: {e}")
        return None


def stop_tensorboard_server():
    """Stop the TensorBoard server if running"""
    global _tensorboard_process

    if _tensorboard_process is not None:
        try:
            logger.info("Stopping TensorBoard server...")
            _tensorboard_process.terminate()

            # Give it a moment to terminate gracefully
            try:
                _tensorboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                _tensorboard_process.kill()
                _tensorboard_process.wait()

            logger.info("✓ TensorBoard server stopped")
        except Exception as e:
            logger.exception(f"Error stopping TensorBoard: {e}")
        finally:
            _tensorboard_process = None


def check_tensorboard_installation() -> bool:
    """Check if TensorBoard is properly installed"""
    try:
        import tensorboard

        return True
    except ImportError:
        return False


def get_tensorboard_install_instructions() -> str:
    """Get platform-specific TensorBoard installation instructions"""
    instructions = """
TensorBoard Installation Instructions:
=====================================

Option 1 (Recommended): Install via pip
  pip install tensorboard

Option 2: Install via conda  
  conda install tensorboard

Option 3: Install with TensorFlow
  pip install tensorflow  # includes tensorboard

After installation, restart your terminal/IDE and try again.
"""
    return instructions


def setup_tensorboard_if_enabled(args: argparse.Namespace):
    """Setup TensorBoard server if enabled in config"""
    # Only rank 0 should attempt to launch the server in distributed runs
    try:
        import accelerate  # type: ignore

        state = accelerate.PartialState()  # type: ignore
        is_main_process = bool(getattr(state, "is_main_process", True))
    except Exception:
        is_main_process = True

    if (
        hasattr(args, "launch_tensorboard_server")
        and args.launch_tensorboard_server
        and is_main_process
    ):
        # Check if TensorBoard is installed
        if not check_tensorboard_installation():
            logger.error("❌ TensorBoard is not installed!")
            logger.info(get_tensorboard_install_instructions())
            logger.warning(
                "TensorBoard server launch is disabled due to missing installation."
            )
            return

        # Ensure logging directory exists
        os.makedirs(args.logging_dir, exist_ok=True)

        # Launch TensorBoard server
        process = launch_tensorboard_server(
            logdir=args.logging_dir,
            host=args.tensorboard_host,
            port=args.tensorboard_port,
            auto_reload=args.tensorboard_auto_reload,
        )

        if process:
            logger.info(
                f"TensorBoard is running in the background (PID: {process.pid})"
            )
        else:
            logger.error("❌ Failed to launch TensorBoard server")
            logger.info(
                f"Try launching TensorBoard manually with: tensorboard --logdir {args.logging_dir} --host {args.tensorboard_host} --port {args.tensorboard_port}"
            )
    else:
        if (
            hasattr(args, "launch_tensorboard_server")
            and args.launch_tensorboard_server
        ):
            logger.info("TensorBoard server launch skipped on non-main process")
        else:
            logger.info("TensorBoard server launch is disabled")


def _estimate_and_log_vram_from_config(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[float, Dict[str, Any]]:
    return estimate_and_log_vram(config, logger)


class UnifiedTrainer:
    """Unified trainer that handles caching and training operations"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config, self.config_content = load_training_config(config_path)
        self.args = create_args_from_config(
            self.config, config_path, self.config_content
        )

        # Configure CUDA from TOML before any CUDA initialization
        try:
            configure_cuda_from_config(self.config, logger)
        except Exception as _cuda_alloc_err:
            # Non-fatal: proceed with default values if misconfigured
            logger.debug(f"CUDA config skipped: {_cuda_alloc_err}")

        # Apply logging level from config
        try:
            level_str = str(getattr(self.args, "logging_level", "INFO")).upper()
            level_val = getattr(logging, level_str, logging.INFO)
            logger.setLevel(level_val)
            for h in logger.handlers:
                h.setLevel(level_val)
        except Exception:
            pass

        # Set global seed for reproducibility
        try:
            set_global_seed(int(getattr(self.args, "seed", 42)))
            logger.info(f"🔒 Global seed set to {getattr(self.args, 'seed', 42)}")
        except Exception as seed_err:
            logger.warning(f"Failed to set global seed: {seed_err}")

        flash_attn, _flash_attn_forward, flash_attn_varlen_func, flash_attn_func = (
            setup_flash_attention()
        )
        sageattn_varlen, sageattn = setup_sageattention()
        xops = setup_xformers()

        # Setup TensorBoard server if enabled
        setup_tensorboard_if_enabled(self.args)

        # Initialize memory tracking if enabled
        initialize_memory_tracking(self.args) if MEMORY_TRACKING_AVAILABLE else None

        # Configure RamTorch Linear replacement from root TOML
        try:
            configure_ramtorch_from_args(self.args)
        except Exception as e:
            logger.debug(f"RamTorch Linear configuration skipped: {e}")

    def show_menu(self) -> str:
        """Display the main menu and get user choice (legacy method)"""
        from menu.operations_menu import create_operations_menu

        menu = create_operations_menu(self)
        return menu.display()

    def cache_latents(self) -> bool:
        """Run latent caching operation"""
        logger.info("Starting Latent Caching...")

        try:
            # Optional memory trace gate
            trace_memory: bool = bool(
                getattr(self, "args", argparse.Namespace()).__dict__.get(
                    "trace_memory", False
                )
            )
            if trace_memory:
                snapshot_gpu_memory("cache_latents/before")
            # Create args namespace with the required arguments from config
            cache_args = argparse.Namespace()
            cache_args.dataset_config = self.args.dataset_config
            cache_args.vae = self.args.vae
            cache_args.vae_dtype = self.args.vae_dtype
            cache_args.vae_cache_cpu = self.args.vae_cache_cpu
            cache_args.clip = self.args.clip
            cache_args.device = self.args.latent_cache_device
            cache_args.debug_mode = self.args.latent_cache_debug_mode
            cache_args.console_width = self.args.latent_cache_console_width
            cache_args.console_back = self.args.latent_cache_console_back
            cache_args.console_num_images = self.args.latent_cache_console_num_images
            cache_args.batch_size = self.args.latent_cache_batch_size
            cache_args.num_workers = self.args.latent_cache_num_workers
            cache_args.skip_existing = self.args.latent_cache_skip_existing
            cache_args.keep_cache = self.args.latent_cache_keep_cache
            cache_args.purge_before_run = self.args.latent_cache_purge

            # Add control LoRA caching arguments
            cache_args.control_lora_type = self.args.control_lora_type
            cache_args.control_preprocessing = self.args.control_preprocessing
            cache_args.control_blur_kernel_size = self.args.control_blur_kernel_size
            cache_args.control_blur_sigma = self.args.control_blur_sigma

            # Add target_model for proper cache extension
            cache_args.target_model = self.args.target_model

            # Set default values for any missing arguments
            if not hasattr(cache_args, "vae") or cache_args.vae is None:
                raise ValueError("VAE checkpoint is required for latent caching")

            logger.info(f"Running latent caching with VAE: {cache_args.vae}")
            logger.info(f"Device: {cache_args.device}")
            logger.debug(f"Debug mode: {cache_args.debug_mode}")
            logger.info(f"Batch size: {cache_args.batch_size}")
            logger.info(f"Num workers: {cache_args.num_workers}")
            logger.info(f"Skip existing: {cache_args.skip_existing}")
            logger.info(f"Keep cache: {cache_args.keep_cache}")

            # Run latent caching using the unified functions

            device = (
                cache_args.device
                if cache_args.device is not None
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
            if isinstance(device, str):
                device = torch.device(device)

            # Load dataset config
            blueprint_generator = BlueprintGenerator(ConfigSanitizer())
            logger.info(f"Load dataset config from {cache_args.dataset_config}")
            user_config = config_utils.load_user_config(cache_args.dataset_config)

            blueprint = blueprint_generator.generate(user_config, cache_args)

            # Combine training and validation dataset blueprints
            all_dataset_blueprints = list(blueprint.train_dataset_group.datasets)
            if len(blueprint.val_dataset_group.datasets) > 0:
                all_dataset_blueprints.extend(blueprint.val_dataset_group.datasets)

            combined_dataset_group_blueprint = config_utils.DatasetGroupBlueprint(
                all_dataset_blueprints
            )

            dataset_group = config_utils.generate_dataset_group_by_blueprint(
                combined_dataset_group_blueprint,
                training=False,
                prior_loss_weight=getattr(cache_args, "prior_loss_weight", 1.0),
            )

            datasets = dataset_group.datasets

            if cache_args.debug_mode is not None:
                show_datasets(
                    datasets,  # type: ignore
                    cache_args.debug_mode,
                    cache_args.console_width,
                    cache_args.console_back,
                    cache_args.console_num_images,
                    fps=16,
                )
                return True

            assert cache_args.vae is not None, "vae checkpoint is required"

            vae_path = cache_args.vae

            # Download model if it's a URL
            if vae_path.startswith(("http://", "https://")):
                logger.info(f"Detected URL for VAE model, downloading: {vae_path}")
                cache_dir = getattr(cache_args, "model_cache_dir", None)
                vae_path = download_model_if_needed(vae_path, cache_dir=cache_dir)
                logger.info(f"Downloaded VAE model to: {vae_path}")

            logger.info(f"Loading VAE model from {vae_path}")
            # Default to float16 for consistency unless explicitly overridden
            vae_dtype = (
                torch.float16
                if cache_args.vae_dtype is None
                else str_to_dtype(cache_args.vae_dtype)
            )
            cache_device = torch.device("cpu") if cache_args.vae_cache_cpu else None
            vae = WanVAE(
                vae_path=vae_path,
                device=str(device),
                dtype=vae_dtype,
                cache_device=cache_device,
            )
            # Convert device string to torch.device for compatibility with encode_and_save_batch
            vae.device = torch.device(vae.device)

            if trace_memory:
                snapshot_gpu_memory("cache_latents/after_load")

            # Encode images
            def encode(one_batch):
                encode_and_save_batch(vae, one_batch, cache_args)

            encode_datasets(datasets, encode, cache_args)  # type: ignore

            # Clean up VAE model from memory
            logger.info("Cleaning up VAE model from memory...")
            if trace_memory:
                snapshot_gpu_memory("cache_latents/before_cleanup")
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if trace_memory:
                force_cuda_cleanup("cache_latents")

            logger.info("Latent caching completed successfully!")
            return True

        except Exception as e:
            logger.exception(f"❌ Error during latent caching: {e}")
            return False

    def cache_text_encoder_outputs(self) -> bool:
        """Run text encoder output caching operation - simplified without temporary files"""
        logger.info("Starting Text Encoder Output Caching...")

        try:
            trace_memory: bool = bool(
                getattr(self, "args", argparse.Namespace()).__dict__.get(
                    "trace_memory", False
                )
            )
            if trace_memory:
                snapshot_gpu_memory("cache_t5/before")
            # Create args namespace with the required arguments from config
            cache_args = argparse.Namespace()
            cache_args.dataset_config = self.args.dataset_config
            cache_args.t5 = self.args.t5
            cache_args.fp8_t5 = self.args.fp8_t5
            cache_args.device = self.args.text_encoder_cache_device
            cache_args.num_workers = self.args.text_encoder_cache_num_workers
            cache_args.skip_existing = self.args.text_encoder_cache_skip_existing
            cache_args.batch_size = self.args.text_encoder_cache_batch_size
            cache_args.keep_cache = self.args.text_encoder_cache_keep_cache
            cache_args.purge_before_run = self.args.text_encoder_cache_purge

            # Add target_model for proper cache extension
            cache_args.target_model = self.args.target_model

            # Set default values for any missing arguments
            if not hasattr(cache_args, "t5") or cache_args.t5 is None:
                raise ValueError(
                    "T5 checkpoint is required for text encoder output caching"
                )

            logger.info(f"Running text encoder caching with T5: {cache_args.t5}")
            logger.info(f"Dataset config: {cache_args.dataset_config}")
            logger.info(f"Device: {cache_args.device}")
            logger.info(f"Batch size: {cache_args.batch_size}")
            logger.info(f"Num workers: {cache_args.num_workers}")
            logger.info(f"Skip existing: {cache_args.skip_existing}")
            logger.info(f"Keep cache: {cache_args.keep_cache}")
            logger.info(f"FP8 T5: {cache_args.fp8_t5}")

            # Run text encoder caching using the imported functions

            device = (
                cache_args.device
                if cache_args.device is not None
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
            if isinstance(device, str):
                device = torch.device(device)

            # Load dataset config
            blueprint_generator = BlueprintGenerator(ConfigSanitizer())
            logger.info(f"Load dataset config from {cache_args.dataset_config}")
            user_config = config_utils.load_user_config(cache_args.dataset_config)

            blueprint = blueprint_generator.generate(user_config, cache_args)

            # Combine training and validation dataset blueprints
            all_dataset_blueprints = list(blueprint.train_dataset_group.datasets)
            if len(blueprint.val_dataset_group.datasets) > 0:
                all_dataset_blueprints.extend(blueprint.val_dataset_group.datasets)

            combined_dataset_group_blueprint = config_utils.DatasetGroupBlueprint(
                all_dataset_blueprints
            )

            dataset_group = config_utils.generate_dataset_group_by_blueprint(
                combined_dataset_group_blueprint,
                training=False,
                prior_loss_weight=getattr(cache_args, "prior_loss_weight", 1.0),
            )

            datasets = dataset_group.datasets

            # define accelerator for fp8 inference
            # Select T5 config based on mapped task
            if getattr(self.args, "task", "t2v-14B") == "t2v-A14B":
                from wan.configs import wan_t2v_A14B as _wan_cfg

                config = _wan_cfg.t2v_A14B  # type: ignore
            else:
                config = wan_t2v_14B.t2v_14B
            accelerator = None
            if cache_args.fp8_t5:
                mixed_precision = getattr(cache_args, "mixed_precision", None)
                if not mixed_precision or mixed_precision == "no":
                    mixed_precision = "bf16" if config.t5_dtype == torch.bfloat16 else "fp16"  # type: ignore
                accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

            # prepare cache files and paths
            all_cache_files_for_dataset, all_cache_paths_for_dataset = (
                prepare_cache_files_and_paths(datasets)  # type: ignore
            )

            # Download T5 model if it's a URL
            t5_path = cache_args.t5
            if t5_path.startswith(("http://", "https://")):
                logger.info(f"Detected URL for T5 model, downloading: {t5_path}")
                cache_dir = getattr(cache_args, "model_cache_dir", None)
                t5_path = download_model_if_needed(t5_path, cache_dir=cache_dir)
                logger.info(f"Downloaded T5 model to: {t5_path}")

            # Load T5
            logger.info(f"Loading T5: {t5_path}")
            text_encoder = T5EncoderModel(
                text_len=config.text_len,  # type: ignore
                dtype=config.t5_dtype,  # type: ignore
                device=device,  # type: ignore
                weight_path=t5_path,
                fp8=cache_args.fp8_t5,
            )

            if trace_memory:
                snapshot_gpu_memory("cache_t5/after_load")

            # Encode with T5
            logger.info("Encoding with T5")

            def encode_for_text_encoder(batch):
                encode_and_save_text_encoder_output_batch(
                    text_encoder,
                    batch,
                    device,
                    accelerator,
                    cache_args,  # <-- ADD cache_args
                )

            # Mark the encoder closure with purge flag so the batch processor can purge before run
            setattr(
                encode_for_text_encoder,
                "_purge_before_run",
                cache_args.purge_before_run,
            )

            process_text_encoder_batches(
                cache_args.num_workers,
                cache_args.skip_existing,
                cache_args.batch_size,
                datasets,  # type: ignore
                all_cache_files_for_dataset,
                all_cache_paths_for_dataset,
                encode_for_text_encoder,
            )

            # Clean up text encoder model from memory
            logger.info("Cleaning up text encoder model from memory...")
            if trace_memory:
                snapshot_gpu_memory("cache_t5/before_cleanup")
            del text_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if trace_memory:
                force_cuda_cleanup("cache_t5")

            # remove cache files not in dataset
            post_process_cache_files(
                datasets,  # type: ignore
                all_cache_files_for_dataset,
                all_cache_paths_for_dataset,
                cache_args.keep_cache,
            )

            logger.info("✅ Text encoder output caching completed successfully!")
            return True

        except Exception as e:
            logger.exception(f"❌ Error during text encoder output caching: {e}")
            return False

    def _estimate_latent_cache_chunks(self) -> int:
        from caching.chunk_estimator import estimate_latent_cache_chunks

        return estimate_latent_cache_chunks(self.args.dataset_config, self.args)

    def free_vram_aggressively(self) -> bool:
        """Attempt to free as much VRAM as possible using safe mechanisms."""
        try:
            try:
                snapshot_gpu_memory("free_vram/before")
            except Exception:
                pass
            # CPU garbage collect
            try:
                gc.collect()
            except Exception:
                pass
            # CUDA cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # ipc_collect may not exist on all builds; ignore if missing
                try:
                    torch.cuda.ipc_collect()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            # Use existing utility to enforce cleanup
            try:
                force_cuda_cleanup("manual_free")
            except Exception:
                pass
            try:
                snapshot_gpu_memory("free_vram/after")
            except Exception:
                pass
            logger.info("Attempted aggressive VRAM cleanup.")
            return True
        except Exception as e:
            logger.exception(f"❌ Error during VRAM cleanup: {e}")
            return False

    def _log_timestep_configuration(self) -> None:
        """Log comprehensive timestep configuration information."""
        args = self.args
        # Honor configured logging level
        level_str = str(getattr(args, "logging_level", "INFO")).upper()
        level_val = getattr(logging, level_str, logging.INFO)
        logger = get_logger(__name__, level=level_val)

        logger.info("Timestep Sampling Configuration:")
        logger.info(f"   Method: '{args.timestep_sampling}'")

        # Strategy summary (explicit)
        use_precomputed = bool(getattr(args, "use_precomputed_timesteps", False))
        num_buckets = getattr(args, "num_timestep_buckets", None)
        has_bucketting = isinstance(num_buckets, int) and num_buckets >= 2

        strategy_parts: list[str] = [str(getattr(args, "timestep_sampling", "unknown"))]
        if use_precomputed:
            strategy_parts.append("precomputed-quantiles")
        if has_bucketting:
            strategy_parts.append(f"dataset-bucketing({num_buckets})")
        logger.info(f"   Strategy: {' + '.join(strategy_parts)}")

        # Dataset-driven timestep bucketing
        if has_bucketting:
            logger.info(f"   Dataset bucketing: ENABLED — num_buckets={num_buckets}")
        else:
            logger.info("   Dataset bucketing: DISABLED")

        # Show method-specific parameters
        if args.timestep_sampling in ["sigmoid", "shift", "logit_normal"]:
            logger.info(f"   Sigmoid scale: {getattr(args, 'sigmoid_scale', 1.0)}")
        if args.timestep_sampling == "shift":
            logger.info(f"   Flow shift: {getattr(args, 'discrete_flow_shift', 3.0)}")

        # Show optimization settings
        if use_precomputed:
            logger.info(f"   Precomputed quantiles: ENABLED")
            logger.info(
                f"   Bucket count: {getattr(args, 'precomputed_timestep_buckets', 10000):,}"
            )
        else:
            logger.info(f"   Optimization: Original random sampling")

        # Show timestep constraints
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        if min_timestep is not None or max_timestep is not None:
            min_val = min_timestep if min_timestep is not None else 0
            max_val = max_timestep if max_timestep is not None else 1000
            logger.info(f"   Range: [{min_val}, {max_val}] (out of [0, 1000])")
        else:
            logger.info(f"   Range: [0, 1000] (full range)")

        # Additional sampling toggles
        logger.info(
            f"   Round to schedule steps: {bool(getattr(args, 'round_training_timesteps', False))}"
        )
        logger.info(
            f"   Preserve distribution shape: {args.preserve_distribution_shape}"
        )
        logger.info(
            f"   Skip extra in-range constraint: {args.skip_extra_timestep_constraint}"
        )
        logger.info(
            f"   Constraint epsilon: {float(getattr(args, 'timestep_constraint_epsilon', 1e-6))}"
        )
        if args.preserve_distribution_shape:
            logger.info(
                "   Shape preservation settings:"
                + f" weighting='{getattr(args, 'weighting_scheme', 'none')}',"
                + f" mode_scale={getattr(args, 'mode_scale', 1.29)},"
                + f" bell_center={getattr(args, 'bell_center', 0.5)},"
                + f" bell_std={getattr(args, 'bell_std', 0.2)},"
                + f" logit_mean={getattr(args, 'logit_mean', 0.0)},"
                + f" logit_std={getattr(args, 'logit_std', 1.0)}"
            )

        # Show compatibility status
        supported_methods = ["uniform", "sigmoid", "shift", "logit_normal"]
        if use_precomputed:
            if args.timestep_sampling in supported_methods:
                logger.info(f"   Compatibility: fully supported")
            else:
                logger.info(f"   Compatibility: will fall back to original sampling")
                if args.timestep_sampling == "flux_shift":
                    logger.info(
                        f"      Reason: Spatial dependency (image size affects timesteps)"
                    )
                elif args.timestep_sampling == "fopp":
                    logger.info(f"      Reason: Complex AR-Diffusion logic")

        logger.info("")  # Empty line for readability

    def _log_acceleration_configuration(self) -> None:
        """Log comprehensive acceleration configuration information."""
        args = self.args
        # Honor configured logging level
        level_str = str(getattr(args, "logging_level", "INFO")).upper()
        level_val = getattr(logging, level_str, logging.INFO)
        logger = get_logger(__name__, level=level_val)

        logger.info("🚀 Acceleration Configuration:")

        # Check which acceleration techniques are enabled
        acceleration_methods = []

        # Core attention optimizations
        if getattr(args, "sdpa", False):
            acceleration_methods.append("SDPA (Scaled Dot-Product Attention)")

        if getattr(args, "flash_attn", False):
            acceleration_methods.append("FlashAttention")

        if getattr(args, "sage_attn", False):
            acceleration_methods.append("SageAttention")

        if getattr(args, "xformers", False):
            acceleration_methods.append("Xformers")

        if getattr(args, "flash3", False):
            acceleration_methods.append("FlashAttention 3.0")

        if getattr(args, "split_attn", False):
            acceleration_methods.append("Split Attention")

        # Precision optimizations
        mixed_precision = getattr(args, "mixed_precision", "no")
        if mixed_precision != "no":
            acceleration_methods.append(f"Mixed Precision ({mixed_precision})")

        if getattr(args, "full_fp16", False):
            acceleration_methods.append("Full FP16")
        if getattr(args, "full_bf16", False):
            acceleration_methods.append("Full BF16")

        # FP8 optimizations
        if getattr(args, "fp8_scaled", False):
            acceleration_methods.append("FP8 Scaled")
        if getattr(args, "fp8_base", False):
            acceleration_methods.append("FP8 Base")
        if getattr(args, "fp8_t5", False):
            acceleration_methods.append("FP8 T5")
        if (
            getattr(args, "fp8_scaled", False)
            or getattr(args, "fp8_base", False)
            or getattr(args, "fp8_t5", False)
        ):
            fp8_format = getattr(args, "fp8_format", "e4m3")
            acceleration_methods.append(f"FP8 Format: {fp8_format.upper()}")

        # Memory and compute optimizations
        if getattr(args, "gradient_checkpointing", False):
            acceleration_methods.append("Gradient Checkpointing")

        if getattr(args, "persistent_data_loader_workers", False):
            acceleration_methods.append("Persistent DataLoader Workers")

        # Recent optimization techniques
        if getattr(args, "optimized_torch_compile", False):
            compile_args = getattr(args, "compile_args", None)
            if compile_args and len(compile_args) >= 2:
                backend, mode = compile_args[0], compile_args[1]
                acceleration_methods.append(
                    f"Optimized torch.compile ({backend}, {mode})"
                )
            else:
                acceleration_methods.append("Optimized torch.compile")

        if getattr(args, "lean_attn_math", False):
            fp32_default = getattr(args, "lean_attention_fp32_default", False)
            policy = "FP32 default" if fp32_default else "Input dtype default"
            acceleration_methods.append(f"Lean Attention Math ({policy})")

        if getattr(args, "lower_precision_attention", False):
            acceleration_methods.append("Lower Precision Attention (FP16)")

        if getattr(args, "simple_modulation", False):
            acceleration_methods.append("Simple Modulation (Wan 2.1 style)")

        rope_func = getattr(args, "rope_func", "default")
        if rope_func != "default":
            acceleration_methods.append(f"RoPE type: {rope_func}")

        # TREAD optimization
        if getattr(args, "enable_tread", False):
            tread_mode = getattr(args, "tread_mode", "full")
            if tread_mode != "full":
                acceleration_methods.append(f"TREAD Optimization ({tread_mode})")
            else:
                acceleration_methods.append("TREAD Optimization")

        # Compilation optimizations
        dynamo_backend = getattr(args, "dynamo_backend", "NO")
        if dynamo_backend != "NO":
            acceleration_methods.append(f"Dynamo ({dynamo_backend})")

        # Log the results
        if acceleration_methods:
            logger.info("   ✅ Enabled acceleration techniques:")
            for method in acceleration_methods:
                logger.info(f"      • {method}")
        else:
            logger.info("   ⚠️  No acceleration techniques enabled")

        # Check for potential conflicts
        conflicts = []
        if getattr(args, "flash_attn", False) and getattr(args, "xformers", False):
            conflicts.append("FlashAttention and Xformers may conflict")
        if getattr(args, "flash_attn", False) and getattr(args, "sage_attn", False):
            conflicts.append("FlashAttention and SageAttention may conflict")
        if getattr(args, "xformers", False) and getattr(args, "sage_attn", False):
            conflicts.append("Xformers and SageAttention may conflict")
        if getattr(args, "optimized_torch_compile", False) and dynamo_backend != "NO":
            conflicts.append(
                "Optimized Torch Compile and Dynamo are mutually exclusive"
            )

        if conflicts:
            logger.info("   ⚠️  Potential conflicts detected:")
            for conflict in conflicts:
                logger.info(f"      • {conflict}")

        # Log device information
        device = getattr(args, "device", None)
        if device:
            logger.info(f"   🖥️  Target device: {device}")

        logger.info("")  # Empty line for readability

    def train_model(self) -> bool:
        """Run training operation"""
        logger.info("Starting Model Training...")

        # Log comprehensive timestep configuration
        self._log_timestep_configuration()

        # Log acceleration configuration
        self._log_acceleration_configuration()

        # Estimate and store VRAM usage for later validation (if enabled)
        from core.handlers.vram_validation_handler import estimate_and_store_vram

        estimate_and_store_vram(self.args, self.config)

        try:
            # Route to RCM pipeline if enabled via config
            if getattr(getattr(self.args, "rcm", None), "enabled", False):
                if dispatch_rcm_pipeline is None:
                    logger.error(
                        "RCM pipeline requested but distillation.rcm_bridge is unavailable"
                    )
                    return False
                logger.info(
                    "?? RCM pipeline enabled via config – dispatching distillation runner"
                )
                return dispatch_rcm_pipeline(
                    args=self.args,
                    raw_config=self.config,
                    raw_config_content=self.config_content,
                    config_path=self.config_path,
                )

            # Select trainer based on network_module configuration
            network_module = getattr(self.args, "network_module", "networks.lora_wan")

            if network_module == "networks.wan_finetune":
                # Use WanFinetune trainer for full model fine-tuning
                from core.wan_finetune_trainer import WanFinetuneTrainer

                trainer = WanFinetuneTrainer()
                logger.info("🔥 Using WanFinetune trainer for full model fine-tuning")
            else:
                # Use default WAN LoRA trainer
                trainer = WanNetworkTrainer()
                logger.info("🔧 Using WAN LoRA trainer for LoRA training")

            # Store config content in trainer for state saving
            trainer.original_config_content = self.config_content  # type: ignore
            trainer.original_config_path = self.config_path  # type: ignore
            trainer.train(self.args)

            logger.info("Training completed successfully!")
            return True

        except Exception as e:
            logger.exception(f"❌ Error during training: {e}")
            return False

    def reload_config(self) -> bool:
        """Reload configuration from file"""
        logger.info("Reloading Configuration...")

        try:
            # Clean up existing temporary files and stop TensorBoard
            self.cleanup()

            # Reload configuration
            self.config, self.config_content = load_training_config(self.config_path)
            self.args = create_args_from_config(
                self.config, self.config_path, self.config_content
            )

            # Setup TensorBoard server with new configuration
            setup_tensorboard_if_enabled(self.args)

            logger.info(f"Configuration reloaded successfully from: {self.config_path}")
            logger.info("All settings have been updated.")
            return True

        except Exception as e:
            logger.exception(f"❌ Error reloading configuration: {e}")
            return False

    def run(self):
        """Main loop for the unified trainer"""
        logger.info(f"Loaded configuration from: {self.config_path}")

        # Use the new menu system
        from menu.operations_menu import create_operations_menu

        menu = create_operations_menu(self)
        menu.run(self)

    def cleanup(self):
        """Clean up temporary files"""
        # No temp files to clean up anymore

        # Stop TensorBoard server if running
        stop_tensorboard_server()


def main():
    """Entry point: preserves interactive menu by default, adds non-interactive flags."""
    parser = argparse.ArgumentParser(
        description="Takenoko unified operations (interactive by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config path: support positional or --config for convenience
    parser.add_argument("config", nargs="?", help="Path to training config TOML file")
    parser.add_argument(
        "--config", dest="config_opt", help="Path to training config TOML file"
    )

    # Non-interactive operation flags
    parser.add_argument(
        "--cache-latents", action="store_true", help="Run latent caching and exit"
    )
    parser.add_argument(
        "--cache-text-encoder",
        action="store_true",
        help="Run text encoder output caching and exit",
    )
    parser.add_argument("--train", action="store_true", help="Run training and exit")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run cache-latents, cache-text-encoder, then train",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive menu; requires at least one action flag or --all",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config_opt or args.config
    if not config_path:
        parser.error("missing config path (positional 'config' or --config)")

    # Validate config path early
    if not os.path.exists(config_path):
        logger.error(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Determine requested actions
    action_flags = [args.cache_latents, args.cache_text_encoder, args.train, args.all]
    non_interactive_requested = args.non_interactive or any(action_flags)

    try:
        trainer = UnifiedTrainer(config_path)

        if non_interactive_requested:
            # Build ordered action list
            actions: List[str] = []
            if args.all:
                actions = ["cache_latents", "cache_text_encoder", "train"]
            else:
                if args.cache_latents:
                    actions.append("cache_latents")
                if args.cache_text_encoder:
                    actions.append("cache_text_encoder")
                if args.train:
                    actions.append("train")

            if len(actions) == 0:
                logger.error(
                    "--non-interactive requires at least one action flag or --all"
                )
                sys.exit(2)

            overall_success = True
            for action in actions:
                if action == "cache_latents":
                    success = trainer.cache_latents()
                elif action == "cache_text_encoder":
                    success = trainer.cache_text_encoder_outputs()
                elif action == "train":
                    success = trainer.train_model()
                else:
                    logger.error(f"Unknown action: {action}")
                    success = False

                if not success:
                    overall_success = False
                    break

            sys.exit(0 if overall_success else 1)
        else:
            # Preserve existing behavior: interactive menu
            trainer.run()

    except SystemExit:
        # bubble up argparse or explicit exits
        raise
    except Exception as e:
        logger.exception(f"💥 CRITICAL ERROR: {e}")
        sys.exit(1)
    finally:
        if "trainer" in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()
