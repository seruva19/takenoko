## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/wan_train_network.py (Apache)

import argparse
import logging
import traceback

from dataset import config_utils
from wan.modules.vae import WanVAE
from utils.model_utils import str_to_dtype
from common.model_downloader import download_model_if_needed
import torch
import gc
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

from core.config import TrainerConfig
from common.dependencies import (
    setup_flash_attention,
    setup_sageattention,
    setup_xformers,
)
import toml
import os

import sys
import subprocess
import atexit
import time
from typing import Dict, Any, Optional, Tuple, List

from dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
    validate_dataset_config,
)

import accelerate
from utils.memory_utils import configure_cuda_allocator_from_config
from common.vram_estimator import (
    estimate_peak_vram_gb_from_config as shared_estimate_vram,
)


def load_training_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Load training configuration from TOML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()

    config = toml.loads(config_content)
    return config, config_content


def create_args_from_config(
    config: Dict[str, Any], config_path: str = None, config_content: str = None  # type: ignore
) -> argparse.Namespace:
    """Convert config dictionary to argparse.Namespace for compatibility"""
    # Removed pprint import - not needed

    args = argparse.Namespace()

    # Store original config information for state saving
    args.config_file = config_path
    args.config_content = config_content

    # Set default values for all possible arguments
    # Model settings
    args.task = config.get("task", "t2v-14B")

    # Map target_model to appropriate task if specified
    target_model = config.get("target_model", "wan21")  # for backwards compatibility

    # Validate target_model is a string
    if not isinstance(target_model, str):
        logger.warning(
            f"⚠️  Invalid target_model type '{type(target_model)}'. Expected string. Using default 'wan21'"
        )
        target_model = "wan21"

    args.target_model = (
        target_model  # Store target_model in args for dataset configuration
    )

    if target_model:
        target_model_mapping = {
            "wan21": "t2v-14B",
            "wan22": "t2v-A14B",
        }
        if target_model in target_model_mapping:
            args.task = target_model_mapping[target_model]
            logger.info(
                f"📋 Mapped target_model '{target_model}' to task '{args.task}'"
            )
        else:
            logger.warning(
                f"⚠️  Unknown target_model '{target_model}'. Using task '{args.task}'"
            )
    else:
        logger.info(f"📋 Using task '{args.task}' (no target_model specified)")

    args.fp8_scaled = config.get("fp8_scaled", False)
    args.fp8_base = config.get("fp8_base", False)
    # Quantization behavior controls (default False to preserve prior behavior)
    args.upcast_quantization = bool(config.get("upcast_quantization", False))
    args.upcast_linear = bool(config.get("upcast_linear", False))
    # New FP8 optimization flags (gated for safety - default to False)
    args.exclude_ffn_from_scaled_mm = bool(
        config.get("exclude_ffn_from_scaled_mm", False)
    )
    args.scale_input_tensor = config.get("scale_input_tensor", None)
    # Allow loading mixed precision transformer (per-tensor dtypes preserved)
    args.mixed_precision_transformer = bool(
        config.get("mixed_precision_transformer", False)
    )
    # Optional uniform cast dtype for DiT weights ("fp16"|"bf16"|None)
    args.dit_cast_dtype = config.get("dit_cast_dtype", None)
    args.t5 = config.get("t5")
    args.fp8_t5 = config.get("fp8_t5", False)
    args.clip = config.get("clip")
    args.vae_cache_cpu = config.get("vae_cache_cpu", False)
    args.dit = config.get("dit")
    args.vae = config.get("vae")
    args.vae_dtype = config.get("vae_dtype")
    args.model_cache_dir = config.get("model_cache_dir")
    args.fp16_accumulation = config.get("fp16_accumulation", False)
    # Memory tracing (optional)
    args.trace_memory = config.get("trace_memory", False)

    # TREAD configuration (optional)
    # 1) Native TOML tables: tread_config.routes = [ {selection_ratio=..., start_layer_idx=..., end_layer_idx=...}, ... ]
    # 2) Shorthand strings: tread_config_route1 = "selection_ratio=0.2; start_layer_idx=2; end_layer_idx=-2"
    # 3) Simplified frame-based block: tread = { start_layer=2, end_layer=28, keep_ratio=0.5 }
    # Enable flag gates activation
    args.enable_tread = config.get("enable_tread", False)
    args.tread_mode = config.get(
        "tread_mode", "full"
    )  # "full" | "frame_contiguous" | "frame_stride"

    def _parse_route_kv_string(s: str) -> Dict[str, Any]:
        route: Dict[str, Any] = {}
        for part in s.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            k, v = [t.strip() for t in part.split("=", 1)]
            # try to cast to int, then float, else keep string
            try:
                if v.lower() == "true":
                    route[k] = True
                elif v.lower() == "false":
                    route[k] = False
                elif v.startswith("-") and v[1:].isdigit() or v.isdigit():
                    route[k] = int(v)
                else:
                    route[k] = float(v)
            except Exception:
                route[k] = v
        return route

    routes: list[Dict[str, Any]] = []
    # Collect from native TOML if present
    if isinstance(config.get("tread_config"), dict):
        routes.extend(config["tread_config"].get("routes", []) or [])
    # Collect from simplified 'tread' block if present (maps to one route)
    if isinstance(config.get("tread"), dict):
        _t = config["tread"]
        try:
            start_layer = int(_t.get("start_layer", -1))
            end_layer = int(_t.get("end_layer", -1))
            keep_ratio = float(_t.get("keep_ratio", 1.0))
            # Convert frame keep_ratio to token drop selection_ratio used by router path
            selection_ratio = max(0.0, min(1.0, 1.0 - keep_ratio))
            routes.append(
                {
                    "selection_ratio": selection_ratio,
                    "start_layer_idx": start_layer,
                    "end_layer_idx": end_layer,
                }
            )
        except Exception:
            pass
    # Collect shorthand: any top-level key like tread_config_routeX
    for key, val in list(config.items()):
        if isinstance(key, str) and key.lower().startswith("tread_config_route"):
            if isinstance(val, str):
                route = _parse_route_kv_string(val)
                if route:
                    routes.append(route)
    # Normalize: only set when enabled and we have routes
    args.tread_config = (
        {"routes": routes} if args.enable_tread and len(routes) > 0 else None
    )

    # Wan model gated features (RoPE/time embedding/safety)
    args.rope_on_the_fly = bool(config.get("rope_on_the_fly", False))
    args.broadcast_time_embed = bool(config.get("broadcast_time_embed", False))
    args.strict_e_slicing_checks = bool(config.get("strict_e_slicing_checks", False))

    # Dataset config - set to the same as config file since it's included in main config
    args.dataset_config = config_path

    # Training settings
    args.max_train_steps = config.get("max_train_steps", 1600)
    args.prior_loss_weight = config.get("prior_loss_weight", 1.0)

    # START OF DOP ADDITION
    args.diff_output_preservation = config.get("diff_output_preservation", False)
    args.diff_output_preservation_trigger_word = config.get(
        "diff_output_preservation_trigger_word"
    )
    args.diff_output_preservation_class = config.get("diff_output_preservation_class")
    args.diff_output_preservation_multiplier = config.get(
        "diff_output_preservation_multiplier", 1.0
    )
    # END OF DOP ADDITION

    args.max_train_epochs = config.get("max_train_epochs")
    args.max_data_loader_n_workers = config.get("max_data_loader_n_workers", 8)
    args.persistent_data_loader_workers = config.get(
        "persistent_data_loader_workers", False
    )
    args.seed = config.get("seed", 42)
    args.gradient_checkpointing = config.get("gradient_checkpointing", False)
    args.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    args.mixed_precision = config.get("mixed_precision", "no")

    # Optimizer settings
    args.optimizer_type = config.get("optimizer_type", "")
    args.optimizer_args = config.get("optimizer_args", [])
    args.learning_rate = config.get("learning_rate", 5e-5)
    args.max_grad_norm = config.get("max_grad_norm", 1.0)

    # LR Scheduler settings
    args.lr_scheduler = config.get("lr_scheduler", "constant")
    args.lr_warmup_steps = config.get("lr_warmup_steps", 0)
    args.lr_decay_steps = config.get("lr_decay_steps", 0)
    args.lr_scheduler_num_cycles = config.get("lr_scheduler_num_cycles", 1)
    args.lr_scheduler_power = config.get("lr_scheduler_power", 1.0)
    args.lr_scheduler_timescale = config.get("lr_scheduler_timescale", 0.0)
    args.lr_scheduler_min_lr_ratio = config.get("lr_scheduler_min_lr_ratio", 0.0)
    args.lr_scheduler_type = config.get("lr_scheduler_type", "")
    args.lr_scheduler_args = config.get("lr_scheduler_args", "")

    # Network settings
    args.network_module = config.get("network_module", "networks.lora_wan")
    args.network_dim = config.get("network_dim", 32)
    args.network_alpha = config.get("network_alpha", 32)
    args.network_weights = config.get("network_weights")
    args.network_dropout = config.get("network_dropout", 0)
    # Normalize network_args to a list of strings
    raw_net_args = config.get("network_args", [])
    if isinstance(raw_net_args, str):
        # Backward compatibility: allow single string or comma-separated
        if raw_net_args.strip() == "":
            args.network_args = []
        else:
            # split on commas if present, else wrap as single arg
            if "," in raw_net_args:
                args.network_args = [
                    s.strip() for s in raw_net_args.split(",") if s.strip()
                ]
            else:
                args.network_args = [raw_net_args.strip()]
    elif isinstance(raw_net_args, list):
        args.network_args = [str(v) for v in raw_net_args]
    else:
        args.network_args = []

    # Extract LoRA-GGPO parameters from network_args (e.g., "ggpo_sigma=0.03")
    args.ggpo_sigma = None
    args.ggpo_beta = None
    for net_arg in args.network_args:
        if isinstance(net_arg, str) and "=" in net_arg:
            key, value = net_arg.split("=", 1)
            k = key.strip().lower()
            v = value.strip()
            if k == "ggpo_sigma":
                try:
                    args.ggpo_sigma = float(v)
                except Exception:
                    args.ggpo_sigma = None
            elif k == "ggpo_beta":
                try:
                    args.ggpo_beta = float(v)
                except Exception:
                    args.ggpo_beta = None
    args.training_comment = config.get("training_comment", "trained with Takenoko")
    args.dim_from_weights = config.get("dim_from_weights", False)
    args.lycoris = config.get("lycoris", False)
    args.verbose_network = config.get("verbose_network", False)
    args.scale_weight_norms = config.get("scale_weight_norms", None)
    args.base_weights = config.get("base_weights")
    args.base_weights_multiplier = config.get("base_weights_multiplier", 1.0)

    # Reward LoRA settings (prefixed with reward_*)
    args.enable_reward_lora = config.get("enable_reward_lora", False)
    # prompts: either file path or enumerated keys like reward_prompt1, reward_prompt2, ...
    # We do not support reading external files for reward prompts; only TOML
    # derived keys are used. Keep these as None to avoid accidental use.
    args.reward_prompt_path = None
    args.reward_prompt_column = None
    # collect enumerated reward_promptN keys
    reward_prompts: list[str] = []
    for key, val in list(config.items()):
        if (
            isinstance(key, str)
            and key.lower().startswith("reward_prompt")
            and key.lower() != "reward_prompt_path"
        ):
            if isinstance(val, str) and val.strip():
                reward_prompts.append(val)
    args.reward_prompts = reward_prompts

    # core reward hyperparameters (defaults adapted from reference)
    args.reward_train_batch_size = config.get("reward_train_batch_size", 1)
    args.reward_train_sample_height = config.get("reward_train_sample_height", 256)
    args.reward_train_sample_width = config.get("reward_train_sample_width", 256)
    # support alias reward_num_frames -> reward_video_length
    args.reward_video_length = config.get(
        "reward_video_length", config.get("reward_num_frames", 81)
    )
    args.reward_num_inference_steps = config.get("reward_num_inference_steps", 50)
    args.reward_guidance_scale = config.get("reward_guidance_scale", 6.0)
    args.reward_num_decoded_latents = config.get("reward_num_decoded_latents", 1)
    args.reward_validation_steps = config.get("reward_validation_steps", 10000)
    args.reward_validation_prompt_path = config.get("reward_validation_prompt_path")

    # reward function selection
    args.reward_fn = config.get("reward_fn", "HPSReward")
    # keep kwargs as a raw json string to be parsed by the core
    rf_kwargs = config.get("reward_fn_kwargs", None)
    if isinstance(rf_kwargs, dict):
        import json as _json

        args.reward_fn_kwargs = _json.dumps(rf_kwargs)
    else:
        args.reward_fn_kwargs = rf_kwargs

    # backprop strategy
    args.reward_backprop = config.get("reward_backprop", True)
    args.reward_backprop_strategy = config.get("reward_backprop_strategy", "tail")
    args.reward_backprop_num_steps = config.get("reward_backprop_num_steps", 5)
    args.reward_backprop_step_list = config.get("reward_backprop_step_list")
    args.reward_backprop_random_start_step = config.get(
        "reward_backprop_random_start_step", 0
    )
    args.reward_backprop_random_end_step = config.get(
        "reward_backprop_random_end_step", 50
    )
    args.reward_stop_latent_model_input_gradient = config.get(
        "reward_stop_latent_model_input_gradient", False
    )

    # Enhanced progress bar and logging
    args.enhanced_progress_bar = config.get(
        "enhanced_progress_bar", True
    )  # Default to True for better UX
    args.logging_level = config.get("logging_level", "INFO")

    # Attention metrics (gated; off by default)
    args.enable_attention_metrics = bool(config.get("enable_attention_metrics", False))
    args.attention_metrics_interval = int(config.get("attention_metrics_interval", 500))
    args.attention_metrics_max_layers = int(
        config.get("attention_metrics_max_layers", 2)
    )
    args.attention_metrics_max_queries = int(
        config.get("attention_metrics_max_queries", 1024)
    )
    args.attention_metrics_topk = int(config.get("attention_metrics_topk", 16))
    args.attention_metrics_log_prefix = str(
        config.get("attention_metrics_log_prefix", "attn")
    )

    # Attention heatmap logging (fully gated; off by default)
    args.attention_metrics_log_heatmap = bool(
        config.get("attention_metrics_log_heatmap", False)
    )
    args.attention_metrics_heatmap_max_heads = int(
        config.get("attention_metrics_heatmap_max_heads", 1)
    )
    args.attention_metrics_heatmap_max_queries = int(
        config.get("attention_metrics_heatmap_max_queries", 64)
    )
    args.attention_metrics_heatmap_log_prefix = str(
        config.get("attention_metrics_heatmap_log_prefix", "attn_hm")
    )
    # Attention heatmap rendering preferences
    args.attention_metrics_heatmap_cmap = str(
        config.get("attention_metrics_heatmap_cmap", "magma")
    )
    args.attention_metrics_heatmap_norm = str(
        config.get("attention_metrics_heatmap_norm", "log")
    )  # "log" | "linear"
    try:
        args.attention_metrics_heatmap_vmin_pct = float(
            config.get("attention_metrics_heatmap_vmin_pct", 60.0)
        )
    except Exception:
        args.attention_metrics_heatmap_vmin_pct = 60.0
    try:
        args.attention_metrics_heatmap_vmax_pct = float(
            config.get("attention_metrics_heatmap_vmax_pct", 99.5)
        )
    except Exception:
        args.attention_metrics_heatmap_vmax_pct = 99.5
    # Figure size (inches)
    try:
        args.attention_metrics_heatmap_fig_w = float(
            config.get("attention_metrics_heatmap_fig_w", 6.0)
        )
    except Exception:
        args.attention_metrics_heatmap_fig_w = 6.0
    try:
        args.attention_metrics_heatmap_fig_h = float(
            config.get("attention_metrics_heatmap_fig_h", 4.0)
        )
    except Exception:
        args.attention_metrics_heatmap_fig_h = 4.0

    # Extra training metrics (periodic)
    args.log_extra_train_metrics = config.get("log_extra_train_metrics", True)
    args.train_metrics_interval = config.get("train_metrics_interval", 50)
    # Prefer essential SNR metrics under `snr/` and move others to `snr_other/`
    args.snr_split_namespaces = config.get("snr_split_namespaces", True)
    # Prefer essential Validation metrics under `val/` and move others to `val_other/`
    args.val_split_namespaces = config.get("val_split_namespaces", True)
    # Append small emoji hints to TensorBoard tags (e.g., loss 📉, throughput 📈)
    args.tensorboard_append_direction_hints = bool(
        config.get("tensorboard_append_direction_hints", True)
    )
    # EMA loss display config
    args.ema_loss_beta = float(config.get("ema_loss_beta", 0.98))
    args.ema_loss_bias_warmup_steps = int(config.get("ema_loss_bias_warmup_steps", 100))

    # Loss-vs-timestep scatter logging
    args.log_loss_scatterplot = config.get("log_loss_scatterplot", False)
    args.log_loss_scatterplot_interval = config.get(
        "log_loss_scatterplot_interval", 500
    )

    # Control LoRA settings
    args.enable_control_lora = config.get("enable_control_lora", False)
    args.control_lora_type = config.get("control_lora_type", "tile")
    args.control_preprocessing = config.get("control_preprocessing", "blur")
    args.control_blur_kernel_size = config.get("control_blur_kernel_size", 15)
    args.control_blur_sigma = config.get("control_blur_sigma", 4.0)
    args.control_scale_factor = config.get("control_scale_factor", 1.0)
    args.input_lr_scale = config.get("input_lr_scale", 1.0)
    # Match reference default (CFHW -> channel dim 0). Training (BCFHW) remaps to dim=1 at runtime.
    args.control_concatenation_dim = config.get("control_concatenation_dim", 0)
    args.load_control = config.get("load_control", False)
    args.control_suffix = config.get("control_suffix", "_control")
    args.control_inject_noise = config.get("control_inject_noise", 0.0)
    args.save_control_videos = config.get("save_control_videos", False)
    args.control_video_save_all = config.get("control_video_save_all", False)
    args.control_video_save_dir = config.get("control_video_save_dir", "control_videos")

    # ControlNet settings
    args.enable_controlnet = config.get("enable_controlnet", False)
    args.controlnet_weight = config.get("controlnet_weight", 1.0)
    args.controlnet_stride = config.get("controlnet_stride", 1)
    args.controlnet = config.get("controlnet", {})
    # Optional separate gradient clipping for ControlNet
    args.controlnet_max_grad_norm = config.get("controlnet_max_grad_norm")

    args.output_dir = config.get("output_dir", "output")
    args.output_name = config.get("output_name", "wan21_lora")
    args.resume = config.get("resume")
    args.auto_resume = config.get("auto_resume", True)
    args.save_every_n_epochs = config.get("save_every_n_epochs", None)
    args.save_every_n_steps = config.get("save_every_n_steps", 1000)
    args.save_last_n_epochs = config.get("save_last_n_epochs", None)
    args.save_last_n_epochs_state = config.get("save_last_n_epochs_state", None)
    args.save_last_n_steps = config.get("save_last_n_steps", None)
    args.save_last_n_steps_state = config.get("save_last_n_steps_state", None)
    args.save_state = config.get("save_state", True)
    args.save_state_on_train_end = config.get("save_state_on_train_end", False)

    # Sampling settings
    args.sample_every_n_steps = config.get("sample_every_n_steps", None)
    args.sample_at_first = config.get("sample_at_first", False)
    args.sample_every_n_epochs = config.get("sample_every_n_epochs", None)
    args.sample_prompts = config.get("sample_prompts", [])

    # Validation settings
    args.validate_every_n_steps = config.get("validate_every_n_steps", None)
    args.validate_on_epoch_end = config.get("validate_on_epoch_end", False)
    args.validation_timesteps = config.get("validation_timesteps", "500")
    # Dynamic validation timesteps controls
    # mode: "fixed" | "random" | "jitter"
    args.validation_timesteps_mode = config.get("validation_timesteps_mode", "fixed")
    # number of timesteps to draw when mode == "random"
    args.validation_timesteps_count = int(config.get("validation_timesteps_count", 4))
    # bounds for random/jittered timesteps (inclusive). If None, fall back to args.min/max_timestep or [0,1000]
    args.validation_timesteps_min = config.get("validation_timesteps_min")
    args.validation_timesteps_max = config.get("validation_timesteps_max")
    # integer jitter radius applied per base timestep when mode == "jitter"
    args.validation_timesteps_jitter = int(config.get("validation_timesteps_jitter", 0))
    args.use_unique_noise_per_batch = config.get("use_unique_noise_per_batch", True)
    # SNR / perceptual validation toggles
    args.enable_perceptual_snr = config.get("enable_perceptual_snr", False)
    args.perceptual_snr_max_items = config.get("perceptual_snr_max_items", 4)

    # LPIPS validation (optional)
    args.enable_lpips = bool(config.get("enable_lpips", False))
    args.lpips_max_items = int(config.get("lpips_max_items", 2))
    args.lpips_network = str(config.get("lpips_network", "vgg"))  # vgg|alex|squeeze
    args.lpips_frame_stride = int(config.get("lpips_frame_stride", 8))

    # Temporal SSIM (adjacent-frame) validation (optional)
    args.enable_temporal_ssim = bool(config.get("enable_temporal_ssim", False))
    args.temporal_ssim_max_items = int(config.get("temporal_ssim_max_items", 2))
    args.temporal_ssim_frame_stride = int(config.get("temporal_ssim_frame_stride", 1))

    # Temporal LPIPS (adjacent-frame) validation (optional)
    args.enable_temporal_lpips = bool(config.get("enable_temporal_lpips", False))
    args.temporal_lpips_max_items = int(config.get("temporal_lpips_max_items", 2))
    args.temporal_lpips_frame_stride = int(config.get("temporal_lpips_frame_stride", 2))

    # Flow-warped SSIM (RAFT via torchvision, optional)
    args.enable_flow_warped_ssim = bool(config.get("enable_flow_warped_ssim", False))
    args.flow_warped_ssim_model = str(
        config.get("flow_warped_ssim_model", "torchvision_raft_small")
    )
    args.flow_warped_ssim_max_items = int(config.get("flow_warped_ssim_max_items", 2))
    args.flow_warped_ssim_frame_stride = int(
        config.get("flow_warped_ssim_frame_stride", 2)
    )

    # FVD (Fréchet Video Distance) validation (optional)
    args.enable_fvd = bool(config.get("enable_fvd", False))
    args.fvd_model = str(config.get("fvd_model", "torchvision_r3d_18"))
    args.fvd_max_items = int(config.get("fvd_max_items", 2))
    args.fvd_clip_len = int(config.get("fvd_clip_len", 16))
    args.fvd_frame_stride = int(config.get("fvd_frame_stride", 2))

    # VMAF (requires ffmpeg with libvmaf; optional)
    args.enable_vmaf = bool(config.get("enable_vmaf", False))
    args.vmaf_model_path = config.get("vmaf_model_path")  # optional explicit path
    args.vmaf_max_items = int(config.get("vmaf_max_items", 1))
    args.vmaf_clip_len = int(config.get("vmaf_clip_len", 16))
    args.vmaf_frame_stride = int(config.get("vmaf_frame_stride", 2))
    args.vmaf_ffmpeg_path = config.get("vmaf_ffmpeg_path", "ffmpeg")

    # Logging settings
    args.logging_dir = config.get("logging_dir", "logs")

    args.log_with = config.get("log_with", "tensorboard")
    args.log_prefix = config.get("log_prefix", "")
    args.log_tracker_name = config.get("log_tracker_name", "")
    args.log_tracker_config = config.get("log_tracker_config", "")
    args.log_config = config.get("log_config", False)

    # Timestep distribution logging settings
    args.log_timestep_distribution = config.get("log_timestep_distribution", "off")
    args.log_timestep_distribution_interval = config.get(
        "log_timestep_distribution_interval", 1000
    )
    args.log_timestep_distribution_bins = config.get(
        "log_timestep_distribution_bins", 100
    )
    args.log_timestep_distribution_init = config.get(
        "log_timestep_distribution_init", True
    )
    args.log_timestep_distribution_samples = config.get(
        "log_timestep_distribution_samples", 20000
    )
    args.log_timestep_distribution_window = config.get(
        "log_timestep_distribution_window", 10000
    )
    args.log_timestep_distribution_bands = config.get(
        "log_timestep_distribution_bands", "0,100,200,300,400,500,600,700,800,900,1000"
    )
    args.log_timestep_distribution_init_once = config.get(
        "log_timestep_distribution_init_once", True
    )
    args.log_timestep_distribution_pmf = config.get(
        "log_timestep_distribution_pmf", False
    )

    # Throughput metrics logging settings
    args.log_throughput_metrics = config.get("log_throughput_metrics", True)
    args.throughput_window_size = config.get("throughput_window_size", 100)

    # TensorBoard server settings
    args.launch_tensorboard_server = config.get("launch_tensorboard_server", False)
    args.tensorboard_host = config.get("tensorboard_host", "127.0.0.1")
    args.tensorboard_port = config.get("tensorboard_port", 6006)
    args.tensorboard_auto_reload = config.get("tensorboard_auto_reload", True)

    # Acceleration settings
    args.sdpa = config.get("sdpa", False)
    args.flash_attn = config.get("flash_attn", False)
    args.sage_attn = config.get("sage_attn", False)
    args.xformers = config.get("xformers", False)
    args.flash3 = config.get("flash3", False)
    args.split_attn = config.get("split_attn", False)

    # DDP settings
    args.ddp_timeout = config.get("ddp_timeout")
    args.ddp_gradient_as_bucket_view = config.get("ddp_gradient_as_bucket_view", False)
    args.ddp_static_graph = config.get("ddp_static_graph", False)

    # Dynamo settings
    args.dynamo_backend = config.get("dynamo_backend", "NO")
    args.dynamo_mode = config.get("dynamo_mode", "default")
    args.dynamo_fullgraph = config.get("dynamo_fullgraph", False)
    args.dynamo_dynamic = config.get("dynamo_dynamic", False)

    # Full precision settings (commented out in parser but used in code)
    args.full_fp16 = config.get("full_fp16", False)
    args.full_bf16 = config.get("full_bf16", False)

    # Timestep and flow matching settings
    args.timestep_sampling = config.get("timestep_sampling", "shift")

    # Parse new timestep optimization flags
    args.use_precomputed_timesteps = config.get("use_precomputed_timesteps", False)
    args.precomputed_timestep_buckets = config.get(
        "precomputed_timestep_buckets", 10000
    )
    args.discrete_flow_shift = config.get("discrete_flow_shift", 3.0)
    args.sigmoid_scale = config.get("sigmoid_scale", 1.0)
    # Enhanced sigmoid optional bias
    args.sigmoid_bias = config.get("sigmoid_bias", 0.0)
    # Bell-shaped distribution parameters
    args.bell_center = config.get("bell_center", 0.5)
    args.bell_std = config.get("bell_std", 0.2)
    # LogNormal blend control
    args.lognorm_blend_alpha = config.get("lognorm_blend_alpha", 0.75)
    # Content/style blend control
    args.content_style_blend_ratio = config.get("content_style_blend_ratio", 0.5)

    # Performance logging verbosity
    args.performance_verbosity = config.get("performance_verbosity", "standard")
    args.weighting_scheme = config.get("weighting_scheme", "none")
    args.logit_mean = config.get("logit_mean", 0.0)
    args.logit_std = config.get("logit_std", 1.0)
    args.mode_scale = config.get("mode_scale", 1.29)
    args.min_timestep = config.get("min_timestep", 0)
    args.max_timestep = config.get("max_timestep", 1000)
    args.skip_extra_timestep_constraint = config.get(
        "skip_extra_timestep_constraint", True
    )
    args.fast_rejection_sampling = config.get("fast_rejection_sampling", False)
    # Fine-tuning knobs for fast rejection sampling
    args.rejection_overdraw_factor = config.get("rejection_overdraw_factor", 4.0)
    args.rejection_max_iters = config.get("rejection_max_iters", 10)
    args.timestep_constraint_epsilon = config.get("timestep_constraint_epsilon", 1e-6)
    # Optional: round training timesteps to the nearest integer schedule step
    args.round_training_timesteps = config.get("round_training_timesteps", False)
    args.preserve_distribution_shape = config.get("preserve_distribution_shape", False)
    # Optional override for precomputed mid-shift area (used by flux/qwen/qinglong precompute path)
    args.precomputed_midshift_area = config.get("precomputed_midshift_area")
    args.show_timesteps = config.get("show_timesteps")
    args.guidance_scale = config.get("guidance_scale", 1.0)

    # Offloading settings
    args.blocks_to_swap = config.get("blocks_to_swap", 0)
    args.allow_mixed_block_swap_offload = config.get(
        "allow_mixed_block_swap_offload", False
    )

    # Dual model training settings
    args.enable_dual_model_training = config.get("enable_dual_model_training", False)
    args.dit_high_noise = config.get("dit_high_noise")
    args.timestep_boundary = config.get("timestep_boundary", 875)
    args.offload_inactive_dit = config.get("offload_inactive_dit", True)

    # Dual-mode timestep bucketing strategy
    args.dual_timestep_bucket_strategy = config.get(
        "dual_timestep_bucket_strategy", "hybrid"
    )
    args.dual_timestep_bucket_max_retries = config.get(
        "dual_timestep_bucket_max_retries", 100
    )
    args.dual_timestep_bucket_eps = config.get("dual_timestep_bucket_eps", 1e-4)

    # Metadata settings
    args.no_metadata = config.get("no_metadata", False)
    args.embed_config_in_metadata = config.get("embed_config_in_metadata", True)
    args.metadata_title = config.get("metadata_title", "")
    args.metadata_author = config.get("metadata_author", "")
    args.metadata_description = config.get("metadata_description", "")
    args.metadata_license = config.get("metadata_license", "")
    args.metadata_tags = config.get("metadata_tags", "")

    # Device settings
    args.device = config.get("device")

    # Fluxflow settings
    args.enable_fluxflow = config.get("enable_fluxflow", False)
    args.fluxflow_mode = config.get("fluxflow_mode", "frame")
    args.fluxflow_frame_perturb_ratio = config.get("fluxflow_frame_perturb_ratio", 0.25)
    args.fluxflow_block_size = config.get("fluxflow_block_size", 4)
    args.fluxflow_block_perturb_prob = config.get("fluxflow_block_perturb_prob", 0.5)
    args.fluxflow_frame_dim_in_batch = config.get("fluxflow_frame_dim_in_batch", 2)

    # FVDM settings
    args.enable_fvdm = config.get("enable_fvdm", False)
    args.fvdm_ptss_p = config.get("fvdm_ptss_p", 0.2)

    # FOPP settings
    args.fopp_num_timesteps = config.get("fopp_num_timesteps", 1000)
    args.fopp_schedule_type = config.get("fopp_schedule_type", "linear")
    args.fopp_beta_start = config.get("fopp_beta_start", 0.0001)
    args.fopp_beta_end = config.get("fopp_beta_end", 0.002)
    args.fopp_seed = config.get("fopp_seed", None)

    # Nabla settings
    args.nabla_sparse_attention = config.get("nabla_sparse_attention", False)
    args.nabla_sparse_algo = config.get("nabla_sparse_algo", "nabla-0.7_sta-11-24-24")

    # Contrastive Flow Matching (ΔFM) settings
    args.enable_contrastive_flow_matching = config.get(
        "enable_contrastive_flow_matching", False
    )
    args.contrastive_flow_lambda = config.get("contrastive_flow_lambda", 0.05)

    # Timestep bucketing (dataset-driven, per-epoch stratified uniform pool)
    # None disables bucketing; set to >=2 to enable
    args.num_timestep_buckets = config.get("num_timestep_buckets")

    # Dispersive Loss Regularization settings
    args.enable_dispersive_loss = config.get("enable_dispersive_loss", False)
    args.dispersive_loss_lambda = config.get("dispersive_loss_lambda", 0.0)
    args.dispersive_loss_tau = config.get("dispersive_loss_tau", 0.5)
    # None disables extraction, non-negative integer selects a block index
    args.dispersive_loss_target_block = config.get("dispersive_loss_target_block")
    args.dispersive_loss_metric = config.get(
        "dispersive_loss_metric", "l2_sq"
    )  # "l2_sq" or "cosine"
    # Optional: pool spatial tokens per frame before dispersion ("none" or "frame_mean")
    args.dispersive_loss_pooling = config.get("dispersive_loss_pooling", "none")

    # Optical Flow Loss (RAFT-based) settings
    args.enable_optical_flow_loss = config.get("enable_optical_flow_loss", False)
    args.lambda_optical_flow = config.get("lambda_optical_flow", 0.0)

    # REPA (Representation Alignment) settings
    args.enable_repa = config.get("enable_repa", False)
    args.repa_encoder_name = config.get("repa_encoder_name", "dinov2_vitb14")
    args.repa_alignment_depth = config.get("repa_alignment_depth", 8)
    args.repa_loss_lambda = config.get("repa_loss_lambda", 0.5)
    args.repa_similarity_fn = config.get("repa_similarity_fn", "cosine")

    # Use original config file directly - no need for temporary files!
    # The caching scripts can handle full config files and extract what they need

    # Validate that we have dataset configuration
    if "datasets" not in config and "val_datasets" not in config:
        raise ValueError(
            "No dataset configuration found in the config file. Please include [[datasets]] and/or [[val_datasets]] sections."
        )

    # Validate the dataset configuration
    logger.info("🔍 Validating dataset configuration...")
    try:
        validate_dataset_config(args.dataset_config, test_dataset_creation=False)
        logger.info("✅ Dataset configuration validation passed!")
    except Exception as e:
        logger.exception(f"❌ Dataset configuration validation failed: {e}")
        raise ValueError(f"Dataset configuration validation failed: {e}")

    # Set default values for compatibility
    args.dit_dtype = None  # automatically detected
    if args.vae_dtype is None:
        args.vae_dtype = "float16"  # make float16 as default for VAE

    # Read latent cache settings from config
    if "datasets" in config and "latent_cache" in config["datasets"]:
        latent_cache_config = config["datasets"]["latent_cache"]
        args.vae = latent_cache_config.get("vae", args.vae)
        args.vae_cache_cpu = latent_cache_config.get(
            "vae_cache_cpu", args.vae_cache_cpu
        )
        args.vae_dtype = latent_cache_config.get("vae_dtype", args.vae_dtype)
        args.latent_cache_device = latent_cache_config.get("device", args.device)
        args.latent_cache_batch_size = latent_cache_config.get("batch_size")
        args.latent_cache_num_workers = latent_cache_config.get("num_workers")
        args.latent_cache_skip_existing = latent_cache_config.get(
            "skip_existing", False
        )
        args.latent_cache_keep_cache = latent_cache_config.get("keep_cache", False)
        args.latent_cache_purge = latent_cache_config.get("purge_before_run", False)
        args.latent_cache_debug_mode = latent_cache_config.get("debug_mode")
        args.latent_cache_console_width = latent_cache_config.get("console_width", 80)
        args.latent_cache_console_back = latent_cache_config.get(
            "console_back", "black"
        )
        args.latent_cache_console_num_images = latent_cache_config.get(
            "console_num_images", 1
        )
    else:
        # Set defaults for latent cache if section not found
        args.latent_cache_device = args.device
        args.latent_cache_batch_size = None
        args.latent_cache_num_workers = None
        args.latent_cache_skip_existing = False
        args.latent_cache_keep_cache = False
        args.latent_cache_purge = False
        args.latent_cache_debug_mode = None
        args.latent_cache_console_width = 80
        args.latent_cache_console_back = "black"
        args.latent_cache_console_num_images = 1

    # Read text encoder cache settings from config
    if "datasets" in config and "text_encoder_cache" in config["datasets"]:
        text_encoder_cache_config = config["datasets"]["text_encoder_cache"]
        args.t5 = text_encoder_cache_config.get("t5", args.t5)
        args.fp8_t5 = text_encoder_cache_config.get("fp8_t5", args.fp8_t5)
        args.text_encoder_cache_device = text_encoder_cache_config.get(
            "device", args.device
        )
        args.text_encoder_cache_batch_size = text_encoder_cache_config.get("batch_size")
        args.text_encoder_cache_num_workers = text_encoder_cache_config.get(
            "num_workers"
        )
        args.text_encoder_cache_skip_existing = text_encoder_cache_config.get(
            "skip_existing", False
        )
        args.text_encoder_cache_keep_cache = text_encoder_cache_config.get(
            "keep_cache", False
        )
        args.text_encoder_cache_purge = text_encoder_cache_config.get(
            "purge_before_run", False
        )
    else:
        # Set defaults for text encoder cache if section not found
        args.text_encoder_cache_device = args.device
        args.text_encoder_cache_batch_size = None
        args.text_encoder_cache_num_workers = None
        args.text_encoder_cache_skip_existing = False
        args.text_encoder_cache_keep_cache = False
        args.text_encoder_cache_purge = False

    # No explicit load_mask toggle: mask_path presence implies mask loading

    # Self-correction (draft, fully gated by self_correction_enabled) - flat keys only
    args.self_correction_enabled = config.get("self_correction_enabled", False)
    args.self_correction_warmup_steps = config.get("self_correction_warmup_steps", 1000)
    args.self_correction_update_frequency = config.get(
        "self_correction_update_frequency", 1000
    )
    args.self_correction_cache_size = config.get("self_correction_cache_size", 200)
    args.self_correction_clip_len = config.get("self_correction_clip_len", 32)
    args.self_correction_batch_ratio = config.get("self_correction_batch_ratio", 0.2)
    args.self_correction_sample_steps = config.get("self_correction_sample_steps", 16)
    args.self_correction_width = config.get("self_correction_width", 256)
    args.self_correction_height = config.get("self_correction_height", 256)
    args.self_correction_guidance_scale = config.get(
        "self_correction_guidance_scale", 5.0
    )

    # Inline self-correction prompts in config (preferred over external files)
    args.self_correction_prompts = None
    if "self_correction_prompts" in config:
        try:
            sc_prompts = config.get("self_correction_prompts", [])
            if isinstance(sc_prompts, list):
                normalized: list[dict] = []
                for i, p in enumerate(sc_prompts):
                    if isinstance(p, str):
                        normalized.append({"text": p, "enum": i})
                    elif isinstance(p, dict):
                        d = dict(p)
                        if "text" not in d:
                            # If a dict without text appears, skip it gracefully
                            continue
                        d.setdefault("enum", i)
                        normalized.append(d)
                args.self_correction_prompts = normalized
        except Exception:
            args.self_correction_prompts = None

    return args


# Removed cleanup_temp_files - no longer needed since we don't create temporary files


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


def _estimate_peak_vram_gb_from_config(
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    return shared_estimate_vram(config)


class UnifiedTrainer:
    """Unified trainer that handles caching and training operations"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config, self.config_content = load_training_config(config_path)
        self.args = create_args_from_config(
            self.config, config_path, self.config_content
        )

        # Configure CUDA allocator from TOML before any CUDA initialization
        try:
            configure_cuda_allocator_from_config(self.config, logger)
        except Exception as _cuda_alloc_err:
            # Non-fatal: proceed with default allocator if misconfigured
            logger.debug(f"CUDA allocator config skipped: {_cuda_alloc_err}")

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

    def show_menu(self) -> str:
        """Display the main menu and get user choice"""
        print("\n" + "=" * 50)
        print("Takenoko - Unified Operations Menu")
        print("=" * 50)
        print("1. Cache Latents")
        print("2. Cache Text Encoder Outputs")
        print("3. Train Model")
        print("4. Estimate VRAM Usage (from current config)")
        print("5. Estimate latent cache chunks (by frame extraction mode)")
        print("6. Reload Config File")
        print("7. Free VRAM (aggressive)")
        print("8. Return to Config Selection")
        print("=" * 50)

        while True:
            choice = input("Enter your choice (1-8): ").strip()
            if choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                return choice
            else:
                print("Invalid choice. Please enter 1-8.")

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
            f"   Preserve distribution shape: {bool(getattr(args, 'preserve_distribution_shape', False))}"
        )
        logger.info(
            f"   Skip extra in-range constraint: {bool(getattr(args, 'skip_extra_timestep_constraint', False))}"
        )
        logger.info(
            f"   Constraint epsilon: {float(getattr(args, 'timestep_constraint_epsilon', 1e-6))}"
        )
        if bool(getattr(args, "preserve_distribution_shape", False)):
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

        # Check mixed precision settings
        mixed_precision = getattr(args, "mixed_precision", "no")
        if mixed_precision != "no":
            acceleration_methods.append(f"Mixed Precision ({mixed_precision})")

        # Check full precision settings
        if getattr(args, "full_fp16", False):
            acceleration_methods.append("Full FP16")
        if getattr(args, "full_bf16", False):
            acceleration_methods.append("Full BF16")

        # Check FP8 settings
        if getattr(args, "fp8_scaled", False):
            acceleration_methods.append("FP8 Scaled")
        if getattr(args, "fp8_base", False):
            acceleration_methods.append("FP8 Base")
        if getattr(args, "fp8_t5", False):
            acceleration_methods.append("FP8 T5")

        # Check gradient checkpointing
        if getattr(args, "gradient_checkpointing", False):
            acceleration_methods.append("Gradient Checkpointing")

        # Check Dynamo settings
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

        try:
            # Use the unified trainer directly
            trainer = WanNetworkTrainer()
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

        while True:
            choice = self.show_menu()

            if choice == "1":
                success = self.cache_latents()
                if not success:
                    logger.error(
                        "Latent caching failed. Please check the error messages above."
                    )
                    input("Press Enter to continue...")

            elif choice == "2":
                success = self.cache_text_encoder_outputs()
                if not success:
                    logger.error(
                        "Text encoder output caching failed. Please check the error messages above."
                    )
                    input("Press Enter to continue...")

            elif choice == "3":
                success = self.train_model()
                if not success:
                    logger.error(
                        "Training failed. Please check the error messages above."
                    )
                    input("Press Enter to continue...")

            elif choice == "4":
                try:
                    gb, details = _estimate_peak_vram_gb_from_config(self.config)
                    logger.info(
                        "🧠 Estimated peak VRAM usage (per device): {:.2f} GB".format(
                            gb
                        )
                    )
                    logger.info(
                        "   Shape: B={} F={} H={} W={} → lat {}x{} tokens={}".format(
                            details["batch_size"],
                            details["frames"],
                            details["height"],
                            details["width"],
                            details["lat_h"],
                            details["lat_w"],
                            details["tokens_per_sample"],
                        )
                    )
                    logger.info(
                        "   Precision={} ({} bytes/elem), checkpointing={}, control_lora={}, dual={} (offload_inactive_dit={})".format(
                            details["mixed_precision"],
                            details["bytes_per_elem"],
                            details["gradient_checkpointing"],
                            details["enable_control_lora"],
                            details["enable_dual_model_training"],
                            details["offload_inactive_dit"],
                        )
                    )
                    logger.info(
                        "   Breakdown: activations={:.2f} GB, latents={:.2f} GB, text={:.2f} GB, overhead≈{:.2f} GB".format(
                            details["activations_gb"],
                            details["latents_gb"],
                            details["text_gb"],
                            details["overhead_gb"],
                        )
                    )
                except Exception as e:
                    logger.exception(f"❌ Error estimating VRAM usage: {e}")
                input("Press Enter to continue...")

            elif choice == "5":
                try:
                    # Show both total and per-dataset breakdown
                    from caching.chunk_estimator import (
                        estimate_latent_cache_chunks,
                        estimate_latent_cache_chunks_per_dataset,
                    )

                    total_chunks = estimate_latent_cache_chunks(
                        self.args.dataset_config, self.args
                    )
                    per_ds = estimate_latent_cache_chunks_per_dataset(
                        self.args.dataset_config, self.args
                    )
                    logger.info(
                        f"🧮 Estimated latent cache chunks: {total_chunks} (across all video datasets)"
                    )
                    for entry in per_ds:
                        logger.info(
                            f"   - {entry['video_directory']}: {entry['chunks']} chunks"
                        )
                except Exception as e:
                    logger.exception(f"❌ Error estimating cache chunks: {e}")
                input("Press Enter to continue...")

            elif choice == "6":
                success = self.reload_config()
                if not success:
                    logger.error(
                        "Config reload failed. Please check the error messages above."
                    )
                input("Press Enter to continue...")

            elif choice == "7":
                success = self.free_vram_aggressively()
                if not success:
                    logger.error("VRAM cleanup failed.")
                input("Press Enter to continue...")

            elif choice == "8":
                logger.info("Returning to config selection menu...")
                sys.exit(100)

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
