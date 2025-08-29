import argparse
import logging
from typing import Any, Dict
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

from dataset.config_utils import (
    validate_dataset_config,
)


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
    args.fp8_format = config.get("fp8_format", "e4m3")
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

    # Optional: lean attention math to reduce fp32 intermediates in Wan2.2 blocks
    args.lean_attn_math = bool(config.get("lean_attn_math", False))
    # Lean attention compute policy: default to fp32 unless disabled
    args.lean_attention_fp32_default = bool(
        config.get("lean_attention_fp32_default", False)
    )
    # RoPE variant (advanced)
    args.rope_func = str(config.get("rope_func", "default"))

    # Optional: force lower precision attention compute (fp16) for additional VRAM savings
    args.lower_precision_attention = bool(
        config.get("lower_precision_attention", False)
    )
    # Optional: use Wan 2.1 style modulation on Wan 2.2 to save VRAM
    args.simple_modulation = bool(config.get("simple_modulation", False))
    # Optional: optimized selective compile for critical paths (safely gated)
    args.optimized_torch_compile = bool(config.get("optimized_torch_compile", False))
    # Optional: torch.compile args for optimized_torch_compile
    _ca = config.get("compile_args")
    if isinstance(_ca, list) and len(_ca) == 4:
        args.compile_args = _ca
    else:
        # Default to inductor if not specified
        args.compile_args = ["inductor", "default", "auto", "False"]

    # Dataset config - set to the same as config file since it's included in main config
    args.dataset_config = config_path

    # Training settings
    args.max_train_steps = config.get("max_train_steps", 1600)
    args.prior_loss_weight = config.get("prior_loss_weight", 1.0)

    # DOP
    args.diff_output_preservation = config.get("diff_output_preservation", False)
    args.diff_output_preservation_trigger_word = config.get(
        "diff_output_preservation_trigger_word"
    )
    args.diff_output_preservation_class = config.get("diff_output_preservation_class")
    args.diff_output_preservation_multiplier = config.get(
        "diff_output_preservation_multiplier", 1.0
    )

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
    args.fvd_model = str(
        config.get("fvd_model", "torchvision_r3d_18")
    )  # torchvision_r3d_18|reference_i3d
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

    # Validation data pixels loading
    # When enabled, validation datasets will include original/decoded pixels in batches
    # as `batch["pixels"]`, allowing perceptual metrics without altering model inputs.
    args.load_val_pixels = bool(config.get("load_val_pixels", False))

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
    # Progress postfix alternation settings
    args.alternate_perf_postfix = config.get("alternate_perf_postfix", True)

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

    # Enable adaptive timestep sampling (default: false)
    args.enable_adaptive_timestep_sampling = config.get(
        "enable_adaptive_timestep_sampling", False
    )

    # Core parameters
    args.adaptive_focus_strength = config.get("adaptive_focus_strength", 2.0)
    args.adaptive_warmup_steps = config.get("adaptive_warmup_steps", 500)
    args.adaptive_analysis_window = config.get("adaptive_analysis_window", 1000)
    args.adaptive_importance_threshold = config.get(
        "adaptive_importance_threshold", 1.5
    )
    args.adaptive_update_frequency = config.get("adaptive_update_frequency", 100)

    # Timestep constraints
    args.adaptive_min_timesteps = config.get("adaptive_min_timesteps", 50)
    args.adaptive_max_timesteps = config.get("adaptive_max_timesteps", 200)

    # Video-specific features
    args.adaptive_video_specific = config.get("adaptive_video_specific", True)
    args.adaptive_motion_weight = config.get("adaptive_motion_weight", 1.0)
    args.adaptive_detail_weight = config.get("adaptive_detail_weight", 1.0)
    args.adaptive_temporal_weight = config.get("adaptive_temporal_weight", 1.0)

    # Research alignment parameters
    args.adaptive_use_beta_sampler = config.get("adaptive_use_beta_sampler", False)
    args.adaptive_feature_selection_size = config.get(
        "adaptive_feature_selection_size", 3
    )
    args.adaptive_sampler_update_frequency = config.get(
        "adaptive_sampler_update_frequency", 40
    )
    args.adaptive_use_neural_sampler = config.get("adaptive_use_neural_sampler", False)
    args.adaptive_beta_alpha_init = config.get("adaptive_beta_alpha_init", 1.0)
    args.adaptive_beta_beta_init = config.get("adaptive_beta_beta_init", 1.0)
    args.adaptive_neural_hidden_size = config.get("adaptive_neural_hidden_size", 64)

    # Enhanced research modes
    args.adaptive_kl_exact_mode = config.get("adaptive_kl_exact_mode", False)
    args.adaptive_comparative_logging = config.get(
        "adaptive_comparative_logging", False
    )
    args.adaptive_research_mode_enabled = config.get(
        "adaptive_research_mode_enabled", False
    )

    # Complementary approach configuration
    args.adaptive_use_importance_weighting = config.get(
        "adaptive_use_importance_weighting", True
    )
    args.adaptive_use_kl_reward_learning = config.get(
        "adaptive_use_kl_reward_learning", False
    )
    args.adaptive_use_replay_buffer = config.get("adaptive_use_replay_buffer", False)
    args.adaptive_use_statistical_features = config.get(
        "adaptive_use_statistical_features", False
    )
    args.adaptive_weight_combination = config.get(
        "adaptive_weight_combination", "fallback"
    )
    args.adaptive_replay_buffer_size = config.get("adaptive_replay_buffer_size", 100)
    args.adaptive_rl_learning_rate = config.get("adaptive_rl_learning_rate", 1e-4)
    args.adaptive_entropy_coefficient = config.get("adaptive_entropy_coefficient", 0.01)
    args.adaptive_kl_update_frequency = config.get("adaptive_kl_update_frequency", 20)

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
