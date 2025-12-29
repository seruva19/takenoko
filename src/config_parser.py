import argparse
import logging
from typing import Any, Dict
from common.logger import get_logger
from enhancements.semanticgen.config import parse_semanticgen_config
from memory.config import parse_memory_optimization_config
from optimizers.q_galore_config import apply_q_galore_config
from transition.configuration import parse_transition_config

logger = get_logger(__name__, level=logging.INFO)

from dataset.config_utils import (
    validate_dataset_config,
)
from configs.relora_config import apply_relora_config
from configs.glance_config import apply_glance_config


def create_args_from_config(
    config: Dict[str, Any], config_path: str = None, config_content: str = None  # type: ignore
) -> argparse.Namespace:
    """Convert config dictionary to argparse.Namespace for compatibility"""
    # Removed pprint import - not needed

    args = argparse.Namespace()

    # Store original config information for state saving
    args.config_file = config_path
    args.config_content = config_content
    args.raw_config = config

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

    args.fp8_percentile = config.get("fp8_percentile", None)
    args.fp8_exclude_keys = config.get("fp8_exclude_keys", None)

    # Enhanced FP8 quantization parameters (new features - gated by fp8_use_enhanced)
    args.fp8_use_enhanced = bool(config.get("fp8_use_enhanced", True))
    args.fp8_quantization_mode = config.get("fp8_quantization_mode", "block")
    args.fp8_block_size = config.get("fp8_block_size", 64)
    args.fp8_block_wise_fallback_to_channel = bool(
        config.get("fp8_block_wise_fallback_to_channel", True)
    )

    # TorchAO FP8 quantization parameters (alternative backend)
    args.torchao_fp8_enabled = bool(config.get("torchao_fp8_enabled", False))
    args.torchao_fp8_weight_dtype = config.get("torchao_fp8_weight_dtype", "e4m3fn")
    args.torchao_fp8_target_modules = config.get("torchao_fp8_target_modules", None)
    args.torchao_fp8_exclude_modules = config.get("torchao_fp8_exclude_modules", None)
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
    args.trace_memory = config.get("trace_memory", True)
    # Safetensors loading optimisations (gated; default disabled)
    args.enable_memory_mapping = bool(config.get("enable_memory_mapping", False))
    args.enable_zero_copy_loading = bool(config.get("enable_zero_copy_loading", False))
    args.enable_non_blocking_transfers = bool(
        config.get("enable_non_blocking_transfers", False)
    )
    args.memory_mapping_threshold = int(
        config.get("memory_mapping_threshold", 10 * 1024 * 1024)
    )

    args.transition_training = parse_transition_config(config)

    # RamTorch linear replacement
    args.use_ramtorch_linear = bool(config.get("use_ramtorch_linear", False))
    args.ramtorch_device = config.get("ramtorch_device", None)
    args.ramtorch_strength = float(config.get("ramtorch_strength", 1.0))
    args.ramtorch_min_features = int(config.get("ramtorch_min_features", 0))
    args.ramtorch_verbose = bool(config.get("ramtorch_verbose", False))
    args.ramtorch_fp32_io = bool(config.get("ramtorch_fp32_io", True))

    # TREAD configuration (optional)
    # 1) Native TOML tables: tread_config.routes = [ {selection_ratio=..., start_layer_idx=..., end_layer_idx=...}, ... ]
    # 2) Shorthand strings: tread_config_route1 = "selection_ratio=0.2; start_layer_idx=2; end_layer_idx=-2"
    # 3) Simplified frame-based block: tread = { start_layer=2, end_layer=28, keep_ratio=0.5 }
    # Enable flag gates activation
    args.enable_tread = config.get("enable_tread", False)
    # Enhanced TREAD mode validation with spatial routing support
    specified_tread_mode = config.get("tread_mode", "full")
    valid_modes = {
        "full",  # Content-aware routing (existing)
        "frame_contiguous",  # Frame-based routing (existing)
        "frame_stride",  # Frame-based routing (existing)
        "row_contiguous",  # Row-based routing (new)
        "row_stride",  # Row-based routing (new)
        "row_random",  # Row-based routing (new)
        "spatial_auto",  # Auto-detection: F=1→rows, F>1→frames (new)
    }

    if specified_tread_mode not in valid_modes:
        logger.warning(
            f"Invalid tread_mode '{specified_tread_mode}'. Supported modes: {sorted(valid_modes)}. "
            f"Falling back to 'full'."
        )
        specified_tread_mode = "full"

    args.tread_mode = specified_tread_mode

    # Fallback control for row-based TREAD with mixed datasets
    args.row_tread_auto_fallback = config.get("row_tread_auto_fallback", True)

    # Add descriptive logging for new modes
    if args.enable_tread and specified_tread_mode.startswith("row_"):
        logger.info(
            f"Row-based TREAD enabled: mode='{specified_tread_mode}' (spatial routing for images)"
        )
        if args.row_tread_auto_fallback:
            logger.info("Auto-fallback enabled: video content will use frame routing")
    elif args.enable_tread and specified_tread_mode == "spatial_auto":
        logger.info("Spatial auto TREAD enabled: F=1→rows, F>1→frames (hybrid mode)")

    # Validate fallback setting
    if args.row_tread_auto_fallback and not isinstance(
        args.row_tread_auto_fallback, bool
    ):
        logger.warning("row_tread_auto_fallback must be boolean, defaulting to True")
        args.row_tread_auto_fallback = True

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
    # RoPE precision optimization (float32 vs float64)
    args.rope_use_float32 = bool(config.get("rope_use_float32", False))

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
    args.diff_output_preservation = bool(config.get("diff_output_preservation", False))
    args.diff_output_preservation_trigger_word = config.get(
        "diff_output_preservation_trigger_word"
    )
    args.diff_output_preservation_class = config.get("diff_output_preservation_class")
    args.diff_output_preservation_multiplier = float(
        config.get("diff_output_preservation_multiplier", 1.0)
    )

    args.blank_prompt_preservation = bool(
        config.get("blank_prompt_preservation", False)
    )
    _bpp_multiplier = config.get("blank_prompt_preservation_multiplier", 1.0)
    try:
        args.blank_prompt_preservation_multiplier = float(_bpp_multiplier)
    except (TypeError, ValueError):
        raise ValueError(
            "blank_prompt_preservation_multiplier must be a numeric value"
        ) from None
    if args.blank_prompt_preservation_multiplier < 0.0:
        raise ValueError("blank_prompt_preservation_multiplier must be >= 0.0")

    if args.diff_output_preservation and args.blank_prompt_preservation:
        raise ValueError(
            "Cannot enable both diff_output_preservation and blank_prompt_preservation."
        )

    if args.blank_prompt_preservation:
        logger.info(
            "🧭 Blank prompt preservation enabled (multiplier %.3f)",
            args.blank_prompt_preservation_multiplier,
        )

    args.max_train_epochs = config.get("max_train_epochs")
    args.max_data_loader_n_workers = config.get("max_data_loader_n_workers", 8)
    args.persistent_data_loader_workers = config.get(
        "persistent_data_loader_workers", False
    )
    args.data_loader_pin_memory = bool(config.get("data_loader_pin_memory", False))
    _prefetch_factor = config.get("data_loader_prefetch_factor", 0)
    try:
        args.data_loader_prefetch_factor = int(_prefetch_factor)
    except (TypeError, ValueError):
        raise ValueError("data_loader_prefetch_factor must be an int") from None
    if args.data_loader_prefetch_factor < 0:
        raise ValueError("data_loader_prefetch_factor must be >= 0")
    args.bucket_shuffle_across_datasets = bool(
        config.get("bucket_shuffle_across_datasets", False)
    )
    args.seed = config.get("seed", 42)
    args.gradient_checkpointing = config.get("gradient_checkpointing", False)
    args.gradient_checkpointing_cpu_offload = config.get(
        "gradient_checkpointing_cpu_offload", False
    )
    args.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    args.mixed_precision = config.get("mixed_precision", "no")

    # VAE training knobs (defaults preserve legacy behaviour)
    args.vae_training_mode = str(config.get("vae_training_mode", "full"))
    args.vae_kl_weight = float(config.get("vae_kl_weight", 1e-6))
    args.vae_reconstruction_loss = str(config.get("vae_reconstruction_loss", "mse"))
    args.vae_mse_weight = float(config.get("vae_mse_weight", 1.0))
    args.vae_mae_weight = float(config.get("vae_mae_weight", 0.0))
    args.vae_lpips_weight = float(config.get("vae_lpips_weight", 0.0))
    args.vae_edge_weight = float(config.get("vae_edge_weight", 0.0))
    args.vae_loss_balancer_window = int(config.get("vae_loss_balancer_window", 0))
    args.vae_loss_balancer_percentile = int(
        config.get("vae_loss_balancer_percentile", 95)
    )

    _decoder_mean_default = args.vae_training_mode == "decoder_only"
    args.vae_decoder_latent_mean = bool(
        config.get("vae_decoder_latent_mean", _decoder_mean_default)
    )

    # Stochastic rounding for BF16 training stability
    args.use_stochastic_rounding = config.get("use_stochastic_rounding", True)
    args.use_stochastic_rounding_cuda = config.get(
        "use_stochastic_rounding_cuda", False
    )

    # Optimizer settings
    args.optimizer_type = config.get("optimizer_type", "")
    args.optimizer_args = config.get("optimizer_args", [])
    args = apply_q_galore_config(args, config, logger)
    args.ivon_ess = config.get("ivon_ess")
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

    # Slider training configuration (detected by network_module)
    args.slider_guidance_strength = config.get("slider_guidance_strength", 3.0)
    args.slider_anchor_strength = config.get("slider_anchor_strength", 1.0)

    # Advanced guidance parameters
    args.slider_guidance_scale = config.get("slider_guidance_scale", 1.0)
    args.slider_guidance_embedding_scale = config.get(
        "slider_guidance_embedding_scale", 1.0
    )
    args.slider_target_guidance_scale = config.get("slider_target_guidance_scale", 1.0)

    # Prompt configuration
    args.slider_positive_prompt = config.get("slider_positive_prompt", "")
    args.slider_negative_prompt = config.get("slider_negative_prompt", "")
    args.slider_target_class = config.get("slider_target_class", "")
    args.slider_anchor_class = config.get("slider_anchor_class", None)

    # Training parameters
    args.slider_learning_rate_multiplier = config.get(
        "slider_learning_rate_multiplier", 1.0
    )
    args.slider_cache_embeddings = config.get("slider_cache_embeddings", True)

    # T5 text encoder settings (follows Takenoko's pattern)
    args.slider_t5_device = config.get("slider_t5_device", "cpu")
    args.slider_cache_on_init = config.get("slider_t5_cache_on_init", True)

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

    apply_relora_config(args, config, logger)
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

    from polylora.config import apply_polylora_to_args

    args = apply_polylora_to_args(args, config)

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

    # Gradient norm and parameter statistics logging
    args.log_gradient_norm = config.get("log_gradient_norm", False)
    args.log_param_stats = config.get("log_param_stats", False)
    args.param_stats_every_n_steps = config.get("param_stats_every_n_steps", 100)
    args.max_param_stats_logged = config.get("max_param_stats_logged", 20)
    args.log_per_source_loss = config.get("log_per_source_loss", False)

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
    # Optional model weight EMA (averaged parameters for eval/saving)
    args.enable_weight_ema = bool(config.get("enable_weight_ema", False))
    args.weight_ema_decay = float(config.get("weight_ema_decay", 0.999))
    if not 0.0 < args.weight_ema_decay < 1.0:
        raise ValueError("weight_ema_decay must be between 0 and 1 (exclusive)")
    args.weight_ema_start_step = max(int(config.get("weight_ema_start_step", 0)), 0)
    args.weight_ema_trainable_only = bool(config.get("weight_ema_trainable_only", True))
    args.weight_ema_use_for_eval = bool(config.get("weight_ema_use_for_eval", True))
    args.weight_ema_update_interval = max(
        int(config.get("weight_ema_update_interval", 1)), 1
    )
    weight_ema_device = str(config.get("weight_ema_device", "accelerator")).lower()
    if weight_ema_device not in ("accelerator", "cpu"):
        raise ValueError("weight_ema_device must be 'accelerator' or 'cpu'")
    args.weight_ema_device = weight_ema_device
    weight_ema_eval_mode = str(config.get("weight_ema_eval_mode", "ema")).lower()
    if weight_ema_eval_mode not in ("off", "ema", "compare"):
        raise ValueError("weight_ema_eval_mode must be 'off', 'ema', or 'compare'")
    args.weight_ema_eval_mode = weight_ema_eval_mode
    args.weight_ema_save_separately = bool(
        config.get("weight_ema_save_separately", False)
    )

    # Loss-vs-timestep scatter logging
    args.log_loss_scatterplot = config.get("log_loss_scatterplot", False)
    args.log_loss_scatterplot_interval = config.get(
        "log_loss_scatterplot_interval", 500
    )

    # Advanced metrics (gradient stability, convergence, noise split, oscillation bounds)
    args.enable_advanced_metrics = config.get("enable_advanced_metrics", False)
    args.advanced_metrics_features = config.get("advanced_metrics_features", None)
    args.advanced_metrics_max_history = config.get(
        "advanced_metrics_max_history", 10000
    )
    args.gradient_watch_threshold = config.get("gradient_watch_threshold", 0.5)
    args.gradient_stability_window = config.get("gradient_stability_window", 10)
    args.convergence_window_sizes = config.get(
        "convergence_window_sizes", [10, 25, 50, 100]
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

    # Enhanced resume functionality
    args.initial_step = config.get("initial_step")
    args.initial_epoch = config.get("initial_epoch")
    args.skip_until_initial_step = config.get("skip_until_initial_step", False)

    # Latent quality analysis
    args.latent_quality_analysis = config.get("latent_quality_analysis", False)
    args.latent_mean_threshold = config.get("latent_mean_threshold", 0.16)
    args.latent_std_threshold = config.get("latent_std_threshold", 1.35)
    args.latent_quality_visualizer = config.get("latent_quality_visualizer", False)
    args.latent_quality_tensorboard = config.get("latent_quality_tensorboard", True)
    args.latent_quality_video_analysis = config.get(
        "latent_quality_video_analysis", True
    )

    args.save_every_n_epochs = config.get("save_every_n_epochs", None)
    args.save_every_n_steps = config.get("save_every_n_steps", 1000)
    args.save_checkpoint_before_sampling = bool(
        config.get("save_checkpoint_before_sampling", True)
    )
    args.save_last_n_epochs = config.get("save_last_n_epochs", None)
    args.save_last_n_epochs_state = config.get("save_last_n_epochs_state", None)
    args.save_last_n_steps = config.get("save_last_n_steps", None)
    args.save_last_n_steps_state = config.get("save_last_n_steps_state", None)
    args.save_state = config.get("save_state", True)
    args.save_state_on_train_end = config.get("save_state_on_train_end", False)

    # New parameters for independent state saving frequency
    args.save_state_every_n_epochs = config.get("save_state_every_n_epochs", None)
    args.save_state_every_n_steps = config.get("save_state_every_n_steps", None)

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

    # Two-tier validation: fast validation runs more frequently on a subset of data
    args.validate_fast_every_n_steps = config.get("validate_fast_every_n_steps", None)
    args.validation_fast_subset_fraction = float(
        config.get("validation_fast_subset_fraction", 0.1)
    )
    args.validation_fast_random_subset = bool(
        config.get("validation_fast_random_subset", True)
    )

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
    args.log_timestep_distribution_compare_baseline = config.get(
        "log_timestep_distribution_compare_baseline", True
    )
    args.log_timestep_distribution_compare_interval = int(
        config.get("log_timestep_distribution_compare_interval", 100)
    )
    args.log_timestep_distribution_compare_bins = int(
        config.get("log_timestep_distribution_compare_bins", 100)
    )

    # Throughput metrics logging settings
    args.log_throughput_metrics = config.get("log_throughput_metrics", True)
    args.throughput_window_size = config.get("throughput_window_size", 100)
    # Progress postfix alternation settings
    args.alternate_perf_postfix = config.get("alternate_perf_postfix", True)

    # Debug batch content logging
    args.log_batch_item_info = bool(config.get("log_batch_item_info", False))
    args.log_batch_item_info_interval = int(
        config.get("log_batch_item_info_interval", 1)
    )
    args.log_batch_item_info_max_items = int(
        config.get("log_batch_item_info_max_items", 8)
    )

    # VRAM estimation validation logging
    args.log_vram_validation = config.get("log_vram_validation", False)

    # TensorBoard server settings
    args.launch_tensorboard_server = config.get("launch_tensorboard_server", True)
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
    args.ddp_timeout = config.get("ddp_timeout", 1)
    args.ddp_gradient_as_bucket_view = config.get("ddp_gradient_as_bucket_view", False)
    args.ddp_static_graph = config.get("ddp_static_graph", False)

    # Dynamo settings
    args.dynamo_backend = config.get("dynamo_backend", "NO")
    args.dynamo_mode = config.get("dynamo_mode", "default")
    args.dynamo_fullgraph = config.get("dynamo_fullgraph", False)
    args.dynamo_dynamic = config.get("dynamo_dynamic", False)

    # Full precision settings
    args.full_fp16 = config.get("full_fp16", False)
    args.full_bf16 = config.get("full_bf16", False)

    # BF16 checkpoint conversion (finetune trainer only)
    args.use_or_convert_bf16 = config.get("use_or_convert_bf16", True)

    # WanFinetune specific settings
    args.fine_tune_ratio = config.get("fine_tune_ratio", 1.0)
    args.finetune_text_encoder = config.get("finetune_text_encoder", False)
    args.fused_backward_pass = config.get("fused_backward_pass", False)
    args.mem_eff_save = config.get("mem_eff_save", True)
    args.verify_weight_dynamics_every_n_steps = config.get(
        "verify_weight_dynamics_every_n_steps", 0
    )

    # Direct checkpoint loading settings
    args.direct_checkpoint_loading = config.get("direct_checkpoint_loading", True)

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
    # Dim-aware timestep shift (default off; helps large video latents when enabled)
    args.enable_dim_aware_time_shift = bool(
        config.get("enable_dim_aware_time_shift", False)
    )
    args.dim_aware_shift_base = float(config.get("dim_aware_shift_base", 4096.0))
    if args.dim_aware_shift_base <= 0:
        raise ValueError(
            f"dim_aware_shift_base must be > 0, got {args.dim_aware_shift_base}"
        )
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
    # Mode shift parameters
    args.time_shift_mu = config.get("time_shift_mu", 1.0)
    args.time_shift_sigma = config.get("time_shift_sigma", 1.0)
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

    # Enhanced FVDM settings
    args.fvdm_adaptive_ptss = config.get("fvdm_adaptive_ptss", False)
    args.fvdm_ptss_initial = config.get("fvdm_ptss_initial", 0.3)
    args.fvdm_ptss_final = config.get("fvdm_ptss_final", 0.1)
    args.fvdm_ptss_warmup = config.get("fvdm_ptss_warmup", 1000)
    args.fvdm_temporal_consistency_weight = config.get(
        "fvdm_temporal_consistency_weight", 0.0
    )
    args.fvdm_allow_async_with_temporal_loss = config.get(
        "fvdm_allow_async_with_temporal_loss", False
    )
    args.fvdm_pin_first_frame = config.get("fvdm_pin_first_frame", True)
    args.fvdm_frame_diversity_weight = config.get("fvdm_frame_diversity_weight", 0.0)
    args.fvdm_integrate_adaptive_timesteps = config.get(
        "fvdm_integrate_adaptive_timesteps", False
    )
    args.fvdm_eval_temporal_metrics = config.get("fvdm_eval_temporal_metrics", False)
    args.fvdm_eval_frequency = config.get("fvdm_eval_frequency", 1000)

    # FOPP settings
    args.fopp_num_timesteps = config.get("fopp_num_timesteps", 1000)
    args.fopp_schedule_type = config.get("fopp_schedule_type", "linear")
    args.fopp_beta_start = config.get("fopp_beta_start", 0.0001)
    args.fopp_beta_end = config.get("fopp_beta_end", 0.002)
    args.fopp_seed = config.get("fopp_seed", None)

    # Nabla settings
    args.nabla_sparse_attention = config.get("nabla_sparse_attention", False)
    args.nabla_sparse_algo = config.get("nabla_sparse_algo", "nabla-0.7_sta-11-24-24")

    # Contrastive Flow Matching (ΔFM) settings - Enhanced Implementation
    args.enable_contrastive_flow_matching = config.get(
        "enable_contrastive_flow_matching", False
    )
    # WanVideo LoRA cross-batch CFM regularizer (disabled by default)
    args.enable_wanvideo_cfm = bool(config.get("enable_wanvideo_cfm", False))
    args.wanvideo_cfm_weighting = str(config.get("wanvideo_cfm_weighting", "uniform"))
    if args.wanvideo_cfm_weighting not in {"uniform", "linear"}:
        raise ValueError(
            f"wanvideo_cfm_weighting must be 'uniform' or 'linear', got {args.wanvideo_cfm_weighting}"
        )
    args.wanvideo_cfm_lambda = float(config.get("wanvideo_cfm_lambda", 0.05))
    if args.wanvideo_cfm_lambda < 0:
        raise ValueError(
            f"wanvideo_cfm_lambda must be >= 0, got {args.wanvideo_cfm_lambda}"
        )
    args.contrastive_flow_lambda = config.get("contrastive_flow_lambda", 0.05)
    args.contrastive_flow_class_conditioning = config.get(
        "contrastive_flow_class_conditioning", True
    )
    args.contrastive_flow_skip_unconditional = config.get(
        "contrastive_flow_skip_unconditional", False
    )
    args.contrastive_flow_null_class_idx = config.get(
        "contrastive_flow_null_class_idx", None
    )

    # Timestep bucketing (dataset-driven, per-epoch stratified uniform pool)
    # None disables bucketing; set to >=2 to enable
    args.num_timestep_buckets = config.get("num_timestep_buckets")

    # Dispersive Loss Regularization settings
    args.enable_dispersive_loss = config.get("enable_dispersive_loss", False)
    args.dispersive_loss_lambda = config.get("dispersive_loss_lambda", 0.25)
    args.dispersive_loss_tau = config.get("dispersive_loss_tau", 1.0)
    # None disables extraction; use "last" (default) to select the final block
    args.dispersive_loss_target_block = config.get(
        "dispersive_loss_target_block", "last"
    )
    args.dispersive_loss_metric = config.get(
        "dispersive_loss_metric", "l2_sq"
    )  # "l2_sq", "l2_sq_legacy", or "cosine"
    # Optional: pool spatial tokens per frame before dispersion ("none" or "frame_mean")
    args.dispersive_loss_pooling = config.get("dispersive_loss_pooling", "none")

    # EqM mode
    args.enable_eqm_mode = config.get("enable_eqm_mode", False)
    eqm_prediction = str(config.get("eqm_prediction", "velocity"))
    allowed_eqm_predictions = {"velocity", "score", "noise"}
    if eqm_prediction.lower() not in allowed_eqm_predictions:
        raise ValueError(
            f"Unsupported eqm_prediction '{eqm_prediction}'. "
            f"Expected one of {sorted(allowed_eqm_predictions)}."
        )
    args.eqm_prediction = eqm_prediction
    args.eqm_path_type = config.get("eqm_path_type", "Linear")
    args.eqm_loss_weight = float(config.get("eqm_loss_weight", 1.0))
    args.eqm_transport_weighting = config.get("eqm_transport_weighting")
    args.eqm_train_eps = config.get("eqm_train_eps")
    args.eqm_sample_eps = config.get("eqm_sample_eps")
    args.eqm_step_size = config.get("eqm_step_size", 0.0017)
    args.eqm_momentum = config.get("eqm_momentum", 0.0)
    args.eqm_sampler = config.get("eqm_sampler", "gd")
    args.eqm_use_adaptive_sampler = bool(config.get("eqm_use_adaptive_sampler", False))
    args.eqm_adaptive_step_min = config.get("eqm_adaptive_step_min", 1e-5)
    args.eqm_adaptive_step_max = config.get("eqm_adaptive_step_max", 0.01)
    args.eqm_adaptive_growth = config.get("eqm_adaptive_growth", 1.05)
    args.eqm_adaptive_shrink = config.get("eqm_adaptive_shrink", 0.5)
    args.eqm_adaptive_restart_patience = int(
        config.get("eqm_adaptive_restart_patience", 4)
    )
    args.eqm_adaptive_alignment_threshold = config.get(
        "eqm_adaptive_alignment_threshold", 0.0
    )
    args.eqm_energy_head = bool(config.get("eqm_energy_head", False))
    args.eqm_energy_mode = config.get("eqm_energy_mode", "dot")
    args.eqm_weighting_schedule = config.get("eqm_weighting_schedule")
    raw_weight_steps = config.get("eqm_weighting_steps")
    args.eqm_weighting_steps = (
        int(raw_weight_steps) if raw_weight_steps is not None else None
    )
    args.eqm_ode_method = config.get("eqm_ode_method", "dopri5")
    args.eqm_ode_steps = int(config.get("eqm_ode_steps", 50))
    args.eqm_ode_atol = float(config.get("eqm_ode_atol", 1e-6))
    args.eqm_ode_rtol = float(config.get("eqm_ode_rtol", 1e-3))
    args.eqm_ode_reverse = bool(config.get("eqm_ode_reverse", False))
    args.eqm_ode_likelihood_atol = float(
        config.get("eqm_ode_likelihood_atol", args.eqm_ode_atol)
    )
    args.eqm_ode_likelihood_rtol = float(
        config.get("eqm_ode_likelihood_rtol", args.eqm_ode_rtol)
    )
    args.eqm_ode_likelihood_trace_samples = int(
        config.get("eqm_ode_likelihood_trace_samples", 1)
    )
    args.eqm_sde_method = config.get("eqm_sde_method", "Euler")
    args.eqm_sde_steps = int(config.get("eqm_sde_steps", 250))
    args.eqm_sde_last_step = config.get("eqm_sde_last_step", "Mean")
    args.eqm_sde_last_step_size = float(config.get("eqm_sde_last_step_size", 0.04))
    args.eqm_sde_diffusion_form = config.get("eqm_sde_diffusion_form", "SBDM")
    args.eqm_sde_diffusion_norm = float(config.get("eqm_sde_diffusion_norm", 1.0))
    args.eqm_save_npz = bool(config.get("eqm_save_npz", False))
    args.eqm_npz_dir = config.get("eqm_npz_dir")
    raw_npz_limit = config.get("eqm_npz_limit")
    args.eqm_npz_limit = int(raw_npz_limit) if raw_npz_limit is not None else None

    # Loss function settings
    args.loss_type = config.get("loss_type", "mse")

    # Pseudo-Huber loss parameters
    args.pseudo_huber_c = config.get("pseudo_huber_c", 0.5)
    args.pseudo_huber_schedule_type = config.get("pseudo_huber_schedule_type", "linear")
    args.pseudo_huber_c_min = config.get("pseudo_huber_c_min", 0.1)
    args.pseudo_huber_c_max = config.get("pseudo_huber_c_max", 1.0)

    # Huber loss parameters (for pure_huber loss type)
    args.huber_delta = config.get("huber_delta", 1.0)

    # Stepped loss parameters
    args.stepped_step_size = config.get("stepped_step_size", 50)
    args.stepped_multiplier = config.get("stepped_multiplier", 10.0)

    # Fourier loss parameters
    args.fourier_weight = config.get("fourier_weight", 0.05)
    args.fourier_mode = config.get(
        "fourier_mode", "weighted"
    )  # "basic", "weighted", "multiscale", "adaptive"
    args.fourier_norm = config.get("fourier_norm", "l2")  # "l1", "l2"
    args.fourier_dims = tuple(
        config.get("fourier_dims", [-2, -1])
    )  # Dimensions for FFT
    args.fourier_eps = config.get("fourier_eps", 1e-8)  # Numerical stability
    # Note: fourier_normalize removed as fourier functions don't support this parameter
    args.fourier_multiscale_factors = config.get(
        "fourier_multiscale_factors", [1, 2, 4]
    )
    args.fourier_adaptive_threshold = config.get("fourier_adaptive_threshold", 0.1)
    args.fourier_adaptive_alpha = config.get("fourier_adaptive_alpha", 0.5)
    args.fourier_high_freq_weight = config.get("fourier_high_freq_weight", 2.0)

    # DWT/Wavelet loss parameters
    args.wavelet_type = config.get("wavelet_type", "haar")  # "haar", "db1", "db4", etc.
    args.wavelet_levels = config.get(
        "wavelet_levels", 1
    )  # Number of decomposition levels
    args.wavelet_mode = config.get("wavelet_mode", "zero")  # Border mode for wavelets

    # Clustered MSE loss parameters
    args.clustered_mse_num_clusters = config.get("clustered_mse_num_clusters", 8)
    args.clustered_mse_cluster_weight = config.get("clustered_mse_cluster_weight", 1.0)

    # EW loss parameters
    args.ew_boundary_shift = config.get("ew_boundary_shift", 0.0)

    args.use_explicit_video_loss_reduction = config.get(
        "use_explicit_video_loss_reduction", False
    )
    args.enable_custom_loss_target = config.get("enable_custom_loss_target", False)

    # Optical Flow Loss (RAFT-based) settings
    args.enable_optical_flow_loss = config.get("enable_optical_flow_loss", False)
    args.lambda_optical_flow = config.get("lambda_optical_flow", 0.0)

    # HASTE (Holistic Alignment with Stage-wise Termination) settings
    from enhancements.haste.config_parser import parse_haste_config

    parse_haste_config(config, args)

    # CDC-FM (Carre du Champ Flow Matching) settings
    from enhancements.cdc.config_parser import parse_cdc_config

    parse_cdc_config(config, args, logger)

    # REPA (Representation Alignment) settings
    args.enable_repa = config.get("enable_repa", False)
    args.enable_irepa = bool(config.get("enable_irepa", False))
    args.irepa_projection_type = str(config.get("irepa_projection_type", "conv"))
    allowed_irepa_projection = {"mlp", "conv"}
    if args.irepa_projection_type not in allowed_irepa_projection:
        raise ValueError(
            f"Invalid irepa_projection_type '{args.irepa_projection_type}'. "
            f"Expected one of {sorted(allowed_irepa_projection)}."
        )
    args.irepa_proj_kernel = int(config.get("irepa_proj_kernel", 3))
    if args.irepa_proj_kernel < 3 or args.irepa_proj_kernel % 2 == 0:
        raise ValueError(
            f"irepa_proj_kernel must be an odd integer >= 3, got {args.irepa_proj_kernel}"
        )
    args.irepa_spatial_norm = str(config.get("irepa_spatial_norm", "zscore"))
    allowed_irepa_norms = {"none", "zscore"}
    if args.irepa_spatial_norm not in allowed_irepa_norms:
        raise ValueError(
            f"Invalid irepa_spatial_norm '{args.irepa_spatial_norm}'. "
            f"Expected one of {sorted(allowed_irepa_norms)}."
        )
    args.irepa_zscore_alpha = float(config.get("irepa_zscore_alpha", 1.0))
    if args.irepa_zscore_alpha <= 0:
        raise ValueError(
            f"irepa_zscore_alpha must be > 0 for numerical stability, got {args.irepa_zscore_alpha}"
        )
    if args.enable_irepa:
        logger.info(
            "iREPA enabled (projection=%s, spatial_norm=%s, kernel=%d, alpha=%.3f)",
            args.irepa_projection_type,
            args.irepa_spatial_norm,
            args.irepa_proj_kernel,
            args.irepa_zscore_alpha,
        )
    args.repa_encoder_name = config.get("repa_encoder_name", "dinov2-vit-b")
    args.repa_alignment_depth = config.get("repa_alignment_depth", 8)
    args.repa_loss_lambda = config.get("repa_loss_lambda", 0.5)
    args.repa_similarity_fn = config.get("repa_similarity_fn", "cosine")

    # REG (Representation Entanglement for Generation) settings
    args.enable_reg = bool(config.get("enable_reg", False))
    args.reg_encoder_name = config.get("reg_encoder_name", "dinov2-vit-b")
    args.reg_alignment_depth = int(config.get("reg_alignment_depth", 8))
    if args.reg_alignment_depth <= 0:
        raise ValueError(
            f"reg_alignment_depth must be > 0, got {args.reg_alignment_depth}"
        )
    args.reg_proj_coeff = float(config.get("reg_proj_coeff", 0.5))
    if args.reg_proj_coeff <= 0:
        raise ValueError(f"reg_proj_coeff must be > 0, got {args.reg_proj_coeff}")
    args.reg_cls_loss_weight = float(config.get("reg_cls_loss_weight", 0.03))
    if args.reg_cls_loss_weight < 0:
        raise ValueError(
            f"reg_cls_loss_weight must be >= 0, got {args.reg_cls_loss_weight}"
        )
    if args.enable_reg and args.reg_cls_loss_weight <= 0:
        raise ValueError("reg_cls_loss_weight must be > 0 when REG is enabled")
    args.reg_similarity_fn = str(config.get("reg_similarity_fn", "cosine"))
    allowed_reg_similarity = {"cosine", "mse"}
    if args.reg_similarity_fn not in allowed_reg_similarity:
        raise ValueError(
            f"Invalid reg_similarity_fn '{args.reg_similarity_fn}'. "
            f"Expected one of {sorted(allowed_reg_similarity)}."
        )
    args.reg_input_resolution = int(config.get("reg_input_resolution", 256))
    if args.reg_input_resolution not in (256, 512):
        raise ValueError(
            f"reg_input_resolution must be 256 or 512, got {args.reg_input_resolution}"
        )
    args.reg_cls_dim = int(config.get("reg_cls_dim", 0))
    if args.reg_cls_dim < 0:
        raise ValueError(f"reg_cls_dim must be >= 0, got {args.reg_cls_dim}")
    args.reg_target_type = str(config.get("reg_target_type", "flow"))
    allowed_reg_target_types = {"flow", "velocity"}
    if args.reg_target_type not in allowed_reg_target_types:
        raise ValueError(
            f"Invalid reg_target_type '{args.reg_target_type}'. "
            f"Expected one of {sorted(allowed_reg_target_types)}."
        )
    args.reg_spatial_align = bool(config.get("reg_spatial_align", True))
    if args.enable_reg:
        logger.info(
            "REG enabled (encoder=%s, depth=%d, lambda=%.3f, beta=%.3f, cls_dim=%d, target=%s)",
            args.reg_encoder_name,
            args.reg_alignment_depth,
            args.reg_proj_coeff,
            args.reg_cls_loss_weight,
            args.reg_cls_dim,
            args.reg_target_type,
        )

    # LayerSync (self-alignment) settings
    from enhancements.layer_sync.config import parse_layer_sync_config

    parse_layer_sync_config(config, args, logger)

    # Enhanced REPA settings (now the only REPA implementation)
    args.repa_input_resolution = config.get("repa_input_resolution", 256)
    args.repa_ensemble_mode = config.get("repa_ensemble_mode", "individual")
    args.repa_shared_projection = config.get("repa_shared_projection", False)
    args.repa_spatial_align = config.get("repa_spatial_align", True)

    # CREPA (Cross-frame Representation Alignment) settings
    args.crepa_enabled = bool(config.get("crepa_enabled", False))
    args.crepa_block_index = config.get("crepa_block_index", None)
    if args.crepa_block_index is not None:
        args.crepa_block_index = int(args.crepa_block_index)
    args.crepa_teacher_block_index = config.get("crepa_teacher_block_index", None)
    if args.crepa_teacher_block_index is not None:
        args.crepa_teacher_block_index = int(args.crepa_teacher_block_index)
    args.crepa_lambda = float(config.get("crepa_lambda", 0.5))

    adjacency_fallback = config.get("crepa_adjacency", None)
    tau_fallback = config.get("crepa_temperature", None)
    args.crepa_adjacent_distance = int(
        config.get(
            "crepa_adjacent_distance",
            1 if adjacency_fallback is None else adjacency_fallback,
        )
    )
    args.crepa_adjacent_tau = float(
        config.get("crepa_adjacent_tau", 1.0 if tau_fallback is None else tau_fallback)
    )
    args.crepa_cumulative_neighbors = bool(
        config.get("crepa_cumulative_neighbors", False)
    )
    args.crepa_encoder = config.get(
        "crepa_encoder", config.get("crepa_model", "dinov2_vitg14")
    )
    args.crepa_encoder_image_size = int(config.get("crepa_encoder_image_size", 518))
    raw_crepa_encoder_frame_chunk_size = config.get(
        "crepa_encoder_frame_chunk_size", -1
    )
    if raw_crepa_encoder_frame_chunk_size is None:
        raw_crepa_encoder_frame_chunk_size = -1
    args.crepa_encoder_frame_chunk_size = int(raw_crepa_encoder_frame_chunk_size)
    args.crepa_spatial_align = bool(config.get("crepa_spatial_align", True))
    args.crepa_use_backbone_features = bool(
        config.get("crepa_use_backbone_features", False)
    )
    args.crepa_drop_vae_encoder = bool(config.get("crepa_drop_vae_encoder", False))
    args.crepa_normalize_by_frames = config.get("crepa_normalize_by_frames", True)
    # Back-compat aliases for existing REPA-based CREPA path
    args.crepa_adjacency = args.crepa_adjacent_distance
    args.crepa_temperature = args.crepa_adjacent_tau

    if args.crepa_adjacent_distance < 0:
        raise ValueError(
            f"crepa_adjacent_distance must be non-negative, got {args.crepa_adjacent_distance}"
        )
    if args.crepa_adjacent_tau <= 0:
        raise ValueError(
            f"crepa_adjacent_tau must be > 0, got {args.crepa_adjacent_tau}"
        )
    if args.crepa_lambda < 0:
        raise ValueError(f"crepa_lambda must be non-negative, got {args.crepa_lambda}")
    if args.crepa_enabled and args.crepa_lambda <= 0:
        raise ValueError("crepa_lambda must be > 0 when CREPA is enabled")
    if args.crepa_encoder_image_size <= 0:
        raise ValueError(
            f"crepa_encoder_image_size must be > 0, got {args.crepa_encoder_image_size}"
        )
    if args.crepa_enabled and args.crepa_block_index is None:
        raise ValueError("crepa_block_index must be set when CREPA is enabled")
    if args.crepa_enabled:
        logger.info(
            "CREPA enabled (block=%s, teacher=%s, lambda=%.3f, distance=%d, tau=%.3f, backbone=%s)",
            args.crepa_block_index,
            args.crepa_teacher_block_index,
            args.crepa_lambda,
            args.crepa_adjacent_distance,
            args.crepa_adjacent_tau,
            args.crepa_use_backbone_features,
        )

    # SemanticGen LoRA settings
    parse_semanticgen_config(args, config, logger)

    # SARA (Structural and Adversarial Representation Alignment) settings
    args.sara_enabled = config.get("sara_enabled", False)
    args.sara_encoder_name = config.get("sara_encoder_name", "dinov2_vitb14")
    args.sara_alignment_depth = config.get("sara_alignment_depth", 8)

    # Loss weights
    args.sara_patch_loss_weight = float(config.get("sara_patch_loss_weight", 0.5))
    args.sara_autocorr_loss_weight = float(config.get("sara_autocorr_loss_weight", 0.5))
    args.sara_adversarial_loss_weight = float(
        config.get("sara_adversarial_loss_weight", 0.05)
    )

    # Structural alignment settings
    args.sara_autocorr_normalize = bool(config.get("sara_autocorr_normalize", True))
    args.sara_autocorr_use_frobenius = bool(
        config.get("sara_autocorr_use_frobenius", True)
    )

    # Adversarial discriminator settings
    args.sara_adversarial_enabled = bool(config.get("sara_adversarial_enabled", True))
    args.sara_discriminator_arch = config.get("sara_discriminator_arch", "resnet18")
    args.sara_discriminator_lr = float(config.get("sara_discriminator_lr", 2e-4))
    args.sara_discriminator_updates_per_step = int(
        config.get("sara_discriminator_updates_per_step", 1)
    )
    args.sara_discriminator_warmup_steps = int(
        config.get("sara_discriminator_warmup_steps", 500)
    )
    args.sara_discriminator_update_interval = int(
        config.get("sara_discriminator_update_interval", 5)
    )

    # Advanced training controls
    args.sara_similarity_fn = config.get("sara_similarity_fn", "cosine")
    args.sara_gradient_penalty_weight = float(
        config.get("sara_gradient_penalty_weight", 0.0)
    )
    args.sara_feature_matching = bool(config.get("sara_feature_matching", False))
    args.sara_feature_matching_weight = float(
        config.get("sara_feature_matching_weight", 0.1)
    )

    # Memory and stability settings
    args.sara_cache_encoder_outputs = bool(
        config.get("sara_cache_encoder_outputs", True)
    )
    args.sara_use_mixed_precision = bool(config.get("sara_use_mixed_precision", True))
    max_grad_norm = config.get("sara_discriminator_max_grad_norm", None)
    args.sara_discriminator_max_grad_norm = (
        None if max_grad_norm is None else float(max_grad_norm)
    )
    args.sara_discriminator_scheduler_step = int(
        config.get("sara_discriminator_scheduler_step", 0)
    )
    args.sara_discriminator_scheduler_gamma = float(
        config.get("sara_discriminator_scheduler_gamma", 0.1)
    )
    args.sara_log_detailed_metrics = bool(
        config.get("sara_log_detailed_metrics", False)
    )

    # Sprint configuration (parsed by Sprint module)
    try:
        from enhancements.sprint.config_parser import parse_sprint_config

        parse_sprint_config(config, args)
    except ImportError:
        # Sprint module not available, set defaults
        args.enable_sprint = False

    # MemFlow guidance configuration (parsed by MemFlow module)
    try:
        from enhancements.memflow_guidance.config_parser import (
            parse_memflow_guidance_config,
        )

        parse_memflow_guidance_config(config, args)
    except ImportError:
        args.enable_memflow_guidance = False

    # Masked Training Configuration
    args.use_masked_training_with_prior = config.get(
        "use_masked_training_with_prior", False
    )
    args.unmasked_probability = config.get("unmasked_probability", 0.1)
    args.unmasked_weight = config.get("unmasked_weight", 0.1)
    args.masked_prior_preservation_weight = config.get(
        "masked_prior_preservation_weight", 0.0
    )
    args.normalize_masked_area_loss = config.get("normalize_masked_area_loss", False)
    args.mask_interpolation_mode = config.get("mask_interpolation_mode", "area")
    args.enable_prior_computation = config.get("enable_prior_computation", True)
    args.prior_computation_method = config.get(
        "prior_computation_method", "lora_disabled"
    )

    # Video-specific parameters
    args.temporal_consistency_weight = config.get("temporal_consistency_weight", 0.0)
    args.frame_consistency_mode = config.get("frame_consistency_mode", "adjacent")

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

    # Enable temporal consistency enhancement for video training
    args.enable_frequency_domain_temporal_consistency = config.get(
        "enable_frequency_domain_temporal_consistency", False
    )
    args.freq_temporal_enable_motion_coherence = config.get(
        "freq_temporal_enable_motion_coherence", False
    )
    args.freq_temporal_enable_prediction_loss = config.get(
        "freq_temporal_enable_prediction_loss", False
    )
    args.freq_temporal_low_threshold = config.get("freq_temporal_low_threshold", 0.25)
    args.freq_temporal_high_threshold = config.get("freq_temporal_high_threshold", 0.7)
    args.freq_temporal_consistency_weight = config.get(
        "freq_temporal_consistency_weight", 0.1
    )
    args.freq_temporal_motion_weight = config.get("freq_temporal_motion_weight", 0.05)
    args.freq_temporal_prediction_weight = config.get(
        "freq_temporal_prediction_weight", 0.08
    )
    args.freq_temporal_max_distance = config.get("freq_temporal_max_distance", 4)
    args.freq_temporal_decay_factor = config.get("freq_temporal_decay_factor", 0.8)
    args.freq_temporal_min_frames = config.get("freq_temporal_min_frames", 4)
    args.freq_temporal_motion_threshold = config.get(
        "freq_temporal_motion_threshold", 0.1
    )
    args.freq_temporal_preserve_dc = config.get("freq_temporal_preserve_dc", True)
    args.freq_temporal_adaptive_threshold = config.get(
        "freq_temporal_adaptive_threshold", False
    )
    args.freq_temporal_adaptive_range = tuple(
        config.get("freq_temporal_adaptive_range", [0.15, 0.35])
    )
    args.freq_temporal_start_step = config.get("freq_temporal_start_step", 0)
    args.freq_temporal_end_step = config.get("freq_temporal_end_step", None)
    args.freq_temporal_warmup_steps = config.get("freq_temporal_warmup_steps", 100)
    args.freq_temporal_enable_caching = config.get("freq_temporal_enable_caching", True)
    args.freq_temporal_cache_size = config.get("freq_temporal_cache_size", 500)
    args.freq_temporal_batch_parallel = config.get("freq_temporal_batch_parallel", True)
    args.freq_temporal_apply_every_n_steps = config.get(
        "freq_temporal_apply_every_n_steps", 1
    )
    args.freq_temporal_max_frames_per_batch = config.get(
        "freq_temporal_max_frames_per_batch", 16
    )

    # Logging cadence for temporal consistency
    args.freq_temporal_log_every_steps = config.get(
        "freq_temporal_log_every_steps", 500
    )
    args.freq_temporal_tb_log_every_steps = config.get(
        "freq_temporal_tb_log_every_steps", 500
    )
    args.freq_temporal_weight_strategy = config.get(
        "freq_temporal_weight_strategy", "exponential"
    )
    args.freq_temporal_loss_reduction = config.get(
        "freq_temporal_loss_reduction", "mean"
    )
    args.freq_temporal_apply_to_latent = config.get(
        "freq_temporal_apply_to_latent", True
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

    # Parse memory optimization configuration (delegated to memory module)
    args = parse_memory_optimization_config(args, config)

    # Validate enhanced FP8 configuration
    if args.fp8_use_enhanced:
        # Validate quantization mode
        valid_modes = ["tensor", "channel", "block"]
        if args.fp8_quantization_mode not in valid_modes:
            logger.warning(
                f"⚠️  Invalid fp8_quantization_mode '{args.fp8_quantization_mode}'. "
                f"Valid options: {valid_modes}. Using 'tensor' as default."
            )
            args.fp8_quantization_mode = "tensor"

        # Validate FP8 format
        valid_formats = ["e4m3", "e5m2"]
        if args.fp8_format not in valid_formats:
            logger.warning(
                f"⚠️  Invalid fp8_format '{args.fp8_format}'. "
                f"Valid options: {valid_formats}. Using 'e4m3' as default."
            )
            args.fp8_format = "e4m3"

        # Validate block size for block quantization
        if args.fp8_quantization_mode == "block":
            if args.fp8_block_size is None:
                args.fp8_block_size = 128
                logger.info("📋 Using default block size 128 for block quantization")
            elif not isinstance(args.fp8_block_size, int) or args.fp8_block_size <= 0:
                logger.warning(
                    f"⚠️  Invalid fp8_block_size '{args.fp8_block_size}'. Using default 128."
                )
                args.fp8_block_size = 128

        # Validate percentile
        if args.fp8_percentile is not None:
            if not isinstance(args.fp8_percentile, (int, float)) or not (
                0.0 < args.fp8_percentile <= 1.0
            ):
                logger.warning(
                    f"⚠️  Invalid fp8_percentile '{args.fp8_percentile}'. "
                    "Must be between 0.0 and 1.0 or null. Using default 0.999."
                )
                args.fp8_percentile = 0.999

        # Convert exclude keys to list if it's a string
        if args.fp8_exclude_keys is not None:
            if isinstance(args.fp8_exclude_keys, str):
                args.fp8_exclude_keys = [
                    k.strip() for k in args.fp8_exclude_keys.split(",")
                ]
            elif not isinstance(args.fp8_exclude_keys, list):
                logger.warning(
                    f"⚠️  Invalid fp8_exclude_keys type '{type(args.fp8_exclude_keys)}'. "
                    "Expected list or comma-separated string. Ignoring."
                )
                args.fp8_exclude_keys = None

        logger.info(
            f"🔧 Enhanced FP8 enabled - Mode: {args.fp8_quantization_mode}, "
            f"Format: {args.fp8_format}, Block size: {args.fp8_block_size}, "
            f"Percentile: {args.fp8_percentile}, "
            f"Fallback to channel: {args.fp8_block_wise_fallback_to_channel}"
        )

    # Validate TorchAO FP8 configuration
    if args.torchao_fp8_enabled:
        # Check for conflicts with other FP8 methods
        if args.fp8_use_enhanced:
            logger.warning(
                "⚠️  Both torchao_fp8_enabled and fp8_use_enhanced are True. "
                "TorchAO will take precedence over enhanced FP8."
            )

        # Validate TorchAO weight dtype
        valid_torchao_dtypes = ["e4m3fn", "e5m2"]
        if args.torchao_fp8_weight_dtype not in valid_torchao_dtypes:
            logger.warning(
                f"⚠️  Invalid torchao_fp8_weight_dtype '{args.torchao_fp8_weight_dtype}'. "
                f"Valid options: {valid_torchao_dtypes}. Using 'e4m3fn' as default."
            )
            args.torchao_fp8_weight_dtype = "e4m3fn"

        # Convert target/exclude modules to lists if strings
        for attr_name in ["torchao_fp8_target_modules", "torchao_fp8_exclude_modules"]:
            attr_value = getattr(args, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, str):
                    setattr(args, attr_name, [k.strip() for k in attr_value.split(",")])
                elif not isinstance(attr_value, list):
                    logger.warning(
                        f"⚠️  Invalid {attr_name} type '{type(attr_value)}'. "
                        "Expected list or comma-separated string. Ignoring."
                    )
                    setattr(args, attr_name, None)

        logger.info(
            f"🚀 TorchAO FP8 enabled - Weight dtype: {args.torchao_fp8_weight_dtype}"
        )

    # Glance distillation mode configuration (WAN training)
    apply_glance_config(args, config, logger)

    # RCM distillation pipeline configuration (root-level keys, prefixed with rcm_)
    rcm_extra_args = config.get("rcm_extra_args", {}) or {}
    if not isinstance(rcm_extra_args, dict):
        logger.warning(
            "??  Invalid rcm_extra_args type '%s'. Expected table or inline table. Ignoring.",
            type(rcm_extra_args),
        )
        rcm_extra_args = {}
    args.rcm = argparse.Namespace(
        enabled=bool(config.get("rcm_enabled", False)),
        config_path=None,  # legacy field retained for compatibility
        override_wan=bool(config.get("rcm_override_wan", True)),
        accelerator_mode=config.get("rcm_accelerator_mode", "auto"),
        trainer_variant=config.get("rcm_trainer_variant", "distill"),
        max_steps=config.get("rcm_max_steps"),
        mixed_precision=config.get("rcm_mixed_precision", "bf16"),
        extra_args=rcm_extra_args,
        cpu_debug=bool(config.get("rcm_cpu_debug", False)),
    )

    if args.rcm.enabled and args.rcm.override_wan:
        args.pipeline_override = "rcm"
    else:
        args.pipeline_override = getattr(args, "pipeline_override", None)

    if args.rcm.enabled and getattr(args, "finetune_mode", False):
        raise ValueError("RCM pipeline cannot run with finetune_mode enabled.")

    # SRPO (Semantic Relative Preference Optimization) training configuration
    args.enable_srpo_training = bool(config.get("enable_srpo_training", False))

    if args.enable_srpo_training:
        # Reward model configuration
        args.srpo_reward_model_name = config.get("srpo_reward_model_name", "hps")
        args.srpo_reward_model_dtype = config.get("srpo_reward_model_dtype", "float32")
        args.srpo_srp_control_weight = float(config.get("srpo_srp_control_weight", 1.0))
        args.srpo_srp_positive_words = config.get("srpo_srp_positive_words", None)
        args.srpo_srp_negative_words = config.get("srpo_srp_negative_words", None)

        # Direct-Align algorithm parameters
        args.srpo_sigma_interpolation_method = config.get(
            "srpo_sigma_interpolation_method", "linear"
        )
        args.srpo_sigma_interpolation_min = float(
            config.get("srpo_sigma_interpolation_min", 0.0)
        )
        args.srpo_sigma_interpolation_max = float(
            config.get("srpo_sigma_interpolation_max", 1.0)
        )
        args.srpo_num_inference_steps = int(config.get("srpo_num_inference_steps", 50))
        args.srpo_guidance_scale = float(config.get("srpo_guidance_scale", 1.0))
        args.srpo_enable_sd3_time_shift = bool(
            config.get("srpo_enable_sd3_time_shift", True)
        )
        args.srpo_sd3_time_shift_value = float(
            config.get("srpo_sd3_time_shift_value", 3.0)
        )

        # Discount schedules
        args.srpo_discount_denoise_min = float(
            config.get("srpo_discount_denoise_min", 0.0)
        )
        args.srpo_discount_denoise_max = float(
            config.get("srpo_discount_denoise_max", 1.0)
        )
        args.srpo_discount_inversion_start = float(
            config.get("srpo_discount_inversion_start", 1.0)
        )
        args.srpo_discount_inversion_end = float(
            config.get("srpo_discount_inversion_end", 0.0)
        )

        # Training hyperparameters
        args.srpo_batch_size = int(config.get("srpo_batch_size", 1))
        args.srpo_gradient_accumulation_steps = int(
            config.get("srpo_gradient_accumulation_steps", 4)
        )
        args.srpo_num_training_steps = int(config.get("srpo_num_training_steps", 500))

        # Validation parameters
        args.srpo_validation_prompts = config.get("srpo_validation_prompts", None)
        args.srpo_validation_frequency = int(
            config.get("srpo_validation_frequency", 50)
        )
        args.srpo_save_validation_videos = bool(
            config.get("srpo_save_validation_videos", True)
        )
        args.srpo_save_validation_as_images = bool(
            config.get("srpo_save_validation_as_images", False)
        )

        # WAN-specific parameters
        args.srpo_vae_scale_factor = int(config.get("srpo_vae_scale_factor", 8))
        args.srpo_latent_channels = int(config.get("srpo_latent_channels", 16))

        # Multi-frame reward parameters
        args.srpo_reward_frame_strategy = config.get(
            "srpo_reward_frame_strategy", "first"
        )
        args.srpo_reward_num_frames = int(config.get("srpo_reward_num_frames", 1))
        args.srpo_reward_aggregation = config.get("srpo_reward_aggregation", "mean")

        # Video-specific reward parameters
        args.srpo_enable_video_rewards = bool(
            config.get("srpo_enable_video_rewards", False)
        )
        args.srpo_temporal_consistency_weight = float(
            config.get("srpo_temporal_consistency_weight", 0.0)
        )
        args.srpo_optical_flow_weight = float(
            config.get("srpo_optical_flow_weight", 0.0)
        )
        args.srpo_motion_quality_weight = float(
            config.get("srpo_motion_quality_weight", 0.0)
        )

        logger.info(
            f"🎯 SRPO training enabled - Reward model: {args.srpo_reward_model_name}, "
            f"Steps: {args.srpo_num_training_steps}, Batch size: {args.srpo_batch_size}"
        )

    return args
