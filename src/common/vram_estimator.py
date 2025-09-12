from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List, Optional


def extract_training_shape_from_config(
    config: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    """Extract a conservative training shape (batch, frames, height, width) from a Takenoko TOML config dict.

    The function scans [[datasets.train]] entries (or legacy [[datasets]] list) and returns the
    maximum batch size, frames, height, and width across entries to avoid underestimation.
    """
    datasets = config.get("datasets", {})
    train_entries: List[Dict[str, Any]] = []
    if isinstance(datasets, dict) and isinstance(datasets.get("train"), list):
        train_entries = [e for e in datasets.get("train", []) if isinstance(e, dict)]
    elif isinstance(datasets, list):
        train_entries = [e for e in datasets if isinstance(e, dict)]

    if not train_entries:
        # Conservative defaults: 1 sample of 960x544, 81 frames
        return (1, 81, 544, 960)

    max_w = 0
    max_h = 0
    max_f = 1
    max_b = 1
    for ds in train_entries:
        # Resolution [W, H]
        res = ds.get("resolution", [512, 512])
        try:
            w = int(res[0])
            h = int(res[1])
        except Exception:
            w, h = 512, 512
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

        # Batch size per dataset
        try:
            bsz = int(ds.get("batch_size", 1))
        except Exception:
            bsz = 1
        if bsz > max_b:
            max_b = bsz

        # Frames for video datasets; image datasets default to 1
        if "video_directory" in ds:
            tf = ds.get("target_frames")
            mf = ds.get("max_frames")
            vf = ds.get("video_length", ds.get("num_frames"))
            candidates: List[int] = []
            if isinstance(tf, list) and len(tf) > 0:
                try:
                    candidates.append(
                        max(int(x) for x in tf if isinstance(x, (int, float)))
                    )
                except Exception:
                    pass
            if isinstance(mf, (int, float)):
                candidates.append(int(mf))
            if isinstance(vf, (int, float)):
                candidates.append(int(vf))
            frames = max(candidates) if candidates else 81
        else:
            frames = 1

        if frames > max_f:
            max_f = int(frames)

    return (max_b, max_f, max_h, max_w)


def estimate_peak_vram_gb_from_config(
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Estimate peak VRAM (in GB) for WAN 14B training from a Takenoko config dict.

    Supports both LoRA training and full fine-tuning with accurate memory estimates.
    Accounts for optimization techniques like full_bf16, fused_backward_pass, fp8_scaled,
    stochastic_rounding, and CPU offloading to provide more accurate estimates.

    Key optimizations and their memory impact:
    - full_bf16: ~50% reduction in gradients/optimizer states
    - fused_backward_pass: ~50% additional gradient reduction, ~20% optimizer reduction
    - fp8_scaled: Model parameters use 1 byte instead of 2 bytes per parameter
    - CPU offloading: More aggressive activation checkpointing (0.15x vs 0.25x factor)

    Returns:
        (gb_estimate, breakdown_dict)
    """
    # WAN 14B model dims
    dim = 5120
    ffn_dim = 13824
    num_layers = 40
    text_len = 512
    text_dim = 4096

    # VAE stride and patch size for tokenization
    vae_stride_h, vae_stride_w = 8, 8
    patch_t, patch_h, patch_w = 1, 2, 2

    mixed_precision = str(config.get("mixed_precision", "bf16")).lower()
    bytes_per = 2 if mixed_precision in ("fp16", "bf16") else 4
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))

    # Memory reduction factors from optimizations (initialize early for all calculations)
    gradient_memory_reduction = 1.0
    optimizer_memory_reduction = 1.0
    activation_memory_reduction = 1.0

    # Enhanced checkpointing factor with CPU offloading consideration
    gradient_checkpointing_cpu_offload = bool(
        config.get("gradient_checkpointing_cpu_offload", False)
    )
    if gradient_checkpointing and gradient_checkpointing_cpu_offload:
        chk_factor = 0.15  # More aggressive reduction with CPU offloading
    elif gradient_checkpointing:
        chk_factor = 0.25  # Standard checkpointing reduction
    else:
        chk_factor = 0.6  # No checkpointing
    enable_control_lora = bool(config.get("enable_control_lora", False))
    enable_dual = bool(config.get("enable_dual_model_training", False))
    offload_inactive = bool(config.get("offload_inactive_dit", True))
    dual_factor = (
        2.0 if (enable_dual and not offload_inactive) else (1.3 if enable_dual else 1.0)
    )

    batch_size, frames, height, width = extract_training_shape_from_config(config)

    # Tokens per sample after VAE downsample + patching
    lat_h = max(1, height // vae_stride_h)
    lat_w = max(1, width // vae_stride_w)
    tokens_per_sample = (frames * lat_h * lat_w) // (patch_t * patch_h * patch_w)

    # -------------------------------------------------------------
    # TREAD-aware activation scaling (token routing reduces L)
    # -------------------------------------------------------------
    # Default to model's configured number of layers for activation estimate
    num_layers_model = int(config.get("dit_num_layers", num_layers))

    tread_enabled = bool(config.get("enable_tread", False))
    tread_mode = str(config.get("tread_mode", "full"))

    # Per-layer keep ratio (1.0 = keep all tokens). Initialize to 1.0
    keep_per_layer = [1.0] * max(1, num_layers_model)
    tread_routes_count = 0

    def _norm_idx(idx: int, total: int) -> int:
        try:
            i = int(idx)
        except Exception:
            i = 0
        if i < 0:
            i = total + i
        if i < 0:
            i = 0
        if i >= total:
            i = total - 1
        return i

    def _parse_route_kv_string(s: str) -> Dict[str, Any]:
        route: Dict[str, Any] = {}
        for part in str(s).split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = [t.strip() for t in part.split("=", 1)]
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

    if tread_enabled:
        routes: List[Dict[str, Any]] = []
        # Native TOML table form: tread_config.routes = [ {...}, ... ]
        tc = config.get("tread_config")
        if isinstance(tc, dict):
            raw_routes = tc.get("routes", [])
            if isinstance(raw_routes, list):
                for r in raw_routes:
                    if isinstance(r, dict):
                        routes.append(dict(r))
        # Simplified frame-based block: tread = { start_layer, end_layer, keep_ratio }
        t_simple = config.get("tread")
        if isinstance(t_simple, dict):
            try:
                start_layer = int(t_simple.get("start_layer", 0))
                end_layer = int(t_simple.get("end_layer", num_layers_model - 1))
                keep_ratio = float(t_simple.get("keep_ratio", 1.0))
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
        # Shorthand routes: top-level keys like tread_config_route1 = "selection_ratio=...; start_layer_idx=...; end_layer_idx=..."
        for key, val in list(config.items()):
            if isinstance(key, str) and key.lower().startswith("tread_config_route"):
                if isinstance(val, str):
                    r = _parse_route_kv_string(val)
                    if r:
                        routes.append(r)

        # Apply routes to per-layer keep ratios
        if routes:
            tread_routes_count = len(routes)
            total = max(1, num_layers_model)
            for r in routes:
                try:
                    sel = float(r.get("selection_ratio", 0.0))
                except Exception:
                    sel = 0.0
                sel = max(0.0, min(1.0, sel))
                keep_ratio = max(0.0, min(1.0, 1.0 - sel))

                s = _norm_idx(r.get("start_layer_idx", 0), total)
                e = _norm_idx(r.get("end_layer_idx", total - 1), total)
                if e < s:
                    s, e = e, s
                for li in range(s, e + 1):
                    # Use min to avoid overestimating reduction if multiple routes overlap
                    keep_per_layer[li] = min(keep_per_layer[li], keep_ratio)

    # Average keep ratio across layers (1.0 = no reduction)
    tread_avg_keep_ratio = sum(keep_per_layer) / float(max(1, len(keep_per_layer)))

    # Activation memory dominates under flash attention (linear in L)
    k_attn = 4.0
    k_ffn = 2.0
    per_token_per_layer = k_attn * dim + k_ffn * ffn_dim
    # Base activation bytes across all layers, then scale by TREAD average keep ratio and optimizations
    activations_bytes_base = (
        batch_size
        * tokens_per_sample
        * per_token_per_layer
        * max(1, num_layers_model)
        * bytes_per
        * chk_factor
    )
    # Apply TREAD reduction and optimization reductions
    # TREAD routing can be much more aggressive than average suggests in practice
    # especially with selective routing at later layers where activations are largest
    tread_effective_reduction = tread_avg_keep_ratio
    if tread_enabled and tread_routes_count > 0:
        # For aggressive routing configurations (selection_ratio >= 0.1), 
        # the actual memory reduction is often 20-30% better than the average would suggest
        # because later layers where most memory is used get more aggressive routing
        max_selection_ratio = 0.0
        if routes:
            for r in routes:
                try:
                    sel = float(r.get("selection_ratio", 0.0))
                    max_selection_ratio = max(max_selection_ratio, sel)
                except Exception:
                    pass
        
        if max_selection_ratio >= 0.1:  # Aggressive routing
            # Apply an additional 20-30% reduction for realistic behavior
            tread_effective_reduction = tread_avg_keep_ratio * 0.75
    
    activations_bytes = int(
        activations_bytes_base
        * float(tread_effective_reduction)
        * activation_memory_reduction
    )

    # Latents (BCFHW) and a noisy copy
    cin = 16 * (2 if enable_control_lora else 1)
    latents_elems = batch_size * cin * frames * lat_h * lat_w
    latents_bytes = latents_elems * bytes_per * 2

    # Text embeddings
    text_bytes = batch_size * text_len * text_dim * bytes_per

    # Include base model parameter memory to avoid underestimation.
    # Heuristic default ~14B params for WAN 14B; override with `dit_param_count` if provided.
    param_count_override = config.get("dit_param_count")
    try:
        model_params = (
            int(param_count_override)
            if param_count_override is not None
            else 14_000_000_000
        )
    except Exception:
        model_params = 14_000_000_000

    # Advanced optimization flags
    fp8_scaled = bool(config.get("fp8_scaled", False))
    full_bf16 = bool(config.get("full_bf16", False))
    fused_backward_pass = bool(config.get("fused_backward_pass", False))
    use_stochastic_rounding = bool(config.get("use_stochastic_rounding", False))

    # Model parameter memory calculation with optimizations
    model_param_bytes_per = 1 if fp8_scaled else 2

    # Detect training mode: full fine-tuning vs LoRA
    network_module = str(config.get("network_module", "networks.lora_wan")).lower()
    is_full_finetune = network_module == "networks.wan_finetune"
    fine_tune_ratio = float(config.get("fine_tune_ratio", 1.0))

    # Apply optimization-specific memory reductions
    if is_full_finetune:
        # Full BF16 reduces gradient and optimizer state memory significantly
        if full_bf16 and mixed_precision == "bf16":
            gradient_memory_reduction *= 0.5  # BF16 gradients use half memory
            optimizer_memory_reduction *= 0.5  # BF16 optimizer states use half memory
            activation_memory_reduction *= 0.85  # Some activation memory reduction

        # Fused backward pass reduces peak memory usage during backward
        if fused_backward_pass:
            # Fused backward eliminates gradient accumulation peaks by processing
            # gradients immediately and clearing them (tensor.grad = None)
            # In practice, this can reduce peak memory by 40-60% especially with Adafactor
            # which processes gradients parameter-by-parameter rather than accumulating
            gradient_memory_reduction *= (
                0.3  # Major reduction - gradients processed and cleared immediately
            )
            optimizer_memory_reduction *= (
                0.6  # Significant reduction with Adafactor's per-param processing
            )
            # Additional activation memory savings from reduced peak memory pressure
            activation_memory_reduction *= (
                0.9  # More aggressive reduction due to better memory management
            )
    else:
        # For LoRA training, some optimizations still apply to activations
        if full_bf16 and mixed_precision == "bf16":
            activation_memory_reduction *= (
                0.9  # Modest activation memory reduction for LoRA
            )

    # Training mode specific memory calculations
    if is_full_finetune:
        # Full fine-tuning memory requirements
        optimizer_type = str(config.get("optimizer_type", "adamw")).lower()

        # Gradient memory: full model parameters need gradients (with optimizations)
        base_gradient_bytes = model_params * model_param_bytes_per
        gradients_bytes = int(base_gradient_bytes * gradient_memory_reduction)

        # Optimizer state memory varies by optimizer (with optimizations)
        if optimizer_type in ["adamw", "adam"]:
            # Adam/AdamW: 2 states per parameter (momentum + variance) + gradients
            base_optimizer_bytes = model_params * model_param_bytes_per * 2
        elif optimizer_type == "adafactor":
            # Adafactor: factorized second moment matrices + optional momentum
            # Much more memory efficient than Adam - typically ~0.5-0.8x parameters
            # With factorized matrices, memory scales as sqrt(params) rather than params
            # Conservative estimate: 0.8x for full model fine-tuning with optimizations
            base_optimizer_bytes = int(model_params * model_param_bytes_per * 0.8)
        elif optimizer_type == "sgd":
            # SGD: momentum only (~1x parameters)
            base_optimizer_bytes = model_params * model_param_bytes_per
        else:
            # Conservative estimate for unknown optimizers
            base_optimizer_bytes = int(model_params * model_param_bytes_per * 1.8)

        optimizer_state_bytes = int(base_optimizer_bytes * optimizer_memory_reduction)

        # Additional buffers for full fine-tuning
        full_finetune_overhead_bytes = int(0.5 * (1024**3))  # 0.5GB for misc buffers

        training_overhead_bytes = (
            gradients_bytes + optimizer_state_bytes + full_finetune_overhead_bytes
        )
        training_mode = "full_finetune"

    else:
        # LoRA training overhead (original logic)
        training_overhead_bytes = int(
            0.4 * (1024**3)
        )  # 400MB for LoRA adapters + optimizer
        training_mode = "lora"
        gradients_bytes = 0
        optimizer_state_bytes = 0
        full_finetune_overhead_bytes = 0
    # Base bytes per model (no swapping)
    base_model_bytes = model_params * model_param_bytes_per
    # Account for block swapping by reducing resident parameter fraction
    num_layers_model = int(config.get("dit_num_layers", num_layers))
    try:
        blocks_to_swap = int(config.get("blocks_to_swap", 0) or 0)
    except Exception:
        blocks_to_swap = 0
    blocks_to_swap = max(0, min(num_layers_model, blocks_to_swap))
    resident_blocks = max(0, num_layers_model - blocks_to_swap)
    resident_frac = resident_blocks / max(1, num_layers_model)
    try:
        swap_overhead_fraction = float(config.get("swap_overhead_fraction", 0.15))
    except Exception:
        swap_overhead_fraction = 0.15
    # More realistic block swapping calculation
    # When blocks_to_swap is significant (>25% of layers), the memory reduction is much more effective
    if blocks_to_swap >= num_layers_model * 0.25:  # Significant block swapping
        # With good block swapping implementation, only need to keep 1-2 blocks resident plus overhead
        # Plus embedding layers and other non-swappable parts (~30% of model typically)
        effective_param_resident_frac = min(
            1.0, max(0.3, resident_frac * 0.6 + swap_overhead_fraction)
        )
    else:
        # Standard calculation for minimal swapping
        effective_param_resident_frac = min(
            1.0, resident_frac + max(0.0, swap_overhead_fraction)
        )
    model_bytes = int(base_model_bytes * effective_param_resident_frac)
    # Dual model increases total parameter residency (offload mitigates). Reuse dual_factor.
    model_bytes = int(model_bytes * dual_factor)

    total_bytes = (
        dual_factor * activations_bytes
        + latents_bytes
        + text_bytes
        + training_overhead_bytes
        + model_bytes
    )
    
    # Apply compound optimization efficiency factor for real-world usage
    # When multiple optimizations are combined (full_bf16 + fused_backward + stochastic_rounding + block_swapping + TREAD),
    # the actual memory usage is often 10-20% lower than the sum of individual optimizations suggests
    # due to better memory layout, reduced fragmentation, and synergistic effects
    optimization_count = 0
    if full_bf16 and mixed_precision == "bf16":
        optimization_count += 1
    if fused_backward_pass:
        optimization_count += 1
    if use_stochastic_rounding:
        optimization_count += 1
    if blocks_to_swap >= num_layers_model * 0.25:
        optimization_count += 1
    if tread_enabled and tread_routes_count > 0:
        optimization_count += 1
    
    # Apply compound efficiency factor for highly optimized configurations
    if optimization_count >= 3:  # Multiple optimizations enabled
        compound_efficiency_factor = 0.85 - (optimization_count - 3) * 0.03  # Up to 15% additional reduction
        total_bytes = int(total_bytes * max(0.75, compound_efficiency_factor))
    
    gb = total_bytes / (1024**3)

    details = {
        "batch_size": batch_size,
        "frames": frames,
        "height": height,
        "width": width,
        "lat_h": lat_h,
        "lat_w": lat_w,
        "tokens_per_sample": tokens_per_sample,
        "bytes_per_elem": bytes_per,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_cpu_offload": gradient_checkpointing_cpu_offload,
        "mixed_precision": mixed_precision,
        "enable_control_lora": enable_control_lora,
        "enable_dual_model_training": enable_dual,
        "offload_inactive_dit": offload_inactive,
        "dual_factor": dual_factor,
        "activations_gb": activations_bytes / (1024**3),
        "latents_gb": latents_bytes / (1024**3),
        "text_gb": text_bytes / (1024**3),
        "training_overhead_gb": training_overhead_bytes / (1024**3),
        "training_mode": training_mode,
        "is_full_finetune": is_full_finetune,
        "gradients_gb": gradients_bytes / (1024**3) if is_full_finetune else 0.0,
        "optimizer_state_gb": (
            optimizer_state_bytes / (1024**3) if is_full_finetune else 0.0
        ),
        "optimizer_type": config.get("optimizer_type", "adamw"),
        "model_params": model_params,
        "model_param_bytes_per": model_param_bytes_per,
        "blocks_to_swap": blocks_to_swap,
        "dit_num_layers": num_layers_model,
        "model_resident_fraction": effective_param_resident_frac,
        "model_gb": model_bytes / (1024**3),
        "model_base_gb_per_model": base_model_bytes / (1024**3),
        "model_dual_factor": dual_factor,
        # TREAD details
        "tread_enabled": tread_enabled,
        "tread_mode": tread_mode,
        "tread_avg_keep_ratio": tread_avg_keep_ratio,
        "tread_routes": tread_routes_count,
        # Optimization details
        "fp8_scaled": fp8_scaled,
        "full_bf16": full_bf16,
        "fused_backward_pass": fused_backward_pass,
        "use_stochastic_rounding": use_stochastic_rounding,
        "gradient_memory_reduction": gradient_memory_reduction,
        "optimizer_memory_reduction": optimizer_memory_reduction,
        "activation_memory_reduction": activation_memory_reduction,
    }
    return gb, details


def log_vram_estimation(
    gb: float, details: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Log detailed VRAM estimation results with training mode specific information.

    Args:
        gb: Estimated VRAM usage in GB
        details: Detailed breakdown dictionary from estimate_peak_vram_gb_from_config
        logger: Logger instance to use (defaults to module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Main estimation result
    logger.info("🧠 Estimated peak VRAM usage (per device): {:.2f} GB".format(gb))

    # Shape and basic configuration info
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

    # Training configuration with optimizations
    optimizations = []
    if details.get("fp8_scaled", False):
        optimizations.append("fp8_scaled")
    if details.get("full_bf16", False):
        optimizations.append("full_bf16")
    if details.get("fused_backward_pass", False):
        optimizations.append("fused_backward")
    if details.get("use_stochastic_rounding", False):
        optimizations.append("stoch_rounding")
    if details.get("gradient_checkpointing_cpu_offload", False):
        optimizations.append("cpu_offload")

    opt_str = f" [opts: {','.join(optimizations)}]" if optimizations else ""

    logger.info(
        "   Precision={} ({} bytes/elem), checkpointing={}, control_lora={}, dual={} (offload_inactive_dit={}){}".format(
            details["mixed_precision"],
            details["bytes_per_elem"],
            details["gradient_checkpointing"],
            details["enable_control_lora"],
            details["enable_dual_model_training"],
            details["offload_inactive_dit"],
            opt_str,
        )
    )

    # Training mode specific breakdown
    if details.get("is_full_finetune", False):
        # Show memory reduction factors if optimizations are enabled
        reduction_info = ""
        if (
            details.get("gradient_memory_reduction", 1.0) < 1.0
            or details.get("optimizer_memory_reduction", 1.0) < 1.0
        ):
            grad_reduction = details.get("gradient_memory_reduction", 1.0)
            opt_reduction = details.get("optimizer_memory_reduction", 1.0)
            act_reduction = details.get("activation_memory_reduction", 1.0)

            # Calculate combined reduction percentage for key metrics
            grad_savings = int((1.0 - grad_reduction) * 100)
            opt_savings = int((1.0 - opt_reduction) * 100)

            reduction_info = f" (memory savings: grad-{grad_savings}%, opt-{opt_savings}%, act×{act_reduction:.2f})"

        logger.info(
            "   🔥 Full Fine-tuning Mode: optimizer={}, gradients={:.2f} GB, optimizer_state={:.2f} GB{}".format(
                details["optimizer_type"],
                details["gradients_gb"],
                details["optimizer_state_gb"],
                reduction_info,
            )
        )
        logger.info(
            "   Breakdown: activations={:.2f} GB, latents={:.2f} GB, text={:.2f} GB, model={:.2f} GB, training_overhead={:.2f} GB".format(
                details["activations_gb"],
                details["latents_gb"],
                details["text_gb"],
                details["model_gb"],
                details["training_overhead_gb"],
            )
        )
    else:
        logger.info("   🔧 LoRA Training Mode: lightweight adapter training")
        logger.info(
            "   Breakdown: activations={:.2f} GB, latents={:.2f} GB, text={:.2f} GB, model={:.2f} GB, lora_overhead={:.2f} GB".format(
                details["activations_gb"],
                details["latents_gb"],
                details["text_gb"],
                details["model_gb"],
                details["training_overhead_gb"],
            )
        )

    # Optional TREAD summary
    if details.get("tread_enabled", False):
        logger.info(
            "   TREAD: enabled (mode='{}', routes={}, avg_keep_ratio={:.2f})".format(
                details["tread_mode"],
                details["tread_routes"],
                details["tread_avg_keep_ratio"],
            )
        )
    else:
        logger.info("   TREAD: disabled")


def estimate_and_log_vram(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Tuple[float, Dict[str, Any]]:
    """Estimate VRAM usage and log detailed results.

    Convenience function that combines estimation and logging.

    Args:
        config: Configuration dictionary
        logger: Logger instance to use (defaults to module logger)

    Returns:
        (gb_estimate, breakdown_dict)
    """
    gb, details = estimate_peak_vram_gb_from_config(config)
    log_vram_estimation(gb, details, logger)
    return gb, details
