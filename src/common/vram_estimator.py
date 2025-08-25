from __future__ import annotations

from typing import Dict, Any, Tuple, List


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
    chk_factor = 0.25 if gradient_checkpointing else 0.6
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
    # Base activation bytes across all layers, then scale by TREAD average keep ratio
    activations_bytes_base = (
        batch_size
        * tokens_per_sample
        * per_token_per_layer
        * max(1, num_layers_model)
        * bytes_per
        * chk_factor
    )
    activations_bytes = int(activations_bytes_base * float(tread_avg_keep_ratio))

    # Latents (BCFHW) and a noisy copy
    cin = 16 * (2 if enable_control_lora else 1)
    latents_elems = batch_size * cin * frames * lat_h * lat_w
    latents_bytes = latents_elems * bytes_per * 2

    # Text embeddings
    text_bytes = batch_size * text_len * text_dim * bytes_per

    # LoRA/optimizer/buffers heuristic overhead
    lora_overhead_bytes = int(0.4 * (1024**3))

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

    fp8_scaled = bool(config.get("fp8_scaled", False))
    model_param_bytes_per = 1 if fp8_scaled else 2
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
        + lora_overhead_bytes
        + model_bytes
    )
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
        "mixed_precision": mixed_precision,
        "enable_control_lora": enable_control_lora,
        "enable_dual_model_training": enable_dual,
        "offload_inactive_dit": offload_inactive,
        "dual_factor": dual_factor,
        "activations_gb": activations_bytes / (1024**3),
        "latents_gb": latents_bytes / (1024**3),
        "text_gb": text_bytes / (1024**3),
        "overhead_gb": lora_overhead_bytes / (1024**3),
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
    }
    return gb, details
