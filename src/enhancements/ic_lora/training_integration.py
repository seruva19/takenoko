from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from common.logger import get_logger


logger = get_logger(__name__)
_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)-(\d+)$")
_PANEL_TAG_RE = re.compile(r"\[image\d+\]", flags=re.IGNORECASE)
_WARNED_MISSING_SYNC_SCORE_FLAG = "_ic_lora_warned_missing_sync_scores"
_SYNC_SCORE_CACHE: dict[tuple[str, str], Optional[float]] = {}


def is_ic_lora_enabled(args: Any) -> bool:
    enabled = getattr(args, "enable_ic_lora", None)
    if enabled is not None:
        return bool(enabled)
    return str(getattr(args, "network_module", "") or "") == "networks.ic_lora_wan"


def _normalize_reference_latents(
    reference: torch.Tensor,
    target_latents: torch.Tensor,
    dtype: torch.dtype,
    args: Any,
) -> Optional[torch.Tensor]:
    """Normalize reference tensor to [B, C, F, H, W] latent layout."""
    ref = reference
    target_c = int(target_latents.shape[1])
    target_h = int(target_latents.shape[3])
    target_w = int(target_latents.shape[4])

    if ref.dim() != 5:
        return None

    # Common variants:
    # - B,C,F,H,W (preferred)
    # - B,F,H,W,C (raw numpy frame stacks)
    # - B,F,C,H,W
    if ref.shape[1] == target_c:
        pass
    elif ref.shape[-1] == target_c:
        ref = ref.permute(0, 4, 1, 2, 3).contiguous()
    elif ref.shape[2] == target_c:
        ref = ref.permute(0, 2, 1, 3, 4).contiguous()
    else:
        return None

    ref_h = int(ref.shape[-2])
    ref_w = int(ref.shape[-1])
    expected_scale = int(getattr(args, "ic_lora_reference_downscale_factor", 1) or 1)
    inferred_scale: Optional[int] = None
    if target_h == ref_h and target_w == ref_w:
        inferred_scale = 1
    elif target_h % ref_h == 0 and target_w % ref_w == 0:
        scale_h = target_h // ref_h
        scale_w = target_w // ref_w
        if scale_h == scale_w and scale_h >= 1:
            inferred_scale = scale_h

    if inferred_scale is None and expected_scale != 1:
        raise ValueError(
            "IC-LoRA reference dimensions are incompatible with "
            f"ic_lora_reference_downscale_factor={expected_scale}: "
            f"target={target_h}x{target_w}, reference={ref_h}x{ref_w}."
        )
    if inferred_scale is not None and inferred_scale != expected_scale:
        raise ValueError(
            "IC-LoRA reference scale mismatch: "
            f"expected {expected_scale}, got {inferred_scale} "
            f"(target={target_h}x{target_w}, reference={ref_h}x{ref_w})."
        )

    runtime_scale = getattr(args, "_ic_lora_runtime_reference_downscale_factor", None)
    runtime_candidate = inferred_scale if inferred_scale is not None else expected_scale
    if runtime_scale is None:
        setattr(args, "_ic_lora_runtime_reference_downscale_factor", runtime_candidate)
    elif int(runtime_scale) != int(runtime_candidate):
        raise ValueError(
            "IC-LoRA reference downscale factor changed during training: "
            f"previous={runtime_scale}, current={runtime_candidate}."
        )

    if ref.shape[-2] != target_h or ref.shape[-1] != target_w:
        # Keep temporal length intact; only align spatial size.
        ref = F.interpolate(
            ref.to(torch.float32),
            size=(int(ref.shape[2]), target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )

    return ref.to(device=target_latents.device, dtype=dtype)


def _maybe_parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_score_from_payload(payload: Any, key: str) -> Optional[float]:
    direct = _maybe_parse_float(payload)
    if direct is not None:
        return direct
    if not isinstance(payload, dict):
        return None

    dot_parts = [part for part in key.split(".") if part]
    node: Any = payload
    if dot_parts:
        for part in dot_parts:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        nested = _maybe_parse_float(node)
        if nested is not None:
            return nested

    for candidate in (
        key,
        key.lower(),
        "sync_score",
        "syncScore",
        "sync",
    ):
        score = _maybe_parse_float(payload.get(candidate))
        if score is not None:
            return score
    return None


def _strip_window_suffix(path_no_ext: str) -> str:
    basename = os.path.basename(path_no_ext)
    match = _WINDOW_SUFFIX_RE.search(basename)
    if not match:
        return path_no_ext
    return os.path.join(os.path.dirname(path_no_ext), basename[: match.start()])


def _extract_sync_score_from_caption(caption: str, key: str) -> Optional[float]:
    if not caption:
        return None
    escaped = re.escape(key)
    patterns = (
        rf"(?i)\b{escaped}\b\s*[:=]\s*([0-9]*\.?[0-9]+)",
        rf"(?i)\"{escaped}\"\s*:\s*([0-9]*\.?[0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, caption)
        if match:
            return _maybe_parse_float(match.group(1))
    return None


def _extract_sync_score_from_sidecar(item: Any, key: str) -> Optional[float]:
    item_key = getattr(item, "item_key", None)
    if not isinstance(item_key, str) or not item_key:
        return None

    base_no_ext = _strip_window_suffix(os.path.splitext(item_key)[0])
    candidates = (
        f"{base_no_ext}.json",
        f"{base_no_ext}.metadata.json",
        f"{base_no_ext}_meta.json",
    )
    for json_path in candidates:
        cache_key = (json_path, key)
        if cache_key in _SYNC_SCORE_CACHE:
            score = _SYNC_SCORE_CACHE[cache_key]
            if score is not None:
                return score
            continue
        if not os.path.exists(json_path):
            _SYNC_SCORE_CACHE[cache_key] = None
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            score = _extract_score_from_payload(payload, key)
        except Exception:
            score = None
        _SYNC_SCORE_CACHE[cache_key] = score
        if score is not None:
            return score
    return None


def _extract_item_sync_score(item: Any, key: str) -> Optional[float]:
    direct = _maybe_parse_float(getattr(item, key, None))
    if direct is not None:
        return direct
    attr_key = key.replace(".", "_")
    direct = _maybe_parse_float(getattr(item, attr_key, None))
    if direct is not None:
        return direct
    caption = getattr(item, "caption", "")
    if isinstance(caption, str):
        parsed = _extract_sync_score_from_caption(caption, key)
        if parsed is not None:
            return parsed
    return _extract_sync_score_from_sidecar(item, key)


def _build_ic_mask_from_batch(
    batch: dict[str, Any],
    latents: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build a broadcastable [B,1,F,H,W] mask from batch['mask_signal']."""
    mask = batch.get("mask_signal")
    if not isinstance(mask, torch.Tensor):
        return None

    mask_t = mask
    if mask_t.dim() == 4:
        mask_t = mask_t.unsqueeze(1)  # B,F,H,W -> B,1,F,H,W
    elif mask_t.dim() != 5:
        return None

    if mask_t.shape[1] != 1:
        # Accept mask channels and collapse to one mask channel.
        mask_t = mask_t.mean(dim=1, keepdim=True)

    # Normalize to [0,1].
    mask_t = mask_t.to(device=latents.device, dtype=torch.float32)
    if mask_t.min() < 0.0 or mask_t.max() > 1.0:
        mask_t = (mask_t + 1.0) / 2.0
    mask_t = torch.clamp(mask_t, 0.0, 1.0)

    target_f = int(latents.shape[2])
    target_h = int(latents.shape[3])
    target_w = int(latents.shape[4])
    if (
        int(mask_t.shape[2]) != target_f
        or int(mask_t.shape[3]) != target_h
        or int(mask_t.shape[4]) != target_w
    ):
        mask_t = F.interpolate(
            mask_t,
            size=(target_f, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
    return mask_t


def should_skip_ic_lora_batch(args: Any, batch: dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return whether the current batch should be skipped by IC-LoRA rules."""
    if not is_ic_lora_enabled(args):
        return False, None

    require_reference = bool(getattr(args, "ic_lora_require_reference", True))
    presence = batch.get("control_signal_present")
    if require_reference and isinstance(presence, torch.Tensor):
        missing_count = int((~presence.to(torch.bool)).sum().item())
        if missing_count > 0:
            return (
                True,
                f"IC-LoRA skipped batch: missing paired references for {missing_count}/{presence.numel()} samples.",
            )

    item_infos = batch.get("item_info")
    if not isinstance(item_infos, list):
        item_infos = []
    if bool(getattr(args, "ic_lora_enforce_panel_tags", False)) and item_infos:
        missing_tags = 0
        for item in item_infos:
            caption = getattr(item, "caption", "")
            if not isinstance(caption, str) or not _PANEL_TAG_RE.search(caption):
                missing_tags += 1
        if missing_tags > 0:
            return (
                True,
                f"IC-LoRA skipped batch: {missing_tags}/{len(item_infos)} captions are missing panel tags like [IMAGE1].",
            )

    min_sync_score = getattr(args, "ic_lora_min_sync_score", None)
    if min_sync_score is None:
        return False, None
    min_sync_score = float(min_sync_score)

    if len(item_infos) == 0:
        return False, None

    sync_key = str(getattr(args, "ic_lora_sync_score_key", "sync_score") or "sync_score")
    scores: list[float] = []
    for item in item_infos:
        score = _extract_item_sync_score(item, sync_key)
        if score is not None:
            scores.append(float(score))

    if not scores:
        if not bool(getattr(args, _WARNED_MISSING_SYNC_SCORE_FLAG, False)):
            logger.warning(
                "IC-LoRA sync filtering enabled (min=%.3f) but no '%s' score found in captions/sidecars; filtering is skipped.",
                min_sync_score,
                sync_key,
            )
            setattr(args, _WARNED_MISSING_SYNC_SCORE_FLAG, True)
        return False, None

    batch_min = min(scores)
    if batch_min < min_sync_score:
        return (
            True,
            f"IC-LoRA skipped batch: minimum sync score {batch_min:.3f} is below threshold {min_sync_score:.3f}.",
        )
    return False, None


def prepare_ic_lora_model_input(
    args: Any,
    batch: dict[str, Any],
    latents: torch.Tensor,
    noisy_model_input: torch.Tensor,
    network_dtype: torch.dtype,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build model input for IC-LoRA by prepending clean reference latents.

    Returns:
        model_input: Tensor for model forward.
        reference_frame_count: Number of prepended reference frames.
        conditioned_first_frame_mask: Optional bool tensor [B], True where first-frame
            conditioning was applied.
        masked_loss_mask: Optional float tensor [B,1,F,H,W] where 1 indicates
            regions to optimize when masked conditioning is enabled.
    """
    if not is_ic_lora_enabled(args):
        return noisy_model_input, 0, None, None

    reference = batch.get("control_signal")
    require_reference = bool(getattr(args, "ic_lora_require_reference", True))

    if not isinstance(reference, torch.Tensor):
        if require_reference:
            raise ValueError(
                "IC-LoRA requires reference latents in batch['control_signal']. "
                "Enable paired reference loading and ensure cached latents are available."
            )
        return noisy_model_input, 0, None, None

    presence = batch.get("control_signal_present")
    if (
        require_reference
        and isinstance(presence, torch.Tensor)
        and bool((~presence.to(torch.bool)).any().item())
    ):
        missing_count = int((~presence.to(torch.bool)).sum().item())
        raise ValueError(
            f"IC-LoRA requires references for all batch items; {missing_count} items are missing references."
        )

    ref_latents = _normalize_reference_latents(reference, latents, network_dtype, args)
    if ref_latents is None:
        if require_reference:
            raise ValueError(
                "IC-LoRA reference tensor shape is incompatible with target latents. "
                "Expected a latent-like tensor convertible to [B,C,F,H,W]."
            )
        return noisy_model_input, 0, None, None

    ref_dropout_p = float(getattr(args, "ic_lora_reference_dropout_p", 0.0) or 0.0)
    if ref_dropout_p > 0.0:
        drop_mask = torch.rand((ref_latents.shape[0],), device=ref_latents.device) < ref_dropout_p
        if bool(drop_mask.any().item()):
            ref_latents = ref_latents.clone()
            ref_latents[drop_mask] = 0.0

    conditioned_first_frame_mask: Optional[torch.Tensor] = None
    masked_loss_mask: Optional[torch.Tensor] = None
    first_frame_p = float(getattr(args, "ic_lora_first_frame_conditioning_p", 0.0))
    target_noisy = noisy_model_input
    if first_frame_p > 0.0 and target_noisy.shape[2] > 0:
        apply_mask = torch.rand(
            (target_noisy.shape[0],), device=target_noisy.device
        ) < first_frame_p
        if bool(apply_mask.any().item()):
            conditioned_first_frame_mask = apply_mask
            target_noisy = target_noisy.clone()
            target_noisy[apply_mask, :, :1, :, :] = latents[
                apply_mask, :, :1, :, :
            ].to(
                device=target_noisy.device,
                dtype=target_noisy.dtype,
            )

    if bool(getattr(args, "ic_lora_use_masked_conditioning", False)):
        spatial_mask = _build_ic_mask_from_batch(batch=batch, latents=latents)
        if spatial_mask is None:
            logger.warning(
                "IC-LoRA masked conditioning enabled but batch['mask_signal'] is missing or invalid; using unmasked conditioning."
            )
        else:
            mask_bool = spatial_mask > 0.5
            target_noisy = torch.where(
                mask_bool,
                target_noisy,
                latents.to(device=target_noisy.device, dtype=target_noisy.dtype),
            )
            if bool(getattr(args, "ic_lora_masked_loss_only", False)):
                masked_loss_mask = mask_bool.to(dtype=target_noisy.dtype)

    model_input = torch.cat([ref_latents, target_noisy], dim=2)
    return (
        model_input,
        int(ref_latents.shape[2]),
        conditioned_first_frame_mask,
        masked_loss_mask,
    )
