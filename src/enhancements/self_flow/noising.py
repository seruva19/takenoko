from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch

from scheduling.timestep_utils import _apply_timestep_constraints, map_uniform_to_sampling


@dataclass
class SelfFlowDualTimestepContext:
    """Container for Self-Flow dual-timestep tensors used in loss computation."""

    student_noisy_model_input: torch.Tensor
    teacher_noisy_model_input: torch.Tensor
    student_model_timesteps: torch.Tensor
    teacher_model_timesteps: torch.Tensor
    base_timesteps: torch.Tensor
    masked_token_ratio: float
    tau_mean: float
    tau_min_mean: float
    sequence_length: int


def _resolve_patch_size(patch_size: Tuple[int, int, int], ndim: int) -> Tuple[int, int, int]:
    if ndim == 5:
        return int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    # 4D image latents are treated as F=1.
    return 1, int(patch_size[1]), int(patch_size[2])


def _token_shape_from_latents(
    latents: torch.Tensor,
    patch_size: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    if latents.ndim == 5:
        _, _, f, h, w = latents.shape
    elif latents.ndim == 4:
        _, _, h, w = latents.shape
        f = 1
    else:
        raise ValueError(f"Self-Flow expects 4D or 5D latents, got ndim={latents.ndim}")

    pt, ph, pw = _resolve_patch_size(patch_size, latents.ndim)
    if f % pt != 0 or h % ph != 0 or w % pw != 0:
        raise ValueError(
            "Self-Flow dual-timestep requires latent dimensions divisible by patch_size. "
            f"Got latents=(F={f},H={h},W={w}), patch_size=({pt},{ph},{pw})."
        )
    return f // pt, h // ph, w // pw


def _expand_tokenwise_tau_to_latents(
    tau_tokenwise: torch.Tensor,
    latents: torch.Tensor,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    bsz = latents.shape[0]
    ft, ht, wt = _token_shape_from_latents(latents, patch_size)
    pt, ph, pw = _resolve_patch_size(patch_size, latents.ndim)

    if latents.ndim == 5:
        _, _, f, h, w = latents.shape
        tau_grid = tau_tokenwise.view(bsz, ft, ht, wt)
        tau_latent = (
            tau_grid.repeat_interleave(pt, dim=1)
            .repeat_interleave(ph, dim=2)
            .repeat_interleave(pw, dim=3)
        )
        tau_latent = tau_latent[:, :f, :h, :w].unsqueeze(1)
    else:
        _, _, h, w = latents.shape
        tau_grid = tau_tokenwise.view(bsz, ht, wt)
        tau_latent = tau_grid.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)
        tau_latent = tau_latent[:, :h, :w].unsqueeze(1)

    return tau_latent.to(device=latents.device, dtype=latents.dtype)


def _infer_modality_hint(
    latents: torch.Tensor,
    batch: Mapping[str, Any] | None,
    args: Any,
) -> str:
    """
    Infer modality for Self-Flow auto mask ratio routing.

    Priority:
    1) Explicit arg hint (`self_flow_input_modality`) for future model ports.
    2) Batch-level hints (modality/data_type keys or audio feature tensors).
    3) Latent layout heuristic (video if F>1, else image).
    """
    explicit = str(getattr(args, "self_flow_input_modality", "")).strip().lower()
    if explicit in {"image", "video", "audio"}:
        return explicit

    if batch:
        for key in ("modality", "data_type", "dataset_type"):
            value = batch.get(key)
            if isinstance(value, str):
                hint = value.strip().lower()
                if "audio" in hint:
                    return "audio"
                if "video" in hint:
                    return "video"
                if "image" in hint:
                    return "image"

        audio_keys = (
            "audio",
            "audio_latents",
            "audio_features",
            "audio_embeddings",
            "mel",
            "mel_spec",
            "spectrogram",
            "waveform",
            "wav",
        )
        for key in audio_keys:
            value = batch.get(key)
            if torch.is_tensor(value):
                return "audio"
            if isinstance(value, (list, tuple)) and any(
                torch.is_tensor(item) for item in value
            ):
                return "audio"

    if latents.ndim == 5 and latents.shape[2] > 1:
        return "video"
    return "image"


def _select_mask_ratio(
    args: Any,
    latents: torch.Tensor,
    batch: Mapping[str, Any] | None = None,
) -> float:
    mode = str(getattr(args, "self_flow_mask_ratio_mode", "auto")).lower()
    if mode == "fixed":
        ratio = float(getattr(args, "self_flow_mask_ratio", 0.25))
    else:
        modality = _infer_modality_hint(latents, batch, args)
        if modality == "audio":
            ratio = float(getattr(args, "self_flow_mask_ratio_audio", 0.50))
        elif modality == "video":
            ratio = float(getattr(args, "self_flow_mask_ratio_video", 0.10))
        else:
            ratio = float(getattr(args, "self_flow_mask_ratio_image", 0.25))
    return max(0.0, min(0.5, ratio))


def reduce_model_timesteps_for_runtime(model_timesteps: torch.Tensor) -> torch.Tensor:
    """
    Reduce tokenwise/self-flow timesteps to per-sample shape `[B]`.

    This is used for components that only accept one timestep per sample
    (loss weighting, adapter timestep gates, and similar utilities).
    """
    if model_timesteps.dim() <= 1:
        return model_timesteps
    reduced = model_timesteps.float().view(model_timesteps.size(0), -1).mean(dim=1)
    return reduced.to(dtype=model_timesteps.dtype)


def _sample_secondary_timestep(
    args: Any,
    latents: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    u = torch.rand((batch_size,), device=device, dtype=torch.float32)
    sampled = map_uniform_to_sampling(args, u, latents)
    sampled = _apply_timestep_constraints(
        sampled.view(-1),
        args,
        batch_size,
        device,
        latents,
        presampled_uniform=None,
    )

    lower = float(getattr(args, "self_flow_timestep_lower_bound", 0.001))
    upper = float(getattr(args, "self_flow_timestep_upper_bound", 0.999))
    return sampled.view(batch_size).clamp(lower, upper)


def maybe_apply_self_flow_dual_timestep(
    *,
    args: Any,
    latents: torch.Tensor,
    noise: torch.Tensor,
    base_timesteps: torch.Tensor,
    patch_size: Tuple[int, int, int],
    batch: Mapping[str, Any] | None = None,
) -> SelfFlowDualTimestepContext | None:
    """Apply Self-Flow dual-timestep noising and return context tensors."""
    if not bool(getattr(args, "enable_self_flow", False)):
        return None
    if not bool(getattr(args, "self_flow_enable_dual_timestep", True)):
        return None

    if latents.shape != noise.shape:
        raise ValueError(
            "Self-Flow dual-timestep requires latents and noise shapes to match. "
            f"Got latents={tuple(latents.shape)} noise={tuple(noise.shape)}."
        )

    bsz = latents.shape[0]
    strict_mode = bool(getattr(args, "self_flow_strict_mode", True))
    if base_timesteps.dim() != 1:
        if strict_mode:
            raise ValueError(
                "Self-Flow strict mode requires scalar base timesteps of shape [B]. "
                f"Got shape {tuple(base_timesteps.shape)}."
            )
        t_base = base_timesteps.float().view(bsz, -1).mean(dim=1)
    else:
        t_base = base_timesteps.float()

    t_base = ((t_base - 1.0) / 1000.0).clamp(
        float(getattr(args, "self_flow_timestep_lower_bound", 0.001)),
        float(getattr(args, "self_flow_timestep_upper_bound", 0.999)),
    )
    s_base = _sample_secondary_timestep(args, latents, bsz, latents.device)

    ft, ht, wt = _token_shape_from_latents(latents, patch_size)
    seq_len = int(ft * ht * wt)
    t_tokens = t_base.unsqueeze(1).expand(-1, seq_len)
    s_tokens = s_base.unsqueeze(1).expand(-1, seq_len)

    mask_ratio = _select_mask_ratio(args, latents, batch=batch)
    mask = torch.rand((bsz, seq_len), device=latents.device) < mask_ratio
    tau_tokens = torch.where(mask, s_tokens, t_tokens)
    tau_min_tokens = torch.minimum(t_tokens, s_tokens)

    tau_latent = _expand_tokenwise_tau_to_latents(tau_tokens, latents, patch_size)
    tau_min_latent = _expand_tokenwise_tau_to_latents(
        tau_min_tokens, latents, patch_size
    )

    student_noisy = (1.0 - tau_latent) * latents + tau_latent * noise
    teacher_noisy = (1.0 - tau_min_latent) * latents + tau_min_latent * noise

    force_tokenwise = bool(getattr(args, "self_flow_force_tokenwise_timesteps", True))
    if force_tokenwise:
        student_t = tau_tokens * 1000.0 + 1.0
        teacher_t = tau_min_tokens * 1000.0 + 1.0
    else:
        student_t = tau_tokens.mean(dim=1) * 1000.0 + 1.0
        teacher_t = tau_min_tokens.mean(dim=1) * 1000.0 + 1.0

    return SelfFlowDualTimestepContext(
        student_noisy_model_input=student_noisy,
        teacher_noisy_model_input=teacher_noisy,
        student_model_timesteps=student_t.to(dtype=base_timesteps.dtype),
        teacher_model_timesteps=teacher_t.to(dtype=base_timesteps.dtype),
        base_timesteps=base_timesteps,
        masked_token_ratio=float(mask.float().mean().item()),
        tau_mean=float(tau_tokens.mean().item()),
        tau_min_mean=float(tau_min_tokens.mean().item()),
        sequence_length=seq_len,
    )
