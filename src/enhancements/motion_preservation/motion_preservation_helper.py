import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from common.logger import get_logger
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from scheduling.timestep_utils import get_noisy_model_input_and_timesteps
from .motion_attention import (
    AttentionMapRecorder,
    collect_motion_attention_modules,
    compute_motion_attention_loss,
    filter_motion_attention_modules_for_swap,
)
from .motion_runtime import MotionReplayHealthTracker

logger = get_logger(__name__)


@dataclass
class MotionAnchorVariant:
    sigma: float
    noisy_input: torch.Tensor
    model_timesteps: torch.Tensor
    teacher_pred: torch.Tensor
    teacher_attention_maps: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class MotionAnchor:
    latents: torch.Tensor
    noise: torch.Tensor
    batch: Dict[str, Any]
    variants: List[MotionAnchorVariant]
    source: str


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _move_to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, list):
        return [_move_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, dtype) for item in value)
    if isinstance(value, dict):
        return {
            key: moved
            for key, item in value.items()
            if (moved := _move_to_device(item, device, dtype)) is not None
        }
    return value


@contextmanager
def _temporary_network_multiplier(
    accelerator: Optional[Accelerator],
    network: Optional[Any],
    multiplier: float,
):
    if network is None:
        yield
        return

    try:
        unwrapped = (
            accelerator.unwrap_model(network)
            if accelerator is not None
            else network
        )
    except Exception:
        unwrapped = network

    setter = getattr(unwrapped, "set_multiplier", None)
    had_multiplier_attr = hasattr(unwrapped, "multiplier")
    previous_multiplier = getattr(unwrapped, "multiplier", None)
    changed = False
    try:
        if callable(setter):
            setter(multiplier)
            changed = True
        elif had_multiplier_attr:
            setattr(unwrapped, "multiplier", multiplier)
            changed = True
        yield
    finally:
        if not changed:
            return
        try:
            if callable(setter):
                setter(previous_multiplier)
            elif had_multiplier_attr:
                setattr(unwrapped, "multiplier", previous_multiplier)
        except Exception:
            logger.warning("Failed to restore network multiplier after motion replay teacher pass.")


def _extract_anchor_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    anchor_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"latents", "pixels"}:
            continue
        cloned = _clone_to_cpu(value)
        if cloned is not None:
            anchor_batch[key] = cloned
    if "t5" not in anchor_batch and "text_embeds" in anchor_batch:
        text_embeds = anchor_batch["text_embeds"]
        anchor_batch["t5"] = text_embeds if isinstance(text_embeds, list) else [text_embeds]
    return anchor_batch


def _build_synthetic_motion_latents(
    base_latents: torch.Tensor,
    *,
    target_frames: int,
    temporal_corr: float,
    content_seeded: bool,
) -> torch.Tensor:
    if base_latents.dim() != 5:
        raise ValueError(
            f"Expected 5D base latents for synthetic motion anchors, got {tuple(base_latents.shape)}"
        )

    batch_size, channels, _, height, width = base_latents.shape
    frames = max(2, int(target_frames))
    corr = max(0.0, min(0.999, float(temporal_corr)))
    noise_scale = math.sqrt(max(1e-6, 1.0 - corr * corr))

    synth = torch.empty(
        (batch_size, channels, frames, height, width),
        device=base_latents.device,
        dtype=base_latents.dtype,
    )
    if content_seeded:
        prev = base_latents[:, :, 0, :, :].clone()
    else:
        prev = torch.randn(
            (batch_size, channels, height, width),
            device=base_latents.device,
            dtype=base_latents.dtype,
        )
    synth[:, :, 0, :, :] = prev
    for frame_idx in range(1, frames):
        prev = corr * prev + noise_scale * torch.randn_like(prev)
        synth[:, :, frame_idx, :, :] = prev

    if not content_seeded:
        base_mean = base_latents.float().mean()
        base_std = base_latents.float().std(unbiased=False).clamp_min(1e-6)
        synth_mean = synth.float().mean()
        synth_std = synth.float().std(unbiased=False).clamp_min(1e-6)
        synth = (synth - synth_mean.to(dtype=synth.dtype)) * (
            base_std / synth_std
        ).to(dtype=synth.dtype)
        synth = synth + base_mean.to(dtype=synth.dtype)
    return synth


def _resolve_motion_sigmas(args: argparse.Namespace) -> List[float]:
    explicit = getattr(args, "motion_preservation_sigma_values", None)
    if explicit:
        sigmas = [float(sigma) for sigma in explicit]
    else:
        num_sigmas = int(getattr(args, "motion_preservation_num_sigmas", 1) or 1)
        if num_sigmas <= 1:
            return []
        sigma_min = float(getattr(args, "motion_preservation_sigma_min", 0.2))
        sigma_max = float(getattr(args, "motion_preservation_sigma_max", 0.8))
        sigmas = torch.linspace(
            sigma_min, sigma_max, steps=num_sigmas, dtype=torch.float32
        ).tolist()
    unique_sigmas = sorted({round(float(sigma), 6) for sigma in sigmas})
    return unique_sigmas


def _compute_sigma_weights(
    sigmas: List[float], args: argparse.Namespace
) -> Optional[List[float]]:
    if len(sigmas) <= 1:
        return None
    mode = str(
        getattr(args, "motion_preservation_sigma_sampling", "uniform") or "uniform"
    ).lower()
    if mode != "logsnr":
        return None
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    logsnr = torch.log(((1.0 - sigma_tensor) ** 2) / (sigma_tensor**2))
    weights = 1.0 / (1.0 + logsnr.abs())
    power = float(getattr(args, "motion_preservation_sigma_sampling_power", 1.0) or 1.0)
    weights = weights.clamp_min(1e-8).pow(power)
    weights = weights / weights.sum().clamp_min(1e-8)
    return [float(weight) for weight in weights.tolist()]


def _sample_sigma_index(sigmas: List[float], args: argparse.Namespace) -> int:
    if len(sigmas) <= 1:
        return 0
    weights = _compute_sigma_weights(sigmas, args)
    if not weights:
        return random.randrange(len(sigmas))
    return int(torch.multinomial(torch.tensor(weights), 1, replacement=True).item())


def _build_noisy_input_for_sigma(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
    noise_scheduler: FlowMatchDiscreteScheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_tensor = latents.new_full((latents.shape[0],), float(sigma))
    sigma_broadcast = sigma_tensor.view(-1, *([1] * (latents.dim() - 1)))
    noisy_input = sigma_broadcast * noise + (1.0 - sigma_broadcast) * latents
    max_ts = float(
        getattr(getattr(noise_scheduler, "config", object()), "num_train_timesteps", 1000)
    )
    model_timesteps = sigma_tensor * max_ts
    return noisy_input, model_timesteps


def _infer_sigma_from_timesteps(
    model_timesteps: torch.Tensor,
    noise_scheduler: FlowMatchDiscreteScheduler,
) -> float:
    max_ts = float(
        getattr(getattr(noise_scheduler, "config", object()), "num_train_timesteps", 1000)
    )
    return float((model_timesteps.float().mean() / max(max_ts, 1.0)).item())


def _path_mtime(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    return float(os.path.getmtime(path))


def _signature_hash(signature: Dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache_payload(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        logger.exception("Failed to load motion anchor cache, rebuilding: %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _save_cache_payload(path: str, payload: Dict[str, Any]) -> None:
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    torch.save(payload, path)


def _chunked_motion_replay_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    mode: str,
    chunk_frames: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if (
        teacher_pred.device.type != "cpu"
        or chunk_frames <= 0
        or student_pred.dim() != 5
        or teacher_pred.dim() != 5
    ):
        return None

    if mode == "temporal" and student_pred.shape[2] > 1 and teacher_pred.shape[2] > 1:
        total = student_pred.new_tensor(0.0, dtype=torch.float32)
        total_count = 0
        for start in range(0, student_pred.shape[2] - 1, chunk_frames):
            end = min(student_pred.shape[2], start + chunk_frames + 1)
            student_chunk = student_pred[:, :, start:end, :, :].to(torch.float32)
            teacher_chunk = teacher_pred[:, :, start:end, :, :].to(
                device=student_pred.device,
                dtype=dtype,
                non_blocking=True,
            ).to(torch.float32)
            student_delta = student_chunk[:, :, 1:, :, :] - student_chunk[:, :, :-1, :, :]
            teacher_delta = teacher_chunk[:, :, 1:, :, :] - teacher_chunk[:, :, :-1, :, :]
            diff = student_delta - teacher_delta
            total = total + diff.square().sum()
            total_count += diff.numel()
        return (total / max(total_count, 1)).to(dtype=dtype)

    if mode == "full":
        total = student_pred.new_tensor(0.0, dtype=torch.float32)
        total_count = 0
        for start in range(0, student_pred.shape[2], chunk_frames):
            end = min(student_pred.shape[2], start + chunk_frames)
            student_chunk = student_pred[:, :, start:end, :, :].to(torch.float32)
            teacher_chunk = teacher_pred[:, :, start:end, :, :].to(
                device=student_pred.device,
                dtype=dtype,
                non_blocking=True,
            ).to(torch.float32)
            diff = student_chunk - teacher_chunk
            total = total + diff.square().sum()
            total_count += diff.numel()
        return (total / max(total_count, 1)).to(dtype=dtype)

    return None


def _extract_model_pred(model_result: Any) -> torch.Tensor:
    if isinstance(model_result, tuple):
        return model_result[0]
    return model_result


def _compute_motion_replay_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    mode: str,
    second_order_weight: float,
    chunk_frames: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, bool]:
    use_temporal = (
        mode == "temporal"
        and student_pred.dim() == 5
        and teacher_pred.dim() == 5
        and student_pred.shape[2] > 1
        and teacher_pred.shape[2] > 1
    )
    if teacher_pred.device.type == "cpu" and chunk_frames > 0 and second_order_weight <= 0.0:
        chunked = _chunked_motion_replay_loss(
            student_pred,
            teacher_pred,
            mode="temporal" if use_temporal else "full",
            chunk_frames=chunk_frames,
            dtype=dtype,
        )
        if chunked is not None:
            return chunked, mode == "temporal" and not use_temporal
    if teacher_pred.device != student_pred.device:
        teacher_pred = teacher_pred.to(
            device=student_pred.device,
            dtype=dtype,
            non_blocking=True,
        )
    else:
        teacher_pred = teacher_pred.to(dtype=dtype)
    if not use_temporal:
        return F.mse_loss(student_pred, teacher_pred), mode == "temporal"

    student_delta = student_pred[:, :, 1:, :, :] - student_pred[:, :, :-1, :, :]
    teacher_delta = teacher_pred[:, :, 1:, :, :] - teacher_pred[:, :, :-1, :, :]
    loss = F.mse_loss(student_delta, teacher_delta)
    if second_order_weight > 0.0 and student_pred.shape[2] > 2 and teacher_pred.shape[2] > 2:
        student_accel = (
            student_pred[:, :, 2:, :, :]
            - 2.0 * student_pred[:, :, 1:-1, :, :]
            + student_pred[:, :, :-2, :, :]
        )
        teacher_accel = (
            teacher_pred[:, :, 2:, :, :]
            - 2.0 * teacher_pred[:, :, 1:-1, :, :]
            + teacher_pred[:, :, :-2, :, :]
        )
        loss = loss + float(second_order_weight) * F.mse_loss(
            student_accel, teacher_accel
        )
    return loss, False


class MotionPreservationHelper:
    def __init__(self, training_core: Any, args: argparse.Namespace, config: Any):
        self.training_core = training_core
        self.args = args
        self.config = config
        self.anchor_cache: List[MotionAnchor] = []
        self.replay_sigmas = _resolve_motion_sigmas(args)
        self.attention_modules: List[tuple[str, torch.nn.Module]] = []
        self.last_loss: Optional[torch.Tensor] = None
        self.last_weight: Optional[torch.Tensor] = None
        self.last_sigma: Optional[torch.Tensor] = None
        self.last_anchor_source: Optional[torch.Tensor] = None
        self.last_temporal_fallback: Optional[torch.Tensor] = None
        self.last_attention_loss: Optional[torch.Tensor] = None
        self.last_anchor_frames: Optional[torch.Tensor] = None
        self.last_total_to_task_ratio: Optional[torch.Tensor] = None
        self.health = MotionReplayHealthTracker()
        self._runtime_state_filename = "motion_preservation_runtime_state.json"
        self._publish_runtime_attributes()

    def setup_hooks(self) -> None:
        return None

    def remove_hooks(self) -> None:
        return None

    def save_runtime_state(self, output_dir: str) -> None:
        payload = {
            "version": 1,
            "health": self.health.as_dict(),
        }
        path = os.path.join(output_dir, self._runtime_state_filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def load_runtime_state(self, input_dir: str) -> None:
        path = os.path.join(input_dir, self._runtime_state_filename)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("health"), dict):
                self.health.load_dict(payload["health"])
                self._publish_runtime_attributes()
        except Exception as exc:
            logger.warning("Failed to load motion-preservation runtime state: %s", exc)

    def has_anchors(self) -> bool:
        return len(self.anchor_cache) > 0

    def count_temporal_anchors(self) -> int:
        return sum(
            1
            for anchor in self.anchor_cache
            if anchor.latents.dim() == 5 and int(anchor.latents.shape[2]) > 1
        )

    def _strict_mode_enabled(self) -> bool:
        return bool(getattr(self.args, "motion_preservation_strict_mode", True))

    def _publish_runtime_attributes(self) -> None:
        self.health.publish_to_args(self.args)

    def _maybe_log_health_summary(self) -> None:
        interval = int(
            getattr(self.args, "motion_preservation_health_log_interval", 100) or 0
        )
        if interval <= 0 or self.health.invocations <= 0:
            return
        if self.health.invocations % interval != 0:
            return
        snapshot = self.health.as_dict()
        logger.info(
            "Motion replay health: invocations=%d applied=%d apply_rate=%.3f schedule_skip_rate=%.3f temporal_fallback_rate=%.3f anchors=%d temporal=%d synthetic=%d dataset=%d attention_applied=%d last_error=%s",
            snapshot["invocations"],
            snapshot["applied"],
            snapshot["apply_rate"],
            snapshot["schedule_skip_rate"],
            snapshot["temporal_fallback_rate"],
            snapshot["anchor_cache_size"],
            snapshot["temporal_anchor_count"],
            snapshot["synthetic_anchor_count"],
            snapshot["dataset_anchor_count"],
            snapshot["attention_applied"],
            snapshot["last_error"] or "none",
        )

    def record_skip(self, reason: str) -> None:
        self.health.note_skip(reason)
        self._publish_runtime_attributes()
        self._maybe_log_health_summary()

    def _handle_validation_issue(self, message: str, *, reason: str) -> bool:
        self.health.note_error(message)
        if self._strict_mode_enabled():
            raise ValueError(message)
        logger.warning("%s", message)
        self.health.note_skip(reason)
        self._publish_runtime_attributes()
        self._maybe_log_health_summary()
        return False

    def _validate_variant(
        self,
        *,
        anchor_latents: torch.Tensor,
        variant: MotionAnchorVariant,
    ) -> bool:
        if not isinstance(variant.noisy_input, torch.Tensor):
            return self._handle_validation_issue(
                "Motion replay anchor is missing noisy_input tensor.",
                reason="invalid_anchor",
            )
        if not isinstance(variant.model_timesteps, torch.Tensor):
            return self._handle_validation_issue(
                "Motion replay anchor is missing model_timesteps tensor.",
                reason="invalid_anchor",
            )
        if not isinstance(variant.teacher_pred, torch.Tensor):
            return self._handle_validation_issue(
                "Motion replay anchor is missing teacher_pred tensor.",
                reason="invalid_anchor",
            )
        if tuple(variant.noisy_input.shape) != tuple(anchor_latents.shape):
            return self._handle_validation_issue(
                f"Motion replay anchor noisy_input shape mismatch: expected {tuple(anchor_latents.shape)} got {tuple(variant.noisy_input.shape)}.",
                reason="invalid_anchor",
            )
        if tuple(variant.teacher_pred.shape) != tuple(anchor_latents.shape):
            return self._handle_validation_issue(
                f"Motion replay teacher_pred shape mismatch: expected {tuple(anchor_latents.shape)} got {tuple(variant.teacher_pred.shape)}.",
                reason="invalid_anchor",
            )
        timestep_batch = int(variant.model_timesteps.shape[0]) if variant.model_timesteps.ndim > 0 else 1
        if timestep_batch not in {1, int(anchor_latents.shape[0])}:
            return self._handle_validation_issue(
                f"Motion replay model_timesteps batch mismatch: expected 1 or {int(anchor_latents.shape[0])} got {timestep_batch}.",
                reason="invalid_anchor",
            )
        return True

    def configure_attention_modules(
        self,
        transformer: torch.nn.Module,
        accelerator: Accelerator,
        *,
        blocks_to_swap: int = 0,
    ) -> None:
        self.attention_modules = []
        if not bool(getattr(self.args, "motion_attention_preservation", False)):
            return

        modules = collect_motion_attention_modules(
            transformer,
            getattr(self.args, "motion_attention_preservation_blocks", None),
        )
        modules = filter_motion_attention_modules_for_swap(
            modules,
            transformer=transformer,
            accelerator=accelerator,
            blocks_to_swap=blocks_to_swap,
        )
        if not modules:
            logger.warning(
                "motion_attention_preservation requested but no matching Wan self-attention modules were found; disabling it."
            )
            self.args.motion_attention_preservation = False
            return

        self.attention_modules = modules
        logger.info(
            "Motion attention preservation active: modules=%d queries=%d keys=%d loss=%s per_head=%s temperature=%.3f symmetric_kl=%s",
            len(self.attention_modules),
            int(getattr(self.args, "motion_attention_preservation_queries", 32) or 32),
            int(getattr(self.args, "motion_attention_preservation_keys", 64) or 64),
            str(getattr(self.args, "motion_attention_preservation_loss", "kl")),
            str(bool(getattr(self.args, "motion_attention_preservation_per_head", False))),
            float(getattr(self.args, "motion_attention_preservation_temperature", 1.0) or 1.0),
            str(bool(getattr(self.args, "motion_attention_preservation_symmetric_kl", False))),
        )

    def resolve_anchor_cache_size(self, dataloader_len: int) -> int:
        requested = int(getattr(self.args, "motion_preservation_anchor_cache_size", 0) or 0)
        if requested > 0:
            return requested
        if not bool(getattr(self.args, "motion_preservation_anchor_cache_auto", False)):
            return requested
        ratio = float(
            getattr(self.args, "motion_preservation_anchor_cache_auto_ratio", 0.2) or 0.2
        )
        min_size = int(
            getattr(self.args, "motion_preservation_anchor_cache_auto_min", 8) or 8
        )
        max_size = int(
            getattr(self.args, "motion_preservation_anchor_cache_auto_max", 256) or 256
        )
        resolved = int(math.ceil(max(1, dataloader_len) * ratio))
        resolved = max(min_size, min(max_size, resolved))
        self.args.motion_preservation_anchor_cache_size = resolved
        return resolved

    def _cache_signature(self) -> Dict[str, Any]:
        checkpoint_path = getattr(self.args, "dit", None)
        dataset_config = getattr(self.args, "dataset_config", None)
        return {
            "schema": 1,
            "kind": "motion_anchor_cache",
            "checkpoint": checkpoint_path,
            "checkpoint_mtime": _path_mtime(checkpoint_path),
            "dataset_config": dataset_config,
            "dataset_config_mtime": _path_mtime(dataset_config),
            "anchor_source": getattr(self.args, "motion_preservation_anchor_source", "synthetic"),
            "anchor_cache_size": int(getattr(self.args, "motion_preservation_anchor_cache_size", 0) or 0),
            "motion_mode": getattr(self.args, "motion_preservation_mode", "temporal"),
            "num_sigmas": int(getattr(self.args, "motion_preservation_num_sigmas", 1) or 1),
            "sigma_values": [float(sigma) for sigma in self.replay_sigmas],
            "synthetic_frames": int(getattr(self.args, "motion_preservation_synthetic_frames", 8) or 8),
            "synthetic_temporal_corr": float(getattr(self.args, "motion_preservation_synthetic_temporal_corr", 0.92)),
            "synthetic_dataset_mix": float(getattr(self.args, "motion_preservation_synthetic_dataset_mix", 0.25)),
            "synthetic_content_seeded": bool(getattr(self.args, "motion_preservation_synthetic_content_seeded", True)),
            "attention_preservation": bool(getattr(self.args, "motion_attention_preservation", False)),
            "attention_queries": int(getattr(self.args, "motion_attention_preservation_queries", 32) or 32),
            "attention_keys": int(getattr(self.args, "motion_attention_preservation_keys", 64) or 64),
            "attention_per_head": bool(getattr(self.args, "motion_attention_preservation_per_head", False)),
            "attention_blocks": getattr(self.args, "motion_attention_preservation_blocks", None),
        }

    def _load_anchor_cache(self, path: str) -> Optional[List[MotionAnchor]]:
        payload = _load_cache_payload(path)
        if not payload:
            return None
        if payload.get("signature_hash") != _signature_hash(self._cache_signature()):
            logger.info("Motion anchor cache signature mismatch; rebuilding: %s", path)
            return None
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            logger.warning("Motion anchor cache is empty or malformed; rebuilding: %s", path)
            return None

        entries: List[MotionAnchor] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            latents = raw_entry.get("latents")
            noise = raw_entry.get("noise")
            batch = raw_entry.get("batch")
            if not isinstance(latents, torch.Tensor) or not isinstance(noise, torch.Tensor) or not isinstance(batch, dict):
                continue

            variants: List[MotionAnchorVariant] = []
            for raw_variant in raw_entry.get("variants", []):
                if not isinstance(raw_variant, dict):
                    continue
                teacher_pred = raw_variant.get("teacher_pred")
                noisy_input = raw_variant.get("noisy_input")
                model_timesteps = raw_variant.get("model_timesteps")
                if not all(
                    isinstance(item, torch.Tensor)
                    for item in (teacher_pred, noisy_input, model_timesteps)
                ):
                    continue
                variants.append(
                    MotionAnchorVariant(
                        sigma=float(raw_variant.get("sigma", 0.0)),
                        noisy_input=noisy_input,
                        model_timesteps=model_timesteps,
                        teacher_pred=teacher_pred,
                        teacher_attention_maps=raw_variant.get("teacher_attention_maps"),
                    )
                )
            if not variants:
                continue
            anchor = MotionAnchor(
                latents=latents,
                noise=noise,
                batch=batch,
                variants=variants,
                source=str(raw_entry.get("source", "dataset")),
            )
            if not all(
                self._validate_variant(anchor_latents=anchor.latents, variant=variant)
                for variant in anchor.variants
            ):
                continue
            entries.append(anchor)
        if entries:
            logger.info("Loaded %d motion anchors from cache: %s", len(entries), path)
            self.health.note_cache(entries, loaded=True)
            self._publish_runtime_attributes()
        return entries or None

    def _save_anchor_cache(self, path: str, entries: List[MotionAnchor]) -> None:
        payload = {
            "version": 1,
            "signature_hash": _signature_hash(self._cache_signature()),
            "signature": self._cache_signature(),
            "created_at": float(time.time()),
            "entries": [
                {
                    "latents": entry.latents,
                    "noise": entry.noise,
                    "batch": entry.batch,
                    "source": entry.source,
                    "variants": [
                        {
                            "sigma": variant.sigma,
                            "noisy_input": variant.noisy_input,
                            "model_timesteps": variant.model_timesteps,
                            "teacher_pred": variant.teacher_pred,
                            "teacher_attention_maps": variant.teacher_attention_maps,
                        }
                        for variant in entry.variants
                    ],
                }
                for entry in entries
            ],
        }
        _save_cache_payload(path, payload)
        logger.info("Saved motion anchor cache: %s (entries=%d)", path, len(entries))

    def build_anchor_cache(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        network_dtype: torch.dtype,
        timestep_distribution: Any = None,
        network: Optional[Any] = None,
        use_base_model_teacher: bool = False,
    ) -> List[MotionAnchor]:
        cache_size = int(
            getattr(self.args, "motion_preservation_anchor_cache_size", 0) or 0
        )
        if cache_size <= 0:
            self.anchor_cache = []
            return self.anchor_cache
        cache_path = getattr(self.args, "motion_preservation_anchor_cache_path", None)
        rebuild_cache = bool(
            getattr(self.args, "motion_preservation_anchor_cache_rebuild", False)
        )
        if cache_path and not rebuild_cache:
            cached_entries = self._load_anchor_cache(cache_path)
            if cached_entries is not None:
                self.anchor_cache = cached_entries
                return self.anchor_cache
        elif cache_path and rebuild_cache:
            logger.info(
                "Motion anchor cache rebuild requested; ignoring existing cache: %s",
                cache_path,
            )

        anchor_source = str(
            getattr(self.args, "motion_preservation_anchor_source", "synthetic")
            or "synthetic"
        ).lower()
        synthetic_frames = int(
            getattr(self.args, "motion_preservation_synthetic_frames", 8) or 8
        )
        synthetic_temporal_corr = float(
            getattr(self.args, "motion_preservation_synthetic_temporal_corr", 0.92)
        )
        dataset_mix = float(
            getattr(self.args, "motion_preservation_synthetic_dataset_mix", 0.25)
        )
        content_seeded = bool(
            getattr(self.args, "motion_preservation_synthetic_content_seeded", True)
        )
        max_attempts = max(cache_size * 4, cache_size)
        entries: List[MotionAnchor] = []
        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=getattr(self.args, "discrete_flow_shift", 3.0),
            reverse=True,
            solver="euler",
        )
        skipped_bad_latents = 0
        single_frame_dataset_anchors = 0
        was_training = transformer.training

        logger.info(
            "Motion preservation anchor build: source=%s size=%d sigmas=%s",
            anchor_source,
            cache_size,
            self.replay_sigmas if self.replay_sigmas else ["schedule"],
        )
        if anchor_source in {"synthetic", "hybrid"}:
            logger.info(
                "Synthetic replay anchors: frames=%d temporal_corr=%.3f dataset_mix=%.2f content_seeded=%s",
                synthetic_frames,
                synthetic_temporal_corr,
                dataset_mix,
                content_seeded,
            )

        pbar = tqdm(
            total=cache_size,
            desc="prep: motion anchors",
            leave=False,
            disable=not accelerator.is_local_main_process,
        )
        transformer.eval()
        try:
            with torch.no_grad():
                for attempt, batch in enumerate(train_dataloader, start=1):
                    if len(entries) >= cache_size or attempt > max_attempts:
                        break
                    latents = batch.get("latents")
                    if not isinstance(latents, torch.Tensor) or latents.dim() != 5:
                        skipped_bad_latents += 1
                        continue

                    latents = latents.to(device=accelerator.device, dtype=network_dtype)
                    use_synthetic = anchor_source == "synthetic" or (
                        anchor_source == "hybrid" and random.random() >= dataset_mix
                    )
                    if use_synthetic:
                        anchor_latents = _build_synthetic_motion_latents(
                            latents,
                            target_frames=synthetic_frames,
                            temporal_corr=synthetic_temporal_corr,
                            content_seeded=content_seeded,
                        )
                        anchor_source_name = "synthetic"
                    else:
                        anchor_latents = latents.clone()
                        anchor_source_name = "dataset"
                        if anchor_latents.shape[2] <= 1:
                            single_frame_dataset_anchors += 1
                    anchor_noise = torch.randn_like(anchor_latents)
                    anchor_batch_cpu = _extract_anchor_batch(batch)
                    anchor_batch = _move_to_device(
                        anchor_batch_cpu,
                        device=accelerator.device,
                        dtype=network_dtype,
                    )

                    variants: List[MotionAnchorVariant] = []
                    sigma_iter = self.replay_sigmas if self.replay_sigmas else [None]
                    for sigma in sigma_iter:
                        if sigma is None:
                            noisy_input, model_timesteps, sampled_sigmas = (
                                get_noisy_model_input_and_timesteps(
                                    self.args,
                                    anchor_noise,
                                    anchor_latents,
                                    noise_scheduler,
                                    accelerator.device,
                                    network_dtype,
                                    timestep_distribution=timestep_distribution,
                                    item_info=anchor_batch.get("item_info"),
                                )
                            )
                            resolved_sigma = (
                                float(sampled_sigmas.float().mean().item())
                                if sampled_sigmas is not None
                                else _infer_sigma_from_timesteps(
                                    model_timesteps, noise_scheduler
                                )
                            )
                        else:
                            noisy_input, model_timesteps = _build_noisy_input_for_sigma(
                                anchor_latents,
                                anchor_noise,
                                sigma,
                                noise_scheduler,
                            )
                            resolved_sigma = float(sigma)

                        teacher_attention_maps = None
                        teacher_context = (
                            _temporary_network_multiplier(
                                accelerator,
                                network,
                                0.0,
                            )
                            if use_base_model_teacher
                            else nullcontext()
                        )
                        with teacher_context:
                            if (
                                bool(
                                    getattr(
                                        self.args, "motion_attention_preservation", False
                                    )
                                )
                                and self.attention_modules
                            ):
                                with AttentionMapRecorder(
                                    self.attention_modules,
                                    max_queries=int(
                                        getattr(
                                            self.args,
                                            "motion_attention_preservation_queries",
                                            32,
                                        )
                                        or 32
                                    ),
                                    max_keys=int(
                                        getattr(
                                            self.args,
                                            "motion_attention_preservation_keys",
                                            64,
                                        )
                                        or 64
                                    ),
                                    capture_grad=False,
                                    keep_heads=bool(
                                        getattr(
                                            self.args,
                                            "motion_attention_preservation_per_head",
                                            False,
                                        )
                                    ),
                                ) as recorder:
                                    model_result = self.training_core.call_dit(
                                        self.args,
                                        accelerator,
                                        transformer,
                                        anchor_latents,
                                        anchor_batch,
                                        anchor_noise,
                                        noisy_input,
                                        model_timesteps,
                                        network_dtype,
                                        model_timesteps_override=model_timesteps,
                                        apply_stable_velocity_target=False,
                                    )
                                    teacher_attention_maps = {
                                        name: _clone_to_cpu(attn_map)
                                        for name, attn_map in recorder.collect_maps().items()
                                    }
                            else:
                                model_result = self.training_core.call_dit(
                                    self.args,
                                    accelerator,
                                    transformer,
                                    anchor_latents,
                                    anchor_batch,
                                    anchor_noise,
                                    noisy_input,
                                    model_timesteps,
                                    network_dtype,
                                    model_timesteps_override=model_timesteps,
                                    apply_stable_velocity_target=False,
                                )
                        teacher_pred = _extract_model_pred(model_result)
                        variants.append(
                            MotionAnchorVariant(
                                sigma=resolved_sigma,
                                noisy_input=_clone_to_cpu(noisy_input),
                                model_timesteps=_clone_to_cpu(model_timesteps),
                                teacher_pred=_clone_to_cpu(teacher_pred),
                                teacher_attention_maps=teacher_attention_maps,
                            )
                        )

                    anchor_entry = MotionAnchor(
                        latents=_clone_to_cpu(anchor_latents),
                        noise=_clone_to_cpu(anchor_noise),
                        batch=anchor_batch_cpu,
                        variants=variants,
                        source=anchor_source_name,
                    )
                    if not all(
                        self._validate_variant(
                            anchor_latents=anchor_entry.latents,
                            variant=variant,
                        )
                        for variant in anchor_entry.variants
                    ):
                        continue
                    entries.append(anchor_entry)
                    pbar.update(1)
        finally:
            pbar.close()
            if was_training:
                transformer.train()

        if not entries:
            logger.warning(
                "Motion preservation requested, but no anchors were built (bad_latent_batches=%d).",
                skipped_bad_latents,
            )
        else:
            logger.info(
                "Built %d motion anchors (single_frame_dataset=%d bad_latent_batches=%d).",
                len(entries),
                single_frame_dataset_anchors,
                skipped_bad_latents,
            )
            if (
                getattr(self.args, "motion_preservation_mode", "temporal") == "temporal"
                and single_frame_dataset_anchors > 0
            ):
                logger.info(
                    "Temporal replay will fall back to full-output matching for %d single-frame dataset anchors.",
                    single_frame_dataset_anchors,
                )
            if cache_path:
                self._save_anchor_cache(cache_path, entries)
        self.health.note_cache(entries, loaded=False)
        self._publish_runtime_attributes()
        self.anchor_cache = entries
        return self.anchor_cache

    def should_apply_replay(self, global_step: int) -> bool:
        probability = getattr(self.args, "motion_preservation_probability", None)
        if probability is not None:
            return random.random() < float(probability)
        interval = int(getattr(self.args, "motion_preservation_interval", 1) or 1)
        return global_step % interval == 0

    def _current_weight(self, global_step: int) -> float:
        multiplier = float(
            getattr(self.args, "motion_preservation_multiplier", 0.0) or 0.0
        )
        warmup_steps = int(
            getattr(self.args, "motion_preservation_warmup_steps", 0) or 0
        )
        if warmup_steps <= 0:
            return multiplier
        warmup_ratio = min(1.0, max(0.0, float(global_step) / float(warmup_steps)))
        return multiplier * warmup_ratio

    def compute_loss(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network_dtype: torch.dtype,
        *,
        global_step: int,
        base_task_loss: Optional[torch.Tensor] = None,
        skip_schedule_check: bool = False,
    ) -> Optional[torch.Tensor]:
        self.last_loss = None
        self.last_weight = None
        self.last_sigma = None
        self.last_anchor_source = None
        self.last_temporal_fallback = None
        self.last_attention_loss = None
        self.last_anchor_frames = None
        self.last_total_to_task_ratio = None

        if not self.anchor_cache:
            self.record_skip("no_anchor")
            return None
        if not skip_schedule_check and not self.should_apply_replay(global_step):
            self.record_skip("schedule")
            return None

        weight = self._current_weight(global_step)
        if weight <= 0.0:
            self.record_skip("zero_weight")
            return None

        anchor = random.choice(self.anchor_cache)
        sigmas = [variant.sigma for variant in anchor.variants]
        variant = anchor.variants[_sample_sigma_index(sigmas, self.args)]
        if not self._validate_variant(anchor_latents=anchor.latents, variant=variant):
            return None

        anchor_latents = anchor.latents.to(device=accelerator.device, dtype=network_dtype)
        anchor_noise = anchor.noise.to(device=accelerator.device, dtype=network_dtype)
        anchor_noisy_input = variant.noisy_input.to(
            device=accelerator.device, dtype=network_dtype
        )
        anchor_model_timesteps = variant.model_timesteps.to(
            device=accelerator.device, dtype=network_dtype
        )
        teacher_pred = variant.teacher_pred
        anchor_batch = _move_to_device(
            anchor.batch,
            device=accelerator.device,
            dtype=network_dtype,
        )

        student_attention_maps: Dict[str, torch.Tensor] = {}
        if (
            bool(getattr(self.args, "motion_attention_preservation", False))
            and self.attention_modules
        ):
            with AttentionMapRecorder(
                self.attention_modules,
                max_queries=int(
                    getattr(self.args, "motion_attention_preservation_queries", 32)
                    or 32
                ),
                max_keys=int(
                    getattr(self.args, "motion_attention_preservation_keys", 64) or 64
                ),
                capture_grad=True,
                keep_heads=bool(
                    getattr(self.args, "motion_attention_preservation_per_head", False)
                ),
            ) as recorder:
                with accelerator.autocast():
                    model_result = self.training_core.call_dit(
                        self.args,
                        accelerator,
                        transformer,
                        anchor_latents,
                        anchor_batch,
                        anchor_noise,
                        anchor_noisy_input,
                        anchor_model_timesteps,
                        network_dtype,
                        global_step=global_step,
                        model_timesteps_override=anchor_model_timesteps,
                        apply_stable_velocity_target=False,
                    )
                student_attention_maps = recorder.collect_maps()
        else:
            with accelerator.autocast():
                model_result = self.training_core.call_dit(
                    self.args,
                    accelerator,
                    transformer,
                    anchor_latents,
                    anchor_batch,
                    anchor_noise,
                    anchor_noisy_input,
                    anchor_model_timesteps,
                    network_dtype,
                    global_step=global_step,
                    model_timesteps_override=anchor_model_timesteps,
                    apply_stable_velocity_target=False,
                )
        student_pred = _extract_model_pred(model_result)
        if tuple(student_pred.shape) != tuple(anchor_latents.shape):
            if not self._handle_validation_issue(
                f"Motion replay student prediction shape mismatch: expected {tuple(anchor_latents.shape)} got {tuple(student_pred.shape)}.",
                reason="invalid_anchor",
            ):
                return None
        replay_loss_raw, temporal_fallback = _compute_motion_replay_loss(
            student_pred,
            teacher_pred,
            mode=str(
                getattr(self.args, "motion_preservation_mode", "temporal")
                or "temporal"
            ).lower(),
            second_order_weight=float(
                getattr(self.args, "motion_preservation_second_order_weight", 0.0) or 0.0
            ),
            chunk_frames=int(
                getattr(self.args, "motion_preservation_teacher_chunk_frames", 0) or 0
            ),
            dtype=network_dtype,
        )
        motion_total_loss = replay_loss_raw * weight

        attention_loss_raw = compute_motion_attention_loss(
            student_attention_maps,
            variant.teacher_attention_maps,
            loss_type=str(
                getattr(self.args, "motion_attention_preservation_loss", "kl") or "kl"
            ).lower(),
            temperature=float(
                getattr(
                    self.args,
                    "motion_attention_preservation_temperature",
                    1.0,
                )
                or 1.0
            ),
            symmetric_kl=bool(
                getattr(
                    self.args, "motion_attention_preservation_symmetric_kl", False
                )
            ),
        )
        if (
            bool(getattr(self.args, "motion_attention_preservation", False))
            and variant.teacher_attention_maps
            and attention_loss_raw is None
        ):
            teacher_keys = set(variant.teacher_attention_maps.keys())
            student_keys = set(student_attention_maps.keys())
            if teacher_keys and student_keys and teacher_keys.isdisjoint(student_keys):
                if not self._handle_validation_issue(
                    "Motion attention preservation found no overlapping teacher/student attention modules during replay.",
                    reason="invalid_anchor",
                ):
                    return None
        if attention_loss_raw is not None:
            attention_loss = attention_loss_raw * float(
                getattr(self.args, "motion_attention_preservation_weight", 0.0) or 0.0
            )
            motion_total_loss = motion_total_loss + attention_loss
            self.last_attention_loss = attention_loss.detach()

        loss_detached = motion_total_loss.detach()
        self.last_loss = loss_detached
        self.last_weight = loss_detached.new_tensor(weight)
        self.last_sigma = loss_detached.new_tensor(float(variant.sigma))
        self.last_anchor_source = loss_detached.new_tensor(
            1.0 if anchor.source == "synthetic" else 0.0
        )
        self.last_temporal_fallback = loss_detached.new_tensor(
            1.0 if temporal_fallback else 0.0
        )
        self.last_anchor_frames = loss_detached.new_tensor(float(anchor.latents.shape[2]))
        if isinstance(base_task_loss, torch.Tensor):
            base_task_abs = base_task_loss.detach().to(torch.float32).abs().clamp_min(1e-8)
            self.last_total_to_task_ratio = (
                motion_total_loss.detach().to(torch.float32).abs() / base_task_abs
            ).to(loss_detached.device, dtype=loss_detached.dtype)
        self.health.note_applied(
            attention_applied=self.last_attention_loss is not None,
            temporal_fallback=bool(temporal_fallback),
        )
        self._publish_runtime_attributes()
        self._maybe_log_health_summary()
        return motion_total_loss

    def maybe_compute_loss(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network_dtype: torch.dtype,
        *,
        global_step: int,
        base_task_loss: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        return self.compute_loss(
            accelerator,
            transformer,
            network_dtype,
            global_step=global_step,
            base_task_loss=base_task_loss,
            skip_schedule_check=False,
        )
