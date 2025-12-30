from __future__ import annotations

import logging
import os
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class ErrorRecyclingState:
    used_clean_input: bool
    grid_index: int
    applied_error: bool
    applied_y_error: bool


class ErrorRecyclingHelper:
    """Train-time error recycling helper (LoRA-only, gated, default off)."""

    def __init__(self, args: Any, noise_scheduler: Any) -> None:
        self.enabled = bool(getattr(args, "enable_error_recycling", False))
        self.error_buffer_size = int(getattr(args, "error_buffer_k", 500))
        self.timestep_grid_size = int(getattr(args, "timestep_grid_size", 25))
        self.num_grids = int(getattr(args, "num_grids", 50))
        self.buffer_replacement_strategy = str(
            getattr(args, "buffer_replacement_strategy", "random")
        )
        self.buffer_warmup_iter = int(getattr(args, "buffer_warmup_iter", 50))
        self.error_modulate_factor = float(
            getattr(args, "error_modulate_factor", 0.0)
        )
        self.noise_prob = float(getattr(args, "noise_prob", 0.9))
        self.latent_prob = float(getattr(args, "latent_prob", 0.9))
        self.y_prob = float(getattr(args, "y_prob", 0.9))
        self.clean_prob = float(getattr(args, "clean_prob", 0.1))
        self.clean_buffer_update_prob = float(
            getattr(args, "clean_buffer_update_prob", 0.1)
        )
        self.y_error_num = int(getattr(args, "y_error_num", 1))
        self.y_error_sample_from_all_grids = bool(
            getattr(args, "y_error_sample_from_all_grids", False)
        )
        self.y_error_sample_range = self._parse_y_error_sample_range(
            getattr(args, "y_error_sample_range", None)
        )
        self.use_last_y_error = bool(getattr(args, "use_last_y_error", False))
        self.error_setting = int(getattr(args, "error_setting", 1))
        self.require_scheduler_errors = bool(
            getattr(args, "error_recycling_require_scheduler_errors", False)
        )
        self.enable_svi_y_builder = bool(getattr(args, "enable_svi_y_builder", False))
        self.svi_y_motion_source = str(
            getattr(args, "svi_y_motion_source", "latent_last")
        )
        self.svi_y_num_motion_latent = int(
            getattr(args, "svi_y_num_motion_latent", 1)
        )
        self.svi_y_first_clip_mode = str(
            getattr(args, "svi_y_first_clip_mode", "zeros")
        )
        self.svi_y_replay_buffer_size = int(
            getattr(args, "svi_y_replay_buffer_size", 256)
        )
        self.svi_y_replay_key_mode = str(
            getattr(args, "svi_y_replay_key_mode", "item_key")
        )
        self.svi_y_replay_use_sequence_index = bool(
            getattr(args, "svi_y_replay_use_sequence_index", False)
        )
        self.svi_y_replay_sequence_pattern = str(
            getattr(args, "svi_y_replay_sequence_pattern", r"_(\d+)-(\d+)$")
        )

        self.iteration_count = 0
        self._warned_missing_y = False
        self._warned_empty_buffer = False
        self._warned_missing_anchor = False
        self._warned_svi_shape = False
        self._warned_missing_item_info = False
        self._warned_motion_shape = False

        num_train_timesteps = getattr(
            getattr(noise_scheduler, "config", object()),
            "num_train_timesteps",
            1000,
        )
        self.num_train_timesteps = int(num_train_timesteps)

        if self.timestep_grid_size <= 0:
            self.timestep_grid_size = 25
        if self.num_grids <= 0:
            self.num_grids = max(1, self.num_train_timesteps // self.timestep_grid_size)

        self.error_buffer: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_grids)
        }
        self.y_error_buffer: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_grids)
        }
        self._svi_motion_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._svi_key_strip_re = re.compile(r"_(\d+)-(\d+)$")
        self._svi_sequence_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._svi_sequence_re = self._compile_sequence_pattern(
            self.svi_y_replay_sequence_pattern
        )

    def _get_svi_replay_key(self, item_info: Any) -> Optional[str]:
        if item_info is None:
            return None
        item_key = getattr(item_info, "item_key", None)
        if not isinstance(item_key, str):
            return None
        if self.svi_y_replay_key_mode == "base_key":
            return self._svi_key_strip_re.sub("", item_key)
        return item_key

    def _parse_sequence_key_index(
        self, item_key: str
    ) -> Optional[Tuple[str, int]]:
        stem = os.path.splitext(os.path.basename(item_key))[0]
        match = self._svi_sequence_re.search(stem)
        if not match:
            return None
        try:
            index = int(match.group(1))
        except ValueError:
            return None
        seq_key = stem[: match.start()]
        return seq_key, index

    def _get_item_key(self, item_info: Any) -> Optional[str]:
        if item_info is None:
            return None
        item_key = getattr(item_info, "item_key", None)
        if not isinstance(item_key, str) or not item_key:
            return None
        return item_key

    def _get_first_clip_motion(
        self,
        latents: torch.Tensor,
        batch_index: int,
        num_motion: int,
    ) -> torch.Tensor:
        if self.svi_y_first_clip_mode == "current":
            return latents[batch_index, :, -num_motion:, ...]
        return torch.zeros(
            (
                latents.shape[1],
                num_motion,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

    def _build_replay_motion_latents(
        self,
        latents: torch.Tensor,
        item_infos: Optional[Iterable[Any]],
        num_motion: int,
    ) -> Optional[torch.Tensor]:
        if not item_infos:
            if not self._warned_missing_item_info:
                logger.warning(
                    "SVI y builder replay requested but batch has no item_info."
                )
                self._warned_missing_item_info = True
            return None
        items = list(item_infos)
        if len(items) != latents.shape[0]:
            if not self._warned_missing_item_info:
                logger.warning(
                    "SVI y builder replay skipped: item_info size %d does not match batch %d.",
                    len(items),
                    latents.shape[0],
                )
                self._warned_missing_item_info = True
            return None
        motion_list: List[torch.Tensor] = []
        for idx, item_info in enumerate(items):
            item_key = self._get_item_key(item_info)
            key = self._get_svi_replay_key(item_info)
            cached = None
            if self.svi_y_replay_use_sequence_index and key:
                parsed = self._parse_sequence_key_index(item_key or "")
                if parsed is None:
                    if not self._warned_motion_shape:
                        logger.warning(
                            "SVI y builder replay skipped: item key '%s' does not match sequence pattern.",
                            key,
                        )
                        self._warned_motion_shape = True
                else:
                    seq_key, index = parsed
                    prev_key = f"{seq_key}:{index - 1}"
                    cached = self._svi_sequence_cache.get(prev_key)
            else:
                cached = self._svi_motion_cache.get(key) if key else None
            if cached is None:
                motion_list.append(
                    self._get_first_clip_motion(latents, idx, num_motion)
                )
                continue
            cached = cached.to(device=latents.device, dtype=latents.dtype)
            if cached.dim() == 4:
                cached = cached.unsqueeze(0)
            if cached.shape[-2:] != latents.shape[-2:] or cached.shape[2] != num_motion:
                if not self._warned_motion_shape:
                    logger.warning(
                        "SVI y builder replay skipped: cached motion shape %s does not match expected (C=%d,F=%d,H=%d,W=%d).",
                        tuple(cached.shape),
                        latents.shape[1],
                        num_motion,
                        latents.shape[3],
                        latents.shape[4],
                    )
                    self._warned_motion_shape = True
                motion_list.append(
                    self._get_first_clip_motion(latents, idx, num_motion)
                )
                continue
            motion_list.append(cached[0])
        if not motion_list:
            return None
        return torch.stack(motion_list, dim=0)

    def maybe_build_svi_y(
        self, batch: Dict[str, Any], latents: torch.Tensor
    ) -> None:
        if not self.enabled or not self.enable_svi_y_builder:
            return
        if not isinstance(batch, dict):
            return

        image_emb = batch.get("image_emb")
        if isinstance(image_emb, dict) and torch.is_tensor(image_emb.get("y")):
            return

        anchor_latent = batch.get("svi_y_anchor")
        if not torch.is_tensor(anchor_latent):
            if not self._warned_missing_anchor:
                logger.warning(
                    "SVI y builder enabled but no svi_y_anchor found in batch."
                )
                self._warned_missing_anchor = True
            return

        anchor_latent = anchor_latent.to(
            device=latents.device, dtype=latents.dtype
        )
        if anchor_latent.dim() == 4:
            anchor_latent = anchor_latent.unsqueeze(0)
        if latents.dim() != 5 or anchor_latent.dim() != 5:
            if not self._warned_svi_shape:
                logger.warning(
                    "SVI y builder skipped: latents shape %s, anchor shape %s.",
                    tuple(latents.shape),
                    tuple(anchor_latent.shape),
                )
                self._warned_svi_shape = True
            return

        batch_size = latents.shape[0]
        if anchor_latent.shape[0] != batch_size:
            if anchor_latent.shape[0] == 1:
                anchor_latent = anchor_latent.expand(batch_size, -1, -1, -1, -1)
            else:
                if not self._warned_svi_shape:
                    logger.warning(
                        "SVI y builder skipped: anchor batch %d does not match latents %d.",
                        anchor_latent.shape[0],
                        batch_size,
                    )
                    self._warned_svi_shape = True
                return

        if anchor_latent.shape[1] != latents.shape[1]:
            if not self._warned_svi_shape:
                logger.warning(
                    "SVI y builder skipped: anchor channels %d do not match latents %d.",
                    anchor_latent.shape[1],
                    latents.shape[1],
                )
                self._warned_svi_shape = True
            return

        if anchor_latent.shape[2] != 1:
            anchor_latent = anchor_latent[:, :, :1]

        latent_frames = int(latents.shape[2])
        num_frames = (latent_frames - 1) * 4 + 1
        num_motion = int(max(0, min(self.svi_y_num_motion_latent, latent_frames - 1)))

        if self.svi_y_motion_source == "replay_buffer" and num_motion > 0:
            item_infos = None
            try:
                item_infos = batch.get("item_info")
            except Exception:
                item_infos = None
            motion_latent = self._build_replay_motion_latents(
                latents=latents,
                item_infos=item_infos,
                num_motion=num_motion,
            )
            if motion_latent is None:
                motion_latent = torch.zeros(
                    (
                        batch_size,
                        latents.shape[1],
                        num_motion,
                        latents.shape[3],
                        latents.shape[4],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
        elif self.svi_y_motion_source == "zeros" or num_motion == 0:
            motion_latent = torch.zeros(
                (
                    batch_size,
                    latents.shape[1],
                    num_motion,
                    latents.shape[3],
                    latents.shape[4],
                ),
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            motion_latent = latents[:, :, -num_motion:, ...]

        padding_frames = max(0, latent_frames - 1 - num_motion)
        padding = torch.zeros(
            (
                batch_size,
                latents.shape[1],
                padding_frames,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )
        y_latent = torch.cat([anchor_latent, motion_latent, padding], dim=2)

        msk = torch.ones(
            (1, num_frames, latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )
        msk[:, 1:] = 0
        msk = torch.cat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, latents.shape[3], latents.shape[4])
        msk = msk.transpose(1, 2)
        msk = msk.expand(batch_size, -1, -1, -1, -1)

        y = torch.cat([msk, y_latent], dim=1)
        if not isinstance(image_emb, dict):
            image_emb = {}
        image_emb["y"] = y
        batch["image_emb"] = image_emb

    def update_svi_motion_cache(
        self, batch: Dict[str, Any], latents: torch.Tensor
    ) -> None:
        if not self.enabled or not self.enable_svi_y_builder:
            return
        if self.svi_y_motion_source != "replay_buffer":
            return
        if self.svi_y_replay_buffer_size <= 0:
            return
        if latents.dim() != 5:
            return

        item_infos = None
        try:
            item_infos = batch.get("item_info")
        except Exception:
            item_infos = None
        if not item_infos:
            return

        num_motion = int(self.svi_y_num_motion_latent)
        if num_motion <= 0:
            return
        max_motion = min(num_motion, max(1, latents.shape[2]))

        for idx, item_info in enumerate(item_infos):
            item_key = self._get_item_key(item_info)
            key = self._get_svi_replay_key(item_info)
            if not key:
                continue
            motion = latents[idx, :, -max_motion:, ...].detach().cpu()
            if max_motion < num_motion:
                pad = torch.zeros(
                    (
                        motion.shape[0],
                        num_motion - max_motion,
                        motion.shape[2],
                        motion.shape[3],
                    ),
                    dtype=motion.dtype,
                )
                motion = torch.cat([motion, pad], dim=1)
            if self.svi_y_replay_use_sequence_index:
                parsed = self._parse_sequence_key_index(item_key or "")
                if parsed is None:
                    continue
                seq_key, index = parsed
                cache_key = f"{seq_key}:{index}"
                if cache_key in self._svi_sequence_cache:
                    self._svi_sequence_cache.move_to_end(cache_key)
                self._svi_sequence_cache[cache_key] = motion
                while (
                    len(self._svi_sequence_cache) > self.svi_y_replay_buffer_size
                ):
                    self._svi_sequence_cache.popitem(last=False)
            else:
                if key in self._svi_motion_cache:
                    self._svi_motion_cache.move_to_end(key)
                self._svi_motion_cache[key] = motion
                while len(self._svi_motion_cache) > self.svi_y_replay_buffer_size:
                    self._svi_motion_cache.popitem(last=False)

    def _compile_sequence_pattern(self, pattern: str) -> "re.Pattern[str]":
        try:
            return re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"Invalid svi_y_replay_sequence_pattern '{pattern}': {exc}"
            ) from exc

    def _parse_y_error_sample_range(
        self, value: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        if not value:
            return None
        if not isinstance(value, str):
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            return None
        try:
            start = int(parts[0])
            end = int(parts[1])
        except Exception:
            return None
        if start < 0 or end < start:
            return None
        return start, end

    def _get_grid_index(self, timesteps: torch.Tensor) -> int:
        if torch.is_tensor(timesteps):
            t = timesteps.detach().float()
            if t.numel() > 1:
                t = t.mean()
            t_val = float(t.item())
        else:
            t_val = float(timesteps)

        if t_val <= 1.0:
            t_val *= float(self.num_train_timesteps)
        grid_idx = int(t_val // float(self.timestep_grid_size))
        return max(0, min(grid_idx, self.num_grids - 1))

    def _has_y_buffer_data(self, grid_idx: int) -> bool:
        if self.y_error_sample_range is not None:
            start, end = self.y_error_sample_range
            return any(
                len(self.y_error_buffer[idx]) > 0
                for idx in range(start, min(end + 1, self.num_grids))
            )
        if self.y_error_sample_from_all_grids:
            return any(len(buf) > 0 for buf in self.y_error_buffer.values())
        return len(self.y_error_buffer[grid_idx]) > 0

    def _sample_error(self, grid_idx: int) -> Optional[torch.Tensor]:
        if not self.error_buffer[grid_idx]:
            return None
        return random.choice(self.error_buffer[grid_idx])

    def _sample_y_error(self, grid_idx: int) -> Optional[torch.Tensor]:
        if self.y_error_sample_range is not None:
            start, end = self.y_error_sample_range
            candidates = []
            for idx in range(start, min(end + 1, self.num_grids)):
                candidates.extend(self.y_error_buffer[idx])
            if not candidates:
                return None
            return random.choice(candidates)
        if self.y_error_sample_from_all_grids:
            candidates = []
            for buf in self.y_error_buffer.values():
                candidates.extend(buf)
            if not candidates:
                return None
            return random.choice(candidates)
        if not self.y_error_buffer[grid_idx]:
            return None
        return random.choice(self.y_error_buffer[grid_idx])

    def _modulate_error(self, error_sample: torch.Tensor) -> torch.Tensor:
        if self.error_modulate_factor <= 0.0:
            return error_sample
        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        return error_sample * random.uniform(min_mod, max_mod)

    def _resolve_y_target(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        image_emb = batch.get("image_emb") if isinstance(batch, dict) else None
        if isinstance(image_emb, dict):
            y_val = image_emb.get("y")
            if torch.is_tensor(y_val):
                return y_val
        return None

    def inject_errors(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, ErrorRecyclingState]:
        self.iteration_count += 1
        grid_idx = self._get_grid_index(timesteps)

        has_error_buffer = len(self.error_buffer[grid_idx]) > 0
        has_y_buffer = self._has_y_buffer_data(grid_idx)

        allow_noise = True
        allow_latent = True
        allow_y = True
        if self.error_setting == 0:
            allow_latent = False
            allow_y = False
        elif self.error_setting == 2:
            allow_y = False
        elif self.error_setting == 3:
            allow_noise = False
            allow_latent = False
        elif self.error_setting == 4:
            allow_noise = False
            allow_y = False

        add_error_noise = allow_noise and random.random() < self.noise_prob
        add_error_latent = allow_latent and random.random() < self.latent_prob
        add_error_y = allow_y and random.random() < self.y_prob

        use_clean_input = False
        if random.random() < self.clean_prob:
            add_error_noise = False
            add_error_latent = False
            add_error_y = False
            use_clean_input = True

        applied_error = False
        if (add_error_noise or add_error_latent) and has_error_buffer:
            error_sample = self._sample_error(grid_idx)
            if error_sample is not None:
                error_sample = self._modulate_error(error_sample)
                if error_sample.shape == noise.shape:
                    error_sample = error_sample.to(device=noise.device, dtype=noise.dtype)
                    if add_error_noise:
                        noise = noise + error_sample
                    if add_error_latent:
                        latents = latents + error_sample.to(
                            device=latents.device, dtype=latents.dtype
                        )
                    applied_error = True
                else:
                    logger.debug(
                        "Error recycling skipped: buffer shape %s mismatched with noise %s.",
                        tuple(error_sample.shape),
                        tuple(noise.shape),
                    )

        applied_y_error = False
        if add_error_y and has_y_buffer:
            y_target = self._resolve_y_target(batch)
            if y_target is None:
                if not self._warned_missing_y:
                    logger.warning(
                        "Error recycling enabled y_error but no image_emb['y'] found in batch."
                    )
                    self._warned_missing_y = True
            else:
                y_error_sample = self._sample_y_error(grid_idx)
                if y_error_sample is not None:
                    y_error_sample = self._modulate_error(y_error_sample).to(
                        device=y_target.device, dtype=y_target.dtype
                    )
                    if y_error_sample.shape == y_target.shape:
                        max_start_idx = max(0, y_error_sample.shape[2] - self.y_error_num)
                        if self.use_last_y_error:
                            frame_idx = max_start_idx
                        else:
                            frame_idx = random.randint(0, max_start_idx)
                        end_idx = frame_idx + self.y_error_num
                        y_target[:, :, frame_idx:end_idx, ...] = (
                            y_target[:, :, frame_idx:end_idx, ...]
                            + y_error_sample[:, :, frame_idx:end_idx, ...]
                        )
                        applied_y_error = True
                    else:
                        logger.debug(
                            "Error recycling skipped: y_error shape %s mismatched with y target %s.",
                            tuple(y_error_sample.shape),
                            tuple(y_target.shape),
                        )

        if not has_error_buffer and not self._warned_empty_buffer:
            logger.info(
                "Error recycling enabled but buffers are empty; waiting for warmup updates."
            )
            self._warned_empty_buffer = True

        state = ErrorRecyclingState(
            used_clean_input=use_clean_input,
            grid_index=grid_idx,
            applied_error=applied_error,
            applied_y_error=applied_y_error,
        )
        return noise, latents, state

    def apply_to_inputs(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: Optional[torch.Tensor],
        batch: Dict[str, Any],
        noise_scheduler: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[ErrorRecyclingState]]:
        """Apply error recycling and rebuild noisy_model_input when possible."""
        if not self.enabled:
            return noise, latents, None, None

        original_noise = noise
        noise, latents_for_noisy, state = self.inject_errors(
            noise=noise,
            latents=latents,
            timesteps=timesteps,
            batch=batch,
        )

        rebuilt = self._rebuild_noisy_model_input(
            latents_for_noisy=latents_for_noisy,
            noise_for_noisy=noise,
            timesteps_for_noisy=timesteps,
            sigmas_for_noisy=sigmas,
            noise_scheduler=noise_scheduler,
        )
        if rebuilt is None:
            noise = original_noise
            return noise, latents, None, None
        return noise, latents_for_noisy, rebuilt, state

    def compute_buffer_errors(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: Any,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pred = model_pred
        if pred.dim() == target.dim() + 1 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.shape != target.shape:
            logger.debug(
                "Error recycling skipped: model_pred shape %s vs target %s.",
                tuple(pred.shape),
                tuple(target.shape),
            )
            return None, None

        if hasattr(noise_scheduler, "step"):
            try:
                x0_pred = noise_scheduler.step(
                    pred, timesteps, noisy_model_input, to_final=True, self_corr=True
                )
                noise_corr_gt = noise_scheduler.step(
                    target, timesteps, noisy_model_input, to_final=True, self_corr=True
                )
                x1_pred = noise_scheduler.step(
                    pred, timesteps, noisy_model_input, to_final=True, self_corr=False
                )
                latent_corr_gt = noise_scheduler.step(
                    target,
                    timesteps,
                    noisy_model_input,
                    to_final=True,
                    self_corr=False,
                )
                return x0_pred - noise_corr_gt, x1_pred - latent_corr_gt
            except Exception:
                pass

        if self.require_scheduler_errors:
            return None, None

        return pred - target, None

    def _rebuild_noisy_model_input(
        self,
        latents_for_noisy: torch.Tensor,
        noise_for_noisy: torch.Tensor,
        timesteps_for_noisy: torch.Tensor,
        sigmas_for_noisy: Optional[torch.Tensor],
        noise_scheduler: Any,
    ) -> Optional[torch.Tensor]:
        if sigmas_for_noisy is not None:
            return sigmas_for_noisy * noise_for_noisy + (
                1.0 - sigmas_for_noisy
            ) * latents_for_noisy
        if timesteps_for_noisy.dim() > 1:
            return None
        try:
            max_ts = int(
                getattr(
                    getattr(noise_scheduler, "config", object()),
                    "num_train_timesteps",
                    1000,
                )
                or 1000
            )
        except Exception:
            max_ts = 1000
        t = (timesteps_for_noisy.float() - 1.0) / float(max(1, max_ts))
        if latents_for_noisy.ndim == 5:
            t = t.view(-1, 1, 1, 1, 1)
        else:
            t = t.view(-1, 1, 1, 1)
        return (1.0 - t) * latents_for_noisy + t * noise_for_noisy

    def update_buffers(
        self,
        error_tensor: torch.Tensor,
        timesteps: torch.Tensor,
        accelerator: Optional[Any],
        used_clean_input: bool,
        y_error_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        if self.error_buffer_size <= 0:
            return
        if used_clean_input and random.random() > self.clean_buffer_update_prob:
            return

        error_tensor = error_tensor.detach()
        timesteps = timesteps.detach()
        if y_error_tensor is not None:
            y_error_tensor = y_error_tensor.detach()

        gathered_errors = error_tensor
        gathered_timesteps = timesteps
        gathered_y_errors = y_error_tensor

        use_gather = (
            accelerator is not None
            and getattr(accelerator, "num_processes", 1) > 1
            and self.iteration_count <= self.buffer_warmup_iter
        )
        if use_gather:
            try:
                gathered_errors = accelerator.gather(error_tensor)
                gathered_timesteps = accelerator.gather(timesteps)
                if y_error_tensor is not None:
                    gathered_y_errors = accelerator.gather(y_error_tensor)
            except Exception as exc:
                logger.debug("Error recycling gather skipped: %s", exc)

        self._update_buffers_from_batch(
            gathered_errors, gathered_timesteps, gathered_y_errors
        )

    def _update_buffers_from_batch(
        self,
        error_tensor: torch.Tensor,
        timesteps: torch.Tensor,
        y_error_tensor: Optional[torch.Tensor],
    ) -> None:
        if error_tensor.dim() == 0:
            error_tensor = error_tensor.unsqueeze(0)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        batch_size = int(error_tensor.shape[0])
        for idx in range(batch_size):
            error_sample = error_tensor[idx]
            timestep_sample = timesteps[idx] if timesteps.numel() > 1 else timesteps
            grid_idx = self._get_grid_index(timestep_sample)
            self._add_error_to_buffer(self.error_buffer, grid_idx, error_sample)

            if y_error_tensor is not None:
                y_sample = y_error_tensor[idx]
                self._add_error_to_buffer(self.y_error_buffer, grid_idx, y_sample)

    def _add_error_to_buffer(
        self, buffer: Dict[int, List[torch.Tensor]], grid_idx: int, sample: torch.Tensor
    ) -> None:
        buffer_list = buffer[grid_idx]
        sample_cpu = sample.detach().cpu()
        if len(buffer_list) < self.error_buffer_size:
            buffer_list.append(sample_cpu)
            return

        strategy = self.buffer_replacement_strategy
        if strategy == "fifo":
            buffer_list.pop(0)
            buffer_list.append(sample_cpu)
            return
        if strategy == "random":
            replace_idx = random.randint(0, len(buffer_list) - 1)
            buffer_list[replace_idx] = sample_cpu
            return
        if strategy in ("l2_similarity", "l2_batch"):
            self._replace_most_similar(buffer_list, sample_cpu, batch_mode=strategy == "l2_batch")
            return

        # Fallback to random if strategy is unknown
        replace_idx = random.randint(0, len(buffer_list) - 1)
        buffer_list[replace_idx] = sample_cpu

    def _replace_most_similar(
        self, buffer_list: List[torch.Tensor], sample: torch.Tensor, batch_mode: bool
    ) -> None:
        sample_flat = sample.float().flatten()
        if batch_mode:
            try:
                stacked = torch.stack(
                    [buf.float().flatten() for buf in buffer_list], dim=0
                )
                distances = torch.norm(stacked - sample_flat.unsqueeze(0), dim=1)
                replace_idx = int(torch.argmin(distances).item())
            except Exception:
                replace_idx = random.randint(0, len(buffer_list) - 1)
        else:
            replace_idx = 0
            best_distance = None
            for i, buf in enumerate(buffer_list):
                dist = torch.norm(buf.float().flatten() - sample_flat)
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    replace_idx = i
        buffer_list[replace_idx] = sample.cpu()

    def get_stats(self, grid_idx: int) -> Dict[str, int]:
        total_error = sum(len(buf) for buf in self.error_buffer.values())
        total_y = sum(len(buf) for buf in self.y_error_buffer.values())
        return {
            "error_buffer_size": total_error,
            "y_error_buffer_size": total_y,
            "error_buffer_grid_size": len(self.error_buffer[grid_idx]),
            "y_error_buffer_grid_size": len(self.y_error_buffer[grid_idx]),
            "error_buffer_grid_index": grid_idx,
            "error_recycling_iteration": self.iteration_count,
        }
