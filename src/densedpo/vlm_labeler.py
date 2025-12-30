"""VLM-based dense segment labeling for DenseDPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

logger = logging.getLogger(__name__)


@dataclass
class DenseDPOVLMConfig:
    model_path: str
    dtype: torch.dtype
    device: torch.device
    prompt: str
    max_new_tokens: int
    temperature: float
    cache_dir: Optional[str]
    max_frames: int


class DenseDPOVLMLabeler:
    """Compute per-segment preferences using a local VLM scorer."""

    def __init__(self, config: DenseDPOVLMConfig) -> None:
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.model.eval()

        self._cache: Dict[str, float] = {}
        self._cache_path = None
        if config.cache_dir:
            cache_dir = Path(config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / "densedpo_vlm_scores.jsonl"
            self._load_cache()

    def compute_preferences(
        self,
        *,
        policy_frames: List[torch.Tensor] | List[List[torch.Tensor]],
        reference_frames: List[torch.Tensor] | List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return preferences plus policy/reference scores per segment."""
        policy_batches = _normalize_frame_batches(policy_frames)
        reference_batches = _normalize_frame_batches(reference_frames)
        if len(policy_batches) != len(reference_batches):
            raise ValueError("VLM policy/reference batch size mismatch.")

        policy_scores = []
        reference_scores = []
        for policy_clips, reference_clips in zip(
            policy_batches, reference_batches
        ):
            if len(policy_clips) != len(reference_clips):
                raise ValueError("VLM segment count mismatch within batch.")
            policy_scores.append(
                [self._score_clip(clip) for clip in policy_clips]
            )
            reference_scores.append(
                [self._score_clip(clip) for clip in reference_clips]
            )
        policy_scores_t = torch.tensor(policy_scores, dtype=torch.float32)
        reference_scores_t = torch.tensor(reference_scores, dtype=torch.float32)
        preferences = (policy_scores_t > reference_scores_t).to(torch.float32)
        return preferences, policy_scores_t, reference_scores_t

    def _score_clip(self, frames: torch.Tensor) -> float:
        frames = self._limit_frames(frames, self.config.max_frames)
        clip_hash = self._hash_frames(frames)
        cached = self._cache.get(clip_hash)
        if cached is not None:
            return cached
        score = self._run_vlm(frames)
        self._cache[clip_hash] = score
        self._append_cache(clip_hash, score)
        return score

    def _run_vlm(self, frames: torch.Tensor) -> float:
        images = self._to_pil_list(frames)
        inputs = self.processor(
            images=images,
            text=self.config.prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )
        text = decoded[0] if decoded else ""
        return _extract_first_float(text)

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = payload.get("hash")
                    score = payload.get("score")
                    if key and isinstance(score, (int, float)):
                        self._cache[str(key)] = float(score)
        except Exception as exc:
            logger.warning("Failed to load DenseDPO VLM cache: %s", exc)

    def _append_cache(self, key: str, score: float) -> None:
        if self._cache_path is None:
            return
        try:
            with self._cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"hash": key, "score": score}) + "\n")
        except Exception as exc:
            logger.warning("Failed to write DenseDPO VLM cache: %s", exc)

    @staticmethod
    def _hash_frames(frames: torch.Tensor) -> str:
        frames = frames.detach().to(torch.uint8).contiguous()
        digest = hashlib.sha1(frames.cpu().numpy().tobytes()).hexdigest()
        return digest

    @staticmethod
    def _to_pil_list(frames: torch.Tensor) -> List[Image.Image]:
        frames = frames.detach()
        if frames.dtype != torch.uint8:
            frames = frames.float()
            frames = (frames + 1.0) * 127.5
            frames = frames.clamp(0, 255).to(torch.uint8)
        np_frames = frames.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(frame) for frame in np_frames]

    @staticmethod
    def _limit_frames(frames: torch.Tensor, max_frames: int) -> torch.Tensor:
        if max_frames <= 0 or frames.shape[0] <= max_frames:
            return frames
        indices = np.linspace(0, frames.shape[0] - 1, max_frames).astype(int)
        return frames[torch.tensor(indices, device=frames.device)]


def _extract_first_float(text: str) -> float:
    if not text:
        return 0.0
    tokens = text.replace(",", " ").split()
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def _normalize_frame_batches(
    frames: List[torch.Tensor] | List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    if not frames:
        return []
    first = frames[0]
    if isinstance(first, list):
        return [list(item) for item in frames]  # type: ignore[arg-type]
    return [list(frames)]  # type: ignore[list-item]
