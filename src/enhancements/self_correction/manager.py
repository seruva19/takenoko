from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator

from core.sampling_manager import SamplingManager
from dataset import config_utils

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SelfCorrectionManager:
    """Generates lightweight correction clips and refreshes a small on-disk cache.

    Non-invasive draft implementation:
    - Generates short videos via SamplingManager using prompts from either config or dataset captions
    - Writes captions as .txt and lets the existing caching pipeline handle latents/TE
    - Keeps only up to `self_correction_cache_size` clips (FIFO style)
    """

    def __init__(
        self,
        args: Any,
        accelerator: Accelerator,
        sampling_manager: SamplingManager,
        blueprint: config_utils.Blueprint,
        vae_dtype: torch.dtype,
    ) -> None:
        self.args = args
        self.accelerator = accelerator
        self.sampling_manager = sampling_manager
        self.blueprint = blueprint
        self.vae_dtype = vae_dtype

        # Config keys (all prefixed with self_correction_)
        self.enabled: bool = bool(getattr(args, "self_correction_enabled", False))
        self.cache_dir: str = os.path.join(args.output_dir, "self_correction_cache")
        self.cache_size: int = int(getattr(args, "self_correction_cache_size", 200))
        self.clip_len: int = int(getattr(args, "self_correction_clip_len", 32))
        self.batch_ratio: float = float(
            getattr(args, "self_correction_batch_ratio", 0.2)
        )
        self.sample_steps: int = int(getattr(args, "self_correction_sample_steps", 16))
        self.sample_width: int = int(getattr(args, "self_correction_width", 256))
        self.sample_height: int = int(getattr(args, "self_correction_height", 256))
        self.guidance_scale: float = float(
            getattr(args, "self_correction_guidance_scale", 5.0)
        )

        os.makedirs(self.cache_dir, exist_ok=True)

    def _sample_prompts_from_dataset(self, max_count: int) -> List[Dict[str, Any]]:
        # Use the train dataset captions inferred from the blueprint and config utils
        train_group = config_utils.generate_dataset_group_by_blueprint(
            self.blueprint.train_dataset_group, training=False
        )
        prompts: List[Dict[str, Any]] = []
        try:
            # Access per-dataset datasource captions via retrieve_text_encoder_output_cache_batches path
            # Fallback: random empty strings when captions are unavailable
            num_workers = 0
            for dataset in train_group.datasets:  # type: ignore[attr-defined]
                collected = 0
                try:
                    for _, batch in dataset.retrieve_text_encoder_output_cache_batches(
                        num_workers
                    ):
                        # Guard typing in case linter infers wrong type
                        try:
                            iterable_batch = list(batch)  # type: ignore
                        except Exception:
                            iterable_batch = []
                        for item in iterable_batch:
                            # batch items are ItemInfo
                            try:
                                text = getattr(item, "caption", "") or ""
                            except Exception:
                                text = ""
                            prompts.append({"text": text, "enum": len(prompts)})
                            collected += 1
                            if len(prompts) >= max_count:
                                return prompts
                        if collected >= max_count:
                            break
                except Exception:
                    continue
        except Exception:
            pass
        # Pad with empty prompts if dataset captions were not available
        while len(prompts) < max_count:
            prompts.append({"text": "", "enum": len(prompts)})
        return prompts

    def _build_sampling_parameter(
        self, prompt_text: str, enum_idx: int
    ) -> Dict[str, Any]:
        return {
            "prompt": prompt_text,
            "height": self.sample_height,
            "width": self.sample_width,
            "frame_count": self.clip_len,
            "sample_steps": self.sample_steps,
            "guidance_scale": self.guidance_scale,
            "enum": enum_idx,
        }

    def _gather_prompts(self, count: int) -> List[Dict[str, Any]]:
        # Prefer inline prompts from args if provided
        inline_prompts = getattr(self.args, "self_correction_prompts", None)
        if isinstance(inline_prompts, list) and len(inline_prompts) > 0:
            return inline_prompts[:count]

        return self._sample_prompts_from_dataset(count)

    @torch.no_grad()
    def update_cache(self, transformer: Any) -> None:
        """Regenerate a small batch of correction clips and prune old ones.

        - Uses SamplingManager.do_inference through sample_images path for consistency
        - Saves .mp4 via existing helpers in SamplingManager
        - Triggers latent/text-encoder caching over the new folder
        """
        if not self.enabled:
            return

        logger.info("Updating self-correction cache ...")

        # 1) Prepare prompts
        count = max(1, int(self.cache_size))
        prompts = self._gather_prompts(count)

        # 2) Convert prompts to sampling parameter dicts (with embeds pre-encoded by SamplingManager)
        # Reuse SamplingManager.process_sample_prompts to get T5 embeddings for prompts
        prompt_dicts = [
            {"text": p.get("text", ""), "enum": p.get("enum", i)}
            for i, p in enumerate(prompts)
        ]
        sample_params = self.sampling_manager.process_sample_prompts(
            self.args, self.accelerator, prompt_dicts
        )
        if not sample_params:
            logger.warning(
                "No sample parameters produced; skipping self-correction update"
            )
            return

        # 3) Ensure VAE can be loaded lazily by SamplingManager
        vae_config = {
            "args": self.args,
            "vae_dtype": self.vae_dtype,
            "vae_path": getattr(self.args, "vae", None),
        }
        self.sampling_manager.set_vae_config(vae_config)

        # 4) Generate videos into cache_dir (reusing sample_images logic but overriding save_dir)
        save_root = self.cache_dir
        os.makedirs(save_root, exist_ok=True)

        device_dtype = (
            torch.bfloat16
            if str(getattr(self.args, "dit_dtype", "bf16")) in ("bfloat16", "bf16")
            else torch.float16
        )

        # Temporarily override output_dir to redirect saves into cache_dir
        old_output_dir = self.args.output_dir
        try:
            self.args.output_dir = save_root
            # Perform generation
            # Provide VAE via SamplingManager lazy path by passing None
            self.sampling_manager.sample_images(
                self.accelerator,
                self.args,
                epoch=None,
                steps=0,
                vae=None,  # type: ignore[arg-type]
                transformer=transformer,
                sample_parameters=sample_params,
                dit_dtype=device_dtype,
            )
        finally:
            self.args.output_dir = old_output_dir

        # 5) Prune oldest files to maintain cache_size (count .mp4 only)
        try:
            videos = [
                os.path.join(save_root, f)
                for f in os.listdir(save_root)
                if f.lower().endswith(".mp4")
            ]
            if len(videos) > self.cache_size:
                videos.sort(key=lambda p: os.path.getmtime(p))
                for path in videos[: len(videos) - self.cache_size]:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception:
            pass

        logger.info("Self-correction cache update complete")
