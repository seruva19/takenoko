import argparse
import os
import re
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from common.logger import get_logger
from dataset.image_video_dataset import BaseDataset
from dataset.item_info import ItemInfo
from enhancements.structure_from_tracking.groundingdino_prompter import (
    GroundingDINOPrompter,
)
from enhancements.structure_from_tracking.sam2_teacher import SAM2Teacher

logger = get_logger(__name__)


def _build_clean_pixels_tensor(
    batch: list[ItemInfo],
    device: torch.device,
) -> tuple[torch.Tensor, list[ItemInfo]]:
    """Convert batch item content arrays into [B, C, F, H, W] float tensor in [-1, 1]."""
    valid_items = []
    tensors = []
    for item in batch:
        content = item.content
        if not isinstance(content, np.ndarray):
            continue
        arr = content
        if arr.ndim == 3:
            # image HWC -> 1HWC
            arr = arr[None, ...]
        if arr.ndim != 4:
            continue
        # FHWC -> CFHW
        tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2).contiguous()
        tensors.append(tensor)
        valid_items.append(item)

    if not tensors:
        raise ValueError("No valid pixel content found in batch for SFT teacher caching.")

    clean_pixels = torch.stack(tensors, dim=0).to(device=device)
    clean_pixels = clean_pixels / 127.5 - 1.0
    return clean_pixels, valid_items


def _build_mask_hints(
    valid_items: list[ItemInfo],
    clean_pixels: torch.Tensor,
) -> Optional[torch.Tensor]:
    masks = []
    has_any = False
    target_frames = int(clean_pixels.shape[2])
    target_h = int(clean_pixels.shape[3])
    target_w = int(clean_pixels.shape[4])
    for item in valid_items:
        raw = getattr(item, "mask_content", None)
        if raw is None:
            masks.append(torch.zeros(target_frames, target_h, target_w, dtype=torch.float32))
            continue
        mask = torch.as_tensor(raw, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.shape[-1] in (1, 3, 4):
            mask = mask.mean(dim=-1, keepdim=False).unsqueeze(0)
        elif mask.dim() == 4 and mask.shape[-1] in (1, 3, 4):
            mask = mask.mean(dim=-1)
        if mask.dim() != 3:
            masks.append(torch.zeros(target_frames, target_h, target_w, dtype=torch.float32))
            continue

        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(target_frames, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        mask = mask.clamp(0.0, 1.0)
        if mask.max().item() > 0:
            has_any = True
        masks.append(mask)

    if not has_any:
        return None
    return torch.stack(masks, dim=0).to(device=clean_pixels.device)


def _purge_teacher_cache(cache_dir: str) -> int:
    if not os.path.isdir(cache_dir):
        return 0
    removed = 0
    for filename in os.listdir(cache_dir):
        if filename.endswith("_sft_teacher.safetensors"):
            path = os.path.join(cache_dir, filename)
            try:
                os.remove(path)
                removed += 1
            except Exception as exc:
                logger.warning("Failed to remove SFT teacher cache file %s: %s", path, exc)
    return removed


_FRAME_START_RE = re.compile(r"_(\d{5})-(\d{3})(?:-\d+)?(?:\.[^.]+)?$")


def _extract_frame_start(item_key: str) -> Optional[int]:
    match = _FRAME_START_RE.search(str(item_key))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _filter_items_by_frame_start_stride(
    batch: list[ItemInfo],
    stride: int,
) -> list[ItemInfo]:
    if stride <= 1:
        return batch
    filtered = []
    for item in batch:
        frame_start = _extract_frame_start(item.item_key)
        if frame_start is None:
            # Keep when unknown format to avoid accidental data loss.
            filtered.append(item)
            continue
        if frame_start % stride == 0:
            filtered.append(item)
    return filtered


def cache_sft_teacher_features(
    datasets: list[BaseDataset],
    args: argparse.Namespace,
) -> None:
    """Precompute and persist SFT teacher features for all dataset items."""
    if str(getattr(args, "sft_teacher_cache_dir", "")).strip() == "":
        raise ValueError(
            "sft_teacher_cache_dir must be set for SFT teacher feature caching."
        )

    device = (
        args.device
        if args.device is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if isinstance(device, str):
        device = torch.device(device)

    teacher_mode = str(getattr(args, "sft_teacher_cache_mode", "off")).lower()
    if teacher_mode == "off":
        teacher_mode = "read_write" if bool(getattr(args, "sft_teacher_cache_skip_existing", True)) else "write"
    elif teacher_mode == "write" and bool(getattr(args, "sft_teacher_cache_skip_existing", True)):
        teacher_mode = "read_write"

    # Clone args so runtime cache mode used for precompute does not mutate global trainer args.
    teacher_args = argparse.Namespace(**vars(args))
    teacher_args.device = str(device)
    teacher_args.sft_teacher_cache_mode = teacher_mode
    teacher = SAM2Teacher(args=teacher_args, device=device)
    dino_prompter = GroundingDINOPrompter(args=teacher_args, device=device)

    cache_dir = str(getattr(args, "sft_teacher_cache_dir", ""))
    os.makedirs(cache_dir, exist_ok=True)
    if bool(getattr(args, "sft_teacher_cache_purge", False)):
        removed = _purge_teacher_cache(cache_dir)
        logger.info("Purged %d existing SFT teacher cache files.", removed)

    include_backward = bool(getattr(args, "sft_enable_backward_teacher", True))
    paper_strict_mode = bool(getattr(args, "sft_paper_strict_mode", False))
    num_workers = int(getattr(args, "sft_teacher_cache_num_workers", 4))
    batch_size = int(getattr(args, "sft_teacher_cache_batch_size", 1))
    frame_start_stride = int(getattr(args, "sft_teacher_cache_frame_start_stride", 1))
    logger.info(
        "Starting SFT teacher feature caching (mode=%s, backward=%s, workers=%d, batch_size=%d, frame_start_stride=%d).",
        teacher_mode,
        include_backward,
        num_workers,
        batch_size,
        frame_start_stride,
    )

    total_items = 0
    for dataset_idx, dataset in enumerate(datasets):
        logger.info("Caching SFT teacher features for dataset [%d]", dataset_idx)
        for _key, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            for start in range(0, len(batch), batch_size):
                batch_slice = batch[start : start + batch_size]
                batch_slice = _filter_items_by_frame_start_stride(
                    batch=batch_slice,
                    stride=frame_start_stride,
                )
                if not batch_slice:
                    continue
                try:
                    clean_pixels, valid_items = _build_clean_pixels_tensor(
                        batch=batch_slice,
                        device=device,
                    )
                except ValueError:
                    continue
                item_keys = [item.item_key for item in valid_items]
                mask_hints = None
                if bool(getattr(args, "sft_use_mask_prompting", False)):
                    mask_hints = _build_mask_hints(valid_items, clean_pixels)
                    if mask_hints is None and dino_prompter.enabled:
                        mask_hints = dino_prompter.generate_mask_hints(
                            clean_pixels=clean_pixels,
                            item_info=valid_items,
                        )
                        if mask_hints is not None:
                            mask_hints = mask_hints.to(
                                device=clean_pixels.device,
                                dtype=torch.float32,
                            )
                    if paper_strict_mode and mask_hints is None:
                        raise ValueError(
                            "Structure-From-Tracking strict mode requires usable mask hints during cache precompute."
                        )

                teacher.extract_bidirectional_features(
                    clean_pixels=clean_pixels,
                    include_backward=include_backward,
                    item_keys=item_keys,
                    mask_hints=mask_hints,
                )
                total_items += len(valid_items)

    logger.info("SFT teacher feature caching complete. Processed %d items.", total_items)
