"""Cache optical flow tensors for EquiVDM consistent noise training."""

from __future__ import annotations

import argparse
import gc
import logging
import os
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from common.logger import get_logger
from dataset.cache import save_optical_flow_cache_wan
from dataset.item_info import ItemInfo
from dataset.base_dataset import BaseDataset

logger = get_logger(__name__, level=logging.INFO)


def _to_raft_range(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0


def _load_raft_model(model_name: str, device: torch.device) -> torch.nn.Module:
    import torchvision
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
    )

    if model_name == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        model = torchvision.models.optical_flow.raft_small(weights=weights)
    elif model_name == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = torchvision.models.optical_flow.raft_large(weights=weights)
    else:
        raise ValueError(f"Unknown optical flow model: {model_name}")

    return model.to(device=device, dtype=torch.float32).eval()


def _prepare_video_tensor(item: ItemInfo) -> Optional[torch.Tensor]:
    if item.content is None:
        return None
    arr = item.content
    if not isinstance(arr, np.ndarray):
        return None
    if arr.ndim == 3:
        return None
    if arr.ndim != 4:
        raise ValueError(f"Unsupported content shape for {item.item_key}: {arr.shape}")
    tensor = torch.from_numpy(arr).to(torch.float32)  # F, H, W, C
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # F, C, H, W
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


def _compute_flow_batch(
    raft_model: torch.nn.Module,
    video_batch: torch.Tensor,
) -> torch.Tensor:
    # video_batch: B, F, C, H, W in [0,1]
    flows = []
    for t in range(video_batch.shape[1] - 1):
        f1 = _to_raft_range(video_batch[:, t])
        f2 = _to_raft_range(video_batch[:, t + 1])
        flow = raft_model(f1, f2)[-1]
        flows.append(flow)
    return torch.stack(flows, dim=1)  # B, F-1, 2, H, W


def compute_and_save_batch(
    raft_model: torch.nn.Module,
    batch: list[ItemInfo],
    args: argparse.Namespace,
    device: torch.device,
    frame_stride: Optional[int],
) -> None:
    ready_items: list[ItemInfo] = []
    tensors: list[torch.Tensor] = []
    for item in batch:
        flow_path = getattr(item, "optical_flow_cache_path", None)
        if flow_path is None and item.latent_cache_path is not None:
            flow_path = item.latent_cache_path.replace(".safetensors", "_flow.safetensors")
            item.optical_flow_cache_path = flow_path
        if flow_path is None:
            continue
        if args.skip_existing and os.path.exists(flow_path):
            continue
        tensor = _prepare_video_tensor(item)
        if tensor is None or tensor.shape[0] < 2:
            continue
        ready_items.append(item)
        tensors.append(tensor)

    if not ready_items:
        return

    video_batch = torch.stack(tensors, dim=0)  # B, F, C, H, W
    video_batch = video_batch.to(device=device, dtype=torch.float32)
    video_batch = video_batch / 255.0

    with torch.no_grad():
        flows = _compute_flow_batch(raft_model, video_batch)

    for item, flow in zip(ready_items, flows):
        save_optical_flow_cache_wan(
            item,
            flow.to(torch.float32),
            frame_stride=frame_stride,
        )


def cache_optical_flow(datasets: list[BaseDataset], args: argparse.Namespace) -> None:
    device = (
        torch.device(args.device)
        if isinstance(args.device, str)
        else args.device
    )
    raft_model = _load_raft_model(args.model, device)

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else max(1, os.cpu_count() - 1)  # type: ignore
    )

    if getattr(args, "purge_before_run", False):
        total_purged = 0
        for dataset in datasets:
            try:
                for cache_file in dataset.get_all_optical_flow_cache_files():
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        total_purged += 1
            except Exception as exc:
                logger.warning("Failed to purge flow cache files: %s", exc)
        logger.info("🧹 Purged %d optical flow cache files before caching", total_purged)

    for i, dataset in enumerate(datasets):
        logger.info("Caching optical flow for dataset [%d]", i)
        all_flow_cache_paths = []
        frame_stride = getattr(dataset, "frame_stride", None)
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            for item in batch:
                tensor = _prepare_video_tensor(item)
                if tensor is None or tensor.shape[0] < 2:
                    continue
                flow_path = getattr(item, "optical_flow_cache_path", None)
                if flow_path is None and item.latent_cache_path is not None:
                    flow_path = item.latent_cache_path.replace(
                        ".safetensors", "_flow.safetensors"
                    )
                    item.optical_flow_cache_path = flow_path
                if flow_path is not None:
                    all_flow_cache_paths.append(flow_path)
            bs = args.batch_size if args.batch_size is not None else len(batch)
            for j in range(0, len(batch), bs):
                compute_and_save_batch(
                    raft_model,
                    batch[j : j + bs],
                    args,
                    device,
                    frame_stride,
                )

        all_flow_cache_paths = {os.path.normpath(p) for p in all_flow_cache_paths}
        for cache_file in dataset.get_all_optical_flow_cache_files():
            if os.path.normpath(cache_file) not in all_flow_cache_paths:
                if args.keep_cache:
                    logger.info("Keep cache file not in the dataset: %s", cache_file)
                else:
                    try:
                        os.remove(cache_file)
                        logger.info("Removed old cache file: %s", cache_file)
                    except Exception as exc:
                        logger.warning("Failed to remove old cache file: %s", exc)

    del raft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
