## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/cache_text_encoder_outputs.py (Apache)

import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.cache import save_text_encoder_output_cache_wan
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import BaseDataset
from dataset.item_info import ItemInfo
from wan.configs import wan_t2v_14B
from wan.modules.t5 import T5EncoderModel

import accelerate
import logging
from common.logger import get_logger
from utils.model_utils import str_to_dtype
from common.model_downloader import download_model_if_needed

logger = get_logger(__name__, level=logging.INFO)


def prepare_cache_files_and_paths(datasets: list[BaseDataset]):
    """Prepare cache files and paths for text encoder output caching"""
    all_cache_files_for_dataset = []  # existing cache files
    all_cache_paths_for_dataset = []  # all cache paths in the dataset
    for dataset in datasets:
        all_cache_files = [
            os.path.normpath(file)
            for file in dataset.get_all_text_encoder_output_cache_files()
        ]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)

        all_cache_paths_for_dataset.append(set())
    return all_cache_files_for_dataset, all_cache_paths_for_dataset


def process_text_encoder_batches(
    num_workers: Optional[int],
    skip_existing: bool,
    batch_size: int,
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    encode: callable,  # type: ignore
):
    """Process text encoder batches across all datasets"""
    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)  # type: ignore
    # Optional purge step: remove all existing text-encoder cache files in each dataset cache dir
    purge_before_run = getattr(encode, "_purge_before_run", False)
    if purge_before_run:
        total_purged = 0
        for dataset in datasets:
            try:
                for cache_file in dataset.get_all_text_encoder_output_cache_files():
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        total_purged += 1
            except Exception as e:
                logger.warning(f"Failed to purge text encoder cache files: {e}")
        logger.info(f"🧹 Purged {total_purged} text encoder cache files before caching")
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for batch in tqdm(
            dataset.retrieve_text_encoder_output_cache_batches(num_workers)
        ):
            # update cache files (it's ok if we update it multiple times)
            all_cache_paths.update(
                [
                    os.path.normpath(item.text_encoder_output_cache_path)
                    for item in batch
                ]
            )

            # skip existing cache files
            if skip_existing:
                filtered_batch = [
                    item
                    for item in batch
                    if not os.path.normpath(item.text_encoder_output_cache_path)
                    in all_cache_files
                ]
                # print(f"Filtered {len(batch) - len(filtered_batch)} existing cache files")
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = batch_size if batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])


def post_process_cache_files(
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    keep_cache: bool,
):
    """Post-process cache files to remove old ones not in dataset"""
    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def encode_and_save_text_encoder_output_batch(
    text_encoder: T5EncoderModel,
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
    args: Optional[argparse.Namespace] = None,  # <-- ADD THIS ARGUMENT
):
    """Encode and save a batch of text prompts using WAN T5 encoder"""
    prompts = [item.caption for item in batch]

    dop_enabled = getattr(args, "diff_output_preservation", False)
    trigger_word = getattr(args, "diff_output_preservation_trigger_word", None)
    preservation_class = getattr(args, "diff_output_preservation_class", None)

    prompts_for_preservation = []
    has_dop_prompts = False

    if dop_enabled and trigger_word and preservation_class:
        for prompt in prompts:
            if trigger_word in prompt:
                prompts_for_preservation.append(
                    prompt.replace(trigger_word, preservation_class)
                )
                has_dop_prompts = True
            else:
                # If a prompt doesn't have the trigger, it doesn't need a preservation version.
                # We add a placeholder to keep the list lengths aligned.
                prompts_for_preservation.append(None)

    # Combine original and preservation prompts for a single encoding pass
    combined_prompts = list(prompts)
    prompts_to_encode = []
    if has_dop_prompts:
        # Only add non-None preservation prompts to the encoding batch
        prompts_to_encode = [p for p in prompts_for_preservation if p is not None]
        combined_prompts.extend(prompts_to_encode)

    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                # Encode all prompts in one go
                context = text_encoder(combined_prompts, device)
        else:
            context = text_encoder(combined_prompts, device)

    # Split the results back
    training_context = context[: len(prompts)]
    preservation_context_encoded = context[len(prompts) :]

    # Re-align preservation embeddings with the original batch
    preservation_context_full = []
    encoded_idx = 0
    if has_dop_prompts:
        for prompt in prompts_for_preservation:
            if prompt is not None:
                preservation_context_full.append(
                    preservation_context_encoded[encoded_idx]
                )
                encoded_idx += 1
            else:
                preservation_context_full.append(None)

    # save prompt cache
    for i, item in enumerate(batch):
        ctx = training_context[i]

        # Get the corresponding preservation context
        preservation_ctx = preservation_context_full[i] if has_dop_prompts else None

        # Pass both embeddings to the save function
        save_text_encoder_output_cache_wan(item, ctx, preservation_ctx)
