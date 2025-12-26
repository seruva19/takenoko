import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from common.logger import get_logger
from dataset import config_utils
from dataset.cache import save_semantic_encoder_output_cache_wan
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.datasource_utils import load_video
from dataset.image_video_dataset import BaseDataset, VideoDataset
from dataset.item_info import ItemInfo
from enhancements.semanticgen.semantic_encoder import SemanticEncoder, sample_frames

logger = get_logger(__name__)


@dataclass
class SemanticCacheOptions:
    dataset_config: str
    device: str
    encoder_name: str
    encoder_type: str
    encoder_dtype: str
    encoder_resolution: int
    semantic_fps: float
    semantic_stride: int
    frame_limit: Optional[int]
    skip_existing: bool
    model_cache_dir: str = "models"


def _prepare_datasets(options: SemanticCacheOptions) -> list[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(options.dataset_config)
    blueprint = blueprint_generator.generate(user_config, options)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.train_dataset_group,
        training=False,
    )
    return list(dataset_group.datasets)


def _encode_dataset(
    dataset: BaseDataset,
    encoder: SemanticEncoder,
    options: SemanticCacheOptions,
):
    latent_cache_files = dataset.get_all_latent_cache_files()
    if not latent_cache_files:
        logger.warning("No latent cache files found for %s.", dataset)
        return

    frame_limit = (
        options.frame_limit
        if options.frame_limit is not None and options.frame_limit > 0
        else None
    )
    for cache_file in tqdm(latent_cache_files):
        item_key = os.path.basename(cache_file).rsplit("_", 2)[0]
        semantic_cache_path = dataset.get_semantic_encoder_output_cache_path_from_key(
            item_key
        )
        if options.skip_existing and os.path.exists(semantic_cache_path):
            continue

        if isinstance(dataset, VideoDataset):
            source_path = dataset._get_original_video_path_from_cache(
                cache_file, item_key
            )
            frames_np = (
                load_video(source_path)
                if source_path is not None
                else None
            )
        else:
            source_path = dataset._get_original_image_path_from_cache(
                cache_file, item_key
            )
            frames_np = None
            if source_path is not None and os.path.exists(source_path):
                with Image.open(source_path) as img:
                    frames_np = [np.array(img.convert("RGB"))]

        if source_path is None or not os.path.exists(source_path) or not frames_np:
            logger.warning("Missing source file for %s", cache_file)
            continue

        frame_tensor = torch.from_numpy(
            np.stack(frames_np, axis=0)
        ).permute(0, 3, 1, 2)
        frame_tensor = sample_frames(
            frame_tensor,
            target_fps=float(getattr(dataset, "target_fps", 16)),
            semantic_fps=float(options.semantic_fps),
            stride=int(options.semantic_stride),
            frame_limit=frame_limit,
        )
        frame_tensor = frame_tensor.to(device=options.device)
        embeddings = encoder.encode_frames(frame_tensor)

        item_info = ItemInfo(item_key, "", (0, 0))
        item_info.semantic_encoder_output_cache_path = semantic_cache_path
        save_semantic_encoder_output_cache_wan(item_info, embeddings)


def cache_semantic_embeddings(
    *,
    dataset_config: str,
    encoder_name: str,
    device: str = "cuda",
    encoder_type: str = "repa",
    encoder_dtype: str = "float16",
    encoder_resolution: int = 256,
    semantic_fps: float = 2.0,
    semantic_stride: int = 1,
    frame_limit: Optional[int] = None,
    skip_existing: bool = False,
    model_cache_dir: str = "models",
) -> None:
    options = SemanticCacheOptions(
        dataset_config=dataset_config,
        device=device,
        encoder_name=encoder_name,
        encoder_type=encoder_type,
        encoder_dtype=encoder_dtype,
        encoder_resolution=encoder_resolution,
        semantic_fps=semantic_fps,
        semantic_stride=semantic_stride,
        frame_limit=frame_limit,
        skip_existing=skip_existing,
        model_cache_dir=model_cache_dir,
    )
    device_obj = torch.device(options.device)
    encoder = SemanticEncoder(
        model_name=options.encoder_name,
        encoder_type=options.encoder_type,
        device=str(device_obj),
        dtype=getattr(torch, options.encoder_dtype),
        input_resolution=int(options.encoder_resolution),
        cache_dir=options.model_cache_dir,
    )
    datasets = _prepare_datasets(options)
    for dataset in datasets:
        _encode_dataset(dataset, encoder, options)
