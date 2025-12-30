"""DenseDPO setup and initialization utilities."""

from __future__ import annotations

from typing import Any, Dict
import logging
import torch
import torch.nn as nn
from pathlib import Path

from densedpo.densedpo_config_schema import DenseDPOConfig
from densedpo.densedpo_training_core import DenseDPOTrainingCore
from densedpo.labeler import DenseDPOLabeler
from densedpo.vlm_labeler import DenseDPOVLMConfig, DenseDPOVLMLabeler
from srpo.srpo_reward_models import create_reward_model

logger = logging.getLogger(__name__)


def setup_densedpo_training(
    args,
    accelerator,
    transformer: nn.Module,
    network: nn.Module,
    optimizer,
    lr_scheduler,
    vae: nn.Module,
    model_config: Dict[str, Any],
) -> DenseDPOTrainingCore:
    """Set up DenseDPO training components and return the training core."""
    logger.info("Setting up DenseDPO training")

    densedpo_config = DenseDPOConfig(
        densedpo_partial_noise_eta=args.densedpo_partial_noise_eta,
        densedpo_num_inference_steps=args.densedpo_num_inference_steps,
        densedpo_segment_frames=args.densedpo_segment_frames,
        densedpo_beta=args.densedpo_beta,
        densedpo_label_source=args.densedpo_label_source,
        densedpo_segment_preference_key=args.densedpo_segment_preference_key,
        densedpo_reward_model_name=args.densedpo_reward_model_name,
        densedpo_reward_model_dtype=args.densedpo_reward_model_dtype,
        densedpo_reward_frame_strategy=args.densedpo_reward_frame_strategy,
        densedpo_reward_num_frames=args.densedpo_reward_num_frames,
        densedpo_reward_aggregation=args.densedpo_reward_aggregation,
        densedpo_vlm_model_path=args.densedpo_vlm_model_path,
        densedpo_vlm_dtype=args.densedpo_vlm_dtype,
        densedpo_vlm_prompt=args.densedpo_vlm_prompt,
        densedpo_vlm_max_new_tokens=args.densedpo_vlm_max_new_tokens,
        densedpo_vlm_temperature=args.densedpo_vlm_temperature,
        densedpo_vlm_cache_dir=args.densedpo_vlm_cache_dir,
        densedpo_vlm_max_frames=args.densedpo_vlm_max_frames,
    )

    labeler = None
    if densedpo_config.densedpo_label_source == "reward":
        if vae is None:
            raise ValueError(
                "DenseDPO reward labeling requires a VAE checkpoint (set 'vae' in config)."
            )
        reward_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[densedpo_config.densedpo_reward_model_dtype]
        reward_model = create_reward_model(
            reward_model_name=densedpo_config.densedpo_reward_model_name,
            device=accelerator.device,
            dtype=reward_dtype,
        )
        reward_model.requires_grad_(False)
        reward_model.eval()
        labeler = DenseDPOLabeler(
            reward_model=reward_model,
            vae=vae,
            device=accelerator.device,
            reward_frame_strategy=densedpo_config.densedpo_reward_frame_strategy,
            reward_num_frames=densedpo_config.densedpo_reward_num_frames,
            reward_aggregation=densedpo_config.densedpo_reward_aggregation,
        )
        logger.info("DenseDPO reward labeler initialized")
    elif densedpo_config.densedpo_label_source == "vlm":
        if vae is None:
            raise ValueError(
                "DenseDPO VLM labeling requires a VAE checkpoint (set 'vae' in config)."
            )
        if densedpo_config.densedpo_vlm_model_path is None:
            raise ValueError(
                "DenseDPO VLM labeling requires densedpo_vlm_model_path."
            )
        model_path = Path(densedpo_config.densedpo_vlm_model_path)
        if not model_path.exists():
            raise ValueError(
                f"DenseDPO VLM model path does not exist: {model_path}"
            )
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        vlm_config = DenseDPOVLMConfig(
            model_path=str(model_path),
            dtype=dtype_map.get(densedpo_config.densedpo_vlm_dtype, torch.bfloat16),
            device=accelerator.device,
            prompt=densedpo_config.densedpo_vlm_prompt,
            max_new_tokens=densedpo_config.densedpo_vlm_max_new_tokens,
            temperature=densedpo_config.densedpo_vlm_temperature,
            cache_dir=densedpo_config.densedpo_vlm_cache_dir,
            max_frames=densedpo_config.densedpo_vlm_max_frames,
        )
        labeler = DenseDPOVLMLabeler(vlm_config)
        logger.info("DenseDPO VLM labeler initialized")

    if hasattr(network, "prepare_network"):
        network.prepare_network(transformer)

    lora_param_count = sum(
        p.numel() for p in network.parameters() if p.requires_grad
    )
    if lora_param_count == 0:
        raise RuntimeError(
            "DenseDPO requires LoRA parameters; none were found."
        )
    logger.info("DenseDPO LoRA parameters: %s", lora_param_count)

    return DenseDPOTrainingCore(
        densedpo_config=densedpo_config,
        model_config=model_config,
        accelerator=accelerator,
        transformer=transformer,
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        vae=vae,
        args=args,
        labeler=labeler,
    )
