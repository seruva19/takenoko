"""Utilities for translating Takenoko datasets into RCM replay buffers."""

from __future__ import annotations

import os
from multiprocessing import Value
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from common.logger import get_logger
from dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
)
from distillation.rcm_core.buffers import RCMReplayBuffer, RCMReplaySample
from distillation.rcm_core.config_loader import RCMConfig
from utils.train_utils import collator_class

logger = get_logger(__name__)


def _build_dataset_group(
    args: Any,
    raw_config: Dict[str, Any] | None,
) -> DataLoader:
    sanitizer = ConfigSanitizer()
    generator = BlueprintGenerator(sanitizer)
    user_config = raw_config or getattr(args, "raw_config", None)
    if user_config is None:
        raise RuntimeError("RCM dataset adapter requires raw configuration data.")

    blueprint = generator.generate(user_config, args)

    shared_epoch = Value("i", 0)
    shared_step = Value("i", 0)

    train_group = generate_dataset_group_by_blueprint(
        blueprint.train_dataset_group,
        training=True,
        load_pixels_for_batches=False,
        prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
        num_timestep_buckets=getattr(args, "num_timestep_buckets", None),
        shared_epoch=shared_epoch,
    )

    max_workers = getattr(args, "max_data_loader_n_workers", 0) or 0
    num_workers = min(max_workers, os.cpu_count() or 1)

    ds_for_collator = train_group if num_workers == 0 else None
    collator = collator_class(shared_epoch, shared_step, ds_for_collator)

    loader = DataLoader(
        train_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=bool(
            getattr(args, "persistent_data_loader_workers", False) and num_workers > 0
        ),
    )
    return loader


def _extract_observations(batch: Dict[str, Any], device: torch.device | None) -> torch.Tensor:
    candidate_keys = (
        "latents",
        "latent",
        "vae_latents",
    )
    for key in candidate_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            return tensor.to(device=device or tensor.device)

    pixels = batch.get("pixels")
    if isinstance(pixels, torch.Tensor):
        return pixels.detach().to(device=device or pixels.device)
    if isinstance(pixels, list) and pixels:
        first = pixels[0]
        if isinstance(first, torch.Tensor):
            return first.detach().to(device=device or first.device)

    caption = batch.get("caption")
    if isinstance(caption, str):
        tokens = torch.tensor([ord(ch) % 256 for ch in caption], dtype=torch.float32)
        return tokens.to(device=device or tokens.device)

    raise RuntimeError("Unable to derive observations from dataset batch.")


def _first_available(batch: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in batch:
            return batch[key]
    return None


def _ensure_tensor_sequence(value: Any, *, expected_length: Optional[int] = None) -> Optional[List[torch.Tensor]]:
    if value is None:
        return None

    tensors: List[torch.Tensor] | None = None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if expected_length is not None and tensor.ndim > 0 and tensor.shape[0] == expected_length:
            tensors = [tensor[idx].detach() for idx in range(expected_length)]
        else:
            tensors = [tensor]
    elif isinstance(value, (list, tuple)):
        items = [item.detach() for item in value if isinstance(item, torch.Tensor)]
        if items:
            tensors = items

    if tensors is None:
        return None

    return [tensor.detach() for tensor in tensors]


def _sequence_length_from_embedding(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 1
    if tensor.ndim == 1:
        return int(tensor.shape[0])
    if tensor.ndim >= 2:
        return int(tensor.shape[-2])
    return 1


def _infer_attention_mask(tensor: torch.Tensor) -> torch.Tensor:
    seq_len = _sequence_length_from_embedding(tensor)
    mask = torch.ones(seq_len, dtype=torch.float32, device=tensor.device)
    return mask


def _infer_pooled_output(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        pooled = tensor.mean(dim=-2)
    else:
        pooled = tensor.clone()
    return pooled.detach()


def _extract_payload(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise dataset tensors into WAN-style replay payload entries."""

    payload: Dict[str, Any] = {}

    latent_value = _first_available(batch, ("latents", "latent", "vae_latents"))
    if isinstance(latent_value, torch.Tensor):
        payload["latents"] = latent_value.detach()

    timesteps_val = batch.get("timesteps")
    if isinstance(timesteps_val, torch.Tensor):
        payload["timesteps"] = timesteps_val.detach()
    elif isinstance(timesteps_val, (list, tuple)):
        try:
            payload["timesteps"] = torch.tensor(list(timesteps_val), dtype=torch.float32)
        except (TypeError, ValueError):
            pass
    elif timesteps_val is not None:
        try:
            payload["timesteps"] = torch.tensor([float(timesteps_val)], dtype=torch.float32)
        except (TypeError, ValueError):
            pass

    weight_val = batch.get("weight")
    if isinstance(weight_val, torch.Tensor):
        payload["weight"] = weight_val.detach()
    elif isinstance(weight_val, (list, tuple)):
        try:
            payload["weight"] = torch.tensor(weight_val, dtype=torch.float32)
        except (TypeError, ValueError):
            pass

    for aux_key in ("control_signal", "mask_signal", "pixels", "t5_preservation"):
        aux_val = batch.get(aux_key)
        if isinstance(aux_val, torch.Tensor):
            payload[aux_key] = aux_val.detach()
        elif isinstance(aux_val, list) and aux_val and all(isinstance(item, torch.Tensor) for item in aux_val):
            payload[aux_key] = [item.detach() for item in aux_val]

    positive_embeddings = _ensure_tensor_sequence(
        _first_available(
            batch,
            (
                "t5",
                "t5_text_embeddings",
                "t5_embeddings",
                "positive_prompt_embeds",
            ),
        )
    )

    if positive_embeddings:
        payload["t5"] = positive_embeddings

        positive_mask = _ensure_tensor_sequence(
            _first_available(
                batch,
                (
                    "t5_attention_mask",
                    "t5_mask",
                    "prompt_attention_mask",
                ),
            ),
            expected_length=len(positive_embeddings),
        )
        if positive_mask is None:
            positive_mask = [_infer_attention_mask(t) for t in positive_embeddings]
        payload["t5_attention_mask"] = [mask.to(dtype=torch.float32) for mask in positive_mask]

        positive_pooled = _ensure_tensor_sequence(
            _first_available(
                batch,
                (
                    "t5_pooled_output",
                    "pooled_output",
                    "prompt_pooled_output",
                ),
            ),
            expected_length=len(positive_embeddings),
        )
        if positive_pooled is None:
            positive_pooled = [_infer_pooled_output(t) for t in positive_embeddings]
        payload["t5_pooled_output"] = positive_pooled

        preservation_embeddings = _ensure_tensor_sequence(
            batch.get("t5_preservation"), expected_length=len(positive_embeddings)
        )
        if preservation_embeddings:
            payload.setdefault("t5_preservation", preservation_embeddings)

        negative_embeddings = _ensure_tensor_sequence(
            _first_available(
                batch,
                (
                    "t5_negative",
                    "neg_t5_text_embeddings",
                    "negative_prompt_embeds",
                ),
            ),
            expected_length=len(positive_embeddings),
        )

        if negative_embeddings is None and preservation_embeddings:
            negative_embeddings = preservation_embeddings

        if negative_embeddings is None:
            negative_embeddings = [torch.zeros_like(t) for t in positive_embeddings]

        payload["t5_negative"] = negative_embeddings

        negative_mask = _ensure_tensor_sequence(
            _first_available(
                batch,
                (
                    "t5_negative_attention_mask",
                    "neg_t5_attention_mask",
                ),
            ),
            expected_length=len(negative_embeddings),
        )
        if negative_mask is None:
            negative_mask = [_infer_attention_mask(t) for t in negative_embeddings]
        payload["t5_negative_attention_mask"] = [mask.to(dtype=torch.float32) for mask in negative_mask]

        negative_pooled = _ensure_tensor_sequence(
            _first_available(
                batch,
                (
                    "t5_negative_pooled_output",
                    "neg_t5_pooled_output",
                ),
            ),
            expected_length=len(negative_embeddings),
        )
        if negative_pooled is None:
            negative_pooled = [_infer_pooled_output(t) for t in negative_embeddings]
        payload["t5_negative_pooled_output"] = negative_pooled

    return payload



def _build_metadata(
    batch: Dict[str, Any],
    config_path: Optional[str],
    index: int,
    observations: torch.Tensor,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": "takenoko_dataset",
        "config_path": config_path,
        "index": index,
    }
    if "caption" in batch:
        metadata["caption"] = batch["caption"]
    if "dataset_name" in batch:
        metadata["dataset_name"] = batch["dataset_name"]

    if observations.ndim >= 5:
        modality = "video" if observations.shape[2] > 1 else "image"
    else:
        modality = "image"
    metadata["modality"] = modality

    metadata["payload_keys"] = sorted(payload.keys())

    latents = payload.get("latents")
    if isinstance(latents, torch.Tensor):
        metadata["latents_shape"] = list(latents.shape)

    if "t5" in payload:
        seq_lengths = []
        for tensor in payload["t5"]:
            if isinstance(tensor, torch.Tensor):
                seq_lengths.append(_sequence_length_from_embedding(tensor))
        if seq_lengths:
            metadata["t5_sequence_lengths"] = seq_lengths

    metadata["has_negative_condition"] = bool(payload.get("t5_negative"))

    pos_mask = payload.get("t5_attention_mask")
    if isinstance(pos_mask, list) and pos_mask:
        first_mask = pos_mask[0]
        if isinstance(first_mask, torch.Tensor):
            metadata["t5_mask_shape"] = list(first_mask.shape)

    neg_mask = payload.get("t5_negative_attention_mask")
    if isinstance(neg_mask, list) and neg_mask:
        first_mask = neg_mask[0]
        if isinstance(first_mask, torch.Tensor):
            metadata["t5_negative_mask_shape"] = list(first_mask.shape)

    timesteps_val = payload.get("timesteps") or batch.get("timesteps")
    if isinstance(timesteps_val, torch.Tensor):
        metadata["timesteps"] = timesteps_val.detach().cpu().float().tolist()
    elif isinstance(timesteps_val, (list, tuple)):
        metadata["timesteps"] = [float(x) for x in timesteps_val]

    return metadata


_ALIAS_KEYS = {
    "t5_text_embeddings": "t5",
    "t5_embeddings": "t5",
    "positive_prompt_embeds": "t5",
    "neg_t5_text_embeddings": "t5_negative",
    "negative_prompt_embeds": "t5_negative",
    "t5_attention_mask": "t5_attention_mask",
    "neg_t5_attention_mask": "t5_negative_attention_mask",
    "t5_pooled_output": "t5_pooled_output",
    "neg_t5_pooled_output": "t5_negative_pooled_output",
}


def _apply_payload_aliases(batch: Dict[str, Any], payload: Dict[str, Any]) -> None:
    for src_key, dst_key in _ALIAS_KEYS.items():
        if src_key in batch and dst_key not in payload:
            payload[dst_key] = batch[src_key]

def create_rcm_dataset(
    args: Any,
    config: RCMConfig,
    *,
    raw_config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    device: torch.device | None = None,
) -> RCMReplayBuffer:
    """Construct an :class:`RCMReplayBuffer` using Takenoko dataset configuration."""

    synthetic_samples = config.extra_args.get("synthetic_samples")
    seed = int(config.extra_args.get("seed", getattr(args, "seed", 42) or 42))
    if synthetic_samples:
        logger.info(
            "RCM adapter generating %s synthetic samples (seed=%s)",
            synthetic_samples,
            seed,
        )
        return _create_synthetic_buffer(config, device, seed)

    try:
        dataloader = _build_dataset_group(args, raw_config)
    except Exception as exc:
        logger.warning("Failed to build dataset group (%s). Falling back to synthetic buffer.", exc)
        return _create_synthetic_buffer(config, device, seed)

    default_capacity = max(config.max_steps or 0, 32)
    buffer_capacity = int(config.extra_args.get("buffer_capacity", default_capacity))
    action_dim = max(int(config.extra_args.get("action_dim", 16)), 2)

    buffer = RCMReplayBuffer(capacity=buffer_capacity, device=device)

    for index, batch in enumerate(dataloader):
        if index >= buffer_capacity:
            break

        if isinstance(batch, list):
            batch = batch[0]

        observations = _extract_observations(batch, device)
        teacher_logits = batch.get("teacher_logits")
        if isinstance(teacher_logits, torch.Tensor):
            teacher_logits = teacher_logits.to(device=device or observations.device)
        else:
            teacher_logits = torch.zeros(action_dim, device=device or observations.device)

        actions = batch.get("actions")
        if isinstance(actions, torch.Tensor):
            actions = actions.to(device=device or observations.device)
        else:
            actions = torch.zeros(1, dtype=torch.long, device=device or observations.device)

        payload = _extract_payload(batch)
        _apply_payload_aliases(batch, payload)

        metadata = _build_metadata(batch, config_path, index, observations, payload)
        if "caption" in batch:
            metadata["caption"] = batch["caption"]
        if "dataset_name" in batch:
            metadata["dataset_name"] = batch["dataset_name"]

        sample = RCMReplaySample(
            observations=observations,
            actions=actions,
            teacher_logits=teacher_logits,
            metadata=metadata,
            payload=payload,
        )
        buffer.insert(sample)

    buffer.finalize()
    logger.info(
        "RCM replay buffer populated with %s samples from Takenoko datasets.",
        len(buffer),
    )
    return buffer


def _create_synthetic_buffer(
    config: RCMConfig,
    device: torch.device | None,
    seed: int,
) -> RCMReplayBuffer:
    """Generate a deterministic synthetic replay buffer for smoke tests."""

    sample_count = int(config.extra_args.get("synthetic_samples", 32))
    observation_dim = max(int(config.extra_args.get("observation_dim", 512)), 8)
    embedding_dim = max(int(config.extra_args.get("embedding_dim", 128)), 8)
    action_dim = max(int(config.extra_args.get("action_dim", 16)), 2)

    latent_channels = max(int(config.extra_args.get("synthetic_latent_channels", 4)), 1)
    latent_frames = max(int(config.extra_args.get("synthetic_latent_frames", 1)), 1)
    latent_height = max(int(config.extra_args.get("synthetic_latent_height", 16)), 4)
    latent_width = max(int(config.extra_args.get("synthetic_latent_width", 16)), 4)
    t5_seq_len = max(int(config.extra_args.get("synthetic_t5_length", 77)), 4)
    t5_hidden = max(int(config.extra_args.get("synthetic_t5_dim", embedding_dim)), 8)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    buffer = RCMReplayBuffer(capacity=sample_count, device=device)
    for index in range(sample_count):
        observations = torch.randn(observation_dim, generator=generator)
        teacher_embedding = torch.randn(embedding_dim, generator=generator)
        teacher_logits = torch.randn(action_dim, generator=generator)
        actions = torch.randint(0, action_dim, (1,), generator=generator)

        latents = torch.randn(
            1,
            latent_channels,
            latent_frames,
            latent_height,
            latent_width,
            generator=generator,
        )
        timesteps = torch.rand(1, generator=generator)
        t5_tensor = torch.randn(t5_seq_len, t5_hidden, generator=generator)
        t5_negative = torch.zeros_like(t5_tensor)
        attention_mask = torch.ones(t5_seq_len, dtype=torch.float32)
        pooled_output = t5_tensor.mean(dim=0)
        negative_pooled = t5_negative.mean(dim=0)

        payload = {
            "latents": latents,
            "timesteps": timesteps,
            "t5": [t5_tensor],
            "t5_negative": [t5_negative],
            "t5_attention_mask": [attention_mask],
            "t5_negative_attention_mask": [attention_mask.clone()],
            "t5_pooled_output": [pooled_output],
            "t5_negative_pooled_output": [negative_pooled],
        }

        metadata = {
            "source": "synthetic",
            "index": index,
            "modality": "image",
        }

        sample = RCMReplaySample(
            observations=observations,
            actions=actions,
            teacher_logits=teacher_logits,
            metadata=metadata,
            payload=payload,
        )

        # Store additional synthetic teacher embedding for fallback paths.
        sample.payload["teacher_embedding"] = teacher_embedding
        buffer.insert(sample)

    buffer.finalize()
    logger.info("RCM synthetic replay buffer generated with %s samples.", len(buffer))
    return buffer
