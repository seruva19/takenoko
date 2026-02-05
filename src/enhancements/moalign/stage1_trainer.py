"""Stage-1 trainer for MOALIGN motion teacher pretraining."""

from __future__ import annotations

import copy
import gc
import os
from multiprocessing import Value
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from common.logger import get_logger
from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from enhancements.moalign.stage1_model import Stage1FlowPredictor, Stage1MotionProjector
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from utils.train_utils import collator_class

logger = get_logger(__name__)


def _resolve_device(device_arg: Any) -> torch.device:
    if isinstance(device_arg, torch.device):
        return device_arg
    if isinstance(device_arg, str) and device_arg:
        try:
            return torch.device(device_arg)
        except Exception:
            logger.warning("Invalid device '%s', falling back to auto device.", device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_encoder_features(features: Any) -> torch.Tensor:
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            features = features["x_norm_patchtokens"]
        elif "x_norm_clstoken" in features:
            features = features["x_norm_clstoken"].unsqueeze(1)
        else:
            tensor_candidate = None
            for value in features.values():
                if isinstance(value, torch.Tensor):
                    tensor_candidate = value
                    break
            if tensor_candidate is None:
                raise ValueError("MOALIGN Stage-1: encoder output did not contain tensor features")
            features = tensor_candidate

    if not isinstance(features, torch.Tensor):
        raise ValueError("MOALIGN Stage-1: unsupported encoder output type")

    if features.dim() == 2:
        features = features.unsqueeze(1)
    elif features.dim() > 3:
        n_s, c_feat, h_feat, w_feat = features.shape
        features = features.view(n_s, c_feat, h_feat * w_feat).transpose(1, 2)
    return features


def _extract_encoder_tokens(
    encoder: torch.nn.Module,
    encoder_type: str,
    clean_pixels: torch.Tensor,
) -> torch.Tensor:
    if clean_pixels.dim() != 5:
        raise ValueError(
            f"MOALIGN Stage-1: clean_pixels must be [B,C,F,H,W], got {tuple(clean_pixels.shape)}"
        )

    bsz, channels, frames, height, width = clean_pixels.shape
    images = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
        bsz * frames, channels, height, width
    )
    with torch.no_grad():
        images = ((images + 1.0) / 2.0).clamp(0, 1) * 255.0
        images = preprocess_raw_image(images, encoder_type)
        features = encoder.forward_features(images)
        features = _parse_encoder_features(features)
    return features.view(bsz, frames, features.shape[1], features.shape[2])


def _extract_pixels_batch(
    batch: dict[str, Any], device: torch.device
) -> Optional[torch.Tensor]:
    pixels = batch.get("pixels")
    if not isinstance(pixels, list) or len(pixels) == 0:
        return None

    first = pixels[0]
    if not isinstance(first, torch.Tensor) or first.dim() != 4:
        return None
    expected_shape = tuple(first.shape)
    if expected_shape[1] < 2:
        return None

    for tensor in pixels:
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
            return None
        if tuple(tensor.shape) != expected_shape:
            return None

    return torch.stack(pixels, dim=0).to(device=device, dtype=torch.float32)


def _extract_cached_flow(
    batch: dict[str, Any],
    expected_batch_size: int,
    expected_frames: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    flow = batch.get("optical_flow")
    if not isinstance(flow, torch.Tensor):
        return None
    if flow.dim() == 4:
        flow = flow.unsqueeze(0)
    if flow.dim() != 5:
        return None
    if flow.shape[0] != expected_batch_size:
        return None
    if flow.shape[1] != expected_frames - 1:
        return None
    if flow.shape[2] != 2:
        return None
    return flow.to(device=device, dtype=torch.float32)


def _compute_raft_flow(
    raft_model: torch.nn.Module,
    clean_pixels: torch.Tensor,
) -> torch.Tensor:
    from caching.cache_optical_flow import _compute_flow_batch

    video_batch = ((clean_pixels + 1.0) / 2.0).clamp(0, 1)
    video_batch = video_batch.permute(0, 2, 1, 3, 4).contiguous()  # B,F,C,H,W
    with torch.no_grad():
        return _compute_flow_batch(raft_model, video_batch).to(dtype=torch.float32)


def _resolve_checkpoint_path(args: Any, global_step: int, intermediate: bool = False) -> str:
    configured_path = str(getattr(args, "moalign_stage1_checkpoint", "") or "")
    if configured_path:
        base_path = configured_path
    else:
        output_dir = str(getattr(args, "output_dir", "") or "output")
        base_path = os.path.join(output_dir, "moalign_stage1_teacher.pt")

    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".pt"
    if intermediate:
        path = f"{root}-step{global_step:06d}{ext}"
    else:
        path = f"{root}{ext}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def _save_checkpoint(
    args: Any,
    motion_projector: Stage1MotionProjector,
    flow_predictor: Stage1FlowPredictor,
    global_step: int,
    mean_loss: float,
    intermediate: bool = False,
) -> str:
    checkpoint_path = _resolve_checkpoint_path(args, global_step, intermediate=intermediate)
    payload = {
        "format": "moalign_stage1_v1",
        "global_step": int(global_step),
        "mean_loss": float(mean_loss),
        "encoder_name": str(getattr(args, "moalign_encoder_name", "dinov2-vit-b")),
        "input_resolution": int(getattr(args, "moalign_input_resolution", 256)),
        "motion_projector": motion_projector.state_dict(),
        "flow_predictor": flow_predictor.state_dict(),
        "input_dim": int(motion_projector.input_dim),
        "hidden_dim": int(motion_projector.hidden_dim),
        "output_dim": int(motion_projector.output_dim),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _build_stage1_dataloader(args: Any) -> Tuple[Any, Value, Value]:
    stage1_args = copy.deepcopy(args)
    stage1_args.batch_size = int(getattr(args, "moalign_stage1_batch_size", 1))

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(stage1_args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, stage1_args)

    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.train_dataset_group,
        training=True,
        load_pixels_for_batches=True,
        prior_loss_weight=getattr(stage1_args, "prior_loss_weight", 1.0),
        shared_epoch=current_epoch,
    )
    if hasattr(train_dataset_group, "set_max_train_steps"):
        train_dataset_group.set_max_train_steps(
            int(getattr(args, "moalign_stage1_max_steps", 2000))
        )

    num_workers = int(getattr(args, "moalign_stage1_num_workers", 0))
    ds_for_collator = train_dataset_group if num_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)
    dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=(bool(getattr(args, "persistent_data_loader_workers", False)) and num_workers > 0),
        pin_memory=bool(getattr(args, "data_loader_pin_memory", False)),
    )
    return dataloader, current_epoch, current_step


def train_moalign_stage1(args: Any) -> str:
    """Train MOALIGN Stage-1 teacher and return checkpoint path."""

    device = _resolve_device(getattr(args, "device", None))
    logger.info("MOALIGN Stage-1: using device=%s", device)

    cache_dir = getattr(args, "model_cache_dir", "models")
    resolution = int(getattr(args, "moalign_input_resolution", 256))
    encoder_spec = getattr(args, "moalign_encoder_name", "dinov2-vit-b")

    manager = EncoderManager(device=str(device), cache_dir=cache_dir)
    encoders, encoder_types, _ = manager.load_encoders(encoder_spec, resolution=resolution)
    encoder = encoders[0]
    encoder_type = encoder_types[0]
    if len(encoders) > 1:
        logger.warning(
            "MOALIGN Stage-1: multiple encoders provided; using first encoder '%s'.",
            encoder_spec.split(",")[0].strip(),
        )
    encoder.eval()
    encoder.requires_grad_(False)

    input_dim = int(getattr(encoder, "embed_dim", 0))
    if input_dim <= 0:
        raise ValueError("MOALIGN Stage-1: failed to infer encoder embed_dim.")

    motion_projector = Stage1MotionProjector(
        input_dim=input_dim,
        hidden_dim=int(getattr(args, "moalign_projection_hidden_dim", 256)),
        output_dim=int(getattr(args, "moalign_projection_out_dim", 64)),
    ).to(device=device, dtype=torch.float32)
    flow_predictor = Stage1FlowPredictor(
        input_dim=int(getattr(args, "moalign_projection_out_dim", 64))
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        list(motion_projector.parameters()) + list(flow_predictor.parameters()),
        lr=float(getattr(args, "moalign_stage1_learning_rate", 1e-4)),
        weight_decay=float(getattr(args, "moalign_stage1_weight_decay", 0.0)),
    )

    dataloader, current_epoch, current_step = _build_stage1_dataloader(args)

    num_epochs = int(getattr(args, "moalign_stage1_num_epochs", 1))
    max_steps = int(getattr(args, "moalign_stage1_max_steps", 2000))
    log_interval = int(getattr(args, "moalign_stage1_log_interval", 20))
    save_interval = int(getattr(args, "moalign_stage1_save_interval", 500))
    grad_clip_norm = float(getattr(args, "moalign_stage1_grad_clip_norm", 1.0))
    token_reg_weight = float(getattr(args, "moalign_stage1_token_reg_weight", 0.0))
    flow_source = str(getattr(args, "moalign_stage1_flow_source", "auto")).lower()
    allow_raft_fallback = bool(getattr(args, "moalign_stage1_allow_raft_fallback", True))
    raft_model_name = str(getattr(args, "moalign_stage1_raft_model", "raft_small"))
    logger.info(
        "MOALIGN Stage-1 hyperparameters: epochs=%d max_steps=%d lr=%.6f flow_source=%s token_reg_weight=%.6f",
        num_epochs,
        max_steps,
        float(getattr(args, "moalign_stage1_learning_rate", 1e-4)),
        flow_source,
        token_reg_weight,
    )

    raft_model: Optional[torch.nn.Module] = None

    def get_raft_model() -> torch.nn.Module:
        nonlocal raft_model
        if raft_model is None:
            from caching.cache_optical_flow import _load_raft_model

            logger.info("MOALIGN Stage-1: loading RAFT model '%s'.", raft_model_name)
            raft_model = _load_raft_model(raft_model_name, device)
        return raft_model

    global_step = 0
    running_loss = 0.0
    skipped_for_missing_pixels = 0
    skipped_for_missing_flow = 0

    motion_projector.train()
    flow_predictor.train()

    for epoch in range(num_epochs):
        current_epoch.value = epoch
        for batch in dataloader:
            if global_step >= max_steps:
                break
            current_step.value = global_step

            clean_pixels = _extract_pixels_batch(batch, device=device)
            if clean_pixels is None:
                skipped_for_missing_pixels += 1
                continue

            expected_batch_size = clean_pixels.shape[0]
            expected_frames = clean_pixels.shape[2]
            target_flow = None

            if flow_source in {"cache", "auto"}:
                target_flow = _extract_cached_flow(
                    batch,
                    expected_batch_size=expected_batch_size,
                    expected_frames=expected_frames,
                    device=device,
                )

            need_raft = flow_source == "raft" or (
                flow_source == "auto" and target_flow is None and allow_raft_fallback
            )
            if need_raft:
                target_flow = _compute_raft_flow(get_raft_model(), clean_pixels)

            if target_flow is None:
                skipped_for_missing_flow += 1
                continue

            with torch.no_grad():
                encoder_tokens = _extract_encoder_tokens(encoder, encoder_type, clean_pixels)

            motion_tokens, motion_maps = motion_projector.forward_tokens(encoder_tokens)
            pred_flow = flow_predictor(
                motion_maps,
                target_height=int(target_flow.shape[-2]),
                target_width=int(target_flow.shape[-1]),
                target_frames=int(target_flow.shape[1]),
            )

            flow_loss = F.smooth_l1_loss(pred_flow, target_flow)
            token_reg = motion_tokens.square().mean()
            total_loss = flow_loss + token_reg_weight * token_reg

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(motion_projector.parameters()) + list(flow_predictor.parameters()),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

            global_step += 1
            running_loss += float(total_loss.detach().item())

            if global_step % log_interval == 0:
                logger.info(
                    "MOALIGN Stage-1: step=%d/%d epoch=%d loss=%.6f",
                    global_step,
                    max_steps,
                    epoch + 1,
                    running_loss / max(1, global_step),
                )

            if save_interval > 0 and global_step % save_interval == 0:
                checkpoint_path = _save_checkpoint(
                    args,
                    motion_projector,
                    flow_predictor,
                    global_step=global_step,
                    mean_loss=running_loss / max(1, global_step),
                    intermediate=True,
                )
                logger.info("MOALIGN Stage-1: saved intermediate checkpoint: %s", checkpoint_path)

        if global_step >= max_steps:
            break

    if global_step == 0:
        raise RuntimeError(
            "MOALIGN Stage-1: no optimization steps completed. "
            "Check dataset video frames and optical-flow availability."
        )

    final_checkpoint_path = _save_checkpoint(
        args,
        motion_projector,
        flow_predictor,
        global_step=global_step,
        mean_loss=running_loss / max(1, global_step),
        intermediate=False,
    )
    logger.info(
        "MOALIGN Stage-1 complete: steps=%d, avg_loss=%.6f, skipped_pixels=%d, skipped_flow=%d, checkpoint=%s",
        global_step,
        running_loss / max(1, global_step),
        skipped_for_missing_pixels,
        skipped_for_missing_flow,
        final_checkpoint_path,
    )

    if raft_model is not None:
        del raft_model
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_checkpoint_path
