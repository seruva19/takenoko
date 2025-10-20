"""Factory helpers for constructing RCM teacher/student models."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from accelerate import Accelerator
import torch
import torch.nn as nn

from common.logger import get_logger

from core.model_manager import ModelManager
from distillation.rcm_core.config_loader import RCMConfig
from distillation.rcm_core.fake_score import SimpleRCMFakeScoreHead

logger = get_logger(__name__)


class SimpleRCMEncoder(nn.Module):
    """Lightweight MLP encoder used as a stand-in for the clean-room port."""

    def __init__(self, observation_dim: int, embedding_dim: int, action_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(embedding_dim, action_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.view(inputs.size(0), -1)
        embedding = self.encoder(x)
        logits = self.policy_head(embedding)
        return embedding, logits


def _load_checkpoint_if_available(module: nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(
            "RCM teacher checkpoint '%s' not found; using random weights.", checkpoint_path
        )
        return

    try:
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict, strict=False)
        logger.info("Loaded RCM teacher weights from %s", checkpoint_path)
    except Exception as exc:
        logger.warning("Failed to load teacher checkpoint '%s': %s", checkpoint_path, exc)


def create_rcm_models(
    args: Any,
    config: RCMConfig,
    *,
    accelerator: Accelerator,
    raw_config: Dict[str, Any] | None = None,
    config_path: str | None = None,
    device: torch.device | None = None,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    """Return teacher/student (and optional EMA/fake-score) modules for RCM."""

    # If explicit teacher checkpoint provided, load lightweight encoder fallbacks.
    teacher_checkpoint = config.extra_args.get("teacher_checkpoint")
    if teacher_checkpoint:
        observation_dim = max(int(config.extra_args.get("observation_dim", 1024)), 8)
        embedding_dim = max(int(config.extra_args.get("embedding_dim", 256)), 8)
        action_dim = max(int(config.extra_args.get("action_dim", 16)), 2)

        teacher = SimpleRCMEncoder(observation_dim, embedding_dim, action_dim)
        student = SimpleRCMEncoder(observation_dim, embedding_dim, action_dim)
        _load_checkpoint_if_available(teacher, teacher_checkpoint)

        teacher_ema = copy.deepcopy(student) if bool(config.extra_args.get("rcm_enable_ema", True)) else None
        fake_score = None
        if bool(config.extra_args.get("rcm_fake_score_enabled", False)):
            fake_dim = max(int(config.extra_args.get("rcm_fake_score_dim", embedding_dim)), 8)
            fake_score = SimpleRCMFakeScoreHead(fake_dim)

        if device is not None:
            teacher.to(device)
            student.to(device)
            if teacher_ema is not None:
                teacher_ema.to(device)
            if fake_score is not None:
                fake_score.to(device)

        return teacher, student, teacher_ema, fake_score

    # Otherwise, reuse Takenoko's WAN model loader so the teacher matches training semantics.
    model_manager = ModelManager()
    attn_mode = model_manager.get_attention_mode(args)
    split_attn = bool(getattr(args, "split_attn", False))
    dit_path = getattr(args, "dit", None)
    if not dit_path:
        raise ValueError("RCM pipeline requires 'dit' path in the configuration.")

    transformer, _ = model_manager.load_transformer(
        accelerator=accelerator,
        args=args,
        dit_path=dit_path,
        attn_mode=attn_mode,
        split_attn=split_attn,
        loading_device=accelerator.device,
        dit_weight_dtype=None,
        config=raw_config or {},
    )

    transformer.eval()
    transformer.requires_grad_(False)

    student = copy.deepcopy(transformer)
    student.train()
    student.requires_grad_(True)

    teacher_ema: Optional[nn.Module] = None
    if bool(config.extra_args.get("rcm_enable_ema", True)):
        teacher_ema = copy.deepcopy(student)
        teacher_ema.eval()
        teacher_ema.requires_grad_(False)

    fake_score: Optional[nn.Module] = None
    if bool(config.extra_args.get("rcm_fake_score_enabled", False)):
        fake_dim = max(int(config.extra_args.get("rcm_fake_score_dim", 1024)), 8)
        fake_score = SimpleRCMFakeScoreHead(fake_dim)

    if device is not None:
        transformer.to(device)
        student.to(device)
        if teacher_ema is not None:
            teacher_ema.to(device)
        if fake_score is not None:
            fake_score.to(device)

    return transformer, student, teacher_ema, fake_score
