"""Core primitives for the Takenoko RCM-style distillation pipeline."""

from .config_loader import RCMConfig, load_rcm_config
from .buffers import RCMReplayBuffer, RCMReplaySample
from .losses import (
    ReconstructionLossConfig,
    BehaviorCloningLossConfig,
    DistillationLossOutput,
    compute_distillation_losses,
)
from .validation import validate_replay_buffer, summarize_replay_buffer, compare_replay_statistics
from .tokenizer import RCMTokenizer, TokenizerAssets, build_conditioning

__all__ = [
    "RCMConfig",
    "load_rcm_config",
    "RCMReplayBuffer",
    "RCMReplaySample",
    "ReconstructionLossConfig",
    "BehaviorCloningLossConfig",
    "DistillationLossOutput",
    "compute_distillation_losses",
    "validate_replay_buffer",
    "summarize_replay_buffer",
    "compare_replay_statistics",
    "RCMTokenizer",
    "TokenizerAssets",
    "build_conditioning",
]
