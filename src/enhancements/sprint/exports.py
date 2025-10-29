"""
Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers

Achieves significant training speedup with token dropping while maintaining generation quality.

Key Components:
- SparseDenseFusion: Core fusion module partitioning blocks into encoder/middle/decoder
- TokenSampler: Video-aware 3D token sampling with temporal coherence
- SprintTrainingScheduler: Two-stage training (pretraining + fine-tuning)

Usage:
    from enhancements.sprint import create_sprint_fusion, SprintTrainingScheduler

    # Create fusion module
    fusion = create_sprint_fusion(
        dim=2048,
        num_layers=32,
        token_drop_ratio=0.75,
        sampling_strategy="temporal_coherent"
    )

    # Create scheduler (optional, for two-stage training)
    scheduler = SprintTrainingScheduler(
        pretrain_steps=10000,
        finetune_steps=1000
    )

    # In training loop:
    drop_ratio = scheduler.get_drop_ratio(step)
    fusion.token_sampler.keep_ratio = 1.0 - drop_ratio
"""

from .sparse_dense_fusion import SparseDenseFusion, create_sprint_fusion
from .token_sampling import (
    TokenSampler,
    sample_tokens_3d,
    apply_token_sampling,
    restore_sequence_with_padding,
)
from .training_scheduler import SprintTrainingScheduler
from .model_integration import (
    setup_sprint_fusion,
    can_use_sprint,
    apply_sprint_forward,
    create_sprint_scheduler,
    update_sprint_drop_ratio,
    set_sprint_training_state,
    enable_sprint_with_validation,
)
from .diagnostics import (
    SprintDiagnostics,
    initialize_diagnostics,
    get_diagnostics,
    record_sprint_step,
)

__all__ = [
    # Core fusion
    "SparseDenseFusion",
    "create_sprint_fusion",
    # Token sampling
    "TokenSampler",
    "sample_tokens_3d",
    "apply_token_sampling",
    "restore_sequence_with_padding",
    # Training scheduler
    "SprintTrainingScheduler",
    # Model integration helpers
    "setup_sprint_fusion",
    "can_use_sprint",
    "apply_sprint_forward",
    "create_sprint_scheduler",
    "update_sprint_drop_ratio",
    "set_sprint_training_state",
    "enable_sprint_with_validation",
    # Diagnostics
    "SprintDiagnostics",
    "initialize_diagnostics",
    "get_diagnostics",
    "record_sprint_step",
]

__version__ = "1.0.4"
