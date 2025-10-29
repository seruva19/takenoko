"""
Sprint Training Scheduler

Implements two-stage training scheduler for Sprint:
1. Pretraining: Long phase with token dropping for efficiency
2. Fine-tuning: Brief phase with full tokens to close train-inference gap

Two-stage approach: Pre-train with aggressive token dropping, then fine-tune
with full-token processing to bridge the train-inference discrepancy.
"""

import torch
import logging
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class SprintTrainingScheduler:
    """
    Manages transition from token-dropping pretraining to full-token fine-tuning.

    Two-stage training schedule:
    - Stage 1 (Pretrain): Steps 0 to pretrain_steps → token_drop_ratio
    - Stage 2 (Finetune): Steps pretrain_steps to (pretrain_steps + finetune_steps) → 0.0 drop ratio

    The drop ratio can optionally have a linear warmup/cooldown for smooth transitions.

    Usage:
        scheduler = SprintTrainingScheduler(
            pretrain_steps=10000,
            finetune_steps=1000,
            initial_drop_ratio=0.75
        )

        # In training loop:
        for step in range(total_steps):
            current_drop_ratio = scheduler.get_drop_ratio(step)
            model.sprint_fusion.token_sampler.keep_ratio = 1.0 - current_drop_ratio

            # ... training ...

            if scheduler.should_log_transition(step):
                logger.info(f"Sprint stage transition at step {step}")
    """

    def __init__(
        self,
        pretrain_steps: int = 10000,
        finetune_steps: int = 1000,
        initial_drop_ratio: float = 0.75,
        warmup_steps: int = 0,
        cooldown_steps: int = 100,
    ):
        """
        Initialize Sprint training scheduler.

        Args:
            pretrain_steps: Number of steps for token-dropping pretraining
                            If 0, entire training uses token dropping (no finetune stage)
            finetune_steps: Number of steps for full-token fine-tuning
                            If 0, no fine-tuning stage (entire training uses token dropping)
            initial_drop_ratio: Token drop ratio during pretraining (default 0.75)
            warmup_steps: Number of steps to linearly warmup drop ratio from 0 → initial_drop_ratio
                          (default 0, no warmup)
            cooldown_steps: Number of steps to linearly cooldown drop ratio from initial → 0
                            during transition to fine-tuning (default 100)
        """
        self.pretrain_steps = pretrain_steps
        self.finetune_steps = finetune_steps
        self.initial_drop_ratio = initial_drop_ratio
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps

        # Calculate total steps and transition point
        self.total_steps = pretrain_steps + finetune_steps
        self.transition_start = max(0, pretrain_steps - cooldown_steps)
        self.transition_end = pretrain_steps

        # Track stage transitions for logging
        self._logged_warmup = False
        self._logged_pretrain = False
        self._logged_transition = False
        self._logged_finetune = False

        # Validation
        if pretrain_steps < 0 or finetune_steps < 0:
            raise ValueError("pretrain_steps and finetune_steps must be non-negative")
        if not 0.0 <= initial_drop_ratio <= 1.0:
            raise ValueError("initial_drop_ratio must be in [0.0, 1.0]")
        if warmup_steps < 0 or cooldown_steps < 0:
            raise ValueError("warmup_steps and cooldown_steps must be non-negative")

        logger.info(
            f"Sprint scheduler initialized: "
            f"pretrain={pretrain_steps} steps (drop_ratio={initial_drop_ratio:.2f}), "
            f"finetune={finetune_steps} steps (drop_ratio=0.0), "
            f"warmup={warmup_steps}, cooldown={cooldown_steps}"
        )

    def get_drop_ratio(self, step: int) -> float:
        """
        Get current token drop ratio based on training step.

        Schedule:
        - Steps 0 to warmup_steps: Linear warmup from 0 → initial_drop_ratio
        - Steps warmup_steps to transition_start: Constant initial_drop_ratio
        - Steps transition_start to transition_end: Linear cooldown initial_drop_ratio → 0
        - Steps transition_end+: Constant 0 (full tokens)

        Args:
            step: Current training step (0-indexed)

        Returns:
            Token drop ratio for current step
        """
        # Stage 1: Warmup (if enabled)
        if step < self.warmup_steps:
            # Linear warmup from 0 to initial_drop_ratio
            return self.initial_drop_ratio * (step / self.warmup_steps)

        # Stage 2: Pretraining (constant drop ratio)
        if step < self.transition_start:
            return self.initial_drop_ratio

        # Stage 3: Transition / Cooldown
        if step < self.transition_end:
            # Linear cooldown from initial_drop_ratio to 0
            progress = (step - self.transition_start) / self.cooldown_steps
            return self.initial_drop_ratio * (1.0 - progress)

        # Stage 4: Fine-tuning (no dropping)
        return 0.0

    def get_keep_ratio(self, step: int) -> float:
        """Get keep ratio (1 - drop_ratio) for current step."""
        return 1.0 - self.get_drop_ratio(step)

    def is_pretraining(self, step: int) -> bool:
        """Check if currently in pretraining stage."""
        return step < self.transition_start

    def is_finetuning(self, step: int) -> bool:
        """Check if currently in fine-tuning stage."""
        return step >= self.transition_end

    def is_transitioning(self, step: int) -> bool:
        """Check if currently in transition (cooldown) phase."""
        return self.transition_start <= step < self.transition_end

    def get_stage_name(self, step: int) -> str:
        """Get human-readable stage name for current step."""
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.transition_start:
            return "pretrain"
        elif step < self.transition_end:
            return "transition"
        else:
            return "finetune"

    def should_log_transition(self, step: int) -> bool:
        """
        Check if this step is a stage transition that should be logged.

        Returns True once per stage transition for clean logging.
        """
        stage = self.get_stage_name(step)

        if stage == "warmup" and not self._logged_warmup:
            self._logged_warmup = True
            return True
        elif stage == "pretrain" and not self._logged_pretrain:
            self._logged_pretrain = True
            return True
        elif stage == "transition" and not self._logged_transition:
            self._logged_transition = True
            return True
        elif stage == "finetune" and not self._logged_finetune:
            self._logged_finetune = True
            return True

        return False

    def log_schedule_info(self):
        """Log schedule information (call once at training start)."""
        logger.info("=" * 60)
        logger.info("Sprint Training Schedule")
        logger.info("=" * 60)

        if self.warmup_steps > 0:
            logger.info(f"Warmup:     Steps 0-{self.warmup_steps} (drop ratio 0.0 → {self.initial_drop_ratio:.2f})")

        logger.info(
            f"Pretrain:   Steps {self.warmup_steps}-{self.transition_start} "
            f"(drop ratio {self.initial_drop_ratio:.2f}, {self.transition_start - self.warmup_steps} steps)"
        )

        if self.cooldown_steps > 0:
            logger.info(
                f"Transition: Steps {self.transition_start}-{self.transition_end} "
                f"(drop ratio {self.initial_drop_ratio:.2f} → 0.0, {self.cooldown_steps} steps)"
            )

        if self.finetune_steps > 0:
            logger.info(
                f"Finetune:   Steps {self.transition_end}-{self.total_steps} "
                f"(drop ratio 0.0, {self.finetune_steps} steps)"
            )

        logger.info("=" * 60)

    def get_progress_string(self, step: int) -> str:
        """Get formatted progress string for logging."""
        stage = self.get_stage_name(step)
        drop_ratio = self.get_drop_ratio(step)
        keep_ratio = self.get_keep_ratio(step)

        return (
            f"Step {step}/{self.total_steps} | "
            f"Stage: {stage} | "
            f"Drop ratio: {drop_ratio:.3f} | "
            f"Keep ratio: {keep_ratio:.3f}"
        )

    @staticmethod
    def from_args(args) -> Optional['SprintTrainingScheduler']:
        """
        Create scheduler from argparse arguments.

        Args:
            args: Argparse namespace with sprint_pretrain_steps, sprint_finetune_steps, etc.

        Returns:
            SprintTrainingScheduler if Sprint is enabled, None otherwise
        """
        if not getattr(args, 'enable_sprint', False):
            return None

        pretrain_steps = getattr(args, 'sprint_pretrain_steps', 0)
        finetune_steps = getattr(args, 'sprint_finetune_steps', 0)

        # If both are 0, use entire training as pretraining
        if pretrain_steps == 0 and finetune_steps == 0:
            pretrain_steps = getattr(args, 'max_train_steps', 10000)
            finetune_steps = 0
            logger.info(
                f"Sprint two-stage training not configured; "
                f"using entire training ({pretrain_steps} steps) with token dropping"
            )

        token_drop_ratio = getattr(args, 'sprint_token_drop_ratio', 0.75)
        warmup_steps = getattr(args, 'sprint_warmup_steps', 0)
        cooldown_steps = getattr(args, 'sprint_cooldown_steps', 100)

        return SprintTrainingScheduler(
            pretrain_steps=pretrain_steps,
            finetune_steps=finetune_steps,
            initial_drop_ratio=token_drop_ratio,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
        )
