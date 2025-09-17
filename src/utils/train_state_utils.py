"""Train state utilities for enhanced resume functionality.

This module handles saving and loading of training state information
to enable precise resume from specific steps and epochs.
"""

import json
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

try:
    from common.logger import get_logger
except ImportError:
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

logger = get_logger(__name__, level=logging.INFO)


class TrainStateManager:
    """Manages training state persistence for enhanced resume functionality."""

    @staticmethod
    def save_train_state(output_dir: str, current_epoch: int, current_step: int) -> None:
        """Save current training state to train_state.json.

        Args:
            output_dir: Directory to save the state file
            current_epoch: Current epoch number (1-indexed)
            current_step: Current step number (0-indexed, will be incremented for resume)
        """
        try:
            train_state_file = os.path.join(output_dir, "train_state.json")

            # +1 is needed because the state is saved before current_step is set from global_step
            state_data = {
                "current_epoch": current_epoch,
                "current_step": current_step + 1
            }

            logger.info(f"💾 Saving train state to {train_state_file} at epoch {current_epoch} step {current_step + 1}")

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to save train state: {e}")

    @staticmethod
    def load_train_state(input_dir: str) -> Optional[Dict[str, Any]]:
        """Load training state from train_state.json.

        Args:
            input_dir: Directory containing the state file

        Returns:
            Dictionary with current_epoch and current_step, or None if not found
        """
        try:
            train_state_file = os.path.join(input_dir, "train_state.json")

            if not os.path.exists(train_state_file):
                logger.debug(f"Train state file not found: {train_state_file}")
                return None

            with open(train_state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            logger.info(f"🔄 Loaded train state from {train_state_file}: {state_data}")
            return state_data

        except Exception as e:
            logger.warning(f"⚠️ Failed to load train state: {e}")
            return None

    @staticmethod
    def calculate_initial_step(
        args: Any,
        train_dataloader_length: int,
        num_processes: int,
        gradient_accumulation_steps: int,
        steps_from_state: Optional[int] = None
    ) -> tuple[int, int]:
        """Calculate initial step and epoch to start from.

        Args:
            args: Training arguments
            train_dataloader_length: Length of training dataloader
            num_processes: Number of accelerator processes
            gradient_accumulation_steps: Gradient accumulation steps
            steps_from_state: Steps loaded from train_state.json

        Returns:
            Tuple of (initial_step, epoch_to_start)
        """
        initial_step = 0

        # Priority: args.initial_step > args.initial_epoch > steps_from_state
        if getattr(args, "initial_step", None) is not None or getattr(args, "initial_epoch", None) is not None:
            # If initial_epoch or initial_step is specified, steps_from_state is ignored
            if steps_from_state is not None:
                logger.warning(
                    "Steps from train_state.json ignored because initial_step/initial_epoch is specified"
                )

            if getattr(args, "initial_step", None) is not None:
                initial_step = args.initial_step
            else:
                # Calculate steps from initial_epoch
                steps_per_epoch = math.ceil(
                    train_dataloader_length / num_processes / gradient_accumulation_steps
                )
                initial_step = (args.initial_epoch - 1) * steps_per_epoch

        elif steps_from_state is not None:
            # Use steps from train_state.json
            initial_step = steps_from_state

        # Calculate epoch to start from
        epoch_to_start = 0
        if initial_step > 0:
            steps_per_epoch = math.ceil(
                train_dataloader_length / num_processes / gradient_accumulation_steps
            )
            epoch_to_start = initial_step // steps_per_epoch

        return initial_step, epoch_to_start

    @staticmethod
    def should_skip_epoch(initial_step: int, steps_per_epoch: int) -> tuple[bool, int]:
        """Check if current epoch should be skipped and calculate remaining steps.

        Args:
            initial_step: Initial step to start from
            steps_per_epoch: Steps per epoch

        Returns:
            Tuple of (should_skip, remaining_initial_step)
        """
        if initial_step > steps_per_epoch:
            return True, initial_step - steps_per_epoch
        return False, initial_step


def save_step_to_state_dir(state_dir: str, global_step: int) -> None:
    """Save step information to a checkpoint state directory.

    This function is used for compatibility with existing checkpoint saving.

    Args:
        state_dir: State directory path
        global_step: Current global step
    """
    try:
        step_file = os.path.join(state_dir, "step_info.json")
        step_data = {"global_step": global_step}

        with open(step_file, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2)

        logger.debug(f"📝 Saved step info to {step_file}: step {global_step}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save step info: {e}")


# Import math for calculations
import math