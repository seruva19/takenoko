"""Model saving logic handler for training loop."""

import argparse
from typing import Any, Optional, Callable


def handle_step_saving(
    should_saving: bool,
    accelerator: Any,
    save_model: Optional[Callable],
    remove_model: Optional[Callable],
    args: argparse.Namespace,
    network: Any,
    global_step: int,
    epoch: int,
) -> None:
    """Handle model saving during training step.
    
    Args:
        should_saving: Whether saving should occur
        accelerator: Accelerator instance
        save_model: Model saving function
        remove_model: Model removal function
        args: Training arguments
        network: Network model
        global_step: Current global step
        epoch: Current epoch (0-indexed)
    """
    if not should_saving:
        return
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and save_model:
        from utils import train_utils

        ckpt_name = train_utils.get_step_ckpt_name(
            args.output_name, global_step
        )
        save_model(
            ckpt_name,
            accelerator.unwrap_model(network),
            global_step,
            epoch + 1,
        )

        if args.save_state:
            train_utils.save_and_remove_state_stepwise(
                args, accelerator, global_step
            )

        remove_step_no = train_utils.get_remove_step_no(
            args, global_step
        )
        if remove_step_no is not None and remove_model:
            remove_ckpt_name = train_utils.get_step_ckpt_name(
                args.output_name, remove_step_no
            )
            remove_model(remove_ckpt_name)


def handle_epoch_end_saving(
    args: argparse.Namespace,
    epoch: int,
    num_train_epochs: int,
    is_main_process: bool,
    save_model: Optional[Callable],
    remove_model: Optional[Callable],
    accelerator: Any,
    network: Any,
    global_step: int,
) -> None:
    """Handle model saving at end of epoch.
    
    Args:
        args: Training arguments
        epoch: Current epoch (0-indexed)
        num_train_epochs: Total number of training epochs
        is_main_process: Whether this is the main process
        save_model: Model saving function
        remove_model: Model removal function
        accelerator: Accelerator instance
        network: Network model
        global_step: Current global step
    """
    if args.save_every_n_epochs is None:
        return
    
    saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
        epoch + 1
    ) < num_train_epochs
    
    if not (is_main_process and saving and save_model):
        return
    
    from utils import train_utils

    ckpt_name = train_utils.get_epoch_ckpt_name(
        args.output_name, epoch + 1
    )
    save_model(
        ckpt_name,
        accelerator.unwrap_model(network),
        global_step,
        epoch + 1,
    )

    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
    if remove_epoch_no is not None and remove_model:
        remove_ckpt_name = train_utils.get_epoch_ckpt_name(
            args.output_name, remove_epoch_no
        )
        remove_model(remove_ckpt_name)

    if args.save_state:
        from utils import train_utils

        train_utils.save_and_remove_state_on_epoch_end(
            args, accelerator, epoch + 1
        )