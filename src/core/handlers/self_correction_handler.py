"""Self-correction logic handler for training loop."""

import argparse
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def handle_self_correction_update(
    args: argparse.Namespace,
    global_step: int,
    accelerator: Any,
    transformer: Any,
    training_model: Any,
) -> None:
    """Handle periodic self-correction cache refresh during training.
    
    This function manages the lightweight periodic self-correction cache refresh
    that is fully gated behind configuration flags.
    
    Args:
        args: Training arguments containing self-correction configuration
        global_step: Current global training step
        accelerator: Accelerator instance for distributed coordination
        transformer: Transformer model
        training_model: Training model wrapper
    """
    try:
        # Check if self-correction is enabled and conditions are met
        if (
            bool(getattr(args, "self_correction_enabled", False))
            and global_step > int(getattr(args, "self_correction_warmup_steps", 1000))
            and int(getattr(args, "self_correction_update_frequency", 1000)) > 0
            and global_step % int(getattr(args, "self_correction_update_frequency", 1000)) == 0
        ):
            # Access manager via trainer if present
            mgr = getattr(accelerator.state, "_self_correction_manager", None)
            
            # Fallback: some callers may attach it to the transformer
            if mgr is None:
                try:
                    mgr = getattr(transformer, "_self_correction_manager", None)
                except Exception:
                    mgr = None
                    
            if mgr is not None:
                # Switch to eval for generation
                try:
                    training_was = training_model.training
                    training_model.eval()
                except Exception:
                    training_was = None
                    
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    mgr.update_cache(accelerator.unwrap_model(transformer))  # type: ignore
                accelerator.wait_for_everyone()
                
                # Restore mode
                try:
                    if training_was is True:
                        training_model.train()
                except Exception:
                    pass
                    
    except Exception as _sc_err:
        # Non-fatal path: continue training
        if accelerator.is_main_process:
            logger.debug(f"Self-correction update skipped: {_sc_err}")