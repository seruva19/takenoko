"""Fira optimizer implementation for WAN network trainer.

This module provides Fira and FiraPT optimizer implementations with proper
parameter group handling and Fira-specific parameter management.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class FiraOptimizerManager:
    """Handles Fira optimizer creation and configuration."""

    @staticmethod
    def create_fira_optimizer(
        args: Any,
        transformer: torch.nn.Module,  # We still receive the transformer for context
        trainable_params: List[torch.nn.Parameter],
        lr: float,
        optimizer_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Create Fira optimizer using fira.FiraAdamW with enhanced logging and parameter handling."""
        from fira import FiraAdamW, divide_params

        optimizer_class = FiraAdamW

        # ====================================================================
        # The `transformer` is frozen. Gradients are only computed for the LoRA
        # `network` parameters, which are passed in via `trainable_params`.
        #
        # The Fira documentation example performs full fine-tuning, so all model
        # parameters get gradients. Here, we must only give the optimizer the
        # parameters that are actually trainable.
        #
        # The `divide_params` function adds Fira-specific attributes to the
        # parameters. To use it correctly in a LoRA context, we must run it on
        # the trainable network module, NOT the frozen base model.
        # ====================================================================

        # Enhanced logging from original method
        logger.info(f"Model type: {type(transformer).__name__}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Network dimension: {args.network_dim}")

        # Simple and Correct Fix:
        # Use the already prepared trainable_params. The Fira optimizer will apply its
        # logic to these parameters. We will add the fira-specific args to the groups.
        param_groups = trainable_params

        # Fira-specific parameters (similar to FiraPT method)
        fira_params = {
            "rank": args.network_dim,
            "update_proj_gap": 50,
            "alpha": 1.0,
            "proj_type": "std",
        }

        # Add Fira-specific parameters to each group
        for group in param_groups:
            if isinstance(group, dict):
                group.update(fira_params)  # type: ignore

        logger.info(f"Fira parameters configured: {fira_params}")
        logger.info(
            f"Configuring Fira with {len(param_groups)} trainable parameter groups."
        )

        # Enhanced parameter group logging (from original method)
        total_params = 0
        if param_groups:
            for i, group in enumerate(param_groups):
                if isinstance(group, dict) and "params" in group:
                    # Handle the params correctly - they might be a dict_values object
                    params_obj = group.get("params", [])

                    # Convert to list of parameters if it's a dict_values or dict
                    if hasattr(params_obj, "values"):
                        # It's a dict or dict_values object
                        params_list = list(params_obj.values())
                    elif isinstance(params_obj, list):
                        # It's already a list
                        params_list = params_obj
                    else:
                        # Fallback
                        params_list = (
                            list(params_obj)
                            if hasattr(params_obj, "__iter__")
                            else [params_obj]
                        )

                    # Update the group with the corrected params list
                    if isinstance(group, dict):
                        group["params"] = params_list  # type: ignore
                    total_params += len(params_list)

                    logger.info(f"  Group {i}: {len(params_list)} parameters")
                    logger.info(
                        f"  Group {i} keys: {list(group.keys()) if isinstance(group, dict) else 'N/A'}"
                    )

                    # Log parameter types for debugging
                    if params_list:
                        param_types = set(type(p).__name__ for p in params_list)
                        logger.info(f"  Group {i} parameter types: {param_types}")
                else:
                    logger.info(
                        f"  Group {i}: {type(group)} - {len(group) if hasattr(group, '__len__') else 'unknown'}"
                    )

        logger.info(f"Total trainable parameters: {total_params}")

        # Create optimizer with proper arguments
        try:
            optimizer = optimizer_class(param_groups, lr=lr, **optimizer_kwargs)  # type: ignore
            logger.info(f"Successfully created {optimizer_class.__name__} optimizer")
        except Exception as e:
            logger.error(f"Failed to create FiraAdamW optimizer: {e}")
            raise e

        # Initialize Fira-specific state if needed (from original method)
        init_state_method = getattr(optimizer, "init_state", None)
        if init_state_method is not None and callable(init_state_method):
            logger.info("Initializing Fira-specific state...")
            init_state_method()
            logger.info("Fira state initialization completed")

        # Log optimizer state for debugging
        logger.info(f"Optimizer state keys: {list(optimizer.state_dict().keys())}")
        logger.info(f"Optimizer parameter groups: {len(optimizer.param_groups)}")

        return optimizer, {"train_fn": lambda: None, "eval_fn": lambda: None}
