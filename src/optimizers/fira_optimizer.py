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
        """Create a FIRA AdamW optimizer configured for the provided LoRA parameter groups."""
        from vendor.fira.fira_adamw import FiraAdamW

        optimizer_class = FiraAdamW

        # ====================================================================
        # The `transformer` stays frozen; only the LoRA `network` parameters
        # supplied via `trainable_params` should be optimized.
        #
        # Upstream FIRA examples assume full-model fine tuning and call
        # `divide_params` to build parameter groups. In our LoRA workflow those
        # groups are prepared by the caller, so we simply attach the Fira
        # settings to the incoming groups before constructing the optimizer.
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

        # Determine target device from provided parameter groups
        target_device = None
        for group in param_groups:
            params = None
            if isinstance(group, dict):
                params = group.get("params")
            elif hasattr(group, '__iter__'):
                params = group
            if not params:
                continue
            first_param = next(iter(params), None)
            if isinstance(first_param, torch.nn.Parameter):
                target_device = first_param.device
                break

        # Create optimizer with proper arguments
        try:
            optimizer = optimizer_class(param_groups, lr=lr, **optimizer_kwargs)  # type: ignore
            logger.info(f"Successfully created {optimizer_class.__name__} optimizer")
        except Exception as e:
            logger.error(f"Failed to create FiraAdamW optimizer: {e}")
            raise e

        if target_device is not None:
            FiraOptimizerManager.fix_fira_device_state(optimizer, target_device)

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


    @staticmethod
    def fix_fira_device_state(optimizer: Any, target_device: torch.device) -> None:
        """Align cached projector state with the provided device and vendored implementation."""
        from vendor.fira.gradient_projection import GradientProjector

        if optimizer is None or not hasattr(optimizer, "state"):
            return

        for group_index, group in enumerate(getattr(optimizer, "param_groups", [])):
            if not isinstance(group, dict):
                continue
            rank = group.get("rank")
            if rank is None:
                continue
            params = group.get("params") or []

            for param in params:
                if param is None:
                    continue

                state = optimizer.state.get(param)  # type: ignore[index]
                if not isinstance(state, dict):
                    continue

                projector = state.get("projector")
                if projector is None:
                    continue

                logger.debug(
                    "Checking FIRA projector for group %s param %s: %s",
                    group_index,
                    hex(id(param)),
                    f"{type(projector).__module__}.{type(projector).__name__}",
                )

                if not isinstance(projector, GradientProjector):
                    try:
                        old_projector = projector
                        projector = GradientProjector(
                            rank,
                            update_proj_gap=group.get("update_proj_gap", 200),
                            alpha=group.get("alpha", 1.0),
                            proj_type=group.get("proj_type", "std"),
                        )
                        if hasattr(old_projector, "ortho_matrix"):
                            projector.ortho_matrix = getattr(old_projector, "ortho_matrix")
                        state["projector"] = projector
                        logger.debug(
                            "Replaced checkpointed projector %s with vendored implementation",
                            hex(id(old_projector)),
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(f"Failed to replace projector: {exc}")
                        continue

                ortho = getattr(projector, "ortho_matrix", None)
                if ortho is None:
                    continue

                try:
                    if isinstance(ortho, list):
                        projector.ortho_matrix = [
                            matrix.to(device=target_device) if isinstance(matrix, torch.Tensor) else matrix
                            for matrix in ortho
                        ]
                    elif isinstance(ortho, torch.Tensor):
                        projector.ortho_matrix = ortho.to(device=target_device)
                    logger.debug("Synchronized projector matrices to %s", target_device)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"Failed to move projector matrices to {target_device}: {exc}")

                align_method = getattr(projector, "_align_ortho_matrix", None)
                if callable(align_method):
                    dummy = torch.empty(0, device=target_device)
                    align_method(dummy)
                    logger.debug("Invoked _align_ortho_matrix for projector %s", hex(id(projector)))


