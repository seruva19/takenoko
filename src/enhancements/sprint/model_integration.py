"""
Sprint Model Integration Helpers

Helper functions for integrating Sprint sparse-dense fusion into WanModel.
Keeps model.py clean by encapsulating Sprint-specific logic here.
"""

import logging
from typing import Optional, List, Dict, Any
import torch.nn as nn

from common.logger import get_logger
from .exceptions import (
    SprintImportError,
    SprintConfigurationError,
    SprintCompatibilityError,
    SprintModelStateError,
)
from .device_utils import (
    validate_sprint_device_compatibility,
    optimize_memory_for_sprint,
)
from .sparse_dense_fusion import create_sprint_fusion
from .training_scheduler import SprintTrainingScheduler
from .config_parser import validate_sprint_block_partitioning

logger = get_logger(__name__, level=logging.INFO)


def setup_sprint_fusion(
    model: nn.Module,
    token_drop_ratio: float = 0.75,
    encoder_layers: Optional[int] = None,
    middle_layers: Optional[int] = None,
    sampling_strategy: str = "temporal_coherent",
    path_drop_prob: float = 0.1,
    partitioning_strategy: str = "percentage",
    encoder_ratio: float = 0.25,
    middle_ratio: float = 0.50,
    use_learnable_mask_token: bool = False,
    enable_uncond_path_drop: bool = False,
) -> Optional[nn.Module]:
    """
    Create and initialize Sprint sparse-dense fusion module for a WanModel.

    Args:
        model: WanModel instance with .blocks, .dim, .device attributes
        token_drop_ratio: Ratio of tokens to drop (default 0.75 for 75% dropping)
        encoder_layers: Number of encoder blocks (if None, auto-calculate based on strategy)
        middle_layers: Number of middle blocks (if None, auto-calculate based on strategy)
        sampling_strategy: Token sampling strategy - "uniform", "temporal_coherent", "spatial_coherent"
        path_drop_prob: Probability of replacing sparse path with MASK during training (default 0.1)
        partitioning_strategy: "percentage" (default) or "fixed" (2-N-2 style)
        encoder_ratio: Ratio of blocks to use as encoder (default 0.25 for percentage strategy)
        middle_ratio: Ratio of blocks to use as middle (default 0.50 for percentage strategy)

    Returns:
        SparseDenseFusion module ready for use, or None if setup fails
    """
    try:
        total_blocks = len(model.blocks)  # type: ignore
        dim = model.dim  # type: ignore
        device = model.device  # type: ignore

        # Validate Sprint compatibility with model configuration
        if hasattr(model, "rope_on_the_fly") and model.rope_on_the_fly:  # type: ignore
            raise ValueError(
                "Sprint requires cached rotary position embeddings (RoPE). "
                "The model is configured with rope_on_the_fly=True, which is incompatible. "
                "Please set rope_on_the_fly=False in your config to use Sprint."
            )

        # Create fusion module (supports both manual and automatic partitioning)
        sprint_fusion = create_sprint_fusion(
            dim=dim,
            num_layers=total_blocks,
            token_drop_ratio=token_drop_ratio,
            sampling_strategy=sampling_strategy,
            encoder_ratio=encoder_ratio,
            middle_ratio=middle_ratio,
            path_drop_prob=path_drop_prob,
            encoder_layers=encoder_layers,  # Manual override (if specified)
            middle_layers=middle_layers,  # Manual override (if specified)
            partitioning_strategy=partitioning_strategy,
            use_learnable_mask_token=use_learnable_mask_token,
        ).to(device)
        sprint_fusion.enable_uncond_path_drop = enable_uncond_path_drop

        return sprint_fusion

    except Exception as e:
        logger.error(f"Failed to setup Sprint fusion: {e}")
        return None


def can_use_sprint(
    sprint_fusion: Optional[nn.Module],
    is_training: bool,
    skip_block_indices: Optional[List[int]],
    sparse_attention: bool,
    controlnet_states: Optional[List],
    tread_routes: Optional[List],
    blocks_to_swap: Optional[List] = None,
) -> bool:
    """
    Check if Sprint can be used for the current forward pass.

    Sprint is incompatible with:
    - Inference mode (Sprint is training-only)
    - CPU offloading (Sprint bypasses standard block iteration loop)
    - TREAD routing (dynamic token selection conflicts)
    - ControlNet injection (requires per-block hooks)
    - Block skipping (Sprint manages block iteration)
    - Sparse attention patches (local patching conflicts with Sprint sampling)

    Args:
        sprint_fusion: Sprint fusion module (or None if not enabled)
        is_training: Whether model is in training mode
        skip_block_indices: List of block indices to skip
        sparse_attention: Whether sparse attention (Nabla/STA) is enabled
        controlnet_states: ControlNet conditioning states
        tread_routes: TREAD routing configuration
        blocks_to_swap: CPU offloading configuration (if set, Sprint is incompatible)

    Returns:
        True if Sprint can be safely used, False otherwise
    """
    if sprint_fusion is None:
        return False

    if not is_training and not getattr(sprint_fusion, "enable_uncond_path_drop", False):
        # Sprint is training-only unless explicitly enabled for evaluation path drop
        return False

    # Check for incompatible features
    has_routing = tread_routes is not None and len(tread_routes) > 0
    has_skip = skip_block_indices is not None and len(skip_block_indices) > 0
    has_controlnet = controlnet_states is not None
    has_cpu_offloading = blocks_to_swap is not None and len(blocks_to_swap) > 0

    if has_routing:
        return False
    if has_skip:
        return False
    if has_controlnet:
        return False
    if sparse_attention:
        # Sprint has its own token sampling, conflicts with Nabla/STA sparse attention
        return False
    if has_cpu_offloading:
        # Sprint bypasses standard block loop, incompatible with CPU offloading
        logger.warning(
            "Sprint is incompatible with CPU offloading (blocks_to_swap). "
            "Disabling Sprint for this forward pass."
        )
        return False

    return True


def apply_sprint_forward(
    sprint_fusion: nn.Module,
    x,
    blocks,
    kwargs: dict,
    batch_idx: Optional[int] = None,
    global_step: Optional[int] = None,
    drop_ratio: float = 0.75,
    stage_name: Optional[str] = None,
    force_path_drop: bool = False,
):
    """
    Apply Sprint sparse-dense fusion forward pass with diagnostics.

    Args:
        sprint_fusion: Sprint fusion module
        x: Input tensor [B, L, C]
        blocks: Model blocks (nn.ModuleList)
        kwargs: Dict containing e, context, seq_lens, grid_sizes, freqs, context_lens, sparse_attention
        batch_idx: Optional batch index for deterministic token sampling
        global_step: Current global training step (for diagnostics)
        drop_ratio: Current token drop ratio (for diagnostics)
        stage_name: Training stage name (for diagnostics)

    Returns:
        Output tensor after Sprint processing, or raises exception on failure
    """
    # Execute Sprint forward pass first
    output = sprint_fusion(
        x=x,
        blocks=blocks,
        e=kwargs["e"],
        context=kwargs["context"],
        seq_lens=kwargs["seq_lens"],
        grid_sizes=kwargs["grid_sizes"],
        freqs=kwargs["freqs"],
        context_lens=kwargs["context_lens"],
        sparse_attention=False,  # Sprint has own sampling
        batched_rotary=None,
        batch_idx=batch_idx,
        force_path_drop=force_path_drop,
    )

    # Record diagnostics AFTER forward pass (so _last_sparse_seq_lens is available)
    try:
        from .diagnostics import get_diagnostics

        diagnostics = get_diagnostics()
        if diagnostics and diagnostics.enabled:
            # Get sparse seq_lens from fusion module (now set during forward pass)
            sparse_seq_lens = None
            if hasattr(sprint_fusion, "_last_sparse_seq_lens"):
                sparse_seq_lens = sprint_fusion._last_sparse_seq_lens

            from .diagnostics import record_sprint_step

            record_sprint_step(
                sprint_active=True,
                drop_ratio=drop_ratio,
                seq_lens=kwargs["seq_lens"],
                sparse_seq_lens=sparse_seq_lens,
                stage_name=stage_name,
                global_step=global_step,
            )
    except Exception:
        pass  # Don't let diagnostics break training

    return output


def create_sprint_scheduler(
    pretrain_steps: int = 0,
    finetune_steps: int = 0,
    initial_drop_ratio: float = 0.75,
    warmup_steps: int = 0,
    cooldown_steps: int = 100,
) -> Optional[SprintTrainingScheduler]:
    """
    Create Sprint two-stage training scheduler.

    This scheduler manages the transition from token-dropping pretraining to
    full-token fine-tuning.

    Args:
        pretrain_steps: Number of steps for token-dropping pretraining (if 0, no scheduler)
        finetune_steps: Number of steps for full-token fine-tuning (if 0, no fine-tuning)
        initial_drop_ratio: Token drop ratio during pretraining (default 0.75)
        warmup_steps: Steps to linearly warmup drop ratio from 0 → initial_drop_ratio
        cooldown_steps: Steps to linearly cooldown drop ratio during transition

    Returns:
        SprintTrainingScheduler if pretrain_steps > 0, else None

    Usage in training loop:
        # 1. Create scheduler during setup (if Sprint two-stage training enabled)
        sprint_scheduler = None
        if args.enable_sprint and args.sprint_pretrain_steps > 0:
            sprint_scheduler = create_sprint_scheduler(
                pretrain_steps=args.sprint_pretrain_steps,
                finetune_steps=args.sprint_finetune_steps,
                initial_drop_ratio=args.sprint_token_drop_ratio,
                warmup_steps=args.sprint_warmup_steps,
                cooldown_steps=args.sprint_cooldown_steps,
            )

        # 2. Update token drop ratio at each training step
        for step in range(total_steps):
            if sprint_scheduler is not None:
                current_drop_ratio = sprint_scheduler.get_drop_ratio(step)
                # Update Sprint fusion module's token sampler
                transformer.sprint_fusion.token_sampler.keep_ratio = 1.0 - current_drop_ratio

                # Log stage transitions
                if sprint_scheduler.should_log_transition(step):
                    stage = sprint_scheduler.get_stage_name(step)
                    logger.info(f"Sprint stage: {stage} at step {step}")

            # ... rest of training step ...
    """
    if pretrain_steps <= 0:
        logger.info("Sprint two-stage training disabled (pretrain_steps=0)")
        return None

    scheduler = SprintTrainingScheduler(
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        initial_drop_ratio=initial_drop_ratio,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )

    logger.info(
        f"Sprint two-stage scheduler created: "
        f"{pretrain_steps} pretrain steps → {finetune_steps} finetune steps"
    )

    return scheduler


def update_sprint_drop_ratio(
    transformer: nn.Module,
    sprint_scheduler: Optional[SprintTrainingScheduler],
    global_step: int,
) -> None:
    """
    Update Sprint token drop ratio based on training schedule.

    Call this at the beginning of each training step to automatically adjust
    the token drop ratio according to the two-stage training schedule.

    Args:
        transformer: WanModel with sprint_fusion attribute
        sprint_scheduler: Sprint training scheduler (or None if disabled)
        global_step: Current global training step

    Usage in training loop:
        for step in range(total_steps):
            # Update Sprint drop ratio for this step
            update_sprint_drop_ratio(transformer, sprint_scheduler, global_step)

            # ... rest of training step ...
    """
    if sprint_scheduler is None:
        return

    # Get drop ratio for current step
    current_drop_ratio = sprint_scheduler.get_drop_ratio(global_step)

    # Update transformer's Sprint fusion module
    if hasattr(transformer, "sprint_fusion") and transformer.sprint_fusion is not None:
        transformer.sprint_fusion.token_sampler.keep_ratio = 1.0 - current_drop_ratio

        # Log stage transitions
        if sprint_scheduler.should_log_transition(global_step):
            stage = sprint_scheduler.get_stage_name(global_step)
            logger.info(
                f"Sprint stage transition: {stage} at step {global_step} "
                f"(drop_ratio={current_drop_ratio:.3f})"
            )


# New helper functions for improved error handling and state management


def handle_sprint_import_error(import_error: Exception, logger: logging.Logger) -> None:
    """
    Handle Sprint import errors with helpful user guidance.

    Args:
        import_error: The import exception that occurred
        logger: Logger instance for error reporting
    """
    error_msg = str(import_error).lower()

    if "no module named" in error_msg:
        if "enhancements.sprint" in error_msg:
            logger.error("❌ Sprint module not found.")
            logger.error("Solution: Ensure enhancements.sprint is in your Python path.")
            logger.error(
                "Check that src/ is in PYTHONPATH or install the package properly."
            )
        else:
            missing_module = error_msg.split("no module named")[-1].strip().strip("'\"")
            logger.error(f"❌ Missing dependency: {missing_module}")
            logger.error(f"Solution: Install with: pip install {missing_module}")
    else:
        logger.error(f"❌ Sprint import failed: {import_error}")
        logger.error("Solution: Check your Python environment and dependencies.")

    logger.error(
        "⚠️ Sprint will be disabled. Training will continue without Sprint optimization."
    )
    logger.error("For help, see: docs/enhancements/sprint/INTEGRATION_GUIDE.md")


def validate_sprint_model_compatibility(model: nn.Module) -> None:
    """
    Comprehensive validation of model compatibility with Sprint.

    Args:
        model: The model instance to validate

    Raises:
        SprintCompatibilityError: If model is incompatible with Sprint
        SprintConfigurationError: If model configuration is invalid
    """
    # Check basic model attributes
    required_attrs = ["blocks", "dim", "device"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise SprintCompatibilityError(
                f"Model missing required attribute '{attr}' for Sprint integration",
                incompatible_feature=f"missing_{attr}",
                alternative="Ensure model has proper attributes before Sprint setup",
            )

    # Validate device compatibility
    device = getattr(model, "device", None)
    if device is not None:
        validate_sprint_device_compatibility(device)

    # Validate model size
    blocks = getattr(model, "blocks", [])
    if len(blocks) < 4:
        raise SprintConfigurationError(
            f"Model too small for Sprint: {len(blocks)} blocks. Minimum 4 blocks required.",
            config_key="model_size",
            config_value=len(blocks),
            expected=">= 4 blocks",
        )

    # Check for incompatible features
    if hasattr(model, "rope_on_the_fly") and getattr(model, "rope_on_the_fly", False):
        raise SprintCompatibilityError(
            "Sprint requires cached rotary position embeddings. rope_on_the_fly=True is incompatible.",
            incompatible_feature="rope_on_the_fly",
            alternative="Set rope_on_the_fly=False in model configuration",
        )

    # Optimize memory for Sprint
    optimize_memory_for_sprint(device)


def get_sprint_status_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive Sprint status information for debugging.

    Args:
        model: Model instance to check

    Returns:
        Dictionary with Sprint status information
    """
    status = {
        "sprint_enabled": False,
        "sprint_fusion_loaded": False,
        "import_error": None,
        "compatibility_issues": [],
        "model_info": {},
        "configuration": {},
    }

    # Check if Sprint is enabled
    if hasattr(model, "_sprint_enabled"):
        status["sprint_enabled"] = getattr(model, "_sprint_enabled", False)

    # Check if fusion module is loaded
    if (
        hasattr(model, "sprint_fusion")
        and getattr(model, "sprint_fusion", None) is not None
    ):
        status["sprint_fusion_loaded"] = True
        fusion = model.sprint_fusion

        # Get fusion configuration
        if hasattr(fusion, "token_sampler"):
            token_sampler = fusion.token_sampler
            status["configuration"] = {
                "sampling_strategy": getattr(token_sampler, "strategy", "unknown"),
                "keep_ratio": getattr(token_sampler, "keep_ratio", None),
                "drop_ratio": 1.0 - getattr(token_sampler, "keep_ratio", 0.0),
            }

    # Check for import errors
    if hasattr(model, "_sprint_import_error"):
        status["import_error"] = getattr(model, "_sprint_import_error", None)

    # Get model information
    if hasattr(model, "blocks"):
        status["model_info"]["total_blocks"] = len(model.blocks)
    if hasattr(model, "dim"):
        status["model_info"]["hidden_size"] = getattr(model, "dim", None)
    if hasattr(model, "device"):
        status["model_info"]["device"] = str(getattr(model, "device", None))

    # Check compatibility
    try:
        validate_sprint_model_compatibility(model)
    except (SprintCompatibilityError, SprintConfigurationError) as e:
        status["compatibility_issues"].append(str(e))

    return status


def safe_sprint_setup(model: nn.Module, **kwargs) -> Optional[nn.Module]:
    """
    Safely setup Sprint with comprehensive error handling and validation.

    Args:
        model: Model instance to setup Sprint for
        **kwargs: Arguments to pass to setup_sprint_fusion

    Returns:
        Sprint fusion module if successful, None otherwise
    """
    try:
        # Validate model compatibility first
        validate_sprint_model_compatibility(model)

        # Extract and validate block partitioning
        total_blocks = len(model.blocks)
        encoder_layers = kwargs.get("encoder_layers")
        middle_layers = kwargs.get("middle_layers")

        validate_sprint_block_partitioning(encoder_layers, middle_layers, total_blocks)

        # Setup Sprint fusion
        fusion = setup_sprint_fusion(model, **kwargs)

        if fusion is not None:
            logger.info("✅ Sprint setup completed successfully")
            return fusion
        else:
            logger.error("❌ Sprint setup failed during fusion creation")
            return None

    except (SprintCompatibilityError, SprintConfigurationError) as e:
        logger.error(f"❌ Sprint compatibility error: {e}")
        if e.details.get("alternative"):
            logger.error(f"💡 Suggestion: {e.details['alternative']}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error during Sprint setup: {e}")
        logger.error("This may indicate a bug or configuration issue.")
        logger.error("Please report this issue with your configuration details.")
        return None


def enable_sprint_with_validation(
    model: nn.Module,
    token_drop_ratio: float = 0.75,
    encoder_layers: Optional[int] = None,
    middle_layers: Optional[int] = None,
    sampling_strategy: str = "temporal_coherent",
    path_drop_prob: float = 0.1,
    partitioning_strategy: str = "percentage",
    encoder_ratio: float = 0.25,
    middle_ratio: float = 0.50,
    use_learnable_mask_token: bool = False,
    enable_uncond_path_drop: bool = False,
) -> bool:
    """
    Enable Sprint with comprehensive validation and error handling.
    This function handles ALL Sprint setup logic and state management.

    Args:
        model: Model instance to enable Sprint for
        token_drop_ratio: Ratio of tokens to drop (default 0.75 for 75% dropping)
        encoder_layers: Number of encoder blocks (if None, auto-calculate based on strategy)
        middle_layers: Number of middle blocks (if None, auto-calculate based on strategy)
        sampling_strategy: Token sampling strategy - "uniform", "temporal_coherent", "spatial_coherent"
        path_drop_prob: Probability of replacing sparse path with MASK during training (default 0.1)
        partitioning_strategy: "percentage" (default) or "fixed" (2-N-2 style)
        encoder_ratio: Ratio of blocks to use as encoder (default 0.25 for percentage strategy)
        middle_ratio: Ratio of blocks to use as middle (default 0.50 for percentage strategy)

    Returns:
        True if Sprint was successfully enabled, False otherwise

    Side effects:
        - Sets model.sprint_fusion to the fusion module (or None)
        - Sets model._sprint_enabled to True/False
        - Sets model._sprint_import_error to error details (if any)
    """
    # Initialize Sprint state attributes
    model._sprint_enabled = False
    model._sprint_import_error = None
    model.sprint_fusion = None

    try:
        # Setup Sprint with comprehensive error handling
        fusion = safe_sprint_setup(
            model=model,
            token_drop_ratio=token_drop_ratio,
            encoder_layers=encoder_layers,
            middle_layers=middle_layers,
            sampling_strategy=sampling_strategy,
            path_drop_prob=path_drop_prob,
            partitioning_strategy=partitioning_strategy,
            encoder_ratio=encoder_ratio,
            middle_ratio=middle_ratio,
            use_learnable_mask_token=use_learnable_mask_token,
            enable_uncond_path_drop=enable_uncond_path_drop,
        )

        if fusion is not None:
            model.sprint_fusion = fusion
            model._sprint_enabled = True
            logger.info("✅ Sprint successfully enabled and validated")
            return True
        else:
            logger.warning("⚠️ Sprint setup returned None - Sprint will be disabled")
            return False

    except (SprintCompatibilityError, SprintConfigurationError) as e:
        model._sprint_enabled = False
        model._sprint_import_error = str(e)
        logger.error(f"❌ Sprint compatibility error: {e}")
        if hasattr(e, "details") and e.details.get("alternative"):
            logger.error(f"💡 Suggestion: {e.details['alternative']}")
        return False
    except ImportError as e:
        model._sprint_enabled = False
        model._sprint_import_error = str(e)
        handle_sprint_import_error(e, logger)
        return False
    except Exception as e:
        model._sprint_enabled = False
        model._sprint_import_error = str(e)
        logger.error(f"❌ Unexpected error during Sprint setup: {e}")
        logger.error("This may indicate a bug or configuration issue.")
        return False


def set_sprint_training_state(
    model: nn.Module,
    global_step: Optional[int] = None,
    stage_name: Optional[str] = None,
) -> None:
    """
    Set Sprint training state for diagnostics logging.

    This function should be called from the training loop to enable
    detailed Sprint diagnostics with TensorBoard logging.

    Args:
        model: Model instance with Sprint integration
        global_step: Current global training step
        stage_name: Training stage name ("warmup", "pretrain", "transition", "finetune")

    Example:
        # Import the function
        from enhancements.sprint import set_sprint_training_state

        # In training loop
        for step in range(total_steps):
            # Update Sprint state
            set_sprint_training_state(
                model=model,
                global_step=step,
                stage_name=sprint_scheduler.get_stage_name(step) if sprint_scheduler else None
            )
            # Forward/backward pass
            loss = model(...)
    """
    # Initialize Sprint state attributes if they don't exist
    if not hasattr(model, "_sprint_global_step"):
        model._sprint_global_step = None
    if not hasattr(model, "_sprint_stage_name"):
        model._sprint_stage_name = None

    # Only update state if Sprint is actually enabled
    if hasattr(model, "_sprint_enabled") and model._sprint_enabled:
        model._sprint_global_step = global_step
        model._sprint_stage_name = stage_name
    else:
        # Sprint is not enabled, but still set the attributes for consistency
        model._sprint_global_step = global_step
        model._sprint_stage_name = stage_name


def create_sprint_state_tracker() -> "SprintStateTracker":
    """
    Create a Sprint state tracker for monitoring model state during training.

    Returns:
        SprintStateTracker instance
    """
    return SprintStateTracker()


class SprintStateTracker:
    """
    Tracks Sprint state during training to detect inconsistencies.
    """

    def __init__(self):
        self._last_known_state = {}
        self._state_history = []
        self._anomaly_count = 0

    def record_state(self, model: nn.Module, step: int) -> None:
        """
        Record current Sprint state.

        Args:
            model: Model instance to check
            step: Current training step
        """
        current_state = {
            "step": step,
            "sprint_enabled": getattr(model, "_sprint_enabled", False),
            "sprint_fusion_exists": hasattr(model, "sprint_fusion")
            and model.sprint_fusion is not None,
            "fusion_keep_ratio": None,
        }

        if hasattr(model, "sprint_fusion") and model.sprint_fusion is not None:
            fusion = model.sprint_fusion
            if hasattr(fusion, "token_sampler") and hasattr(
                fusion.token_sampler, "keep_ratio"
            ):
                current_state["fusion_keep_ratio"] = fusion.token_sampler.keep_ratio

        # Check for anomalies
        if self._last_known_state and step > 0:
            if self._detect_state_anomaly(self._last_known_state, current_state):
                self._anomaly_count += 1
                logger.warning(
                    f"🚨 Sprint state anomaly detected at step {step} "
                    f"(total anomalies: {self._anomaly_count})"
                )
                logger.warning(f"Previous state: {self._last_known_state}")
                logger.warning(f"Current state: {current_state}")

        self._last_known_state = current_state.copy()
        self._state_history.append(current_state)

        # Keep only recent history
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-50:]

    def _detect_state_anomaly(self, prev_state: Dict, curr_state: Dict) -> bool:
        """Detect if current state represents an anomaly compared to previous."""
        # Sprint should not spontaneously enable/disable
        if prev_state["sprint_enabled"] != curr_state["sprint_enabled"]:
            return True

        # Fusion should not disappear if Sprint is enabled
        if (
            curr_state["sprint_enabled"]
            and prev_state["sprint_fusion_exists"]
            and not curr_state["sprint_fusion_exists"]
        ):
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked state information."""
        return {
            "total_steps_tracked": len(self._state_history),
            "anomalies_detected": self._anomaly_count,
            "last_known_state": self._last_known_state,
            "recent_states": self._state_history[-10:] if self._state_history else [],
        }
