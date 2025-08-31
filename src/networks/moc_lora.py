"""
MoC-LoRA: Mixture of Contexts LoRA implementation for Takenoko.
Integrates MoC sparse attention with LoRA training for efficient long-context video processing.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, cast

import logging
from common.logger import get_logger

# Import base LoRA implementation
from networks.lora_wan import LoRAModule, LoRANetwork

# Import MoC attention (with fallback)
try:
    from modules.moc_attention import MoCAttention, is_moc_available

    MOC_AVAILABLE = is_moc_available()
except ImportError:
    MOC_AVAILABLE = False

logger = get_logger(__name__, level=logging.INFO)

if not MOC_AVAILABLE:
    logger.warning(
        "🚨 MoC attention not available. MoC-LoRA will use standard attention."
    )
else:
    logger.info("✅ MoC attention is available and ready.")

WAN_ATTENTION_REPLACE_BLOCKS: list[str] = ["WanAttentionBlock"]


class MoCLoRAModule(LoRAModule):
    """
    LoRA module enhanced with Mixture of Contexts sparse attention.

    Extends standard LoRA with:
    1. Sparse attention routing for long context efficiency
    2. Content-aligned chunking for video sequences
    3. Progressive sparsification during training
    4. Graceful fallback to standard LoRA if MoC unavailable
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        split_dims: Optional[List[int]] = None,
        # MoC-specific parameters
        moc_chunk_size: int = 1024,
        moc_top_k: int = 5,
        moc_enable_causality: bool = True,
        moc_progressive_sparsify: bool = False,
        moc_context_dropout: float = 0.0,
        moc_implementation: str = "optimized",  # "original" or "optimized"
        # Memory control parameters
        moc_max_layers: Optional[int] = None,
        moc_max_modules: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize MoC-LoRA module.

        Args:
            Standard LoRA parameters (same as LoRAModule)
            moc_chunk_size: Size of content-aligned chunks
            moc_top_k: Number of chunks each query attends to
            moc_enable_causality: Whether to enforce causal attention
            moc_progressive_sparsify: Whether to progressively increase sparsity
            moc_context_dropout: Dropout rate for context regularization
        """
        # Initialize base LoRA
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            split_dims,
            **kwargs,
        )

        # Store MoC parameters
        self.moc_chunk_size = moc_chunk_size
        self.moc_top_k = moc_top_k
        self.moc_enable_causality = moc_enable_causality
        self.moc_progressive_sparsify = moc_progressive_sparsify
        self.moc_context_dropout = moc_context_dropout
        self.moc_implementation = moc_implementation

        # Store memory control parameters
        self.moc_max_layers = moc_max_layers
        self.moc_max_modules = moc_max_modules

        # Debug logging for module initialization
        logger.debug(f"Initializing MoCLoRAModule for {lora_name}:")
        logger.debug(f"  Module class: {org_module.__class__.__name__}")
        logger.debug(f"  MOC_AVAILABLE: {MOC_AVAILABLE}")
        logger.debug(f"  Module type check: {self._is_attention_layer(org_module)}")

        # DEBUGGING: Re-enable MoC module detection but with limited scope
        should_use_moc = MOC_AVAILABLE and self._is_attention_layer(org_module)
        passes_memory_filter = self._passes_memory_filter(lora_name)

        # Production: Enable MoC for all qualifying modules (respects your config limits)
        self.use_moc = should_use_moc and passes_memory_filter

        if self.use_moc:
            logger.debug(f"✅ Enabling MoC for {lora_name}")

        # Initialize MoC attention for qualifying modules
        if self.use_moc:
            try:
                logger.debug(f"🔧 Initializing MoC attention for {lora_name}")
                self._initialize_moc_attention(org_module)
                logger.debug(f"✅ MoC attention initialized for {lora_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize MoC for {lora_name}: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.use_moc = False

        # Training state for progressive sparsification
        self.training_step = 0
        self.sparsification_schedule = None
        if self.moc_progressive_sparsify:
            self._setup_sparsification_schedule()

    def _passes_memory_filter(self, lora_name: str) -> bool:
        """Check if this module passes memory filtering constraints."""
        if self.moc_max_layers is None and self.moc_max_modules is None:
            return True  # No filtering - all modules pass

        # Extract layer information from module name
        parts = lora_name.split("_")
        if (
            len(parts) >= 5
            and parts[2] == "unet"
            and parts[3] == "blocks"
            and parts[4].isdigit()
        ):
            layer_idx = int(parts[4])

            # Check layer limit
            if self.moc_max_layers is not None and layer_idx >= self.moc_max_layers:
                return False

            # Count how many modules from this layer we've already seen
            # This is a simplified check - in practice, the network-level filtering
            # will be more accurate, but this prevents MoC init for obvious cases
            module_name = "_".join(parts[5:]) if len(parts) > 5 else "unknown"
            attention_modules = [
                "self_attn_q",
                "self_attn_k",
                "self_attn_v",
                "self_attn_o",
                "cross_attn_q",
                "cross_attn_k",
                "cross_attn_v",
                "cross_attn_o",
            ]

            if self.moc_max_modules is not None:
                # Find position of this module in the typical attention module order
                try:
                    module_position = attention_modules.index(module_name)
                    if module_position >= self.moc_max_modules:
                        return False
                except ValueError:
                    # Module not in standard list - conservatively allow it
                    pass

        return True

    def _is_attention_layer(self, module: nn.Module) -> bool:
        """Check if module is an attention layer that can benefit from MoC."""
        module_name = module.__class__.__name__
        lora_name_lower = self.lora_name.lower()

        # Check if this is an attention-related module by class name
        attention_keywords = ["attention", "attn", "multihead"]
        class_is_attention = any(
            keyword in module_name.lower() for keyword in attention_keywords
        )

        # Check if this is an attention-related module by LoRA name
        name_is_attention = any(
            keyword in lora_name_lower for keyword in attention_keywords
        )

        # Check if it's a large linear layer that could benefit from MoC
        is_large_linear = False
        if isinstance(module, nn.Linear):
            # Use MoC for large linear layers (likely transformer components)
            input_dim = module.in_features
            output_dim = module.out_features
            # More strict heuristic: use MoC only for very large linear layers
            # AND only if they're likely attention-related based on naming
            if input_dim >= 1024 and output_dim >= 1024:
                attention_like_names = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "qkv",
                    "proj",
                ]
                is_attention_like = any(
                    name in lora_name_lower for name in attention_like_names
                )
                if is_attention_like:
                    is_large_linear = True

        return class_is_attention or name_is_attention or is_large_linear

    def _initialize_moc_attention(self, org_module: nn.Module):
        """Initialize MoC attention for the original module."""
        # Extract attention parameters from original module
        if hasattr(org_module, "attention"):
            attention = org_module.attention
            dim = getattr(
                attention, "embed_dim", getattr(attention, "hidden_size", 768)
            )
            num_heads = getattr(attention, "num_heads", 8)
        elif isinstance(org_module, nn.Linear):
            # For linear layers, infer parameters
            dim = org_module.in_features
            num_heads = max(1, dim // 64)  # Heuristic: 64 dims per head
        else:
            # Default parameters
            dim = 768
            num_heads = 8

        # Create MoC attention
        self.moc_attention = MoCAttention(
            dim=dim,
            num_heads=num_heads,
            chunk_size=self.moc_chunk_size,
            top_k=self.moc_top_k,
            enable_causality=self.moc_enable_causality,
            context_dropout=self.moc_context_dropout,
            implementation=self.moc_implementation,
        )

    def _setup_sparsification_schedule(self):
        """Setup progressive sparsification schedule following the original exactly."""
        # Original's exact training: "chunk size gradually decreasing from 10240, 5120, 2560 to 1280, and top-k=5"
        self.sparsification_schedule = {
            "chunk_sizes": [10240, 5120, 2560, 1280],  # Original's exact sequence
            "top_k": 5,  # Original uses constant top-k=5 throughout
            "phase_steps": 500,  # Steps per phase (adjust based on training length)
        }

    def update_training_step(self, step: int):
        """Update training step for progressive sparsification following the original exactly."""
        self.training_step = step

        if (
            self.use_moc
            and self.moc_progressive_sparsify
            and self.sparsification_schedule
        ):
            schedule = self.sparsification_schedule
            chunk_sizes = schedule["chunk_sizes"]
            target_top_k = schedule["top_k"]
            phase_steps = schedule["phase_steps"]

            # Determine current phase (0-3 for 4 phases)
            current_phase = min(step // phase_steps, len(chunk_sizes) - 1)
            current_chunk_size = chunk_sizes[current_phase]
            current_top_k = target_top_k  # Original uses constant top-k=5

            # Update MoC attention parameters
            if hasattr(self, "moc_attention"):
                if (
                    self.moc_attention.chunk_size != current_chunk_size
                    or self.moc_attention.top_k != current_top_k
                ):
                    logger.info(
                        f"Progressive sparsification step {step} (phase {current_phase}): "
                        f"chunk_size {self.moc_attention.chunk_size} -> {current_chunk_size}, "
                        f"top_k {self.moc_attention.top_k} -> {current_top_k}"
                    )

                self.moc_attention.chunk_size = current_chunk_size
                self.moc_attention.top_k = current_top_k

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with optional MoC attention.

        Args:
            x: Input tensor
            **kwargs: Additional arguments (attention_mask, token_type_ids, etc.)

        Returns:
            Output tensor
        """
        # Get original module output
        org_output = self.org_forward(x)

        # Apply LoRA adaptation
        if self.split_dims is None:
            lora_output = self._compute_lora_output(x, **kwargs)
        else:
            lora_output = self._compute_split_lora_output(x, **kwargs)

        # Combine original and LoRA outputs
        return org_output + lora_output * self.multiplier * self.scale

    def _compute_lora_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute LoRA output with optional MoC attention."""
        # Standard LoRA computation
        lx = self.lora_down(x)

        # Apply dropout if specified
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # Apply rank dropout if specified
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale / (1.0 - self.rank_dropout)
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        # Apply MoC attention if available and appropriate
        if self.use_moc and hasattr(self, "moc_attention") and self.training:
            try:
                import time

                start_time = time.time()

                # Apply MoC attention
                lx = self.moc_attention(lx, **kwargs)

                elapsed = time.time() - start_time
                # Concise logging: one line per block with timing
                logger.debug(f"🧠 MoC: {self.lora_name} completed in {elapsed:.2f}s")

                # Track statistics for phase-end reporting
                if not hasattr(self.__class__, "_moc_stats"):
                    self.__class__._moc_stats = {
                        "total_time": 0,
                        "call_count": 0,
                        "modules": set(),
                    }
                self.__class__._moc_stats["total_time"] += elapsed
                self.__class__._moc_stats["call_count"] += 1
                self.__class__._moc_stats["modules"].add(self.lora_name)

            except Exception as e:
                logger.error(f"❌ MoC failed for {self.lora_name}: {e}")
                # Fallback to regular LoRA on error

        return lx

    def _compute_split_lora_output(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute split LoRA output (for split_dims case)."""
        lxs = [lora_down(x) for lora_down in self.lora_down]

        # Apply dropout and rank dropout
        if self.dropout is not None and self.training:
            lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

        if self.rank_dropout is not None and self.training:
            masks = [
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
                for lx in lxs
            ]
            for i in range(len(lxs)):
                if len(lxs[i].size()) == 3:
                    masks[i] = masks[i].unsqueeze(1)
                elif len(lxs[i].size()) == 4:
                    masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                lxs[i] = lxs[i] * masks[i]
            scale = self.scale / (1.0 - self.rank_dropout)
        else:
            scale = self.scale

        lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

        return torch.cat(lxs, dim=-1)


class MoCLoRANetwork(LoRANetwork):
    """
    MoC-LoRA network that manages multiple MoC-LoRA modules.

    Extends LoRANetwork with:
    1. MoC-specific parameter management
    2. Progressive sparsification coordination
    3. Training state synchronization
    4. Performance monitoring
    """

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        # MoC-specific parameters
        moc_chunk_size: int = 1024,
        moc_top_k: int = 5,
        moc_enable_causality: bool = True,
        moc_progressive_sparsify: bool = False,
        moc_context_dropout: float = 0.0,
        moc_implementation: str = "optimized",  # "original" or "optimized"
        # Memory control parameters
        moc_max_layers: Optional[int] = None,
        moc_max_modules: Optional[int] = None,
        **kwargs,
    ):
        """Initialize MoC-LoRA network."""

        # Store MoC parameters
        self.moc_chunk_size = moc_chunk_size
        self.moc_top_k = moc_top_k
        self.moc_enable_causality = moc_enable_causality
        self.moc_progressive_sparsify = moc_progressive_sparsify
        self.moc_context_dropout = moc_context_dropout
        self.moc_implementation = moc_implementation

        # Store memory control parameters
        self.moc_max_layers = moc_max_layers
        self.moc_max_modules = moc_max_modules

        # Log memory control settings early
        if moc_max_layers is not None:
            logger.info(
                f"🎯 MEMORY: Limiting MoC-LoRA to first {moc_max_layers} layers for memory control"
            )
        if moc_max_modules is not None:
            logger.info(
                f"🎯 MEMORY: Limiting MoC-LoRA to first {moc_max_modules} modules per layer for memory control"
            )

        # Initialize base LoRA network with MoC module class
        # We need to customize the module creation to pass MoC parameters
        super().__init__(
            target_replace_modules=target_replace_modules,
            prefix=prefix,
            text_encoders=text_encoders,
            unet=unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            module_class=self._create_moc_lora_module,  # Use custom factory
            **kwargs,
        )

        # Log MoC configuration
        moc_enabled_count = sum(
            1 for lora in self.unet_loras if hasattr(lora, "use_moc") and lora.use_moc
        )

        total_modules = len(self.unet_loras)

        # Critical check: If MoC-LoRA is explicitly requested but no modules use MoC, fail training
        if total_modules == 0:
            error_msg = (
                "❌ CRITICAL ERROR: No LoRA modules were created!\n"
                "\n🔍 This means your WAN model doesn't contain any modules matching:\n"
                f"   {WAN_ATTENTION_REPLACE_BLOCKS}\n"
                "\n🛠️  Solutions:\n"
                "   1. Check your model structure - it should contain 'WanAttentionBlock' modules\n"
                "   2. If your model uses different attention module names, update WAN_TARGET_REPLACE_MODULES\n"
                "   3. Use standard LoRA instead: network_module = 'networks.lora_wan'\n"
                "\n💡 Common WAN attention module names: WanAttentionBlock, Attention, SelfAttention\n"
                "\n🚨 Training stopped to prevent silent failure."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        elif moc_enabled_count == 0:
            # TEMPORARILY DISABLED: Allow training without MoC for debugging
            logger.warning(
                f"🚨 DEBUG MODE: MoC-LoRA requested but no modules using MoC attention.\n"
                f"📊 Found {total_modules} LoRA modules, 0 using MoC.\n"
                f"🔧 This is expected while MoC attention is disabled for debugging.\n"
                f"🚀 Training will continue with standard LoRA behavior."
            )
            # Original error (commented out for debugging):
            # error_msg = (
            #     "❌ CRITICAL ERROR: MoC-LoRA was requested but no modules are using MoC attention!\n"
            #     f"\n📊 Found {total_modules} LoRA modules, but 0 are using MoC.\n"
            #     "\n🔍 This usually means:\n"
            #     "   • Your modules don't match MoC attention criteria\n"
            #     "   • MoC attention initialization failed\n"
            #     "\n🛠️  Solutions:\n"
            #     "   1. Set logging_level = 'DEBUG' to see detailed module analysis\n"
            #     "   2. Check that your modules have attention layers or are large linear layers\n"
            #     "   3. Verify MoC attention imports are working\n"
            #     "\n🚨 Training stopped to prevent silent fallback to standard LoRA."
            # )
            # logger.error(error_msg)
            # raise RuntimeError(error_msg)

        logger.info(
            f"☄️ INFO: MoC-LoRA Network initialized: {moc_enabled_count}/{total_modules} modules using MoC"
        )

        if moc_enabled_count > 0:
            logger.info(
                f"☄️ INFO: MoC Configuration: chunk_size={moc_chunk_size}, top_k={moc_top_k}, "
                f"causality={moc_enable_causality}, progressive={moc_progressive_sparsify}"
            )

    def _create_moc_lora_module(
        self, lora_name, org_module, multiplier, lora_dim, alpha, **kwargs
    ):
        """Custom factory method to create MoCLoRAModule with additional parameters."""
        return MoCLoRAModule(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            # MoC-specific parameters
            moc_chunk_size=self.moc_chunk_size,
            moc_top_k=self.moc_top_k,
            moc_enable_causality=self.moc_enable_causality,
            moc_progressive_sparsify=self.moc_progressive_sparsify,
            moc_context_dropout=self.moc_context_dropout,
            moc_implementation=self.moc_implementation,
            # Memory control parameters
            moc_max_layers=self.moc_max_layers,
            moc_max_modules=self.moc_max_modules,
            **kwargs,
        )

    def _apply_memory_filtering(self):
        """Apply memory filtering to limit number of MoC modules."""
        if self.moc_max_layers is None and self.moc_max_modules is None:
            return  # No filtering requested

        original_count = len(self.unet_loras)

        # Group modules by layer (based on module name pattern)
        layer_modules = {}
        for lora in self.unet_loras:
            # Extract layer number from module name like "moc_lora_unet_blocks_0_..."
            parts = lora.lora_name.split("_")
            if (
                len(parts) >= 5
                and parts[2] == "unet"
                and parts[3] == "blocks"
                and parts[4].isdigit()
            ):
                layer_idx = int(parts[4])
                if layer_idx not in layer_modules:
                    layer_modules[layer_idx] = []
                layer_modules[layer_idx].append(lora)
            else:
                # Put non-layer modules in a special group
                if -1 not in layer_modules:
                    layer_modules[-1] = []
                layer_modules[-1].append(lora)

        filtered_modules = []
        layers_processed = 0

        # Process layers in order
        for layer_idx in sorted(layer_modules.keys()):
            if (
                self.moc_max_layers is not None
                and layers_processed >= self.moc_max_layers
            ):
                logger.info(
                    f"🎯 MEMORY: Skipping layer {layer_idx} (max_layers={self.moc_max_layers})"
                )
                break

            layer_loras = layer_modules[layer_idx]

            # Apply per-layer module limit
            if self.moc_max_modules is not None:
                layer_loras = layer_loras[: self.moc_max_modules]
                if len(layer_modules[layer_idx]) > self.moc_max_modules:
                    logger.info(
                        f"🎯 MEMORY: Layer {layer_idx}: Limited to {self.moc_max_modules}/{len(layer_modules[layer_idx])} modules"
                    )

            filtered_modules.extend(layer_loras)
            if layer_idx >= 0:  # Don't count the special -1 group as a layer
                layers_processed += 1

        # Update the module list
        self.unet_loras = filtered_modules

        filtered_count = len(self.unet_loras)
        if filtered_count < original_count:
            logger.warning(
                f"🎯 MEMORY: Filtered modules from {original_count} to {filtered_count} "
                f"(max_layers={self.moc_max_layers}, max_modules={self.moc_max_modules})"
            )
            logger.info(f"🎯 MEMORY: This reduces RAM usage during MoC initialization")

    def _apply_pre_moc_memory_filtering(self):
        """Apply memory filtering before MoC initialization to prevent RAM overload."""
        if self.moc_max_layers is None and self.moc_max_modules is None:
            return  # No filtering requested

        original_count = len(self.unet_loras)
        logger.info(
            f"🎯 MEMORY: Applying pre-MoC filtering on {original_count} modules"
        )

        # Check if any modules have already been initialized with MoC (they shouldn't at this stage)
        moc_initialized = sum(
            1
            for lora in self.unet_loras
            if hasattr(lora, "use_moc") and hasattr(lora, "moc_attention")
        )
        if moc_initialized > 0:
            logger.warning(
                f"🎯 MEMORY: Warning - {moc_initialized} modules already have MoC initialized!"
            )

        # Group modules by layer (based on module name pattern)
        layer_modules = {}
        for lora in self.unet_loras:
            # Extract layer number from module name like "moc_lora_unet_blocks_0_..."
            parts = lora.lora_name.split("_")
            if (
                len(parts) >= 5
                and parts[2] == "unet"
                and parts[3] == "blocks"
                and parts[4].isdigit()
            ):
                layer_idx = int(parts[4])
                if layer_idx not in layer_modules:
                    layer_modules[layer_idx] = []
                layer_modules[layer_idx].append(lora)
            else:
                # Put non-layer modules in a special group
                if -1 not in layer_modules:
                    layer_modules[-1] = []
                layer_modules[-1].append(lora)

        filtered_modules = []
        layers_processed = 0

        # Process layers in order
        for layer_idx in sorted(layer_modules.keys()):
            if (
                self.moc_max_layers is not None
                and layers_processed >= self.moc_max_layers
            ):
                logger.info(
                    f"🎯 MEMORY: Skipping layer {layer_idx} (max_layers={self.moc_max_layers}) - preventing MoC init"
                )
                break

            layer_loras = layer_modules[layer_idx]

            # Apply per-layer module limit
            if self.moc_max_modules is not None:
                layer_loras = layer_loras[: self.moc_max_modules]
                if len(layer_modules[layer_idx]) > self.moc_max_modules:
                    logger.info(
                        f"🎯 MEMORY: Layer {layer_idx}: Limited to {self.moc_max_modules}/{len(layer_modules[layer_idx])} modules - preventing MoC init"
                    )

            filtered_modules.extend(layer_loras)
            if layer_idx >= 0:  # Don't count the special -1 group as a layer
                layers_processed += 1

        # Update the module list BEFORE MoC initialization happens
        self.unet_loras = filtered_modules

        filtered_count = len(self.unet_loras)
        if filtered_count < original_count:
            logger.warning(
                f"🎯 MEMORY: Pre-filtered modules from {original_count} to {filtered_count} "
                f"(max_layers={self.moc_max_layers}, max_modules={self.moc_max_modules})"
            )
            logger.info(
                f"🎯 MEMORY: Prevented MoC initialization for {original_count - filtered_count} modules - major RAM savings!"
            )

    def update_training_step(self, step: int):
        """Update training step for all MoC modules."""
        for lora in self.unet_loras:
            if hasattr(lora, "update_training_step"):
                lora.update_training_step(step)

    def get_moc_statistics(self) -> Dict[str, float]:
        """Get MoC usage statistics."""
        stats = {
            "total_modules": len(self.unet_loras),
            "moc_enabled": 0,
            "avg_chunk_size": 0,
            "avg_top_k": 0,
            "sparsity_ratio": 0,
        }

        moc_modules = []
        for lora in self.unet_loras:
            if hasattr(lora, "use_moc") and lora.use_moc:
                stats["moc_enabled"] += 1
                if hasattr(lora, "moc_attention"):
                    moc_modules.append(lora.moc_attention)

        if moc_modules:
            stats["avg_chunk_size"] = sum(m.chunk_size for m in moc_modules) / len(
                moc_modules
            )
            stats["avg_top_k"] = sum(m.top_k for m in moc_modules) / len(moc_modules)
            # Estimate sparsity: (top_k * avg_chunk_size) / total_sequence_length
            # This is a rough estimate - actual sparsity depends on sequence length
            estimated_chunks = 180000 // stats["avg_chunk_size"]  # Assume 180k tokens
            stats["sparsity_ratio"] = 1.0 - (
                stats["avg_top_k"] / max(estimated_chunks, 1)
            )

        return stats

    def log_moc_statistics(self, step: int = 0):
        """Log current MoC statistics."""
        stats = self.get_moc_statistics()
        if stats["moc_enabled"] > 0:
            logger.info(
                f"MoC Stats [Step {step}]: "
                f"{stats['moc_enabled']}/{stats['total_modules']} modules active, "
                f"chunk_size={stats['avg_chunk_size']:.0f}, "
                f"top_k={stats['avg_top_k']:.1f}, "
                f"estimated_sparsity={stats['sparsity_ratio']:.1%}"
            )

    @classmethod
    def log_moc_phase_stats(cls):
        """Log MoC performance statistics for the current phase."""
        if hasattr(cls, "_moc_stats") and cls._moc_stats["call_count"] > 0:
            stats = cls._moc_stats
            avg_time = stats["total_time"] / stats["call_count"]
            logger.info(
                f"📊 MoC Phase Stats: {len(stats['modules'])} modules, {stats['call_count']} calls, "
                f"avg {avg_time:.3f}s/call, total {stats['total_time']:.2f}s"
            )
            # Reset stats for next phase
            cls._moc_stats = {"total_time": 0, "call_count": 0, "modules": set()}

    def set_training_step(self, step: int):
        """Update the training step for all MoC modules (progressive sparsification)."""
        for lora in self.unet_loras:
            if hasattr(lora, "moc_attention") and lora.moc_attention is not None:
                lora.moc_attention.set_training_step(step)

        # Log phase changes
        if hasattr(self, "_last_logged_step"):
            schedule = {0: 10240, 500: 5120, 1000: 2560, 1500: 1280}
            for threshold, chunk_size in schedule.items():
                if self._last_logged_step < threshold <= step:
                    logger.info(
                        f"📈 MoC Phase Change: Step {step} → chunk_size={chunk_size}"
                    )
        else:
            self._last_logged_step = -1

        self._last_logged_step = step


# Factory functions for creating MoC-LoRA networks
def create_moc_lora_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> Union[MoCLoRANetwork, LoRANetwork]:
    """Create MoC-LoRA network with default parameters."""

    if not MOC_AVAILABLE:
        logger.warning("MoC not available. Creating standard LoRA network instead.")
        from networks.lora_wan import create_arch_network

        return create_arch_network(
            multiplier,
            network_dim,
            network_alpha,
            vae,
            text_encoders,
            unet,
            neuron_dropout,
            **kwargs,
        )

    # Parse MoC-specific arguments
    moc_chunk_size = int(kwargs.pop("moc_chunk_size", 1024))
    moc_top_k = int(kwargs.pop("moc_top_k", 5))
    moc_enable_causality = (
        str(kwargs.pop("moc_enable_causality", "true")).lower() == "true"
    )
    moc_progressive_sparsify = (
        str(kwargs.pop("moc_progressive_sparsify", "false")).lower() == "true"
    )
    moc_context_dropout = float(kwargs.pop("moc_context_dropout", 0.0))
    moc_implementation = str(kwargs.pop("moc_implementation", "optimized"))

    # Memory control parameters
    moc_max_layers = kwargs.pop("moc_max_layers", None)
    moc_max_modules = kwargs.pop("moc_max_modules", None)

    if moc_max_layers is not None:
        moc_max_layers = int(moc_max_layers)
        logger.info(
            f"🎯 MEMORY: Limiting MoC-LoRA to first {moc_max_layers} layers for memory control"
        )

    if moc_max_modules is not None:
        moc_max_modules = int(moc_max_modules)
        logger.info(
            f"🎯 MEMORY: Limiting MoC-LoRA to first {moc_max_modules} modules per layer for memory control"
        )

    # Set default values
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    # Create MoC-LoRA network (only pass supported LoRANetwork parameters)
    lora_kwargs = {
        "target_replace_modules": WAN_ATTENTION_REPLACE_BLOCKS,
        "prefix": "moc_lora_unet",
        "text_encoders": text_encoders,
        "unet": unet,
        "multiplier": multiplier,
        "lora_dim": network_dim,
        "alpha": network_alpha,
        "dropout": neuron_dropout,
        # Pass only supported LoRANetwork parameters from kwargs
        "rank_dropout": kwargs.get("rank_dropout"),
        "module_dropout": kwargs.get("module_dropout"),
        "conv_lora_dim": kwargs.get("conv_lora_dim"),
        "conv_alpha": kwargs.get("conv_alpha"),
        "modules_dim": kwargs.get("modules_dim"),
        "modules_alpha": kwargs.get("modules_alpha"),
        # MoC-specific parameters
        "moc_chunk_size": moc_chunk_size,
        "moc_top_k": moc_top_k,
        "moc_enable_causality": moc_enable_causality,
        "moc_progressive_sparsify": moc_progressive_sparsify,
        "moc_context_dropout": moc_context_dropout,
        "moc_implementation": moc_implementation,
        # Memory control parameters
        "moc_max_layers": moc_max_layers,
        "moc_max_modules": moc_max_modules,
    }

    # Remove None values to avoid issues (but keep required parameters even if None)
    required_params = {"target_replace_modules", "prefix", "text_encoders", "unet"}
    lora_kwargs = {
        k: v for k, v in lora_kwargs.items() if v is not None or k in required_params
    }

    network = MoCLoRANetwork(**lora_kwargs)

    return network


# Alias for compatibility with existing network loading system
def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> Union[MoCLoRANetwork, LoRANetwork]:
    """Alias for create_moc_lora_network for compatibility."""
    return create_moc_lora_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        **kwargs,
    )
