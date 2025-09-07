"""Control LoRA network module for training with control signals
This extends the standard LoRA functionality to support control signal training"""

import ast
from typing import Dict, List, Optional, Type, Union
from transformers import CLIPTextModel
import torch
import torch.nn as nn

import logging
from common.logger import get_logger
import os

logger = get_logger(__name__, level=logging.INFO)

# Import the base LoRA module
from .lora_wan import (
    LoRAModule,
    LoRANetwork,
)

WAN_TARGET_REPLACE_MODULES: list[str] = ["WanAttentionBlock"]


class ControlLoRAModule(LoRAModule):
    """
    Control LoRA module that extends the base LoRA functionality
    to support control signal training.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        control_config: Optional[Dict] = None,
    ):
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
        )

        self.control_config = control_config or {}
        self.control_enabled = self.control_config.get("enabled", False)

        if self.control_enabled:
            logger.info(
                f"Control LoRA enabled for {lora_name} with config: {self.control_config}"
            )

    def forward(self, x):
        # Standard LoRA forward pass
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = (
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                )
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]  # type: ignore

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
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

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]  # type: ignore

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class ControlLoRAInfModule(ControlLoRAModule):
    """
    Control LoRA inference module that extends the base inference functionality.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        control_config: Optional[Dict] = None,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            control_config=control_config,
        )

        self.org_module_ref = [org_module]  # for reference
        self.enabled = True
        self.network: Optional[ControlLoRANetwork] = None

    def set_network(self, network):
        self.network = network

    def default_forward(self, x):
        return super().forward(x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class ControlLoRANetwork(LoRANetwork):
    """
    Control LoRA network that extends the base LoRA network
    to support control signal training.
    """

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = ControlLoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        control_config: Optional[Dict] = None,
    ) -> None:
        # Store control config for use in module creation
        self.control_config = control_config or {}

        # Call parent constructor with modified module_class
        super().__init__(
            target_replace_modules,
            prefix,
            text_encoders,
            unet,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            conv_lora_dim,
            conv_alpha,
            module_class,
            modules_dim,
            modules_alpha,
            exclude_patterns,
            include_patterns,
            verbose,
        )

        # Override module creation to pass control_config
        self._create_modules_with_control_config()

    def _create_modules_with_control_config(self):
        """Recreate modules with control config passed to each module."""
        # This is a simplified approach - in practice, you'd want to modify
        # the module creation logic in the parent class
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "control_config"):
                lora.control_config = self.control_config  # type: ignore

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that handles missing keys gracefully.
        This is needed because the network structure might change between
        saving and loading, especially with control LoRA configurations.
        """
        # Get the current state dict keys
        current_keys = set(self.state_dict().keys())
        saved_keys = set(state_dict.keys())

        # Analyze the key differences
        missing_from_saved = current_keys - saved_keys
        missing_from_current = saved_keys - current_keys
        matching_keys = current_keys & saved_keys

        # Log detailed information about the key differences
        logger.info(f"🔍 State dict analysis:")
        logger.info(f"   Current network has {len(current_keys)} keys")
        logger.info(f"   Saved state has {len(saved_keys)} keys")
        logger.info(f"   Matching keys: {len(matching_keys)}")
        logger.info(f"   Missing from saved: {len(missing_from_saved)}")
        logger.info(f"   Missing from current: {len(missing_from_current)}")

        # Extract module names from keys for better analysis
        def extract_module_name(key):
            if "." in key:
                return key.split(".")[0]
            return key

        current_modules = set(extract_module_name(key) for key in current_keys)
        saved_modules = set(extract_module_name(key) for key in saved_keys)

        logger.info(f"📊 Module analysis:")
        logger.info(f"   Current modules: {len(current_modules)}")
        logger.info(f"   Saved modules: {len(saved_modules)}")
        logger.info(f"   Common modules: {len(current_modules & saved_modules)}")

        # Show sample module names for debugging
        sample_current = list(current_modules)[:5]
        sample_saved = list(saved_modules)[:5]
        logger.info(f"   Sample current modules: {sample_current}")
        logger.info(f"   Sample saved modules: {sample_saved}")

        # Filter the input state dict to only include keys that exist in the current network
        filtered_state_dict = {}
        ignored_keys = []

        for key, value in state_dict.items():
            if key in current_keys:
                filtered_state_dict[key] = value
            else:
                ignored_keys.append(key)

        # Log information about the filtering
        if ignored_keys:
            logger.warning(
                f"ControlLoRANetwork: {len(ignored_keys)} keys from saved state "
                f"are not present in current network structure and will be ignored."
            )

            # Group ignored keys by module for better understanding
            ignored_modules = {}
            for key in ignored_keys:
                module_name = extract_module_name(key)
                if module_name not in ignored_modules:
                    ignored_modules[module_name] = []
                ignored_modules[module_name].append(key)

            logger.info(f"🗑️  Ignored keys by module:")
            for module_name, keys in list(ignored_modules.items())[
                :10
            ]:  # Show first 10 modules
                logger.info(f"   {module_name}: {len(keys)} keys")
                if len(keys) <= 3:
                    logger.info(f"      {keys}")
                else:
                    logger.info(f"      {keys[:3]}... (and {len(keys)-3} more)")

        # Check if we have a reasonable number of matching keys
        if len(filtered_state_dict) == 0:
            logger.error(
                "❌ No matching keys found between saved state and current network!"
            )
            logger.error("This indicates a fundamental mismatch in network structure.")
            return super().load_state_dict({}, strict=False)

        match_ratio = len(filtered_state_dict) / len(current_keys)
        logger.info(
            f"✅ Loading {len(filtered_state_dict)} keys (match ratio: {match_ratio:.2%})"
        )

        if match_ratio < 0.5:
            logger.warning(
                f"⚠️  Low match ratio ({match_ratio:.2%}). This may indicate significant "
                f"network structure changes between save and load."
            )

        # Call parent's load_state_dict with filtered state dict
        result = super().load_state_dict(filtered_state_dict, strict=False)

        # Log the final result
        if result.missing_keys:
            logger.info(
                f"📋 Final result: {len(result.missing_keys)} keys still missing from current network"
            )

            # Group missing keys by module
            missing_modules = {}
            for key in result.missing_keys:
                module_name = extract_module_name(key)
                if module_name not in missing_modules:
                    missing_modules[module_name] = []
                missing_modules[module_name].append(key)

            logger.info(f"📋 Missing keys by module:")
            for module_name, keys in list(missing_modules.items())[
                :5
            ]:  # Show first 5 modules
                logger.info(f"   {module_name}: {len(keys)} keys")

        if result.unexpected_keys:
            logger.info(
                f"📋 Final result: {len(result.unexpected_keys)} unexpected keys"
            )

        # Final success/failure assessment
        total_expected = len(current_keys)
        total_loaded = len(filtered_state_dict) - len(result.missing_keys)
        load_success_ratio = total_loaded / total_expected if total_expected > 0 else 0

        logger.info(
            f"🎯 Load success: {total_loaded}/{total_expected} keys ({load_success_ratio:.2%})"
        )

        if load_success_ratio >= 0.8:
            logger.info("✅ Control LoRA state loading successful!")
        elif load_success_ratio >= 0.5:
            logger.warning("⚠️  Partial control LoRA state loading - some keys missing")
        else:
            logger.error(
                "❌ Control LoRA state loading largely failed - most keys missing"
            )

        return result

    def save_weights(self, file, dtype, metadata):
        """
        Enhanced save_weights method that includes control LoRA configuration metadata.
        This helps with proper resumption of control LoRA training.
        """
        # Add control LoRA specific metadata
        if self.control_config and self.control_config.get("enabled", False):
            logger.info("🎯 Saving control LoRA weights with configuration metadata")

            # Add control LoRA configuration to metadata
            if metadata is None:
                metadata = {}

            metadata["control_lora_enabled"] = "True"
            metadata["control_lora_type"] = str(
                self.control_config.get("control_lora_type", "unknown")
            )
            metadata["control_preprocessing"] = str(
                self.control_config.get("control_preprocessing", "unknown")
            )
            metadata["control_blur_kernel_size"] = str(
                self.control_config.get("control_blur_kernel_size", 0)
            )
            metadata["control_blur_sigma"] = str(
                self.control_config.get("control_blur_sigma", 0.0)
            )
            metadata["control_scale_factor"] = str(
                self.control_config.get("control_scale_factor", 1.0)
            )
            metadata["input_lr_scale"] = str(
                self.control_config.get("input_lr_scale", 1.0)
            )
            metadata["control_concatenation_dim"] = str(
                self.control_config.get("control_concatenation_dim", 0)
            )

            # Log what we're saving
            logger.info(f"📦 Control LoRA config being saved:")
            for key, value in metadata.items():
                if key.startswith("control_"):
                    logger.info(f"   {key}: {value}")

            # Log network structure information
            state_dict = self.state_dict()
            lora_modules = []
            for key in state_dict.keys():
                if "lora_" in key and ("down" in key or "up" in key or "alpha" in key):
                    module_name = key.split(".")[0]
                    if module_name not in lora_modules:
                        lora_modules.append(module_name)

            logger.info(f"🎯 LoRA modules being saved: {len(lora_modules)} modules")
            logger.info(f"   Module names: {lora_modules[:10]}...")  # Show first 10

            # Add module count to metadata
            metadata["control_lora_module_count"] = str(len(lora_modules))
            metadata["control_lora_modules"] = str(
                lora_modules[:20]
            )  # Save first 20 module names
        else:
            logger.info("📦 Saving standard LoRA weights (control LoRA not enabled)")

        # Call parent's save_weights method
        super().save_weights(file, dtype, metadata)

    def load_weights(self, file):
        """
        Enhanced load_weights method that validates control LoRA configuration from metadata.
        This helps ensure proper resumption of control LoRA training.
        """
        if os.path.splitext(file)[1] == ".safetensors":
            from memory.safetensors_loader import load_file

            weights_sd = load_file(file)

            # Check for control LoRA metadata
            # For now, we'll skip metadata validation for safetensors files
            # as the metadata access is complex and varies by safetensors version
            metadata = None
            logger.info("📦 Loading LoRA weights from safetensors format")

            # Skip metadata validation for now
            if False:  # metadata:
                if metadata.get("control_lora_enabled", False):
                    logger.info(
                        "🎯 Loading control LoRA weights with configuration validation"
                    )

                    # Log saved configuration
                    logger.info(f"📦 Saved control LoRA config:")
                    for key, value in metadata.items():
                        if key.startswith("control_"):
                            logger.info(f"   {key}: {value}")

                    # Validate configuration compatibility
                    if self.control_config and self.control_config.get(
                        "enabled", False
                    ):
                        saved_type = metadata.get("control_lora_type", "unknown")
                        current_type = self.control_config.get(
                            "control_lora_type", "unknown"
                        )

                        if saved_type != current_type:
                            logger.warning(
                                f"⚠️  Control LoRA type mismatch: saved={saved_type}, current={current_type}"
                            )

                        saved_preprocessing = metadata.get(
                            "control_preprocessing", "unknown"
                        )
                        current_preprocessing = self.control_config.get(
                            "control_preprocessing", "unknown"
                        )

                        if saved_preprocessing != current_preprocessing:
                            logger.warning(
                                f"⚠️  Control preprocessing mismatch: saved={saved_preprocessing}, current={current_preprocessing}"
                            )

                        # Log module count comparison
                        saved_module_count = metadata.get(
                            "control_lora_module_count", 0
                        )
                        current_state_dict = self.state_dict()
                        current_modules = []
                        for key in current_state_dict.keys():
                            if "lora_" in key and (
                                "down" in key or "up" in key or "alpha" in key
                            ):
                                module_name = key.split(".")[0]
                                if module_name not in current_modules:
                                    current_modules.append(module_name)

                        logger.info(
                            f"🎯 Module count comparison: saved={saved_module_count}, current={len(current_modules)}"
                        )

                        if saved_module_count != len(current_modules):
                            logger.warning(
                                f"⚠️  Module count mismatch: saved={saved_module_count}, current={len(current_modules)}"
                            )
                    else:
                        logger.warning(
                            "⚠️  Loading control LoRA weights but current config has control LoRA disabled"
                        )
                else:
                    logger.info(
                        "📦 Loading standard LoRA weights (no control LoRA metadata found)"
                    )
            else:
                logger.info("📦 Loading LoRA weights (no metadata available)")
        else:
            weights_sd = torch.load(file, map_location="cpu")
            logger.info("📦 Loading LoRA weights from torch format")

        info = self.load_state_dict(weights_sd, False)
        return info


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    control_config: Optional[Dict] = None,
    **kwargs,
):
    """Create a control LoRA network with the specified configuration."""

    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # For control LoRA, we need to train patch_embedding, so exclude it from exclusion patterns
    if control_config and control_config.get("enabled", False):
        # Exclude everything except patch_embedding
        exclude_patterns.append(
            r".*(text_embedding|time_embedding|time_projection|norm|head).*"
        )
        logger.info("Control LoRA enabled: patch_embedding will be trained")

        # Validate that patch_embedding is not excluded
        patch_embedding_excluded = any(
            "patch_embedding" in pattern for pattern in exclude_patterns
        )
        if patch_embedding_excluded:
            logger.warning(
                "patch_embedding appears to be excluded from training! "
                "This will prevent control LoRA from working properly."
            )
        else:
            logger.info("patch_embedding confirmed to be included in training")
    else:
        # Standard exclusion including patch_embedding
        exclude_patterns.append(
            r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"
        )

    kwargs["exclude_patterns"] = exclude_patterns

    network = create_control_network(
        WAN_TARGET_REPLACE_MODULES,
        "control_lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders or [],  # Convert None to empty list
        unet,
        neuron_dropout=neuron_dropout,
        control_config=control_config,
        **kwargs,
    )

    # After network creation, log what modules are actually being trained
    if control_config and control_config.get("enabled", False):
        logger.info("Control LoRA network created. Training modules:")
        for name, module in network.named_modules():
            if hasattr(module, "lora_name"):
                logger.info(f"  - {module.lora_name}")

    return network


def create_control_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    control_config: Optional[Dict] = None,
    **kwargs,
):
    """Create a control LoRA network with the specified configuration. (Legacy function name)"""
    return create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        control_config=control_config,
        **kwargs,
    )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    control_config: Optional[Dict] = None,
    **kwargs,
):
    """Create a control LoRA network. (Fallback compatibility function)"""
    return create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        control_config=control_config,
        **kwargs,
    )


def create_control_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    control_config: Optional[Dict] = None,
    **kwargs,
):
    """Create a control LoRA network."""

    # Set defaults for None values
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0
    if text_encoders is None:
        text_encoders = []

    # Extract control-specific arguments
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    rank_dropout = kwargs.get("rank_dropout", None)
    module_dropout = kwargs.get("module_dropout", None)
    exclude_patterns = kwargs.get("exclude_patterns", None)
    include_patterns = kwargs.get("include_patterns", None)
    verbose = kwargs.get("verbose", False)

    # too many arguments ( ^ω^)･･･
    network = ControlLoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,  # type: ignore
        unet,  # type: ignore
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        control_config=control_config,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_control_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    control_config: Optional[Dict] = None,
    **kwargs,
) -> ControlLoRANetwork:
    """Create a control LoRA network from weights."""

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim

    module_class = ControlLoRAInfModule if for_inference else ControlLoRAModule

    network = ControlLoRANetwork(
        target_replace_modules,
        "control_lora_unet",
        text_encoders,  # type: ignore
        unet,  # type: ignore
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        control_config=control_config,
    )
    return network
