"""SingLoRA network implementation for WAN models.
Complete implementation with all Takenoko integration points.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import logging

from common.logger import get_logger
logger = get_logger(__name__, level=logging.INFO)

# WAN-specific target modules (must match existing pattern)
WAN_SINGLORA_TARGET_MODULES = ["WanAttentionBlock"]

class SingLoRAModule(nn.Module):
    """
    SingLoRA module with complete Takenoko integration.
    Implements all required methods and patterns from existing LoRA modules.
    """
    
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        rank: int = 4,
        alpha: float = 1.0,
        ramp_up_steps: int = 1000,
        dropout: Optional[float] = None,
        init_method: str = "kaiming_uniform",
        use_rslora: bool = False,
        adapter_name: str = "default",
    ):
        super().__init__()
        self.lora_name = lora_name
        self.org_module = org_module
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.ramp_up_steps = ramp_up_steps
        self.init_method = init_method
        self.use_rslora = use_rslora
        self.adapter_name = adapter_name
        
        # Store original forward method (Takenoko pattern)
        self.org_forward = None
        
        # Get dimensions
        if isinstance(org_module, nn.Linear):
            self.in_features = org_module.in_features
            self.out_features = org_module.out_features
        elif isinstance(org_module, nn.Conv2d):
            self.in_features = org_module.in_channels
            self.out_features = org_module.out_channels
        else:
            raise ValueError(f"Unsupported module type: {type(org_module)}")
            
        # Enhanced non-square matrix handling
        self.larger_dim = max(self.in_features, self.out_features)
        
        # Create single matrix A with larger dimension
        self.A = nn.Parameter(torch.zeros(self.larger_dim, self.rank))
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate scaling factor
        if self.use_rslora:
            self.scaling = self.alpha / math.sqrt(self.rank)
        else:
            self.scaling = self.alpha / self.rank
        
        # Training step counter
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.float32))
        
        # Dropout if specified
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        
        # Required properties for Takenoko compatibility
        self.enabled = True  # Required by set_enabled()
        
    def _initialize_weights(self):
        """Initialize weights with chosen method."""
        if self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        elif self.init_method == "gaussian":
            nn.init.normal_(self.A, std=1.0 / self.rank)
        elif self.init_method == "zeros":
            nn.init.zeros_(self.A)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
            
    def _get_ramp_up_factor(self) -> float:
        """Calculate ramp-up factor u(t) = min(t/T, 1)."""
        return min(self.training_step.item() / self.ramp_up_steps, 1.0)
        
    def _get_update_weight(self) -> torch.Tensor:
        """Enhanced non-square matrix handling from PEFT-SingLoRA."""
        ramp_up_factor = self._get_ramp_up_factor()
        A = self.A  # Shape: (larger_dim, rank)
        
        # Enhanced non-square matrix handling
        if self.in_features == self.out_features:
            # Square matrix: standard A @ A.T
            update = A @ A.T
        else:
            # Non-square matrix: sophisticated approach
            smaller_dim = min(self.in_features, self.out_features)
            A_star = A[:smaller_dim, :]  # Truncated matrix
            
            if self.in_features < self.out_features:
                # Wide matrix case: A* @ A.T gives (d_in, d_out)
                update = A_star @ A.T
            else:
                # Tall matrix case: A @ A*.T gives (d_out, d_in)
                update = A @ A_star.T
                
        return ramp_up_factor * self.scaling * self.multiplier * update
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - this replaces the original module's forward."""
        # Call original forward
        if self.org_forward is not None:
            result = self.org_forward(x)
        else:
            result = self.org_module(x)
            
        # Skip if disabled
        if not self.enabled:
            return result
            
        # Apply dropout if specified
        if self.dropout is not None and self.training:
            x_dropped = self.dropout(x)
        else:
            x_dropped = x
            
        # Get update matrix
        update_weight = self._get_update_weight()
        
        # Apply adaptation based on layer type
        if isinstance(self.org_module, nn.Linear):
            result = result + F.linear(x_dropped, update_weight, None)
        elif isinstance(self.org_module, nn.Conv2d):
            # Reshape for conv2d
            update_weight_conv = update_weight.view(self.org_module.weight.shape)
            result = result + F.conv2d(
                x_dropped, update_weight_conv, None,
                self.org_module.stride, self.org_module.padding
            )
        
        return result
        
    def apply_to(self):
        """Apply SingLoRA by replacing the original module's forward method."""
        # Store original forward method (Takenoko pattern)
        self.org_forward = self.org_module.forward
        # Replace with our forward method
        self.org_module.forward = self.forward
        # Remove reference to avoid circular dependency
        del self.org_module
        
    @property
    def device(self) -> torch.device:
        """Required property for Takenoko compatibility."""
        return next(self.parameters()).device

    @property 
    def dtype(self) -> torch.dtype:
        """Required property for Takenoko compatibility."""
        return next(self.parameters()).dtype
        
    def update_global_step(self, global_step: int):
        """Update global step for coordinated ramp-up."""
        self.training_step.fill_(global_step)

class SingLoRANetworkWan(nn.Module):
    """
    Complete SingLoRA network with all required Takenoko methods.
    Implements exact interface expected by training core.
    """
    
    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Optional[List[nn.Module]],
        unet: nn.Module,
        multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Store required attributes (Takenoko pattern)
        self.multiplier = multiplier
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        
        # Parse network_args if provided in kwargs
        network_args = kwargs.get("network_args", [])
        if isinstance(network_args, dict):
            config = network_args
        else:
            config = self._parse_network_args(network_args)
        
        # Extract configuration with defaults
        self.rank = config.get("rank", 4)
        self.alpha = config.get("alpha", 4.0)
        self.ramp_up_steps = config.get("ramp_up_steps", 1000)
        self.dropout = config.get("dropout", None)
        self.init_method = config.get("init_method", "kaiming_uniform")
        self.use_rslora = config.get("use_rslora", False)
        self.target_modules = self._parse_target_modules(
            config.get("target_modules", target_replace_modules)
        )
        
        # Store model references
        self.text_encoders = text_encoders
        self.unet = unet
        
        # Create LoRA modules (following Takenoko pattern)
        self.text_encoder_loras: List[SingLoRAModule] = []  # Empty for WAN
        self.unet_loras: List[SingLoRAModule] = []
        
        self._create_singlora_modules()
        
        logger.info(f"Created SingLoRA network with {len(self.unet_loras)} modules")
        
    def _parse_network_args(self, network_args: List[str]) -> Dict[str, Any]:
        """Parse Takenoko network_args format: ["key=value", "key=value", ...]"""
        config = {}
        
        for arg in network_args:
            if isinstance(arg, str) and "=" in arg:
                key, value = arg.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Type conversion
                if value.lower() in ["true", "false"]:
                    config[key] = value.lower() == "true"
                elif value.lower() in ["null", "none"]:
                    config[key] = None
                elif value.replace(".", "").replace("-", "").isdigit():
                    config[key] = float(value) if "." in value else int(value)
                elif value.startswith("[") and value.endswith("]"):
                    # Parse array values
                    items = value[1:-1].split(",")
                    config[key] = [item.strip().strip('"\'') for item in items if item.strip()]
                else:
                    config[key] = value
                    
        return config
        
    def _parse_target_modules(self, target_modules) -> List[str]:
        """Parse target modules into list format."""
        if isinstance(target_modules, str):
            return [target_modules]
        elif isinstance(target_modules, list):
            return target_modules
        else:
            return self.target_replace_modules
            
    def _create_singlora_modules(self):
        """Create SingLoRA modules for target layers (following Takenoko pattern)."""
        for name, module in self.unet.named_modules():
            if self._is_target_module(name, module):
                singlora_module = SingLoRAModule(
                    lora_name=name,
                    org_module=module,
                    multiplier=self.multiplier,
                    rank=self.rank,
                    alpha=self.alpha,
                    ramp_up_steps=self.ramp_up_steps,
                    dropout=self.dropout,
                    init_method=self.init_method,
                    use_rslora=self.use_rslora,
                )
                self.unet_loras.append(singlora_module)
                
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should have SingLoRA applied."""
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
            
        module_type = module.__class__.__name__
        return any(target in module_type for target in self.target_modules)
        
    # REQUIRED: Network lifecycle methods called by training core
    
    def on_epoch_start(self, unet):
        """Called at the start of each epoch by training core."""
        self.train()  # Set to training mode
        
    def on_step_start(self):
        """Called at the start of each training step by training core."""
        pass  # Hook for per-step logic
        
    def on_step_start_with_global_step(self, global_step: int):
        """Enhanced on_step_start that includes SingLoRA global step update."""
        self.on_step_start()  # Call original method
        self.update_global_step(global_step)  # Update SingLoRA global step
        
    def apply_to(self, text_encoders, unet, apply_text_encoder=False, apply_unet=True):
        """Apply SingLoRA modules to the model (called by model manager)."""
        if apply_text_encoder:
            logger.info(f"enable SingLoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable SingLoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        # Apply all LoRA modules and add them as submodules
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)
            
    def prepare_optimizer_params(self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs):
        """Prepare parameters for optimizer (called by training setup)."""
        self.requires_grad_(True)
        all_params = []
        
        # Group parameters by module (following Takenoko pattern)
        for lora in self.unet_loras:
            for name, param in lora.named_parameters():
                param_data = {"params": [param], "lr": unet_lr}
                all_params.append(param_data)
                
        return all_params
        
    # REQUIRED: Network management methods
    
    def set_multiplier(self, multiplier):
        """Set multiplier for all modules (called by training logic)."""
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier
            
    def set_enabled(self, is_enabled):
        """Enable/disable all LoRA modules (Takenoko compatibility)."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled
            
    # REQUIRED: Weight management methods
    
    def save_weights(self, file, dtype, metadata):
        """Save SingLoRA weights (following Takenoko pattern)."""
        if metadata is not None and len(metadata) == 0:
            metadata = None
            
        state_dict = self.state_dict()
        
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
                
        if file.endswith(".safetensors"):
            from safetensors.torch import save_file
            from utils import model_utils
            
            # Add model hashes (Takenoko pattern)
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            metadata["singlora_version"] = "1.0"
            
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
            
    def load_weights(self, file):
        """Load SingLoRA weights (following Takenoko pattern)."""
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
            
        info = self.load_state_dict(weights_sd, False)
        return info
        
    # OPTIONAL: Enhanced methods for training integration
    
    def update_global_step(self, global_step: int):
        """Update global step for all SingLoRA modules."""
        for lora in self.unet_loras:
            lora.update_global_step(global_step)
            
    def prepare_network(self, args):
        """Called after network creation (Takenoko pattern)."""
        pass  # Hook for additional setup
        
    def prepare_grad_etc(self, unet):
        """Prepare gradients (Takenoko pattern)."""
        self.requires_grad_(True)
        
    def get_trainable_params(self):
        """Get trainable parameters (Takenoko compatibility)."""
        return self.parameters()
        
    def is_mergeable(self):
        """Check if network can be merged (Takenoko compatibility)."""
        return True

    # OPTIONAL: Training integration utilities
    
    @staticmethod
    def safe_update_singlora_global_step(network: Any, global_step: int) -> bool:
        """
        Safely update SingLoRA global step.
        Can be called from on_step_start() or training loop.
        
        Args:
            network: Network instance (should be SingLoRA network)
            global_step: Current training step
            
        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            if hasattr(network, 'update_global_step') and callable(network.update_global_step):
                network.update_global_step(global_step)
                return True
        except Exception as e:
            logger.debug(f"Failed to update SingLoRA step: {e}")
        return False

# REQUIRED: Factory functions following Takenoko patterns

def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: Optional[nn.Module],
    text_encoders: Optional[List[nn.Module]],
    transformer: nn.Module,
    neuron_dropout: Optional[float] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> SingLoRANetworkWan:
    """
    Create SingLoRANetworkWan with standard WAN target modules.
    EXACT signature matching existing Takenoko networks.
    """
    # Convert parameters to network_args format for internal use
    network_args = {}
    if network_dim is not None:
        network_args["rank"] = network_dim
    if network_alpha is not None:
        network_args["alpha"] = network_alpha
    if neuron_dropout is not None:
        network_args["dropout"] = neuron_dropout
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if key not in ["network_args"]:  # Avoid double-adding
            network_args[key] = value
    
    logger.info(f"Creating SingLoRA network with rank={network_dim}, alpha={network_alpha}")
    
    network = SingLoRANetworkWan(
        target_replace_modules=WAN_SINGLORA_TARGET_MODULES,
        prefix="singlora_unet",
        text_encoders=text_encoders,
        unet=transformer,
        multiplier=multiplier,
        network_args=network_args,
    )
    
    return network

def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> SingLoRANetworkWan:
    """
    Create SingLoRANetworkWan from weights with default WAN target modules.
    EXACT signature matching existing Takenoko networks.
    """
    return create_network_from_weights(
        WAN_SINGLORA_TARGET_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )

def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> SingLoRANetworkWan:
    """
    Create SingLoRANetworkWan from weights with custom target modules.
    EXACT signature and logic matching existing Takenoko networks.
    """
    # Extract dim/alpha mapping from weights (following lora_wan.py pattern)
    modules_dim = {}
    modules_alpha = {}
    
    for key, value in weights_sd.items():
        if "." not in key:
            continue
            
        lora_name = key.split(".")[0]
        
        if key.endswith(".A"):
            # Extract rank from A matrix shape (larger_dim, rank)
            rank = value.shape[1]
            modules_dim[lora_name] = rank
        elif "alpha" in key:
            modules_alpha[lora_name] = value.item()
    
    # Build network_args from extracted configuration
    network_args = {}
    if modules_dim:
        # Use the first module's rank as default
        default_rank = next(iter(modules_dim.values()))
        network_args["rank"] = default_rank
    if modules_alpha:
        # Use the first module's alpha as default
        default_alpha = next(iter(modules_alpha.values()))
        network_args["alpha"] = default_alpha
    
    logger.info(f"Creating SingLoRA network from weights with {len(modules_dim)} modules")
    
    network = SingLoRANetworkWan(
        target_replace_modules=target_replace_modules,
        prefix="singlora_unet",
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        network_args=network_args,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        **kwargs,
    )
    
    return network