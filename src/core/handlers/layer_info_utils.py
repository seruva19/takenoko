"""Layer information utilities for training progress display."""

from typing import Any, Dict, Optional, Tuple
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def get_model_layer_info(transformer: Any, network: Optional[Any] = None) -> Dict[str, Any]:
    """Extract comprehensive layer information from WAN model.
    
    Args:
        transformer: The WAN transformer model
        network: Optional LoRA network (for fine-tuning scenarios)
        
    Returns:
        Dictionary containing layer statistics
    """
    try:
        layer_info = {
            'total_layers': 0,
            'trainable_layers': 0,
            'frozen_layers': 0,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layer_percentage': 0.0,
            'parameter_percentage': 0.0,
            'is_full_finetune': False,
            'layer_types': {},
            'training_mode': 'unknown'
        }
        
        # Get transformer layers
        try:
            if hasattr(transformer, 'blocks'):
                blocks = transformer.blocks
                layer_info['total_layers'] = len(blocks)
                logger.debug(f"Found blocks via transformer.blocks: {len(blocks)} layers")
            elif hasattr(transformer, 'num_layers') and hasattr(transformer, 'blocks'):
                blocks = transformer.blocks
                layer_info['total_layers'] = len(blocks)
                logger.debug(f"Found blocks via transformer.blocks with num_layers: {len(blocks)} layers")
            else:
                # Fallback: count all direct submodules
                if hasattr(transformer, 'named_children'):
                    blocks = [module for name, module in transformer.named_children() 
                             if 'block' in name.lower() or 'layer' in name.lower()]
                    layer_info['total_layers'] = len(blocks)
                    logger.debug(f"Found blocks via named_children: {len(blocks)} layers")
                else:
                    blocks = []
                    layer_info['total_layers'] = 0
                    logger.debug("No blocks found via any method")
        except Exception as e:
            logger.warning(f"Error getting blocks from transformer: {e}")
            blocks = []
            layer_info['total_layers'] = 0
        
        # Analyze each layer
        trainable_layer_count = 0
        for i, block in enumerate(blocks):
            layer_name = f"block_{i}"
            layer_type = type(block).__name__
            
            # Track layer types
            if layer_type not in layer_info['layer_types']:
                layer_info['layer_types'][layer_type] = 0
            layer_info['layer_types'][layer_type] += 1
            
            # Check if layer has trainable parameters
            has_trainable_params = any(p.requires_grad for p in block.parameters())
            if has_trainable_params:
                trainable_layer_count += 1
        
        layer_info['trainable_layers'] = trainable_layer_count
        layer_info['frozen_layers'] = layer_info['total_layers'] - trainable_layer_count
        
        # Calculate percentages
        if layer_info['total_layers'] > 0:
            layer_info['layer_percentage'] = (trainable_layer_count / layer_info['total_layers']) * 100
        
        # Get parameter counts
        total_params = sum(p.numel() for p in transformer.parameters())
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        
        layer_info['total_parameters'] = total_params
        layer_info['trainable_parameters'] = trainable_params
        
        if total_params > 0:
            layer_info['parameter_percentage'] = (trainable_params / total_params) * 100
        
        # Check network (LoRA) parameters if provided
        if network is not None:
            network_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
            layer_info['trainable_parameters'] += network_trainable
            
            # Recalculate percentage with network params
            total_with_network = total_params + sum(p.numel() for p in network.parameters())
            if total_with_network > 0:
                layer_info['parameter_percentage'] = (layer_info['trainable_parameters'] / total_with_network) * 100
        
        # Determine training mode
        if layer_info['layer_percentage'] >= 95.0:
            layer_info['training_mode'] = 'full_finetune'
            layer_info['is_full_finetune'] = True
        elif layer_info['layer_percentage'] >= 1.0:
            layer_info['training_mode'] = 'partial_finetune'
        else:
            layer_info['training_mode'] = 'lora_finetune'
        
        return layer_info
        
    except Exception as e:
        logger.warning(f"Error extracting layer info: {e}")
        return {
            'total_layers': 0,
            'trainable_layers': 0,
            'layer_percentage': 0.0,
            'training_mode': 'error',
            'is_full_finetune': False
        }


def format_layer_display(layer_info: Dict[str, Any]) -> str:
    """Format layer information for display in progress bar.
    
    Args:
        layer_info: Layer information dictionary
        
    Returns:
        Formatted string for display
    """
    try:
        if layer_info['training_mode'] == 'error':
            return "Layer info unavailable"
        
        trainable = layer_info['trainable_layers']
        total = layer_info['total_layers']
        percentage = layer_info['layer_percentage']
        mode = layer_info['training_mode']
        
        # Format based on training mode
        if mode == 'full_finetune':
            return f"FULL FT: {trainable}/{total} layers ({percentage:.1f}%)"
        elif mode == 'partial_finetune':
            return f"PARTIAL FT: {trainable}/{total} layers ({percentage:.1f}%)"
        else:  # lora_finetune
            param_pct = layer_info['parameter_percentage']
            return f"LoRA FT: {trainable}/{total} layers ({percentage:.1f}%) params:{param_pct:.1f}%"
            
    except Exception as e:
        logger.warning(f"Error formatting layer display: {e}")
        return "Layer info error"


def get_compact_layer_stats(layer_info: Dict[str, Any]) -> Dict[str, str]:
    """Get compact layer statistics for progress bar postfix.
    
    Args:
        layer_info: Layer information dictionary
        
    Returns:
        Dictionary with compact statistics
    """
    try:
        if layer_info['training_mode'] == 'error':
            return {"layers": "err"}
        
        stats = {}
        
        # Basic layer info
        stats["layers"] = f"{layer_info['trainable_layers']}/{layer_info['total_layers']}"
        
        # Training mode indicator
        mode = layer_info['training_mode']
        if mode == 'full_finetune':
            stats["mode"] = "FULL"
        elif mode == 'partial_finetune':
            stats["mode"] = "PART"
        else:
            stats["mode"] = "LoRA"
        
        # Parameter percentage
        stats["train%"] = f"{layer_info['parameter_percentage']:.1f}%"
        
        return stats
        
    except Exception as e:
        logger.warning(f"Error getting compact layer stats: {e}")
        return {"layers": "err"}


def validate_finetuning_progress(layer_info: Dict[str, Any]) -> bool:
    """Validate that actual fine-tuning is happening.
    
    Args:
        layer_info: Layer information dictionary
        
    Returns:
        True if fine-tuning is detected, False otherwise
    """
    try:
        # Check if any layers are actually being trained
        if layer_info['trainable_layers'] == 0:
            return False
        
        # Check if trainable parameters exist
        if layer_info['trainable_parameters'] == 0:
            return False
        
        # Check if we have some reasonable percentage
        if layer_info['layer_percentage'] <= 0:
            return False
        
        return True
        
    except Exception:
        return False


def get_layer_structure_info(transformer: Any) -> Dict[str, Any]:
    """Extract comprehensive layer information from WAN transformer.
    
    Args:
        transformer: The WAN transformer model
        
    Returns:
        Dictionary containing layer statistics
    """
    layer_info = {
        'total_layers': 0,
        'trainable_layers': 0,
        'frozen_layers': 0,
        'total_parameters': 0,
        'trainable_parameters': 0,
        'layer_percentage': 0.0,
        'parameter_percentage': 0.0,
        'is_full_finetune': False,
        'layer_types': {},
        'subcomponent_types': {},
        'training_mode': 'unknown'
    }
    
    try:
        # Get transformer layers
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
            layer_info['total_layers'] = len(blocks)
        elif hasattr(transformer, 'num_layers') and hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
            layer_info['total_layers'] = len(blocks)
        else:
            # Fallback: count all direct submodules
            if hasattr(transformer, 'named_children'):
                blocks = [module for name, module in transformer.named_children() 
                         if 'block' in name.lower() or 'layer' in name.lower()]
                layer_info['total_layers'] = len(blocks)
            else:
                blocks = []
                layer_info['total_layers'] = 0
        
        # Analyze each layer and its subcomponents
        trainable_layer_count = 0
        for i, block in enumerate(blocks):
            layer_type = type(block).__name__
            
            # Track main layer types
            if layer_type not in layer_info['layer_types']:
                layer_info['layer_types'][layer_type] = 0
            layer_info['layer_types'][layer_type] += 1
            
            # Check if main layer has trainable parameters
            has_trainable_params = any(p.requires_grad for p in block.parameters())
            if has_trainable_params:
                trainable_layer_count += 1
            
            # Analyze subcomponents for WanAttentionBlock
            if layer_type == 'WanAttentionBlock':
                for name, module in block.named_children():
                    subcomponent_type = type(module).__name__
                    
                    # Track subcomponent types
                    if subcomponent_type not in layer_info['subcomponent_types']:
                        layer_info['subcomponent_types'][subcomponent_type] = 0
                    layer_info['subcomponent_types'][subcomponent_type] += 1
                    
                    # Also track by common names for better readability
                    if 'norm' in name.lower() and 'LayerNorm' in subcomponent_type:
                        component_name = 'LayerNorm'
                    elif 'attn' in name.lower():
                        component_name = f'Attention_{name}'
                    elif 'ffn' in name.lower():
                        component_name = 'FFN'
                    elif 'Linear' in subcomponent_type:
                        component_name = 'Linear'
                    elif 'GELU' in subcomponent_type:
                        component_name = 'GELU'
                    else:
                        component_name = subcomponent_type
                    
                    if component_name not in layer_info['subcomponent_types']:
                        layer_info['subcomponent_types'][component_name] = 0
                    layer_info['subcomponent_types'][component_name] += 1
        
        layer_info['trainable_layers'] = trainable_layer_count
        layer_info['frozen_layers'] = layer_info['total_layers'] - trainable_layer_count
        
        # Calculate percentages
        if layer_info['total_layers'] > 0:
            layer_info['layer_percentage'] = (trainable_layer_count / layer_info['total_layers']) * 100
        
        # Get parameter counts
        if HAS_TORCH:
            total_params = sum(p.numel() for p in transformer.parameters())
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            
            layer_info['total_parameters'] = total_params
            layer_info['trainable_parameters'] = trainable_params
            
            if total_params > 0:
                layer_info['parameter_percentage'] = (trainable_params / total_params) * 100
        
        # Determine training mode
        if layer_info['layer_percentage'] >= 95.0:
            layer_info['training_mode'] = 'full_finetune'
            layer_info['is_full_finetune'] = True
        elif layer_info['layer_percentage'] >= 1.0:
            layer_info['training_mode'] = 'partial_finetune'
        else:
            layer_info['training_mode'] = 'lora_finetune'
            
    except Exception as e:
        logger.warning(f"Error extracting layer info: {e}")
        
    return layer_info