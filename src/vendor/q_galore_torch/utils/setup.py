import numpy as np
import torch 
import os 
from .quantization import QGaLoreLinear

def saving_model_weight(model, path):
    """
    Save model weight to file
    """
    checkpoint = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, QGaLoreLinear):
            checkpoint[name + '.weight'] = module.weight
            if module.bias is not None:
                checkpoint[name + '.bias'] = module.bias
            checkpoint[name + '.scales'] = module.weight.scales
            checkpoint[name + '.zeros'] = module.weight.zeros
            checkpoint[name + '.group_size'] = module.weight.group_size
            checkpoint[name + '.saved_data_dtype'] = module.weight.saved_data_dtype
            checkpoint[name + '.stochastic_round'] = module.weight.stochastic_round
    torch.save(checkpoint, path)

def load_model_weight(model, path):
    """
    Load model weight from file
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    for name, module in model.named_modules():
        if isinstance(module, QGaLoreLinear):
            weight_key = name + '.weight'
            if weight_key in checkpoint:
                module.weight.data.copy_(
                    checkpoint[weight_key].to(device=module.weight.device)
                )
            bias_key = name + '.bias'
            if module.bias is not None and bias_key in checkpoint:
                module.bias.data.copy_(
                    checkpoint[bias_key].to(
                        device=module.bias.device,
                        dtype=module.bias.dtype,
                    )
                )
            scales_key = name + '.scales'
            if scales_key in checkpoint:
                module.weight.scales = checkpoint[scales_key].to(
                    device=module.weight.device
                )
            zeros_key = name + '.zeros'
            if zeros_key in checkpoint:
                module.weight.zeros = checkpoint[zeros_key].to(
                    device=module.weight.device
                )
            group_size_key = name + '.group_size'
            if group_size_key in checkpoint:
                module.weight.group_size = checkpoint[group_size_key]
            saved_dtype_key = name + '.saved_data_dtype'
            if saved_dtype_key in checkpoint:
                module.weight.saved_data_dtype = checkpoint[saved_dtype_key]
            stochastic_round_key = name + '.stochastic_round'
            if stochastic_round_key in checkpoint:
                module.weight.stochastic_round = checkpoint[stochastic_round_key]

    return model
