import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Union, Optional
import logging
from tqdm import tqdm
from common.logger import get_logger
from utils.safetensors_utils import MemoryEfficientSafeOpen

logger = get_logger(__name__, level=logging.INFO)

from utils.device_utils import clean_memory_on_device, synchronize_device


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign).

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"

    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    if exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max

    raise ValueError(
        f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}"
    )


def quantize_tensor_to_fp8(
    tensor,
    scale,
    exp_bits=4,
    mantissa_bits=3,
    sign_bits=1,
    max_value=None,
    min_value=None,
):
    """
    Quantize a tensor to FP8 format.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits

    Returns:
        tuple: (quantized_tensor, scale_factor)
    """
    # Create scaled tensor
    scaled_tensor = tensor / scale

    # Calculate FP8 parameters
    bias = 2 ** (exp_bits - 1) - 1

    if max_value is None:
        # Calculate max and min values
        max_value = calculate_fp8_maxval(exp_bits, mantissa_bits, sign_bits)
        min_value = -max_value if sign_bits > 0 else 0.0

    # Clamp tensor to range
    clamped_tensor = torch.clamp(scaled_tensor, min_value, max_value)

    # Quantization process
    abs_values = torch.abs(clamped_tensor)
    nonzero_mask = abs_values > 0

    # Calculate log scales (only for non-zero elements)
    log_scales = torch.zeros_like(clamped_tensor)
    if nonzero_mask.any():
        log_scales[nonzero_mask] = (
            torch.floor(torch.log2(abs_values[nonzero_mask]) + bias)
            .detach()
            .to(log_scales.dtype)
        )

    # Limit log scales and calculate quantization factor
    log_scales = torch.clamp(log_scales, min=1.0)
    quant_factor = 2.0 ** (log_scales - mantissa_bits - bias)

    # Quantize and dequantize
    quantized = torch.round(clamped_tensor / quant_factor) * quant_factor

    return quantized, scale


def optimize_state_dict_with_fp8(
    state_dict,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    quant_dtype: Optional[torch.dtype] = None,
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format.

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits (default: 4 for E4M3FN format, original behavior)
        mantissa_bits (int): Number of mantissa bits (default: 3 for E4M3FN format, original behavior)
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"⛔ Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Create optimized state dict
    optimized_count = 0
    per_tensor_quantization_error: list[float] = []

    # Enumerate tarket keys
    target_state_dict_keys = []
    for key in state_dict.keys():
        # Check if it's a weight key and matches target patterns
        is_target = (
            target_layer_keys is None
            or any(pattern in key for pattern in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(
            pattern in key for pattern in exclude_layer_keys
        )
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Process each key
    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        # Save original device and dtype
        original_device = value.device
        original_dtype = value.dtype

        # Move to calculation device
        if calc_device is not None:
            value = value.to(calc_device)

        # Optionally upcast for scale computation and quantization for improved accuracy
        value_q = value.to(quant_dtype) if quant_dtype is not None else value

        # Calculate scale factor
        scale = torch.max(torch.abs(value_q.flatten())) / max_value
        # print(f"Optimizing {key} with scale: {scale}")

        # Quantize weight to FP8
        quantized_weight, _ = quantize_tensor_to_fp8(
            value_q, scale, exp_bits, mantissa_bits, 1, max_value, min_value
        )

        # Compute per-tensor mean relative error (%) before moving tensors
        try:
            reconstructed = quantized_weight.to(value_q.dtype) * scale
            denom = torch.mean(torch.abs(value_q)) + 1e-8
            q_err = (torch.mean(torch.abs(value_q - reconstructed)) / denom) * 100.0
            per_tensor_quantization_error.append(float(q_err.item()))
        except Exception:
            # Best-effort; skip error calc if any unexpected dtype/device issue
            pass

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        quantized_weight = quantized_weight.to(fp8_dtype)

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        scale_tensor = torch.tensor(
            [scale],
            dtype=(quant_dtype or original_dtype),
            device=quantized_weight.device,
        )

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    if optimized_count > 0 and len(per_tensor_quantization_error) > 0:
        errs = torch.tensor(per_tensor_quantization_error)
        mean_err = float(errs.mean().item())
        min_err = float(errs.min().item())
        max_err = float(errs.max().item())
        logger.info(
            f"Mean FP8 quantization error: {mean_err:.2f}% (min {min_err:.2f}%, max {max_err:.2f}%)"
        )
    if (
        enable_non_blocking_transfers
        and move_to_device
        and direct_transfer_device is not None
        and direct_transfer_device.type == "cuda"
    ):
        synchronize_device(direct_transfer_device)
    return state_dict


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quant_dtype: Optional[torch.dtype] = None,
    *,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
):
    """
    Load weight tensors from safetensors files and merge LoRA weights into the state dict with explicit FP8 optimization.

    Args:
        model_files (list[str]): List of model files to load
        calc_device (str or torch.device): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target for optimization (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude from optimization
        exp_bits (int): Number of exponent bits (default: 4 for E4M3FN format, original behavior)
        mantissa_bits (int): Number of mantissa bits (default: 3 for E4M3FN format, original behavior)
        move_to_device (bool): Move optimized tensors to the calculating device
        weight_hook (callable, optional): Function to apply to each weight tensor before optimization

    Returns:
        dict: FP8 optimized state dict
    """
    calc_device_obj: Optional[torch.device]
    if calc_device is None:
        calc_device_obj = None
    elif isinstance(calc_device, torch.device):
        calc_device_obj = calc_device
    else:
        calc_device_obj = torch.device(calc_device)

    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Define function to determine if a key is a target key. target means fp8 optimization, not for weight hook.
    def is_target_key(key):
        # Check if weight key matches target patterns and does not match exclude patterns
        is_target = (
            target_layer_keys is None
            or any(pattern in key for pattern in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(
            pattern in key for pattern in exclude_layer_keys
        )
        return is_target and not is_excluded

    # Create optimized state dict
    optimized_count = 0
    per_tensor_quantization_error: list[float] = []

    # Process each file
    state_dict = {}
    direct_transfer_device = (
        calc_device_obj if move_to_device and weight_hook is None else None
    )

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()
            for key in tqdm(
                keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"
            ):
                value = f.get_tensor(
                    key,
                    device=direct_transfer_device,
                    enable_memory_mapping=enable_memory_mapping,
                    enable_zero_copy_loading=enable_zero_copy_loading,
                    enable_non_blocking_transfers=enable_non_blocking_transfers,
                    memory_mapping_threshold=memory_mapping_threshold,
                )
                original_device_pre_hook = value.device
                original_dtype_pre_hook = value.dtype

                if weight_hook is not None:
                    # Apply weight hook if provided
                    value = weight_hook(
                        key,
                        value,
                        keep_on_calc_device=(calc_device_obj is not None),
                    )

                if not is_target_key(key):
                    target_device = (
                        calc_device_obj
                        if (calc_device_obj is not None and move_to_device)
                        else original_device_pre_hook
                    )
                    if value.device != target_device:
                        value = value.to(target_device)
                    state_dict[key] = value
                    continue

                # Save original device and dtype
                original_device = original_device_pre_hook
                original_dtype = original_dtype_pre_hook

                # Move to calculation device when we didn't request a direct transfer
                if direct_transfer_device is None and calc_device_obj is not None:
                    value = value.to(calc_device_obj)

                # Optionally upcast for scale computation and quantization for improved accuracy
                value_q = value.to(quant_dtype) if quant_dtype is not None else value

                # Calculate scale factor
                scale = torch.max(torch.abs(value_q.flatten())) / max_value
                # print(f"Optimizing {key} with scale: {scale}")

                # Quantize weight to FP8
                quantized_weight, _ = quantize_tensor_to_fp8(
                    value_q, scale, exp_bits, mantissa_bits, 1, max_value, min_value
                )

                # Compute per-tensor mean relative error (%) before moving tensors
                try:
                    reconstructed = quantized_weight.to(value_q.dtype) * scale
                    denom = torch.mean(torch.abs(value_q)) + 1e-8
                    q_err = (
                        torch.mean(torch.abs(value_q - reconstructed)) / denom
                    ) * 100.0
                    per_tensor_quantization_error.append(float(q_err.item()))
                except Exception:
                    pass

                # Add to state dict using original key for weight and new key for scale
                fp8_key = key  # Maintain original key
                scale_key = key.replace(".weight", ".scale_weight")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                quantized_weight = quantized_weight.to(fp8_dtype)

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                scale_tensor = torch.tensor(
                    [scale],
                    dtype=(quant_dtype or original_dtype),
                    device=quantized_weight.device,
                )

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    if optimized_count > 0 and len(per_tensor_quantization_error) > 0:
        errs = torch.tensor(per_tensor_quantization_error)
        mean_err = float(errs.mean().item())
        min_err = float(errs.min().item())
        max_err = float(errs.max().item())
        logger.info(
            f"Mean FP8 quantization error: {mean_err:.2f}% (min {min_err:.2f}%, max {max_err:.2f}%)"
        )
    if (
        enable_non_blocking_transfers
        and move_to_device
        and direct_transfer_device is not None
        and direct_transfer_device.type == "cuda"
    ):
        synchronize_device(direct_transfer_device)
    return state_dict


def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)
        max_value (float): Maximum value for FP8 quantization. If None, no quantization is applied for input tensor.

    Returns:
        torch.Tensor: Result of linear transformation
    """
    if use_scaled_mm:
        input_dtype = x.dtype
        original_weight_dtype = getattr(
            self, "original_dtype_enum", self.scale_weight.dtype
        )
        weight_dtype = self.weight.dtype
        target_dtype = torch.float8_e5m2
        assert (
            weight_dtype == torch.float8_e4m3fn
        ), "Only FP8 E4M3FN format is supported"
        assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, hidden_dim)"

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            x, _ = quantize_tensor_to_fp8(x, scale_x, 5, 2, 1, max_value, -max_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[2]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # out_dtype should reflect original weight dtype per reference
            o = torch._scaled_mm(
                x,
                weight,
                out_dtype=original_weight_dtype,  # type: ignore
                bias=self.bias,
                scale_a=scale_x,
                scale_b=scale_weight,  # type: ignore
            )
        else:
            o = torch._scaled_mm(
                x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight  # type: ignore
            )

        reshaped = o.reshape(original_shape[0], original_shape[1], -1)
        target_dtype = getattr(self, "original_dtype_enum", None)
        if isinstance(target_dtype, torch.dtype):
            return reshaped.to(target_dtype)
        return reshaped

    else:
        # Dequantize the weight
        original_dtype = getattr(self, "original_dtype_enum", self.scale_weight.dtype)

        # Optional upcast of the linear matmul for accuracy (ignored in scaled_mm path)
        do_upcast = bool(getattr(self, "upcast_linear", False))
        if do_upcast:
            x_cast = x.to(torch.float32)
            dequantized_weight = self.weight.to(torch.float32) * self.scale_weight.to(torch.float32)  # type: ignore
        else:
            x_cast = x
            dequantized_weight = self.weight.to(original_dtype) * self.scale_weight  # type: ignore

        # Perform linear transformation
        if self.bias is not None:
            output = F.linear(x_cast, dequantized_weight, self.bias)
        else:
            output = F.linear(x_cast, dequantized_weight)

        # Cast back to input dtype if we upcasted
        return output.to(x.dtype) if do_upcast else output


def apply_fp8_monkey_patch(
    model,
    optimized_state_dict,
    use_scaled_mm=False,
    upcast_linear: bool = False,
    quant_dtype: Optional[torch.dtype] = None,
    exclude_ffn_from_scaled_mm: bool = False,
    scale_input_tensor: Optional[str] = None,
):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)
        upcast_linear (bool): Whether to upcast the linear transformation to float32
        quant_dtype (Optional[torch.dtype]): Quantization dtype used during optimization
        exclude_ffn_from_scaled_mm (bool): Exclude feedforward layers from scaled_mm (useful for WAN models)
        scale_input_tensor (Optional[str]): Scale input tensor format ("e4m3" or "e5m2") for scaled_mm

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # Calculate max_value for input tensor scaling if enabled
    max_value = None
    if use_scaled_mm:
        # Set model-level flag for FP8 optimization detection
        setattr(model, "fp8_matmul_enabled", True)

        if scale_input_tensor is not None:
            if "e4m3" in scale_input_tensor.lower():
                max_value = calculate_fp8_maxval(4, 3)
            elif "e5m2" in scale_input_tensor.lower():
                max_value = calculate_fp8_maxval(5, 2)
            else:
                logger.warning(
                    f"Unknown scale_input_tensor format: {scale_input_tensor}"
                )

    # Log configuration summary for clarity
    if exclude_ffn_from_scaled_mm:
        logger.info("FFNs will be excluded from scaled_mm patching (Wan mode)")
    if upcast_linear:
        logger.info(
            f"Linear transformations for scaled layers will be upcast to float32 {'except when using scaled_mm' if use_scaled_mm else ''}"
        )
    # We'll determine original dtype per-layer for better accuracy

    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0
    q_dtype = None

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # Register the scale_weight as a buffer to load the state_dict
            setattr(
                module, "original_dtype_enum", module.weight.dtype
            )  # More accurate per-layer dtype
            q_dtype = optimized_state_dict[f"{name}.scale_weight"].dtype
            module.register_buffer("scale_weight", torch.tensor(1.0, dtype=q_dtype))
            setattr(module, "upcast_linear", upcast_linear)

            # Determine if this layer should use scaled_mm (with FFN exclusion support)
            really_use_scaled_mm = use_scaled_mm
            if exclude_ffn_from_scaled_mm and _is_ffn_layer(name):
                really_use_scaled_mm = False

            # Create a new forward method with the patched version.
            # Use default arguments to capture variables properly in closure
            def new_forward(
                self, x, _use_scaled_mm=really_use_scaled_mm, _max_value=max_value
            ):
                return fp8_linear_forward_patch(self, x, _use_scaled_mm, _max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Quantization done in {q_dtype}")
    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model


def _is_ffn_layer(layer_name: str) -> bool:
    """
    Check if a layer is a feedforward network layer based on common naming patterns.

    Args:
        layer_name (str): Name of the layer

    Returns:
        bool: True if the layer is likely a feedforward layer
    """
    ffn_patterns = ["ffn", "feed_forward", "mlp", "fc", "feedforward"]
    layer_name_lower = layer_name.lower()
    return any(pattern in layer_name_lower for pattern in ffn_patterns)
