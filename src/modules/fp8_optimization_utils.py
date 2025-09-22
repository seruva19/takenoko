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


# unused
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
    Quantize a tensor to FP8 format (legacy implementation).

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


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """
    Quantize a tensor to FP8 format using PyTorch's native FP8 dtype support.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        fp8_dtype (torch.dtype): Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8

    Returns:
        torch.Tensor: Quantized tensor in FP8 format
    """
    tensor = tensor.to(torch.float32)  # ensure tensor is in float32 for division

    # Create scaled tensor with NaN handling
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)  # handle NaN values

    # Clamp tensor to range
    tensor = tensor.clamp_(min=min_value, max=max_value)

    # Convert to FP8 dtype
    tensor = tensor.to(fp8_dtype)

    return tensor


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
    percentile: Optional[float] = None,
):
    """
    Quantize a weight tensor using block-wise, channel-wise, or tensor-wise scaling.

    Args:
        key (str): Layer key for logging
        tensor (torch.Tensor): Weight tensor to quantize
        fp8_dtype (torch.dtype): Target FP8 dtype
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8
        quantization_mode (str): "tensor", "channel", or "block"
        block_size (int): Block size for block-wise quantization
        percentile (Optional[float]): Percentile for scale calculation (None for max value)

    Returns:
        tuple: (quantized_weight, scale_tensor)
    """
    original_shape = tensor.shape

    # Determine quantization mode
    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "channel"  # fallback to per-channel
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by block_size {block_size}, fallback to per-channel quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(
                    out_features, num_blocks, block_size
                )  # [out, num_blocks, block_size]
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor

    # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
    # value shape is expected to be [out_features, in_features] for Linear weights
    if quantization_mode == "channel" or quantization_mode == "block":
        # row-wise scaling to avoid being dominated by outliers
        # result shape: [out_features, 1] or [out_features, num_blocks, 1]
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)

        if percentile is None:
            # Use max value (original behavior)
            row_scale_val = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        else:
            # Use percentile-based scaling
            row_scale_val = torch.quantile(
                abs_w.to(torch.float32), q=percentile, dim=scale_dim, keepdim=True
            )
        scale = row_scale_val / max_value

    else:
        # per-tensor
        if percentile is None:
            tensor_scale_val = torch.max(torch.abs(tensor).view(-1))
        else:
            abs_tensor = torch.abs(tensor).view(-1).to(torch.float32)
            # For very large tensors, use chunked processing to avoid memory issues
            if abs_tensor.numel() > 8192 * 4096:
                num_chunks = (abs_tensor.numel() + 8192 * 4096 - 1) // (8192 * 4096)
                chunked_abs = torch.chunk(abs_tensor, num_chunks)
                chunked_q = [
                    torch.quantile(chunk, q=percentile) for chunk in chunked_abs
                ]
                tensor_scale_val = torch.max(torch.stack(chunked_q, dim=0))
            else:
                tensor_scale_val = torch.quantile(abs_tensor, q=percentile)

        scale = tensor_scale_val / max_value

    # numerical safety - apply after mode-specific calculation
    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)  # ensure scale is in float32 for division

    # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
    quantized_weight = quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value)

    # If block-wise, restore original shape
    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(
            original_shape
        )  # restore to original shape [out, in]

    return quantized_weight, scale


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    quant_dtype: Optional[torch.dtype] = None,
    percentile: Optional[float] = None,
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

        # Quantize weight to FP8 using new function
        quantized_weight, scale_tensor = quantize_weight(
            key,
            value_q,
            fp8_dtype,
            max_value,
            min_value,
            quantization_mode,
            block_size,
            percentile,
        )

        # Compute per-tensor mean relative error (%) before moving tensors
        try:
            q_w = quantized_weight.to(value_q.dtype)
            if getattr(scale_tensor, "ndim", 0) >= 3:
                # Block-wise scale: reshape to [out_features, num_blocks, block_size] for correct broadcasting
                out_features, num_blocks, _ = scale_tensor.shape
                q_w = q_w.contiguous().view(out_features, num_blocks, -1)
                q_w = q_w * scale_tensor.to(q_w.dtype)
                reconstructed = q_w.view_as(value_q)
            else:
                # Per-tensor or per-channel scales broadcast directly
                reconstructed = q_w * scale_tensor.to(q_w.dtype)
            denom = torch.mean(torch.abs(value_q)) + 1e-8
            q_err = (torch.mean(torch.abs(value_q - reconstructed)) / denom) * 100.0
            per_tensor_quantization_error.append(float(q_err.item()))
        except Exception:
            # Best-effort; skip error calc if any unexpected dtype/device/device issue
            pass

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        quantized_weight = quantized_weight.to(fp8_dtype)

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        # Keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
        scale_tensor = scale_tensor.to(
            dtype=(quant_dtype or original_dtype), device=quantized_weight.device
        )

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    if optimized_count > 0:
        mode_info = f"{quantization_mode} mode"
        if quantization_mode == "block" and block_size:
            mode_info += f" (block_size={block_size})"
        logger.info(
            f"🧮 FP8 optimization completed: {optimized_count} layers quantized ({mode_info})"
        )
        if len(per_tensor_quantization_error) > 0:
            errs = torch.tensor(per_tensor_quantization_error)
            mean_err = float(errs.mean().item())
            min_err = float(errs.min().item())
            max_err = float(errs.max().item())
            logger.info(
                f"📊 FP8 quantization error: {mean_err:.2f}% avg (range: {min_err:.2f}%-{max_err:.2f}%)"
            )
    else:
        logger.info("🧮 FP8 optimization: No layers quantized")
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
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    quant_dtype: Optional[torch.dtype] = None,
    percentile: Optional[float] = None,
    *,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
) -> dict:
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

                # Quantize weight to FP8 using new function
                quantized_weight, scale_tensor = quantize_weight(
                    key,
                    value_q,
                    fp8_dtype,
                    max_value,
                    min_value,
                    quantization_mode,
                    block_size,
                    percentile,
                )

                # Compute per-tensor mean relative error (%) before moving tensors
                try:
                    q_w = quantized_weight.to(value_q.dtype)
                    if getattr(scale_tensor, "ndim", 0) >= 3:
                        # Block-wise scale: reshape to [out_features, num_blocks, block_size] for correct broadcasting
                        out_features, num_blocks, _ = scale_tensor.shape
                        q_w = q_w.contiguous().view(out_features, num_blocks, -1)
                        q_w = q_w * scale_tensor.to(q_w.dtype)
                        reconstructed = q_w.view_as(value_q)
                    else:
                        # Per-tensor or per-channel scales broadcast directly
                        reconstructed = q_w * scale_tensor.to(q_w.dtype)
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

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # Keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                scale_tensor = scale_tensor.to(
                    dtype=(quant_dtype or original_dtype),
                    device=quantized_weight.device,
                )

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    if optimized_count > 0:
        mode_info = f"{quantization_mode} mode"
        if quantization_mode == "block" and block_size:
            mode_info += f" (block_size={block_size})"
        logger.info(
            f"🧮 FP8 optimization completed: {optimized_count} layers quantized ({mode_info})"
        )
        if len(per_tensor_quantization_error) > 0:
            errs = torch.tensor(per_tensor_quantization_error)
            mean_err = float(errs.mean().item())
            min_err = float(errs.min().item())
            max_err = float(errs.max().item())
            logger.info(
                f"📊 FP8 quantization error: {mean_err:.2f}% avg (range: {min_err:.2f}%-{max_err:.2f}%)"
            )
    else:
        logger.info("🧮 FP8 optimization: No layers quantized")
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
        # **not tested**
        # _scaled_mm only works for per-tensor scale for now (per-channel scale does not work in certain cases)
        if self.scale_weight.ndim != 1:
            raise ValueError("scaled_mm only supports per-tensor scale_weight for now.")

        input_dtype = x.dtype
        original_weight_dtype = getattr(
            self, "original_dtype_enum", self.scale_weight.dtype
        )
        weight_dtype = self.weight.dtype
        target_dtype = self.weight.dtype
        # assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, hidden_dim)"

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            fp8_max_value = torch.finfo(target_dtype).max
            fp8_min_value = torch.finfo(target_dtype).min
            x = quantize_fp8(x, scale_x, target_dtype, fp8_max_value, fp8_min_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # Use input_dtype consistently for both bias/no-bias cases
            o = torch._scaled_mm(
                x,
                weight,
                out_dtype=input_dtype,  # type: ignore
                bias=self.bias,
                scale_a=scale_x,
                scale_b=scale_weight,  # type: ignore
            )
        else:
            o = torch._scaled_mm(
                x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight  # type: ignore
            )

        o = (
            o.reshape(original_shape[0], original_shape[1], -1)
            if len(original_shape) == 3
            else o.reshape(original_shape[0], -1)
        )
        return o.to(input_dtype)

    else:
        # Dequantize the weight
        original_dtype = getattr(self, "original_dtype_enum", self.scale_weight.dtype)

        # Optional upcast of the linear matmul for accuracy (ignored in scaled_mm path)
        do_upcast = bool(getattr(self, "upcast_linear", False))

        # Handle different scale tensor shapes for block-wise quantization
        if self.scale_weight.ndim < 3:
            # Per-tensor or per-channel quantization, we can broadcast
            if do_upcast:
                x_cast = x.to(torch.float32)
                dequantized_weight = self.weight.to(torch.float32) * self.scale_weight.to(torch.float32)  # type: ignore
            else:
                x_cast = x
                dequantized_weight = self.weight.to(original_dtype) * self.scale_weight  # type: ignore
        else:
            # Block-wise quantization, need to reshape weight to match scale shape for broadcasting
            out_features, num_blocks, _ = self.scale_weight.shape
            if do_upcast:
                x_cast = x.to(torch.float32)
                dequantized_weight = (
                    self.weight.to(torch.float32)
                    .contiguous()
                    .view(out_features, num_blocks, -1)
                )
                dequantized_weight = dequantized_weight * self.scale_weight.to(torch.float32)  # type: ignore
                dequantized_weight = dequantized_weight.view(self.weight.shape)
            else:
                x_cast = x
                dequantized_weight = (
                    self.weight.to(original_dtype)
                    .contiguous()
                    .view(out_features, num_blocks, -1)
                )
                dequantized_weight = dequantized_weight * self.scale_weight  # type: ignore
                dequantized_weight = dequantized_weight.view(self.weight.shape)

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

    # Enumerate patched layers and store scale shape info (for enhanced compatibility)
    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

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
            scale_shape = scale_shape_info[name]
            module.register_buffer(
                "scale_weight", torch.ones(scale_shape, dtype=q_dtype)
            )
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

    logger.info(f"🧮 FP8 quantization active: {q_dtype} format")
    if patched_count > 0:
        logger.info(f"🔧 FP8 monkey patch applied to {patched_count} layers")
    else:
        logger.info("🔧 FP8 monkey patch: No layers patched")
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
