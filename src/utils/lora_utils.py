import os
import re
from typing import Dict, List, Optional, Union
import torch

import logging
from modules.fp8_optimization_utils import (
    load_safetensors_with_fp8_optimization,
    optimize_state_dict_with_fp8,
)
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

from tqdm import tqdm

from utils.safetensors_utils import (
    MemoryEfficientSafeOpen,
    load_safetensors,
)
from utils.device_utils import synchronize_device


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(
            f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}"
        )

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {
            k: v for k, v in weights_sd.items() if not regex_exclude.search(k)
        }
        logger.info(
            f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
        )

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning(f"No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[Dict[str, torch.Tensor]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    quant_dtype: Optional[torch.dtype] = None,
    fp8_format: str = "e4m3",
    fp8_per_channel: bool = False,
    fp8_percentile: Optional[float] = None,
    *,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): Dictionary of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        target_keys (Optional[List[str]]): Keys to target for optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from optimization.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        basename = os.path.basename(model_file)
        match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
        if match:
            prefix = basename[: match.start(2)]
            count = int(match.group(3))
            state_dict = {}
            for i in range(count):
                filename = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
                filepath = os.path.join(os.path.dirname(model_file), filename)
                if os.path.exists(filepath):
                    extended_model_files.append(filepath)
                else:
                    raise FileNotFoundError(f"File {filepath} not found")
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # load LoRA weights
    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []  # type: ignore
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())  # type: ignore
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        # Merge LoRA weights into the state dict
        logger.info(
            f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}"
        )

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            original_dtype = model_weight.dtype
            if original_device != calc_device:
                model_weight = model_weight.to(
                    calc_device
                )  # to make calculation faster

            for lora_weight_keys, lora_sd, multiplier in zip(
                list_of_lora_weight_keys, lora_weights_list, lora_multipliers  # type: ignore
            ):
                # check if this weight has LoRA weights
                lora_name = model_weight_key.rsplit(".", 1)[
                    0
                ]  # remove trailing ".weight"
                lora_name = "lora_unet_" + lora_name.replace(".", "_")
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                    continue

                # get LoRA weights
                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                # W <- W + U * D
                if len(model_weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    model_weight = (
                        model_weight + multiplier * (up_weight @ down_weight) * scale
                    )
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    model_weight = (
                        model_weight
                        + multiplier
                        * (
                            up_weight.squeeze(3).squeeze(2)
                            @ down_weight.squeeze(3).squeeze(2)
                        )
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3), up_weight
                    ).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    model_weight = model_weight + multiplier * conved * scale

                # remove LoRA keys from set
                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

            model_weight = model_weight.to(
                original_device, original_dtype
            )  # move back to original device and dtype
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
        quant_dtype=quant_dtype,
        fp8_format=fp8_format,
        fp8_per_channel=fp8_per_channel,
        fp8_percentile=fp8_percentile,
        enable_memory_mapping=enable_memory_mapping,
        enable_zero_copy_loading=enable_zero_copy_loading,
        enable_non_blocking_transfers=enable_non_blocking_transfers,
        memory_mapping_threshold=memory_mapping_threshold,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # check if all LoRA keys are used
        if len(lora_weight_keys) > 0:
            # if there are still LoRA keys left, it means they are not used in the model
            # this is a warning, not an error
            logger.warning(
                f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}"
            )

    return state_dict


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,  # type: ignore
    quant_dtype: Optional[torch.dtype] = None,
    fp8_format: str = "e4m3",
    fp8_per_channel: bool = False,
    fp8_percentile: Optional[float] = None,
    *,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. Hook enabled: {weight_hook is not None}"
        )

        # dit_weight_dtype is not used because we use fp8 optimization
        # Parse fp8_format to get exp_bits and mantissa_bits
        if fp8_format.lower() == "e4m3":
            exp_bits, mantissa_bits = 4, 3
        elif fp8_format.lower() == "e5m2":
            exp_bits, mantissa_bits = 5, 2
        else:
            logger.warning(
                f"Unknown fp8_format '{fp8_format}', using E4M3FN as default"
            )
            exp_bits, mantissa_bits = 4, 3

        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            exp_bits=exp_bits,
            mantissa_bits=mantissa_bits,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            quant_dtype=quant_dtype,
            enable_memory_mapping=enable_memory_mapping,
            enable_zero_copy_loading=enable_zero_copy_loading,
            enable_non_blocking_transfers=enable_non_blocking_transfers,
            memory_mapping_threshold=memory_mapping_threshold,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. Hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        direct_device = calc_device if move_to_device and weight_hook is None else None
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                for key in tqdm(
                    f.keys(),
                    desc=f"Loading {os.path.basename(model_file)}",
                    leave=False,
                ):
                    value = f.get_tensor(
                        key,
                        device=direct_device,
                        dtype=dit_weight_dtype if direct_device is not None else None,
                        enable_memory_mapping=enable_memory_mapping,
                        enable_zero_copy_loading=enable_zero_copy_loading,
                        enable_non_blocking_transfers=enable_non_blocking_transfers,
                        memory_mapping_threshold=memory_mapping_threshold,
                    )
                    if weight_hook is not None:
                        value = weight_hook(key, value)
                    if direct_device is None:
                        if move_to_device:
                            if dit_weight_dtype is None:
                                value = value.to(calc_device)
                            else:
                                value = value.to(calc_device, dtype=dit_weight_dtype)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)
                    state_dict[key] = value

        if (
            enable_non_blocking_transfers
            and direct_device is not None
            and direct_device.type == "cuda"
        ):
            synchronize_device(direct_device)

    return state_dict
