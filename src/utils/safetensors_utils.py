import os
import re
import json
import struct
from typing import Dict, Any, Union, Optional

import numpy as np
import torch

from memory.safetensors_loader import load_file
from utils.device_utils import synchronize_device


def _normalize_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
    if device is None:
        return None
    if isinstance(device, str):
        return torch.device(device)
    return device


def mem_eff_save_file(
    tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None  # type: ignore
):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(
                    f"Warning: Metadata value for key '{key}' is not a string. Converting to string."
                )
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    # print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {
                "dtype": _TYPES[v.dtype],
                "shape": list(v.shape),
                "data_offsets": [offset, offset],
            }
        else:
            size = v.numel() * v.element_size()
            header[k] = {
                "dtype": _TYPES[v.dtype],
                "shape": list(v.shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if (
                        v.dim() == 0
                    ):  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    """Lightweight safetensors reader that supports optional zero-copy optimisations."""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def get_tensor(
        self,
        key: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        enable_memory_mapping: bool = False,
        enable_zero_copy_loading: bool = False,
        enable_non_blocking_transfers: bool = False,
        memory_mapping_threshold: int = 10 * 1024 * 1024,
    ) -> torch.Tensor:
        """Return a tensor by key using optional acceleration strategies.

        All optimisation flags default to ``False`` so existing behaviour is preserved
        unless explicitly enabled by configuration.
        """

        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        normalized_device = _normalize_device(device)
        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=normalized_device)

        tensor_offset = self.header_size + 8 + offset_start
        non_blocking = enable_non_blocking_transfers and normalized_device is not None and normalized_device.type == "cuda"

        if (
            enable_memory_mapping
            and num_bytes > memory_mapping_threshold
            and normalized_device is not None
            and normalized_device.type != "cpu"
        ):
            mm = np.memmap(
                self.filename,
                mode="c",
                dtype=np.uint8,
                offset=tensor_offset,
                shape=(num_bytes,),
            )
            byte_tensor = torch.from_numpy(mm)
            del mm

            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor

            result = cpu_tensor.to(device=normalized_device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return result

        self.file.seek(tensor_offset)

        if enable_zero_copy_loading:
            numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
            byte_tensor = torch.from_numpy(numpy_array)
            del numpy_array
        else:
            tensor_bytes = self.file.read(num_bytes)
            if tensor_bytes is None:
                byte_tensor = torch.empty(0, dtype=torch.uint8)
            else:
                tensor_bytes = bytearray(tensor_bytes)
                byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        return tensor.to(device=normalized_device, dtype=target_dtype, non_blocking=non_blocking)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, byte_tensor_or_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if isinstance(byte_tensor_or_bytes, torch.Tensor):
            byte_tensor = byte_tensor_or_bytes
        elif byte_tensor_or_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(byte_tensor_or_bytes)
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # # convert to float16 if float8 is not supported
            # print(f"Warning: {dtype_str} is not supported in this PyTorch version. Converting to float16.")
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(
                f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)"
            )


def load_safetensors(
    path: str,
    device: Optional[Union[str, torch.device]],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    *,
    enable_memory_mapping: bool = False,
    enable_zero_copy_loading: bool = False,
    enable_non_blocking_transfers: bool = False,
    memory_mapping_threshold: int = 10 * 1024 * 1024,
) -> dict[str, torch.Tensor]:
    normalized_device = _normalize_device(device)

    if disable_mmap:
        state_dict = {}
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(
                    key,
                    device=normalized_device,
                    dtype=dtype,
                    enable_memory_mapping=enable_memory_mapping,
                    enable_zero_copy_loading=enable_zero_copy_loading,
                    enable_non_blocking_transfers=enable_non_blocking_transfers,
                    memory_mapping_threshold=memory_mapping_threshold,
                )
        if (
            enable_non_blocking_transfers
            and normalized_device is not None
            and normalized_device.type == "cuda"
        ):
            synchronize_device(normalized_device)
        return state_dict

    try:
        state_dict = load_file(
            path, device=str(normalized_device) if normalized_device is not None else None
        )
    except Exception:
        state_dict = load_file(path)

    if dtype is not None:
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(dtype=dtype)

    return state_dict


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load split weights from a file. If the file name ends with 00001-of-00004 etc, it will load all files with the same prefix.
    dtype is as is, no conversion is done.
    """
    device = torch.device(device)

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        state_dict = {}
        for i in range(count):
            filename = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                state_dict.update(
                    load_safetensors(filepath, device=device, disable_mmap=disable_mmap)
                )
            else:
                raise FileNotFoundError(f"File {filepath} not found")
    else:
        state_dict = load_safetensors(
            file_path, device=device, disable_mmap=disable_mmap
        )
    return state_dict


def extract_config_from_metadata(metadata: Dict[str, str]) -> Optional[str]:
    """Extract the original config content from safetensors metadata."""
    return metadata.get("takenoko_config_content")


def extract_config_file_from_metadata(metadata: Dict[str, str]) -> Optional[str]:
    """Extract the original config filename from safetensors metadata."""
    return metadata.get("takenoko_config_file")


def load_metadata_from_safetensors_file(file_path: str) -> Dict[str, str]:
    """Load metadata from a safetensors file."""
    try:
        with open(file_path, "rb") as f:
            # Read the header length (first 8 bytes)
            header_length_bytes = f.read(8)
            header_length = int.from_bytes(header_length_bytes, "little")

            # Read the header
            header_bytes = f.read(header_length)
            header_str = header_bytes.decode("utf-8")

            # Parse the header JSON
            header_data = json.loads(header_str)

            # Extract metadata
            metadata = header_data.get("__metadata__", {})
            return metadata
    except Exception as e:
        print(f"Error loading metadata from {file_path}: {e}")
        return {}


def save_config_from_safetensors(
    file_path: str, output_path: Optional[str] = None
) -> bool:
    """Extract and save the config content from a safetensors file."""
    metadata = load_metadata_from_safetensors_file(file_path)

    config_content = extract_config_from_metadata(metadata)
    if config_content is None:
        print(f"No config content found in metadata for {file_path}")
        return False

    config_filename = extract_config_file_from_metadata(metadata)
    if config_filename is None:
        config_filename = "extracted_config.toml"

    if output_path is None:
        # Save in the same directory as the safetensors file
        base_dir = os.path.dirname(file_path)
        output_path = os.path.join(base_dir, config_filename)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Config saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving config to {output_path}: {e}")
        return False
