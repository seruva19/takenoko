import os
import re
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file


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
    # does not support metadata loading
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

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)  # type: ignore

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
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        # logger.info(f"Loading without mmap (experimental)")
        state_dict = {}
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(device, dtype=dtype)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)  # type: ignore
        except:
            state_dict = load_file(path)  # prevent device invalid Error
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
