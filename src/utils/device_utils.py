## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/utils/device_utils.py (Apache)

import torch


def clean_memory_on_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
