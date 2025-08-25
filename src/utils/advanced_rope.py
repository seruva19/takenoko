## Based on https://github.com/Sarania/blissful-tuner/blob/main/src/blissful_tuner/advanced_rope.py (Apache 2.0)

"""
Created on Wed Apr 16 19:25:53 2025
Advanced rope functions for Blissful Tuner extension
License: Apache 2.0

@author: blyss
"""
import torch
import torch.nn as nn
from einops import rearrange
from typing import List


# From ComfyUI
def apply_rope_comfy(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)  # type: ignore
