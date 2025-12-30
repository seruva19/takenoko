"""CAT-LVDM corruption helpers for conditioning embeddings."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class CatLVDMCorruptionHelper:
    """Applies CAT-LVDM corruption (BCNI or SACN) to conditioning embeddings."""

    def __init__(self, noise_type: str, noise_ratio: float) -> None:
        self.noise_type = noise_type
        self.noise_ratio = float(noise_ratio)
        self._warned_sacn_failure = False

    def apply_to_context(
        self,
        context: List[Tensor],
        attention_masks: Optional[Iterable[Tensor]] = None,
    ) -> List[Tensor]:
        if not context or self.noise_ratio <= 0:
            return context

        if self.noise_type == "bcni":
            return self._apply_bcni(context, attention_masks)
        if self.noise_type == "sacn":
            return self._apply_sacn(context)
        return context

    def _apply_bcni(
        self,
        context: List[Tensor],
        attention_masks: Optional[Iterable[Tensor]],
    ) -> List[Tensor]:
        mask_list = self._build_attention_masks(context, attention_masks)
        batch = pad_sequence(context, batch_first=True)
        mask = pad_sequence(mask_list, batch_first=True, padding_value=0.0).unsqueeze(-1)

        orig_dtype = batch.dtype
        batch_fp32 = batch.float()
        mask_fp32 = mask.float()

        token_counts = mask_fp32.sum(dim=0).clamp_min(1.0)
        batch_mean = (batch_fp32 * mask_fp32).sum(dim=0) / token_counts
        distance = torch.norm(batch_fp32 - batch_mean, dim=-1, keepdim=True)
        noise = (torch.rand_like(distance) - 0.5) * 2.0
        noise = noise * distance * self.noise_ratio
        noise = noise * mask_fp32

        noisy = (batch_fp32 + noise).to(orig_dtype)
        return [
            noisy[idx, :tensor.shape[0]].to(tensor.dtype)
            for idx, tensor in enumerate(context)
        ]

    def _apply_sacn(self, context: List[Tensor]) -> List[Tensor]:
        output: List[Tensor] = []
        for tensor in context:
            output.append(tensor + self._sacn_noise(tensor))
        return output

    def _sacn_noise(self, embedding: Tensor) -> Tensor:
        if embedding.numel() == 0:
            return torch.zeros_like(embedding)

        orig_dtype = embedding.dtype
        embedding_fp32 = embedding.float()

        try:
            if embedding_fp32.ndim == 1:
                embedding_fp32 = embedding_fp32.unsqueeze(0)

            u, s, v = torch.linalg.svd(embedding_fp32, full_matrices=False)
            freq_weights = torch.exp(
                -torch.arange(s.shape[-1], device=s.device) / s.shape[-1]
            )
            spectral_noise = torch.randn_like(s) * freq_weights
            local_scale = embedding_fp32.pow(2).mean(-1, keepdim=True)
            noise = torch.matmul(u * spectral_noise.unsqueeze(-1), v)
            noise = noise * torch.sqrt(local_scale)
            if embedding.ndim == 1:
                noise = noise.squeeze(0)
            return (noise * self.noise_ratio).to(orig_dtype)
        except Exception as exc:
            if not self._warned_sacn_failure:
                logger.warning(
                    "CAT-LVDM SACN noise generation failed; disabling SACN noise for"
                    " this batch (%s).",
                    exc,
                )
                self._warned_sacn_failure = True
            return torch.zeros_like(embedding)

    @staticmethod
    def _build_attention_masks(
        context: List[Tensor],
        attention_masks: Optional[Iterable[Tensor]],
    ) -> List[Tensor]:
        if attention_masks is None:
            return [
                torch.ones(tensor.shape[0], device=tensor.device, dtype=torch.float32)
                for tensor in context
            ]

        masks: List[Tensor] = []
        for tensor, mask in zip(context, attention_masks):
            if mask is None:
                masks.append(
                    torch.ones(
                        tensor.shape[0], device=tensor.device, dtype=torch.float32
                    )
                )
                continue
            if mask.ndim > 1:
                mask = mask.squeeze(0)
            mask = mask.to(device=tensor.device, dtype=torch.float32)
            if mask.shape[0] != tensor.shape[0]:
                mask = torch.ones(
                    tensor.shape[0], device=tensor.device, dtype=torch.float32
                )
            masks.append(mask)
        return masks
