"""Tokenizer utilities backing the rCM distillation integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from wan.modules.tokenizers import HuggingfaceTokenizer


@dataclass(slots=True)
class TokenizerAssets:
    """Configuration bundle used to build the shared Huggingface tokenizer."""

    text_model: str
    clean_mode: Optional[str] = None
    max_length: Optional[int] = None
    cache_dir: Optional[str] = None


@dataclass(slots=True)
class TokenizerOutput:
    """Container holding token IDs and attention mask tensors."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class RCMTokenizer:
    """Thin wrapper over :class:`HuggingfaceTokenizer` for rCM."""

    def __init__(self, assets: TokenizerAssets) -> None:
        self.assets = assets
        self.tokenizer = HuggingfaceTokenizer(
            name=assets.text_model,
            seq_len=assets.max_length,
            clean=assets.clean_mode,
            cache_dir=assets.cache_dir,
        )

    def encode(self, prompts: Iterable[str]) -> TokenizerOutput:
        input_ids, attention_mask = self.tokenizer(
            list(prompts),
            return_mask=True,
            add_special_tokens=True,
        )
        return TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)


def build_conditioning(
    tokenizer: RCMTokenizer,
    prompts: Iterable[str],
    device: torch.device,
) -> TokenizerOutput:
    """Encode prompts and move tensors onto ``device``."""

    outputs = tokenizer.encode(prompts)
    return TokenizerOutput(
        input_ids=outputs.input_ids.to(device),
        attention_mask=outputs.attention_mask.to(device),
    )
