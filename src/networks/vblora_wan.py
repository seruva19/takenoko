
"""VB-LoRA network module for WAN models.

This implementation follows the VB-LoRA core idea used in PEFT:
- one shared trainable vector bank
- per-layer logits (A/B) over the bank
- top-k sparse composition for low-rank factors

The path is opt-in through `network_module = "networks.vblora_wan"`.
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)

_DEFAULT_NUM_VECTORS = 256
_DEFAULT_VECTOR_LENGTH = 256
_DEFAULT_TOPK = 2
_DEFAULT_DROPOUT = 0.0
_DEFAULT_INIT_VECTOR_BANK_BOUND = 0.02
_DEFAULT_INIT_LOGITS_STD = 0.1
_DEFAULT_BANK_LR_RATIO = 1.0
_DEFAULT_LOGITS_LR_RATIO = 1.0
_DEFAULT_TRAIN_VECTOR_BANK = True
_DEFAULT_SAVE_ONLY_TOPK_WEIGHTS = False

_TOPK_INDICES_SUFFIX = "_topk_indices"
_TOPK_WEIGHTS_SUFFIX = "_topk_weights"


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
        return default
    return bool(raw)


def _parse_optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _parse_patterns(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return None


def _select_vblora_indices_dtype(num_vectors: int) -> torch.dtype:
    if num_vectors < 2**8:
        return torch.uint8
    if num_vectors < 2**15:
        return torch.int16
    if num_vectors < 2**31:
        return torch.int32
    return torch.int64


def _decode_vblora_topk_state_dict(
    state_dict: Dict[str, Tensor],
    fallback_num_vectors: int,
) -> Dict[str, Tensor]:
    decoded = dict(state_dict)
    num_vectors = int(max(1, int(fallback_num_vectors)))

    bank = decoded.get("vblora_vector_bank", None)
    if isinstance(bank, Tensor) and bank.ndim == 2:
        num_vectors = int(bank.shape[0])

    keys = list(decoded.keys())
    for key in keys:
        if not key.endswith(_TOPK_INDICES_SUFFIX):
            continue
        if key not in decoded:
            continue

        weights_key = key[: -len(_TOPK_INDICES_SUFFIX)] + _TOPK_WEIGHTS_SUFFIX
        if weights_key not in decoded:
            logger.warning(
                "VB-LoRA checkpoint missing %s for %s; skipping compressed logit recovery.",
                weights_key,
                key,
            )
            continue

        indices = decoded[key]
        topk_weights = decoded[weights_key]
        if not isinstance(indices, Tensor) or not isinstance(topk_weights, Tensor):
            continue
        if indices.ndim == 0 or topk_weights.ndim == 0:
            continue
        if tuple(indices.shape[:-1]) != tuple(topk_weights.shape[:-1]):
            logger.warning(
                "VB-LoRA compressed tensor shape mismatch for %s and %s; skipping recovery.",
                key,
                weights_key,
            )
            continue

        if indices.numel() > 0:
            max_index = int(indices.max().item()) + 1
            if max_index > num_vectors:
                num_vectors = max_index

        full_topk_weights = torch.cat(
            [topk_weights, 1 - topk_weights.sum(dim=-1, keepdim=True)],
            dim=-1,
        )
        topk_logits = torch.log(full_topk_weights)
        recovered = (
            torch.zeros([*topk_logits.shape[:-1], num_vectors], device=topk_logits.device, dtype=topk_logits.dtype)
            .fill_(float("-inf"))
            .scatter(-1, indices.to(torch.long), topk_logits)
        )

        original_key = key[: -len(_TOPK_INDICES_SUFFIX)]
        decoded[original_key] = recovered
        del decoded[key]
        del decoded[weights_key]

    return decoded


def _encode_vblora_topk_state_dict(
    state_dict: Dict[str, Tensor],
    topk: int,
    num_vectors: int,
) -> Dict[str, Tensor]:
    encoded: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if "vblora_logits" not in key:
            encoded[key] = value
            continue

        effective_topk = max(1, min(int(topk), int(value.shape[-1])))
        logits, indices = value.topk(effective_topk, dim=-1)
        encoded[key + _TOPK_INDICES_SUFFIX] = indices.to(
            dtype=_select_vblora_indices_dtype(int(max(1, num_vectors)))
        )
        encoded[key + _TOPK_WEIGHTS_SUFFIX] = (
            torch.softmax(logits, dim=-1)[..., :-1].contiguous()
        )
    return encoded


class VBLoRAModule(LoRAModule):
    """LoRA module using VB-LoRA vector-bank composition."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        split_dims: Optional[List[int]] = None,
        initialize: Optional[str] = None,
        pissa_niter: Optional[int] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
        vblora_num_vectors: int = _DEFAULT_NUM_VECTORS,
        vblora_vector_length: int = _DEFAULT_VECTOR_LENGTH,
        vblora_topk: int = _DEFAULT_TOPK,
        vblora_dropout: float = _DEFAULT_DROPOUT,
        vblora_init_logits_std: float = _DEFAULT_INIT_LOGITS_STD,
    ) -> None:
        if split_dims is not None:
            raise ValueError("VB-LoRA does not support split_dims modules.")
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("VB-LoRA currently supports linear-like modules only.")

        in_features = getattr(org_module, "in_features", None)
        out_features = getattr(org_module, "out_features", None)
        if in_features is None or out_features is None:
            raise RuntimeError(
                "VB-LoRA requires linear-like modules with in_features/out_features."
            )

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=None,
            initialize=initialize,
            pissa_niter=pissa_niter,
            ggpo_sigma=ggpo_sigma,
            ggpo_beta=ggpo_beta,
        )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.vblora_num_vectors = int(max(1, int(vblora_num_vectors)))
        self.vblora_vector_length = int(max(1, int(vblora_vector_length)))
        self.vblora_topk = int(max(1, int(vblora_topk)))
        self.vblora_dropout = float(vblora_dropout)
        self.vblora_init_logits_std = float(vblora_init_logits_std)

        if self.vblora_num_vectors < self.vblora_topk:
            raise ValueError(
                f"vblora_num_vectors ({self.vblora_num_vectors}) must be >= vblora_topk ({self.vblora_topk})."
            )
        if self.in_features % self.vblora_vector_length != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by vblora_vector_length ({self.vblora_vector_length})."
            )
        if self.out_features % self.vblora_vector_length != 0:
            raise ValueError(
                f"out_features ({self.out_features}) must be divisible by vblora_vector_length ({self.vblora_vector_length})."
            )
        if not (0.0 <= self.vblora_dropout < 1.0):
            raise ValueError(f"vblora_dropout must be in [0, 1), got {self.vblora_dropout}.")
        if self.vblora_init_logits_std <= 0.0:
            raise ValueError(
                f"vblora_init_logits_std must be > 0, got {self.vblora_init_logits_std}."
            )

        # VB-LoRA does not train inherited LoRA up/down matrices.
        self.lora_down.weight.requires_grad_(False)  # type: ignore[attr-defined]
        self.lora_up.weight.requires_grad_(False)  # type: ignore[attr-defined]
        self._ggpo_enabled = False

        self.vblora_logits_A = nn.Parameter(
            torch.zeros(
                self.lora_dim,
                self.in_features // self.vblora_vector_length,
                self.vblora_num_vectors,
                dtype=torch.float32,
            )
        )
        self.vblora_logits_B = nn.Parameter(
            torch.zeros(
                self.out_features // self.vblora_vector_length,
                self.lora_dim,
                self.vblora_num_vectors,
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            nn.init.normal_(self.vblora_logits_A, 0.0, self.vblora_init_logits_std)
            nn.init.normal_(self.vblora_logits_B, 0.0, self.vblora_init_logits_std)

        self._shared_vector_bank: Optional[Tensor] = None

    def set_vector_bank(self, vector_bank: Tensor) -> None:
        self._shared_vector_bank = vector_bank

    def _resolve_vector_bank(self) -> Tensor:
        if self._shared_vector_bank is None:
            raise RuntimeError(
                "VB-LoRA shared vector bank is not attached to module "
                f"{self.lora_name}. Ensure network initialization completed."
            )
        return self._shared_vector_bank

    def _get_low_rank_matrix(self, logits: Tensor, vector_bank: Tensor, topk: int) -> Tensor:
        topk_logits, indices = logits.topk(topk, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        return (topk_weights.unsqueeze(-1) * vector_bank[indices]).sum(-2)

    def _get_lora_matrices(
        self,
        cast_to_fp32: bool = False,
        override_bank: Optional[Tensor] = None,
        override_logits_A: Optional[Tensor] = None,
        override_logits_B: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        logits_A = override_logits_A if override_logits_A is not None else self.vblora_logits_A
        logits_B = override_logits_B if override_logits_B is not None else self.vblora_logits_B
        vector_bank = override_bank if override_bank is not None else self._resolve_vector_bank()

        if self.training and logits_A[0, 0].isinf().any():
            raise RuntimeError(
                "Found infinity values in VB-LoRA logits. Ensure training was not resumed "
                "from a checkpoint saved with vblora_save_only_topk_weights=true."
            )

        vector_bank = vector_bank.to(logits_A.device)
        if cast_to_fp32:
            logits_A = logits_A.float()
            logits_B = logits_B.float()
            vector_bank = vector_bank.float()

        A = self._get_low_rank_matrix(logits_A, vector_bank, self.vblora_topk).reshape(
            logits_A.shape[0], -1
        )
        B = (
            self._get_low_rank_matrix(logits_B, vector_bank, self.vblora_topk)
            .transpose(1, 2)
            .reshape(-1, logits_B.shape[1])
        )
        return A, B

    def forward(self, x: Tensor) -> Tensor:
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < float(self.module_dropout):
                return org_forwarded

        lora_input = x
        if self.vblora_dropout > 0.0 and self.training:
            lora_input = F.dropout(lora_input, p=self.vblora_dropout)

        A, B = self._get_lora_matrices()
        lora_input = lora_input.to(self._resolve_vector_bank().dtype)
        rank_proj = F.linear(lora_input, A)

        scale = self.scale
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((rank_proj.shape[0], self.lora_dim), device=rank_proj.device)
                > float(self.rank_dropout)
            )
            if rank_proj.ndim == 3:
                mask = mask.unsqueeze(1)
            elif rank_proj.ndim == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            rank_proj = rank_proj * mask.to(dtype=rank_proj.dtype)
            scale = scale * (1.0 / max(1.0 - float(self.rank_dropout), 1e-6))

        delta = F.linear(rank_proj, B).to(dtype=org_forwarded.dtype, device=org_forwarded.device)
        return org_forwarded + delta * self.multiplier * scale

    def get_weight(self, multiplier: Optional[float] = None) -> Tensor:
        if multiplier is None:
            multiplier = self.multiplier
        cast_to_fp32 = self.vblora_logits_A.device.type == "cpu" and self.vblora_logits_A.dtype == torch.float16
        A, B = self._get_lora_matrices(cast_to_fp32=cast_to_fp32)
        delta = B @ A
        return delta * float(multiplier) * float(self.scale)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        del non_blocking
        if isinstance(sd, dict):
            sd = _decode_vblora_topk_state_dict(
                sd, fallback_num_vectors=self.vblora_num_vectors
            )
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        work_device = org_device if device is None else device
        work_dtype = org_dtype if dtype is None else dtype

        override_bank = None
        override_logits_A = None
        override_logits_B = None
        if isinstance(sd, dict):
            if "vblora_vector_bank" in sd:
                override_bank = sd["vblora_vector_bank"]
            if "vblora_logits_A" in sd:
                override_logits_A = sd["vblora_logits_A"]
            if "vblora_logits_B" in sd:
                override_logits_B = sd["vblora_logits_B"]

        A, B = self._get_lora_matrices(
            cast_to_fp32=True,
            override_bank=override_bank,
            override_logits_A=override_logits_A,
            override_logits_B=override_logits_B,
        )
        delta = (B @ A).to(device=work_device, dtype=torch.float32)

        merged = weight.to(device=work_device, dtype=torch.float32) + (
            delta * float(self.multiplier) * float(self.scale)
        )
        org_sd["weight"] = merged.to(device=org_device, dtype=work_dtype)
        self.org_module.load_state_dict(org_sd)


class VBLoRAInfModule(VBLoRAModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional[VBLoRANetwork] = None

    def set_network(self, network) -> None:
        self.network = network

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return self.org_forward(x)
        return super().forward(x)

class VBLoRANetwork(LoRANetwork):
    """LoRA network with shared VB-LoRA vector bank."""

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        vblora_num_vectors: int = _DEFAULT_NUM_VECTORS,
        vblora_vector_length: int = _DEFAULT_VECTOR_LENGTH,
        vblora_topk: int = _DEFAULT_TOPK,
        vblora_dropout: float = _DEFAULT_DROPOUT,
        vblora_init_vector_bank_bound: float = _DEFAULT_INIT_VECTOR_BANK_BOUND,
        vblora_init_logits_std: float = _DEFAULT_INIT_LOGITS_STD,
        vblora_bank_lr_ratio: float = _DEFAULT_BANK_LR_RATIO,
        vblora_logits_lr_ratio: float = _DEFAULT_LOGITS_LR_RATIO,
        vblora_train_vector_bank: bool = _DEFAULT_TRAIN_VECTOR_BANK,
        vblora_save_only_topk_weights: bool = _DEFAULT_SAVE_ONLY_TOPK_WEIGHTS,
        module_class: Optional[Type[object]] = None,
        **kwargs,
    ) -> None:
        self.vblora_num_vectors = int(max(1, int(vblora_num_vectors)))
        self.vblora_vector_length = int(max(1, int(vblora_vector_length)))
        self.vblora_topk = int(max(1, int(vblora_topk)))
        self.vblora_dropout = float(vblora_dropout)
        self.vblora_init_vector_bank_bound = float(vblora_init_vector_bank_bound)
        self.vblora_init_logits_std = float(vblora_init_logits_std)
        self.vblora_bank_lr_ratio = float(vblora_bank_lr_ratio)
        self.vblora_logits_lr_ratio = float(vblora_logits_lr_ratio)
        self.vblora_train_vector_bank = bool(vblora_train_vector_bank)
        self.vblora_save_only_topk_weights = bool(vblora_save_only_topk_weights)

        if self.vblora_num_vectors < self.vblora_topk:
            raise ValueError(
                f"vblora_num_vectors ({self.vblora_num_vectors}) must be >= vblora_topk ({self.vblora_topk})."
            )
        if not (0.0 <= self.vblora_dropout < 1.0):
            raise ValueError(f"vblora_dropout must be in [0, 1), got {self.vblora_dropout}.")
        if self.vblora_init_vector_bank_bound <= 0.0:
            raise ValueError(
                "vblora_init_vector_bank_bound must be > 0, "
                f"got {self.vblora_init_vector_bank_bound}."
            )
        if self.vblora_init_logits_std <= 0.0:
            raise ValueError(
                f"vblora_init_logits_std must be > 0, got {self.vblora_init_logits_std}."
            )
        if self.vblora_bank_lr_ratio <= 0.0:
            raise ValueError(
                f"vblora_bank_lr_ratio must be > 0, got {self.vblora_bank_lr_ratio}."
            )
        if self.vblora_logits_lr_ratio <= 0.0:
            raise ValueError(
                f"vblora_logits_lr_ratio must be > 0, got {self.vblora_logits_lr_ratio}."
            )

        self.vblora_vector_bank = nn.Parameter(
            torch.zeros(self.vblora_num_vectors, self.vblora_vector_length, dtype=torch.float32)
        )
        with torch.no_grad():
            nn.init.uniform_(
                self.vblora_vector_bank,
                -self.vblora_init_vector_bank_bound,
                self.vblora_init_vector_bank_bound,
            )
        self.vblora_vector_bank.requires_grad_(self.vblora_train_vector_bank)

        super().__init__(
            target_replace_modules=target_replace_modules,
            prefix=prefix,
            text_encoders=text_encoders,
            unet=unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            module_class=module_class or self._create_vblora_module,
            **kwargs,
        )
        self._attach_shared_vector_bank()

    def _create_vblora_module(self, lora_name, org_module, multiplier, lora_dim, alpha, **kwargs):
        return VBLoRAModule(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            vblora_num_vectors=self.vblora_num_vectors,
            vblora_vector_length=self.vblora_vector_length,
            vblora_topk=self.vblora_topk,
            vblora_dropout=self.vblora_dropout,
            vblora_init_logits_std=self.vblora_init_logits_std,
            **kwargs,
        )

    def _attach_shared_vector_bank(self) -> None:
        for module in self.text_encoder_loras + self.unet_loras:
            if isinstance(module, VBLoRAModule):
                module.set_vector_bank(self.vblora_vector_bank)

    def prepare_network(self, args) -> None:
        logger.info(
            "VB-LoRA enabled (vectors=%s, vector_length=%s, topk=%s, dropout=%s, train_bank=%s, save_only_topk=%s).",
            self.vblora_num_vectors,
            self.vblora_vector_length,
            self.vblora_topk,
            self.vblora_dropout,
            self.vblora_train_vector_bank,
            self.vblora_save_only_topk_weights,
        )

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, input_lr_scale: float = 1.0, **kwargs):
        del input_lr_scale, kwargs
        self.requires_grad_(False)

        logits_params: Dict[str, nn.Parameter] = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            if not isinstance(lora, VBLoRAModule):
                continue
            lora.vblora_logits_A.requires_grad_(True)
            lora.vblora_logits_B.requires_grad_(True)
            logits_params[f"{lora.lora_name}.vblora_logits_A"] = lora.vblora_logits_A
            logits_params[f"{lora.lora_name}.vblora_logits_B"] = lora.vblora_logits_B

        all_params: List[Dict[str, Any]] = []
        lr_descriptions: List[str] = []

        if logits_params:
            logits_group: Dict[str, Any] = {"params": logits_params.values()}
            if unet_lr is not None:
                logits_group["lr"] = float(unet_lr) * float(self.vblora_logits_lr_ratio)
            all_params.append(logits_group)
            lr_descriptions.append("unet")

        if not self.vblora_train_vector_bank:
            self.vblora_vector_bank.requires_grad_(False)
            return all_params, lr_descriptions

        self.vblora_vector_bank.requires_grad_(True)
        bank_group: Dict[str, Any] = {"params": [self.vblora_vector_bank]}
        if unet_lr is not None:
            bank_group["lr"] = float(unet_lr) * float(self.vblora_bank_lr_ratio)
            if bank_group["lr"] <= 0:
                return all_params, lr_descriptions
        all_params.append(bank_group)
        lr_descriptions.append("unet vblora_bank")
        return all_params, lr_descriptions

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        del text_encoders, unet
        decoded_weights_sd = _decode_vblora_topk_state_dict(
            weights_sd,
            fallback_num_vectors=self.vblora_num_vectors,
        )
        vector_bank = decoded_weights_sd.get("vblora_vector_bank", self.vblora_vector_bank)
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for lora in self.text_encoder_loras + self.unet_loras:
                sd_for_lora = {}
                for key in decoded_weights_sd.keys():
                    if key.startswith(lora.lora_name):
                        sd_for_lora[key[len(lora.lora_name) + 1 :]] = decoded_weights_sd[key]
                if not sd_for_lora:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue
                sd_for_lora["vblora_vector_bank"] = vector_bank
                futures.append(
                    executor.submit(
                        lora.merge_to,
                        sd_for_lora,
                        dtype,
                        device,
                        non_blocking,  # type: ignore[arg-type]
                    )
                )
        for future in futures:
            future.result()
        logger.info("weights are merged")

    def load_state_dict(self, state_dict, strict=True):
        decoded_state_dict = _decode_vblora_topk_state_dict(
            state_dict,
            fallback_num_vectors=self.vblora_num_vectors,
        )
        return super().load_state_dict(decoded_state_dict, strict=strict)

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if self.vblora_save_only_topk_weights:
            state_dict = _encode_vblora_topk_state_dict(
                state_dict,
                topk=self.vblora_topk,
                num_vectors=self.vblora_num_vectors,
            )

        if dtype is not None:
            for key in list(state_dict.keys()):
                value = state_dict[key]
                value = value.detach().clone().to("cpu")
                if torch.is_floating_point(value):
                    value = value.to(dtype)
                state_dict[key] = value

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from utils import model_utils

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(
                state_dict,
                metadata,
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
            return

        torch.save(state_dict, file)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> VBLoRANetwork:
    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules, default=False)

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    excluded_parts = ["patch_embedding", "text_embedding", "norm", "head"]
    if not bool(include_time_modules):
        excluded_parts.extend(["time_embedding", "time_projection"])
    exclude_patterns.append(r".*(" + "|".join(excluded_parts) + r").*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        prefix="vblora_unet",
        multiplier=multiplier,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> VBLoRANetwork:
    del vae
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    rank_dropout = _parse_optional_float(kwargs.get("rank_dropout", None))
    module_dropout = _parse_optional_float(kwargs.get("module_dropout", None))
    loraplus_lr_ratio = _parse_optional_float(kwargs.get("loraplus_lr_ratio", None))

    include_patterns = _parse_patterns(kwargs.get("include_patterns", None))
    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns", None))
    extra_include_patterns = _parse_patterns(kwargs.get("extra_include_patterns", None))
    extra_exclude_patterns = _parse_patterns(kwargs.get("extra_exclude_patterns", None))

    include_time_modules = _parse_bool(kwargs.get("include_time_modules", False), False)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    vector_length_raw = kwargs.get("vblora_vector_length", kwargs.get("vblora_vector_dim", _DEFAULT_VECTOR_LENGTH))
    init_logits_std_raw = kwargs.get("vblora_init_logits_std", kwargs.get("vblora_init_std", _DEFAULT_INIT_LOGITS_STD))
    vblora_dropout_raw = kwargs.get("vblora_dropout", kwargs.get("dropout", _DEFAULT_DROPOUT))

    vblora_num_vectors = int(kwargs.get("vblora_num_vectors", _DEFAULT_NUM_VECTORS))
    vblora_vector_length = int(vector_length_raw)
    vblora_topk = int(kwargs.get("vblora_topk", _DEFAULT_TOPK))
    vblora_dropout = float(vblora_dropout_raw if vblora_dropout_raw is not None else _DEFAULT_DROPOUT)
    vblora_init_vector_bank_bound = float(
        kwargs.get("vblora_init_vector_bank_bound", _DEFAULT_INIT_VECTOR_BANK_BOUND)
    )
    vblora_init_logits_std = float(init_logits_std_raw)
    vblora_bank_lr_ratio = float(kwargs.get("vblora_bank_lr_ratio", _DEFAULT_BANK_LR_RATIO))
    vblora_logits_lr_ratio = float(kwargs.get("vblora_logits_lr_ratio", _DEFAULT_LOGITS_LR_RATIO))
    vblora_train_vector_bank = _parse_bool(
        kwargs.get("vblora_train_vector_bank", _DEFAULT_TRAIN_VECTOR_BANK),
        _DEFAULT_TRAIN_VECTOR_BANK,
    )
    vblora_save_only_topk_weights = _parse_bool(
        kwargs.get(
            "vblora_save_only_topk_weights",
            _DEFAULT_SAVE_ONLY_TOPK_WEIGHTS,
        ),
        _DEFAULT_SAVE_ONLY_TOPK_WEIGHTS,
    )

    if vblora_num_vectors < 1:
        raise ValueError("vblora_num_vectors must be >= 1.")
    if vblora_vector_length < 1:
        raise ValueError("vblora_vector_length must be >= 1.")
    if vblora_topk < 1:
        raise ValueError("vblora_topk must be >= 1.")
    if vblora_topk > vblora_num_vectors:
        raise ValueError("vblora_topk must be <= vblora_num_vectors.")
    if not (0.0 <= vblora_dropout < 1.0):
        raise ValueError("vblora_dropout must be in [0, 1).")
    if vblora_init_vector_bank_bound <= 0.0:
        raise ValueError("vblora_init_vector_bank_bound must be > 0.")
    if vblora_init_logits_std <= 0.0:
        raise ValueError("vblora_init_logits_std must be > 0.")
    if vblora_bank_lr_ratio <= 0.0:
        raise ValueError("vblora_bank_lr_ratio must be > 0.")
    if vblora_logits_lr_ratio <= 0.0:
        raise ValueError("vblora_logits_lr_ratio must be > 0.")

    network = VBLoRANetwork(
        target_replace_modules=target_replace_modules,
        prefix=prefix,
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=float(network_alpha),
        dropout=float(neuron_dropout) if neuron_dropout is not None else None,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=None,
        conv_alpha=None,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        vblora_num_vectors=vblora_num_vectors,
        vblora_vector_length=vblora_vector_length,
        vblora_topk=vblora_topk,
        vblora_dropout=vblora_dropout,
        vblora_init_vector_bank_bound=vblora_init_vector_bank_bound,
        vblora_init_logits_std=vblora_init_logits_std,
        vblora_bank_lr_ratio=vblora_bank_lr_ratio,
        vblora_logits_lr_ratio=vblora_logits_lr_ratio,
        vblora_train_vector_bank=vblora_train_vector_bank,
        vblora_save_only_topk_weights=vblora_save_only_topk_weights,
    )

    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    logger.info(
        "VB-LoRA initialized: rank=%s, vectors=%s, vector_length=%s, topk=%s, save_only_topk=%s",
        int(network_dim),
        vblora_num_vectors,
        vblora_vector_length,
        vblora_topk,
        vblora_save_only_topk_weights,
    )
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> VBLoRANetwork:
    return create_network_from_weights(
        target_replace_modules=WAN_TARGET_REPLACE_MODULES,
        multiplier=multiplier,
        weights_sd=weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> VBLoRANetwork:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    inferred_num_vectors = int(kwargs.get("vblora_num_vectors", _DEFAULT_NUM_VECTORS))
    inferred_vector_length = int(kwargs.get("vblora_vector_length", kwargs.get("vblora_vector_dim", _DEFAULT_VECTOR_LENGTH)))

    for key, value in weights_sd.items():
        if "." in key:
            lora_name = key.split(".")[0]
            if key.endswith(".alpha"):
                modules_alpha[lora_name] = value
            elif key.endswith(".vblora_logits_A"):
                modules_dim[lora_name] = int(value.shape[0])
                inferred_num_vectors = int(value.shape[-1])
            elif key.endswith(f".vblora_logits_A{_TOPK_INDICES_SUFFIX}"):
                modules_dim[lora_name] = int(value.shape[0])
                if isinstance(value, Tensor) and value.numel() > 0:
                    inferred_num_vectors = max(
                        inferred_num_vectors,
                        int(value.max().item()) + 1,
                    )
        elif key == "vblora_vector_bank" and value.ndim == 2:
            inferred_num_vectors = int(value.shape[0])
            inferred_vector_length = int(value.shape[1])

    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules, default=False)
    include_time_modules = bool(include_time_modules)
    if not extra_include_patterns:
        extra_include_patterns = []
    if include_time_modules:
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)
    else:
        for lora_name in modules_dim.keys():
            if "time_embedding" in lora_name or "time_projection" in lora_name:
                for pattern in ("^time_embedding\\.", "^time_projection\\."):
                    if pattern not in extra_include_patterns:
                        extra_include_patterns.append(pattern)
                break

    vblora_topk = int(kwargs.get("vblora_topk", _DEFAULT_TOPK))
    vblora_dropout = float(kwargs.get("vblora_dropout", _DEFAULT_DROPOUT))
    vblora_init_vector_bank_bound = float(
        kwargs.get("vblora_init_vector_bank_bound", _DEFAULT_INIT_VECTOR_BANK_BOUND)
    )
    vblora_init_logits_std = float(kwargs.get("vblora_init_logits_std", kwargs.get("vblora_init_std", _DEFAULT_INIT_LOGITS_STD)))
    vblora_bank_lr_ratio = float(kwargs.get("vblora_bank_lr_ratio", _DEFAULT_BANK_LR_RATIO))
    vblora_logits_lr_ratio = float(kwargs.get("vblora_logits_lr_ratio", _DEFAULT_LOGITS_LR_RATIO))
    vblora_train_vector_bank = _parse_bool(
        kwargs.get("vblora_train_vector_bank", _DEFAULT_TRAIN_VECTOR_BANK),
        _DEFAULT_TRAIN_VECTOR_BANK,
    )
    vblora_save_only_topk_weights = _parse_bool(
        kwargs.get(
            "vblora_save_only_topk_weights",
            _DEFAULT_SAVE_ONLY_TOPK_WEIGHTS,
        ),
        _DEFAULT_SAVE_ONLY_TOPK_WEIGHTS,
    )

    module_impl: Type[object] = VBLoRAInfModule if for_inference else VBLoRAModule

    def _module_factory(lora_name, org_module, module_multiplier, lora_dim, alpha, **kw):
        return module_impl(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=module_multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            vblora_num_vectors=inferred_num_vectors,
            vblora_vector_length=inferred_vector_length,
            vblora_topk=vblora_topk,
            vblora_dropout=vblora_dropout,
            vblora_init_logits_std=vblora_init_logits_std,
            **kw,
        )

    network = VBLoRANetwork(
        target_replace_modules=target_replace_modules,
        prefix="vblora_unet",
        text_encoders=text_encoders,  # type: ignore[arg-type]
        unet=unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        lora_dim=1,
        alpha=1.0,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=_module_factory,
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        vblora_num_vectors=inferred_num_vectors,
        vblora_vector_length=inferred_vector_length,
        vblora_topk=vblora_topk,
        vblora_dropout=vblora_dropout,
        vblora_init_vector_bank_bound=vblora_init_vector_bank_bound,
        vblora_init_logits_std=vblora_init_logits_std,
        vblora_bank_lr_ratio=vblora_bank_lr_ratio,
        vblora_logits_lr_ratio=vblora_logits_lr_ratio,
        vblora_train_vector_bank=vblora_train_vector_bank,
        vblora_save_only_topk_weights=vblora_save_only_topk_weights,
    )

    bank = weights_sd.get("vblora_vector_bank", None)
    if isinstance(bank, Tensor):
        if tuple(bank.shape) != tuple(network.vblora_vector_bank.shape):
            logger.warning(
                "VB-LoRA vector bank shape mismatch in checkpoint: expected %s, got %s.",
                tuple(network.vblora_vector_bank.shape),
                tuple(bank.shape),
            )
        else:
            with torch.no_grad():
                network.vblora_vector_bank.copy_(bank.to(dtype=torch.float32))
    return network
