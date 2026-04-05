import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from common.logger import get_logger
from wan.modules.model import WanSelfAttention

logger = get_logger(__name__)


class AttentionMapRecorder:
    def __init__(
        self,
        modules: List[tuple[str, torch.nn.Module]],
        *,
        max_queries: int,
        max_keys: int,
        capture_grad: bool,
        keep_heads: bool,
    ) -> None:
        self.modules = modules
        self.max_queries = max(1, int(max_queries))
        self.max_keys = max(1, int(max_keys))
        self.capture_grad = bool(capture_grad)
        self.keep_heads = bool(keep_heads)
        self._active_names: set[str] = set()

    def collect_maps(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, module in self.modules:
            if name not in self._active_names:
                continue
            attn_map = getattr(module, "_motion_record_attn_map", None)
            if isinstance(attn_map, torch.Tensor):
                out[name] = attn_map
        return out

    def __enter__(self) -> "AttentionMapRecorder":
        self._active_names = set()
        for name, module in self.modules:
            setattr(module, "_motion_record_enabled", True)
            setattr(module, "_motion_record_max_queries", self.max_queries)
            setattr(module, "_motion_record_max_keys", self.max_keys)
            setattr(module, "_motion_record_capture_grad", self.capture_grad)
            setattr(module, "_motion_record_keep_heads", self.keep_heads)
            setattr(module, "_motion_record_attn_map", None)
            self._active_names.add(name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for _, module in self.modules:
            setattr(module, "_motion_record_enabled", False)
            setattr(module, "_motion_record_attn_map", None)
        self._active_names = set()
        return False


def parse_block_index_spec(spec: Optional[str]) -> set[int]:
    if not spec:
        return set()
    out: set[int] = set()
    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(
                    f"Invalid block range in motion_attention_preservation_blocks: {token!r}"
                )
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def extract_self_attn_block_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:model\.)?blocks\.(\d+)\.self_attn$", module_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_motion_attention_modules(
    transformer: torch.nn.Module,
    block_spec: Optional[str],
) -> List[tuple[str, torch.nn.Module]]:
    block_filter = parse_block_index_spec(block_spec)
    apply_filter = len(block_filter) > 0
    out: List[tuple[str, torch.nn.Module]] = []
    for module_name, module in transformer.named_modules():
        if not isinstance(module, WanSelfAttention):
            continue
        block_index = extract_self_attn_block_index(module_name)
        if block_index is None:
            continue
        if apply_filter and block_index not in block_filter:
            continue
        out.append((module_name, module))
    return out


def filter_motion_attention_modules_for_swap(
    modules: List[tuple[str, torch.nn.Module]],
    *,
    transformer: torch.nn.Module,
    accelerator: Accelerator,
    blocks_to_swap: int,
) -> List[tuple[str, torch.nn.Module]]:
    if not modules or int(blocks_to_swap or 0) <= 0:
        return modules

    base_model = transformer.model if hasattr(transformer, "model") else transformer
    blocks = getattr(base_model, "blocks", None)
    if blocks is None:
        logger.warning(
            "motion_attention_preservation: block-swap filtering skipped because blocks are unavailable."
        )
        return modules

    num_blocks = len(blocks)
    if num_blocks <= 0:
        return modules
    swap_start = max(0, num_blocks - int(blocks_to_swap))
    if swap_start <= 0:
        return modules

    kept: List[tuple[str, torch.nn.Module]] = []
    dropped = 0
    for module_name, module in modules:
        block_idx = extract_self_attn_block_index(module_name)
        if block_idx is None or block_idx < swap_start:
            kept.append((module_name, module))
        else:
            dropped += 1
    if not kept:
        logger.warning(
            "motion_attention_preservation: block-swap filtering would drop all modules. Keeping original module list."
        )
        return modules

    logger.info(
        "motion_attention_preservation: filtered modules for block swap on %s: kept=%d dropped=%d",
        str(accelerator.device),
        len(kept),
        dropped,
    )
    return kept


def compute_motion_attention_loss(
    student_maps: Dict[str, torch.Tensor],
    teacher_maps: Optional[Dict[str, torch.Tensor]],
    *,
    loss_type: str,
    temperature: float,
    symmetric_kl: bool,
) -> Optional[torch.Tensor]:
    if not student_maps or not isinstance(teacher_maps, dict):
        return None

    per_block_losses: List[torch.Tensor] = []
    for module_name, student_map in student_maps.items():
        teacher_map = teacher_maps.get(module_name)
        if not isinstance(teacher_map, torch.Tensor):
            continue
        if teacher_map.shape != student_map.shape:
            logger.warning(
                "Motion attention preservation: shape mismatch for %s (student=%s teacher=%s).",
                module_name,
                tuple(student_map.shape),
                tuple(teacher_map.shape),
            )
            continue

        student_dist = student_map.to(torch.float32)
        teacher_dist = teacher_map.to(
            device=student_dist.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if temperature != 1.0:
            inv_temp = 1.0 / max(1e-6, temperature)
            student_dist = student_dist.clamp_min(1e-8).pow(inv_temp)
            teacher_dist = teacher_dist.clamp_min(1e-8).pow(inv_temp)
        student_dist = student_dist.clamp_min(1e-6)
        teacher_dist = teacher_dist.clamp_min(1e-6)
        student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        teacher_dist = teacher_dist / teacher_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if loss_type == "kl":
            kl_t2s = F.kl_div(student_dist.log(), teacher_dist, reduction="batchmean")
            if symmetric_kl:
                kl_s2t = F.kl_div(
                    teacher_dist.log(),
                    student_dist,
                    reduction="batchmean",
                )
                block_loss = 0.5 * (kl_t2s + kl_s2t)
            else:
                block_loss = kl_t2s
        else:
            block_loss = F.mse_loss(student_dist, teacher_dist)
        per_block_losses.append(block_loss)

    if not per_block_losses:
        return None
    return torch.stack(per_block_losses).mean()
