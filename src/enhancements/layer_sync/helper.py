import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class LayerSyncHelper(nn.Module):
    """
    Capture intermediate transformer activations and compute the LayerSync projection loss.

    The helper registers forward hooks on configured transformer blocks, normalizes their
    outputs, and returns a cosine-style alignment loss across one or more (source, target)
    block pairs. All indices are 1-based to match existing DiT-style configuration.
    """

    def __init__(self, transformer: nn.Module, args: any) -> None:  # noqa: ANN401
        super().__init__()
        self.args = args
        self.transformer = self._unwrap_model(transformer)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, torch.Tensor] = {}
        self._warned_missing: bool = False
        self.last_similarity: Optional[torch.Tensor] = None
        self._pair_labels: List[str] = []

        self.pairs = self._collect_pairs()
        self.pair_weights = self._collect_pair_weights()
        self.detach_guidance: bool = bool(
            getattr(args, "layer_sync_detach_guidance", True)
        )
        self.normalization: str = str(
            getattr(args, "layer_sync_normalization", "cosine")
        ).lower()
        if self.normalization not in ("cosine",):
            logger.warning(
                "LayerSync: unsupported normalization '%s', falling back to cosine",
                self.normalization,
            )
            self.normalization = "cosine"

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _collect_pairs(self) -> List[Tuple[int, int]]:
        raw_pairs = getattr(self.args, "layer_sync_pairs", None)
        if isinstance(raw_pairs, Sequence):
            pairs: List[Tuple[int, int]] = []
            for pair in raw_pairs:
                if (
                    isinstance(pair, Sequence)
                    and len(pair) == 2
                    and all(isinstance(v, (int, float)) for v in pair)
                ):
                    src, tgt = int(pair[0]), int(pair[1])
                    if src > 0 and tgt > 0 and src < tgt:
                        pairs.append((src, tgt))
            if pairs:
                return pairs

        # Fallback to single pair
        src_block = int(getattr(self.args, "layer_sync_source_block", 8))
        tgt_block = int(getattr(self.args, "layer_sync_target_block", 16))
        return [(src_block, tgt_block)]

    def _collect_pair_weights(self) -> List[float]:
        raw_weights = getattr(self.args, "layer_sync_pair_weights", None)
        if isinstance(raw_weights, Sequence):
            weights: List[float] = []
            for w in raw_weights:
                try:
                    weight = float(w)
                except Exception:
                    continue
                if weight >= 0:
                    weights.append(weight)
            if weights and len(weights) == len(self._collect_pairs()):
                return weights
        # default: equal weighting
        return [1.0 for _ in self.pairs]

    def _target_blocks(self) -> Tuple[List[nn.Module], int]:
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None and hasattr(self.transformer, "module"):
            blocks = getattr(self.transformer.module, "blocks", None)
        if blocks is None:
            raise ValueError("LayerSyncHelper could not find transformer blocks to hook")
        return list(blocks), len(blocks)

    @staticmethod
    def _resolve_block_index(idx: int, num_blocks: int) -> Optional[int]:
        candidates = [idx - 1, idx]
        for cand in candidates:
            if 0 <= cand < num_blocks:
                return cand + 1
        return None

    def _make_hook(self, block_idx: int):
        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (list, tuple)) and output:
                tensor = output[0]
            if not torch.is_tensor(tensor):
                return
            self.activations[block_idx] = tensor

        return hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        try:
            blocks, num_blocks = self._target_blocks()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(f"LayerSync: unable to attach hooks: {exc}")
            return

        if hasattr(self.transformer, "blocks_to_swap"):
            try:
                blocks_to_swap = int(getattr(self.transformer, "blocks_to_swap", 0))
                if blocks_to_swap > 0:
                    logger.warning(
                        "LayerSync: block swap/offload detected (%s blocks). "
                        "Hooks may miss swapped blocks; monitor activation warnings.",
                        blocks_to_swap,
                    )
            except Exception:
                pass

        resolved_pairs: List[Tuple[int, int]] = []
        resolved_weights: List[float] = []
        resolved_indices: Dict[int, int] = {}
        for pair_idx, (src_idx, tgt_idx) in enumerate(self.pairs):
            resolved_src = self._resolve_block_index(src_idx, num_blocks)
            resolved_tgt = self._resolve_block_index(tgt_idx, num_blocks)
            if resolved_src is None or resolved_tgt is None:
                logger.warning(
                    "LayerSync: requested blocks %s->%s outside available range [0, %s]",
                    src_idx,
                    tgt_idx,
                    num_blocks - 1,
                )
                continue
            resolved_pairs.append((resolved_src, resolved_tgt))
            weight = (
                self.pair_weights[pair_idx]
                if pair_idx < len(self.pair_weights)
                else 1.0
            )
            resolved_weights.append(weight)
            resolved_indices[src_idx] = resolved_src
            resolved_indices[tgt_idx] = resolved_tgt

        self.pairs = resolved_pairs
        self.pair_weights = resolved_weights if resolved_weights else self.pair_weights
        self._pair_labels = [f"{src}->{tgt}" for src, tgt in self.pairs]

        required_indices = sorted({idx for pair in self.pairs for idx in pair})
        for idx in required_indices:
            handle = blocks[idx - 1].register_forward_hook(self._make_hook(idx))
            self.hooks.append(handle)

        if self.hooks:
            logger.info(
                "LayerSync: attached hooks for %s pairs (%s)",
                len(self.pairs),
                ", ".join(self._pair_labels),
            )
        else:
            logger.warning("LayerSync: no valid hooks were attached; feature disabled")

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks = []
        self.activations.clear()
        self.last_similarity = None

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() < 3:
            tensor = tensor.view(tensor.size(0), -1, tensor.size(-1))
        tensor = tensor.flatten(1, -2) if tensor.dim() > 3 else tensor
        return F.normalize(tensor, dim=-1)

    def compute_loss(self) -> Optional[torch.Tensor]:
        if not self.hooks:
            return None

        losses: List[torch.Tensor] = []
        similarities: List[torch.Tensor] = []
        missing_pairs: List[Tuple[int, int]] = []

        for pair_idx, (src_idx, tgt_idx) in enumerate(self.pairs):
            src_act = self.activations.get(src_idx)
            tgt_act = self.activations.get(tgt_idx)
            if src_act is None or tgt_act is None:
                missing_pairs.append((src_idx, tgt_idx))
                continue

            src_norm = self._normalize(src_act)
            tgt_norm = self._normalize(tgt_act)
            if self.detach_guidance:
                tgt_norm = tgt_norm.detach()

            weight = self.pair_weights[pair_idx] if pair_idx < len(self.pair_weights) else 1.0
            if src_norm.shape[1] != tgt_norm.shape[1]:
                min_tokens = min(src_norm.shape[1], tgt_norm.shape[1])
                src_norm = src_norm[:, :min_tokens]
                tgt_norm = tgt_norm[:, :min_tokens]

            similarity = (src_norm * tgt_norm).sum(dim=-1).mean()
            losses.append(weight * (-1.0 * similarity))
            similarities.append(weight * similarity)

        self.activations.clear()

        if missing_pairs and not self._warned_missing:
            logger.warning(
                "LayerSync: missing activations for pairs: %s",
                ", ".join(f"{s}->{t}" for s, t in missing_pairs),
            )
            self._warned_missing = True

        if not losses:
            return None

        total_weight = sum(self.pair_weights) if self.pair_weights else float(len(losses))
        total_weight = total_weight if total_weight > 0 else float(len(losses))
        total_loss = torch.stack(losses).sum() / total_weight
        self.last_similarity = torch.stack(similarities).sum() / total_weight
        return total_loss
