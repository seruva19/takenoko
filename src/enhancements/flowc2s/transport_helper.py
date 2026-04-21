from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger


logger = get_logger(__name__)


class FlowC2STransportHelper(nn.Module):
    """Train-time-only FlowC2S-style current-to-succeeding transport alignment.

    This is intentionally not a full FlowC2S continuation pipeline. Instead, it keeps
    Takenoko's standard inference path intact and adds an auxiliary loss that aligns
    a hooked transformer's current->future hidden-state transport with the same
    video's latent current->future transport.
    """

    def __init__(self, transformer: nn.Module, args: Any) -> None:
        super().__init__()
        self.args = args
        self.transformer = self._unwrap_model(transformer)
        self.enabled = bool(getattr(args, "enable_flowc2s_transport", False))
        self.block_index = getattr(args, "flowc2s_transport_block_index", None)
        self.weight = float(getattr(args, "flowc2s_transport_lambda", 0.1) or 0.0)
        self.loss_type = str(
            getattr(args, "flowc2s_transport_loss_type", "cosine")
        ).lower()
        self.chunk_ratio = float(
            getattr(args, "flowc2s_transport_chunk_ratio", 0.5) or 0.5
        )
        self.min_chunk_frames = int(
            getattr(args, "flowc2s_transport_min_chunk_frames", 2) or 2
        )
        self.normalize_latents = bool(
            getattr(args, "flowc2s_transport_normalize_latents", True)
        )

        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_hidden: Optional[torch.Tensor] = None

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _target_blocks(self) -> tuple[list[nn.Module], int]:
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None and hasattr(self.transformer, "module"):
            blocks = getattr(self.transformer.module, "blocks", None)
        if blocks is None:
            raise ValueError(
                "FlowC2STransportHelper could not find transformer blocks to hook"
            )
        return list(blocks), len(blocks)

    @staticmethod
    def _resolve_block_index(idx: Any, num_blocks: int) -> int:
        try:
            idx_int = int(idx)
        except Exception as exc:
            raise ValueError(
                f"FlowC2S transport block index {idx!r} is not an int."
            ) from exc
        if 0 <= idx_int < num_blocks:
            return idx_int
        raise ValueError(
            "FlowC2S transport block index "
            f"{idx_int} is outside available range [0, {num_blocks - 1}]"
        )

    def _make_hook(self):
        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (list, tuple)) and output:
                tensor = output[0]
            if torch.is_tensor(tensor):
                self._captured_hidden = tensor

        return hook

    def setup_hooks(self) -> None:
        if not self.enabled:
            return
        self.remove_hooks()
        blocks, num_blocks = self._target_blocks()
        if self.block_index is None:
            raise ValueError(
                "flowc2s_transport_block_index must be set when "
                "enable_flowc2s_transport=true"
            )
        student_idx = self._resolve_block_index(self.block_index, num_blocks)
        self.hooks.append(blocks[student_idx].register_forward_hook(self._make_hook()))

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks = []
        self._captured_hidden = None

    def compute_loss(
        self,
        *,
        latents: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[dict[str, float]]]:
        if not self.enabled or self.weight <= 0.0:
            return None, None
        if self._captured_hidden is None:
            raise ValueError(
                "FlowC2S transport is enabled but no intermediate hidden states "
                "were captured."
            )
        if latents is None:
            raise ValueError("FlowC2S transport requires clean latents.")
        if latents.dim() != 5:
            raise ValueError(
                f"FlowC2S transport expects 5D latents [B,C,F,H,W], got {latents.shape}"
            )

        frame_count = int(latents.shape[2])
        split_index = self._resolve_split_index(frame_count)
        if split_index is None:
            self._captured_hidden = None
            return None, None

        student_transport = self._student_transport(self._captured_hidden, frame_count, split_index)
        target_transport = self._target_transport(latents, split_index)

        if student_transport.shape[-1] != target_transport.shape[-1]:
            student_transport = self._resize_last_dim(
                student_transport, target_transport.shape[-1]
            )

        cosine = F.cosine_similarity(
            F.normalize(student_transport, dim=-1, eps=1e-6),
            F.normalize(target_transport, dim=-1, eps=1e-6),
            dim=-1,
        )

        if self.loss_type == "cosine":
            align_loss = (1.0 - cosine).mean() * self.weight
        else:
            align_loss = (
                F.mse_loss(student_transport, target_transport, reduction="mean")
                * self.weight
            )

        log_data = {
            "flowc2s_transport_similarity": float(cosine.mean().detach().item()),
            "flowc2s_transport_split_index": float(split_index),
        }
        self._captured_hidden = None
        return align_loss, log_data

    def _resolve_split_index(self, frame_count: int) -> Optional[int]:
        if frame_count < (self.min_chunk_frames * 2):
            return None
        split_index = int(round(frame_count * self.chunk_ratio))
        split_index = max(self.min_chunk_frames, split_index)
        split_index = min(frame_count - self.min_chunk_frames, split_index)
        if split_index <= 0 or split_index >= frame_count:
            return None
        return split_index

    def _student_transport(
        self,
        hidden_states: torch.Tensor,
        frame_count: int,
        split_index: int,
    ) -> torch.Tensor:
        if hidden_states.ndim == 4:
            pooled = hidden_states.float().mean(dim=2)
        elif hidden_states.ndim == 3:
            pooled = hidden_states.float()
        else:
            raise ValueError(
                "FlowC2S transport expected hidden states with 3 or 4 dims, "
                f"got {hidden_states.shape}"
            )

        if pooled.shape[1] != frame_count:
            pooled = self._resize_temporal(pooled, frame_count)

        current = pooled[:, :split_index, :].mean(dim=1)
        future = pooled[:, split_index:, :].mean(dim=1)
        return future - current

    def _target_transport(
        self,
        latents: torch.Tensor,
        split_index: int,
    ) -> torch.Tensor:
        pooled = latents.float().mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()
        current = pooled[:, :split_index, :].mean(dim=1)
        future = pooled[:, split_index:, :].mean(dim=1)
        delta = future - current
        if self.normalize_latents:
            delta = F.normalize(delta, dim=-1, eps=1e-6)
        return delta

    @staticmethod
    def _resize_temporal(values: torch.Tensor, target_frames: int) -> torch.Tensor:
        if values.shape[1] == target_frames:
            return values
        resized = F.interpolate(
            values.permute(0, 2, 1),
            size=target_frames,
            mode="linear",
            align_corners=False,
        )
        return resized.permute(0, 2, 1)

    @staticmethod
    def _resize_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
        if values.shape[-1] == target_dim:
            return values
        return F.interpolate(
            values.unsqueeze(1),
            size=target_dim,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
