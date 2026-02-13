"""Wan-safe DeT temporal adapter using guarded self-attention output hooks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common.logger import get_logger

logger = get_logger(__name__)


class _TemporalConvAdapter(nn.Module):
    """Local temporal bottleneck with a gated residual output."""

    def __init__(
        self,
        channels: int,
        rank: int,
        kernel_size: int,
        gate_init: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.channels = int(channels)
        self.conv1 = nn.Conv1d(
            self.channels,
            int(rank),
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(
            int(rank),
            self.channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class WanDeTAdapterHelper(nn.Module):
    """Attach a shape-safe temporal adapter to Wan self-attention outputs."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model
        self.args = args

        self.alignment_depths = self._parse_alignment_depths(args)
        self.rank = int(getattr(args, "det_adapter_rank", 128))
        self.kernel_size = int(getattr(args, "det_adapter_kernel_size", 3))
        self.gate_init = float(getattr(args, "det_adapter_gate_init", 0.0))
        self.gate_max = float(getattr(args, "det_adapter_gate_max", 0.25))
        self.require_uniform_grid = bool(
            getattr(args, "det_adapter_require_uniform_grid", True)
        )
        self.allow_sparse_attention = bool(
            getattr(args, "det_adapter_allow_sparse_attention", False)
        )
        self.follow_det_locality_scale = bool(
            getattr(args, "det_adapter_follow_det_locality_scale", False)
        )
        self.locality_scale_source = str(
            getattr(args, "det_adapter_locality_scale_source", "attention_probe")
        ).lower()
        self.locality_min_scale = float(
            getattr(args, "det_adapter_locality_min_scale", 0.0)
        )
        self.gate_warmup_enabled = bool(
            getattr(args, "det_adapter_gate_warmup_enabled", False)
        )
        self.gate_warmup_steps = int(
            getattr(args, "det_adapter_gate_warmup_steps", 0)
        )
        self.gate_warmup_shape = str(
            getattr(args, "det_adapter_gate_warmup_shape", "linear")
        ).lower()
        self.unified_controller_enabled = bool(
            getattr(args, "det_unified_controller_enabled", False)
        )
        self.unified_controller_apply_to_adapter = bool(
            getattr(args, "det_unified_controller_apply_to_adapter", False)
        )
        self.unified_controller_min_scale = float(
            getattr(args, "det_unified_controller_min_scale", 0.1)
        )

        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.adapters: Dict[int, _TemporalConvAdapter] = {}
        self._warned_non_uniform = False
        self._warned_bad_layout = False
        self._warned_sparse = False
        self._warned_missing_locality_scale = False
        self._warned_missing_unified_scale = False

        self._initialize_adapters()

    @staticmethod
    def _parse_alignment_depths(args: Any) -> List[int]:
        raw_depths = getattr(args, "det_adapter_alignment_depths", None)
        if isinstance(raw_depths, Sequence) and not isinstance(raw_depths, (str, bytes)):
            parsed = [int(v) for v in raw_depths]
        else:
            parsed = [int(getattr(args, "det_adapter_alignment_depth", 8))]
        deduped: List[int] = []
        for depth in parsed:
            if depth not in deduped:
                deduped.append(depth)
        return deduped or [8]

    def _locate_blocks(self) -> Sequence[nn.Module]:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks  # type: ignore[return-value]
        if hasattr(self.diffusion_model, "module") and hasattr(self.diffusion_model.module, "blocks"):
            return self.diffusion_model.module.blocks  # type: ignore[return-value]
        raise ValueError("DeT adapter: could not locate Wan transformer blocks.")

    @staticmethod
    def _resolve_depth(depth: int, num_blocks: int) -> Optional[int]:
        if -num_blocks <= depth < num_blocks:
            return depth % num_blocks
        return None

    @staticmethod
    def _infer_channels(block: nn.Module) -> int:
        if hasattr(block, "dim"):
            return int(block.dim)
        self_attn = getattr(block, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "dim"):
            return int(self_attn.dim)
        raise ValueError("DeT adapter: could not infer block hidden width.")

    def _initialize_adapters(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        resolved_depths: List[int] = []
        for depth in self.alignment_depths:
            resolved = self._resolve_depth(depth, num_blocks)
            if resolved is None:
                logger.warning(
                    "DeT adapter: alignment depth %s is outside [-%s, %s).",
                    depth,
                    num_blocks,
                    num_blocks,
                )
                continue
            block = blocks[resolved]
            channels = self._infer_channels(block)
            adapter = _TemporalConvAdapter(
                channels=channels,
                rank=self.rank,
                kernel_size=self.kernel_size,
                gate_init=self.gate_init,
            )
            for param in adapter.parameters():
                param.requires_grad_(True)
            setattr(block, "_det_adapter_module", adapter)
            self.adapters[resolved] = adapter
            resolved_depths.append(resolved)

        if not resolved_depths:
            raise ValueError("DeT adapter: no valid alignment depth resolved.")
        self.alignment_depths = resolved_depths
        logger.info(
            "DeT adapter: attached trainable modules to block indices %s.",
            ", ".join(str(v) for v in self.alignment_depths),
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for depth in self.alignment_depths:
            adapter = self.adapters.get(depth)
            if adapter is None:
                continue
            params.extend(list(adapter.parameters()))
        return params

    @staticmethod
    def _extract_scalar(value: Any, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if torch.is_tensor(value) and value.numel() > 0:
            return int(value.reshape(-1)[0].item())
        return int(default)

    @staticmethod
    def _restore_output_shape(original_output: Any, new_hidden: torch.Tensor) -> Any:
        if torch.is_tensor(original_output):
            return new_hidden
        if isinstance(original_output, tuple) and len(original_output) > 0:
            return (new_hidden, *original_output[1:])
        if isinstance(original_output, list) and len(original_output) > 0:
            output_list = list(original_output)
            output_list[0] = new_hidden
            return output_list
        return original_output

    def _resolve_runtime_locality_scale(self) -> float:
        if not self.follow_det_locality_scale:
            return 1.0

        if self.locality_scale_source == "locality_adaptive":
            attr_name = "_det_locality_scale"
        else:
            attr_name = "_det_attention_locality_scale"
        raw_value = getattr(self.args, attr_name, None)
        if raw_value is None:
            if not self._warned_missing_locality_scale:
                logger.info(
                    "DeT adapter locality-follow enabled but %s is unavailable; using scale=1.0 until metrics are published.",
                    attr_name,
                )
                self._warned_missing_locality_scale = True
            return 1.0

        try:
            value = float(raw_value)
        except Exception:
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(self.locality_min_scale, min(1.0, value))

    def _resolve_runtime_warmup_scale(self) -> float:
        if not self.gate_warmup_enabled or self.gate_warmup_steps <= 0:
            return 1.0
        raw_step = getattr(self.args, "current_step", None)
        try:
            if torch.is_tensor(raw_step):
                step_value = int(raw_step.detach().reshape(-1)[0].item())
            elif raw_step is None:
                step_value = 0
            else:
                step_value = int(raw_step)
        except Exception:
            step_value = 0

        progress = min(max((float(step_value) + 1.0) / float(self.gate_warmup_steps), 0.0), 1.0)
        if self.gate_warmup_shape == "cosine":
            scale = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            scale = progress
        if not math.isfinite(scale):
            return 1.0
        return max(0.0, min(1.0, float(scale)))

    def _resolve_runtime_unified_scale(self) -> float:
        if not self.unified_controller_enabled or not self.unified_controller_apply_to_adapter:
            return 1.0

        raw_value = getattr(self.args, "_det_unified_controller_scale", None)
        if raw_value is None:
            if not self._warned_missing_unified_scale:
                logger.info(
                    "DeT adapter unified-controller scaling enabled but _det_unified_controller_scale is unavailable; using scale=1.0 until metrics are published."
                )
                self._warned_missing_unified_scale = True
            return 1.0

        try:
            value = float(raw_value)
        except Exception:
            return 1.0
        if not math.isfinite(value):
            return 1.0
        min_scale = max(0.0, min(1.0, float(self.unified_controller_min_scale)))
        return max(min_scale, min(1.0, value))

    def _build_hook(self, depth: int):
        def hook(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            adapter = self.adapters.get(depth)
            if adapter is None:
                return output

            hidden = output[0] if isinstance(output, (tuple, list)) and output else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return output

            if len(inputs) < 3:
                if not self._warned_bad_layout:
                    logger.warning("DeT adapter skipped: unexpected Wan self-attn input signature.")
                    self._warned_bad_layout = True
                return output

            seq_lens = inputs[1] if torch.is_tensor(inputs[1]) else None
            grid_sizes = inputs[2] if torch.is_tensor(inputs[2]) else None
            sparse_attention = bool(inputs[4]) if len(inputs) > 4 else False
            extra_tokens = self._extract_scalar(inputs[6], default=0) if len(inputs) > 6 else 0

            if sparse_attention and not self.allow_sparse_attention:
                if not self._warned_sparse:
                    logger.info("DeT adapter skipped on sparse-attention batches by configuration.")
                    self._warned_sparse = True
                return output
            if seq_lens is None or grid_sizes is None or grid_sizes.ndim != 2 or grid_sizes.shape[1] < 3:
                if not self._warned_bad_layout:
                    logger.warning("DeT adapter skipped: missing seq_lens/grid_sizes.")
                    self._warned_bad_layout = True
                return output

            batch_size, seq_capacity, channels = hidden.shape
            if self.require_uniform_grid:
                if not torch.equal(grid_sizes, grid_sizes[0:1].expand_as(grid_sizes)):
                    if not self._warned_non_uniform:
                        logger.warning(
                            "DeT adapter skipped: non-uniform grid_sizes in batch with det_adapter_require_uniform_grid=true."
                        )
                        self._warned_non_uniform = True
                    return output

            modified: Optional[torch.Tensor] = None
            adapter_dtype = next(adapter.parameters()).dtype
            gate = torch.tanh(adapter.gate) * float(self.gate_max)
            runtime_scale = (
                self._resolve_runtime_locality_scale()
                * self._resolve_runtime_warmup_scale()
                * self._resolve_runtime_unified_scale()
            )
            if runtime_scale < 1.0:
                gate = gate * gate.new_tensor(runtime_scale)

            for batch_idx in range(batch_size):
                seq_len_i = int(seq_lens[batch_idx].item()) if seq_lens.numel() > batch_idx else seq_capacity
                frames_i = int(grid_sizes[batch_idx, 0].item())
                height_i = int(grid_sizes[batch_idx, 1].item())
                width_i = int(grid_sizes[batch_idx, 2].item())
                if frames_i <= 1 or height_i <= 0 or width_i <= 0:
                    continue
                token_count = frames_i * height_i * width_i
                start = max(0, extra_tokens)
                end = start + token_count
                if end > seq_len_i or end > seq_capacity:
                    if not self._warned_bad_layout:
                        logger.warning(
                            "DeT adapter skipped: token layout mismatch (end=%s, seq_len=%s, seq_capacity=%s).",
                            end,
                            seq_len_i,
                            seq_capacity,
                        )
                        self._warned_bad_layout = True
                    continue

                token_slice = hidden[batch_idx, start:end, :]
                token_slice = token_slice.reshape(frames_i, height_i * width_i, channels)
                token_slice = token_slice.permute(1, 2, 0)  # [HW, C, F]
                delta = adapter(token_slice.to(dtype=adapter_dtype))
                delta = (delta * gate.to(dtype=delta.dtype)).to(dtype=hidden.dtype)
                delta = delta.permute(2, 0, 1).reshape(token_count, channels)

                if modified is None:
                    modified = hidden.clone()
                modified[batch_idx, start:end, :] = modified[batch_idx, start:end, :] + delta

            if modified is None:
                return output
            return self._restore_output_shape(output, modified)

        return hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        blocks = self._locate_blocks()
        for depth in self.alignment_depths:
            if depth >= len(blocks):
                continue
            block = blocks[depth]
            self_attn = getattr(block, "self_attn", None)
            adapter = self.adapters.get(depth)
            if self_attn is None or adapter is None:
                continue
            adapter.to(dtype=torch.float32)
            handle = self_attn.register_forward_hook(self._build_hook(depth))
            self.hook_handles.append(handle)

        if self.hook_handles:
            logger.info(
                "DeT adapter: hooks attached to Wan self-attn blocks %s.",
                ", ".join(str(d) for d in self.alignment_depths),
            )
        else:
            logger.warning("DeT adapter: no hooks attached; adapter is inactive.")

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles = []
