from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from common.logger import get_logger
from enhancements.self_flow.noising import SelfFlowDualTimestepContext

logger = get_logger(__name__)

SELF_FLOW_PROJECTOR_STATE_FILE = "self_flow_projector.safetensors"
SELF_FLOW_TEACHER_STATE_FILE = "self_flow_teacher_ema.safetensors"


class SelfFlowHelper(nn.Module):
    """Paper-faithful Self-Flow helper (EMA teacher + feature alignment)."""

    def __init__(self, transformer: nn.Module, args: Any, model_config: Any) -> None:
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.enabled = bool(getattr(args, "enable_self_flow", False))
        self.dual_timestep_enabled = bool(
            getattr(args, "self_flow_enable_dual_timestep", True)
        )
        self.feature_alignment_enabled = bool(
            getattr(args, "self_flow_enable_feature_alignment", True)
        )
        self.strict_mode = bool(getattr(args, "self_flow_strict_mode", True))

        self.student = self._unwrap_model(transformer)
        self.teacher: Optional[nn.Module] = None

        self.student_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.teacher_hooks: List[torch.utils.hooks.RemovableHandle] = []

        self.student_features: Optional[torch.Tensor] = None
        self.teacher_features: Optional[torch.Tensor] = None

        self.student_layer_idx, self.teacher_layer_idx = self._resolve_layer_indices()
        self.hidden_dim = self._resolve_hidden_dim()
        mult = max(1, int(getattr(args, "self_flow_projection_hidden_multiplier", 1)))
        self.student_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * mult),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * mult, self.hidden_dim),
        )

        self.rep_loss_weight = float(getattr(args, "self_flow_rep_loss_weight", 0.8))
        self.rep_loss_type = str(
            getattr(args, "self_flow_rep_loss_type", "negative_cosine")
        ).lower()
        self.teacher_momentum = float(getattr(args, "self_flow_teacher_momentum", 0.9999))
        self.teacher_use_ema = bool(getattr(args, "self_flow_teacher_use_ema", True))
        self.teacher_update_interval = int(
            getattr(args, "self_flow_teacher_update_interval", 1)
        )
        self._step_counter = 0

        self.last_cosine_similarity: Optional[torch.Tensor] = None
        self.last_self_flow_loss: Optional[torch.Tensor] = None
        self._warned_missing_context = False

        self._validate_model_compatibility()
        self._build_teacher_snapshot()

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _get_blocks(self, model: nn.Module) -> Tuple[List[nn.Module], int]:
        blocks = getattr(model, "blocks", None)
        if blocks is None and hasattr(model, "module"):
            blocks = getattr(model.module, "blocks", None)
        if blocks is None:
            raise ValueError("Self-Flow requires transformer.blocks to exist")
        return list(blocks), len(blocks)

    def _resolve_hidden_dim(self) -> int:
        if hasattr(self.student, "dim"):
            return int(self.student.dim)
        if hasattr(self.student, "hidden_size"):
            return int(self.student.hidden_size)
        for module in self.student.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        raise ValueError("Could not infer hidden dimension for Self-Flow projector")

    def _resolve_layer_indices(self) -> Tuple[int, int]:
        blocks, depth = self._get_blocks(self.student)
        del blocks
        student_index = int(getattr(self.args, "self_flow_student_layer_index", -1))
        teacher_index = int(getattr(self.args, "self_flow_teacher_layer_index", -1))
        if student_index >= 1 and teacher_index >= 1:
            return student_index, teacher_index

        student_ratio = float(getattr(self.args, "self_flow_student_layer_ratio", 0.3))
        teacher_ratio = float(getattr(self.args, "self_flow_teacher_layer_ratio", 0.7))
        student_idx = max(1, min(depth, int(round(student_ratio * depth))))
        teacher_idx = max(1, min(depth, int(round(teacher_ratio * depth))))
        if teacher_idx <= student_idx:
            teacher_idx = min(depth, student_idx + 1)
        return student_idx, teacher_idx

    def _validate_model_compatibility(self) -> None:
        if not self.enabled:
            return
        if not self.dual_timestep_enabled:
            return
        model_version = str(
            getattr(
                self.args,
                "effective_model_version",
                getattr(
                    self.args,
                    "wan_model_version",
                    getattr(
                        self.student,
                        "effective_model_version",
                        getattr(self.student, "model_version", "2.1"),
                    ),
                ),
            )
        )
        if not model_version.startswith("2.2"):
            message = (
                "Self-Flow dual-timestep requires Wan 2.2 tokenwise timestep path. "
                f"Detected model_version={model_version}."
            )
            if self.strict_mode:
                raise ValueError(message)
            logger.warning("%s Disabling dual-timestep branch.", message)
            self.dual_timestep_enabled = False

    def _build_teacher_snapshot(self) -> None:
        if not self.enabled or not self.feature_alignment_enabled:
            return
        # Keep a teacher snapshot even when EMA updates are disabled. This allows
        # ablations with a fixed teacher while preserving Eq.6-style alignment.
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        tensor = output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            tensor = output[0]
        if torch.is_tensor(tensor):
            return tensor
        return None

    def _make_student_hook(self):
        def _hook(_module, _inputs, output):
            tensor = self._extract_tensor(output)
            if tensor is not None:
                self.student_features = tensor

        return _hook

    def _make_teacher_hook(self):
        def _hook(_module, _inputs, output):
            tensor = self._extract_tensor(output)
            if tensor is not None:
                self.teacher_features = tensor

        return _hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        if not self.enabled or not self.feature_alignment_enabled:
            return

        student_blocks, student_depth = self._get_blocks(self.student)
        if not (1 <= self.student_layer_idx <= student_depth):
            raise ValueError(
                f"self_flow_student_layer_index {self.student_layer_idx} out of range [1,{student_depth}]"
            )
        self.student_hooks.append(
            student_blocks[self.student_layer_idx - 1].register_forward_hook(
                self._make_student_hook()
            )
        )

        if self.teacher is not None:
            teacher_blocks, teacher_depth = self._get_blocks(self.teacher)
            if not (1 <= self.teacher_layer_idx <= teacher_depth):
                raise ValueError(
                    f"self_flow_teacher_layer_index {self.teacher_layer_idx} out of range [1,{teacher_depth}]"
                )
            self.teacher_hooks.append(
                teacher_blocks[self.teacher_layer_idx - 1].register_forward_hook(
                    self._make_teacher_hook()
                )
            )

        logger.info(
            "Self-Flow hooks ready: student_layer=%d teacher_layer=%d dual_timestep=%s feature_alignment=%s ema_updates=%s",
            self.student_layer_idx,
            self.teacher_layer_idx,
            str(self.dual_timestep_enabled).lower(),
            str(self.feature_alignment_enabled).lower(),
            str(self.teacher_use_ema).lower(),
        )

    def remove_hooks(self) -> None:
        for handle in self.student_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        for handle in self.teacher_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.student_hooks = []
        self.teacher_hooks = []
        self.student_features = None
        self.teacher_features = None
        self.last_cosine_similarity = None
        self.last_self_flow_loss = None

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.student_projector.parameters())

    def projector_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone().to(device="cpu")
            for name, tensor in self.student_projector.state_dict().items()
        }

    def load_projector_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        self.student_projector.load_state_dict(state_dict, strict=True)
        logger.info(
            "Self-Flow: loaded projector state (%d tensors).",
            len(state_dict),
        )

    def teacher_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.teacher is None:
            return {}

        state: Dict[str, torch.Tensor] = {
            "__self_flow_step_counter__": torch.tensor(
                [int(self._step_counter)],
                dtype=torch.int64,
            )
        }
        for name, tensor in self.teacher.state_dict().items():
            state[f"teacher::{name}"] = tensor.detach().clone().to(device="cpu")
        return state

    def load_teacher_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.teacher is None or not state_dict:
            return

        step_tensor = state_dict.get("__self_flow_step_counter__")
        if isinstance(step_tensor, torch.Tensor) and step_tensor.numel() > 0:
            self._step_counter = int(step_tensor.flatten()[0].item())

        teacher_state = {
            key[len("teacher::") :]: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and key.startswith("teacher::")
        }
        if not teacher_state:
            return

        self.teacher.load_state_dict(teacher_state, strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        logger.info(
            "Self-Flow: loaded EMA teacher state (%d tensors).",
            len(teacher_state),
        )

    def save_runtime_state(self, output_dir: str) -> None:
        if not self.enabled:
            return

        projector_state = self.projector_state_dict()
        if projector_state:
            save_file(
                projector_state,
                os.path.join(output_dir, SELF_FLOW_PROJECTOR_STATE_FILE),
            )

        teacher_state = self.teacher_state_dict()
        if teacher_state:
            save_file(
                teacher_state,
                os.path.join(output_dir, SELF_FLOW_TEACHER_STATE_FILE),
            )

    def load_runtime_state(self, input_dir: str) -> None:
        if not self.enabled:
            return

        projector_path = os.path.join(input_dir, SELF_FLOW_PROJECTOR_STATE_FILE)
        if os.path.exists(projector_path):
            self.load_projector_state_dict(load_file(projector_path))

        teacher_path = os.path.join(input_dir, SELF_FLOW_TEACHER_STATE_FILE)
        if os.path.exists(teacher_path):
            self.load_teacher_state_dict(load_file(teacher_path))

    def update_teacher_if_needed(self, student_transformer: nn.Module) -> None:
        if not self.enabled or not self.feature_alignment_enabled:
            return
        if self.teacher is None:
            return
        if not self.teacher_use_ema:
            return

        self._step_counter += 1
        if (self._step_counter % max(1, self.teacher_update_interval)) != 0:
            return

        student = self._unwrap_model(student_transformer)
        m = self.teacher_momentum
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), student.parameters()
            ):
                teacher_param.data.mul_(m).add_(student_param.data, alpha=1.0 - m)
            for teacher_buf, student_buf in zip(
                self.teacher.buffers(), student.buffers()
            ):
                teacher_buf.copy_(student_buf)

    @staticmethod
    def _align_tokens(
        student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if student_feat.shape[1] == teacher_feat.shape[1]:
            return student_feat, teacher_feat
        min_tokens = min(student_feat.shape[1], teacher_feat.shape[1])
        return student_feat[:, :min_tokens], teacher_feat[:, :min_tokens]

    def _run_teacher_forward(
        self,
        *,
        teacher_noisy_input: torch.Tensor,
        teacher_timesteps: torch.Tensor,
        batch: dict[str, Any],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.teacher is None:
            return None
        self.teacher_features = None
        context_source: List[torch.Tensor] = []
        if "t5" in batch:
            raw_t5 = batch.get("t5")
            if isinstance(raw_t5, (list, tuple)):
                context_source = [t for t in raw_t5 if torch.is_tensor(t)]
            elif torch.is_tensor(raw_t5):
                context_source = [raw_t5]
        if not context_source and torch.is_tensor(batch.get("text_embeds")):
            context_source = [batch["text_embeds"]]

        if not context_source:
            if self.strict_mode:
                raise ValueError(
                    "Self-Flow feature alignment requires text context in batch "
                    "(expected 't5' or 'text_embeds')."
                )
            if not self._warned_missing_context:
                logger.warning(
                    "Self-Flow teacher forward skipped: missing text context ('t5' or 'text_embeds')."
                )
                self._warned_missing_context = True
            return None

        context = [t.to(device=device, dtype=dtype) for t in context_source]
        with torch.no_grad():
            _ = self.teacher(
                teacher_noisy_input,
                t=teacher_timesteps,
                context=context,
                clip_fea=None,
                seq_len=seq_len,
                y=None,
                force_keep_mask=None,
                controlnet_states=None,
                controlnet_weight=1.0,
                controlnet_stride=1,
                dispersive_loss_target_block=None,
                return_intermediate=False,
                internal_guidance_target_block=None,
                return_internal_guidance=False,
                reg_cls_token=None,
                segment_idx=None,
                bfm_semfeat_tokens=None,
            )
        return self.teacher_features

    def compute_loss(
        self,
        *,
        accelerator: Any,
        network_dtype: torch.dtype,
        batch: dict[str, Any],
        context: SelfFlowDualTimestepContext | None,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or not self.feature_alignment_enabled:
            return None
        if context is None:
            return None
        if self.rep_loss_weight <= 0.0:
            return None
        if self.student_features is None:
            return None

        teacher_feat = self._run_teacher_forward(
            teacher_noisy_input=context.teacher_noisy_model_input.to(
                device=accelerator.device, dtype=network_dtype
            ),
            teacher_timesteps=context.teacher_model_timesteps.to(
                device=accelerator.device, dtype=network_dtype
            ),
            batch=batch,
            seq_len=context.sequence_length,
            device=accelerator.device,
            dtype=network_dtype,
        )
        if teacher_feat is None:
            return None

        student_feat = self.student_features
        student_proj = self.student_projector(student_feat)
        student_proj, teacher_feat = self._align_tokens(student_proj, teacher_feat)

        student_proj = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_feat.detach(), dim=-1)
        cosine = F.cosine_similarity(student_proj, teacher_norm, dim=-1).mean()
        self.last_cosine_similarity = cosine.detach()

        if self.rep_loss_type == "one_minus_cosine":
            rep_loss = 1.0 - cosine
        else:
            rep_loss = -cosine

        total = rep_loss * self.rep_loss_weight
        self.last_self_flow_loss = total.detach()

        # Reset feature tensors to avoid stale reuse across steps.
        self.student_features = None
        self.teacher_features = None
        return total
