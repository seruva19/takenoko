from __future__ import annotations

import copy
import math
import os
import random
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
    """Self-Flow helper for Wan training with optional shared-teacher LoRA modes."""

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
            self._make_activation(),
            nn.Linear(self.hidden_dim * mult, self.hidden_dim),
        )

        self.teacher_mode = str(getattr(args, "self_flow_teacher_mode", "ema")).lower()
        self.projector_lr = getattr(args, "self_flow_projector_lr", None)
        self.mask_focus_loss = bool(getattr(args, "self_flow_mask_focus_loss", False))
        self.max_loss = float(getattr(args, "self_flow_max_loss", 0.0))
        self.temporal_mode = str(getattr(args, "self_flow_temporal_mode", "off")).lower()
        self.lambda_temporal = float(getattr(args, "self_flow_lambda_temporal", 0.0))
        self.lambda_delta = float(getattr(args, "self_flow_lambda_delta", 0.0))
        self.temporal_tau = float(getattr(args, "self_flow_temporal_tau", 1.0))
        self.num_neighbors = int(getattr(args, "self_flow_num_neighbors", 2))
        self.temporal_granularity = str(
            getattr(args, "self_flow_temporal_granularity", "frame")
        ).lower()
        self.patch_spatial_radius = int(
            getattr(args, "self_flow_patch_spatial_radius", 0)
        )
        self.patch_match_mode = str(
            getattr(args, "self_flow_patch_match_mode", "hard")
        ).lower()
        self.patch_match_temperature = float(
            getattr(args, "self_flow_patch_match_temperature", 0.1)
        )
        self.delta_num_steps = int(getattr(args, "self_flow_delta_num_steps", 1))
        self.motion_weighting = str(
            getattr(args, "self_flow_motion_weighting", "none")
        ).lower()
        self.motion_weight_strength = float(
            getattr(args, "self_flow_motion_weight_strength", 0.0)
        )
        self.temporal_schedule = str(
            getattr(args, "self_flow_temporal_schedule", "constant")
        ).lower()
        self.temporal_warmup_steps = int(
            getattr(args, "self_flow_temporal_warmup_steps", 0)
        )
        self.temporal_max_steps = int(
            getattr(args, "self_flow_temporal_max_steps", 0)
        )
        self.offload_teacher_features = bool(
            getattr(args, "self_flow_offload_teacher_features", False)
        )
        self.offload_teacher_params = bool(
            getattr(args, "self_flow_offload_teacher_params", False)
        )
        self.student_layer_stochastic_range = max(
            0, int(getattr(args, "self_flow_student_layer_stochastic_range", 0))
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

        self._capture_mode = "idle"
        self._teacher_strategy = "uninitialized"
        self._shadow_params: Dict[str, torch.Tensor] = {}
        self._last_ema_drift: Optional[float] = None
        self._warned_missing_context = False
        self._warned_invalid_base_mode = False

        self._candidate_student_layer_indices = self._resolve_student_layer_candidates()
        self._active_student_layer_idx = self.student_layer_idx

        self.last_cosine_similarity: Optional[torch.Tensor] = None
        self.last_self_flow_loss: Optional[torch.Tensor] = None
        self.last_frame_cosine: Optional[torch.Tensor] = None
        self.last_delta_cosine: Optional[torch.Tensor] = None
        self.current_rep_loss_weight = self.rep_loss_weight
        self.current_lambda_temporal = self.lambda_temporal
        self.current_lambda_delta = self.lambda_delta

        self._validate_model_compatibility()

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        return model.module if hasattr(model, "module") else model

    def _make_activation(self) -> nn.Module:
        activation = str(
            getattr(self.args, "self_flow_projector_activation", "silu")
        ).lower()
        if activation == "gelu":
            return nn.GELU()
        return nn.SiLU()

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

    def _resolve_student_layer_candidates(self) -> List[int]:
        blocks, depth = self._get_blocks(self.student)
        del blocks, depth
        low = max(1, self.student_layer_idx - self.student_layer_stochastic_range)
        high = min(
            self.teacher_layer_idx - 1,
            self.student_layer_idx + self.student_layer_stochastic_range,
        )
        if high < low:
            return [self.student_layer_idx]
        return list(range(low, high + 1))

    def _validate_model_compatibility(self) -> None:
        if not self.enabled or not self.dual_timestep_enabled:
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
        if not self.enabled or not self.feature_alignment_enabled or self.teacher is not None:
            return
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        if self.offload_teacher_params:
            self.teacher.to(device="cpu")

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        tensor = output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            tensor = output[0]
        if torch.is_tensor(tensor):
            return tensor
        return None

    def _make_student_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            if self._capture_mode != "student":
                return
            if layer_idx != self._active_student_layer_idx:
                return
            tensor = self._extract_tensor(output)
            if tensor is not None:
                self.student_features = tensor

        return _hook

    def _make_teacher_hook(self):
        def _hook(_module, _inputs, output):
            if self._capture_mode != "teacher":
                return
            tensor = self._extract_tensor(output)
            if tensor is not None:
                self.teacher_features = tensor

        return _hook

    def _teacher_hook_target(self) -> nn.Module:
        if self._teacher_strategy == "model_copy" and self.teacher is not None:
            return self.teacher
        return self.student

    def setup_hooks(self) -> None:
        self.remove_hooks()
        if not self.enabled or not self.feature_alignment_enabled:
            return

        student_blocks, student_depth = self._get_blocks(self.student)
        for layer_idx in self._candidate_student_layer_indices:
            if not (1 <= layer_idx <= student_depth):
                raise ValueError(
                    f"self_flow_student_layer_index {layer_idx} out of range [1,{student_depth}]"
                )
            self.student_hooks.append(
                student_blocks[layer_idx - 1].register_forward_hook(
                    self._make_student_hook(layer_idx)
                )
            )

        teacher_target = self._teacher_hook_target()
        teacher_blocks, teacher_depth = self._get_blocks(teacher_target)
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
            "Self-Flow hooks ready: student_layer=%d teacher_layer=%d strategy=%s "
            "teacher_mode=%s dual_timestep=%s feature_alignment=%s ema_updates=%s "
            "mask_focus=%s offload_features=%s offload_params=%s",
            self.student_layer_idx,
            self.teacher_layer_idx,
            self._teacher_strategy,
            self.teacher_mode,
            str(self.dual_timestep_enabled).lower(),
            str(self.feature_alignment_enabled).lower(),
            str(self.teacher_use_ema).lower(),
            str(self.mask_focus_loss).lower(),
            str(self.offload_teacher_features).lower(),
            str(self.offload_teacher_params).lower(),
        )

    def mark_student_forward(self) -> None:
        if not self.enabled or not self.feature_alignment_enabled:
            return
        self._capture_mode = "student"
        self.student_features = None
        if len(self._candidate_student_layer_indices) > 1:
            self._active_student_layer_idx = random.choice(
                self._candidate_student_layer_indices
            )
        else:
            self._active_student_layer_idx = self.student_layer_idx

    def cleanup_step(self) -> None:
        self._capture_mode = "idle"
        self.student_features = None
        self.teacher_features = None

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
        self.cleanup_step()
        self.last_cosine_similarity = None
        self.last_self_flow_loss = None
        self.last_frame_cosine = None
        self.last_delta_cosine = None

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.student_projector.parameters())

    @property
    def last_ema_drift(self) -> Optional[float]:
        return self._last_ema_drift

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

    @staticmethod
    def _resolve_shadow_name(
        param_name: str, shadow_params: Dict[str, torch.Tensor]
    ) -> Optional[str]:
        if param_name in shadow_params:
            return param_name
        if param_name.startswith("module."):
            stripped = param_name[len("module.") :]
            if stripped in shadow_params:
                return stripped
        prefixed = f"module.{param_name}"
        if prefixed in shadow_params:
            return prefixed
        return None

    def _matches_teacher_block(self, param_name: str) -> bool:
        block_idx = max(0, self.teacher_layer_idx - 1)
        patterns = (
            f".blocks.{block_idx}.",
            f"_blocks_{block_idx}_",
            f".transformer_blocks.{block_idx}.",
            f"_transformer_blocks_{block_idx}_",
        )
        return any(pattern in param_name for pattern in patterns)

    def teacher_state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {
            "__self_flow_step_counter__": torch.tensor(
                [int(self._step_counter)],
                dtype=torch.int64,
            )
        }
        if self._shadow_params:
            for name, tensor in self._shadow_params.items():
                state[f"shadow::{name}"] = tensor.detach().clone().to(device="cpu")
            return state
        if self.teacher is not None:
            for name, tensor in self.teacher.state_dict().items():
                state[f"teacher::{name}"] = tensor.detach().clone().to(device="cpu")
            return state
        return {}

    def load_teacher_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        if self.teacher_mode == "base":
            logger.warning(
                "Self-Flow: teacher_mode=base ignores saved teacher EMA state."
            )
            return

        step_tensor = state_dict.get("__self_flow_step_counter__")
        if isinstance(step_tensor, torch.Tensor) and step_tensor.numel() > 0:
            self._step_counter = int(step_tensor.flatten()[0].item())

        shadow_state = {
            key[len("shadow::") :]: value.detach().clone()
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and key.startswith("shadow::")
        }
        if shadow_state:
            if self.offload_teacher_params:
                shadow_state = {
                    key: value.to(device="cpu") for key, value in shadow_state.items()
                }
            self._shadow_params = shadow_state
            self._teacher_strategy = "shadow"
            self.setup_hooks()
            logger.info(
                "Self-Flow: loaded teacher shadow state (%d tensors).",
                len(shadow_state),
            )
            return

        teacher_state = {
            key[len("teacher::") :]: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and key.startswith("teacher::")
        }
        if not teacher_state:
            return

        self._build_teacher_snapshot()
        if self.teacher is None:
            return
        self.teacher.load_state_dict(teacher_state, strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self._teacher_strategy = "model_copy"
        self.setup_hooks()
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

    @staticmethod
    def _is_same_module(lhs: Any, rhs: Any) -> bool:
        if lhs is None or rhs is None:
            return False
        return SelfFlowHelper._unwrap_model(lhs) is SelfFlowHelper._unwrap_model(rhs)

    @staticmethod
    def _is_adapter_module(module: nn.Module) -> bool:
        return hasattr(module, "multiplier")

    @classmethod
    def _collect_adapter_modules(cls, network: nn.Module) -> List[nn.Module]:
        return [module for module in network.modules() if cls._is_adapter_module(module)]

    def _zero_adapter_multipliers(self, network: nn.Module) -> Any:
        network = self._unwrap_model(network)
        if hasattr(network, "set_multiplier") and hasattr(network, "multiplier"):
            saved = float(getattr(network, "multiplier"))
            network.set_multiplier(0.0)
            return ("network", saved)
        modules = self._collect_adapter_modules(network)
        saved = [float(getattr(module, "multiplier")) for module in modules]
        for module in modules:
            try:
                module.multiplier = 0.0
            except Exception:
                pass
        return ("modules", modules, saved)

    @staticmethod
    def _restore_adapter_multipliers(network: nn.Module, state: Any) -> None:
        if not state:
            return
        if state[0] == "network":
            if hasattr(network, "set_multiplier"):
                network.set_multiplier(state[1])
            return
        _, modules, saved = state
        for module, value in zip(modules, saved):
            try:
                module.multiplier = value
            except Exception:
                pass

    def _initialize_shadow_teacher(self, network: nn.Module) -> None:
        if self._shadow_params:
            return
        network = self._unwrap_model(network)
        teacher_mode = self.teacher_mode
        teacher_block_matched = 0
        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            if teacher_mode == "partial_ema" and not self._matches_teacher_block(name):
                continue
            tensor = param.detach()
            if self.offload_teacher_params:
                tensor = tensor.to(device="cpu")
            self._shadow_params[name] = tensor.clone()
            teacher_block_matched += 1

        if teacher_mode == "partial_ema" and teacher_block_matched == 0:
            message = (
                "Self-Flow teacher_mode=partial_ema did not match any trainable "
                "adapter params for the configured teacher block."
            )
            if self.strict_mode:
                raise ValueError(message)
            logger.warning("%s Falling back to full EMA shadow.", message)
            for name, param in network.named_parameters():
                if not param.requires_grad or name in self._shadow_params:
                    continue
                tensor = param.detach()
                if self.offload_teacher_params:
                    tensor = tensor.to(device="cpu")
                self._shadow_params[name] = tensor.clone()

        self._teacher_strategy = "shadow"
        self.setup_hooks()
        logger.info(
            "Self-Flow teacher shadow initialized: mode=%s tensors=%d offload_params=%s",
            teacher_mode,
            len(self._shadow_params),
            str(self.offload_teacher_params).lower(),
        )

    def _swap_in_shadow_teacher(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        network = self._unwrap_model(network)
        backups: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                backups[name] = param.detach().clone()
                if shadow.device != param.device or shadow.dtype != param.dtype:
                    param.copy_(shadow.to(device=param.device, dtype=param.dtype))
                else:
                    param.copy_(shadow)
        return backups

    @staticmethod
    def _restore_shadow_teacher(
        network: nn.Module, backups: Dict[str, torch.Tensor]
    ) -> None:
        if not backups:
            return
        network = SelfFlowHelper._unwrap_model(network)
        with torch.no_grad():
            for name, param in network.named_parameters():
                backup = backups.get(name)
                if backup is None:
                    continue
                param.copy_(backup.to(device=param.device, dtype=param.dtype))

    def _update_shadow_teacher(self, network: nn.Module) -> None:
        network = self._unwrap_model(network)
        momentum = self.teacher_momentum
        drift_sum = 0.0
        drift_count = 0
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                source = param.detach()
                if source.device != shadow.device or source.dtype != shadow.dtype:
                    source = source.to(device=shadow.device, dtype=shadow.dtype)
                drift_sum += (shadow - source).norm().item()
                drift_count += 1
                shadow.mul_(momentum).add_(source, alpha=1.0 - momentum)
        if drift_count > 0:
            self._last_ema_drift = drift_sum / drift_count

    def _ensure_teacher_ready(self, network: Optional[nn.Module]) -> None:
        if not self.enabled or not self.feature_alignment_enabled:
            return
        if self._teacher_strategy != "uninitialized":
            return

        shared_teacher_network = (
            network is not None and not self._is_same_module(network, self.student)
        )
        if self.teacher_mode == "base":
            if shared_teacher_network:
                self._teacher_strategy = "base_shared"
                self.setup_hooks()
                return
            message = (
                "Self-Flow teacher_mode=base requires a separate adapter network. "
                "Using EMA/model-copy teacher instead."
            )
            if not self._warned_invalid_base_mode:
                logger.warning(message)
                self._warned_invalid_base_mode = True

        if shared_teacher_network:
            self._initialize_shadow_teacher(network)
            return

        self._build_teacher_snapshot()
        self._teacher_strategy = "model_copy"
        self.setup_hooks()

    def update_teacher_if_needed(
        self, student_transformer: nn.Module, network: Optional[nn.Module] = None
    ) -> None:
        if not self.enabled or not self.feature_alignment_enabled:
            return
        if not self.teacher_use_ema or self.teacher_mode == "base":
            return

        self._step_counter += 1
        if (self._step_counter % max(1, self.teacher_update_interval)) != 0:
            return

        self._ensure_teacher_ready(network)
        if self._teacher_strategy == "shadow":
            if network is not None:
                self._update_shadow_teacher(network)
            return

        if self.teacher is None:
            self._build_teacher_snapshot()
            self._teacher_strategy = "model_copy"
            self.setup_hooks()
        if self.teacher is None:
            return

        student = self._unwrap_model(student_transformer)
        momentum = self.teacher_momentum
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), student.parameters()
            ):
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1.0 - momentum
                )
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

    @staticmethod
    def _align_mask(
        token_mask: Optional[torch.Tensor], token_count: int
    ) -> Optional[torch.Tensor]:
        if token_mask is None:
            return None
        if token_mask.shape[1] == token_count:
            return token_mask
        if token_mask.shape[1] > token_count:
            return token_mask[:, :token_count]
        return token_mask

    def _schedule_scale(self, global_step: Optional[int]) -> float:
        if global_step is None:
            return 1.0
        schedule = self.temporal_schedule
        if schedule == "constant":
            return 1.0

        warmup_steps = max(0, self.temporal_warmup_steps)
        if warmup_steps > 0 and global_step < warmup_steps:
            return float(global_step) / float(max(1, warmup_steps))

        max_steps = max(0, self.temporal_max_steps)
        if max_steps <= 0:
            return 1.0

        progress = min(
            max(float(global_step - warmup_steps), 0.0)
            / max(float(max_steps - warmup_steps), 1.0),
            1.0,
        )
        if schedule == "linear":
            return 1.0 - progress
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def _apply_lambda_schedule(self, global_step: Optional[int]) -> None:
        scale = self._schedule_scale(global_step)
        self.current_rep_loss_weight = self.rep_loss_weight * scale
        self.current_lambda_temporal = self.lambda_temporal * scale
        self.current_lambda_delta = self.lambda_delta * scale

    def _resolve_patch_size(self, ndim: int) -> Tuple[int, int, int]:
        raw_patch_size = getattr(self.model_config, "patch_size", (1, 2, 2))
        patch_size = tuple(int(value) for value in raw_patch_size)
        if ndim == 5:
            return patch_size[0], patch_size[1], patch_size[2]
        return 1, patch_size[1], patch_size[2]

    def _token_shape_from_latents(self, latents: torch.Tensor) -> Tuple[int, int, int]:
        if latents.ndim == 5:
            _, _, frames, height, width = latents.shape
        elif latents.ndim == 4:
            _, _, height, width = latents.shape
            frames = 1
        else:
            return 0, 0, 0

        pt, ph, pw = self._resolve_patch_size(latents.ndim)
        if (
            pt <= 0
            or ph <= 0
            or pw <= 0
            or frames % pt != 0
            or height % ph != 0
            or width % pw != 0
        ):
            return 0, 0, 0
        return frames // pt, height // ph, width // pw

    def _reshape_temporal_features(
        self, features: torch.Tensor, num_latent_frames: int
    ) -> Optional[torch.Tensor]:
        total_tokens = int(features.shape[1])
        if num_latent_frames <= 1 or total_tokens < num_latent_frames:
            return None
        usable_tokens = (total_tokens // num_latent_frames) * num_latent_frames
        if usable_tokens <= 0:
            return None
        if usable_tokens != total_tokens:
            features = features[:, :usable_tokens]
        spatial_tokens = usable_tokens // num_latent_frames
        return features.reshape(
            features.shape[0],
            num_latent_frames,
            spatial_tokens,
            features.shape[-1],
        )

    def _reshape_temporal_grid(
        self,
        features: torch.Tensor,
        *,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Optional[torch.Tensor]:
        if num_latent_frames <= 1 or latent_height <= 0 or latent_width <= 0:
            return None
        expected_tokens = num_latent_frames * latent_height * latent_width
        if int(features.shape[1]) < expected_tokens:
            return None
        if int(features.shape[1]) != expected_tokens:
            features = features[:, :expected_tokens]
        return features.reshape(
            features.shape[0],
            num_latent_frames,
            latent_height,
            latent_width,
            features.shape[-1],
        )

    @staticmethod
    def _normalize_motion_weights(
        motion: torch.Tensor, strength: float
    ) -> torch.Tensor:
        if float(strength) <= 0.0:
            return torch.ones_like(motion)
        motion = motion.to(dtype=torch.float32)
        baseline = motion.mean()
        if not torch.isfinite(baseline) or float(baseline.item()) <= 1e-8:
            return torch.ones_like(motion)
        normalized = motion / baseline.clamp_min(1e-8)
        weights = 1.0 + float(strength) * normalized
        return weights.to(dtype=motion.dtype)

    @staticmethod
    def _teacher_delta_motion_weights(
        teacher_frames: torch.Tensor,
        *,
        strength: float,
    ) -> torch.Tensor:
        if float(strength) <= 0.0 or teacher_frames.shape[1] <= 1:
            return torch.ones(
                teacher_frames.shape[:-1],
                device=teacher_frames.device,
                dtype=teacher_frames.dtype,
            )
        forward = (teacher_frames[:, 1:] - teacher_frames[:, :-1]).pow(2).mean(dim=-1)
        motion = torch.zeros(
            teacher_frames.shape[:-1],
            device=teacher_frames.device,
            dtype=teacher_frames.dtype,
        )
        motion[:, :-1] = motion[:, :-1] + forward
        motion[:, 1:] = motion[:, 1:] + forward
        return SelfFlowHelper._normalize_motion_weights(motion, strength)

    @staticmethod
    def _neighbor_weighted_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sim = torch.bmm(student_frames, teacher_frames.transpose(1, 2))
        tau = max(float(temporal_tau), 1e-6)
        total = sim.new_zeros(())
        normalizer = sim.new_zeros(())

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                diag = sim.diagonal(dim1=1, dim2=2)
                if motion_weights is None:
                    total = total + weight * diag.sum()
                    normalizer = normalizer + sim.new_tensor(diag.numel() * weight)
                else:
                    cast_weights = motion_weights.to(device=diag.device, dtype=diag.dtype)
                    total = total + weight * (diag * cast_weights).sum()
                    normalizer = normalizer + weight * cast_weights.sum()
                continue

            forward = sim.diagonal(offset=delta, dim1=1, dim2=2)
            backward = sim.diagonal(offset=-delta, dim1=1, dim2=2)
            if motion_weights is None:
                total = total + weight * (forward.sum() + backward.sum())
                normalizer = normalizer + sim.new_tensor(
                    (forward.numel() + backward.numel()) * weight
                )
            else:
                forward_weights = motion_weights[:, :-delta].to(
                    device=forward.device, dtype=forward.dtype
                )
                backward_weights = motion_weights[:, delta:].to(
                    device=backward.device, dtype=backward.dtype
                )
                total = total + weight * (forward * forward_weights).sum()
                total = total + weight * (backward * backward_weights).sum()
                normalizer = normalizer + weight * (
                    forward_weights.sum() + backward_weights.sum()
                )

        if normalizer.item() <= 0.0:
            return sim.diagonal(dim1=1, dim2=2).mean()
        return total / normalizer

    @staticmethod
    def _neighbor_weighted_local_patch_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        spatial_radius: int,
        patch_match_mode: str,
        patch_match_temperature: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_radius <= 0:
            flat_student = student_frames.reshape(
                student_frames.shape[0],
                student_frames.shape[1],
                student_frames.shape[2] * student_frames.shape[3],
                student_frames.shape[4],
            )
            flat_teacher = teacher_frames.reshape(
                teacher_frames.shape[0],
                teacher_frames.shape[1],
                teacher_frames.shape[2] * teacher_frames.shape[3],
                teacher_frames.shape[4],
            )
            sim = torch.einsum("btnd,bsnd->btsn", flat_student, flat_teacher)
            tau = max(float(temporal_tau), 1e-6)
            total = sim.new_zeros(())
            normalizer = sim.new_zeros(())

            def _reduce(
                values: torch.Tensor,
                weight: float,
                weights_slice: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                local_values = values.mean(dim=-1)
                if weights_slice is None:
                    return weight * local_values.sum(), sim.new_tensor(
                        local_values.numel() * weight
                    )
                weights_view = weights_slice.to(
                    device=local_values.device, dtype=local_values.dtype
                ).reshape_as(local_values)
                return (
                    weight * (local_values * weights_view).sum(),
                    weight * weights_view.sum(),
                )

            for delta in range(0, max(0, int(num_neighbors)) + 1):
                weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
                if delta == 0:
                    diag = sim.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
                    part_total, part_norm = _reduce(diag, weight, motion_weights)
                    total = total + part_total
                    normalizer = normalizer + part_norm
                    continue

                forward = sim.diagonal(offset=delta, dim1=1, dim2=2).permute(0, 2, 1)
                backward = sim.diagonal(offset=-delta, dim1=1, dim2=2).permute(0, 2, 1)
                forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
                backward_weights = None if motion_weights is None else motion_weights[:, delta:]
                part_total, part_norm = _reduce(forward, weight, forward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm
                part_total, part_norm = _reduce(backward, weight, backward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm

            if normalizer.item() <= 0.0:
                return sim.diagonal(dim1=1, dim2=2).mean()
            return total / normalizer

        batch_size, num_frames, height, width, channels = teacher_frames.shape
        kernel_size = 2 * int(spatial_radius) + 1
        teacher_bt = teacher_frames.permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_frames, channels, height, width
        )
        teacher_neighborhoods = F.unfold(
            teacher_bt, kernel_size=kernel_size, padding=int(spatial_radius)
        )
        neighborhood_size = kernel_size * kernel_size
        teacher_neighborhoods = teacher_neighborhoods.reshape(
            batch_size, num_frames, channels, neighborhood_size, height, width
        ).permute(0, 1, 4, 5, 3, 2)

        tau = max(float(temporal_tau), 1e-6)
        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())

        def _accumulate(
            student_slice: torch.Tensor,
            teacher_slice: torch.Tensor,
            weight: float,
            weights_slice: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            similarities = (student_slice.unsqueeze(-2) * teacher_slice).sum(dim=-1)
            if patch_match_mode == "soft":
                temperature = max(float(patch_match_temperature), 1e-6)
                attn = torch.softmax(similarities / temperature, dim=-1)
                matched = (attn * similarities).sum(dim=-1)
            else:
                matched = similarities.max(dim=-1).values
            if weights_slice is None:
                return weight * matched.sum(), student_frames.new_tensor(
                    matched.numel() * weight
                )
            local_weights = weights_slice.to(device=matched.device, dtype=matched.dtype)
            return weight * (matched * local_weights).sum(), weight * local_weights.sum()

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                part_total, part_norm = _accumulate(
                    student_frames, teacher_neighborhoods, weight, motion_weights
                )
                total = total + part_total
                normalizer = normalizer + part_norm
                continue

            forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
            backward_weights = None if motion_weights is None else motion_weights[:, delta:]
            part_total, part_norm = _accumulate(
                student_frames[:, :-delta],
                teacher_neighborhoods[:, delta:],
                weight,
                forward_weights,
            )
            total = total + part_total
            normalizer = normalizer + part_norm
            part_total, part_norm = _accumulate(
                student_frames[:, delta:],
                teacher_neighborhoods[:, :-delta],
                weight,
                backward_weights,
            )
            total = total + part_total
            normalizer = normalizer + part_norm

        if normalizer.item() <= 0.0:
            return student_frames.new_zeros(())
        return total / normalizer

    @staticmethod
    def _multi_step_delta_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        delta_num_steps: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        max_step = min(
            max(1, int(delta_num_steps)),
            max(student_frames.shape[1] - 1, 0),
            max(teacher_frames.shape[1] - 1, 0),
        )
        if max_step <= 0:
            return None

        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())
        tau = max(float(temporal_tau), 1e-6)

        for step in range(1, max_step + 1):
            weight = math.exp(-float(step - 1) / tau)
            student_delta = F.normalize(
                student_frames[:, step:] - student_frames[:, :-step], dim=-1
            )
            teacher_delta = F.normalize(
                teacher_frames[:, step:] - teacher_frames[:, :-step], dim=-1
            )
            cosine = F.cosine_similarity(student_delta, teacher_delta, dim=-1)
            step_weights = None if motion_weights is None else motion_weights[:, step:]
            if step_weights is None:
                total = total + weight * cosine.sum()
                normalizer = normalizer + student_frames.new_tensor(
                    cosine.numel() * weight
                )
            else:
                cast_weights = step_weights.to(device=cosine.device, dtype=cosine.dtype)
                total = total + weight * (cosine * cast_weights).sum()
                normalizer = normalizer + weight * cast_weights.sum()

        if normalizer.item() <= 0.0:
            return None
        return total / normalizer

    def _collect_text_context(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[List[torch.Tensor]]:
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
        return [tensor.to(device=device, dtype=dtype) for tensor in context_source]

    @staticmethod
    def _call_wan_model(
        model: nn.Module,
        *,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
    ) -> None:
        _ = model(
            noisy_input,
            t=timesteps,
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

    def _run_teacher_forward(
        self,
        *,
        teacher_noisy_input: torch.Tensor,
        teacher_timesteps: torch.Tensor,
        batch: dict[str, Any],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        network: Optional[nn.Module] = None,
    ) -> Optional[torch.Tensor]:
        self._ensure_teacher_ready(network)
        context = self._collect_text_context(batch, device=device, dtype=dtype)
        if context is None:
            return None

        self.teacher_features = None
        teacher_timesteps = teacher_timesteps.to(device=device, dtype=dtype)
        teacher_noisy_input = teacher_noisy_input.to(device=device, dtype=dtype)

        if self._teacher_strategy == "model_copy":
            if self.teacher is None:
                return None
            teacher_model = self.teacher
            if self.offload_teacher_params:
                teacher_model.to(device=device, dtype=dtype)
            try:
                self._capture_mode = "teacher"
                with torch.no_grad():
                    self._call_wan_model(
                        teacher_model,
                        noisy_input=teacher_noisy_input,
                        timesteps=teacher_timesteps,
                        context=context,
                        seq_len=seq_len,
                    )
            finally:
                self._capture_mode = "idle"
                if self.offload_teacher_params:
                    teacher_model.to(device="cpu")
        else:
            teacher_model = self.student
            if network is None:
                return None
            network = self._unwrap_model(network)
            modifier_state = None
            shadow_backups: Dict[str, torch.Tensor] = {}
            try:
                if self._teacher_strategy == "base_shared":
                    modifier_state = self._zero_adapter_multipliers(network)
                elif self._teacher_strategy == "shadow":
                    shadow_backups = self._swap_in_shadow_teacher(network)
                self._capture_mode = "teacher"
                with torch.no_grad():
                    self._call_wan_model(
                        teacher_model,
                        noisy_input=teacher_noisy_input,
                        timesteps=teacher_timesteps,
                        context=context,
                        seq_len=seq_len,
                    )
            finally:
                self._capture_mode = "idle"
                if self._teacher_strategy == "base_shared":
                    self._restore_adapter_multipliers(network, modifier_state)
                elif self._teacher_strategy == "shadow":
                    self._restore_shadow_teacher(network, shadow_backups)

        if (
            self.offload_teacher_features
            and self.teacher_features is not None
            and self.teacher_features.device.type != "cpu"
        ):
            self.teacher_features = self.teacher_features.to(device="cpu")
        return self.teacher_features

    def compute_loss(
        self,
        *,
        accelerator: Any,
        network_dtype: torch.dtype,
        batch: dict[str, Any],
        context: SelfFlowDualTimestepContext | None,
        network: Optional[nn.Module] = None,
        global_step: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or not self.feature_alignment_enabled:
            return None
        if context is None or self.student_features is None:
            return None

        self._apply_lambda_schedule(global_step)
        if (
            self.current_rep_loss_weight <= 0.0
            and self.current_lambda_temporal <= 0.0
            and self.current_lambda_delta <= 0.0
        ):
            return None
        self.last_frame_cosine = None
        self.last_delta_cosine = None

        teacher_feat = self._run_teacher_forward(
            teacher_noisy_input=context.teacher_noisy_model_input,
            teacher_timesteps=context.teacher_model_timesteps,
            batch=batch,
            seq_len=context.sequence_length,
            device=accelerator.device,
            dtype=network_dtype,
            network=network,
        )
        if teacher_feat is None:
            return None

        student_feat = self.student_features
        if teacher_feat.device != student_feat.device or teacher_feat.dtype != student_feat.dtype:
            teacher_feat = teacher_feat.to(
                device=student_feat.device,
                dtype=student_feat.dtype,
                non_blocking=True,
            )

        student_proj = self.student_projector(student_feat)
        student_proj, teacher_feat = self._align_tokens(student_proj, teacher_feat)
        token_mask = self._align_mask(context.token_mask, student_proj.shape[1])

        student_proj = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_feat.detach(), dim=-1)
        if self.mask_focus_loss and token_mask is not None:
            token_mask = token_mask.to(device=student_proj.device)
            if token_mask.shape[1] < student_proj.shape[1]:
                student_proj = student_proj[:, : token_mask.shape[1]]
                teacher_norm = teacher_norm[:, : token_mask.shape[1]]
            if token_mask.any():
                cosine = F.cosine_similarity(
                    student_proj[token_mask],
                    teacher_norm[token_mask],
                    dim=-1,
                ).mean()
            else:
                cosine = F.cosine_similarity(student_proj, teacher_norm, dim=-1).mean()
        else:
            cosine = F.cosine_similarity(student_proj, teacher_norm, dim=-1).mean()
        self.last_cosine_similarity = cosine.detach()

        if self.rep_loss_type == "one_minus_cosine":
            rep_loss = 1.0 - cosine
        else:
            rep_loss = -cosine

        total = rep_loss * self.current_rep_loss_weight

        token_frames, token_height, token_width = self._token_shape_from_latents(
            context.teacher_noisy_model_input
        )
        temporal_student = None
        temporal_teacher = None
        temporal_student_grid = None
        temporal_teacher_grid = None
        if token_frames > 1:
            temporal_student = self._reshape_temporal_features(student_feat, token_frames)
            temporal_teacher = self._reshape_temporal_features(teacher_feat, token_frames)
            if token_height > 0 and token_width > 0:
                temporal_student_grid = self._reshape_temporal_grid(
                    student_feat,
                    num_latent_frames=token_frames,
                    latent_height=token_height,
                    latent_width=token_width,
                )
                temporal_teacher_grid = self._reshape_temporal_grid(
                    teacher_feat,
                    num_latent_frames=token_frames,
                    latent_height=token_height,
                    latent_width=token_width,
                )

        temporal_motion_weights = None
        temporal_motion_grid_weights = None
        if self.motion_weighting == "teacher_delta":
            if temporal_teacher is not None:
                temporal_motion_weights = self._teacher_delta_motion_weights(
                    temporal_teacher,
                    strength=self.motion_weight_strength,
                )
            if temporal_teacher_grid is not None:
                temporal_motion_grid_weights = self._teacher_delta_motion_weights(
                    temporal_teacher_grid,
                    strength=self.motion_weight_strength,
                )

        if (
            self.temporal_mode in {"frame", "hybrid"}
            and self.current_lambda_temporal > 0.0
            and temporal_student is not None
            and temporal_teacher is not None
        ):
            if self.temporal_granularity == "patch":
                if temporal_student_grid is not None and temporal_teacher_grid is not None:
                    student_frames = F.normalize(temporal_student_grid, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher_grid, dim=-1)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames,
                        teacher_frames,
                        num_neighbors=self.num_neighbors,
                        temporal_tau=self.temporal_tau,
                        spatial_radius=self.patch_spatial_radius,
                        patch_match_mode=self.patch_match_mode,
                        patch_match_temperature=self.patch_match_temperature,
                        motion_weights=temporal_motion_grid_weights,
                    )
                else:
                    student_frames = F.normalize(temporal_student, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher, dim=-1)
                    flat_motion_weights = None
                    if temporal_motion_weights is not None:
                        flat_motion_weights = temporal_motion_weights.mean(dim=-1).unsqueeze(2)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames.reshape(
                            student_frames.shape[0],
                            student_frames.shape[1],
                            1,
                            student_frames.shape[2],
                            student_frames.shape[3],
                        ),
                        teacher_frames.reshape(
                            teacher_frames.shape[0],
                            teacher_frames.shape[1],
                            1,
                            teacher_frames.shape[2],
                            teacher_frames.shape[3],
                        ),
                        num_neighbors=self.num_neighbors,
                        temporal_tau=self.temporal_tau,
                        spatial_radius=0,
                        patch_match_mode=self.patch_match_mode,
                        patch_match_temperature=self.patch_match_temperature,
                        motion_weights=flat_motion_weights,
                    )
            else:
                student_frames = F.normalize(temporal_student.mean(dim=2), dim=-1)
                teacher_frames = F.normalize(temporal_teacher.mean(dim=2), dim=-1)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                frame_cosine = self._neighbor_weighted_cosine(
                    student_frames,
                    teacher_frames,
                    num_neighbors=self.num_neighbors,
                    temporal_tau=self.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            self.last_frame_cosine = frame_cosine.detach()
            total = total + (-frame_cosine if self.rep_loss_type == "negative_cosine" else 1.0 - frame_cosine) * self.current_lambda_temporal

        if (
            self.temporal_mode in {"delta", "hybrid"}
            and self.current_lambda_delta > 0.0
            and temporal_student is not None
            and temporal_teacher is not None
            and temporal_student.shape[1] > 1
            and temporal_teacher.shape[1] > 1
        ):
            if self.temporal_granularity == "patch":
                delta_cosine = self._multi_step_delta_cosine(
                    temporal_student,
                    temporal_teacher,
                    delta_num_steps=self.delta_num_steps,
                    temporal_tau=self.temporal_tau,
                    motion_weights=temporal_motion_weights,
                )
            else:
                student_frames = temporal_student.mean(dim=2)
                teacher_frames = temporal_teacher.mean(dim=2)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                delta_cosine = self._multi_step_delta_cosine(
                    student_frames,
                    teacher_frames,
                    delta_num_steps=self.delta_num_steps,
                    temporal_tau=self.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            if delta_cosine is not None and torch.isfinite(delta_cosine):
                self.last_delta_cosine = delta_cosine.detach()
                total = total + (
                    -delta_cosine
                    if self.rep_loss_type == "negative_cosine"
                    else 1.0 - delta_cosine
                ) * self.current_lambda_delta

        if self.max_loss > 0.0:
            loss_abs = float(total.detach().abs().item())
            if loss_abs > self.max_loss:
                total = total * (self.max_loss / loss_abs)
        self.last_self_flow_loss = total.detach()

        self.cleanup_step()
        return total
