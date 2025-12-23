import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image
from enhancements.repa.enhanced_repa_helper import (
    MultiEncoderProjectionHead,
    interpolate_features_spatial,
)

logger = get_logger(__name__, level=logging.INFO)


def _mean_over_tokens(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean(dim=list(range(1, loss.dim())))


class HasteHelper(nn.Module):
    """Holistic alignment helper with stage-wise termination (HASTE)."""

    def __init__(self, diffusion_model: Any, args: Any, device: Optional[Any] = None):
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.device = device or next(diffusion_model.parameters()).device

        self.enabled = bool(getattr(args, "enable_haste", False))
        self.alignment_depth = int(getattr(args, "haste_alignment_depth", 8))
        self.attn_layer_start = int(getattr(args, "haste_attn_layer_start", 4))
        self.attn_layer_end = int(getattr(args, "haste_attn_layer_end", 8))
        self.input_resolution = int(getattr(args, "haste_input_resolution", 256))
        self.encoder_name = str(
            getattr(args, "haste_encoder_name", "dinov2-vit-b")
        )
        self.use_teacher_attention = bool(
            getattr(args, "haste_use_teacher_attention", True)
        )
        self.teacher_attn_layer_offset = int(
            getattr(args, "haste_teacher_attn_layer_offset", 4)
        )
        self.attn_head_limit = int(getattr(args, "haste_attn_head_limit", 12))

        self.encoder_manager = EncoderManager(self.device)
        self.encoders, self.encoder_types, _ = self.encoder_manager.load_encoders(
            self.encoder_name, resolution=self.input_resolution
        )
        self.encoder_dims = [enc.embed_dim for enc in self.encoders]

        self.diffusion_hidden_dim = self._infer_diffusion_hidden_dim()
        self.projection_heads = MultiEncoderProjectionHead(
            diffusion_hidden_dim=self.diffusion_hidden_dim,
            encoder_dims=self.encoder_dims,
            ensemble_mode="individual",
            shared_projection=False,
            projection_type="mlp",
        )

        self._hooks: List[Any] = []
        self._captured_features: Dict[int, torch.Tensor] = {}
        self._blocks: Optional[Sequence[nn.Module]] = None
        self._warned_missing_layer = False
        self._teacher_attn_hooks: List[Any] = []
        self._teacher_attn_maps: Dict[int, torch.Tensor] = {}
        self._attn_hooks: List[Any] = []
        self._student_attn_logits: Dict[int, torch.Tensor] = {}
        self._warned_missing_attn = False

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning(
            "HASTE: Could not determine diffusion hidden dim; using 1024."
        )
        return 1024

    def _get_blocks(self) -> Sequence[nn.Module]:
        if self._blocks is not None:
            return self._blocks
        if hasattr(self.diffusion_model, "blocks"):
            self._blocks = list(self.diffusion_model.blocks)
        elif hasattr(self.diffusion_model, "layers"):
            self._blocks = list(self.diffusion_model.layers)
        elif hasattr(self.diffusion_model, "transformer_blocks"):
            self._blocks = list(self.diffusion_model.transformer_blocks)
        else:
            blocks = []
            for name, module in self.diffusion_model.named_modules():
                if "block" in name.lower() or "layer" in name.lower():
                    blocks.append(module)
            self._blocks = blocks
        return self._blocks or []

    def _capture_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inp: Any, output: Any) -> None:
            features = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(features):
                self._captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        if not self.enabled:
            return
        blocks = self._get_blocks()
        if not blocks:
            raise ValueError("HASTE: Could not resolve diffusion blocks to hook.")

        max_idx = len(blocks) - 1
        if self.alignment_depth < 0 or self.alignment_depth > max_idx:
            raise ValueError(
                f"HASTE alignment depth {self.alignment_depth} out of range [0, {max_idx}]"
            )
        if self.attn_layer_start < 0 or self.attn_layer_end <= self.attn_layer_start:
            raise ValueError(
                "HASTE attention layer range must be >= 0 with end > start."
            )
        if self.attn_layer_end - 1 > max_idx:
            raise ValueError(
                f"HASTE attention layer end {self.attn_layer_end} exceeds max index {max_idx}."
            )

        layer_indices = {self.alignment_depth}
        layer_indices.update(range(self.attn_layer_start, self.attn_layer_end))

        for idx in sorted(layer_indices):
            handle = blocks[idx].register_forward_hook(self._capture_hook(idx))
            self._hooks.append(handle)
            attn_handle = self._attach_student_attn_hook(idx, blocks[idx])
            if attn_handle is not None:
                self._attn_hooks.append(attn_handle)

        logger.info(
            "HASTE hooks attached (alignment_depth=%d, attn_layers=%s)",
            self.alignment_depth,
            list(range(self.attn_layer_start, self.attn_layer_end)),
        )

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks = []
        self._captured_features = {}
        for handle in self._attn_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._attn_hooks = []
        self._student_attn_logits = {}

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.projection_heads.parameters())

    def _ensure_token_features(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 4:
            bsz, channels, height, width = feats.shape
            feats = feats.view(bsz, channels, height * width).transpose(1, 2)
        elif feats.dim() == 3:
            pass
        elif feats.dim() == 2:
            feats = feats.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected diffusion feature shape: {feats.shape}")
        return feats

    def _compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        # Use token affinity as a lightweight attention proxy.
        normalized = F.normalize(features, dim=-1)
        attn = torch.matmul(normalized, normalized.transpose(-2, -1))
        return F.softmax(attn, dim=-1)

    def _attach_student_attn_hook(
        self, layer_idx: int, block: nn.Module
    ) -> Optional[Any]:
        attn_module = getattr(block, "self_attn", None)
        if attn_module is None:
            return None
        return attn_module.register_forward_hook(
            self._capture_student_attention(layer_idx, attn_module)
        )

    def _capture_student_attention(self, layer_idx: int, attn_module: nn.Module):
        def hook(_module: nn.Module, inputs: Tuple[Any, ...], _output: Any) -> None:
            try:
                logits = self._compute_student_attention_logits(attn_module, inputs)
            except Exception as exc:
                if not self._warned_missing_attn:
                    logger.warning(
                        "HASTE: failed to capture student attention logits at layer %d (%s).",
                        layer_idx,
                        exc,
                    )
                    self._warned_missing_attn = True
                return
            if logits is not None:
                self._student_attn_logits[layer_idx] = logits

        return hook

    def _apply_batched_rotary(
        self, tensor: torch.Tensor, rot: torch.Tensor
    ) -> torch.Tensor:
        bsz, tokens, heads, dim = tensor.shape
        tensor_c = torch.view_as_complex(
            tensor.float().reshape(bsz, tokens, heads, dim // 2, 2)
        )
        rot_c = torch.view_as_complex(
            rot.float().reshape(bsz, tokens, 1, dim // 2, 2)
        )
        out = torch.view_as_real(tensor_c * rot_c).reshape(
            bsz, tokens, heads, dim
        )
        return out.type_as(tensor)

    def _compute_student_attention_logits(
        self, attn_module: nn.Module, inputs: Tuple[Any, ...]
    ) -> Optional[torch.Tensor]:
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        if len(inputs) < 4:
            return None
        grid_sizes = inputs[2]
        freqs = inputs[3]
        sparse_attention = False
        batched_rotary = None
        if len(inputs) >= 5:
            sparse_attention = bool(inputs[4])
        if len(inputs) >= 6:
            batched_rotary = inputs[5]
        if sparse_attention:
            return None

        bsz, tokens = x.shape[0], x.shape[1]
        num_heads = getattr(attn_module, "num_heads", None)
        head_dim = getattr(attn_module, "head_dim", None)
        if num_heads is None or head_dim is None:
            return None

        q = attn_module.q(x)
        k = attn_module.k(x)
        q = attn_module.norm_q(q)
        k = attn_module.norm_k(k)
        q = q.view(bsz, tokens, num_heads, head_dim)
        k = k.view(bsz, tokens, num_heads, head_dim)

        if batched_rotary is not None:
            q = self._apply_batched_rotary(q, batched_rotary)
            k = self._apply_batched_rotary(k, batched_rotary)
        else:
            from wan.modules.model import rope_apply, rope_apply_inplace_cached

            use_comfy = bool(getattr(attn_module, "use_comfy_rope", False))
            rope_func = getattr(attn_module, "rope_func", "default")
            rope_on_the_fly = bool(getattr(attn_module, "rope_on_the_fly", False))
            if use_comfy and rope_func == "comfy":
                try:
                    q, k = attn_module.comfyrope(q, k, freqs)  # type: ignore[attr-defined]
                except Exception:
                    rope_apply_inplace_cached(q, grid_sizes, freqs)
                    rope_apply_inplace_cached(k, grid_sizes, freqs)
            elif rope_on_the_fly:
                q = rope_apply(q, grid_sizes, freqs)
                k = rope_apply(k, grid_sizes, freqs)
            else:
                rope_apply_inplace_cached(q, grid_sizes, freqs)
                rope_apply_inplace_cached(k, grid_sizes, freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scale = 1.0 / math.sqrt(head_dim)
        return torch.matmul(q, k.transpose(-2, -1)) * scale

    def _capture_teacher_attn(self, layer_idx: int, attn_module: nn.Module):
        def hook(_module: nn.Module, inputs: Tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            bsz, tokens, dim = x.shape
            num_heads = getattr(attn_module, "num_heads", None)
            scale = getattr(attn_module, "scale", None)
            qkv = attn_module.qkv(x)
            if num_heads is None:
                return
            head_dim = dim // num_heads
            if scale is None:
                scale = head_dim**-0.5
            qkv = qkv.reshape(bsz, tokens, 3, num_heads, head_dim).permute(
                2, 0, 3, 1, 4
            )
            q, k = qkv[0] * scale, qkv[1]
            attn_logits = torch.matmul(q, k.transpose(-2, -1))
            # Remove class token to match HASTE reference behavior.
            self._teacher_attn_maps[layer_idx] = attn_logits[:, :, 1:, 1:]

        return hook

    def _get_teacher_attention(
        self, encoder: nn.Module, processed_images: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        if not self.use_teacher_attention:
            return {}
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            return {}

        self._teacher_attn_maps = {}
        start_idx = self.attn_layer_start + self.teacher_attn_layer_offset
        end_idx = self.attn_layer_end + self.teacher_attn_layer_offset
        for idx in range(start_idx, end_idx):
            if idx >= len(blocks):
                break
            attn_module = getattr(blocks[idx], "attn", None)
            if attn_module is None or not hasattr(attn_module, "qkv"):
                continue
            handle = attn_module.register_forward_hook(
                self._capture_teacher_attn(idx, attn_module)
            )
            self._teacher_attn_hooks.append(handle)

        _ = encoder.forward_features(processed_images)

        for handle in self._teacher_attn_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._teacher_attn_hooks = []
        return self._teacher_attn_maps

    def _get_teacher_features(
        self, clean_pixels: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[int, torch.Tensor]]:
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0, :, :]

        with torch.no_grad():
            images = ((clean_pixels + 1) / 2.0).clamp(0, 1) * 255.0
            features_list: List[torch.Tensor] = []
            teacher_attn_maps: Dict[int, torch.Tensor] = {}
            for encoder, encoder_type in zip(self.encoders, self.encoder_types):
                processed = preprocess_raw_image(images, encoder_type)
                if (
                    self.use_teacher_attention
                    and "dinov2" in encoder_type
                    and not teacher_attn_maps
                ):
                    teacher_attn_maps = self._get_teacher_attention(
                        encoder.encoder, processed
                    )
                target_features = encoder.forward_features(processed)
                if isinstance(target_features, dict):
                    if "x_norm_patchtokens" in target_features:
                        target_features = target_features["x_norm_patchtokens"]
                    elif "x_norm_clstoken" in target_features:
                        target_features = target_features["x_norm_clstoken"]
                    else:
                        for value in target_features.values():
                            if torch.is_tensor(value) and value.dim() >= 2:
                                target_features = value
                                break
                if not torch.is_tensor(target_features):
                    raise ValueError("HASTE: encoder output is not a tensor.")
                if target_features.dim() == 2:
                    target_features = target_features.unsqueeze(1)
                elif target_features.dim() == 4:
                    bsz, channels, height, width = target_features.shape
                    target_features = target_features.view(
                        bsz, channels, height * width
                    ).transpose(1, 2)
                features_list.append(target_features)
        return features_list, teacher_attn_maps

    def get_haste_losses(
        self, clean_pixels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            zero = torch.tensor(0.0, device=clean_pixels.device)
            return zero, zero

        if self.alignment_depth not in self._captured_features:
            if not self._warned_missing_layer:
                logger.warning(
                    "HASTE: missing captured features for alignment depth %d.",
                    self.alignment_depth,
                )
                self._warned_missing_layer = True
            zero = torch.tensor(0.0, device=clean_pixels.device)
            return zero, zero

        teacher_features_list, teacher_attn_maps = self._get_teacher_features(
            clean_pixels
        )
        student_proj = self._ensure_token_features(
            self._captured_features[self.alignment_depth]
        )

        projected_list = self.projection_heads(student_proj)
        proj_loss = torch.tensor(0.0, device=clean_pixels.device)
        for projected, target in zip(projected_list, teacher_features_list):
            if projected.shape[1] != target.shape[1]:
                target = interpolate_features_spatial(target, projected.shape[1])
            projected = F.normalize(projected, dim=-1)
            target = F.normalize(target, dim=-1)
            proj_loss = proj_loss + _mean_over_tokens(
                -(projected * target).sum(dim=-1)
            )
        proj_loss = proj_loss / max(1, len(projected_list))

        attn_layers = range(self.attn_layer_start, self.attn_layer_end)
        attn_loss = torch.tensor(0.0, device=clean_pixels.device)
        used_layers = 0
        for layer_idx in attn_layers:
            student_logits = self._student_attn_logits.get(layer_idx)
            teacher_logits = teacher_attn_maps.get(
                layer_idx + self.teacher_attn_layer_offset
            )
            if student_logits is None or teacher_logits is None:
                student_feats = self._captured_features.get(layer_idx)
                if student_feats is None:
                    continue
                student_tokens = self._ensure_token_features(student_feats)
                teacher_tokens = teacher_features_list[0]
                if teacher_tokens.shape[1] != student_tokens.shape[1]:
                    teacher_tokens = interpolate_features_spatial(
                        teacher_tokens, student_tokens.shape[1]
                    )
                teacher_attn = self._compute_attention_map(teacher_tokens)
                student_attn = self._compute_attention_map(student_tokens)
                attn_loss = attn_loss + _mean_over_tokens(
                    -(teacher_attn * torch.log(student_attn + 1e-8)).sum(dim=-1)
                )
                used_layers += 1
                continue

            head_limit = self.attn_head_limit
            max_heads = min(student_logits.shape[1], teacher_logits.shape[1])
            if head_limit > 0:
                max_heads = min(max_heads, head_limit)
            student_logits = student_logits[:, :max_heads]
            teacher_logits = teacher_logits[:, :max_heads]
            if student_logits.shape[-1] != teacher_logits.shape[-1]:
                if not self._warned_missing_attn:
                    logger.warning(
                        "HASTE: attention token mismatch at layer %d (student=%d, teacher=%d).",
                        layer_idx,
                        student_logits.shape[-1],
                        teacher_logits.shape[-1],
                    )
                    self._warned_missing_attn = True
                continue

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            attn_loss = attn_loss + _mean_over_tokens(
                -(teacher_probs * student_log_probs).sum(dim=-1)
            )
            used_layers += 1

        attn_count = max(1, used_layers)
        attn_loss = attn_loss / attn_count
        return attn_loss, proj_loss

    def compute_weighted_losses(
        self, clean_pixels: torch.Tensor, global_step: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_loss, proj_loss = self.get_haste_losses(clean_pixels)
        current_step = int(global_step or 0)
        if current_step < int(getattr(self.args, "haste_early_stop_step", 0) or 0):
            attn_loss_value = attn_loss.mean()
            proj_loss_value = proj_loss.mean()
        else:
            attn_loss_value = torch.zeros_like(attn_loss)
            proj_loss_value = torch.zeros_like(proj_loss)

        total = (
            proj_loss_value * float(getattr(self.args, "haste_proj_coeff", 0.0))
            + attn_loss_value * float(getattr(self.args, "haste_attn_coeff", 0.0))
        )
        return total, attn_loss_value, proj_loss_value
