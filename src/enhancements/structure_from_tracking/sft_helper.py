"""Structure-From-Tracking helper for LGF-KL motion distillation."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from common.logger import get_logger
from enhancements.structure_from_tracking.groundingdino_prompter import (
    GroundingDINOPrompter,
)
from enhancements.structure_from_tracking.lgf import (
    compute_lgf_alignment_loss,
    compute_local_gram_flow,
)
from enhancements.structure_from_tracking.sam2_teacher import SAM2Teacher

logger = get_logger(__name__)


def _interpolate_token_count(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token features to match a target token count."""
    if tokens.shape[1] == target_tokens:
        return tokens

    batch_frames, src_tokens, dim = tokens.shape
    src_side = int(math.isqrt(src_tokens))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_tokens:
        x = tokens.permute(0, 2, 1).reshape(batch_frames, dim, src_side, src_side)
        x = nnf.interpolate(
            x,
            size=(tgt_side, tgt_side),
            mode="bilinear",
            align_corners=False,
        )
        return x.reshape(batch_frames, dim, target_tokens).permute(0, 2, 1)

    x = tokens.permute(0, 2, 1)
    x = nnf.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


class _TokenProjector(nn.Module):
    """Three-layer projector with optional GroupNorm and residual skip path."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        use_group_norm: bool = True,
        gn_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError(
                "_TokenProjector hidden_dims must contain exactly 3 entries."
            )
        d1, d2, d3 = [int(v) for v in hidden_dims]
        self.fc1 = nn.Linear(in_dim, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, d3)
        self.fc_out = nn.Linear(d3, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.use_group_norm = bool(use_group_norm)
        if self.use_group_norm:
            g1 = max(1, min(int(gn_groups), d1))
            while d1 % g1 != 0 and g1 > 1:
                g1 -= 1
            g2 = max(1, min(int(gn_groups), d2))
            while d2 % g2 != 0 and g2 > 1:
                g2 -= 1
            g3 = max(1, min(int(gn_groups), d3))
            while d3 % g3 != 0 and g3 > 1:
                g3 -= 1
            self.gn1 = nn.GroupNorm(g1, d1)
            self.gn2 = nn.GroupNorm(g2, d2)
            self.gn3 = nn.GroupNorm(g3, d3)
        self.act = nn.SiLU()

    def _apply_group_norm(self, tensor: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
        # [B, S, C] -> [B, C, S] for GroupNorm -> [B, S, C]
        return norm(tensor.transpose(1, 2)).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.fc1(tokens)
        if self.use_group_norm:
            x = self._apply_group_norm(x, self.gn1)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_group_norm:
            x = self._apply_group_norm(x, self.gn2)
        x = self.act(x)
        x = self.fc3(x)
        if self.use_group_norm:
            x = self._apply_group_norm(x, self.gn3)
        x = self.act(x)
        x = self.fc_out(x)
        return x + self.skip(tokens)


class StructureFromTrackingHelper(nn.Module):
    """Train-time Structure-From-Tracking distillation helper."""

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []
        self._shape_warning_logged = False
        self._mask_warning_logged = False
        self.paper_strict_mode = bool(getattr(args, "sft_paper_strict_mode", False))

        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = SAM2Teacher(args=args, device=torch.device(device))
        self.dino_prompter = GroundingDINOPrompter(
            args=args,
            device=torch.device(device),
        )
        self.teacher_dim = int(self.teacher.embed_dim)

        raw_depths = getattr(args, "sft_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            self.alignment_depths = [int(v) for v in raw_depths]
        else:
            self.alignment_depths = [int(getattr(args, "sft_alignment_depth", 25))]
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        self.projector_hidden_dim = int(
            getattr(args, "sft_projector_hidden_dim", 2048)
        )
        raw_projector_dims = getattr(args, "sft_projector_dims", [512, 256, 256])
        self.projector_dims = [int(v) for v in raw_projector_dims]
        self.projector_group_norm = bool(
            getattr(args, "sft_projector_group_norm", True)
        )
        self.projector_gn_groups = int(getattr(args, "sft_projector_gn_groups", 32))
        self.projectors = nn.ModuleList(
            [
                _TokenProjector(
                    in_dim=self.hidden_dim,
                    hidden_dims=self.projector_dims,
                    out_dim=self.teacher_dim,
                    use_group_norm=self.projector_group_norm,
                    gn_groups=self.projector_gn_groups,
                )
                for _ in self.alignment_depths
            ]
        )
        temporal_kernel_size = int(getattr(args, "sft_temporal_kernel_size", 3))
        self.temporal_mixers = nn.ModuleList(
            [
                nn.Conv1d(
                    self.teacher_dim,
                    self.teacher_dim,
                    kernel_size=temporal_kernel_size,
                    padding=temporal_kernel_size // 2,
                )
                for _ in self.alignment_depths
            ]
        )

        self.loss_lambda = float(getattr(args, "sft_loss_lambda", 0.5))
        self.loss_mode = str(getattr(args, "sft_loss_mode", "kl")).lower()
        self.fusion_mode = str(getattr(args, "sft_fusion_mode", "lgf")).lower()
        self.kernel_size = int(getattr(args, "sft_lgf_kernel_size", 7))
        self.temperature = float(getattr(args, "sft_lgf_temperature", 0.1))
        self.teacher_fusion_weight = float(
            getattr(args, "sft_teacher_fusion_weight", 0.5)
        )
        self.enable_backward_teacher = bool(
            getattr(args, "sft_enable_backward_teacher", True)
        )
        self.temporal_interp_factor = int(getattr(args, "sft_temporal_interp_factor", 4))
        self.max_spatial_tokens = int(getattr(args, "sft_max_spatial_tokens", -1))
        self.spatial_align = bool(getattr(args, "sft_spatial_align", True))
        self.temporal_align = bool(getattr(args, "sft_temporal_align", True))
        self.use_mask_prompting = bool(getattr(args, "sft_use_mask_prompting", False))

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning(
            "Structure-From-Tracking: falling back to hidden_dim=1024."
        )
        return 1024

    def _get_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def _locate_blocks(self) -> Any:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("Structure-From-Tracking: could not locate transformer blocks.")

    def setup_hooks(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        try:
            for idx, depth in enumerate(self.alignment_depths):
                if depth >= num_blocks:
                    raise ValueError(
                        f"SFT alignment depth {depth} exceeds available blocks ({num_blocks})"
                    )
                handle = blocks[depth].register_forward_hook(self._get_hook(idx))
                self.hook_handles.append(handle)
                logger.info(
                    "Structure-From-Tracking: hook attached to layer %d.",
                    depth,
                )
        except Exception:
            self.remove_hooks()
            raise

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()
        self.captured_features = [None] * len(self.alignment_depths)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.projectors.parameters()) + list(self.temporal_mixers.parameters())

    @staticmethod
    def _interpolate_frames(tokens: torch.Tensor, target_frames: int) -> torch.Tensor:
        if tokens.shape[1] == target_frames:
            return tokens
        batch_size, frames, token_count, dim = tokens.shape
        x = tokens.permute(0, 2, 3, 1).reshape(batch_size * token_count, dim, frames)
        x = nnf.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        return x.reshape(batch_size, token_count, dim, target_frames).permute(0, 3, 1, 2)

    def _reshape_source_tokens(
        self,
        projected: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = projected.shape
        if target_frames > 0 and seq_len % target_frames == 0:
            frames = target_frames
        else:
            frames = 1
            if target_frames > 1:
                for candidate in range(target_frames, 0, -1):
                    if seq_len % candidate == 0:
                        frames = candidate
                        break

        tokens_per_frame = seq_len // max(1, frames)
        source = projected.view(batch_size, frames, tokens_per_frame, projected.shape[-1])
        if self.temporal_align and frames != target_frames and target_frames > 0:
            source = self._interpolate_frames(source, target_frames=target_frames)
        return source

    def _apply_temporal_projection(
        self,
        layer_index: int,
        source_tokens: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """Apply temporal interpolation + residual skip projection before LGF."""
        factor = max(1, int(self.temporal_interp_factor))
        if factor <= 1 or source_tokens.shape[1] <= 1:
            return source_tokens

        source_frames = int(source_tokens.shape[1])
        effective_target = target_frames if target_frames > 0 else source_frames
        projected_frames = min(max(source_frames, effective_target), source_frames * factor)

        batch_size, _, token_count, channel_count = source_tokens.shape
        temporal = source_tokens.permute(0, 2, 3, 1).reshape(
            batch_size * token_count,
            channel_count,
            source_frames,
        )
        upsampled = nnf.interpolate(
            temporal,
            size=projected_frames,
            mode="linear",
            align_corners=False,
        )
        mixed = self.temporal_mixers[layer_index](upsampled)
        # Residual skip branch from the upsampled source features.
        mixed = 0.5 * (mixed + upsampled)
        return mixed.reshape(
            batch_size,
            token_count,
            channel_count,
            projected_frames,
        ).permute(0, 3, 1, 2)

    def _match_temporal_and_spatial(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.shape[1] != target.shape[1]:
            if self.temporal_align:
                source = self._interpolate_frames(source, target_frames=target.shape[1])
            else:
                shared_frames = min(source.shape[1], target.shape[1])
                source = source[:, :shared_frames]
                target = target[:, :shared_frames]

        src_tokens = source.shape[2]
        tgt_tokens = target.shape[2]
        if src_tokens != tgt_tokens:
            source_2d = source.reshape(source.shape[0] * source.shape[1], src_tokens, source.shape[3])
            target_2d = target.reshape(target.shape[0] * target.shape[1], tgt_tokens, target.shape[3])
            if self.spatial_align:
                if src_tokens > tgt_tokens:
                    source_2d = _interpolate_token_count(source_2d, tgt_tokens)
                else:
                    target_2d = _interpolate_token_count(target_2d, src_tokens)
            else:
                min_tokens = min(src_tokens, tgt_tokens)
                source_2d = source_2d[:, :min_tokens]
                target_2d = target_2d[:, :min_tokens]
            source = source_2d.view(source.shape[0], source.shape[1], source_2d.shape[1], source.shape[3])
            target = target_2d.view(target.shape[0], target.shape[1], target_2d.shape[1], target.shape[3])

        if self.max_spatial_tokens > 0 and source.shape[2] > self.max_spatial_tokens:
            source_2d = source.reshape(source.shape[0] * source.shape[1], source.shape[2], source.shape[3])
            target_2d = target.reshape(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])
            source_2d = _interpolate_token_count(source_2d, self.max_spatial_tokens)
            target_2d = _interpolate_token_count(target_2d, self.max_spatial_tokens)
            source = source_2d.view(source.shape[0], source.shape[1], self.max_spatial_tokens, source.shape[3])
            target = target_2d.view(target.shape[0], target.shape[1], self.max_spatial_tokens, target.shape[3])
        return source, target

    def _align_teacher_to_source(
        self,
        teacher_tokens: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        aligned = teacher_tokens
        if aligned.shape[1] != reference.shape[1]:
            if self.temporal_align:
                aligned = self._interpolate_frames(aligned, target_frames=reference.shape[1])
            else:
                aligned = aligned[:, : reference.shape[1]]

        if aligned.shape[2] != reference.shape[2]:
            aligned_2d = aligned.reshape(
                aligned.shape[0] * aligned.shape[1],
                aligned.shape[2],
                aligned.shape[3],
            )
            aligned_2d = _interpolate_token_count(aligned_2d, reference.shape[2])
            aligned = aligned_2d.view(
                aligned.shape[0],
                aligned.shape[1],
                reference.shape[2],
                aligned.shape[3],
            )
        return aligned

    @staticmethod
    def _coerce_mask_array(
        raw_mask: Any,
        target_frames: int,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Convert per-item mask-like data into [F, H, W] float tensor."""
        mask = torch.as_tensor(raw_mask, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            if mask.shape[-1] in (1, 3, 4):
                mask = mask.mean(dim=-1, keepdim=False).unsqueeze(0)
            elif mask.shape[0] in (1, 3, 4):
                mask = mask.mean(dim=0, keepdim=True)
        elif mask.dim() == 4:
            if mask.shape[-1] in (1, 3, 4):
                mask = mask.mean(dim=-1)
            elif mask.shape[1] in (1, 3, 4):
                mask = mask.mean(dim=1)
        else:
            raise ValueError(f"Unsupported mask dimensionality: {tuple(mask.shape)}")

        if mask.dim() != 3:
            raise ValueError(f"Mask must resolve to [F,H,W], got {tuple(mask.shape)}")

        if mask.shape[0] != target_frames:
            mask = nnf.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(target_frames, target_height, target_width),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        elif mask.shape[1] != target_height or mask.shape[2] != target_width:
            mask = nnf.interpolate(
                mask.unsqueeze(1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return mask.clamp(0.0, 1.0)

    def _build_mask_hints(
        self,
        clean_pixels: torch.Tensor,
        item_info: Optional[Sequence[Any]],
    ) -> Optional[torch.Tensor]:
        if not self.use_mask_prompting or item_info is None:
            if self.paper_strict_mode:
                raise ValueError(
                    "Structure-From-Tracking strict mode requires mask prompting inputs (item_info) for each sample."
                )
            return None
        if not isinstance(item_info, (list, tuple)) or len(item_info) == 0:
            if self.paper_strict_mode:
                raise ValueError(
                    "Structure-From-Tracking strict mode requires non-empty item_info for mask prompting."
                )
            return None

        target_frames = int(clean_pixels.shape[2]) if clean_pixels.dim() == 5 else 1
        target_height = int(clean_pixels.shape[-2])
        target_width = int(clean_pixels.shape[-1])
        batch_size = int(clean_pixels.shape[0])
        if len(item_info) != batch_size:
            if self.paper_strict_mode:
                raise ValueError(
                    "Structure-From-Tracking strict mode requires item_info batch size to match clean_pixels batch size."
                )
            return None

        has_any_mask = False
        masks: List[torch.Tensor] = []
        missing_indices = []
        for item in item_info:
            raw = getattr(item, "mask_content", None)
            if raw is None:
                masks.append(
                    torch.zeros(
                        target_frames,
                        target_height,
                        target_width,
                        dtype=torch.float32,
                    )
                )
                missing_indices.append(len(masks) - 1)
                continue
            try:
                mask_tensor = self._coerce_mask_array(
                    raw_mask=raw,
                    target_frames=target_frames,
                    target_height=target_height,
                    target_width=target_width,
                )
                has_any_mask = True
                masks.append(mask_tensor)
            except Exception:
                masks.append(
                    torch.zeros(
                        target_frames,
                        target_height,
                        target_width,
                        dtype=torch.float32,
                    )
                )
                missing_indices.append(len(masks) - 1)

        # If enabled, fill missing masks with GroundingDINO-derived box masks.
        if missing_indices and self.dino_prompter.enabled:
            dino_masks = self.dino_prompter.generate_mask_hints(
                clean_pixels=clean_pixels,
                item_info=item_info,
            )
            if dino_masks is not None:
                dino_masks = dino_masks.to(device=clean_pixels.device, dtype=torch.float32)
                for idx in missing_indices:
                    candidate = dino_masks[idx]
                    if candidate.max().item() > 0:
                        masks[idx] = candidate
                        has_any_mask = True

        if not has_any_mask:
            if self.paper_strict_mode:
                raise ValueError(
                    "Structure-From-Tracking strict mode requires usable mask hints; none were resolved from dataset or GroundingDINO."
                )
            if not self._mask_warning_logged:
                logger.warning(
                    "Structure-From-Tracking mask prompting is enabled but no usable masks were found in item_info."
                )
                self._mask_warning_logged = True
            return None

        return torch.stack(masks, dim=0).to(device=clean_pixels.device)

    def get_repa_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
        item_info: Optional[Sequence[Any]] = None,
    ) -> torch.Tensor:
        del vae
        if not any(feat is not None for feat in self.captured_features):
            return clean_pixels.new_tensor(0.0)

        item_keys = None
        if isinstance(item_info, (list, tuple)) and len(item_info) == clean_pixels.shape[0]:
            item_keys = [
                str(getattr(item, "item_key", f"sample_{idx}"))
                for idx, item in enumerate(item_info)
            ]
        mask_hints = self._build_mask_hints(clean_pixels=clean_pixels, item_info=item_info)

        teacher_forward, teacher_backward = self.teacher.extract_bidirectional_features(
            clean_pixels=clean_pixels,
            include_backward=self.enable_backward_teacher,
            item_keys=item_keys,
            mask_hints=mask_hints,
        )
        losses: List[torch.Tensor] = []

        for idx, features in enumerate(self.captured_features):
            if features is None or not isinstance(features, torch.Tensor):
                continue
            if features.dim() != 3:
                if not self._shape_warning_logged:
                    logger.warning(
                        "Structure-From-Tracking: expected diffusion feature shape [B, Seq, C], got %s. Skipping layer loss.",
                        tuple(features.shape),
                    )
                    self._shape_warning_logged = True
                continue

            projected = self.projectors[idx](features)
            source_tokens = self._reshape_source_tokens(
                projected,
                target_frames=teacher_forward.shape[1],
            )
            source_tokens = self._apply_temporal_projection(
                layer_index=idx,
                source_tokens=source_tokens,
                target_frames=teacher_forward.shape[1],
            )
            source_tokens, forward_tokens = self._match_temporal_and_spatial(
                source=source_tokens,
                target=teacher_forward,
            )

            backward_tokens = None
            if teacher_backward is not None:
                backward_tokens = self._align_teacher_to_source(
                    teacher_tokens=teacher_backward,
                    reference=source_tokens,
                )

            source_tokens = nnf.normalize(source_tokens, dim=-1)
            forward_tokens = nnf.normalize(forward_tokens, dim=-1)
            if backward_tokens is not None:
                backward_tokens = nnf.normalize(backward_tokens, dim=-1)

            student_lgf = compute_local_gram_flow(
                source_tokens,
                kernel_size=self.kernel_size,
            )
            forward_lgf = compute_local_gram_flow(
                forward_tokens,
                kernel_size=self.kernel_size,
            )
            if backward_tokens is not None:
                if self.fusion_mode == "feature":
                    teacher_tokens = (
                        self.teacher_fusion_weight * forward_tokens
                        + (1.0 - self.teacher_fusion_weight) * backward_tokens
                    )
                    teacher_lgf = compute_local_gram_flow(
                        teacher_tokens,
                        kernel_size=self.kernel_size,
                    )
                else:
                    backward_lgf = compute_local_gram_flow(
                        backward_tokens,
                        kernel_size=self.kernel_size,
                    )
                    teacher_lgf = (
                        self.teacher_fusion_weight * forward_lgf
                        + (1.0 - self.teacher_fusion_weight) * backward_lgf
                    )
            else:
                teacher_lgf = forward_lgf

            layer_loss = compute_lgf_alignment_loss(
                student_similarity=student_lgf,
                teacher_similarity=teacher_lgf,
                mode=self.loss_mode,
                temperature=self.temperature,
            )
            losses.append(layer_loss)

        self.captured_features = [None] * len(self.alignment_depths)
        if not losses:
            return clean_pixels.new_tensor(0.0)

        total_loss = torch.stack(losses).mean()
        return total_loss * self.loss_lambda
