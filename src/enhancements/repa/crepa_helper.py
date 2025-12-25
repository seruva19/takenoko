import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger


logger = get_logger(__name__)


class CrepaHelper(nn.Module):
    """Cross-frame Representation Alignment (CREPA) regularizer helper."""

    def __init__(self, transformer: nn.Module, args: Any, accelerator: Any) -> None:
        super().__init__()
        self.args = args
        self.transformer = self._unwrap_model(transformer)
        self.device = accelerator.device

        self.enabled = bool(getattr(args, "crepa_enabled", False))
        self.block_index = getattr(args, "crepa_block_index", None)
        self.teacher_block_index = getattr(args, "crepa_teacher_block_index", None)
        self.use_backbone_features = bool(
            getattr(args, "crepa_use_backbone_features", False)
        )

        self.distance = int(getattr(args, "crepa_adjacent_distance", 1))
        self.tau = float(getattr(args, "crepa_adjacent_tau", 1.0))
        self.weight = float(getattr(args, "crepa_lambda", 0.5) or 0.0)
        raw_encoder = getattr(args, "crepa_encoder", "dinov2_vitg14")
        self.encoder_name = self._resolve_encoder_name(raw_encoder)
        self.encoder_image_size = int(
            getattr(args, "crepa_encoder_image_size", 518) or 518
        )
        self.normalize_by_frames = bool(
            getattr(args, "crepa_normalize_by_frames", True)
        )
        self.spatial_align = bool(getattr(args, "crepa_spatial_align", True))
        self.cumulative_neighbors = bool(
            getattr(args, "crepa_cumulative_neighbors", False)
        )

        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_hidden: Optional[torch.Tensor] = None
        self._captured_teacher: Optional[torch.Tensor] = None

        self.hidden_size = self._infer_hidden_size()
        self.encoder: Optional[nn.Module] = None
        self.encoder_dim: Optional[int] = None
        self.projector: Optional[nn.Module] = None

        if self.enabled:
            if self.hidden_size is None:
                raise ValueError("CREPA enabled but unable to infer transformer hidden size.")
            self._init_projector()
            if not self.use_backbone_features:
                self._load_encoder()

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _infer_hidden_size(self) -> Optional[int]:
        config = getattr(self.transformer, "config", None)
        if config is not None:
            heads = getattr(config, "num_attention_heads", None)
            head_dim = getattr(config, "attention_head_dim", None)
            if heads is not None and head_dim is not None:
                return int(heads * head_dim)
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)
            model_dim = getattr(config, "model_dim", None)
            if model_dim is not None:
                return int(model_dim)
        if hasattr(self.transformer, "dim"):
            return int(self.transformer.dim)
        if hasattr(self.transformer, "hidden_size"):
            return int(self.transformer.hidden_size)
        return None

    def _init_projector(self) -> None:
        if self.projector is not None:
            return
        target_dim = self.hidden_size if self.use_backbone_features else None
        if target_dim is None and self.encoder_dim is not None:
            target_dim = self.encoder_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, target_dim or self.hidden_size),
        )
        self.projector.to(device=self.device, dtype=torch.float32)
        self.projector.requires_grad_(True)

    def attach_to_model(self, model: nn.Module) -> None:
        if not self.enabled or self.projector is None:
            return
        if getattr(model, "crepa_projector", None) is not self.projector:
            setattr(model, "crepa_projector", self.projector)

    def _load_encoder(self) -> None:
        if self.encoder is not None:
            return
        # DINOv2 torch hub exports lightweight models; user confirms network usage if needed.
        self.encoder = torch.hub.load("facebookresearch/dinov2", self.encoder_name)
        self.encoder.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

        dummy = torch.zeros(
            1, 3, self.encoder_image_size, self.encoder_image_size, device=self.device
        )
        with torch.no_grad():
            encoded = self._forward_encoder(dummy)
        self.encoder_dim = int(encoded.shape[-1])
        if self.projector is None:
            self._init_projector()
        if self.projector is not None:
            self.projector[-1] = nn.Linear(self.hidden_size, self.encoder_dim)
            self.projector.to(device=self.device, dtype=torch.float32)
            self.projector.requires_grad_(True)

    def get_trainable_params(self) -> list[nn.Parameter]:
        if self.projector is None:
            return []
        return [p for p in self.projector.parameters() if p.requires_grad]

    @staticmethod
    def _resolve_encoder_name(value: Optional[str]) -> str:
        if not value:
            return "dinov2_vitg14"
        value = str(value).strip()
        aliases = {
            "dino_v2_g": "dinov2_vitg14",
            "dinov2_g": "dinov2_vitg14",
            "dinov2-vitg14": "dinov2_vitg14",
            "dino_v2_s": "dinov2_vits14",
            "dinov2_s": "dinov2_vits14",
            "dinov2-vitb14": "dinov2_vitb14",
        }
        return aliases.get(value.lower(), value)

    def _target_blocks(self) -> Tuple[list[nn.Module], int]:
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None and hasattr(self.transformer, "module"):
            blocks = getattr(self.transformer.module, "blocks", None)
        if blocks is None:
            raise ValueError("CrepaHelper could not find transformer blocks to hook")
        return list(blocks), len(blocks)

    def _resolve_block_index(self, idx: Any, num_blocks: int) -> int:
        try:
            idx_int = int(idx)
        except Exception as exc:
            raise ValueError(f"CREPA block index {idx!r} is not an int.") from exc
        if 0 <= idx_int < num_blocks:
            return idx_int
        raise ValueError(
            f"CREPA block index {idx_int} is outside available range [0, {num_blocks - 1}]"
        )

    def _make_hook(self, role: str):
        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (list, tuple)) and output:
                tensor = output[0]
            if not torch.is_tensor(tensor):
                return
            if role == "student":
                self._captured_hidden = tensor
            else:
                self._captured_teacher = tensor

        return hook

    def setup_hooks(self) -> None:
        if not self.enabled:
            return
        self.remove_hooks()
        blocks, num_blocks = self._target_blocks()

        if hasattr(self.transformer, "blocks_to_swap"):
            try:
                blocks_to_swap = int(getattr(self.transformer, "blocks_to_swap", 0))
                if blocks_to_swap > 0:
                    logger.warning(
                        "CREPA: block swap/offload detected (%s blocks). Hooks may miss swapped blocks.",
                        blocks_to_swap,
                    )
            except Exception:
                pass

        if self.block_index is None:
            raise ValueError("crepa_block_index must be set when CREPA is enabled.")
        student_idx = self._resolve_block_index(self.block_index, num_blocks)
        self.hooks.append(
            blocks[student_idx].register_forward_hook(self._make_hook("student"))
        )

        if self.use_backbone_features:
            teacher_idx = (
                self.teacher_block_index
                if self.teacher_block_index is not None
                else self.block_index
            )
            if teacher_idx is None:
                raise ValueError(
                    "crepa_teacher_block_index must be set when backbone mode is enabled."
                )
            resolved_teacher = self._resolve_block_index(teacher_idx, num_blocks)
            self.hooks.append(
                blocks[resolved_teacher].register_forward_hook(
                    self._make_hook("teacher")
                )
            )

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks = []
        self._captured_hidden = None
        self._captured_teacher = None

    def compute_loss(
        self,
        *,
        clean_pixels: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        vae: Optional[nn.Module],
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        if not self.enabled or self.weight == 0:
            return None, None
        if self._captured_hidden is None:
            raise ValueError("CREPA is enabled but no intermediate hidden states were captured.")

        if self.use_backbone_features:
            if self._captured_teacher is None:
                raise ValueError("CREPA backbone feature mode requires teacher hidden states.")
            frame_features = self._normalize_frame_features(self._captured_teacher)
        else:
            if clean_pixels is None:
                if latents is None or vae is None:
                    raise ValueError("CREPA requires clean pixels or (latents + VAE).")
                clean_pixels = self._decode_latents(latents, vae)
            if clean_pixels.dim() == 4:
                clean_pixels = clean_pixels.unsqueeze(1)
            frame_features = self._encode_frames(clean_pixels)

        projected = self._project_hidden_states(self._captured_hidden)
        projected, frame_features = self._maybe_align_temporal(projected, frame_features)
        projected, frame_features = self._maybe_align_tokens(projected, frame_features)

        projected = F.normalize(projected, dim=-1)
        frame_features = F.normalize(frame_features, dim=-1)

        self_sim = (projected * frame_features).sum(dim=-1).mean(dim=-1)
        total_sim = self_sim.clone()
        bsz, num_frames = total_sim.shape
        d = min(self.distance, num_frames - 1)
        tau = max(self.tau, 1e-8)

        if d > 0:
            if self.cumulative_neighbors:
                for offset in range(1, d + 1):
                    weight = math.exp(-float(offset) / tau)
                    fwd = (projected[:, :-offset, ...] * frame_features[:, offset:, ...]).sum(dim=-1).mean(dim=-1)
                    total_sim[:, :-offset] += weight * fwd
                    back = (projected[:, offset:, ...] * frame_features[:, :-offset, ...]).sum(dim=-1).mean(dim=-1)
                    total_sim[:, offset:] += weight * back
            else:
                weight = math.exp(-float(d) / tau)
                fwd = (projected[:, :-d, ...] * frame_features[:, d:, ...]).sum(dim=-1).mean(dim=-1)
                total_sim[:, :-d] += weight * fwd
                back = (projected[:, d:, ...] * frame_features[:, :-d, ...]).sum(dim=-1).mean(dim=-1)
                total_sim[:, d:] += weight * back

        per_video_sum = total_sim.sum(dim=1)
        if self.normalize_by_frames:
            per_video_sum = per_video_sum / float(num_frames)

        align_loss = -per_video_sum.mean() * self.weight
        log_data = {
            "crepa_loss": align_loss.detach().item(),
            "crepa_similarity": total_sim.mean().detach().item(),
        }

        self._captured_hidden = None
        self._captured_teacher = None

        return align_loss, log_data

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.unsqueeze(2)
        elif hidden_states.ndim != 4:
            raise ValueError(
                f"CREPA expected hidden states with 3 or 4 dims, got {hidden_states.shape}"
            )
        b, t, p, d = hidden_states.shape
        projector_dtype = next(self.projector.parameters()).dtype
        flattened = hidden_states.to(dtype=projector_dtype).reshape(b * t * p, d)
        projected = self.projector(flattened)
        return projected.view(b, t, p, -1)

    def _maybe_align_temporal(
        self, projected: torch.Tensor, frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_proj = projected.shape[1]
        t_feat = frame_features.shape[1]
        if t_proj == t_feat:
            return projected, frame_features
        if t_feat > t_proj:
            indices = torch.linspace(0, t_feat - 1, t_proj, device=frame_features.device).long()
            frame_features = frame_features.index_select(1, indices)
        else:
            indices = torch.linspace(0, t_proj - 1, t_feat, device=projected.device).long()
            projected = projected.index_select(1, indices)
        return projected, frame_features

    def _maybe_align_tokens(
        self, projected: torch.Tensor, frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_tokens = projected.shape[2]
        enc_tokens = frame_features.shape[2]
        if proj_tokens == enc_tokens:
            return projected, frame_features
        if not self.spatial_align:
            projected = projected.mean(dim=2, keepdim=True)
            frame_features = frame_features.mean(dim=2, keepdim=True)
            return projected, frame_features
        target_tokens = min(proj_tokens, enc_tokens)
        projected = self._interpolate_tokens(projected, target_tokens)
        frame_features = self._interpolate_tokens(frame_features, target_tokens)
        return projected, frame_features

    def _interpolate_tokens(self, tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if tokens.shape[2] == target_tokens:
            return tokens
        b, t, n, d = tokens.shape
        flat = tokens.reshape(b * t, n, d).permute(0, 2, 1)
        src_size = int(math.sqrt(n))
        tgt_size = int(math.sqrt(target_tokens))
        if src_size * src_size == n and tgt_size * tgt_size == target_tokens:
            flat = flat.view(b * t, d, src_size, src_size)
            interpolated = F.interpolate(
                flat, size=(tgt_size, tgt_size), mode="bilinear", align_corners=False
            )
            interpolated = interpolated.view(b * t, d, target_tokens)
        else:
            interpolated = F.interpolate(flat, size=target_tokens, mode="linear", align_corners=False)
        return interpolated.permute(0, 2, 1).reshape(b, t, target_tokens, d)

    def _forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        enc_dtype = next(self.encoder.parameters()).dtype
        images = images.to(dtype=enc_dtype)
        with torch.no_grad():
            output = self.encoder(images)
        if isinstance(output, dict):
            if "x_norm_patchtokens" in output:
                tokens = output["x_norm_patchtokens"]
            elif "x_norm_clstoken" in output:
                tokens = output["x_norm_clstoken"].unsqueeze(1)
            else:
                tokens = next(iter(output.values()))
        elif torch.is_tensor(output):
            tokens = output
        elif isinstance(output, (list, tuple)):
            tokens = output[0]
        else:
            raise TypeError(f"Unsupported encoder output type: {type(output)}")
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise ValueError(f"Unexpected encoder token shape: {tokens.shape}")
        return tokens

    def _encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frames = F.interpolate(
            frames,
            size=(self.encoder_image_size, self.encoder_image_size),
            mode="bilinear",
            align_corners=False,
        )
        enc_dtype = next(self.encoder.parameters()).dtype
        frames = frames.to(dtype=enc_dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        tokens = self._forward_encoder(frames)
        return tokens.view(b, t, tokens.shape[1], -1)

    def _decode_latents(self, latents: torch.Tensor, vae: nn.Module) -> torch.Tensor:
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
        shift_factor = getattr(getattr(vae, "config", None), "shift_factor", None)
        if shift_factor is not None:
            latents = latents / scaling_factor + shift_factor
        else:
            latents = latents / scaling_factor
        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            view_shape = [1, latents.shape[1]] + [1] * (latents.ndim - 2)
            mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=latents.dtype).view(view_shape)
            std = torch.tensor(vae.config.latents_std, device=self.device, dtype=latents.dtype).view(view_shape)
            latents = latents * std + mean
        with torch.no_grad():
            decoded = vae.decode(latents).sample
        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1.0) * 0.5
        if decoded.ndim != 5:
            raise ValueError(f"Expected decoded video to be 5D, got {decoded.shape}")
        return decoded.permute(0, 2, 1, 3, 4).contiguous()

    def _normalize_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim == 3:
            frame_features = frame_features.unsqueeze(1)
        if frame_features.ndim != 4:
            raise ValueError(f"Unexpected frame feature shape: {frame_features.shape}")
        return frame_features
