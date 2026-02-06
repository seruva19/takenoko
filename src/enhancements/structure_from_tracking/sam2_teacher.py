"""SAM2 teacher feature extraction for Structure-From-Tracking distillation."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from safetensors.torch import save_file

from common.logger import get_logger
from memory.safetensors_loader import load_file

logger = get_logger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def build_causal_memory_features(
    frame_tokens: torch.Tensor,
    decay: float = 0.9,
) -> torch.Tensor:
    """Build causal memory-like features via EMA across time.

    Args:
        frame_tokens: [B, F, T, D] per-frame token features.
        decay: EMA decay in [0, 1).
    """
    if frame_tokens.dim() != 4:
        raise ValueError(
            "build_causal_memory_features expects [B, F, T, D], got "
            f"{tuple(frame_tokens.shape)}"
        )
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    frame_count = frame_tokens.shape[1]
    if frame_count <= 1:
        return frame_tokens

    memory = frame_tokens[:, 0]
    outputs = [memory]
    blend = 1.0 - decay
    for frame_idx in range(1, frame_count):
        current = frame_tokens[:, frame_idx]
        memory = decay * memory + blend * current
        outputs.append(memory)
    return torch.stack(outputs, dim=1)


def build_mask_aware_causal_memory_features(
    frame_tokens: torch.Tensor,
    frame_masks: torch.Tensor,
    decay: float = 0.9,
) -> torch.Tensor:
    """Build causal memory features with mask-aware update rates.

    Higher mask coverage reduces temporal inertia so foreground dynamics are
    integrated faster into recurrent memory.
    """
    if frame_tokens.dim() != 4:
        raise ValueError(
            "build_mask_aware_causal_memory_features expects [B, F, T, D], got "
            f"{tuple(frame_tokens.shape)}"
        )
    if frame_masks.dim() != 4:
        raise ValueError(
            "build_mask_aware_causal_memory_features expects masks [B, F, H, W], got "
            f"{tuple(frame_masks.shape)}"
        )
    if frame_tokens.shape[0] != frame_masks.shape[0] or frame_tokens.shape[1] != frame_masks.shape[1]:
        raise ValueError(
            "Token/mask batch-frame dimensions must match, got "
            f"{tuple(frame_tokens.shape[:2])} vs {tuple(frame_masks.shape[:2])}"
        )
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    frame_count = frame_tokens.shape[1]
    if frame_count <= 1:
        return frame_tokens

    mask_coverage = frame_masks.float().clamp(0.0, 1.0).mean(dim=(-1, -2), keepdim=True)
    memory = frame_tokens[:, 0]
    outputs = [memory]
    for frame_idx in range(1, frame_count):
        current = frame_tokens[:, frame_idx]
        coverage = mask_coverage[:, frame_idx]
        dynamic_decay = torch.clamp(
            decay + (1.0 - coverage) * (1.0 - decay) * 0.5,
            min=0.0,
            max=0.999,
        )
        blend = 1.0 - dynamic_decay
        memory = dynamic_decay * memory + blend * current
        outputs.append(memory)
    return torch.stack(outputs, dim=1)


def build_tracker_memory_features(
    frame_tokens: torch.Tensor,
    frame_masks: Optional[torch.Tensor] = None,
    decay: float = 0.9,
    attention_temperature: float = 0.07,
) -> torch.Tensor:
    """Build tracker-style recurrent memory features with token-attention gating."""
    if frame_tokens.dim() != 4:
        raise ValueError(
            "build_tracker_memory_features expects [B, F, T, D], got "
            f"{tuple(frame_tokens.shape)}"
        )
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    if attention_temperature <= 0:
        raise ValueError(
            "attention_temperature must be > 0, got "
            f"{attention_temperature}"
        )

    frame_count = frame_tokens.shape[1]
    if frame_count <= 1:
        return frame_tokens

    mask_coverage = None
    if frame_masks is not None:
        if frame_masks.dim() != 4:
            raise ValueError(
                "build_tracker_memory_features expects masks [B, F, H, W], got "
                f"{tuple(frame_masks.shape)}"
            )
        if frame_masks.shape[0] != frame_tokens.shape[0] or frame_masks.shape[1] != frame_tokens.shape[1]:
            raise ValueError(
                "Token/mask batch-frame dimensions must match, got "
                f"{tuple(frame_tokens.shape[:2])} vs {tuple(frame_masks.shape[:2])}"
            )
        mask_coverage = frame_masks.float().clamp(0.0, 1.0).mean(
            dim=(-1, -2),
            keepdim=True,
        )

    memory = frame_tokens[:, 0]
    outputs = [memory]
    base_blend = 1.0 - decay
    for frame_idx in range(1, frame_count):
        current = frame_tokens[:, frame_idx]
        sim = torch.nn.functional.cosine_similarity(
            current,
            memory,
            dim=-1,
            eps=1e-6,
        ).unsqueeze(-1)
        gate = torch.sigmoid(sim / attention_temperature)
        if mask_coverage is not None:
            frame_gate = 0.5 + 0.5 * mask_coverage[:, frame_idx]
            gate = gate * frame_gate
        blend = torch.clamp(base_blend * (0.5 + gate), min=0.0, max=1.0)
        memory = memory + blend * (current - memory)
        outputs.append(memory)
    return torch.stack(outputs, dim=1)


def apply_mask_prompt_to_frames(
    frames_bchw: torch.Tensor,
    masks_b1hw: torch.Tensor,
    prompt_strength: float = 0.85,
    blur_kernel: int = 5,
) -> torch.Tensor:
    """Apply soft foreground prompting to preprocessed teacher frames."""
    if frames_bchw.dim() != 4:
        raise ValueError(
            f"apply_mask_prompt_to_frames expects [B, C, H, W], got {tuple(frames_bchw.shape)}"
        )
    if masks_b1hw.dim() != 4 or masks_b1hw.shape[1] != 1:
        raise ValueError(
            f"apply_mask_prompt_to_frames expects mask [B, 1, H, W], got {tuple(masks_b1hw.shape)}"
        )
    if frames_bchw.shape[0] != masks_b1hw.shape[0]:
        raise ValueError("Frame/mask batch sizes must match")
    if not 0.0 <= prompt_strength <= 1.0:
        raise ValueError(f"prompt_strength must be in [0, 1], got {prompt_strength}")
    if blur_kernel <= 0 or blur_kernel % 2 == 0:
        raise ValueError(f"blur_kernel must be an odd integer > 0, got {blur_kernel}")

    mask = masks_b1hw.float().clamp(0.0, 1.0)
    if blur_kernel > 1:
        mask = nnf.avg_pool2d(mask, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    keep_scale = (1.0 - prompt_strength) + prompt_strength * mask
    return frames_bchw * keep_scale.to(dtype=frames_bchw.dtype)


def build_teacher_cache_path(
    cache_dir: str,
    item_key: str,
    *,
    image_size: int,
    max_frames: int,
    include_backward: bool,
    use_mask_prompting: bool,
    teacher_backend: str = "vision_encoder",
    teacher_type: str = "sam2",
) -> str:
    """Create deterministic teacher cache path for a dataset item."""
    base = os.path.splitext(os.path.basename(item_key))[0] or "sample"
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:80]
    signature = (
        f"{item_key}|img{image_size}|maxf{max_frames}|bwd{int(include_backward)}|"
        f"mask{int(use_mask_prompting)}|backend{teacher_backend}|type{teacher_type}"
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"{safe_base}_{digest}_sft_teacher.safetensors")


class SAM2Teacher(nn.Module):
    """Frozen SAM-family feature teacher used by Structure-From-Tracking helper."""

    def __init__(
        self,
        args: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.teacher_type = str(getattr(args, "sft_teacher_type", "sam2")).lower()
        self.teacher_model_id = str(
            getattr(args, "sft_teacher_model_id", "facebook/sam2-hiera-large")
        )
        self.teacher_checkpoint = str(getattr(args, "sft_teacher_checkpoint", "") or "")
        self.image_size = int(getattr(args, "sft_teacher_image_size", 512))
        self.chunk_size = int(getattr(args, "sft_teacher_chunk_size", 8))
        self.max_frames = int(getattr(args, "sft_teacher_max_frames", 0))
        self.detach_teacher = bool(getattr(args, "sft_detach_teacher", True))
        self.teacher_backend = str(
            getattr(args, "sft_teacher_backend", "vision_encoder")
        ).lower()
        self.tracker_memory_sam2_config = str(
            getattr(args, "sft_tracker_memory_sam2_config", "sam2_hiera_l.yaml")
        )
        self.tracker_memory_sam2_checkpoint = str(
            getattr(
                args,
                "sft_tracker_memory_sam2_checkpoint",
                "checkpoints/sam2_hiera_large.pt",
            )
        )
        self.tracker_memory_download_checkpoint = bool(
            getattr(args, "sft_tracker_memory_download_checkpoint", False)
        )
        self.tracker_memory_strict = bool(
            getattr(args, "sft_tracker_memory_strict", False)
        )
        self.use_causal_memory = bool(getattr(args, "sft_use_causal_memory", True))
        self.memory_decay = float(getattr(args, "sft_teacher_memory_decay", 0.9))
        self.use_mask_prompting = bool(getattr(args, "sft_use_mask_prompting", False))
        self.mask_prompt_strength = float(getattr(args, "sft_mask_prompt_strength", 0.85))
        self.mask_prompt_blur_kernel = int(getattr(args, "sft_mask_prompt_blur_kernel", 5))
        self.teacher_cache_mode = str(getattr(args, "sft_teacher_cache_mode", "off")).lower()
        self.teacher_cache_dir = str(getattr(args, "sft_teacher_cache_dir", "") or "")
        self._cache_warned_missing = False
        self.teacher_dtype = _resolve_dtype(
            str(getattr(args, "sft_teacher_dtype", "float16"))
        )
        self.paper_strict_mode = bool(getattr(args, "sft_paper_strict_mode", False))
        if self.teacher_cache_mode in {"write", "read_write"} and self.teacher_cache_dir:
            os.makedirs(self.teacher_cache_dir, exist_ok=True)
        self._tracker_model: Optional[nn.Module] = None
        self._tracker_backend_active = False
        self.model = self._load_teacher()
        if self.teacher_backend == "tracker_memory":
            self._tracker_backend_active = self._load_tracker_memory_backend()
        self.embed_dim = int(self._infer_embed_dim())
        logger.info(
            "Structure-From-Tracking: teacher type=%s backend=%s (active=%s).",
            self.teacher_type,
            self.teacher_backend,
            "tracker_memory" if self._tracker_backend_active else "vision_encoder",
        )

    def _load_teacher(self) -> nn.Module:
        if self.teacher_type == "sam2":
            return self._load_sam2_teacher()
        if self.teacher_type == "sam3":
            return self._load_sam3_teacher()
        raise ValueError(f"Unsupported SFT teacher type: {self.teacher_type}")

    def _load_sam2_teacher(self) -> nn.Module:
        try:
            from transformers import Sam2Model
        except Exception as exc:
            raise ImportError(
                "Structure-From-Tracking requires transformers Sam2Model support. "
                "Install compatible transformers + huggingface-hub versions."
            ) from exc

        source = self.teacher_checkpoint or self.teacher_model_id
        model = Sam2Model.from_pretrained(source)
        vision_encoder = getattr(model, "vision_encoder", None)
        if vision_encoder is None:
            raise ValueError("SAM2 model does not expose vision_encoder.")
        vision_encoder = vision_encoder.to(self.device)
        vision_encoder.eval()
        vision_encoder.requires_grad_(False)
        logger.info("Structure-From-Tracking: loaded SAM2 teacher from %s.", source)
        return vision_encoder

    def _load_sam3_teacher(self) -> nn.Module:
        try:
            import transformers
        except Exception as exc:
            raise ImportError(
                "Structure-From-Tracking SAM3 teacher requires transformers package."
            ) from exc

        model_cls = getattr(transformers, "Sam3Model", None)
        if model_cls is None:
            raise ImportError(
                "Structure-From-Tracking SAM3 teacher requires transformers.Sam3Model."
            )

        source = self.teacher_checkpoint or self.teacher_model_id
        model = model_cls.from_pretrained(source)
        vision_encoder = getattr(model, "vision_encoder", None)
        if vision_encoder is None:
            vision_encoder = getattr(model, "image_encoder", None)
        if vision_encoder is None:
            logger.warning(
                "Structure-From-Tracking: SAM3 model does not expose vision/image encoder; "
                "using full model forward for token extraction."
            )
            vision_encoder = model
        vision_encoder = vision_encoder.to(self.device)
        vision_encoder.eval()
        vision_encoder.requires_grad_(False)
        logger.info("Structure-From-Tracking: loaded SAM3 teacher from %s.", source)
        return vision_encoder

    def _maybe_download_tracker_checkpoint(self) -> None:
        if os.path.exists(self.tracker_memory_sam2_checkpoint):
            return
        if not self.tracker_memory_download_checkpoint:
            return
        try:
            from huggingface_hub import hf_hub_download

            os.makedirs(
                os.path.dirname(self.tracker_memory_sam2_checkpoint) or ".",
                exist_ok=True,
            )
            hf_hub_download(
                "SkalskiP/florence-sam-masking",
                repo_type="space",
                subfolder="checkpoints",
                local_dir="./",
                filename=os.path.basename(self.tracker_memory_sam2_checkpoint),
            )
            logger.info(
                "Structure-From-Tracking: downloaded tracker-memory SAM2 checkpoint to %s.",
                self.tracker_memory_sam2_checkpoint,
            )
        except Exception as exc:
            logger.warning(
                "Structure-From-Tracking: failed to download tracker-memory checkpoint (%s).",
                exc,
            )

    def _load_tracker_memory_backend(self) -> bool:
        """Load optional SAM2 tracker-memory backend. Falls back unless strict."""
        if self.teacher_backend != "tracker_memory":
            return False
        if self.teacher_type != "sam2":
            raise ValueError(
                "tracker_memory backend currently supports only sft_teacher_type='sam2'"
            )
        try:
            self._maybe_download_tracker_checkpoint()
            if not os.path.exists(self.tracker_memory_sam2_checkpoint):
                raise FileNotFoundError(
                    "Tracker-memory SAM2 checkpoint not found: "
                    f"{self.tracker_memory_sam2_checkpoint}"
                )
            from sam2.build_sam import build_sam2

            model = build_sam2(
                self.tracker_memory_sam2_config,
                self.tracker_memory_sam2_checkpoint,
                device=self.device,
            )
            model = model.to(self.device)
            model.eval()
            model.requires_grad_(False)
            self._tracker_model = model
            logger.info(
                "Structure-From-Tracking: tracker-memory backend enabled (%s).",
                self.tracker_memory_sam2_checkpoint,
            )
            return True
        except Exception as exc:
            if self.tracker_memory_strict:
                raise RuntimeError(
                    "Structure-From-Tracking: tracker-memory backend initialization failed "
                    f"with strict mode enabled: {exc}"
                ) from exc
            if self.paper_strict_mode:
                raise RuntimeError(
                    "Structure-From-Tracking strict mode: tracker-memory backend initialization failed."
                ) from exc
            logger.warning(
                "Structure-From-Tracking: tracker-memory backend unavailable (%s). "
                "Falling back to vision_encoder backend.",
                exc,
            )
            return False

    def _infer_embed_dim(self) -> int:
        if self._tracker_backend_active and self._tracker_model is not None:
            config = getattr(self._tracker_model, "config", None)
            if config is not None and hasattr(config, "hidden_size"):
                return int(config.hidden_size)
        config = getattr(self.model, "config", None)
        if config is not None:
            backbone_config = getattr(config, "backbone_config", None)
            if backbone_config is not None:
                embed_dims = getattr(backbone_config, "embed_dim_per_stage", None)
                if isinstance(embed_dims, (list, tuple)) and len(embed_dims) > 0:
                    return int(embed_dims[-1])
            if hasattr(config, "hidden_size"):
                return int(config.hidden_size)

        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            output = self.model(dummy)
        tokens = self._coerce_tokens(output)
        if tokens.dim() != 3:
            raise ValueError(
                f"Unable to infer {self.teacher_type} teacher embed dimension."
            )
        return int(tokens.shape[-1])

    @staticmethod
    def _coerce_tokens(output: Any) -> torch.Tensor:
        hidden: Any = output
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state
        if isinstance(hidden, (list, tuple)):
            for value in hidden:
                if isinstance(value, torch.Tensor):
                    hidden = value
                    break
        if isinstance(hidden, dict):
            for key in (
                "last_hidden_state",
                "hidden_states",
                "x_norm_patchtokens",
                "vision_features",
                "image_embeddings",
            ):
                value = hidden.get(key, None)
                if isinstance(value, torch.Tensor):
                    hidden = value
                    break
                if isinstance(value, (list, tuple)):
                    for v in value:
                        if isinstance(v, torch.Tensor):
                            hidden = v
                            break
                    if isinstance(hidden, torch.Tensor):
                        break
        if not isinstance(hidden, torch.Tensor):
            raise ValueError(f"Expected tensor teacher output, got {type(hidden)}")

        if hidden.dim() == 3:
            return hidden
        if hidden.dim() != 4:
            raise ValueError(
                f"Unsupported SAM2 feature shape {tuple(hidden.shape)}; expected 3D or 4D tensor."
            )

        # Choose the most plausible layout between BCHW and BHWC.
        bsz, d1, d2, d3 = hidden.shape
        score_bchw = (d2 * d3) - d1
        score_bhwc = (d1 * d2) - d3
        if score_bhwc > score_bchw:
            # BHWC -> [B, H*W, C]
            return hidden.view(bsz, d1 * d2, d3)
        # BCHW -> [B, H*W, C]
        return hidden.view(bsz, d1, d2 * d3).transpose(1, 2)

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        frames = (frames + 1.0) * 0.5
        frames = frames.clamp(0.0, 1.0)
        frames = nnf.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(_IMAGENET_MEAN, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std

    @staticmethod
    def _prepare_mask_hints(
        mask_hints: Optional[torch.Tensor],
        batch_size: int,
        frame_count: int,
        target_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Normalize mask hints into [B, F, H, W] with requested temporal/spatial shape."""
        if mask_hints is None:
            return None

        mask = mask_hints
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        if mask.dim() == 4:
            # Ambiguous [B, F, H, W] or [B, C, H, W]; treat channel-1 as single-frame mask.
            if mask.shape[1] == 1:
                mask = mask.expand(-1, frame_count, -1, -1)
        elif mask.dim() == 5:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                raise ValueError(
                    "Unsupported mask_hints layout for 5D tensor; expected [B,1,F,H,W] or [B,F,1,H,W], got "
                    f"{tuple(mask.shape)}"
                )
        else:
            raise ValueError(
                "Unsupported mask_hints dims; expected 3D/4D/5D, got "
                f"{tuple(mask.shape)}"
            )

        if mask.dim() != 4:
            raise ValueError(f"Normalized mask_hints must be 4D [B,F,H,W], got {tuple(mask.shape)}")
        if mask.shape[0] != batch_size:
            raise ValueError(
                "mask_hints batch size mismatch: "
                f"expected {batch_size}, got {mask.shape[0]}"
            )

        mask = mask.to(device=device, dtype=torch.float32)
        if mask.shape[1] != frame_count or mask.shape[2] != target_size or mask.shape[3] != target_size:
            mask = nnf.interpolate(
                mask.unsqueeze(1),
                size=(frame_count, target_size, target_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(1)
        return mask.clamp(0.0, 1.0)

    def _extract_tokens(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        use_amp = self.device.type == "cuda" and self.teacher_dtype != torch.float32
        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.teacher_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            with amp_context:
                output = self.model(frames_bchw)
                tokens = self._coerce_tokens(output)
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    def _extract_tokens_tracker_backend(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        """Extract tokens using tracker-memory backend when available."""
        if self._tracker_model is None:
            return self._extract_tokens(frames_bchw)

        use_amp = self.device.type == "cuda" and self.teacher_dtype != torch.float32
        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.teacher_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        last_error: Optional[Exception] = None
        output = None
        for method_name in ("forward_image", "image_encoder", "forward"):
            method = getattr(self._tracker_model, method_name, None)
            if not callable(method):
                continue
            try:
                with torch.no_grad():
                    with amp_context:
                        output = method(frames_bchw)
                break
            except TypeError:
                continue
            except Exception as exc:
                last_error = exc
                continue

        if output is None:
            if self.tracker_memory_strict:
                raise RuntimeError(
                    "Structure-From-Tracking: tracker-memory backend failed to produce features."
                ) from last_error
            if self.paper_strict_mode:
                raise RuntimeError(
                    "Structure-From-Tracking strict mode: tracker-memory backend failed to produce features."
                ) from last_error
            if last_error is not None:
                logger.warning(
                    "Structure-From-Tracking: tracker-memory token extraction failed (%s). "
                    "Using vision-encoder tokens instead.",
                    last_error,
                )
            return self._extract_tokens(frames_bchw)

        tokens = self._coerce_tokens(output)
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    def extract_features(
        self,
        clean_pixels: torch.Tensor,
        mask_hints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract per-frame token features with shape [B, F, T, D]."""
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        if clean_pixels.dim() != 5:
            raise ValueError(
                "SAM2 teacher expects clean_pixels with shape [B, C, F, H, W], "
                f"got {tuple(clean_pixels.shape)}"
            )

        batch_size, channels, frame_count, _, _ = clean_pixels.shape
        if channels != 3:
            raise ValueError(
                f"SAM2 teacher expects RGB inputs (C=3), got C={channels}"
            )

        source_frame_count = frame_count
        selected_indices = None
        if self.max_frames > 0 and frame_count > self.max_frames:
            indices = (
                torch.linspace(0, frame_count - 1, self.max_frames)
                .long()
                .to(clean_pixels.device)
            )
            clean_pixels = clean_pixels.index_select(2, indices)
            frame_count = clean_pixels.shape[2]
            selected_indices = indices

        frames = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frame_count,
            channels,
            clean_pixels.shape[3],
            clean_pixels.shape[4],
        )
        frames = frames.to(device=self.device, dtype=torch.float32)
        frames = self._preprocess(frames)

        prepared_masks = self._prepare_mask_hints(
            mask_hints=mask_hints,
            batch_size=batch_size,
            frame_count=source_frame_count,
            target_size=self.image_size,
            device=frames.device,
        )
        if prepared_masks is not None and selected_indices is not None:
            prepared_masks = prepared_masks.index_select(1, selected_indices.to(prepared_masks.device))
        if prepared_masks is not None and self.use_mask_prompting:
            flat_masks = prepared_masks.reshape(batch_size * frame_count, 1, self.image_size, self.image_size)
            frames = apply_mask_prompt_to_frames(
                frames_bchw=frames,
                masks_b1hw=flat_masks,
                prompt_strength=self.mask_prompt_strength,
                blur_kernel=self.mask_prompt_blur_kernel,
            )

        chunk_size = max(1, self.chunk_size)
        token_chunks = []
        for start in range(0, frames.shape[0], chunk_size):
            end = min(start + chunk_size, frames.shape[0])
            if self._tracker_backend_active:
                token_chunks.append(
                    self._extract_tokens_tracker_backend(frames[start:end])
                )
            else:
                token_chunks.append(self._extract_tokens(frames[start:end]))
        tokens = torch.cat(token_chunks, dim=0)
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected token tensor [BF, T, D], got {tuple(tokens.shape)}"
            )
        tokens = tokens.view(batch_size, frame_count, tokens.shape[1], tokens.shape[2])
        if self._tracker_backend_active:
            if self.use_causal_memory:
                tokens = build_tracker_memory_features(
                    frame_tokens=tokens,
                    frame_masks=prepared_masks if self.use_mask_prompting else None,
                    decay=self.memory_decay,
                )
        elif self.use_causal_memory:
            if prepared_masks is not None and self.use_mask_prompting:
                tokens = build_mask_aware_causal_memory_features(
                    frame_tokens=tokens,
                    frame_masks=prepared_masks,
                    decay=self.memory_decay,
                )
            else:
                tokens = build_causal_memory_features(tokens, decay=self.memory_decay)
        return tokens

    def extract_bidirectional_features(
        self,
        clean_pixels: torch.Tensor,
        include_backward: bool = True,
        item_keys: Optional[Sequence[str]] = None,
        mask_hints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return forward tokens and optional backward tokens remapped to original order."""
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)

        cache_enabled = (
            self.teacher_cache_mode != "off"
            and bool(self.teacher_cache_dir)
            and item_keys is not None
            and len(item_keys) == clean_pixels.shape[0]
        )
        if not cache_enabled:
            forward_tokens = self.extract_features(clean_pixels, mask_hints=mask_hints)
            if not include_backward or forward_tokens.shape[1] <= 1:
                return forward_tokens, None

            reversed_pixels = clean_pixels.flip(2)
            backward_masks = mask_hints.flip(1) if mask_hints is not None and mask_hints.dim() >= 2 else None
            backward_tokens = self.extract_features(reversed_pixels, mask_hints=backward_masks).flip(1)
            return forward_tokens, backward_tokens

        forward_list = []
        backward_list = []
        all_have_backward = True
        for idx, item_key in enumerate(item_keys):
            sample_pixels = clean_pixels[idx : idx + 1]
            sample_mask = mask_hints[idx : idx + 1] if mask_hints is not None else None
            cache_path = build_teacher_cache_path(
                cache_dir=self.teacher_cache_dir,
                item_key=str(item_key),
                image_size=self.image_size,
                max_frames=self.max_frames,
                include_backward=include_backward,
                use_mask_prompting=self.use_mask_prompting,
                teacher_backend=(
                    "tracker_memory"
                    if self._tracker_backend_active
                    else "vision_encoder"
                ),
                teacher_type=self.teacher_type,
            )
            can_read = self.teacher_cache_mode in {"read", "read_write"}
            can_write = self.teacher_cache_mode in {"write", "read_write"}

            cached_forward = None
            cached_backward = None
            if can_read and os.path.exists(cache_path):
                try:
                    sd = load_file(cache_path)
                    cached_forward = sd.get("sft_forward", None)
                    cached_backward = sd.get("sft_backward", None)
                    if isinstance(cached_forward, torch.Tensor):
                        if cached_forward.dim() == 3:
                            cached_forward = cached_forward.unsqueeze(0)
                        cached_forward = cached_forward.to(device=sample_pixels.device)
                    else:
                        cached_forward = None
                    if isinstance(cached_backward, torch.Tensor):
                        if cached_backward.dim() == 3:
                            cached_backward = cached_backward.unsqueeze(0)
                        cached_backward = cached_backward.to(device=sample_pixels.device)
                    else:
                        cached_backward = None
                except Exception as exc:
                    logger.warning(
                        "Structure-From-Tracking: failed to load teacher cache %s: %s",
                        cache_path,
                        exc,
                    )
                    cached_forward = None
                    cached_backward = None

            has_cached_pair = (
                cached_forward is not None
                and (not include_backward or cached_backward is not None)
            )
            if has_cached_pair:
                forward_tokens = cached_forward
                backward_tokens = cached_backward
            else:
                if self.paper_strict_mode and self.teacher_cache_mode == "read":
                    raise FileNotFoundError(
                        "Structure-From-Tracking strict mode: teacher cache miss in read mode "
                        f"for item_key={item_key!r} at {cache_path}"
                    )
                if self.teacher_cache_mode == "read" and not self._cache_warned_missing:
                    logger.warning(
                        "Structure-From-Tracking: cache miss in read mode; falling back to online teacher extraction."
                    )
                    self._cache_warned_missing = True
                forward_tokens = self.extract_features(sample_pixels, mask_hints=sample_mask)
                backward_tokens = None
                if include_backward and forward_tokens.shape[1] > 1:
                    reversed_pixels = sample_pixels.flip(2)
                    backward_masks = sample_mask.flip(1) if sample_mask is not None else None
                    backward_tokens = self.extract_features(
                        reversed_pixels,
                        mask_hints=backward_masks,
                    ).flip(1)
                if can_write:
                    try:
                        payload = {"sft_forward": forward_tokens[0].detach().cpu()}
                        if backward_tokens is not None:
                            payload["sft_backward"] = backward_tokens[0].detach().cpu()
                        save_file(payload, cache_path)
                    except Exception as exc:
                        logger.warning(
                            "Structure-From-Tracking: failed to save teacher cache %s: %s",
                            cache_path,
                            exc,
                        )

            forward_list.append(forward_tokens)
            if include_backward and forward_tokens.shape[1] > 1:
                if backward_tokens is None:
                    all_have_backward = False
                else:
                    backward_list.append(backward_tokens)

        forward_batch = torch.cat(forward_list, dim=0)
        if not include_backward or forward_batch.shape[1] <= 1 or not all_have_backward:
            return forward_batch, None
        backward_batch = torch.cat(backward_list, dim=0)
        return forward_batch, backward_batch
