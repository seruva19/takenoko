"""M2-REPA inspired multi-expert representation alignment helper."""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger
from enhancements.repa.encoder_manager import EncoderManager, preprocess_raw_image

logger = get_logger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEPTH_ANYTHING_V2_MODEL_IDS = {
    "s": "depth-anything/Depth-Anything-V2-Small-hf",
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "b": "depth-anything/Depth-Anything-V2-Base-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "l": "depth-anything/Depth-Anything-V2-Large-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _interpolate_token_count(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate token features to match a target token count."""
    if tokens.shape[1] == target_tokens:
        return tokens

    batch_frames, src_tokens, dim = tokens.shape
    src_side = int(math.isqrt(src_tokens))
    tgt_side = int(math.isqrt(target_tokens))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_tokens:
        x = tokens.permute(0, 2, 1).reshape(batch_frames, dim, src_side, src_side)
        x = F.interpolate(
            x,
            size=(tgt_side, tgt_side),
            mode="bilinear",
            align_corners=False,
        )
        return x.reshape(batch_frames, dim, target_tokens).permute(0, 2, 1)

    x = tokens.permute(0, 2, 1)
    x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


def _make_projector(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    layers: int,
) -> nn.Sequential:
    modules: List[nn.Module] = []
    current_dim = int(in_dim)
    for _ in range(max(0, int(layers) - 1)):
        modules.append(nn.Linear(current_dim, int(hidden_dim)))
        modules.append(nn.SiLU())
        current_dim = int(hidden_dim)
    modules.append(nn.Linear(current_dim, int(out_dim)))
    return nn.Sequential(*modules)


def _resolve_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _is_depth_anything_v2_spec(spec: str) -> bool:
    value = str(spec).strip().lower()
    return (
        value == "depth-anything-v2"
        or value == "depthanythingv2"
        or value == "depth_anything_v2"
        or value.startswith("depth-anything-v2:")
        or value.startswith("depthanythingv2:")
        or value.startswith("depth_anything_v2:")
        or value.startswith("depth-anything-v2-")
        or value.startswith("depthanythingv2-")
        or value.startswith("depth_anything_v2-")
    )


def _parse_depth_anything_v2_spec(spec: str, default_model_id: str) -> str:
    raw = str(spec).strip()
    value = raw.lower()
    for prefix in ("depth-anything-v2", "depthanythingv2", "depth_anything_v2"):
        if value == prefix:
            return default_model_id
        if value.startswith(prefix + ":"):
            model_id = raw[len(prefix) + 1 :].strip()
            if not model_id:
                raise ValueError(
                    f"M2-REPA Depth Anything V2 expert spec {spec!r} is missing a model ID"
                )
            return model_id
        if value.startswith(prefix + "-"):
            scale = value[len(prefix) + 1 :].strip().replace("_", "-")
            if scale.endswith("-hf"):
                scale = scale[:-3]
            model_id = _DEPTH_ANYTHING_V2_MODEL_IDS.get(scale)
            if model_id is None:
                raise ValueError(
                    "M2-REPA Depth Anything V2 scale must be one of "
                    f"{sorted(_DEPTH_ANYTHING_V2_MODEL_IDS)}, got {scale!r}"
                )
            return model_id
    raise ValueError(f"M2-REPA: unsupported Depth Anything V2 expert spec {spec!r}")


class _DepthAnythingV2Expert(nn.Module):
    """Frozen Depth Anything V2 feature adapter for M2-REPA."""

    def __init__(
        self,
        args: Any,
        spec: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.args = args
        self.device = device
        default_model_id = str(
            getattr(
                args,
                "m2_repa_depth_model_id",
                "depth-anything/Depth-Anything-V2-Base-hf",
            )
        )
        self.model_id = _parse_depth_anything_v2_spec(spec, default_model_id)
        self.checkpoint = str(getattr(args, "m2_repa_depth_checkpoint", "") or "")
        self.source = self.checkpoint or self.model_id
        self.image_size = int(getattr(args, "m2_repa_depth_image_size", 518))
        self.chunk_size = int(getattr(args, "m2_repa_encoder_chunk_size", 0))
        if self.chunk_size <= 0:
            self.chunk_size = 8
        self.feature_source = str(
            getattr(args, "m2_repa_depth_feature_source", "auto")
        ).lower()
        self.teacher_dtype = _resolve_dtype(
            str(getattr(args, "m2_repa_depth_teacher_dtype", "float16"))
        )
        self.detach_teacher = bool(getattr(args, "m2_repa_detach_teacher", True))

        try:
            from transformers import AutoModelForDepthEstimation
        except Exception as exc:
            raise ImportError(
                "M2-REPA Depth Anything V2 experts require transformers "
                "AutoModelForDepthEstimation support."
            ) from exc

        self.model = AutoModelForDepthEstimation.from_pretrained(self.source)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.embed_dim = int(self._infer_embed_dim())
        logger.info(
            "M2-REPA: loaded Depth Anything V2 expert from %s (feature_source=%s, embed_dim=%d).",
            self.source,
            self.feature_source,
            self.embed_dim,
        )

    @staticmethod
    def _maybe_strip_cls(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            return tokens
        token_count = tokens.shape[1]
        is_square = int(math.isqrt(token_count)) ** 2 == token_count
        is_square_minus_one = int(math.isqrt(token_count - 1)) ** 2 == (token_count - 1)
        if is_square_minus_one and not is_square:
            return tokens[:, 1:, :]
        return tokens

    @staticmethod
    def _tensor_to_tokens(value: torch.Tensor) -> Optional[torch.Tensor]:
        if value.dim() == 2:
            return value.unsqueeze(1)
        if value.dim() == 3:
            return _DepthAnythingV2Expert._maybe_strip_cls(value)
        if value.dim() == 4:
            bsz, channels, height, width = value.shape
            return value.view(bsz, channels, height * width).transpose(1, 2)
        return None

    @staticmethod
    def _iter_tensor_candidates(value: Any) -> List[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            tensors: List[torch.Tensor] = []
            for item in value:
                tensors.extend(_DepthAnythingV2Expert._iter_tensor_candidates(item))
            return tensors
        if isinstance(value, dict):
            tensors = []
            for item in value.values():
                tensors.extend(_DepthAnythingV2Expert._iter_tensor_candidates(item))
            return tensors
        return []

    def _extract_hidden_tokens(self, output: Any) -> Optional[torch.Tensor]:
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None and isinstance(output, dict):
            hidden_states = output.get("hidden_states")
        if hidden_states is not None:
            for hidden in reversed(self._iter_tensor_candidates(hidden_states)):
                tokens = self._tensor_to_tokens(hidden)
                if tokens is not None:
                    return tokens

        for attr in ("last_hidden_state", "vision_hidden_states", "backbone_hidden_states"):
            value = getattr(output, attr, None)
            if value is None and isinstance(output, dict):
                value = output.get(attr)
            if value is None:
                continue
            for hidden in reversed(self._iter_tensor_candidates(value)):
                tokens = self._tensor_to_tokens(hidden)
                if tokens is not None:
                    return tokens
        return None

    def _extract_depth_tokens_from_output(self, output: Any) -> torch.Tensor:
        depth = getattr(output, "predicted_depth", None)
        if depth is None and isinstance(output, dict):
            depth = output.get("predicted_depth")
        if not isinstance(depth, torch.Tensor):
            raise ValueError("M2-REPA Depth Anything V2 output did not contain predicted_depth")
        if depth.dim() == 3:
            return depth.flatten(1).unsqueeze(-1)
        if depth.dim() == 4 and depth.shape[1] == 1:
            return depth[:, 0].flatten(1).unsqueeze(-1)
        if depth.dim() == 4:
            bsz, channels, height, width = depth.shape
            return depth.view(bsz, channels, height * width).transpose(1, 2)
        raise ValueError(
            f"M2-REPA Depth Anything V2 predicted_depth has unsupported shape {tuple(depth.shape)}"
        )

    def _coerce_output_tokens(self, output: Any) -> torch.Tensor:
        if self.feature_source in {"auto", "hidden_states"}:
            hidden_tokens = self._extract_hidden_tokens(output)
            if hidden_tokens is not None:
                return hidden_tokens
            if self.feature_source == "hidden_states":
                raise ValueError(
                    "M2-REPA Depth Anything V2 feature_source='hidden_states' "
                    "but the model output did not include hidden states"
                )
        return self._extract_depth_tokens_from_output(output)

    def _infer_embed_dim(self) -> int:
        if self.feature_source == "predicted_depth":
            return 1

        config = getattr(self.model, "config", None)
        candidates = []
        if config is not None:
            candidates.extend(
                [
                    getattr(config, "hidden_size", None),
                    getattr(config, "neck_hidden_sizes", None),
                ]
            )
            backbone_config = getattr(config, "backbone_config", None)
            if backbone_config is not None:
                candidates.extend(
                    [
                        getattr(backbone_config, "hidden_size", None),
                        getattr(backbone_config, "hidden_sizes", None),
                        getattr(backbone_config, "embed_dim", None),
                        getattr(backbone_config, "embed_dims", None),
                    ]
                )
        for candidate in candidates:
            if isinstance(candidate, int) and candidate > 0:
                return candidate
            if isinstance(candidate, (list, tuple)) and candidate:
                last = candidate[-1]
                if isinstance(last, int) and last > 0:
                    return last

        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            output = self.model(
                pixel_values=dummy,
                output_hidden_states=True,
                return_dict=True,
            )
        return int(self._coerce_output_tokens(output).shape[-1])

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        frames = (frames + 1.0) * 0.5
        frames = frames.clamp(0.0, 1.0)
        frames = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(_IMAGENET_MEAN, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std

    def extract_features(self, clean_pixels: torch.Tensor) -> torch.Tensor:
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        if clean_pixels.dim() != 5:
            raise ValueError(
                "Depth Anything V2 expert expects clean_pixels [B, C, F, H, W], "
                f"got {tuple(clean_pixels.shape)}"
            )
        batch_size, channels, frame_count, height, width = clean_pixels.shape
        if channels != 3:
            raise ValueError(f"Depth Anything V2 expert expects RGB inputs, got C={channels}")

        frames = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frame_count,
            channels,
            height,
            width,
        )
        frames = self._preprocess(frames.to(device=self.device, dtype=torch.float32))
        chunks = list(torch.split(frames, self.chunk_size, dim=0))
        use_amp = self.device.type == "cuda" and self.teacher_dtype != torch.float32
        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.teacher_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        token_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            with amp_context:
                for chunk in chunks:
                    output = self.model(
                        pixel_values=chunk,
                        output_hidden_states=self.feature_source != "predicted_depth",
                        return_dict=True,
                    )
                    token_chunks.append(self._coerce_output_tokens(output))
        tokens = torch.cat(token_chunks, dim=0)
        tokens = tokens.view(batch_size, frame_count, tokens.shape[1], tokens.shape[2])
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens


class M2RepaHelper(nn.Module):
    """Train-time multi-expert REPA with modality-style decoupling heads.

    The helper follows the paper's useful transfer pattern for Takenoko: multiple
    frozen expert encoders provide target representations, while separate MLP
    heads map a shared diffusion hidden state into expert-specific subspaces.
    A pairwise linear CKA penalty discourages those subspaces from collapsing
    into redundant features. It does not change inference behavior.
    """

    def __init__(self, diffusion_model: Any, args: Any) -> None:
        super().__init__()
        self.args = args
        self.diffusion_model = diffusion_model
        self.hook_handles: List[Any] = []
        self._shape_warning_logged = False
        self.last_m2_repa_metrics: dict[str, torch.Tensor] = {}

        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = getattr(args, "model_cache_dir", "models")
        resolution = int(getattr(args, "m2_repa_input_resolution", 256))
        self.align_lambda = float(getattr(args, "m2_repa_align_lambda", 0.5))
        self.decouple_lambda = float(getattr(args, "m2_repa_decouple_lambda", 0.05))
        self.max_spatial_tokens = int(getattr(args, "m2_repa_max_spatial_tokens", -1))
        self.decouple_max_samples = int(
            getattr(args, "m2_repa_decouple_max_samples", 4096)
        )
        self.spatial_align = bool(getattr(args, "m2_repa_spatial_align", True))
        self.temporal_align = bool(getattr(args, "m2_repa_temporal_align", True))
        self.detach_teacher = bool(getattr(args, "m2_repa_detach_teacher", True))
        self.encoder_chunk_size = int(getattr(args, "m2_repa_encoder_chunk_size", 0))
        encoder_names = getattr(args, "m2_repa_encoder_name_list", None)
        if not encoder_names:
            encoder_spec = str(getattr(args, "m2_repa_encoder_names", ""))
            encoder_names = [
                name.strip() for name in encoder_spec.split(",") if name.strip()
            ]
        else:
            encoder_names = [
                str(name).strip() for name in encoder_names if str(name).strip()
            ]

        (
            self.encoders,
            self.encoder_types,
            self.encoder_architectures,
            self.expert_kinds,
        ) = self._load_experts(
            encoder_names=encoder_names,
            device=device,
            cache_dir=cache_dir,
            resolution=resolution,
        )
        if len(self.encoders) < 2:
            raise ValueError("M2-REPA requires at least two loaded expert encoders")
        for encoder in self.encoders:
            encoder.eval()
            encoder.requires_grad_(False)

        raw_depths = getattr(args, "m2_repa_alignment_depths", None)
        if isinstance(raw_depths, (list, tuple)) and len(raw_depths) > 0:
            self.alignment_depths = [int(depth) for depth in raw_depths]
        else:
            self.alignment_depths = [int(getattr(args, "m2_repa_alignment_depth", 8))]
        self.captured_features: List[Optional[torch.Tensor]] = [None] * len(
            self.alignment_depths
        )

        self.hidden_dim = self._infer_diffusion_hidden_dim()
        self.projector_hidden_dim = int(
            getattr(args, "m2_repa_projector_hidden_dim", 2048)
        )
        self.projector_layers = int(getattr(args, "m2_repa_projector_layers", 3))
        self.expert_dims = [int(getattr(encoder, "embed_dim", 768)) for encoder in self.encoders]
        self.projectors = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _make_projector(
                            in_dim=self.hidden_dim,
                            hidden_dim=self.projector_hidden_dim,
                            out_dim=expert_dim,
                            layers=self.projector_layers,
                        )
                        for expert_dim in self.expert_dims
                    ]
                )
                for _ in self.alignment_depths
            ]
        )

        logger.info(
            "M2-REPA: initialized with %d experts, hidden_dim=%d, expert_dims=%s, depths=%s.",
            len(self.encoders),
            self.hidden_dim,
            self.expert_dims,
            self.alignment_depths,
        )

    @staticmethod
    def _is_sam_expert_spec(spec: str) -> bool:
        value = str(spec).strip().lower()
        return value == "sam2" or value.startswith("sam2:") or value.startswith("sam3:")

    @staticmethod
    def _is_depth_expert_spec(spec: str) -> bool:
        return _is_depth_anything_v2_spec(spec)

    @staticmethod
    def _parse_sam_expert_spec(spec: str) -> Tuple[str, Optional[str]]:
        raw = str(spec).strip()
        if ":" not in raw:
            return raw.lower(), None
        teacher_type, model_id = raw.split(":", 1)
        model_id = model_id.strip()
        return teacher_type.strip().lower(), model_id or None

    def _load_sam_expert(
        self,
        spec: str,
        device: Any,
    ) -> Tuple[nn.Module, str]:
        teacher_type, inline_model_id = self._parse_sam_expert_spec(spec)
        if teacher_type not in {"sam2", "sam3"}:
            raise ValueError(f"M2-REPA: unsupported SAM-family expert spec {spec!r}")
        if teacher_type == "sam3" and not inline_model_id:
            raise ValueError(
                "M2-REPA SAM3 experts must use an explicit spec such as "
                "'sam3:<huggingface-model-id>'"
            )

        from enhancements.structure_from_tracking.sam2_teacher import SAM2Teacher

        chunk_size = int(getattr(self.args, "m2_repa_encoder_chunk_size", 0))
        if chunk_size <= 0:
            chunk_size = 8
        teacher_args = SimpleNamespace(
            sft_teacher_type=teacher_type,
            sft_teacher_model_id=inline_model_id
            or str(getattr(self.args, "m2_repa_sam_model_id", "facebook/sam2-hiera-large")),
            sft_teacher_checkpoint=str(
                getattr(self.args, "m2_repa_sam_checkpoint", "") or ""
            ),
            sft_teacher_image_size=int(
                getattr(self.args, "m2_repa_sam_image_size", 512)
            ),
            sft_teacher_chunk_size=chunk_size,
            sft_teacher_max_frames=0,
            sft_detach_teacher=bool(
                getattr(self.args, "m2_repa_detach_teacher", True)
            ),
            sft_teacher_backend="vision_encoder",
            sft_tracker_memory_strict=False,
            sft_use_causal_memory=bool(
                getattr(self.args, "m2_repa_sam_use_causal_memory", True)
            ),
            sft_teacher_memory_decay=0.9,
            sft_use_mask_prompting=False,
            sft_teacher_cache_mode="off",
            sft_teacher_cache_dir="",
            sft_teacher_dtype=str(
                getattr(self.args, "m2_repa_sam_teacher_dtype", "float16")
            ),
            sft_paper_strict_mode=False,
        )
        teacher = SAM2Teacher(teacher_args, torch.device(device))
        logger.info(
            "M2-REPA: loaded %s expert from %s.",
            teacher_type.upper(),
            teacher_args.sft_teacher_checkpoint or teacher_args.sft_teacher_model_id,
        )
        return teacher, teacher_type

    def _load_depth_expert(
        self,
        spec: str,
        device: Any,
    ) -> Tuple[nn.Module, str]:
        expert = _DepthAnythingV2Expert(
            args=self.args,
            spec=spec,
            device=torch.device(device),
        )
        return expert, "depth_anything_v2"

    def _load_experts(
        self,
        encoder_names: List[str],
        device: Any,
        cache_dir: str,
        resolution: int,
    ) -> Tuple[List[nn.Module], List[str], List[str], List[str]]:
        encoders: List[nn.Module] = []
        encoder_types: List[str] = []
        architectures: List[str] = []
        expert_kinds: List[str] = []
        manager = EncoderManager(device=device, cache_dir=cache_dir)

        for spec in encoder_names:
            if self._is_sam_expert_spec(spec):
                teacher, teacher_type = self._load_sam_expert(spec, device=device)
                encoders.append(teacher)
                encoder_types.append(teacher_type)
                architectures.append("vision")
                expert_kinds.append("sam")
                continue
            if self._is_depth_expert_spec(spec):
                expert, expert_type = self._load_depth_expert(spec, device=device)
                encoders.append(expert)
                encoder_types.append(expert_type)
                architectures.append("depth")
                expert_kinds.append("depth")
                continue

            loaded_encoders, loaded_types, loaded_architectures = manager.load_encoders(
                spec,
                resolution=resolution,
            )
            encoders.extend(loaded_encoders)
            encoder_types.extend(loaded_types)
            architectures.extend(loaded_architectures)
            expert_kinds.extend(["encoder_manager"] * len(loaded_encoders))

        return encoders, encoder_types, architectures, expert_kinds

    def _infer_diffusion_hidden_dim(self) -> int:
        if hasattr(self.diffusion_model, "dim"):
            return int(self.diffusion_model.dim)
        if hasattr(self.diffusion_model, "hidden_size"):
            return int(self.diffusion_model.hidden_size)
        for module in self.diffusion_model.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("M2-REPA: falling back to hidden_dim=1024")
        return 1024

    def _locate_blocks(self) -> Any:
        if hasattr(self.diffusion_model, "blocks"):
            return self.diffusion_model.blocks
        if hasattr(self.diffusion_model, "layers"):
            return self.diffusion_model.layers
        if hasattr(self.diffusion_model, "transformer_blocks"):
            return self.diffusion_model.transformer_blocks
        raise ValueError("M2-REPA: could not locate transformer block list")

    def _get_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            features = output[0] if isinstance(output, tuple) else output
            self.captured_features[layer_idx] = features

        return hook

    def setup_hooks(self) -> None:
        blocks = self._locate_blocks()
        num_blocks = len(blocks)
        try:
            for i, depth in enumerate(self.alignment_depths):
                if depth >= num_blocks:
                    raise ValueError(
                        f"M2-REPA alignment depth {depth} exceeds available blocks ({num_blocks})"
                    )
                handle = blocks[depth].register_forward_hook(self._get_hook(i))
                self.hook_handles.append(handle)
                logger.info("M2-REPA: hook attached to layer %d.", depth)
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
        return list(self.projectors.parameters())

    @staticmethod
    def _maybe_strip_cls(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            return tokens
        token_count = tokens.shape[1]
        is_square = int(math.isqrt(token_count)) ** 2 == token_count
        is_square_minus_one = int(math.isqrt(token_count - 1)) ** 2 == (token_count - 1)
        if is_square_minus_one and not is_square:
            return tokens[:, 1:, :]
        return tokens

    @staticmethod
    def _coerce_encoder_tokens(features: Any) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            tensor_candidate = None
            for value in features:
                if isinstance(value, torch.Tensor):
                    tensor_candidate = value
                    break
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, torch.Tensor):
                            tensor_candidate = nested
                            break
                if tensor_candidate is not None:
                    break
            if tensor_candidate is None:
                raise ValueError(
                    "M2-REPA: encoder tuple/list output did not contain tensor features"
                )
            features = tensor_candidate

        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                features = features["x_norm_patchtokens"]
            elif "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"].unsqueeze(1)
            else:
                tensor_candidate = None
                for value in features.values():
                    if isinstance(value, torch.Tensor):
                        tensor_candidate = value
                        break
                if tensor_candidate is None:
                    raise ValueError("M2-REPA: encoder output did not contain tensor features")
                features = tensor_candidate

        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        if not isinstance(features, torch.Tensor):
            raise ValueError("M2-REPA: unsupported encoder output type")
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() == 5:
            bsz, channels, frames, height, width = features.shape
            features = features.view(
                bsz,
                channels,
                frames * height * width,
            ).transpose(1, 2)
        elif features.dim() == 4:
            bsz, channels, height, width = features.shape
            features = features.view(bsz, channels, height * width).transpose(1, 2)
        elif features.dim() != 3:
            raise ValueError(
                f"M2-REPA: expected encoder features with 2-5 dims, got {features.dim()}"
            )
        return M2RepaHelper._maybe_strip_cls(features)

    @staticmethod
    def _match_feature_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
        if tokens.shape[-1] == target_dim:
            return tokens
        original_shape = tokens.shape
        flat = tokens.reshape(-1, original_shape[-2], original_shape[-1])
        flat = F.interpolate(
            flat,
            size=target_dim,
            mode="linear",
            align_corners=False,
        )
        return flat.reshape(*original_shape[:-1], target_dim)

    def _extract_sam_tokens(
        self,
        clean_pixels: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        encoder = self.encoders[expert_idx]
        with torch.no_grad():
            tokens = encoder.extract_features(clean_pixels)
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(1)
        if tokens.dim() != 4:
            raise ValueError(
                f"M2-REPA: SAM-family expert returned unsupported shape {tuple(tokens.shape)}"
            )
        expected_dim = self.expert_dims[expert_idx]
        tokens = self._match_feature_dim(tokens, expected_dim)
        if self.max_spatial_tokens > 0 and tokens.shape[2] > self.max_spatial_tokens:
            bsz, frames, token_count, dim = tokens.shape
            tokens_2d = tokens.reshape(bsz * frames, token_count, dim)
            tokens_2d = _interpolate_token_count(tokens_2d, self.max_spatial_tokens)
            tokens = tokens_2d.view(bsz, frames, self.max_spatial_tokens, dim)
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    def _extract_depth_tokens(
        self,
        clean_pixels: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        encoder = self.encoders[expert_idx]
        with torch.no_grad():
            tokens = encoder.extract_features(clean_pixels)
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(1)
        if tokens.dim() != 4:
            raise ValueError(
                "M2-REPA: Depth Anything V2 expert returned unsupported shape "
                f"{tuple(tokens.shape)}"
            )
        expected_dim = self.expert_dims[expert_idx]
        tokens = self._match_feature_dim(tokens, expected_dim)
        if self.max_spatial_tokens > 0 and tokens.shape[2] > self.max_spatial_tokens:
            bsz, frames, token_count, dim = tokens.shape
            tokens_2d = tokens.reshape(bsz * frames, token_count, dim)
            tokens_2d = _interpolate_token_count(tokens_2d, self.max_spatial_tokens)
            tokens = tokens_2d.view(bsz, frames, self.max_spatial_tokens, dim)
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    def _extract_expert_tokens(
        self,
        clean_pixels: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        if clean_pixels.dim() == 4:
            clean_pixels = clean_pixels.unsqueeze(2)
        bsz, channels, frames, height, width = clean_pixels.shape
        if channels != 3:
            raise ValueError(f"M2-REPA expects RGB clean_pixels, got C={channels}")

        expert_kinds = getattr(
            self,
            "expert_kinds",
            ["encoder_manager"] * len(getattr(self, "encoders", [])),
        )
        if expert_idx < len(expert_kinds) and expert_kinds[expert_idx] == "sam":
            return self._extract_sam_tokens(clean_pixels, expert_idx)
        if expert_idx < len(expert_kinds) and expert_kinds[expert_idx] == "depth":
            return self._extract_depth_tokens(clean_pixels, expert_idx)

        images = clean_pixels.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames,
            channels,
            height,
            width,
        )
        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
        images = preprocess_raw_image(images, self.encoder_types[expert_idx])

        chunks = [images]
        if self.encoder_chunk_size > 0 and images.shape[0] > self.encoder_chunk_size:
            chunks = list(torch.split(images, self.encoder_chunk_size, dim=0))

        token_chunks: List[torch.Tensor] = []
        encoder = self.encoders[expert_idx]
        with torch.no_grad():
            for chunk in chunks:
                encoded = encoder.forward_features(chunk)
                token_chunks.append(self._coerce_encoder_tokens(encoded))
        tokens = torch.cat(token_chunks, dim=0)
        expected_dim = self.expert_dims[expert_idx]
        tokens = self._match_feature_dim(tokens, expected_dim)
        if self.max_spatial_tokens > 0 and tokens.shape[1] > self.max_spatial_tokens:
            tokens = _interpolate_token_count(tokens, self.max_spatial_tokens)
        tokens = tokens.view(bsz, frames, tokens.shape[1], tokens.shape[2])
        if self.detach_teacher:
            tokens = tokens.detach()
        return tokens

    @staticmethod
    def _interpolate_frames(tokens: torch.Tensor, target_frames: int) -> torch.Tensor:
        if tokens.shape[1] == target_frames:
            return tokens
        bsz, frames, token_count, dim = tokens.shape
        x = tokens.permute(0, 2, 3, 1).reshape(bsz * token_count, dim, frames)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        return x.reshape(bsz, token_count, dim, target_frames).permute(0, 3, 1, 2)

    def _reshape_source_tokens(
        self,
        projected: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        bsz, seq_len, dim = projected.shape
        if target_frames > 0 and seq_len % target_frames == 0:
            tokens_per_frame = seq_len // target_frames
            return projected.view(bsz, target_frames, tokens_per_frame, dim)
        if not self._shape_warning_logged:
            logger.warning(
                "M2-REPA: seq_len (%d) not divisible by target_frames (%d); using pooled-frame fallback.",
                seq_len,
                target_frames,
            )
            self._shape_warning_logged = True
        return projected.unsqueeze(1)

    def _match_temporal_and_spatial(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.shape[1] != target.shape[1]:
            if self.temporal_align:
                source = self._interpolate_frames(source, target.shape[1])
            else:
                min_frames = min(source.shape[1], target.shape[1])
                source = source[:, :min_frames]
                target = target[:, :min_frames]

        src_tokens = source.shape[2]
        tgt_tokens = target.shape[2]
        if src_tokens != tgt_tokens:
            source_2d = source.reshape(
                source.shape[0] * source.shape[1],
                source.shape[2],
                source.shape[3],
            )
            target_2d = target.reshape(
                target.shape[0] * target.shape[1],
                target.shape[2],
                target.shape[3],
            )
            if self.spatial_align:
                if src_tokens > tgt_tokens:
                    source_2d = _interpolate_token_count(source_2d, tgt_tokens)
                else:
                    target_2d = _interpolate_token_count(target_2d, src_tokens)
            else:
                min_tokens = min(src_tokens, tgt_tokens)
                source_2d = source_2d[:, :min_tokens]
                target_2d = target_2d[:, :min_tokens]
            source = source_2d.view(
                source.shape[0],
                source.shape[1],
                source_2d.shape[1],
                source.shape[3],
            )
            target = target_2d.view(
                target.shape[0],
                target.shape[1],
                target_2d.shape[1],
                target.shape[3],
            )

        if self.max_spatial_tokens > 0 and source.shape[2] > self.max_spatial_tokens:
            source_2d = source.reshape(
                source.shape[0] * source.shape[1],
                source.shape[2],
                source.shape[3],
            )
            target_2d = target.reshape(
                target.shape[0] * target.shape[1],
                target.shape[2],
                target.shape[3],
            )
            source_2d = _interpolate_token_count(source_2d, self.max_spatial_tokens)
            target_2d = _interpolate_token_count(target_2d, self.max_spatial_tokens)
            source = source_2d.view(
                source.shape[0],
                source.shape[1],
                self.max_spatial_tokens,
                source.shape[3],
            )
            target = target_2d.view(
                target.shape[0],
                target.shape[1],
                self.max_spatial_tokens,
                target.shape[3],
            )

        return source, target

    def _flatten_for_cka(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1, tokens.shape[-1])
        if flat.shape[0] > self.decouple_max_samples:
            idx = torch.linspace(
                0,
                flat.shape[0] - 1,
                self.decouple_max_samples,
                device=flat.device,
            ).long()
            flat = flat.index_select(0, idx)
        return flat

    @staticmethod
    def _match_sample_count(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == y.shape[0]:
            return x, y
        sample_count = min(x.shape[0], y.shape[0])
        x_idx = torch.linspace(0, x.shape[0] - 1, sample_count, device=x.device).long()
        y_idx = torch.linspace(0, y.shape[0] - 1, sample_count, device=y.device).long()
        return x.index_select(0, x_idx), y.index_select(0, y_idx)

    @staticmethod
    def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = M2RepaHelper._match_sample_count(x, y)
        if x.shape[0] < 2:
            return x.new_tensor(0.0)
        x_float = x.float()
        y_float = y.float()
        x_centered = x_float - x_float.mean(dim=0, keepdim=True)
        y_centered = y_float - y_float.mean(dim=0, keepdim=True)
        xy = x_centered.transpose(0, 1).matmul(y_centered)
        xx = x_centered.transpose(0, 1).matmul(x_centered)
        yy = y_centered.transpose(0, 1).matmul(y_centered)
        numerator = xy.pow(2).sum()
        denominator = (xx.pow(2).sum() * yy.pow(2).sum()).clamp_min(1e-12).sqrt()
        return (numerator / denominator).to(dtype=x.dtype)

    def _decoupling_loss(self, sources: List[torch.Tensor]) -> torch.Tensor:
        if len(sources) < 2 or self.decouple_lambda <= 0:
            return sources[0].new_tensor(0.0)
        flattened = [self._flatten_for_cka(source) for source in sources]
        losses: List[torch.Tensor] = []
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                losses.append(self._linear_cka(flattened[i], flattened[j]))
        if not losses:
            return sources[0].new_tensor(0.0)
        return torch.stack(losses).mean()

    def get_repa_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
    ) -> torch.Tensor:
        del vae
        if clean_pixels is None:
            return torch.tensor(0.0)
        if not any(feat is not None for feat in self.captured_features):
            return clean_pixels.new_tensor(0.0)

        if clean_pixels.dim() == 4:
            target_frames = 1
        else:
            target_frames = int(clean_pixels.shape[2])

        expert_targets = [
            self._extract_expert_tokens(clean_pixels, idx)
            for idx in range(len(self.encoders))
        ]

        align_losses: List[torch.Tensor] = []
        decouple_losses: List[torch.Tensor] = []
        for depth_idx, features in enumerate(self.captured_features):
            if features is None or not isinstance(features, torch.Tensor):
                continue
            if features.dim() != 3:
                if not self._shape_warning_logged:
                    logger.warning(
                        "M2-REPA: expected diffusion features [B, Seq, C], got %s. Skipping layer loss.",
                        tuple(features.shape),
                    )
                    self._shape_warning_logged = True
                continue
            if features.shape[0] != clean_pixels.shape[0]:
                if not self._shape_warning_logged:
                    logger.warning(
                        "M2-REPA: feature batch size %d does not match pixel batch size %d; skipping layer loss.",
                        features.shape[0],
                        clean_pixels.shape[0],
                    )
                    self._shape_warning_logged = True
                continue

            layer_sources: List[torch.Tensor] = []
            for expert_idx, target_tokens in enumerate(expert_targets):
                projected = self.projectors[depth_idx][expert_idx](features)
                source_tokens = self._reshape_source_tokens(
                    projected,
                    target_frames=target_frames,
                )
                source_tokens, aligned_target = self._match_temporal_and_spatial(
                    source=source_tokens,
                    target=target_tokens.to(
                        device=source_tokens.device,
                        dtype=source_tokens.dtype,
                    ),
                )
                source_norm = F.normalize(source_tokens, dim=-1)
                target_norm = F.normalize(aligned_target, dim=-1)
                align_losses.append((-(source_norm * target_norm).sum(dim=-1)).mean())
                layer_sources.append(source_tokens)

            if layer_sources:
                decouple_losses.append(self._decoupling_loss(layer_sources))

        self.captured_features = [None] * len(self.alignment_depths)
        if not align_losses:
            return clean_pixels.new_tensor(0.0)

        align_loss = torch.stack(align_losses).mean()
        if decouple_losses:
            decouple_loss = torch.stack(decouple_losses).mean()
        else:
            decouple_loss = align_loss.new_tensor(0.0)
        total = self.align_lambda * align_loss + self.decouple_lambda * decouple_loss
        self.last_m2_repa_metrics = {
            "m2_repa_align_loss": align_loss.detach(),
            "m2_repa_decouple_loss": decouple_loss.detach(),
            "m2_repa_total_loss": total.detach(),
        }
        return total
