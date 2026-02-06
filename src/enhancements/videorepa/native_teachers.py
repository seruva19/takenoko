"""Native video-teacher loaders for VideoREPA."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class NativeTeacherBundle:
    model: nn.Module
    teacher_type: str
    patch_size: int
    tubelet_size: int
    input_size: tuple[int, int]


def is_native_teacher_spec(spec: str) -> bool:
    value = str(spec or "").strip().lower()
    return (
        value.startswith("videomaev2-")
        or value.startswith("vjepa-")
        or value.startswith("vjepa2-")
    )


def _require_checkpoint(path: str, teacher_name: str) -> str:
    ckpt = str(path or "").strip()
    if ckpt == "":
        raise ValueError(
            f"{teacher_name} teacher requires videorepa_teacher_checkpoint to be set."
        )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"{teacher_name} checkpoint not found: {ckpt}"
        )
    return ckpt


def _freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _parse_hw(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(
        "videorepa_video_teacher_align_resolution must be [height, width]"
    )


def _load_vjepa_state_dict(
    model: nn.Module,
    checkpoint_path: str,
    checkpoint_key: str,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")

    state_dict = None
    if isinstance(payload, dict):
        if checkpoint_key in payload and isinstance(payload[checkpoint_key], dict):
            state_dict = payload[checkpoint_key]
        elif "encoder" in payload and isinstance(payload["encoder"], dict):
            state_dict = payload["encoder"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in payload.values()):
            state_dict = payload

    if not isinstance(state_dict, dict):
        raise ValueError(
            "VJEPA checkpoint did not contain a usable state dict "
            f"(requested key: {checkpoint_key!r})."
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
        normalized[new_key] = value

    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing:
        # Keep this as warning-level behavior in callers; load should remain permissive.
        pass
    if unexpected:
        pass


def _get_videomaev2_builder(name: str) -> Callable[..., nn.Module]:
    from enhancements.videorepa.models import videomaev2

    mapping: dict[str, Callable[..., nn.Module]] = {
        "vit-s": videomaev2.vit_small_patch16_224,
        "vit-b": videomaev2.vit_base_patch16_224,
        "vit-l": videomaev2.vit_large_patch16_224,
        "vit-h": videomaev2.vit_huge_patch16_224,
        "vit-g": videomaev2.vit_giant_patch14_224,
    }
    if name not in mapping:
        raise ValueError(
            "Unsupported videomaev2 teacher variant. "
            "Use one of: vit-s, vit-b, vit-l, vit-h, vit-g."
        )
    return mapping[name]


def _get_vjepa_builder(name: str) -> Callable[..., nn.Module]:
    from enhancements.videorepa.models import vjepa_vision_transformer as vjepa

    mapping: dict[str, Callable[..., nn.Module]] = {
        "vit-t": vjepa.vit_tiny,
        "vit-s": vjepa.vit_small,
        "vit-b": vjepa.vit_base,
        "vit-l": vjepa.vit_large,
        "vit-h": vjepa.vit_huge,
        "vit-g": vjepa.vit_giant,
        "vit-gg": vjepa.vit_gigantic,
    }
    if name not in mapping:
        raise ValueError(
            "Unsupported vjepa teacher variant. "
            "Use one of: vit-t, vit-s, vit-b, vit-l, vit-h, vit-g, vit-gg."
        )
    return mapping[name]


def _get_vjepa2_hub_entry(name: str) -> str:
    mapping: dict[str, str] = {
        "vit-l": "vjepa2_vit_large",
    }
    if name not in mapping:
        raise ValueError(
            "Unsupported vjepa2 teacher variant. "
            "Use one of: vit-l."
        )
    return mapping[name]


def load_native_teacher(
    spec: str,
    args: Any,
    device: torch.device,
) -> NativeTeacherBundle:
    value = str(spec or "").strip().lower()
    if value.startswith("videomaev2-"):
        variant = value[len("videomaev2-") :]
        frames = int(getattr(args, "videorepa_video_teacher_frames", 16))
        tubelet = int(getattr(args, "videorepa_video_teacher_tubelet_size", 2))
        image_size = int(getattr(args, "videorepa_video_teacher_image_size", 224))
        align_res = _parse_hw(
            getattr(args, "videorepa_video_teacher_align_resolution", [480, 720]),
            default=(480, 720),
        )
        builder = _get_videomaev2_builder(variant)
        model = builder(
            pretrained=False,
            all_frames=frames,
            tubelet_size=tubelet,
            img_size=image_size,
            align_video_resolution=align_res,
            num_classes=0,
        )
        ckpt = _require_checkpoint(
            str(getattr(args, "videorepa_teacher_checkpoint", "")),
            teacher_name="VideoMAEv2",
        )
        if hasattr(model, "from_pretrained"):
            model.from_pretrained(ckpt)
        else:
            raise RuntimeError(
                "VideoMAEv2 model builder does not expose from_pretrained()."
            )
        model = _freeze(model, device=device)
        patch_size = int(getattr(model, "patch_size", 16))
        tubelet_size = int(getattr(model, "tubelet_size", tubelet))
        return NativeTeacherBundle(
            model=model,
            teacher_type="videomaev2",
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            input_size=(image_size, image_size),
        )

    if value.startswith("vjepa-"):
        variant = value[len("vjepa-") :]
        frames = int(getattr(args, "videorepa_video_teacher_frames", 16))
        tubelet = int(getattr(args, "videorepa_video_teacher_tubelet_size", 2))
        image_size = int(getattr(args, "videorepa_video_teacher_image_size", 224))
        patch_size = int(getattr(args, "videorepa_video_teacher_patch_size", 16))
        builder = _get_vjepa_builder(variant)
        model = builder(
            img_size=image_size,
            patch_size=patch_size,
            num_frames=frames,
            tubelet_size=tubelet,
            uniform_power=bool(
                getattr(args, "videorepa_vjepa_uniform_power", True)
            ),
            use_sdpa=bool(getattr(args, "videorepa_vjepa_use_sdpa", True)),
            use_SiLU=bool(getattr(args, "videorepa_vjepa_use_silu", False)),
            tight_SiLU=bool(getattr(args, "videorepa_vjepa_tight_silu", False)),
        )
        ckpt = _require_checkpoint(
            str(getattr(args, "videorepa_teacher_checkpoint", "")),
            teacher_name="VJEPA",
        )
        checkpoint_key = str(
            getattr(args, "videorepa_vjepa_checkpoint_key", "target_encoder")
        )
        _load_vjepa_state_dict(model, ckpt, checkpoint_key=checkpoint_key)
        model = _freeze(model, device=device)
        return NativeTeacherBundle(
            model=model,
            teacher_type="vjepa",
            patch_size=patch_size,
            tubelet_size=tubelet,
            input_size=(image_size, image_size),
        )

    if value.startswith("vjepa2-"):
        variant = value[len("vjepa2-") :]
        image_size = int(getattr(args, "videorepa_video_teacher_image_size", 224))
        patch_size = int(getattr(args, "videorepa_video_teacher_patch_size", 16))
        tubelet = int(getattr(args, "videorepa_video_teacher_tubelet_size", 2))
        hub_entry = _get_vjepa2_hub_entry(variant)
        try:
            loaded = torch.hub.load("facebookresearch/vjepa2", hub_entry)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load VJEPA2 from torch.hub. "
                "Ensure internet access or a cached local torch.hub checkout is available."
            ) from exc

        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        if not isinstance(model, nn.Module):
            raise ValueError(
                "torch.hub VJEPA2 loader did not return a valid torch.nn.Module."
            )
        if hasattr(model, "norm"):
            model.norm = nn.Identity()
        model = _freeze(model, device=device)
        return NativeTeacherBundle(
            model=model,
            teacher_type="vjepa2",
            patch_size=int(getattr(model, "patch_size", patch_size)),
            tubelet_size=int(getattr(model, "tubelet_size", tubelet)),
            input_size=(image_size, image_size),
        )

    raise ValueError(
        f"Unsupported native VideoREPA teacher specification: {spec!r}"
    )
