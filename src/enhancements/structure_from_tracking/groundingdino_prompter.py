"""GroundingDINO-backed prompt masks for Structure-From-Tracking."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
import torch

from common.logger import get_logger

logger = get_logger(__name__)


def build_binary_box_mask(
    height: int,
    width: int,
    boxes_xyxy: torch.Tensor,
) -> torch.Tensor:
    """Rasterize XYXY boxes to a binary mask with shape [H, W]."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be > 0, got ({height}, {width})")
    if boxes_xyxy.numel() == 0:
        return torch.zeros(height, width, dtype=torch.float32)
    if boxes_xyxy.dim() != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(
            "boxes_xyxy must be [N, 4], got "
            f"{tuple(boxes_xyxy.shape)}"
        )

    mask = torch.zeros(height, width, dtype=torch.float32)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if x1 < x0 or y1 < y0:
            continue
        mask[y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return mask


class GroundingDINOPrompter:
    """Optional object-box prompter using GroundingDINO from transformers."""

    def __init__(self, args: Any, device: torch.device):
        self.device = device
        self.enabled = bool(getattr(args, "sft_use_groundingdino_prompts", False))
        self.paper_strict_mode = bool(getattr(args, "sft_paper_strict_mode", False))
        self.model_id = str(
            getattr(args, "sft_groundingdino_model_id", "IDEA-Research/grounding-dino-base")
        )
        self.box_threshold = float(getattr(args, "sft_groundingdino_box_threshold", 0.35))
        self.text_threshold = float(getattr(args, "sft_groundingdino_text_threshold", 0.25))
        self.prompt_source = str(
            getattr(args, "sft_groundingdino_prompt_source", "caption_or_item_key")
        ).lower()
        self.apply_to_last_frame = bool(
            getattr(args, "sft_groundingdino_apply_to_last_frame", True)
        )
        self.use_sam2_refine = bool(
            getattr(args, "sft_groundingdino_use_sam2_refine", False)
        )
        self.sam2_config = str(
            getattr(args, "sft_groundingdino_sam2_config", "sam2_hiera_l.yaml")
        )
        self.sam2_checkpoint = str(
            getattr(args, "sft_groundingdino_sam2_checkpoint", "checkpoints/sam2_hiera_large.pt")
        )
        self.download_sam2_checkpoint = bool(
            getattr(args, "sft_groundingdino_download_sam2_checkpoint", False)
        )

        self._processor = None
        self._model = None
        self._sam2_predictor = None
        self._failed = False
        self._sam2_failed = False
        self._warned_missing_prompt = False

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self._failed:
            return False
        if self._model is not None and self._processor is not None:
            return True
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            logger.info(
                "Structure-From-Tracking: loaded GroundingDINO prompter from %s.",
                self.model_id,
            )
            return True
        except Exception as exc:
            self._failed = True
            if self.paper_strict_mode:
                raise RuntimeError(
                    "Structure-From-Tracking strict mode: GroundingDINO prompter unavailable."
                ) from exc
            logger.warning(
                "Structure-From-Tracking: GroundingDINO prompter unavailable (%s). "
                "Falling back to dataset-provided masks only.",
                exc,
            )
            return False

        # Optional SAM2 refinement path; failure degrades to box masks.
        if self.use_sam2_refine:
            self._load_sam2_predictor()
        return True

    def _maybe_download_sam2_checkpoint(self) -> None:
        if os.path.exists(self.sam2_checkpoint):
            return
        if not self.download_sam2_checkpoint:
            return
        try:
            from huggingface_hub import hf_hub_download

            os.makedirs(os.path.dirname(self.sam2_checkpoint) or ".", exist_ok=True)
            hf_hub_download(
                "SkalskiP/florence-sam-masking",
                repo_type="space",
                subfolder="checkpoints",
                local_dir="./",
                filename=os.path.basename(self.sam2_checkpoint),
            )
            logger.info(
                "Structure-From-Tracking: downloaded SAM2 checkpoint to %s.",
                self.sam2_checkpoint,
            )
        except Exception as exc:
            logger.warning(
                "Structure-From-Tracking: failed to download SAM2 checkpoint (%s).",
                exc,
            )

    def _load_sam2_predictor(self) -> bool:
        if self._sam2_predictor is not None:
            return True
        if self._sam2_failed:
            return False
        try:
            self._maybe_download_sam2_checkpoint()
            if not os.path.exists(self.sam2_checkpoint):
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {self.sam2_checkpoint}"
                )
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam_model=sam_model)
            logger.info(
                "Structure-From-Tracking: loaded SAM2 predictor for GroundingDINO mask refinement."
            )
            return True
        except Exception as exc:
            self._sam2_failed = True
            if self.paper_strict_mode:
                raise RuntimeError(
                    "Structure-From-Tracking strict mode: SAM2 refinement backend unavailable."
                ) from exc
            logger.warning(
                "Structure-From-Tracking: SAM2 refinement unavailable (%s). Falling back to box masks.",
                exc,
            )
            return False

    def _resolve_prompt(self, item: Any) -> Optional[str]:
        caption = str(getattr(item, "caption", "") or "").strip()
        item_key = str(getattr(item, "item_key", "") or "").strip()

        if self.prompt_source == "caption":
            return caption or None
        if self.prompt_source == "item_key":
            return item_key or None
        # caption_or_item_key
        return caption or item_key or None

    @staticmethod
    def _to_pil(image_chw: torch.Tensor):
        from PIL import Image

        image = image_chw.detach().float().clamp(-1.0, 1.0)
        image = ((image + 1.0) * 127.5).round().to(torch.uint8)
        image = image.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(image)

    def _detect_boxes(self, image_chw: torch.Tensor, prompt_text: str) -> torch.Tensor:
        assert self._processor is not None
        assert self._model is not None

        pil_image = self._to_pil(image_chw)
        inputs = self._processor(
            images=pil_image,
            text=prompt_text,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor(
            [[pil_image.height, pil_image.width]],
            device=self.device,
        )
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
        if len(results) == 0:
            return torch.empty((0, 4), dtype=torch.float32)
        boxes = results[0].get("boxes", None)
        if not isinstance(boxes, torch.Tensor):
            return torch.empty((0, 4), dtype=torch.float32)
        return boxes.detach().to(dtype=torch.float32).cpu()

    def _refine_mask_with_sam2(
        self,
        image_chw: torch.Tensor,
        boxes_xyxy: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_sam2_refine:
            return None
        if boxes_xyxy.numel() == 0:
            return None
        if not self._load_sam2_predictor():
            return None
        assert self._sam2_predictor is not None

        pil = self._to_pil(image_chw)
        image_np = np.array(pil.convert("RGB"))
        try:
            self._sam2_predictor.set_image(image_np)
            masks, _scores, _logits = self._sam2_predictor.predict(
                box=boxes_xyxy.numpy(),
                multimask_output=False,
            )
            if masks is None:
                return None
            masks_np = np.array(masks)
            if masks_np.ndim == 4:
                masks_np = np.squeeze(masks_np, axis=1)
            if masks_np.ndim == 2:
                mask_2d = masks_np
            elif masks_np.ndim == 3:
                mask_2d = np.any(masks_np, axis=0)
            else:
                return None
            mask_tensor = torch.from_numpy(mask_2d.astype(np.float32))
            if mask_tensor.shape[0] != image_chw.shape[1] or mask_tensor.shape[1] != image_chw.shape[2]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(image_chw.shape[1], image_chw.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            return mask_tensor.clamp(0.0, 1.0)
        except Exception as exc:
            logger.warning(
                "Structure-From-Tracking: SAM2 refinement failed (%s). Using box mask fallback.",
                exc,
            )
            return None

    def generate_mask_hints(
        self,
        clean_pixels: torch.Tensor,
        item_info: Sequence[Any],
    ) -> Optional[torch.Tensor]:
        """Generate [B, F, H, W] mask hints from prompts and detections."""
        if not self.enabled:
            return None
        if clean_pixels.dim() != 5:
            return None
        if len(item_info) != clean_pixels.shape[0]:
            return None
        if not self._load():
            return None

        batch_size, _channels, frame_count, height, width = clean_pixels.shape
        masks = []
        has_any = False
        for sample_idx in range(batch_size):
            prompt_text = self._resolve_prompt(item_info[sample_idx])
            if not prompt_text:
                if self.paper_strict_mode:
                    raise ValueError(
                        "Structure-From-Tracking strict mode: empty prompt text encountered for GroundingDINO prompting."
                    )
                if not self._warned_missing_prompt:
                    logger.warning(
                        "Structure-From-Tracking: GroundingDINO prompting enabled but prompts are empty."
                    )
                    self._warned_missing_prompt = True
                masks.append(torch.zeros(frame_count, height, width, dtype=torch.float32))
                continue

            first_frame = clean_pixels[sample_idx, :, 0]
            first_boxes = self._detect_boxes(first_frame, prompt_text=prompt_text)
            union_boxes = first_boxes
            refined_candidates = []
            first_refined = self._refine_mask_with_sam2(
                first_frame,
                boxes_xyxy=first_boxes,
            )
            if first_refined is not None:
                refined_candidates.append(first_refined)
            if self.apply_to_last_frame and frame_count > 1:
                last_boxes = self._detect_boxes(
                    clean_pixels[sample_idx, :, frame_count - 1],
                    prompt_text=prompt_text,
                )
                if last_boxes.numel() > 0:
                    union_boxes = (
                        torch.cat([first_boxes, last_boxes], dim=0)
                        if first_boxes.numel() > 0
                        else last_boxes
                    )
                last_refined = self._refine_mask_with_sam2(
                    clean_pixels[sample_idx, :, frame_count - 1],
                    boxes_xyxy=last_boxes,
                )
                if last_refined is not None:
                    refined_candidates.append(last_refined)

            if refined_candidates:
                mask_2d = torch.stack(refined_candidates, dim=0).amax(dim=0)
            else:
                mask_2d = build_binary_box_mask(
                    height=height,
                    width=width,
                    boxes_xyxy=union_boxes,
                )
            if mask_2d.max().item() > 0:
                has_any = True
            masks.append(mask_2d.unsqueeze(0).expand(frame_count, -1, -1))

        if not has_any:
            return None
        return torch.stack(masks, dim=0).to(device=clean_pixels.device)
