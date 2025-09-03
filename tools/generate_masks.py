"""
Command-line tool for automatic mask generation using Florence-2 + SAM2.

This tool generates masks for datasets using modern computer vision models.
All functionality is self-contained in this single file.
"""

import argparse
import logging
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import supervision as sv
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Auto-masking dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class AutoMaskGenerator:
    """
    Automatic mask generation using Florence-2 for detection + SAM2 for segmentation.

    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        florence_model: str = "microsoft/Florence-2-large",
        sam_config: str = "sam2_hiera_l.yaml",
        sam_checkpoint: str = "checkpoints/sam2_hiera_large.pt",
    ):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Auto-masking requires additional dependencies. See requirements.txt"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.florence_model_name = florence_model
        self.sam_config = sam_config
        self.sam_checkpoint = sam_checkpoint

        self.florence_model = None
        self.florence_processor = None
        self.sam_model = None

        logger.info(f"AutoMaskGenerator initialized for device: {self.device}")

    def load_models(self) -> None:
        """Load Florence-2 and SAM2 models."""
        logger.info("Loading Florence-2 model...")
        self._load_florence_model()

        logger.info("Loading SAM2 model...")
        self._load_sam_model()

        logger.info("Models loaded successfully")

    def _load_florence_model(self) -> None:
        """Load Florence-2 model for object detection."""

        # Fix for flash_attn import issues
        def fixed_get_imports(filename):
            if not str(filename).endswith("/modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        # Patch and load
        from unittest.mock import patch

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.florence_model = (
                AutoModelForCausalLM.from_pretrained(
                    self.florence_model_name, trust_remote_code=True
                )
                .to(self.device)
                .eval()
            )
            self.florence_processor = AutoProcessor.from_pretrained(
                self.florence_model_name, trust_remote_code=True
            )

    def _load_sam_model(self) -> None:
        """Load SAM2 model for segmentation."""
        # Download checkpoint if needed
        if not os.path.exists(self.sam_checkpoint):
            self._download_sam_checkpoint()

        # Build and load model
        sam_model = build_sam2(self.sam_config, self.sam_checkpoint, device=self.device)
        self.sam_model = SAM2ImagePredictor(sam_model=sam_model)

    def _download_sam_checkpoint(self) -> None:
        """Download SAM2 checkpoint."""
        from huggingface_hub import hf_hub_download

        os.makedirs("checkpoints", exist_ok=True)
        logger.info("Downloading SAM2 checkpoint...")

        hf_hub_download(
            "SkalskiP/florence-sam-masking",
            repo_type="space",
            subfolder="checkpoints",
            local_dir="./",
            filename="sam2_hiera_large.pt",
        )

    def generate_mask(
        self, image: Image.Image, prompt: str = "person"
    ) -> Optional[Image.Image]:
        """
        Generate mask for a single image.

        Args:
            image: Input PIL image
            prompt: Text prompt for object detection

        Returns:
            Generated mask as PIL image, or None if no objects detected
        """
        if self.florence_model is None or self.sam_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        try:
            # Florence-2 detection
            detections = self._run_florence_detection(image, prompt)
            if len(detections) == 0:
                logger.warning(f"No objects detected for prompt: '{prompt}'")
                return None

            # SAM2 segmentation
            detections = self._run_sam_segmentation(image, detections)

            # Combine masks
            combined_mask = np.any(detections.mask, axis=0)
            mask_image = Image.fromarray(combined_mask.astype("uint8") * 255)

            return mask_image

        except Exception as e:
            logger.error(f"Failed to generate mask: {e}")
            return None

    def _run_florence_detection(self, image: Image.Image, prompt: str) -> sv.Detections:
        """Run Florence-2 object detection."""
        task = "<OPEN_VOCABULARY_DETECTION>"
        full_prompt = task + prompt

        # Prepare inputs
        inputs = self.florence_processor(
            text=full_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode response
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        response = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=image.size
        )

        # Convert to detections
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2, result=response, resolution_wh=image.size
        )

        return detections

    def _run_sam_segmentation(
        self, image: Image.Image, detections: sv.Detections
    ) -> sv.Detections:
        """Run SAM2 segmentation on detected objects."""
        image_np = np.array(image.convert("RGB"))
        self.sam_model.set_image(image_np)

        # Predict masks
        masks, scores, _ = self.sam_model.predict(
            box=detections.xyxy, multimask_output=False
        )

        # Ensure correct dimensions
        if len(masks.shape) == 4:
            masks = np.squeeze(masks)

        detections.mask = masks.astype(bool)
        return detections

    def generate_dataset_masks(
        self,
        input_dir: str,
        output_dir: str,
        prompt: str = "person",
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> Dict[str, Any]:
        """
        Generate masks for entire dataset.

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save masks
            prompt: Detection prompt
            image_extensions: Supported image extensions

        Returns:
            Statistics about mask generation
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find image files
        image_files = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                image_files.append(filename)

        stats = {
            "total_images": len(image_files),
            "masks_generated": 0,
            "masks_skipped": 0,
            "errors": 0,
        }

        logger.info(f"Processing {len(image_files)} images...")

        for filename in image_files:
            input_path = os.path.join(input_dir, filename)

            # Determine output path
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")

            # Skip if mask already exists
            if os.path.exists(output_path):
                logger.debug(f"Mask already exists: {output_path}")
                stats["masks_skipped"] += 1
                continue

            try:
                # Load image
                with Image.open(input_path) as image:
                    image = image.convert("RGB")

                    # Generate mask
                    mask = self.generate_mask(image, prompt)

                    if mask is not None:
                        mask.save(output_path)
                        stats["masks_generated"] += 1
                        logger.info(f"Generated mask: {output_path}")
                    else:
                        stats["errors"] += 1
                        logger.warning(f"No mask generated for: {filename}")

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing {filename}: {e}")

        logger.info(f"Mask generation complete. Stats: {stats}")
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks for dataset using Florence-2 + SAM2"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated masks",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Detection prompt (default: person)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not DEPENDENCIES_AVAILABLE:
        print("ERROR: Auto-masking dependencies not available.")
        print("Install with: pip install supervision transformers sam2")
        return 1

    # Create generator and load models
    generator = AutoMaskGenerator()
    generator.load_models()

    # Generate masks
    stats = generator.generate_dataset_masks(
        input_dir=args.input_dir, output_dir=args.output_dir, prompt=args.prompt
    )

    print(f"\nMask generation complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Masks generated: {stats['masks_generated']}")
    print(f"Masks skipped: {stats['masks_skipped']}")
    print(f"Errors: {stats['errors']}")

    return 0


if __name__ == "__main__":
    exit(main())
