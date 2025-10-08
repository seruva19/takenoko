"""
Reward model implementations for SRPO training.

Supports three reward models:
1. HPS v2.1 (Human Preference Score)
2. PickScore (CLIP-based prompt-image alignment)
3. Aesthetic Predictor v2/v2.5

All models support Semantic Relative Preference (SRP) for relative scoring via
positive/negative prompt augmentation.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SRPORewardModel(ABC, nn.Module):
    """
    Base class for SRPO reward models.

    All reward models must implement:
    - compute_rewards(images, prompts) -> Tensor[B]
    - compute_srp_rewards(images, base_prompts, pos_words, neg_words, k) -> Tensor[B]
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def compute_rewards(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Compute reward scores for images given prompts.

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]
            prompts: List of B text prompts

        Returns:
            Tensor of shape [B] with reward scores
        """
        pass

    def compute_srp_rewards(
        self,
        images: torch.Tensor,
        base_prompts: List[str],
        positive_words: List[str],
        negative_words: List[str],
        k: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute Semantic Relative Preference (SRP) scores.

        Formula (VERIFIED against SRPO.py line 234):
            r_srp = (1 + k) * r_positive - r_negative

        Where:
            r_positive = reward with positive control word appended
            r_negative = reward with negative control word appended

        This uses the relative difference between positive/negative prompt-conditioned
        reward pairs to regularize the reward signal and mitigate reward hacking.

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]
            base_prompts: List of B base prompts
            positive_words: List of positive control words
            negative_words: List of negative control words
            k: SRP scaling parameter (default: 1.0)

        Returns:
            Tensor of shape [B] with SRP-scaled rewards
        """
        import random

        B = len(base_prompts)

        # Sample control words
        pos_prompts = [
            f"{prompt}, {random.choice(positive_words)}" for prompt in base_prompts
        ]
        neg_prompts = [
            f"{prompt}, {random.choice(negative_words)}" for prompt in base_prompts
        ]

        # Compute rewards
        r_pos = self.compute_rewards(images, pos_prompts)  # [B]
        r_neg = self.compute_rewards(images, neg_prompts)  # [B]

        # SRP formula
        r_srp = (1.0 + k) * r_pos - r_neg  # [B]

        return r_srp


class HPSRewardModel(SRPORewardModel):
    """
    Human Preference Score v2.1 reward model.

    Dependency: hpsv2 (already in pyproject.toml)
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device, dtype)

        # Lazy import to avoid loading when not using HPS
        import hpsv2

        self.hps_model = hpsv2.score_model(device=str(device))
        logger.info(f"Loaded HPS v2.1 model on {device}")

    def compute_rewards(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Compute HPS v2.1 scores.

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]
            prompts: List of B prompts

        Returns:
            Tensor of shape [B] with HPS scores
        """
        # HPS expects images in [0, 1] range (no preprocessing needed)
        with torch.no_grad():
            scores = self.hps_model.score(images, prompts)  # [B]

        return scores


class PickScoreRewardModel(SRPORewardModel):
    """
    PickScore (CLIP-based prompt-image alignment) reward model.

    Dependency: transformers (already in pyproject.toml)
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device, dtype)

        from transformers import CLIPModel, CLIPProcessor

        # Load CLIP model for PickScore
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model.eval()
        logger.info(f"Loaded PickScore (CLIP) model on {device}")

    def compute_rewards(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Compute PickScore (CLIP similarity) scores.

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]
            prompts: List of B prompts

        Returns:
            Tensor of shape [B] with PickScore values
        """
        from PIL import Image
        import numpy as np

        B = images.shape[0]

        # Convert tensors to PIL images
        images_pil = []
        for i in range(B):
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images_pil.append(Image.fromarray(img_np))

        # Preprocess
        inputs = self.clip_processor(
            text=prompts, images=images_pil, return_tensors="pt", padding=True
        ).to(self.device)

        # Compute CLIP similarity
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # PickScore = cosine similarity between image and text embeddings
            image_embeds = outputs.image_embeds  # [B, D]
            text_embeds = outputs.text_embeds  # [B, D]

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            scores = (image_embeds * text_embeds).sum(dim=-1)  # [B]

        return scores


class AestheticRewardModel(SRPORewardModel):
    """
    Aesthetic Predictor v2/v2.5 reward model.

    Uses Takenoko's existing improved_aesthetic_predictor.py
    Dependency: None (already in Takenoko codebase)
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device, dtype)

        # Import Takenoko's existing aesthetic predictor
        from reward.improved_aesthetic_predictor import ImprovedAestheticPredictor

        self.aesthetic_model = ImprovedAestheticPredictor(device=device)
        logger.info(f"Loaded Aesthetic Predictor on {device}")

    def compute_rewards(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Compute aesthetic scores (prompts are ignored for aesthetic model).

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]
            prompts: List of B prompts (unused for aesthetic scoring)

        Returns:
            Tensor of shape [B] with aesthetic scores
        """
        with torch.no_grad():
            scores = self.aesthetic_model.score(images)  # [B]

        return scores


def create_reward_model(
    reward_model_name: str, device: torch.device, dtype: torch.dtype
) -> SRPORewardModel:
    """
    Factory function to create reward models.

    Args:
        reward_model_name: One of "hps", "pickscore", "aesthetic"
        device: Torch device
        dtype: Torch dtype

    Returns:
        Instantiated reward model
    """
    if reward_model_name == "hps":
        return HPSRewardModel(device, dtype)
    elif reward_model_name == "pickscore":
        return PickScoreRewardModel(device, dtype)
    elif reward_model_name == "aesthetic":
        return AestheticRewardModel(device, dtype)
    else:
        raise ValueError(
            f"Unknown reward model: {reward_model_name}. "
            "Must be one of: 'hps', 'pickscore', 'aesthetic'"
        )
