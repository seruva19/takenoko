"""Reward-based dense segment labeling for DenseDPO."""

from __future__ import annotations

from typing import List, Tuple
import torch


class DenseDPOLabeler:
    """Compute per-segment preference labels using a reward model."""

    def __init__(
        self,
        *,
        reward_model,
        vae,
        device: torch.device,
        reward_frame_strategy: str,
        reward_num_frames: int,
        reward_aggregation: str,
    ) -> None:
        self.reward_model = reward_model
        self.vae = vae
        self.device = device
        self.reward_frame_strategy = reward_frame_strategy
        self.reward_num_frames = reward_num_frames
        self.reward_aggregation = reward_aggregation

    def compute_preferences(
        self,
        *,
        policy_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        prompts: List[str],
        segment_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-segment preference labels based on reward comparisons.

        Returns:
            Tuple of (preferences, policy_rewards, reference_rewards).
            preferences is a float tensor of shape [B, num_segments] with
            1.0 for policy preferred, 0.0 otherwise.
        """
        policy_rewards = self._compute_segment_rewards(
            policy_latents, prompts, segment_frames
        )
        reference_rewards = self._compute_segment_rewards(
            reference_latents, prompts, segment_frames
        )
        preferences = (policy_rewards > reference_rewards).to(torch.float32)
        return preferences, policy_rewards, reference_rewards

    def _compute_segment_rewards(
        self,
        latents: torch.Tensor,
        prompts: List[str],
        segment_frames: int,
    ) -> torch.Tensor:
        bsz, _, total_frames, _, _ = latents.shape
        rewards = []
        for start in range(0, total_frames, segment_frames):
            end = min(start + segment_frames, total_frames)
            frame_indices = self._select_segment_frames(end - start)
            frame_indices = [start + idx for idx in frame_indices]
            images = self._decode_frames(latents, frame_indices)
            frame_rewards = []
            for idx in range(images.shape[1]):
                frame_reward = self.reward_model.compute_rewards(
                    images[:, idx, :, :, :], prompts
                )
                frame_rewards.append(frame_reward)
            frame_rewards = torch.stack(frame_rewards, dim=1)
            rewards.append(self._aggregate_rewards(frame_rewards))
        return torch.stack(rewards, dim=1).view(bsz, -1)

    def _decode_frames(
        self, latents: torch.Tensor, frame_indices: List[int]
    ) -> torch.Tensor:
        with torch.no_grad():
            bsz = latents.shape[0]
            decoded_frames = []
            for frame_idx in frame_indices:
                frame_latents = latents[:, :, frame_idx, :, :]
                latents_list = [frame_latents[i] for i in range(bsz)]
                images_list = self.vae.decode(latents_list)
                images = torch.stack(images_list)
                images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0.0, 1.0)
                decoded_frames.append(images)
            return torch.stack(decoded_frames, dim=1)

    def _select_segment_frames(self, segment_length: int) -> List[int]:
        if self.reward_num_frames >= segment_length:
            return list(range(segment_length))
        strategy = self.reward_frame_strategy
        if strategy == "first":
            return [0]
        if strategy == "uniform":
            return (
                torch.linspace(0, segment_length - 1, self.reward_num_frames)
                .long()
                .tolist()
            )
        if strategy == "boundary":
            if self.reward_num_frames == 1:
                return [0]
            if self.reward_num_frames == 2:
                return [0, segment_length - 1]
            indices = [0]
            middle = self.reward_num_frames - 2
            if middle > 0:
                middle_indices = (
                    torch.linspace(1, segment_length - 2, middle)
                    .long()
                    .tolist()
                )
                indices.extend(middle_indices)
            indices.append(segment_length - 1)
            return indices
        return list(range(segment_length))

    def _aggregate_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_aggregation == "mean":
            return rewards.mean(dim=1)
        if self.reward_aggregation == "min":
            return rewards.min(dim=1)[0]
        if self.reward_aggregation == "max":
            return rewards.max(dim=1)[0]
        if self.reward_aggregation == "weighted":
            weights = torch.linspace(
                0.5, 1.0, rewards.shape[1], device=rewards.device
            )
            weights = weights / weights.sum()
            return (rewards * weights.unsqueeze(0)).sum(dim=1)
        return rewards.mean(dim=1)
